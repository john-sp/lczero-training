#include "loader/chunk_source/tar_chunk_source.h"

#include <zlib.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "proto/data_loader_config.pb.h"

namespace lczero {
namespace training {
namespace {

// Minimal in-memory tar (ustar/PAX) writer used to drive the TarChunkSource
// indexer through the exact framing shapes it must handle. The parser only
// consults the name, size, and typeflag fields, but we emit valid ustar
// headers (magic + checksum) so the fixtures also load with GNU/BSD tar and
// python's tarfile for cross-checking.
class TarBuilder {
 public:
  // Appends a data-bearing member (regular file by default).
  void AddMember(const std::string& name, const std::string& content,
                 char typeflag = '0') {
    AppendHeader(name, content.size(), typeflag);
    data_ += content;
    PadToBlock();
  }

  // Appends a directory entry (typeflag '5', no payload).
  void AddDir(const std::string& name) { AppendHeader(name, 0, '5'); }

  // Appends a PAX extended header (typeflag 'x' local, 'g' global). `records`
  // is the raw payload; its byte length is what exercises the block-skipping
  // logic (small records fit in one block, large ones span several).
  void AddPax(const std::string& records, bool global = false) {
    AppendHeader(global ? "pax_global_header" : "PaxHeaders/pax", records.size(),
                 global ? 'g' : 'x');
    data_ += records;
    PadToBlock();
  }

  // Terminates the archive with the two zero blocks a real tar writer emits.
  std::string Finish() const {
    std::string out = data_;
    out.append(1024, '\0');
    return out;
  }

 private:
  static void WriteOctal(char* field, size_t width, uint64_t value) {
    // `width - 1` octal digits, NUL-terminated, zero padded (ustar style).
    for (size_t i = width - 1; i-- > 0;) {
      field[i] = static_cast<char>('0' + (value & 7u));
      value >>= 3;
    }
    field[width - 1] = '\0';
  }

  void AppendHeader(const std::string& name, uint64_t size, char typeflag) {
    std::array<char, 512> header{};
    std::memcpy(header.data(), name.data(),
                std::min<size_t>(name.size(), 100));
    WriteOctal(&header[100], 8, 0644);   // mode
    WriteOctal(&header[108], 8, 0);      // uid
    WriteOctal(&header[116], 8, 0);      // gid
    WriteOctal(&header[124], 12, size);  // size
    WriteOctal(&header[136], 12, 0);     // mtime
    header[156] = typeflag;
    std::memcpy(&header[257], "ustar", 5);  // magic
    header[263] = '0';
    header[264] = '0';  // version "00"
    // Checksum: sum of all header bytes with the checksum field treated as
    // spaces, then written back as 6 octal digits + NUL + space.
    std::memset(&header[148], ' ', 8);
    uint32_t checksum = 0;
    for (unsigned char c : header) checksum += c;
    WriteOctal(&header[148], 7, checksum);
    header[155] = ' ';
    data_.append(header.data(), header.size());
  }

  void PadToBlock() {
    const size_t rem = data_.size() % 512;
    if (rem != 0) data_.append(512 - rem, '\0');
  }

  std::string data_;
};

// Gzip a buffer so we can build .gz members matching the real shard-tar shape
// and verify they decompress after being located at the correct offset.
std::string Gzip(const std::string& input) {
  z_stream strm = {};
  EXPECT_EQ(deflateInit2(&strm, Z_DEFAULT_COMPRESSION, Z_DEFLATED,
                         16 + MAX_WBITS, 8, Z_DEFAULT_STRATEGY),
            Z_OK);
  std::string out;
  out.resize(deflateBound(&strm, input.size()) + 32);
  strm.next_in =
      reinterpret_cast<Bytef*>(const_cast<char*>(input.data()));
  strm.avail_in = static_cast<uInt>(input.size());
  strm.next_out = reinterpret_cast<Bytef*>(out.data());
  strm.avail_out = static_cast<uInt>(out.size());
  EXPECT_EQ(deflate(&strm, Z_FINISH), Z_STREAM_END);
  out.resize(out.size() - strm.avail_out);
  deflateEnd(&strm);
  return out;
}

class TarChunkSourceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = std::filesystem::temp_directory_path() /
           ("tar_chunk_source_test_" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir_);
  }

  void TearDown() override {
    if (std::filesystem::exists(dir_)) std::filesystem::remove_all(dir_);
  }

  std::filesystem::path WriteTar(const std::string& bytes) {
    const std::filesystem::path path = dir_ / "fixture.tar";
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.close();
    return path;
  }

  std::filesystem::path dir_;
};

// Reads the full (raw) content of an indexed member back out for comparison.
std::string ReadWhole(TarChunkSource& src, size_t index) {
  auto prefix = src.GetChunkPrefix(index, /*max_bytes=*/1 << 20);
  EXPECT_TRUE(prefix.has_value());
  return prefix.value_or(std::string());
}

// (d) Plain ustar with no PAX headers -- the T91 dataset shape.
TEST_F(TarChunkSourceTest, PlainUstarNoPax) {
  TarBuilder tar;
  tar.AddMember("game0.chunk", "AAAA-first");
  tar.AddMember("game1.chunk", "BBBB-second-longer-payload");
  tar.AddMember("game2.chunk", "CC");
  const auto path = WriteTar(tar.Finish());

  TarChunkSource src(path, ChunkSourceLoaderConfig::V6TrainingData);
  ASSERT_EQ(src.GetChunkCount(), 3u);
  EXPECT_EQ(ReadWhole(src, 0), "AAAA-first");
  EXPECT_EQ(ReadWhole(src, 1), "BBBB-second-longer-payload");
  EXPECT_EQ(ReadWhole(src, 2), "CC");
}

// (a) A small (<512B) PAX 'x' header before every member -- the exact shape
// python's tarfile (default PAX format) emits, and the shape of our shard
// tars. Every member must be found with intact content and ZERO extra chunks.
TEST_F(TarChunkSourceTest, SmallPaxHeaderBeforeEachMember) {
  TarBuilder tar;
  const std::vector<std::string> payloads = {
      "payload-zero", "payload-one-longer", "p2", "payload-three"};
  for (size_t i = 0; i < payloads.size(); ++i) {
    // ~34-byte PAX record, well under one 512-byte block.
    tar.AddPax("30 mtime=1700000000.00000000" + std::to_string(i) + "\n");
    tar.AddMember("game" + std::to_string(i) + ".chunk", payloads[i]);
  }
  const auto path = WriteTar(tar.Finish());

  TarChunkSource src(path, ChunkSourceLoaderConfig::V6TrainingData);
  ASSERT_EQ(src.GetChunkCount(), payloads.size());
  for (size_t i = 0; i < payloads.size(); ++i) {
    EXPECT_EQ(ReadWhole(src, i), payloads[i]) << "member " << i;
  }
}

// (b) A PAX record larger than a single 512-byte block. This is the regression
// the fix targets: the old code skipped only the 512-byte header, so a
// multi-block PAX payload desynced the parser for the rest of the archive.
TEST_F(TarChunkSourceTest, LargePaxRecordSpanningMultipleBlocks) {
  TarBuilder tar;
  // A single ~700-byte attribute (e.g. a very long path override) spans two
  // data blocks.
  std::string big_record = "712 path=";
  big_record.append(690, 'x');
  big_record += "\n";
  tar.AddPax(big_record);
  tar.AddMember("after_big_pax.chunk", "STILL-IN-SYNC");
  tar.AddMember("second.chunk", "SECOND-OK");
  const auto path = WriteTar(tar.Finish());

  TarChunkSource src(path, ChunkSourceLoaderConfig::V6TrainingData);
  ASSERT_EQ(src.GetChunkCount(), 2u);
  EXPECT_EQ(ReadWhole(src, 0), "STILL-IN-SYNC");
  EXPECT_EQ(ReadWhole(src, 1), "SECOND-OK");
}

// (c) A 'g' global extended header at the very start of the archive, followed
// by ordinary members.
TEST_F(TarChunkSourceTest, GlobalPaxHeaderAtStart) {
  TarBuilder tar;
  tar.AddPax("52 comment=global-metadata-for-the-whole-archive\n",
             /*global=*/true);
  tar.AddMember("g0.chunk", "GLOBAL-THEN-DATA");
  tar.AddMember("g1.chunk", "MORE-DATA");
  const auto path = WriteTar(tar.Finish());

  TarChunkSource src(path, ChunkSourceLoaderConfig::V6TrainingData);
  ASSERT_EQ(src.GetChunkCount(), 2u);
  EXPECT_EQ(ReadWhole(src, 0), "GLOBAL-THEN-DATA");
  EXPECT_EQ(ReadWhole(src, 1), "MORE-DATA");
}

// A genuinely-unknown typeflag with a data payload must be skipped (payload and
// all) without desyncing the members around it.
TEST_F(TarChunkSourceTest, UnknownTypeflagWithPayloadIsSkipped) {
  TarBuilder tar;
  tar.AddMember("before.chunk", "BEFORE");
  // Typeflag 'A' is not one we understand; it carries a multi-block payload.
  std::string weird_payload(900, 'Z');
  tar.AddMember("weird.blob", weird_payload, 'A');
  tar.AddMember("after.chunk", "AFTER");
  const auto path = WriteTar(tar.Finish());

  TarChunkSource src(path, ChunkSourceLoaderConfig::V6TrainingData);
  ASSERT_EQ(src.GetChunkCount(), 2u);
  EXPECT_EQ(ReadWhole(src, 0), "BEFORE");
  EXPECT_EQ(ReadWhole(src, 1), "AFTER");
}

// Directories and the LICENSE file are ignored; PAX headers interleaved with
// real .gz members still yield exactly the .gz members, decompressible.
TEST_F(TarChunkSourceTest, GzipMembersWithPaxAndLicense) {
  TarBuilder tar;
  tar.AddDir("shard/");
  tar.AddPax("30 mtime=1700000000.000000000\n");
  tar.AddMember("LICENSE", "not a chunk");
  const std::string decompressed0 = "raw-training-bytes-zero";
  const std::string decompressed1 = "raw-training-bytes-one-longer";
  tar.AddPax("30 mtime=1700000001.000000000\n");
  tar.AddMember("shard/0000.gz", Gzip(decompressed0));
  tar.AddPax("30 mtime=1700000002.000000000\n");
  tar.AddMember("shard/0001.gz", Gzip(decompressed1));
  const auto path = WriteTar(tar.Finish());

  TarChunkSource src(path, ChunkSourceLoaderConfig::V6TrainingData);
  ASSERT_EQ(src.GetChunkCount(), 2u);
  // GetChunkPrefix transparently gunzips .gz members.
  EXPECT_EQ(ReadWhole(src, 0), decompressed0);
  EXPECT_EQ(ReadWhole(src, 1), decompressed1);
}

// Real-data parity check. Point this at a tar and its expected chunk count via
// environment variables to validate against, e.g., a python-tarfile-written
// PAX archive or a real shard tar:
//   TAR_PARITY_FILE=/path/shard.tar TAR_PARITY_COUNT=512 ./tar_chunk_source_test
TEST_F(TarChunkSourceTest, RealDataParityFromEnv) {
  const char* file = std::getenv("TAR_PARITY_FILE");
  const char* count = std::getenv("TAR_PARITY_COUNT");
  if (!file || !count) {
    GTEST_SKIP() << "Set TAR_PARITY_FILE and TAR_PARITY_COUNT to run.";
  }
  TarChunkSource src(std::filesystem::path(file),
                     ChunkSourceLoaderConfig::V6TrainingData);
  EXPECT_EQ(src.GetChunkCount(),
            static_cast<size_t>(std::strtoull(count, nullptr, 10)));
}

}  // namespace
}  // namespace training
}  // namespace lczero
