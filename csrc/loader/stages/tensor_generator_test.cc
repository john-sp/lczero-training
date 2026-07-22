// ABOUTME: Unit tests for TensorGenerator stage in training pipeline.
// ABOUTME: Tests tensor conversion, batching, and data format correctness.

#include "loader/stages/tensor_generator.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "libs/lc0/src/trainingdata/trainingdata_v6.h"
#include "utils/queue.h"
#include "utils/tensor.h"

namespace lczero {
namespace training {

namespace {

template <typename T>
class PassthroughStage : public Stage {
 public:
  explicit PassthroughStage(Queue<T>* queue) : queue_(queue) {}

  void Start() override {}
  void Stop() override {}
  StageMetricProto FlushMetrics() override { return StageMetricProto(); }
  QueueBase* GetOutput(std::string_view name = "") override {
    (void)name;
    return queue_;
  }
  void SetInputs(absl::Span<QueueBase* const> inputs) override {
    if (!inputs.empty()) {
      throw std::runtime_error("PassthroughStage expects no inputs");
    }
  }

 private:
  Queue<T>* queue_;
};

}  // namespace

class TensorGeneratorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    input_queue_ = std::make_unique<Queue<FrameType>>(100);
    config_.set_batch_size(4);
    config_.set_threads(1);
    config_.mutable_output()->set_queue_capacity(10);
  }

  FrameType CreateTestFrame() {
    FrameType frame{};
    std::memset(&frame, 0, sizeof(frame));

    frame.version = 6;
    frame.input_format = 3;

    // Fill probabilities with test values.
    for (ssize_t i = 0; i < 1858; ++i) {
      frame.probabilities[i] = static_cast<float>(i) / 1858.0f;
    }

    // Fill planes with test pattern.
    for (ssize_t i = 0; i < 104; ++i) {
      frame.planes[i] = 0x0F0F0F0F0F0F0F0FULL + i;  // Test pattern
    }

    // Set castling rights.
    frame.castling_us_ooo = 1;
    frame.castling_us_oo = 0;
    frame.castling_them_ooo = 1;
    frame.castling_them_oo = 1;

    // Set other fields.
    frame.side_to_move_or_enpassant = 1;
    frame.rule50_count = 50;

    // Set Q and D values.
    frame.result_q = 0.5f;
    frame.result_d = 0.2f;
    frame.best_q = 0.3f;
    frame.best_d = 0.1f;
    frame.best_m = 42.5f;
    frame.plies_left = 42.5f;

    return frame;
  }

  void VerifyTensorTuple(const TensorTuple& tensors,
                         const std::vector<FrameType>& frames) {
    const size_t batch_size = frames.size();

    // Verify tuple has 5 elements
    ASSERT_EQ(tensors.size(), 5);

    // Verify input tensor: (batch_size, 112, 8, 8)
    const auto* planes_tensor =
        dynamic_cast<const TypedTensor<float>*>(tensors[0].get());
    ASSERT_NE(planes_tensor, nullptr);
    EXPECT_EQ(planes_tensor->shape().size(), 4);
    EXPECT_EQ(planes_tensor->shape()[0], batch_size);
    EXPECT_EQ(planes_tensor->shape()[1], 112);
    EXPECT_EQ(planes_tensor->shape()[2], 8);
    EXPECT_EQ(planes_tensor->shape()[3], 8);

    // Verify probabilities tensor: (batch_size, 1858)
    const auto* probs_tensor =
        dynamic_cast<const TypedTensor<float>*>(tensors[1].get());
    ASSERT_NE(probs_tensor, nullptr);
    EXPECT_EQ(probs_tensor->shape().size(), 2);
    EXPECT_EQ(probs_tensor->shape()[0], batch_size);
    EXPECT_EQ(probs_tensor->shape()[1], 1858);

    // Verify values tensor: (batch_size, 7, 3)
    const auto* values_tensor =
        dynamic_cast<const TypedTensor<float>*>(tensors[2].get());
    ASSERT_NE(values_tensor, nullptr);
    EXPECT_EQ(values_tensor->shape().size(), 3);
    EXPECT_EQ(values_tensor->shape()[0], batch_size);
    EXPECT_EQ(values_tensor->shape()[1], 7);
    EXPECT_EQ(values_tensor->shape()[2], 3);

    // Verify aux indices tensor: (batch_size, 3) int32.
    const auto* aux_indices_tensor =
        dynamic_cast<const TypedTensor<int32_t>*>(tensors[3].get());
    ASSERT_NE(aux_indices_tensor, nullptr);
    EXPECT_EQ(aux_indices_tensor->shape().size(), 2);
    EXPECT_EQ(aux_indices_tensor->shape()[0], batch_size);
    EXPECT_EQ(aux_indices_tensor->shape()[1], 3);

    // Verify aux targets tensor: (batch_size, 2) float32.
    const auto* aux_targets_tensor =
        dynamic_cast<const TypedTensor<float>*>(tensors[4].get());
    ASSERT_NE(aux_targets_tensor, nullptr);
    EXPECT_EQ(aux_targets_tensor->shape().size(), 2);
    EXPECT_EQ(aux_targets_tensor->shape()[0], batch_size);
    EXPECT_EQ(aux_targets_tensor->shape()[1], 2);
  }

  void VerifyTensorData(const TensorTuple& tensors,
                        const std::vector<FrameType>& frames) {
    const size_t batch_size = frames.size();
    const auto* planes_tensor =
        dynamic_cast<const TypedTensor<float>*>(tensors[0].get());
    const auto* probs_tensor =
        dynamic_cast<const TypedTensor<float>*>(tensors[1].get());
    const auto* values_tensor =
        dynamic_cast<const TypedTensor<float>*>(tensors[2].get());

    for (size_t i = 0; i < batch_size; ++i) {
      const auto& frame = frames[i];

      // Verify probabilities data.
      auto probs_slice = probs_tensor->slice({static_cast<ssize_t>(i)});
      for (ssize_t j = 0; j < 1858; ++j) {
        EXPECT_FLOAT_EQ(probs_slice[j], frame.probabilities[j]);
      }

      // Verify values tensor [batch, 6, 3] with raw q/d/m values
      // Index 0: result (q=0.5, d=0.2, m=42.5)
      auto values_slice = values_tensor->slice({static_cast<ssize_t>(i)});
      EXPECT_FLOAT_EQ(values_slice[0 * 3 + 0], 0.5f);   // result_q
      EXPECT_FLOAT_EQ(values_slice[0 * 3 + 1], 0.2f);   // result_d
      EXPECT_FLOAT_EQ(values_slice[0 * 3 + 2], 42.5f);  // result_m

      // Index 1: best (q=0.3, d=0.1, m=42.5)
      EXPECT_FLOAT_EQ(values_slice[1 * 3 + 0], 0.3f);   // best_q
      EXPECT_FLOAT_EQ(values_slice[1 * 3 + 1], 0.1f);   // best_d
      EXPECT_FLOAT_EQ(values_slice[1 * 3 + 2], 42.5f);  // best_m

      // Verify planes data - check first few planes and meta planes.
      auto planes_slice = planes_tensor->slice({static_cast<ssize_t>(i)});

      // Check first plane (plane 0).
      uint64_t expected_plane_0 = 0x0F0F0F0F0F0F0F0FULL;
      for (ssize_t square = 0; square < 64; ++square) {
        float expected =
            static_cast<float>((expected_plane_0 >> (63 - square)) & 1);
        EXPECT_FLOAT_EQ(planes_slice[square], expected);
      }

      // Check meta planes.
      // Plane 104: castling_us_ooo = 1
      for (ssize_t square = 104 * 64; square < 105 * 64; ++square) {
        EXPECT_FLOAT_EQ(planes_slice[square], 1.0f);
      }

      // Plane 105: castling_us_oo = 0
      for (ssize_t square = 105 * 64; square < 106 * 64; ++square) {
        EXPECT_FLOAT_EQ(planes_slice[square], 0.0f);
      }

      // Plane 109: rule50_count = 50, should be 50/99
      for (ssize_t square = 109 * 64; square < 110 * 64; ++square) {
        EXPECT_FLOAT_EQ(planes_slice[square], 50.0f / 99.0f);
      }

      // Plane 110: all zeros
      for (ssize_t square = 110 * 64; square < 111 * 64; ++square) {
        EXPECT_FLOAT_EQ(planes_slice[square], 0.0f);
      }

      // Plane 111: all ones
      for (ssize_t square = 111 * 64; square < 112 * 64; ++square) {
        EXPECT_FLOAT_EQ(planes_slice[square], 1.0f);
      }
    }
  }

  std::unique_ptr<Queue<FrameType>> input_queue_;
  TensorGeneratorConfig config_;
};

TEST_F(TensorGeneratorTest, GeneratesCorrectTensorShapes) {
  TensorGenerator generator(config_);
  generator.SetInputs({input_queue_.get()});
  generator.Start();

  auto producer = input_queue_->CreateProducer();
  std::vector<FrameType> frames;
  for (size_t i = 0; i < config_.batch_size(); ++i) {
    frames.push_back(CreateTestFrame());
    producer.Put(frames.back());
  }
  producer.Close();

  auto tensors = generator.output_queue()->Get();
  VerifyTensorTuple(tensors, frames);
}

TEST_F(TensorGeneratorTest, GeneratesCorrectTensorData) {
  TensorGenerator generator(config_);
  generator.SetInputs({input_queue_.get()});
  generator.Start();

  auto producer = input_queue_->CreateProducer();
  std::vector<FrameType> frames;
  for (size_t i = 0; i < config_.batch_size(); ++i) {
    frames.push_back(CreateTestFrame());
    producer.Put(frames.back());
  }
  producer.Close();

  auto tensors = generator.output_queue()->Get();
  VerifyTensorTuple(tensors, frames);
  VerifyTensorData(tensors, frames);
}

TEST_F(TensorGeneratorTest, HandlesMultipleBatches) {
  TensorGenerator generator(config_);
  generator.SetInputs({input_queue_.get()});
  generator.Start();

  auto producer = input_queue_->CreateProducer();

  // Send two full batches.
  std::vector<FrameType> all_frames;
  for (ssize_t batch = 0; batch < 2; ++batch) {
    for (size_t i = 0; i < config_.batch_size(); ++i) {
      auto frame = CreateTestFrame();
      frame.version = batch * 1000 + i;  // Unique version for each frame
      all_frames.push_back(frame);
      producer.Put(frame);
    }
  }
  producer.Close();

  // Get first batch.
  auto tensors1 = generator.output_queue()->Get();
  std::vector<FrameType> batch1_frames(
      all_frames.begin(), all_frames.begin() + config_.batch_size());
  VerifyTensorTuple(tensors1, batch1_frames);

  // Get second batch.
  auto tensors2 = generator.output_queue()->Get();
  std::vector<FrameType> batch2_frames(
      all_frames.begin() + config_.batch_size(), all_frames.end());
  VerifyTensorTuple(tensors2, batch2_frames);

  // No more batches should be available.
  EXPECT_THROW(generator.output_queue()->Get(), QueueClosedException);
}

TEST_F(TensorGeneratorTest, HandlesDifferentBatchSizes) {
  config_.set_batch_size(2);
  TensorGenerator generator(config_);
  generator.SetInputs({input_queue_.get()});
  generator.Start();

  auto producer = input_queue_->CreateProducer();
  std::vector<FrameType> frames;
  for (size_t i = 0; i < config_.batch_size(); ++i) {
    frames.push_back(CreateTestFrame());
    producer.Put(frames.back());
  }
  producer.Close();

  auto tensors = generator.output_queue()->Get();
  VerifyTensorTuple(tensors, frames);
}

TEST_F(TensorGeneratorTest, HandlesEmptyInput) {
  TensorGenerator generator(config_);
  generator.SetInputs({input_queue_.get()});
  generator.Start();

  // Close input queue without sending data.
  input_queue_->Close();

  // Should not output any tensors.
  EXPECT_THROW(generator.output_queue()->Get(), QueueClosedException);
}

TEST_F(TensorGeneratorTest, VerifiesPlanesConversion) {
  config_.set_batch_size(1);
  TensorGenerator generator(config_);
  generator.SetInputs({input_queue_.get()});
  generator.Start();

  auto producer = input_queue_->CreateProducer();

  FrameType frame = CreateTestFrame();
  // Set specific bit pattern for plane 0.
  frame.planes[0] = 0xAAAAAAAAAAAAAAAAULL;  // Alternating bits
  // Set specific values for meta planes.
  frame.castling_us_ooo = 1;
  frame.castling_us_oo = 0;
  frame.rule50_count = 75;

  producer.Put(frame);
  producer.Close();

  auto tensors = generator.output_queue()->Get();
  const auto* planes_tensor =
      dynamic_cast<const TypedTensor<float>*>(tensors[0].get());

  auto planes_slice = planes_tensor->slice({0});

  // Verify plane 0 bit conversion.
  for (ssize_t square = 0; square < 64; ++square) {
    float expected =
        static_cast<float>((0xAAAAAAAAAAAAAAAAULL >> (63 - square)) & 1);
    EXPECT_FLOAT_EQ(planes_slice[square], expected)
        << "Mismatch at square " << square;
  }

  // Verify rule50_count conversion: 75/99.
  for (ssize_t square = 109 * 64; square < 110 * 64; ++square) {
    EXPECT_FLOAT_EQ(planes_slice[square], 75.0f / 99.0f);
  }
}

TEST_F(TensorGeneratorTest, VerifiesQDConversion) {
  config_.set_batch_size(1);
  TensorGenerator generator(config_);
  generator.SetInputs({input_queue_.get()});
  generator.Start();

  auto producer = input_queue_->CreateProducer();

  FrameType frame = CreateTestFrame();
  // Test specific Q/D values.
  frame.result_q = 0.4f;
  frame.result_d = 0.3f;
  frame.best_q = -0.2f;
  frame.best_d = 0.1f;

  producer.Put(frame);
  producer.Close();

  auto tensors = generator.output_queue()->Get();
  const auto* values_tensor =
      dynamic_cast<const TypedTensor<float>*>(tensors[2].get());

  auto values_slice = values_tensor->slice({0});

  // Verify result values: q=0.4, d=0.3 (raw values, no WDL conversion)
  EXPECT_FLOAT_EQ(values_slice[0 * 3 + 0], 0.4f);  // result_q
  EXPECT_FLOAT_EQ(values_slice[0 * 3 + 1], 0.3f);  // result_d

  // Verify best values: q=-0.2, d=0.1 (raw values, no WDL conversion)
  EXPECT_FLOAT_EQ(values_slice[1 * 3 + 0], -0.2f);  // best_q
  EXPECT_FLOAT_EQ(values_slice[1 * 3 + 1], 0.1f);   // best_d
}

TEST_F(TensorGeneratorTest, V7ReservedSlotsPassthrough) {
  config_.set_batch_size(2);
  TensorGenerator generator(config_);
  generator.SetInputs({input_queue_.get()});
  generator.Start();

  auto producer = input_queue_->CreateProducer();

  // Frame 0: V7 frame with the reserved block filled by the offline
  // rescorer.
  FrameType frame = CreateTestFrame();
  frame.version = 7;
  frame.q_st = 0.11f;
  frame.d_st = 0.22f;
  frame.opp_played_idx = 123;
  frame.next_played_idx = 456;
  frame.played_idx = 789;
  frame.reserved[0] = 2.0f;    // provenance: noise-deblunder
  frame.reserved[1] = -0.5f;   // q_st_censored
  frame.reserved[2] = 0.25f;   // d_st_censored
  frame.reserved[3] = -0.75f;  // played-move child-Q
  producer.Put(frame);

  // Frame 1: V7 last record of a game: sentinel lookahead indices and NaN
  // child-Q must be passed through.
  FrameType last_frame = CreateTestFrame();
  last_frame.version = 7;
  last_frame.q_st = 0.5f;
  last_frame.d_st = 0.1f;
  last_frame.opp_played_idx = 65535;
  last_frame.next_played_idx = 65535;
  last_frame.played_idx = 17;
  last_frame.reserved[0] = 1.0f;  // provenance: tablebase
  last_frame.reserved[1] = 0.9f;
  last_frame.reserved[2] = 0.05f;
  last_frame.reserved[3] = std::numeric_limits<float>::quiet_NaN();
  producer.Put(last_frame);
  producer.Close();

  auto tensors = generator.output_queue()->Get();
  ASSERT_EQ(tensors.size(), 5);
  const auto* values_tensor =
      dynamic_cast<const TypedTensor<float>*>(tensors[2].get());
  const auto* aux_indices_tensor =
      dynamic_cast<const TypedTensor<int32_t>*>(tensors[3].get());
  const auto* aux_targets_tensor =
      dynamic_cast<const TypedTensor<float>*>(tensors[4].get());
  ASSERT_NE(values_tensor, nullptr);
  ASSERT_NE(aux_indices_tensor, nullptr);
  ASSERT_NE(aux_targets_tensor, nullptr);

  // Frame 0: row 6 = st_censored from reserved[1]/[2].
  auto values0 = values_tensor->slice({0});
  EXPECT_FLOAT_EQ(values0[5 * 3 + 0], 0.11f);   // q_st
  EXPECT_FLOAT_EQ(values0[5 * 3 + 1], 0.22f);   // d_st
  EXPECT_TRUE(std::isnan(values0[5 * 3 + 2]));  // st m slot
  EXPECT_FLOAT_EQ(values0[6 * 3 + 0], -0.5f);   // q_st_censored
  EXPECT_FLOAT_EQ(values0[6 * 3 + 1], 0.25f);   // d_st_censored
  EXPECT_TRUE(std::isnan(values0[6 * 3 + 2]));  // st_censored m slot

  auto aux_idx0 = aux_indices_tensor->slice({0});
  EXPECT_EQ(aux_idx0[0], 123);  // opp_played_idx
  EXPECT_EQ(aux_idx0[1], 456);  // next_played_idx
  EXPECT_EQ(aux_idx0[2], 789);  // played_idx

  auto aux_tgt0 = aux_targets_tensor->slice({0});
  EXPECT_FLOAT_EQ(aux_tgt0[0], 2.0f);    // provenance
  EXPECT_FLOAT_EQ(aux_tgt0[1], -0.75f);  // child-Q

  // Frame 1: sentinels and NaN passthrough.
  auto values1 = values_tensor->slice({1});
  EXPECT_FLOAT_EQ(values1[6 * 3 + 0], 0.9f);
  EXPECT_FLOAT_EQ(values1[6 * 3 + 1], 0.05f);

  auto aux_idx1 = aux_indices_tensor->slice({1});
  EXPECT_EQ(aux_idx1[0], 65535);  // sentinel passthrough
  EXPECT_EQ(aux_idx1[1], 65535);  // sentinel passthrough
  EXPECT_EQ(aux_idx1[2], 17);

  auto aux_tgt1 = aux_targets_tensor->slice({1});
  EXPECT_FLOAT_EQ(aux_tgt1[0], 1.0f);
  EXPECT_TRUE(std::isnan(aux_tgt1[1]));  // NaN child-Q passthrough
}

TEST_F(TensorGeneratorTest, StCensoredFallsBackForV6Frames) {
  config_.set_batch_size(2);
  TensorGenerator generator(config_);
  generator.SetInputs({input_queue_.get()});
  generator.Start();

  auto producer = input_queue_->CreateProducer();

  // Frame 0: a raw V6 frame (version 6, reserved block zeroed as done by
  // the chunk sources when widening V6 records into V7 structs).
  FrameType v6_frame = CreateTestFrame();
  v6_frame.version = 6;
  v6_frame.q_st = 0.33f;
  v6_frame.d_st = 0.44f;
  v6_frame.played_idx = 42;
  producer.Put(v6_frame);

  // Frame 1: a V7 frame whose reserved block is all-zero (e.g. produced by
  // a writer that does not fill it). Must also fall back to plain st.
  FrameType v7_zero_frame = CreateTestFrame();
  v7_zero_frame.version = 7;
  v7_zero_frame.q_st = -0.6f;
  v7_zero_frame.d_st = 0.3f;
  v7_zero_frame.opp_played_idx = 5;
  v7_zero_frame.next_played_idx = 65535;
  v7_zero_frame.played_idx = 7;
  producer.Put(v7_zero_frame);
  producer.Close();

  auto tensors = generator.output_queue()->Get();
  const auto* values_tensor =
      dynamic_cast<const TypedTensor<float>*>(tensors[2].get());
  const auto* aux_indices_tensor =
      dynamic_cast<const TypedTensor<int32_t>*>(tensors[3].get());
  const auto* aux_targets_tensor =
      dynamic_cast<const TypedTensor<float>*>(tensors[4].get());

  // Frame 0 (V6): st_censored row falls back to plain st values; lookahead
  // indices are the invalid sentinel; provenance none; child-Q NaN.
  auto values0 = values_tensor->slice({0});
  EXPECT_FLOAT_EQ(values0[6 * 3 + 0], 0.33f);
  EXPECT_FLOAT_EQ(values0[6 * 3 + 1], 0.44f);
  EXPECT_TRUE(std::isnan(values0[6 * 3 + 2]));

  auto aux_idx0 = aux_indices_tensor->slice({0});
  EXPECT_EQ(aux_idx0[0], 65535);
  EXPECT_EQ(aux_idx0[1], 65535);
  EXPECT_EQ(aux_idx0[2], 42);

  auto aux_tgt0 = aux_targets_tensor->slice({0});
  EXPECT_FLOAT_EQ(aux_tgt0[0], 0.0f);
  EXPECT_TRUE(std::isnan(aux_tgt0[1]));

  // Frame 1 (V7, zeroed reserved): st_censored falls back to plain st, but
  // the V7 lookahead indices are used.
  auto values1 = values_tensor->slice({1});
  EXPECT_FLOAT_EQ(values1[6 * 3 + 0], -0.6f);
  EXPECT_FLOAT_EQ(values1[6 * 3 + 1], 0.3f);

  auto aux_idx1 = aux_indices_tensor->slice({1});
  EXPECT_EQ(aux_idx1[0], 5);
  EXPECT_EQ(aux_idx1[1], 65535);
  EXPECT_EQ(aux_idx1[2], 7);

  auto aux_tgt1 = aux_targets_tensor->slice({1});
  EXPECT_FLOAT_EQ(aux_tgt1[0], 0.0f);
  EXPECT_TRUE(std::isnan(aux_tgt1[1]));
}

}  // namespace training
}  // namespace lczero
