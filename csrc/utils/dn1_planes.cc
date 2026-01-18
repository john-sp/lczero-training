/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "utils/dn1_planes.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <span>
#include <vector>

#include "absl/types/span.h"

#include "chess/board.h"
#include "chess/types.h"
#include "loader/frame_type.h"
#include "neural/decoder.h"
#include "neural/encoder.h"
#include "trainingdata/reader.h"
#include "utils/bititer.h"
#include "utils/exception.h"

namespace lczero::training {

namespace {

Square SingleSquare(BitBoard input) {
  for (auto sq : input) {
    return sq;
  }
  assert(false);
  return Square();
}


constexpr bool IsOnBoard(int file, int rank) {
  return file >= 0 && file < 8 && rank >= 0 && rank < 8;
}

void AddAttack(std::array<uint8_t, 64>& counts, Square square) {
  const auto idx = square.as_idx();
  if (counts[idx] < 255) ++counts[idx];
}

void AddSlidingAttacks(std::array<uint8_t, 64>& counts, Square square,
                       absl::Span<const std::pair<int, int>> directions,
                       const BitBoard& occupancy) {
  int file = square.file().idx;
  int rank = square.rank().idx;
  for (const auto& [df, dr] : directions) {
    int f = file + df;
    int r = rank + dr;
    while (IsOnBoard(f, r)) {
      Square target(File::FromIdx(f), Rank::FromIdx(r));
      AddAttack(counts, target);
      if (occupancy.get(target)) break;
      f += df;
      r += dr;
    }
  }
}

std::array<uint8_t, 64> ComputeAttackCounts(const ChessBoard& board,
                                            const BitBoard& side_pieces,
                                            bool side_is_ours) {
  std::array<uint8_t, 64> counts{};
  const BitBoard occupancy = board.ours() | board.theirs();

  const BitBoard pawns = board.pawns() & side_pieces;
  const BitBoard knights = board.knights() & side_pieces;
  const BitBoard bishops = board.bishops() & side_pieces;
  const BitBoard rooks = board.rooks() & side_pieces;
  const BitBoard queens = board.queens() & side_pieces;
  const BitBoard kings = board.kings() & side_pieces;

  const int pawn_dir = side_is_ours ? 1 : -1;
  for (auto square : pawns) {
    const int file = square.file().idx;
    const int rank = square.rank().idx + pawn_dir;
    if (!IsOnBoard(file, rank)) continue;
    if (file > 0) {
      AddAttack(counts, Square(File::FromIdx(file - 1), Rank::FromIdx(rank)));
    }
    if (file < 7) {
      AddAttack(counts, Square(File::FromIdx(file + 1), Rank::FromIdx(rank)));
    }
  }

  static constexpr std::array<std::pair<int, int>, 8> kKnightDeltas = {
      std::pair<int, int>{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
      {1, -2}, {1, 2}, {2, -1}, {2, 1}};
  for (auto square : knights) {
    const int file = square.file().idx;
    const int rank = square.rank().idx;
    for (const auto& [df, dr] : kKnightDeltas) {
      const int f = file + df;
      const int r = rank + dr;
      if (!IsOnBoard(f, r)) continue;
      AddAttack(counts, Square(File::FromIdx(f), Rank::FromIdx(r)));
    }
  }

  static constexpr std::array<std::pair<int, int>, 8> kKingDeltas = {
      std::pair<int, int>{-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
      {0, 1}, {1, -1}, {1, 0}, {1, 1}};
  for (auto square : kings) {
    const int file = square.file().idx;
    const int rank = square.rank().idx;
    for (const auto& [df, dr] : kKingDeltas) {
      const int f = file + df;
      const int r = rank + dr;
      if (!IsOnBoard(f, r)) continue;
      AddAttack(counts, Square(File::FromIdx(f), Rank::FromIdx(r)));
    }
  }

  static constexpr std::array<std::pair<int, int>, 4> kBishopDirs = {
      std::pair<int, int>{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
  static constexpr std::array<std::pair<int, int>, 4> kRookDirs = {
      std::pair<int, int>{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
  static constexpr std::array<std::pair<int, int>, 8> kQueenDirs = {
      std::pair<int, int>{1, 0},  {-1, 0}, {0, 1},  {0, -1},
      {1, 1},  {1, -1}, {-1, 1}, {-1, -1}};

  for (auto square : bishops) {
    AddSlidingAttacks(counts, square, kBishopDirs, occupancy);
  }
  for (auto square : rooks) {
    AddSlidingAttacks(counts, square, kRookDirs, occupancy);
  }
  for (auto square : queens) {
    AddSlidingAttacks(counts, square, kQueenDirs, occupancy);
  }

  return counts;
}

bool IsPassedPawn(const BitBoard& enemy_pawns, Square pawn,
                  bool pawn_is_ours) {
  const int pawn_file = pawn.file().idx;
  const int pawn_rank = pawn.rank().idx;
  for (auto enemy : enemy_pawns) {
    const int enemy_file = enemy.file().idx;
    if (std::abs(enemy_file - pawn_file) > 1) continue;
    const int enemy_rank = enemy.rank().idx;
    if (pawn_is_ours && enemy_rank > pawn_rank) return false;
    if (!pawn_is_ours && enemy_rank < pawn_rank) return false;
  }
  return true;
}

bool IsSquareAttackedByUs(const ChessBoard& board, Square square, bool black_to_move) {
  ChessBoard mirrored = board;
  Square mirrored_square = square;
  if (!black_to_move) {
    mirrored.Mirror();
  }
  mirrored_square.Flip();
  return mirrored.IsUnderAttack(mirrored_square);
}

BitBoard OurDiscoveredChecks(const ChessBoard& board) {
  const Square enemy_king =
      SingleSquare(board.kings() & board.theirs());
  const BitBoard occupied = board.ours() | board.theirs();
  BitBoard result(0);

  static constexpr std::array<std::pair<int, int>, 8> kDirections = {
      std::pair<int, int>{1, 0}, {0, 1}, {-1, 0}, {0, -1},
      std::pair<int, int>{1, 1}, {-1, 1}, {1, -1}, {-1, -1}};

  for (const auto& [df, dr] : kDirections) {
    int f = enemy_king.file().idx + df;
    int r = enemy_king.rank().idx + dr;
    while (IsOnBoard(f, r)) {
      Square sq(File::FromIdx(f), Rank::FromIdx(r));
      if (occupied.get(sq)) {
        if (!board.ours().get(sq)) break;
        Square blocker = sq;
        f += df;
        r += dr;
        while (IsOnBoard(f, r)) {
          Square next_sq(File::FromIdx(f), Rank::FromIdx(r));
          if (occupied.get(next_sq)) {
            if (board.ours().get(next_sq)) {
              const bool is_orth = df == 0 || dr == 0;
              const bool is_diag = std::abs(df) == std::abs(dr);
                const bool is_rook_like = board.rooks().get(next_sq) ||
                            board.queens().get(next_sq);
                const bool is_bishop_like = board.bishops().get(next_sq) ||
                              board.queens().get(next_sq);
              if ((is_orth && is_rook_like) || (is_diag && is_bishop_like)) {
                // TODO: Identify if pawn can capture, if so, then identify it. 
                // If Ray is Vertical and Blocker is a Pawn, it can't step aside (ignoring captures)
                if (df == 0 && board.pawns().get(blocker)) {
                    break; // Don't mark result
                }
                result.set(blocker);
              }
            }
            break;
          }
          f += df;
          r += dr;
        }
        break;
      }
      f += df;
      r += dr;
    }
  }

  return result;
}

struct PieceAtSquare {
  PieceType type;
  bool is_ours;
  bool is_valid;
};


}  // namespace

Dn1Planes ComputeDn1Planes(const FrameType& frame) {
  const auto input_format =
      static_cast<pblczero::NetworkFormat::InputFormat>(frame.input_format);
  InputPlanes planes = PlanesFromTrainingData(frame);
  ChessBoard board;
  int rule50 = 0;
  int gameply = 0;
  PopulateBoard(input_format, planes, &board, &rule50, &gameply);
  
  const bool black_to_move =
      !IsCanonicalFormat(input_format) && frame.side_to_move_or_enpassant != 0;

  const BitBoard our_pieces = board.ours();
  const BitBoard their_pieces = board.theirs();
  //const BitBoard our_king = board.kings() & our_pieces;
  //const BitBoard their_king = board.kings() & their_pieces;

  Dn1Planes out;

  out.our_pins = board.GenerateKingAttackInfo().pinned_pieces_;

  {
    ChessBoard mirrored = board;
    mirrored.Mirror();
    out.their_pins = mirrored.GenerateKingAttackInfo().pinned_pieces_;
    out.their_pins.Mirror();
  }

  out.our_discovered_checks = OurDiscoveredChecks(board);

  const BitBoard our_pawns = board.pawns() & our_pieces;
  const BitBoard their_pawns = board.pawns() & their_pieces;
  for (auto square : our_pawns) {
    if (IsPassedPawn(their_pawns, square, true)) {
      out.our_passed_pawns.set(square);
    }
  }
  for (auto square : their_pawns) {
    if (IsPassedPawn(our_pawns, square, false)) {
      out.their_passed_pawns.set(square);
    }
  }

  const auto our_attacks = ComputeAttackCounts(board, our_pieces, true);
  const auto their_attacks = ComputeAttackCounts(board, their_pieces, false);

  for (auto square : our_pieces) {
    const int idx = square.as_idx();
    const int our_count = our_attacks[idx];
    const int their_count = their_attacks[idx];
    if (our_count == 0 && their_count > 0 ) {
        out.our_hanging.set(square);
    }
  }

  for (auto square : (our_pieces | their_pieces)) {
    const int idx = square.as_idx();
    const int our_count = our_attacks[idx];
    const int their_count = their_attacks[idx];
    
    if (their_count == 0 || our_count == 0) continue;
    if (our_count > their_count) {
      out.control_plus.set(square);
    } else if (our_count == their_count) {
      out.control_equal.set(square);
    } else {
      out.control_minus.set(square);
    }

  }

  const MoveList legal_moves = board.GenerateLegalMoves();
  for (const auto& move : legal_moves) {
    Square dest = move.to();

    ChessBoard copy = board;
    bool isZeroing = copy.ApplyMove(move);
    copy.Mirror();
    
    if (copy.IsUnderCheck()) {
      out.legal_checks.set(dest);
    }
    
    
    const bool is_capture =
        move.is_en_passant() || their_pieces.get(dest);
    if (!is_capture) continue;

    // This is a very expensive operation, an simplified approximation might be needed 
    const int see_value = board.StaticExchangeEvaluation(move);
    if (see_value >= kSeeThreshold) {
      out.see_positive.set(dest);
    } else if (see_value < kSeeThreshold && see_value >= 0) {
      out.see_equal.set(dest);
    } else if (see_value < 0) {
      out.see_negative.set(dest);
    }
  }

  return out;
}

}  // namespace lczero::training
