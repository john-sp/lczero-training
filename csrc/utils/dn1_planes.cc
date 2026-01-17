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

InputPlanes PlanesFromTrainingDataNoTransform(const FrameType& data) {
  InputPlanes result;
  result.reserve(112);
  for (int i = 0; i < 104; ++i) {
    result.emplace_back();
    result.back().mask = ReverseBitsInBytes(data.planes[i]);
  }
  switch (data.input_format) {
    case pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE: {
      result.emplace_back();
      result.back().mask = data.castling_us_ooo != 0 ? ~0LL : 0LL;
      result.emplace_back();
      result.back().mask = data.castling_us_oo != 0 ? ~0LL : 0LL;
      result.emplace_back();
      result.back().mask = data.castling_them_ooo != 0 ? ~0LL : 0LL;
      result.emplace_back();
      result.back().mask = data.castling_them_oo != 0 ? ~0LL : 0LL;
      break;
    }
    case pblczero::NetworkFormat::INPUT_112_WITH_CASTLING_PLANE:
    case pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION:
    case pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_HECTOPLIES:
    case pblczero::NetworkFormat::
        INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON:
    case pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_V2:
    case pblczero::NetworkFormat::
        INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON: {
      result.emplace_back();
      result.back().mask =
          data.castling_us_ooo |
          (static_cast<uint64_t>(data.castling_them_ooo) << 56);
      result.emplace_back();
      result.back().mask = data.castling_us_oo |
                           (static_cast<uint64_t>(data.castling_them_oo)
                            << 56);
      result.emplace_back();
      result.emplace_back();
      break;
    }
    default:
      throw Exception("Unsupported input plane encoding " +
                      std::to_string(data.input_format));
  }
  result.emplace_back();
  auto typed_format =
      static_cast<pblczero::NetworkFormat::InputFormat>(data.input_format);
  if (IsCanonicalFormat(typed_format)) {
    result.back().mask = static_cast<uint64_t>(data.side_to_move_or_enpassant)
                         << 56;
  } else {
    result.back().mask = data.side_to_move_or_enpassant != 0 ? ~0LL : 0LL;
  }
  result.emplace_back();
  if (IsHectopliesFormat(typed_format)) {
    result.back().Fill(data.rule50_count / 100.0f);
  } else {
    result.back().Fill(data.rule50_count);
  }
  result.emplace_back();
  if (IsCanonicalArmageddonFormat(typed_format) &&
      data.invariance_info >= 128) {
    result.back().SetAll();
  }
  result.emplace_back();
  result.back().SetAll();
  return result;
}

constexpr bool IsOnBoard(int file, int rank) {
  return file >= 0 && file < 8 && rank >= 0 && rank < 8;
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

bool IsSquareAttackedByUs(const ChessBoard& board, Square square) {
  ChessBoard mirrored = board;
  mirrored.Mirror();
  Square mirrored_square = square;
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
  InputPlanes planes = PlanesFromTrainingDataNoTransform(frame);
  ChessBoard board;
  int rule50 = 0;
  int gameply = 0;
  PopulateBoard(input_format, planes, &board, &rule50, &gameply);
  // Mirror board if it is black to move
  
  const BitBoard our_pieces = board.ours();
  const BitBoard their_pieces = board.theirs();
  //const BitBoard our_king = board.kings() & our_pieces;
  //const BitBoard their_king = board.kings() & their_pieces;
  if (frame.side_to_move_or_enpassant != 0) {
    // Swap our and their pieces
    std::swap(our_pieces, their_pieces);
  }

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

  const MoveList legal_moves = board.GenerateLegalMoves();
  for (const auto& move : legal_moves) {
    const bool is_capture =
        move.is_en_passant() || their_pieces.get(move.to());
    ChessBoard copy = board;
    copy.ApplyMove(move);
    // Check if move gives check to opponent's king. "Ours" always refers to white, so if side_to_move is black, we need to look at "our" king 
    const Square opponent_king = frame.side_to_move_or_enpassant ? SingleSquare(copy.kings() & copy.theirs()) : SingleSquare(copy.kings() & copy.ours());
    
    if (IsSquareAttackedByUs(copy, opponent_king)) {
      out.legal_checks.set(move.to());
    }
    if (!is_capture) continue;

    const int see_value = board.StaticExchangeEvaluation(move);
    if (see_value >= kSeeThreshold) {
      // LOG(INFO) << "Positive SEE move: " << move.ToString(false) << " value: " << see_value;
      out.see_positive.set(move.to());
    } else if (see_value < kSeeThreshold && see_value >= 0) {
        // LOG(INFO) << "Equal SEE move: " << move.ToString(false) << " value: " << see_value;
      out.see_equal.set(move.to());
    } else if (see_value < 0) {
        // LOG(INFO) << "Negative SEE move: " << move.ToString(false) << " value: " << see_value;
      out.see_negative.set(move.to());
    }
  }

  return out;
}

}  // namespace lczero::training
