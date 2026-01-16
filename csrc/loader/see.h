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

#pragma once

#include "chess/types.h"

namespace lczero::training {

constexpr int kSeeThreshold = 1;

constexpr int kSeePawnValue = 208;
constexpr int kSeeKnightValue = 781;
constexpr int kSeeBishopValue = 825;
constexpr int kSeeRookValue = 1276;
constexpr int kSeeQueenValue = 2538;
constexpr int kSeeKingValue = 0;

inline int SeePieceValue(PieceType type) {
  switch (type.idx) {
    case kPawn.idx:
      return kSeePawnValue;
    case kKnight.idx:
      return kSeeKnightValue;
    case kBishop.idx:
      return kSeeBishopValue;
    case kRook.idx:
      return kSeeRookValue;
    case kQueen.idx:
      return kSeeQueenValue;
    default:
      return kSeeKingValue;
  }
}

}  // namespace lczero::training
