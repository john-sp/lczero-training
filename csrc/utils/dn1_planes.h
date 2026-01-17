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

#include "chess/bitboard.h"
#include "loader/frame_type.h"

namespace lczero::training {

// Planes usable: 91-103 (inclusive)
constexpr int kDn1FirstPlane = 91;
constexpr int kDn1LastPlane = 103;

constexpr int kDn1SeePositivePlane = 91;
constexpr int kDn1SeeEqualPlane = 92;
constexpr int kDn1SeeNegativePlane = 93;

constexpr int kDn1OurPinsPlane = 94;
constexpr int kDn1TheirPinsPlane = 95;
constexpr int kDn1OurDiscoveredChecksPlane = 96;

constexpr int kDn1OurPassedPawnsPlane = 98;
constexpr int kDn1TheirPassedPawnsPlane = 99;

constexpr int kDn1LegalChecksPlane = 100;


constexpr int kSeeThreshold = 1;

struct Dn1Planes {
  BitBoard our_pins{0};
  BitBoard their_pins{0};
  BitBoard our_passed_pawns{0};
  BitBoard their_passed_pawns{0};
  BitBoard legal_checks{0};
  BitBoard our_discovered_checks{0};
  BitBoard see_positive{0};
  BitBoard see_equal{0};
  BitBoard see_negative{0};
};

Dn1Planes ComputeDn1Planes(const FrameType& frame);

}  // namespace lczero::training
