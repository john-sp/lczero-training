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

#include "chess/board.h"
#include "loader/frame_type.h"
#include "neural/decoder.h"
#include "neural/encoder.h"
#include "trainingdata/reader.h"

namespace lczero::training {

Dn1Planes ComputeDn1Planes(const FrameType& frame) {
  const auto input_format =
      static_cast<pblczero::NetworkFormat::InputFormat>(frame.input_format);
  InputPlanes planes = PlanesFromTrainingData(frame);
  ChessBoard board;
  int rule50 = 0;
  int gameply = 0;
  PopulateBoard(input_format, planes, &board, &rule50, &gameply);

  TacticalInfo tactical = board.ComputeTacticalInfo();

  Dn1Planes out;
  out.our_pins = tactical.our_pins;
  out.their_pins = tactical.their_pins;
  out.our_discovered_checks = tactical.our_discovered_checks;
  out.our_passed_pawns = tactical.our_passed_pawns;
  out.their_passed_pawns = tactical.their_passed_pawns;
  out.our_hanging = tactical.our_hanging;
  out.control_plus = tactical.control_plus;
  out.control_equal = tactical.control_equal;
  out.control_minus = tactical.control_minus;
  out.see_positive = tactical.see_positive;
  out.see_equal = tactical.see_equal;
  out.see_negative = tactical.see_negative;
  out.legal_checks = tactical.legal_checks;
  return out;
}

}  // namespace lczero::training
