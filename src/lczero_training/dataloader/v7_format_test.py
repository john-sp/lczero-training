"""Pure-python replica test of the V7TrainingData byte layout.

Verifies the byte offsets used by the C++ loader (csrc/loader/stages/
tensor_generator.cc and chunk_rescorer.cc) against the packed struct
definitions in libs/lc0/src/trainingdata/trainingdata_v6.h and
trainingdata_v7.h (both are #pragma pack(1)):

  V6TrainingData: 8356 bytes
    offset    0: uint32 version
    offset    4: uint32 input_format
    offset    8: float probabilities[1858]
    offset 7440: uint64 planes[104]
    offset 8272: 8x uint8 (castlings, stm, rule50, invariance, dummy)
    offset 8280: 15x float root_q..orig_m
    offset 8340: uint32 visits
    offset 8344: uint16 played_idx
    offset 8346: uint16 best_idx
    offset 8348: float policy_kld
    offset 8352: float q_st
  V7TrainingData tail (total 8396 bytes):
    offset 8356: float d_st
    offset 8360: uint16 opp_played_idx
    offset 8362: uint16 next_played_idx
    offset 8364: float reserved[8]   -> reserved[k] at 8364 + 4*k

Reserved-slot contract (filled by the offline rescorer):
    reserved[0] = provenance (0 none / 1 TB / 2 noise-debl / 3 unint-debl)
    reserved[1] = q_st_censored
    reserved[2] = d_st_censored
    reserved[3] = played-move child-Q (NaN when none)
    reserved[4..7] = 0
"""

import math
import struct

import numpy as np

V6_SIZE = 8356
V7_SIZE = 8396

# Packed little-endian layout mirroring the C++ structs field by field.
V6_FORMAT = "<" + "II" + "1858f" + "104Q" + "8B" + "15f" + "I" + "HH" + "ff"
V7_TAIL_FORMAT = "<" + "f" + "HH" + "8f"

Q_ST_OFFSET = 8352
D_ST_OFFSET = 8356
OPP_PLAYED_IDX_OFFSET = 8360
NEXT_PLAYED_IDX_OFFSET = 8362
RESERVED_OFFSET = 8364
PLAYED_IDX_OFFSET = 8344
VERSION_OFFSET = 0
INVALID_MOVE_INDEX = 65535


def _make_v7_record(
    version: int,
    q_st: float,
    d_st: float,
    played_idx: int,
    opp_played_idx: int,
    next_played_idx: int,
    reserved: list,
) -> bytes:
    v6 = struct.pack(
        V6_FORMAT,
        version,  # version
        3,  # input_format
        *([0.0] * 1858),  # probabilities
        *([0] * 104),  # planes
        *([0] * 8),  # uint8 block
        *([0.0] * 15),  # root_q .. orig_m
        0,  # visits
        played_idx,  # played_idx
        0,  # best_idx
        0.0,  # policy_kld
        q_st,  # q_st
    )
    tail = struct.pack(
        V7_TAIL_FORMAT,
        d_st,
        opp_played_idx,
        next_played_idx,
        *reserved,
    )
    return v6 + tail


def _f32(record: bytes, offset: int) -> float:
    return float(np.frombuffer(record, dtype="<f4", count=1, offset=offset)[0])


def _u16(record: bytes, offset: int) -> int:
    return int(np.frombuffer(record, dtype="<u2", count=1, offset=offset)[0])


def _has_reserved_data(record: bytes) -> bool:
    """Replica of HasReservedData() in tensor_generator.cc."""
    version = int(
        np.frombuffer(record, dtype="<u4", count=1, offset=VERSION_OFFSET)[0]
    )
    if version < 7:
        return False
    for k in range(4):
        v = _f32(record, RESERVED_OFFSET + 4 * k)
        if math.isnan(v) or v != 0.0:
            return True
    return False


def _extract_row6_and_aux(record: bytes) -> tuple:
    """Replica of the row-6/aux extraction in tensor_generator.cc."""
    version = int(
        np.frombuffer(record, dtype="<u4", count=1, offset=VERSION_OFFSET)[0]
    )
    if _has_reserved_data(record):
        row6 = (
            _f32(record, RESERVED_OFFSET + 4),
            _f32(record, RESERVED_OFFSET + 8),
        )
        aux_targets = (
            _f32(record, RESERVED_OFFSET),
            _f32(record, RESERVED_OFFSET + 12),
        )
    else:
        row6 = (_f32(record, Q_ST_OFFSET), _f32(record, D_ST_OFFSET))
        aux_targets = (0.0, float("nan"))
    if version >= 7:
        opp = _u16(record, OPP_PLAYED_IDX_OFFSET)
        nxt = _u16(record, NEXT_PLAYED_IDX_OFFSET)
    else:
        opp = INVALID_MOVE_INDEX
        nxt = INVALID_MOVE_INDEX
    played = _u16(record, PLAYED_IDX_OFFSET)
    return row6, (opp, nxt, played), aux_targets


class TestV7StructLayout:
    def test_struct_sizes(self) -> None:
        assert struct.calcsize(V6_FORMAT) == V6_SIZE
        assert struct.calcsize(V6_FORMAT + V7_TAIL_FORMAT[1:]) == V7_SIZE

    def test_reserved_slot_offsets(self) -> None:
        # d_st, opp_played_idx, next_played_idx, reserved offsets follow
        # directly after the 8356-byte V6 struct.
        assert D_ST_OFFSET == V6_SIZE
        assert OPP_PLAYED_IDX_OFFSET == V6_SIZE + 4
        assert NEXT_PLAYED_IDX_OFFSET == V6_SIZE + 6
        assert RESERVED_OFFSET == V6_SIZE + 8
        for k in range(8):
            assert RESERVED_OFFSET + 4 * k == 8364 + 4 * k
        assert RESERVED_OFFSET + 8 * 4 == V7_SIZE

    def test_field_roundtrip_at_offsets(self) -> None:
        reserved = [2.0, -0.5, 0.25, -0.75, 0.0, 0.0, 0.0, 0.0]
        record = _make_v7_record(
            version=7,
            q_st=0.11,
            d_st=0.22,
            played_idx=789,
            opp_played_idx=123,
            next_played_idx=456,
            reserved=reserved,
        )
        assert len(record) == V7_SIZE
        np.testing.assert_allclose(_f32(record, Q_ST_OFFSET), 0.11, rtol=1e-6)
        np.testing.assert_allclose(_f32(record, D_ST_OFFSET), 0.22, rtol=1e-6)
        assert _u16(record, PLAYED_IDX_OFFSET) == 789
        assert _u16(record, OPP_PLAYED_IDX_OFFSET) == 123
        assert _u16(record, NEXT_PLAYED_IDX_OFFSET) == 456
        for k in range(8):
            np.testing.assert_allclose(
                _f32(record, RESERVED_OFFSET + 4 * k), reserved[k], rtol=1e-6
            )


class TestTensorExtractionReplica:
    def test_v7_reserved_passthrough(self) -> None:
        record = _make_v7_record(
            version=7,
            q_st=0.11,
            d_st=0.22,
            played_idx=789,
            opp_played_idx=123,
            next_played_idx=456,
            reserved=[2.0, -0.5, 0.25, -0.75, 0.0, 0.0, 0.0, 0.0],
        )
        row6, aux_indices, aux_targets = _extract_row6_and_aux(record)
        np.testing.assert_allclose(row6, (-0.5, 0.25), rtol=1e-6)
        assert aux_indices == (123, 456, 789)
        np.testing.assert_allclose(aux_targets, (2.0, -0.75), rtol=1e-6)

    def test_v7_nan_child_q_and_sentinels_passthrough(self) -> None:
        record = _make_v7_record(
            version=7,
            q_st=0.5,
            d_st=0.1,
            played_idx=17,
            opp_played_idx=INVALID_MOVE_INDEX,
            next_played_idx=INVALID_MOVE_INDEX,
            reserved=[1.0, 0.9, 0.05, float("nan"), 0.0, 0.0, 0.0, 0.0],
        )
        row6, aux_indices, aux_targets = _extract_row6_and_aux(record)
        np.testing.assert_allclose(row6, (0.9, 0.05), rtol=1e-6)
        assert aux_indices == (
            INVALID_MOVE_INDEX,
            INVALID_MOVE_INDEX,
            17,
        )
        np.testing.assert_allclose(aux_targets[0], 1.0)
        assert math.isnan(aux_targets[1])

    def test_v6_frame_falls_back_to_plain_st(self) -> None:
        record = _make_v7_record(
            version=6,
            q_st=0.33,
            d_st=0.44,
            played_idx=42,
            opp_played_idx=0,
            next_played_idx=0,
            reserved=[0.0] * 8,
        )
        row6, aux_indices, aux_targets = _extract_row6_and_aux(record)
        np.testing.assert_allclose(row6, (0.33, 0.44), rtol=1e-6)
        assert aux_indices == (INVALID_MOVE_INDEX, INVALID_MOVE_INDEX, 42)
        np.testing.assert_allclose(aux_targets[0], 0.0)
        assert math.isnan(aux_targets[1])

    def test_v7_zeroed_reserved_falls_back_to_plain_st(self) -> None:
        record = _make_v7_record(
            version=7,
            q_st=-0.6,
            d_st=0.3,
            played_idx=7,
            opp_played_idx=5,
            next_played_idx=INVALID_MOVE_INDEX,
            reserved=[0.0] * 8,
        )
        row6, aux_indices, aux_targets = _extract_row6_and_aux(record)
        np.testing.assert_allclose(row6, (-0.6, 0.3), rtol=1e-6)
        assert aux_indices == (5, INVALID_MOVE_INDEX, 7)
        assert math.isnan(aux_targets[1])
