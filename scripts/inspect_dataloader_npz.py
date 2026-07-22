"""Inspect inputs from dataloader_probe .npz outputs."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np


def _format_value(value: Any) -> str:
    try:
        return f"{float(value): .6g}"
    except (TypeError, ValueError):
        return str(value)


def _print_plane(plane: np.ndarray, plane_index: int) -> None:
    print(f"  plane[{plane_index:3d}]:")
    for row_index in range(plane.shape[0]):
        row = plane[row_index]
        row_text = ", ".join(_format_value(v) for v in row)
        print(f"    [{row_index:2d}]: {row_text}")


def _print_inputs(inputs: np.ndarray, batch_index: int, max_planes: int) -> None:
    if inputs.ndim == 4:
        inputs = inputs[batch_index]
    if inputs.ndim != 3 or inputs.shape[1:] != (8, 8):
        raise ValueError(
            "Expected inputs with shape [planes, 8, 8] or "
            f"[batch, planes, 8, 8], got {inputs.shape}"
        )
    plane_count = inputs.shape[0]
    print(
        f"inputs: planes={plane_count} shape={inputs.shape} "
        f"dtype={inputs.dtype}"
    )
    limit = min(plane_count, max_planes)
    for plane_index in range(limit):
        _print_plane(inputs[plane_index], plane_index)
    if limit < plane_count:
        print(f"  ... {plane_count - limit} planes omitted")


def _plane_to_bitboard_mask(plane: np.ndarray) -> int:
    if plane.shape != (8, 8):
        raise ValueError(f"Expected plane shape (8, 8), got {plane.shape}")
    mask = 0
    for rank in range(8):
        for file in range(8):
            if plane[rank, file] != 0:
                idx = rank * 8 + file
                mask |= 1 << idx
    return mask


def _mask_to_squares(mask: int, *, reverse_ranks: bool = False) -> list[str]:
    squares: list[str] = []
    files = "abcdefgh"
    for rank in range(8):
        for file in range(8):
            idx = rank * 8 + file
            if mask & (1 << idx):
                if reverse_ranks:
                    display_rank = 8 - rank
                else:
                    display_rank = rank + 1
                squares.append(f"{files[file]}{display_rank}")
    return squares


def _fen_from_inputs(inputs: np.ndarray) -> str:
    planes = inputs
    if planes.shape[0] < 112:
        raise ValueError(
            f"Expected at least 112 planes, got {planes.shape[0]}"
        )
    
    def _plane_has_any(plane_idx: int) -> bool:
        return np.any(planes[plane_idx] != 0)

    black_to_move = _plane_has_any(108)

    piece_masks = [_plane_to_bitboard_mask(planes[i]) for i in range(12)]
    if black_to_move:
        piece_order = [
            (piece_masks[0], "p"),
            (piece_masks[1], "n"),
            (piece_masks[2], "b"),
            (piece_masks[3], "r"),
            (piece_masks[4], "q"),
            (piece_masks[5], "k"),
            (piece_masks[6], "P"),
            (piece_masks[7], "N"),
            (piece_masks[8], "B"),
            (piece_masks[9], "R"),
            (piece_masks[10], "Q"),
            (piece_masks[11], "K"),
        ]
        rank_range = range(8)
    else:
        piece_order = [
            (piece_masks[0], "P"),
            (piece_masks[1], "N"),
            (piece_masks[2], "B"),
            (piece_masks[3], "R"),
            (piece_masks[4], "Q"),
            (piece_masks[5], "K"),
            (piece_masks[6], "p"),
            (piece_masks[7], "n"),
            (piece_masks[8], "b"),
            (piece_masks[9], "r"),
            (piece_masks[10], "q"),
            (piece_masks[11], "k"),
        ]
        rank_range = range(7, -1, -1)

    rows: list[str] = []
    for rank in rank_range:
        empty = 0
        row = ""
        for file in range(8):
            idx = rank * 8 + file
            piece = ""
            for mask, symbol in piece_order:
                if mask & (1 << idx):
                    piece = symbol
                    break
            if piece:
                if empty:
                    row += str(empty)
                    empty = 0
                row += piece
            else:
                empty += 1
        if empty:
            row += str(empty)
        rows.append(row)

    board = "/".join(rows)

    # Classic 112-plane aux mapping.
    castling = ""
    if _plane_has_any(104):
        castling += "Q"
    if _plane_has_any(105):
        castling += "K"
    if _plane_has_any(106):
        castling += "q"
    if _plane_has_any(107):
        castling += "k"
    if not castling:
        castling = "-"

    side = "b" if black_to_move else "w"

    en_passant = "-"

    rule50 = int(round(float(np.mean(planes[109]))))
    fullmove = max(1, rule50)

    return f"{board} {side} {castling} {en_passant} {rule50} {fullmove}"


def _print_plane_squares(planes: np.ndarray, plane_start: int, plane_end: int) -> None:
    reverse_ranks = False
    if planes.shape[0] > 108:
        reverse_ranks = np.any(planes[108] != 0)
    for plane_idx in range(plane_start, plane_end + 1):
        mask = _plane_to_bitboard_mask(planes[plane_idx])
        squares = _mask_to_squares(mask, reverse_ranks=reverse_ranks)
        print(f"plane[{plane_idx}]: {', '.join(squares) if squares else '-'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect inputs from dataloader_probe .npz files."
    )
    parser.add_argument(
        "npz_path",
        type=str,
        help="Path to .npz created by dataloader_probe.",
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        default=0,
        help="Batch index to print when inputs include a batch dimension.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=1,
        help="Maximum number of batches to print.",
    )
    parser.add_argument(
        "--max-planes",
        type=int,
        default=112,
        help="Maximum number of planes to print per batch.",
    )
    parser.add_argument(
        "--print-fen",
        action="store_true",
        help="Print FEN reconstructed from planes 0-11 and aux planes.",
    )
    parser.add_argument(
        "--highlight-planes",
        type=str,
        default="91-103",
        help="Plane range to list highlighted squares (e.g., 91-103).",
    )
    args = parser.parse_args(argv)

    archive = np.load(args.npz_path, allow_pickle=True)
    batches = archive["batches"]
    print(f"Loaded {len(batches)} batches from {args.npz_path}")

    for index, batch in enumerate(batches[: args.max_batches]):
        inputs = None
        if isinstance(batch, dict):
            inputs = batch.get("inputs")
        elif hasattr(batch, "inputs"):
            inputs = getattr(batch, "inputs")
        if inputs is None:
            keys = []
            if hasattr(batch, "keys"):
                keys = list(batch.keys())  # type: ignore[assignment]
            print(f"batch[{index}]: inputs not found. Keys={keys}")
            continue
        if not isinstance(inputs, np.ndarray):
            inputs = np.asarray(inputs)
        _print_inputs(inputs, args.batch_index, args.max_planes)
        inputs_for_decode = inputs
        if inputs_for_decode.ndim == 4:
            inputs_for_decode = inputs_for_decode[args.batch_index]
        if args.print_fen:
            fen = _fen_from_inputs(inputs_for_decode)
            print(f"FEN: {fen}")
        if args.highlight_planes:
            if "-" in args.highlight_planes:
                start_text, end_text = args.highlight_planes.split("-", 1)
                plane_start = int(start_text)
                plane_end = int(end_text)
            else:
                plane_start = plane_end = int(args.highlight_planes)
            _print_plane_squares(inputs_for_decode, plane_start, plane_end)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
