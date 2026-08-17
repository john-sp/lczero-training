from collections.abc import Sequence


def best_candidate_index(elo_diffs: Sequence[float]) -> int | None:
    """Returns the best improving candidate, or None to retain the base."""
    if not elo_diffs:
        raise ValueError("At least one random-search candidate is required.")
    best_idx = max(range(len(elo_diffs)), key=elo_diffs.__getitem__)
    if elo_diffs[best_idx] <= 0.0:
        return None
    return best_idx
