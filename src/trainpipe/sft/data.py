# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset

from trainpipe.common.jsonl_io import iter_jsonl


def load_progress_dict(path: str, id_key: str, progress_key: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for obj in iter_jsonl(path):
        _id = obj.get(id_key)
        txt = obj.get(progress_key)
        if _id is None or txt is None:
            continue
        _id = str(_id)
        txt = str(txt)
        if txt.strip():
            out[_id] = txt
    if not out:
        raise RuntimeError(f"No valid progress notes loaded from: {path}")
    return out


def load_annotation_dict(path: str, id_key: str, annotation_key: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for obj in iter_jsonl(path):
        _id = obj.get(id_key)
        ann = obj.get(annotation_key)
        if _id is None or ann is None:
            continue
        _id = str(_id)
        ann = str(ann)
        if ann.strip():
            out[_id] = ann
    if not out:
        raise RuntimeError(f"No valid annotations loaded from: {path}")
    return out


def build_pairs_by_id(
    progress: Dict[str, str],
    ann: Dict[str, str],
    max_samples: int = 0,
) -> List[Tuple[str, str, str]]:
    ids = sorted(set(progress.keys()) & set(ann.keys()))
    if not ids:
        raise RuntimeError("No matched ids between progress_jsonl and annotations_jsonl")
    if max_samples and max_samples > 0:
        ids = ids[:max_samples]
    return [(_id, progress[_id], ann[_id]) for _id in ids]


def split_indices(
    n: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("Need at least 2 samples")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0, 1)")
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError("test_ratio must be in [0, 1)")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_test = int(round(n * test_ratio)) if test_ratio > 0 else 0
    n_val = int(round(n * val_ratio))

    if test_ratio > 0:
        n_test = max(1, n_test)
    if val_ratio > 0:
        n_val = max(1, n_val)
    else:
        n_val = 0

    max_test = n - 2
    if test_ratio > 0:
        n_test = min(n_test, max_test)
    else:
        n_test = 0

    max_val = n - 1 - n_test
    n_val = min(n_val, max_val)

    test_idx = idx[:n_test] if n_test > 0 else np.array([], dtype=int)
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]

    return train_idx, val_idx, test_idx


def kfold_indices(
    n: int,
    num_folds: int,
    fold: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("Need at least 2 samples")
    if num_folds < 2:
        raise ValueError("num_folds must be >= 2")
    if fold < 0 or fold >= num_folds:
        raise ValueError("fold must be in [0, num_folds)")

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    folds = np.array_split(idx, num_folds)
    val_idx = folds[fold]
    train_idx = np.concatenate([f for i, f in enumerate(folds) if i != fold])

    if len(val_idx) < 1:
        raise ValueError("Validation fold is empty")
    if len(train_idx) < 1:
        raise ValueError("Train split is empty")

    return train_idx, val_idx


def kfold_split_all(
    n: int,
    num_folds: int,
    seed: int,
) -> List[np.ndarray]:
    """Return a list of fold index arrays after shuffling by seed."""
    if n < 2:
        raise ValueError("Need at least 2 samples")
    if num_folds < 2:
        raise ValueError("num_folds must be >= 2")

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return list(np.array_split(idx, num_folds))


def cv_train_val_test_indices(
    n: int,
    num_folds: int,
    test_fold: int,
    split_seed: int,
    val_size: int,
    val_seed_base: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """5-fold style CV: 1 fold = test, remaining folds = train pool, sample val from train pool.

    - test = the specified fold
    - train_pool = all other folds
    - val = sample val_size examples from train_pool using seed (val_seed_base + test_fold)
    - train = train_pool \ val

    This matches the described protocol.
    """
    if test_fold < 0 or test_fold >= num_folds:
        raise ValueError("test_fold must be in [0, num_folds)")
    if val_size < 1:
        raise ValueError("val_size must be >= 1")

    folds = kfold_split_all(n=n, num_folds=num_folds, seed=split_seed)
    test_idx = folds[test_fold]

    train_pool = np.concatenate([f for i, f in enumerate(folds) if i != test_fold])
    if len(train_pool) < 2:
        raise ValueError("Train pool too small")

    desired = min(int(val_size), len(train_pool) - 1)
    desired = max(1, desired)

    rng = np.random.default_rng(int(val_seed_base) + int(test_fold))
    perm = rng.permutation(len(train_pool))
    val_pos = perm[:desired]
    train_pos = perm[desired:]

    val_idx = train_pool[val_pos]
    train_idx = train_pool[train_pos]

    return train_idx, val_idx, test_idx


def cv_indices(
    n: int,
    num_folds: int,
    fold: int,
    split_seed: int,
    val_ratio: Optional[float] = None,
    val_size: int = 0,
    val_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cross-validation split with optional smaller validation subset.

    Default behavior (recommended):
      - If val_ratio is None and val_size==0: use full fold as validation (classic K-fold).

    If you want a smaller validation set than 1/num_folds (e.g., n=95, k=5 => 19 is too big),
    you can set either:
      - val_size (absolute number of validation samples)
      - or val_ratio (ratio over the whole dataset)

    In that case, we take the held-out fold, then sample *val_size* items from it as validation
    (deterministically), and the remaining items of the held-out fold are moved back into train.

    Notes:
      - No stratification, no grouping.
      - val_size is clipped to [1, len(fold)].
    """
    train_idx, fold_idx = kfold_indices(n=n, num_folds=num_folds, fold=fold, seed=split_seed)

    if (val_ratio is None or val_ratio <= 0.0) and (val_size is None or val_size <= 0):
        return train_idx, fold_idx

    if val_seed is None:
        val_seed = split_seed + int(fold)

    if val_size and val_size > 0:
        desired = int(val_size)
    else:
        desired = int(round(float(n) * float(val_ratio)))

    desired = max(1, desired)
    desired = min(desired, len(fold_idx))

    rng = np.random.default_rng(int(val_seed))
    perm = rng.permutation(len(fold_idx))
    val_pos = perm[:desired]
    rest_pos = perm[desired:]

    val_idx = fold_idx[val_pos]
    # move the remaining portion of the held-out fold back into train
    if len(rest_pos) > 0:
        train_idx = np.concatenate([train_idx, fold_idx[rest_pos]])

    return train_idx, val_idx


def build_raw_splits_from_pairs(
    pairs: List[Tuple[str, str, str]],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: Optional[np.ndarray] = None,
):
    def idx_to_raw(idxs: np.ndarray) -> List[Dict]:
        out: List[Dict] = []
        for i in idxs.tolist():
            _id, note, a = pairs[i]
            out.append({"id": _id, "progress_note": note, "annotation": a})
        return out

    train_raw = idx_to_raw(train_idx)
    val_raw = idx_to_raw(val_idx)
    test_raw = idx_to_raw(test_idx) if test_idx is not None and len(test_idx) > 0 else []
    return train_raw, val_raw, test_raw


def raw_to_hf_dataset(raw: List[Dict]) -> Dataset:
    return Dataset.from_list(raw)
