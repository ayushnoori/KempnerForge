"""Completion-masked SFT dataset for KempnerForge.

``MaskedSftDataset`` backs supervised fine-tuning on pre-tokenized, per-example
chat sequences whose prompt tokens are masked out of the loss. Unlike
``MemoryMappedDataset`` (flat token streams chunked into fixed windows for causal
LM), each row here is a complete SFT example produced upstream with a paired label
row: a ``*.tokens.npy`` shard of shape ``[n, seq_len]`` (uint16/uint32 token ids,
right-padded) and a sibling ``*.labels.npy`` of the same shape (int, ``-100`` on
prompt/pad positions, the true token id on supervised completion positions).

Selected by ``data.masked_sft=true`` in the training config. Like the other
datasets it is map-style and **pre-shifts** each example so the training loop can
call ``loss_fn(logits, labels)`` directly: ``input_ids = tokens[:-1]`` and
``labels = labels[1:]``. The loss is cross-entropy with ``ignore_index=-100``, so
the masked prompt/pad positions contribute no gradient (completion-only SFT).
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MaskedSftDataset(Dataset):
    """Pre-tokenized, completion-masked SFT dataset backed by mmapped numpy files.

    Expects paired 2D ``.npy`` files of shape ``[n, seq_len]``: a token shard and a
    label shard whose name is the token shard's with ``token_suffix`` replaced by
    ``labels_suffix``. Rows are independent examples (no packing / no cross-row
    chunking); multiple shards are concatenated logically.

    Args:
        data_dir: Directory containing the ``*.tokens.npy`` (+ ``*.labels.npy``) shards.
        seq_len: Fixed per-example row width; every shard must be ``[n, seq_len]``.
        file_pattern: Glob for the token shards.
        token_suffix: Token-shard filename suffix (mapped to ``labels_suffix``).
        labels_suffix: Label-shard filename suffix.
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        file_pattern: str = "*.tokens.npy",
        token_suffix: str = ".tokens.npy",
        labels_suffix: str = ".labels.npy",
    ) -> None:
        self.seq_len = seq_len
        self._tmaps: list[np.ndarray] = []
        self._lmaps: list[np.ndarray] = []
        self._cumulative_samples: list[int] = [0]

        data_path = Path(data_dir)
        self._token_files = sorted(data_path.glob(file_pattern))
        if not self._token_files:
            raise FileNotFoundError(f"No files matching {file_pattern!r} in {data_dir}")

        # Memory-map every token shard + its sibling label shard, validating shapes.
        # If any open fails partway, release what we already mapped so the mmaps don't
        # leak via the exception traceback (pytest / logger.exception pin the partial self).
        total_examples = 0
        try:
            for tf in self._token_files:
                lf = tf.with_name(tf.name.replace(token_suffix, labels_suffix))
                if not lf.exists():
                    raise FileNotFoundError(f"Missing labels file for {tf.name} (expected {lf.name})")
                tmap = np.load(str(tf), mmap_mode="r")
                lmap = np.load(str(lf), mmap_mode="r")
                if tmap.ndim != 2 or tmap.shape[1] != seq_len:
                    raise ValueError(f"{tf.name}: expected [n, {seq_len}], got {tmap.shape}")
                if lmap.shape != tmap.shape:
                    raise ValueError(f"{tf.name}: labels shape {lmap.shape} != tokens shape {tmap.shape}")
                self._tmaps.append(tmap)
                self._lmaps.append(lmap)
                total_examples += tmap.shape[0]
                self._cumulative_samples.append(self._cumulative_samples[-1] + tmap.shape[0])
        except Exception:
            self._close_mmaps()
            raise

        self._total_samples = self._cumulative_samples[-1]
        logger.info(
            f"MaskedSftDataset: {len(self._token_files)} files, "
            f"{total_examples:,} examples (seq_len={seq_len})"
        )

        # State for resumption (parity with MemoryMappedDataset)
        self._epoch = 0

    def __len__(self) -> int:
        return self._total_samples

    def _find_file(self, idx: int) -> int:
        """Binary search for the shard index containing global example idx."""
        lo, hi = 0, len(self._token_files) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cumulative_samples[mid + 1] <= idx:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= self._total_samples:
            raise IndexError(f"Index {idx} out of range [0, {self._total_samples})")

        file_idx = self._find_file(idx)
        local_idx = idx - self._cumulative_samples[file_idx]

        tokens = self._tmaps[file_idx][local_idx].astype(np.int64)
        labels = self._lmaps[file_idx][local_idx].astype(np.int64)
        token_tensor = torch.from_numpy(tokens.copy())
        label_tensor = torch.from_numpy(labels.copy())

        # Pre-shift for next-token prediction (as MemoryMappedDataset does), but take
        # the targets from the SEPARATE label row so prompt/pad stay -100. The training
        # loop's cross-entropy uses ignore_index=-100 -> completion-only supervision.
        return {
            "input_ids": token_tensor[:-1],
            "labels": label_tensor[1:],
        }

    def state_dict(self) -> dict:
        """Return checkpoint state. Keys: ``epoch``, ``total_samples``."""
        return {"epoch": self._epoch, "total_samples": self._total_samples}

    def load_state_dict(self, state: dict) -> None:
        """Restore from checkpoint. Only ``epoch`` is restored; sample count is derived."""
        self._epoch = state.get("epoch", 0)

    def _close_mmaps(self) -> None:
        """Release the underlying token + label mmap objects. Idempotent."""
        for mm in [*self._tmaps, *self._lmaps]:
            inner = getattr(mm, "_mmap", None)
            if inner is not None and not inner.closed:
                # BufferError: live views into the mapping still exist — drop the ref
                # and let GC finish it. ValueError: already closed by another path.
                with contextlib.suppress(BufferError, ValueError):
                    inner.close()
        self._tmaps.clear()
        self._lmaps.clear()

    def close(self) -> None:
        """Release the underlying mmaps. Preferred path; do not rely on ``__del__``."""
        self._close_mmaps()

    def __del__(self) -> None:
        """GC safety net only. Prefer explicit :meth:`close`."""
        with contextlib.suppress(Exception):
            self._close_mmaps()
