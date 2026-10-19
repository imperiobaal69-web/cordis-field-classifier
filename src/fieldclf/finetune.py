"""Fine-tune a small transformer encoder end-to-end for multi-label classification.

A plain PyTorch loop (no Trainer) so every choice is visible:

* binary cross-entropy over all labels (independent sigmoids, not a softmax);
* AdamW, linear warm-up then linear decay, gradient clipping at 1.0;
* model selection: the epoch with the best validation macro PR-AUC is kept;
* post-hoc temperature scaling fitted on validation logits, because fine-tuned
  networks are usually over-confident and the API exposes scores.

The same encoder is used frozen in :class:`fieldclf.models.EmbeddingLinear`, so
frozen vs fine-tuned is a controlled comparison.
"""

from __future__ import annotations

import copy
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from fieldclf.models import Texts, _device


def _seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class FineTuned:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        max_length: int = 256,
        epochs: int = 3,
        batch_size: int = 32,
        lr: float = 5e-5,
        warmup: float = 0.06,
        seed: int = 13,
    ) -> None:
        self.model_name = model_name
        self.name = "finetune:" + model_name.split("/")[-1]
        self.params: dict[str, object] = {
            "encoder": model_name,
            "max_length": max_length,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "warmup": warmup,
            "seed": seed,
        }
        self.max_length, self.epochs, self.batch_size = max_length, epochs, batch_size
        self.lr, self.warmup, self.seed = lr, warmup, seed
        self.temperature = 1.0
        self.weights_dir: Path | None = None
        self._net: Any = None
        self._tok: Any = None

    # -- persistence -------------------------------------------------------
    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_net"] = state["_tok"] = None
        state["weights_dir"] = None
        return state

    def export(self, directory: Path) -> None:
        assert self._net is not None and self._tok is not None
        self._net.save_pretrained(directory)
        self._tok.save_pretrained(directory)

    def attach(self, directory: Path) -> None:
        self.weights_dir = directory

    def _ensure_loaded(self) -> None:
        if self._net is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.weights_dir is None:
            raise RuntimeError("weights not attached; load with Classifier.load()")
        self._tok = AutoTokenizer.from_pretrained(self.weights_dir)
        self._net = AutoModelForSequenceClassification.from_pretrained(self.weights_dir)
        self._net.to(_device()).eval()

    # -- inference ---------------------------------------------------------
    def _logits(self, texts: Texts, batch_size: int = 64) -> np.ndarray:
        import torch

        assert self._net is not None and self._tok is not None
        device = next(self._net.parameters()).device
        out = []
        self._net.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self._tok(
                    texts[i : i + batch_size],
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                ).to(device)
                out.append(self._net(**enc).logits.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self._net.config.num_labels))

    def predict_proba(self, texts: Texts) -> np.ndarray:
        self._ensure_loaded()
        return 1.0 / (1.0 + np.exp(-self._logits(texts) / self.temperature))

    # -- training ----------------------------------------------------------
    def fit(self, texts: Texts, y: np.ndarray, val_texts: Texts, val_y: np.ndarray) -> FineTuned:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )

        _seed(self.seed)
        device = _device()
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._net = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=y.shape[1], problem_type="multi_label_classification"
        ).to(device)

        # Tokenise once. Batches are padded to a fixed length: on Apple MPS every new
        # input shape compiles and caches a new graph, and dynamic padding grew the
        # process past 13 GB within one epoch.
        enc = self._tok(texts, truncation=True, max_length=self.max_length)
        ids, masks = enc["input_ids"], enc["attention_mask"]
        targets = torch.tensor(y, dtype=torch.float32)

        steps_per_epoch = math.ceil(len(texts) / self.batch_size)
        total = steps_per_epoch * self.epochs
        opt = torch.optim.AdamW(self._net.parameters(), lr=self.lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(self.warmup * total), total)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        present = val_y.sum(axis=0) > 0
        best_score, best_state, history = -1.0, None, []
        for epoch in range(self.epochs):
            self._net.train()
            order = np.random.permutation(len(texts))
            t0, running = time.perf_counter(), 0.0
            for step in range(steps_per_epoch):
                idx = order[step * self.batch_size : (step + 1) * self.batch_size]
                batch = self._tok.pad(
                    {"input_ids": [ids[i] for i in idx], "attention_mask": [masks[i] for i in idx]},
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(device)
                logits = self._net(**batch).logits
                loss = loss_fn(logits, targets[idx].to(device))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                opt.step()
                sched.step()
                running += loss.item()
                if step % 200 == 0:
                    print(
                        f"epoch {epoch + 1} step {step}/{steps_per_epoch} loss {loss.item():.4f}",
                        flush=True,
                    )
            if device == "mps":
                torch.mps.empty_cache()
            val_logits = self._logits(val_texts)
            score = float(
                average_precision_score(
                    val_y[:, present], (1 / (1 + np.exp(-val_logits)))[:, present], average="macro"
                )
            )
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": round(running / steps_per_epoch, 4),
                    "val_macro_pr_auc": round(score, 4),
                    "seconds": round(time.perf_counter() - t0, 1),
                }
            )
            print(history[-1], flush=True)
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy({k: v.cpu() for k, v in self._net.state_dict().items()})

        assert best_state is not None
        self._net.load_state_dict(best_state)
        self.temperature = _fit_temperature(self._logits(val_texts), val_y)
        self.params |= {"history": history, "temperature": round(self.temperature, 3)}
        return self


def _fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Single scalar T minimising validation BCE of sigmoid(logits / T)."""
    best_t, best_loss = 1.0, math.inf
    for t in np.arange(0.5, 3.01, 0.05):
        p = np.clip(1 / (1 + np.exp(-logits / t)), 1e-7, 1 - 1e-7)
        loss = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        if loss < best_loss:
            best_t, best_loss = float(t), loss
    return best_t
