"""Label-wise attention classifier — the like-for-like supervised control (T3-2).

Reviewers across three rounds made the same point: the paper compares a prompt-based pipeline
against TF-IDF and a plain fine-tuned encoder, but never against the architecture the ICD-coding
literature actually reports state-of-the-art numbers with. Without that, "the harness is competent"
rests on the classical arm alone, and the gap to the published ~0.70 stays unexplained.

This supplies it. The mechanism every strong ICD coder shares — CAML, LAAT, PLM-ICD — is
*label-wise attention*: instead of pooling the document into one vector and classifying it, each
label attends over the token sequence and forms its own document representation. That is what lets
a model find the one sentence that justifies a rare code in a long note, which mean- or CLS-pooling
cannot do.

    for each label l:  alpha_l = softmax(H u_l)        attention over tokens for that label
                       v_l     = alpha_l^T H          label-specific document vector
                       logit_l = v_l . beta_l + b_l   per-label scorer

Deliberately built on the same general-domain encoder as E3, with no clinically pretrained weights,
because the point is to isolate the contribution of the architecture rather than to reproduce a
published number we cannot reach offline. It is therefore a lower bound on what the label-attention
family achieves with a domain-pretrained encoder, and the paper reports it as such.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class LabelAttentionConfig:
    model_path: str
    num_labels: int = 50
    max_length: int = 512
    batch_size: int = 8
    epochs: int = 5
    lr: float = 3e-5
    head_lr: float = 1e-3          # the attention head is new; it needs a faster rate than the encoder
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    device: str = "cuda"
    label_set: list[str] = field(default_factory=list)


class LabelAttentionClassifier:
    """BERT-style encoder + per-label attention pooling + per-label scorer."""

    def __init__(self, cfg: LabelAttentionConfig):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None

    # ------------------------------------------------------------------
    def _build(self):
        import torch
        import torch.nn as nn
        from transformers import AutoModel, AutoTokenizer

        cfg = self.cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, local_files_only=True)
        encoder = AutoModel.from_pretrained(cfg.model_path, local_files_only=True)
        hidden = encoder.config.hidden_size

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = encoder
                # One attention query and one scoring vector per label.
                self.U = nn.Linear(hidden, cfg.num_labels, bias=False)
                self.beta = nn.Linear(hidden, cfg.num_labels)
                nn.init.xavier_uniform_(self.U.weight)
                nn.init.xavier_uniform_(self.beta.weight)

            def forward(self, input_ids, attention_mask):
                H = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                scores = self.U(H)                                    # (B, T, L)
                mask = attention_mask.unsqueeze(-1).to(scores.dtype)  # padding must not receive weight
                scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
                alpha = torch.softmax(scores, dim=1)                  # over tokens, per label
                V = torch.einsum("btl,bth->blh", alpha, H)            # (B, L, hidden)
                logits = (V * self.beta.weight).sum(-1) + self.beta.bias
                return logits

        self.model = Net().to(cfg.device)
        return self.model

    # ------------------------------------------------------------------
    def _encode(self, texts: list[str]):
        return self.tokenizer(texts, truncation=True, padding="max_length",
                              max_length=self.cfg.max_length, return_tensors="pt")

    def _targets(self, gold: list[list[str]]):
        import torch
        idx = {c: i for i, c in enumerate(self.cfg.label_set)}
        y = torch.zeros(len(gold), self.cfg.num_labels)
        for r, codes in enumerate(gold):
            for c in codes:
                j = idx.get(str(c).replace(".", "").strip().upper())
                if j is not None:
                    y[r, j] = 1.0
        return y

    def fit(self, texts: list[str], gold: list[list[str]],
            val_texts: list[str] | None = None, val_gold: list[list[str]] | None = None):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        self._build()

        enc = self._encode(texts)
        ds = TensorDataset(enc["input_ids"], enc["attention_mask"], self._targets(gold))
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

        # The encoder is pretrained and the head is not, so they do not want the same learning rate.
        head_params = [p for n, p in self.model.named_parameters() if not n.startswith("encoder.")]
        enc_params = [p for n, p in self.model.named_parameters() if n.startswith("encoder.")]
        opt = torch.optim.AdamW(
            [{"params": enc_params, "lr": cfg.lr},
             {"params": head_params, "lr": cfg.head_lr}], weight_decay=cfg.weight_decay)
        total = max(1, len(dl) * cfg.epochs)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[cfg.lr, cfg.head_lr], total_steps=total, pct_start=cfg.warmup_ratio)
        lossf = torch.nn.BCEWithLogitsLoss()

        self.model.train()
        for ep in range(cfg.epochs):
            running = 0.0
            for ids, mask, y in dl:
                ids, mask, y = ids.to(cfg.device), mask.to(cfg.device), y.to(cfg.device)
                opt.zero_grad()
                loss = lossf(self.model(ids, mask), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                running += loss.item()
            log.info("epoch %d/%d  loss %.4f", ep + 1, cfg.epochs, running / max(1, len(dl)))
        return self

    # ------------------------------------------------------------------
    def predict_scores(self, texts: list[str], batch_size: int | None = None):
        import torch
        bs = batch_size or self.cfg.batch_size
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                enc = self._encode(texts[i:i + bs])
                ids = enc["input_ids"].to(self.cfg.device)
                mask = enc["attention_mask"].to(self.cfg.device)
                out.append(torch.sigmoid(self.model(ids, mask)).cpu())
        return torch.cat(out) if out else torch.empty(0, self.cfg.num_labels)

    def predict_topk(self, texts: list[str], k: int = 15) -> list[list[str]]:
        """Top-k codes per note — the ladder's fixed-budget protocol."""
        scores = self.predict_scores(texts)
        labels = self.cfg.label_set
        return [[labels[j] for j in row.topk(min(k, len(labels))).indices.tolist()] for row in scores]
