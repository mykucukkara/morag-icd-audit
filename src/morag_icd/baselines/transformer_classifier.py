"""
Transformer-based multi-label classifier baseline (E3).

Supports BioClinicalBERT, ClinicalBERT, or any HuggingFace sequence
classification model.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from packaging.version import Version


class TransformerClassifier:
    """
    Multi-label text classifier using a HuggingFace transformer model.

    Wraps HuggingFace AutoModelForSequenceClassification with multi-label
    training/inference logic.
    """

    def __init__(
        self,
        model_path: str = "",
        device: str = "cuda",
        max_length: int = 256,
        batch_size: int = 2,
        num_labels: int = 50,
        threshold: float = 0.5,
        top_k: int = 10,
        epochs: int = 1,
        learning_rate: float = 2e-5,
        fp16: bool = False,
        allow_cpu_fallback: bool = False,
        use_safetensors: bool = True,
        allow_mock: bool = False,
    ):
        self.model_path = model_path
        self.requested_device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.threshold = threshold
        self.top_k = top_k
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.fp16 = fp16
        self.allow_cpu_fallback = allow_cpu_fallback
        self.use_safetensors = use_safetensors
        self.label_names: List[str] = []
        self.allow_mock = allow_mock
        self.is_mock = False
        self.model = None
        self.tokenizer = None
        self.classifier_load_mode = ""
        self.classifier_model_available = False
        self.classifier_weight_format = "unknown"
        self.cuda_oom_count = 0
        self.json_parse_error_count = 0
        self.device = device
        self._loaded_num_labels: int | None = None

        self._validate_model_path()

    @staticmethod
    def detect_load_mode(model_path: str) -> str:
        path = Path(model_path)
        if path.is_dir() and (path / "config.json").exists():
            return "hf_pretrained_sequence_classification"
        if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".bin"}:
            return "state_dict_checkpoint"
        return "invalid"

    @staticmethod
    def detect_weight_format(model_path: str) -> str:
        path = Path(model_path)
        if path.is_dir():
            has_safetensors = any(path.glob("*.safetensors")) or (path / "model.safetensors.index.json").exists()
            if has_safetensors:
                return "safetensors"
            if (path / "pytorch_model.bin").exists():
                return "bin"
            return "unknown"
        if path.suffix.lower() == ".safetensors":
            return "safetensors"
        if path.suffix.lower() in {".pt", ".pth", ".bin"}:
            return "bin"
        return "unknown"

    @staticmethod
    def _torch_version_lt_26() -> bool:
        import torch

        raw = str(torch.__version__).split("+")[0]
        return Version(raw) < Version("2.6")

    def _validate_model_path(self):
        if not self.model_path:
            raise ValueError("TransformerClassifier requires a valid model_path")
        mode = self.detect_load_mode(self.model_path)
        if mode == "invalid":
            raise ValueError(f"missing_or_invalid_classifier_model: {self.model_path}")
        self.classifier_load_mode = mode
        self.classifier_model_available = True
        self.classifier_weight_format = self.detect_weight_format(self.model_path)

    def _resolve_device(self, requested: str):
        import torch

        req = str(requested or "cuda").lower()
        if req.startswith("cuda"):
            if torch.cuda.is_available():
                return "cuda"
            if self.allow_cpu_fallback:
                return "cpu"
            raise RuntimeError("cuda_unavailable_for_transformer_classifier")
        if req == "cpu":
            return "cpu"
        return req

    def _load_hf_pretrained(self, num_labels: int):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir = Path(self.model_path)
        has_safetensors = any(model_dir.glob("*.safetensors")) or (model_dir / "model.safetensors.index.json").exists()
        if self.use_safetensors and (not has_safetensors) and self._torch_version_lt_26():
            raise RuntimeError(
                "classifier_requires_safetensors_or_torch26: HF directory has no safetensors weights and torch<2.6"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,
            local_files_only=True,
            use_safetensors=bool(self.use_safetensors),
        )

    def _load_state_dict_checkpoint(self, num_labels: int):
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        ckpt_path = Path(self.model_path)
        base_dir = ckpt_path.parent
        if not (base_dir / "config.json").exists():
            raise RuntimeError(
                f"missing_or_invalid_classifier_model: checkpoint parent lacks config.json ({base_dir})"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True)
        config = AutoConfig.from_pretrained(base_dir, local_files_only=True)
        config.num_labels = num_labels
        config.problem_type = "multi_label_classification"
        self.model = AutoModelForSequenceClassification.from_config(config)
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise RuntimeError(f"invalid_state_dict_checkpoint: {ckpt_path}")
        self.model.load_state_dict(state, strict=False)

    def _try_load(self, num_labels: int):
        """Attempt to load model according to detected classifier load mode."""
        try:
            import torch

            if self.classifier_load_mode == "hf_pretrained_sequence_classification":
                self._load_hf_pretrained(num_labels)
            elif self.classifier_load_mode == "state_dict_checkpoint":
                self._load_state_dict_checkpoint(num_labels)
            else:
                raise RuntimeError(f"missing_or_invalid_classifier_model: {self.model_path}")

            self.device = self._resolve_device(self.requested_device)
            self.model.to(self.device)
            self.model.eval()
            self._loaded_num_labels = num_labels
        except Exception as e:
            if self.allow_mock:
                self.is_mock = True
                print(f"Warning: TransformerClassifier could not load model: {e}. Using mock.")
                return
            raise RuntimeError(f"TransformerClassifier could not load model: {e}") from e

    @property
    def run_metadata(self):
        return {
            "classifier_model_path": self.model_path,
            "classifier_model_available": bool(self.classifier_model_available),
            "classifier_load_mode": self.classifier_load_mode,
            "classifier_weight_format": self.classifier_weight_format,
            "classifier_use_safetensors": bool(self.use_safetensors),
            "num_labels": int(self.num_labels),
            "label_names_count": len(self.label_names),
            "max_length": int(self.max_length),
            "batch_size": int(self.batch_size),
            "epochs": int(self.epochs),
            "device": self.device,
            "cuda_oom_count": int(self.cuda_oom_count),
            "json_parse_error_count": int(self.json_parse_error_count),
        }

    def fit(self, X_train: List[str], Y_train, label_names: Optional[List[str]] = None):
        """
        Fine-tune the model on training data.

        Note: Full fine-tuning is out of scope for the benchmark.
        This method provides a placeholder that logs a warning.
        For actual fine-tuning, use Trainer API separately.
        """
        if label_names:
            self.label_names = label_names
            self.num_labels = len(label_names)
        if self.num_labels <= 0:
            raise ValueError("TransformerClassifier requires num_labels > 0")

        if self._loaded_num_labels != self.num_labels:
            self._try_load(num_labels=self.num_labels)

        if self.is_mock:
            return

        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, TensorDataset

        if not X_train or not Y_train:
            return

        labels = torch.tensor(Y_train, dtype=torch.float32)
        encoded = self.tokenizer(
            X_train,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        ds = TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        optim = AdamW(self.model.parameters(), lr=self.learning_rate)
        use_fp16 = bool(self.fp16 and self.device == "cuda" and torch.cuda.is_available())
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

        for _ in range(max(self.epochs, 1)):
            for input_ids, attention_mask, y in dl:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                y = y.to(self.device)
                optim.zero_grad(set_to_none=True)
                try:
                    with torch.cuda.amp.autocast(enabled=use_fp16):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=y,
                        )
                        loss = outputs.loss
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        self.cuda_oom_count += 1
                    raise
        self.model.eval()

    def predict(self, X: List[str]) -> List[List[str]]:
        """
        Predict ICD-10 codes for a list of clinical notes.

        Returns
        -------
        list of list of str (code names/indices for each sample).
        """
        if self.is_mock:
            return [[] for _ in X]

        import torch

        all_preds = []
        for i in range(0, len(X), self.batch_size):
            batch_texts = X[i: i + self.batch_size]
            encoding = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = self.model(**encoding)
                logits = outputs.logits
                probs = torch.sigmoid(logits).cpu().numpy()

            for prob_row in probs:
                predicted_indices = np.where(prob_row >= self.threshold)[0]
                if len(predicted_indices) > self.top_k:
                    ranked = predicted_indices[np.argsort(prob_row[predicted_indices])[::-1][: self.top_k]]
                    predicted_indices = ranked
                elif len(predicted_indices) == 0:
                    # Top-k fallback: a threshold-only rule emits NOTHING whenever the head
                    # is under-trained/calibrated below the threshold (measured: E3 produced
                    # 0 predictions for all 1000 pilot notes). Rank-based output keeps the
                    # baseline comparable to the other systems.
                    predicted_indices = np.argsort(prob_row)[::-1][: self.top_k]
                if self.label_names:
                    pred_codes = [self.label_names[idx] for idx in predicted_indices
                                  if idx < len(self.label_names)]
                else:
                    pred_codes = [str(idx) for idx in predicted_indices]
                all_preds.append(pred_codes)

        return all_preds

    def process_note(self, text: str) -> List[dict]:
        """
        Pipeline-compatible interface: returns list of code dicts.
        """
        result = self.predict([text])
        codes = result[0] if result else []
        return [
            {
                "code": c,
                "confidence": 1.0,
                "supported": None,
                "evidence_score": 0.0,
                "icd_description": "",
                "rationale": "",
                "evidence_preview": "",
                "risk_flag": "transformer_baseline",
            }
            for c in codes
        ]

    def save(self, output_dir: str | Path):
        """Save model and tokenizer."""
        if self.is_mock or self.model is None:
            return
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def set_labels(self, label_names: List[str]):
        """Set the label universe for prediction."""
        self.label_names = label_names
        self.num_labels = len(label_names)
