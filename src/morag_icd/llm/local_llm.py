from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

from ..utils.model_readiness import is_resolved_local_path

class LocalLLM:
    def __init__(self, model_path: str, device="cuda", load_in_4bit=True, allow_mock_llm: bool = False, **kwargs):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.allow_mock_llm = allow_mock_llm
        self.is_mock = False
        self.max_new_tokens = int(kwargs.get("max_new_tokens", 512))
        self.temperature = float(kwargs.get("temperature", 0.1))
        self.do_sample = bool(kwargs.get("do_sample", self.temperature > 0))
        self.num_beams = int(kwargs.get("num_beams", 1))
        self.max_input_tokens = kwargs.get("max_input_tokens")
        self.max_input_tokens = int(self.max_input_tokens) if self.max_input_tokens else None
        self.prompt_max_chars = kwargs.get("prompt_max_chars")
        self.prompt_max_chars = int(self.prompt_max_chars) if self.prompt_max_chars else None
        self.torch_inference_mode = bool(kwargs.get("torch_inference_mode", True))
        
        try:
            if not model_path:
                raise FileNotFoundError("No local LLM path configured")
            if not is_resolved_local_path(model_path):
                raise FileNotFoundError(f"Local LLM path does not exist: {Path(model_path)}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            if load_in_4bit and self.device == "cuda":
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            else:
                quantization_config = None
                
            model_kwargs = {}
            if self.device == "cuda" and bool(kwargs.get("use_fp16", False)) and quantization_config is None:
                model_kwargs["torch_dtype"] = torch.float16
            if quantization_config is not None:
                # bitsandbytes 4-bit needs accelerate's device_map anyway
                model_kwargs["device_map"] = "auto"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                **model_kwargs,
            )
            if self.device == "cuda" and quantization_config is None:
                # plain .to(cuda): no accelerate dependency (device_map="auto" requires it,
                # which is absent in the offline MN5 venv); a 3B fp16 model fits easily.
                self.model = self.model.to("cuda")
            self.model.eval()
            pipe_device = 0 if self.device == "cuda" and quantization_config is None else None
            self.pipe = pipeline(
                "text-generation", model=self.model, tokenizer=self.tokenizer,
                **({"device": pipe_device} if pipe_device is not None else {}),
            )
        except Exception as e:
            if not allow_mock_llm:
                raise RuntimeError(
                    f"Local LLM could not be loaded from {model_path!r}. "
                    f"Enable --allow-mock-llm for smoke runs. Error: {type(e).__name__}: {e}"
                ) from e
            self.is_mock = True
            self.pipe = None
            self.tokenizer = None

    def _mock_generate(self, prompt: str) -> str:
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
        if "one object per candidate" in prompt:
            return self._mock_batch_score(prompt, prompt_hash)
        if "Target Code:" in prompt:
            return self._mock_contrastive(prompt, prompt_hash)
        return self._mock_code_score(prompt, prompt_hash)

    def _mock_batch_score(self, prompt: str, prompt_hash: str) -> str:
        """Deterministic mock for the batched per-note scorer: score each parsed candidate."""
        items = re.findall(
            r"^\s*\d+\.\s*(\S+)\s*\((.*?)\)\s*\n\s*Evidence:\s*(.*)$",
            prompt, flags=re.MULTILINE,
        )
        out = []
        for code, title, evidence in items:
            overlap = self._overlap_score(evidence, title)
            supported = overlap >= 0.15
            item = {
                "code": code,
                "supported": bool(supported),
                "confidence": round(min(0.95, 0.35 + overlap * 0.8), 3),
                "mock_llm": True,
            }
            if supported:  # compact schema: quote/justification only when supported
                item["q"] = evidence.strip()[:80]
                item["r"] = "mock lexical overlap"
            out.append(item)
        return json.dumps(out)

    def _extract_field(self, prompt: str, label: str) -> str:
        pattern = rf"{re.escape(label)}:\s*(.*)"
        m = re.search(pattern, prompt)
        if not m:
            return ""
        return m.group(1).strip().splitlines()[0][:200]

    def _overlap_score(self, haystack: str, needle: str) -> float:
        hay = {tok for tok in re.findall(r"[A-Za-z0-9]+", haystack.lower()) if len(tok) > 2}
        nee = {tok for tok in re.findall(r"[A-Za-z0-9]+", needle.lower()) if len(tok) > 2}
        if not hay or not nee:
            return 0.0
        return len(hay & nee) / max(len(nee), 1)

    def _mock_code_score(self, prompt: str, prompt_hash: str) -> str:
        code = self._extract_field(prompt, "Candidate ICD-10 Code") or "MOCK"
        title = self._extract_field(prompt, "Code Title")
        desc = self._extract_field(prompt, "Code Description")
        evidence = self._extract_field(prompt, "Clinical Evidence")
        overlap = self._overlap_score(evidence, f"{title} {desc}")
        supported = overlap >= 0.15
        confidence = round(min(0.95, 0.35 + overlap * 0.8), 3)
        risk_flag = "none" if supported else ("weak_evidence" if overlap > 0 else "ambiguous")
        payload = {
            "code": code,
            "supported": supported,
            "confidence": confidence,
            "evidence_quote": evidence[:120],
            "rationale": "Mock rationale based on lexical overlap between evidence and ICD description.",
            "missing_evidence": "",
            "risk_flag": risk_flag,
            "mock_llm": True,
            "output_hash": prompt_hash,
        }
        return json.dumps(payload)

    def _mock_contrastive(self, prompt: str, prompt_hash: str) -> str:
        target = self._extract_field(prompt, "Target Code")
        sibling_lines = re.findall(r"-\s*([A-Z0-9.]+):\s*(.*)", prompt)
        preferred = target or (sibling_lines[0][0] if sibling_lines else "MOCK")
        rejected = []
        for code, title in sibling_lines[1:4]:
            rejected.append({"code": code, "reason": f"Less aligned with mock evidence than {preferred}."})
        payload = {
            "preferred_code": preferred,
            "rejected_codes": rejected,
            "contrastive_rationale": "Mock contrastive rationale based on family-level lexical comparison.",
            "confidence": 0.72,
            "mock_llm": True,
            "output_hash": prompt_hash,
        }
        return json.dumps(payload)

    def generate(self, prompt: str, max_new_tokens=None, temperature=None, prompt_max_chars=None) -> str:
        if self.is_mock:
            return self._mock_generate(prompt)

        # Per-call prompt-cap override: None -> instance default; 0 -> no truncation.
        # The batched per-note prompt is pre-capped by batch_prompt_max_chars and MUST NOT
        # be re-truncated to the (small) single-candidate cap — that cuts off the JSON
        # instructions at the end of the prompt and yields non-JSON output.
        pm = self.prompt_max_chars if prompt_max_chars is None else (int(prompt_max_chars) or None)
        if pm:
            prompt = prompt[:pm]
        max_new_tokens = int(max_new_tokens if max_new_tokens is not None else self.max_new_tokens)
        temperature = float(temperature if temperature is not None else self.temperature)
        messages = [{"role": "user", "content": prompt}]

        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": bool(self.do_sample and temperature > 0),
            "num_beams": self.num_beams,
            "pad_token_id": self.tokenizer.eos_token_id,
            # Return ONLY the generation. Slicing the full text by len(input_text) breaks
            # whenever the pipeline truncates a long input (slice lands past the generation
            # and yields ""), which silently zeroed every long batched call.
            "return_full_text": False,
        }
        if self.max_input_tokens:
            generation_kwargs.update({
                "truncation": True,
                "max_length": self.max_input_tokens + max_new_tokens,
            })

        context = torch.inference_mode() if self.torch_inference_mode else torch.no_grad()
        with context:
            outputs = self.pipe(input_text, **generation_kwargs)
        return str(outputs[0]["generated_text"]).strip()
