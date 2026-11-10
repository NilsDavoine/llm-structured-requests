"""A small open model run locally with transformers. No key, no network after download."""
from __future__ import annotations

import os

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# Pinned revisions (Hugging Face commit hashes) so the benchmark is reproducible.
REVISIONS = {
    "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
}


class LocalHFProvider:
    def __init__(self, model: str | None = None, max_new_tokens: int = 600, **_):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model or os.environ.get("STRUCTREQ_LOCAL_MODEL", DEFAULT_MODEL)
        revision = REVISIONS.get(self.model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(self.model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, revision=revision, torch_dtype=dtype).to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        self.name = f"local:{self.model_id}"
        self._torch = torch

    def complete(self, system: str, messages: list[dict]) -> str:
        chat = [{"role": "system", "content": system}] + messages
        ids = self.tok.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            out = self.model.generate(ids, attention_mask=self._torch.ones_like(ids),
                                      max_new_tokens=self.max_new_tokens, do_sample=False,
                                      temperature=None, top_p=None, top_k=None,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
