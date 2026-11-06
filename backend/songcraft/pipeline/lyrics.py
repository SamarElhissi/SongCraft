import logging

from songcraft.device import resolve_device
from songcraft.pipeline.text import parse_lyric_lines

logger = logging.getLogger("songcraft.pipeline")

# New tokens, not a fixed max_length, so a 500-character prompt still fits.
_MAX_NEW_TOKENS = 120


def lyric_messages(prompt: str, line_count: int, syllables: int) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You write singable English lyrics. "
                "Reply with the lyric lines only, one line each. "
                "No title, no section labels, no commentary."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Write {line_count} short rhyming lines about: {prompt}. "
                f"Each line should be about {syllables} syllables."
            ),
        },
    ]


class InstructLyricsGenerator:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    def generate(self, prompt: str, line_count: int = 2, syllables: int = 6) -> str:
        import torch

        model, tokenizer = self._get()
        messages = lyric_messages(prompt, line_count, syllables)
        encoded = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
        input_ids = input_ids.to(model.device)
        attention_mask = torch.ones_like(input_ids)
        try:
            output = self._generate(model, tokenizer, input_ids, attention_mask, line_count, sample=True)
        except RuntimeError:
            # Half-precision sampling can produce a NaN probability and abort the step.
            output = self._generate(model, tokenizer, input_ids, attention_mask, line_count, sample=False)
        text = tokenizer.decode(output[0, input_ids.shape[-1] :], skip_special_tokens=True)
        lines = parse_lyric_lines(text)[: max(line_count, 1)]
        if not lines:
            raise RuntimeError("lyrics model returned no lines")
        return "\n".join(lines)

    def _generate(self, model, tokenizer, input_ids, attention_mask, line_count: int, sample: bool):
        import torch

        kwargs = {
            "attention_mask": attention_mask,
            "max_new_tokens": max(32, line_count * 24),
            "pad_token_id": tokenizer.eos_token_id,
            "do_sample": sample,
        }
        if sample:
            kwargs["temperature"] = 0.8
            kwargs["top_p"] = 0.9
        with torch.no_grad():
            return model.generate(input_ids, **kwargs)

    def _get(self):
        if self._model is None or self._tokenizer is None:
            self._model, self._tokenizer = _load_instruct(self.model_name, self.device)
        return self._model, self._tokenizer


class Gpt2LyricsGenerator:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self._pipeline = None

    def generate(self, prompt: str, line_count: int = 2, syllables: int = 6) -> str:
        pipe = self._get_pipeline()
        result = pipe(
            prompt,
            max_new_tokens=_MAX_NEW_TOKENS,
            do_sample=True,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )
        return str(result[0]["generated_text"]).strip()

    def _get_pipeline(self):
        if self._pipeline is None:
            self._pipeline = _load_text_pipeline(self.model_name, self.device)
        return self._pipeline


def _load_instruct(model_name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = resolve_device(device)
    logger.info("model_loaded step=lyrics model=%s device=%s", model_name, resolved)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if resolved == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(resolved)
    model.eval()
    return model, tokenizer


def _load_text_pipeline(model_name: str, device: str):
    from transformers import pipeline

    resolved = resolve_device(device)
    logger.info("model_loaded step=lyrics model=%s device=%s", model_name, resolved)
    return pipeline(
        "text-generation",
        model=model_name,
        device=_transformers_device(resolved),
    )


def _transformers_device(device: str) -> int | str:
    # The text-generation pipeline uses -1 for CPU and 0 for the first CUDA device.
    if device == "cpu":
        return -1
    if device == "cuda":
        return 0
    return device
