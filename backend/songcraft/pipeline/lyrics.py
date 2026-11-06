import logging

from songcraft.device import resolve_device

logger = logging.getLogger("songcraft.pipeline")

# New tokens, not a fixed max_length, so a 500-character prompt still fits.
_MAX_NEW_TOKENS = 120


class Gpt2LyricsGenerator:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self._pipeline = None

    def generate(self, prompt: str) -> str:
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
