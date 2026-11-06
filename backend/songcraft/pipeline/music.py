import logging
from pathlib import Path

from songcraft.device import resolve_device

logger = logging.getLogger("songcraft.pipeline")


class MusicGenGenerator:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def generate(self, prompt: str, duration_seconds: float, output_path: Path) -> None:
        model = self._get_model()
        model.set_generation_params(
            use_sampling=True,
            top_k=250,
            duration=float(duration_seconds),
        )
        # A progress bar would write over the step logs.
        audio = model.generate(descriptions=[prompt], progress=False, return_tokens=False)
        if isinstance(audio, tuple):
            audio = audio[0]
        _write_wav(audio, output_path, int(model.sample_rate))

    def _get_model(self):
        if self._model is None:
            self._model = _load_musicgen(self.model_name, self.device)
        return self._model


def _load_musicgen(model_name: str, device: str):
    from audiocraft.models import MusicGen

    resolved = resolve_device(device)
    logger.info("model_loaded step=music model=%s device=%s", model_name, resolved)
    return MusicGen.get_pretrained(model_name, device=resolved)


def _write_wav(samples, output_path: Path, sample_rate: int) -> None:
    import torchaudio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    waveform = samples.detach().cpu()
    if waveform.dim() == 3:
        waveform = waveform[0]
    if waveform.dim() != 2:
        raise RuntimeError(f"Unexpected audio shape {tuple(waveform.shape)}")
    torchaudio.save(str(output_path), waveform, sample_rate)
