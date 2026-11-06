from pathlib import Path


class GttsVocalSynthesizer:
    def synthesize(self, line: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # gTTS returns MP3 bytes. The caller passes an .mp3 path.
        _save_speech(line, output_path)


def _save_speech(text: str, output_path: Path) -> None:
    from gtts import gTTS

    gTTS(text=text, lang="en").save(str(output_path))
