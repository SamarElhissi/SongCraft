"""Run inside .venv-acestep. Reads a request json path and writes a result json path."""

import json
import shutil
import sys
import traceback
from pathlib import Path


def main() -> int:
    request_path = Path(sys.argv[1])
    result_path = Path(sys.argv[2])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    try:
        payload = _generate(request)
    except Exception as exc:
        traceback.print_exc()
        payload = {"ok": False, "error": str(exc)}
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    return 0 if payload.get("ok") else 1


def _generate(request: dict) -> dict:
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, create_sample, generate_music
    from acestep.llm_inference import LLMHandler
    from acestep.model_downloader import ensure_lm_model, ensure_main_model

    project_root = request["project_root"]
    checkpoint_dir = request["checkpoint_dir"]
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    ready, message = ensure_main_model(checkpoint_dir)
    if not ready:
        raise RuntimeError(message)
    ready, message = ensure_lm_model(request["lm_model"], checkpoint_dir)
    if not ready:
        raise RuntimeError(message)

    dit = AceStepHandler()
    status, ok = dit.initialize_service(
        project_root=project_root,
        config_path=request["model"],
        device=request["device"],
        use_flash_attention=False,
        compile_model=True,
        offload_to_cpu=True,
        offload_dit_to_cpu=True,
        quantization="int8_weight_only",
        use_mlx_dit=False,
    )
    if not ok:
        raise RuntimeError(status)

    llm = LLMHandler()
    status, ok = llm.initialize(
        checkpoint_dir=checkpoint_dir,
        lm_model_path=request["lm_model"],
        backend="pt",
        device=request["device"],
        offload_to_cpu=True,
    )
    if not ok:
        raise RuntimeError(status)

    sample = create_sample(
        llm,
        query=request["prompt"],
        instrumental=False,
        vocal_language="en",
    )
    if not sample.success:
        raise RuntimeError(sample.error or sample.status_message or "ACE-Step could not plan the song")

    output_path = Path(request["output_path"])
    save_dir = output_path.parent / f".{output_path.stem}-ace"
    save_dir.mkdir(parents=True, exist_ok=True)
    params = GenerationParams(
        task_type="text2music",
        caption=sample.caption or request["prompt"],
        lyrics=sample.lyrics,
        instrumental=False,
        vocal_language=sample.language or "en",
        bpm=sample.bpm,
        keyscale=sample.keyscale or "",
        timesignature=sample.timesignature or "",
        duration=float(request["duration_seconds"]),
        inference_steps=8,
        guidance_scale=1.0,
        shift=3.0,
        infer_method="ode",
        thinking=True,
    )
    config = GenerationConfig(batch_size=1, audio_format="mp3")
    result = generate_music(dit, llm, params, config, save_dir=str(save_dir))
    if not result.success:
        detail = getattr(result, "error", None) or getattr(result, "status_message", None)
        raise RuntimeError(str(detail or "ACE-Step generation failed"))
    audio = result.audios[0]
    produced = Path(str(audio["path"]))
    if produced.resolve() != output_path.resolve():
        shutil.copyfile(produced, output_path)
    shutil.rmtree(save_dir, ignore_errors=True)
    bpm = getattr(result, "bpm", None)
    if bpm is None:
        bpm = audio.get("bpm")
    return {
        "ok": True,
        "lyrics": sample.lyrics,
        "bpm": sample.bpm if sample.bpm is not None else bpm,
    }


if __name__ == "__main__":
    sys.exit(main())
