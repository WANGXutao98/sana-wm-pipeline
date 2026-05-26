"""Qwen3.5-VL (fallback: Qwen2.5-VL) caption runner with rejection-retry.

Model loading is lazy and isolated behind `_load_model`.  Tests inject a
`generate_fn(prompt, images) -> str` to avoid GPU dependence; orchestration
code calls `caption_clip(frames_rgb)` which uses the real Qwen pipeline.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from .postprocess import has_camera_verb, strip_offending_sentences
from .prompts import SCENE_STATIC_PROMPT


# Lazy global model cache: (model, processor)
_MODEL: Optional[tuple] = None


def _try_load_qwen(model_id: str):
    """Load Qwen-VL model + processor; return (model, processor) or None."""
    try:
        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoProcessor,
            AutoModelForVision2Seq,
        )
    except ImportError:
        return None
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        return (model, processor)
    except Exception:
        return None


def _load_model():
    """Load Qwen3.5-VL if available, else Qwen2.5-VL (paper-sanctioned fallback)."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    for mid in ("Qwen/Qwen3.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"):
        loaded = _try_load_qwen(mid)
        if loaded is not None:
            _MODEL = loaded
            return loaded
    raise RuntimeError(
        "No Qwen-VL model could be loaded. "
        "Install transformers, ensure GPU is available, "
        "or pass an explicit `generate_fn` to caption_clip()."
    )


def _sample_keyframes(frames_rgb: np.ndarray, n: int = 8) -> List[np.ndarray]:
    if len(frames_rgb) == 0:
        return []
    idx = np.linspace(0, len(frames_rgb) - 1, n).astype(int)
    return [frames_rgb[i] for i in idx]


def _default_generate(prompt: str, images: List[np.ndarray]) -> str:
    """Real Qwen-VL generation path; only used when generate_fn is not injected."""
    from PIL import Image  # type: ignore
    import torch  # type: ignore

    model, processor = _load_model()
    pil_imgs = [Image.fromarray(im) for im in images]
    content = [{"type": "image", "image": im} for im in pil_imgs] + \
              [{"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=pil_imgs, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=180, do_sample=False)
    decoded = processor.batch_decode(
        out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    )[0]
    return decoded.strip()


def caption_clip(
    frames_rgb: np.ndarray,
    *,
    max_retries: int = 2,
    generate_fn: Optional[Callable[[str, List[np.ndarray]], str]] = None,
) -> str:
    """Generate a scene-static caption with forbidden-verb retry.

    Returns the first generation that passes `has_camera_verb()` check;
    if none of `max_retries + 1` attempts pass, returns the last attempt
    with offending sentences stripped (may be empty).
    """
    gen = generate_fn if generate_fn is not None else _default_generate
    keyframes = _sample_keyframes(frames_rgb)
    last = ""
    for _ in range(max_retries + 1):
        last = gen(SCENE_STATIC_PROMPT, keyframes) or ""
        if not has_camera_verb(last):
            return last
    return strip_offending_sentences(last)
