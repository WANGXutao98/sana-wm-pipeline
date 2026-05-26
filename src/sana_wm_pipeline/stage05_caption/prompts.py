"""Scene-static caption prompt (SANA-WM paper §4, verbatim policy).

Paper quote (§4):
  "When action conditions are available, we use scene-static captions that
   describe objects, layout, and appearance while omitting camera actions
   such as 'pan left' or 'move forward.'  This prevents text from leaking
   trajectory supervision and forces motion control through the pose branch."

We pass this prompt to Qwen3.5-VL (or Qwen2.5-VL fallback).  The post-process
layer (stage05_caption.postprocess) enforces the no-camera-verb rule with
regex, retrying generation if a forbidden phrase leaks through.
"""

SCENE_STATIC_PROMPT = """You are a careful video annotator. The user will provide 8 evenly-sampled keyframes from a 60-second video.

Write ONE concise paragraph (60-120 words) describing the STATIC scene:
- Objects, their materials and colours
- Spatial layout and relative positions
- Lighting, time of day, weather (if visible)
- Visual style (photographic, game-render, animated, etc.)

DO NOT mention any camera motion or rendering verbs, including but not limited to:
  pan, tilt, zoom, dolly, truck, crab, crane, fly-through, walk, walking,
  rotate, spin, orbit, move (forward/back/left/right/up/down), translate,
  slide, approach, retreat, viewpoint changes, "the camera ...",
  "first-person view going ...".

Use present tense, third-person, neutral. Begin with the dominant noun phrase."""
