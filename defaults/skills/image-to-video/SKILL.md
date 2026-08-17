---
name: image-to-video
description: Animate an attached image while preserving its aspect ratio.
---

# Image-to-video generation

Call `generate_video` when the user asks to animate an attached or replied-to image. The backend supplies the source image. Describe motion, camera behavior, timing, continuity, and any requested audio without replacing the source subject or style. Preserve source dimensions when supported and otherwise preserve its aspect ratio unless the user explicitly requests another geometry.
