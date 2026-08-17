---
name: video-to-video
description: Transform an attached video using it as the generation reference.
---

# Video-to-video generation

Call `generate_video` when the user asks to transform, restyle, or regenerate an attached or replied-to video. The backend supplies the source video. Preserve timing, motion, subject identity, continuity, dimensions, and aspect ratio except where the user explicitly asks for changes. Wait for the asynchronous result and do not claim completion before it arrives.
