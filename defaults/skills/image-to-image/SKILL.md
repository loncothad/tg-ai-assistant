---
name: image-to-image
description: Edit or transform an attached image while preserving its geometry.
---

# Image-to-image generation

Call `generate_image` when the user asks to edit, restyle, extend, or otherwise transform an attached or replied-to image. The backend supplies the reference automatically. State the requested changes precisely while preserving all unmentioned content. Unless the user requests a different geometry, retain the source dimensions when supported and otherwise retain its aspect ratio. Wait for the actual tool result.
