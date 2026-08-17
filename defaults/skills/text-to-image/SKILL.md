---
name: text-to-image
description: Generate raster images from text with the selected provider and model.
---

# Text-to-image generation

Call `generate_image` for a request to create a new raster image from text when no image reference is attached. Preserve the user's requested subject, composition, medium, lighting, palette, resolution or aspect ratio, and exclusions. When a named subject needs identifying visual knowledge that is not fully supplied, use the authorized visual-research path first and pass its grounded traits—not raw search results—to the generator. Do not invent a fixed size: use an explicitly requested geometry or let the selected model choose a suitable default. Wait for the tool result and never fabricate a URL or claim generation succeeded early.
