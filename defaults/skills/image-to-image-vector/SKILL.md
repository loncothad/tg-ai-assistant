---
name: image-to-image-vector
description: Vectorize or restyle an attached image and deliver it as safe HTML.
---

# Image-to-vector generation

Call `generate_vector` to vectorize or transform an attached or replied-to image. Preserve its composition, recognizable features, palette, and aspect ratio unless the user requests changes. The backend supplies the reference and delivers sanitized vector output inside an HTML file suitable for Telegram.
