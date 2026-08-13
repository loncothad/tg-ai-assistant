# Image generation skill

Call `generate_image` when the user asks to create or edit an image. Produce a detailed prompt containing subject, composition, medium, lighting, palette, aspect ratio, and exclusions. When the request includes an attached or replied-to image, the backend automatically supplies it to the generation API as an input reference. The backend delivers generated files to Telegram; do not fabricate a URL or claim success before the tool result.
