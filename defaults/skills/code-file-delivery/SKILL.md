---
name: code-file-delivery
description: Deliver long answers, source code, and structured text as Telegram files.
---

# Code and long-answer file delivery

Call `send_file` when the complete answer would be unwieldy as Telegram messages, when the user asks for a downloadable file, or when the primary result is a code, configuration, or data file. Supply a safe filename with an appropriate extension and the complete UTF-8 content. Then give a short summary in chat instead of repeating the entire file. Do not use the tool for short ordinary answers.
