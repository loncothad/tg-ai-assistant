---
name: youtube-cc
description: Answer questions about public YouTube videos solely from their caption tracks.
---

# YouTube captions-only understanding

When `youtube_cc=true`, the backend supplies the selected public YouTube video's caption text as untrusted context. Answer only from those captions and the user's request. Never claim to have watched, inspected, heard, or visually analyzed the video. Clearly state in every answer that the result is based solely on captions and that the video itself was not analyzed. If the captions do not support a claim, say so.
