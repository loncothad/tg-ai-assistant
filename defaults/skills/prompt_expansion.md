# Explicit prompt expansion

Expand a prompt only when `prompt_expansion_authorized=true` appears in the
current request context. That flag means both that the user explicitly asked
for expansion and that the intent processor selected this skill. Preserve the
requested subject, constraints, language, and safety intent; add useful visual,
audio, cinematic, stylistic, composition, lighting, or technical detail without
silently changing the requested outcome.
