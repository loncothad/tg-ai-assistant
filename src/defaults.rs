//! Build-time defaults embedded from ordinary repository files.

pub const SYSTEM_PROMPT: &str = include_str!("../defaults/system.md");

#[derive(Clone, Copy, Debug)]
pub struct BuiltinSkill {
    pub id: &'static str,
    pub description: &'static str,
    pub instructions: &'static str,
}

pub const BUILTIN_SKILLS: &[BuiltinSkill] = &[
    BuiltinSkill {
        id: "search",
        description: "Research current information with OpenRouter, Brave, Exa, or Google/SerpAPI.",
        instructions: include_str!("../defaults/skills/search.md"),
    },
    BuiltinSkill {
        id: "web_fetch",
        description: "Open and extract text from web pages and PDF URLs through OpenRouter.",
        instructions: include_str!("../defaults/skills/web_fetch.md"),
    },
    BuiltinSkill {
        id: "image",
        description: "Generate images through the OpenRouter Image API.",
        instructions: include_str!("../defaults/skills/image.md"),
    },
    BuiltinSkill {
        id: "audio",
        description: "Generate spoken audio through OpenRouter text-to-speech.",
        instructions: include_str!("../defaults/skills/audio.md"),
    },
    BuiltinSkill {
        id: "video",
        description: "Generate videos through the asynchronous OpenRouter Video API.",
        instructions: include_str!("../defaults/skills/video.md"),
    },
    BuiltinSkill {
        id: "media",
        description: "Understand attached images, Telegram videos, and YouTube videos.",
        instructions: include_str!("../defaults/skills/media.md"),
    },
    BuiltinSkill {
        id: "transcription",
        description: "Transcribe Telegram voice notes and audio through OpenRouter.",
        instructions: include_str!("../defaults/skills/transcription.md"),
    },
    BuiltinSkill {
        id: "file",
        description: "Deliver long answers, source code, and structured text as Telegram files.",
        instructions: include_str!("../defaults/skills/file.md"),
    },
    BuiltinSkill {
        id: "model_upgrade",
        description: "Route difficult requests to an administrator-selected, more capable model.",
        instructions: include_str!("../defaults/skills/model_upgrade.md"),
    },
];
