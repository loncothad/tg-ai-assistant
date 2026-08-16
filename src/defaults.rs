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
        id: "text_to_image",
        description: "Generate raster images from text with the selected provider and model.",
        instructions: include_str!("../defaults/skills/text_to_image.md"),
    },
    BuiltinSkill {
        id: "image_to_image",
        description: "Edit or transform an attached image while preserving its geometry.",
        instructions: include_str!("../defaults/skills/image_to_image.md"),
    },
    BuiltinSkill {
        id: "text_to_video",
        description: "Generate video from text with the selected provider and model.",
        instructions: include_str!("../defaults/skills/text_to_video.md"),
    },
    BuiltinSkill {
        id: "image_to_video",
        description: "Animate an attached image while preserving its aspect ratio.",
        instructions: include_str!("../defaults/skills/image_to_video.md"),
    },
    BuiltinSkill {
        id: "video_to_video",
        description: "Transform an attached video using it as the generation reference.",
        instructions: include_str!("../defaults/skills/video_to_video.md"),
    },
    BuiltinSkill {
        id: "text_to_audio",
        description: "Generate music, songs, sound effects, and other non-speech audio from text.",
        instructions: include_str!("../defaults/skills/text_to_audio.md"),
    },
    BuiltinSkill {
        id: "video_to_audio",
        description: "Generate a soundtrack or audio treatment for an attached video.",
        instructions: include_str!("../defaults/skills/video_to_audio.md"),
    },
    BuiltinSkill {
        id: "text_to_speech",
        description: "Generate narration and spoken audio from text.",
        instructions: include_str!("../defaults/skills/text_to_speech.md"),
    },
    BuiltinSkill {
        id: "text_to_3d",
        description: "Generate a downloadable 3D artifact from text through fal.ai.",
        instructions: include_str!("../defaults/skills/text_to_3d.md"),
    },
    BuiltinSkill {
        id: "image_to_3d",
        description: "Reconstruct a downloadable 3D artifact from an attached image through fal.ai.",
        instructions: include_str!("../defaults/skills/image_to_3d.md"),
    },
    BuiltinSkill {
        id: "text_to_image_vector",
        description: "Generate vector artwork from text and deliver it as safe HTML.",
        instructions: include_str!("../defaults/skills/text_to_image_vector.md"),
    },
    BuiltinSkill {
        id: "image_to_image_vector",
        description: "Vectorize or restyle an attached image and deliver it as safe HTML.",
        instructions: include_str!("../defaults/skills/image_to_image_vector.md"),
    },
    BuiltinSkill {
        id: "image_understanding",
        description: "Describe, inspect, and read text from attached images.",
        instructions: include_str!("../defaults/skills/image_understanding.md"),
    },
    BuiltinSkill {
        id: "video_understanding",
        description: "Describe, inspect, and answer questions about attached videos.",
        instructions: include_str!("../defaults/skills/video_understanding.md"),
    },
    BuiltinSkill {
        id: "transcription",
        description: "Transcribe speech from Telegram voice notes, audio files, and videos.",
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
    BuiltinSkill {
        id: "youtube",
        description: "Describe and analyze public YouTube videos with the selected video-understanding model.",
        instructions: include_str!("../defaults/skills/youtube.md"),
    },
    BuiltinSkill {
        id: "prompt_expansion",
        description: "Expand a prompt only after the user explicitly asks and the intent processor authorizes it.",
        instructions: include_str!("../defaults/skills/prompt_expansion.md"),
    },
];
