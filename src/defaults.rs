//! Build-time defaults embedded from ordinary repository files.
//!
//! Each built-in capability is a Claude Code-style skill package at
//! `defaults/skills/<name>/SKILL.md`. YAML frontmatter supplies discoverable
//! metadata while only the Markdown body is placed in model instructions.

pub const SYSTEM_PROMPT: &str = include_str!("../defaults/system.md");

/// One build-time embedded skill package.
#[derive(Clone, Copy, Debug)]
pub struct BuiltinSkill {
    /// Stable backend capability ID retained for saved settings compatibility.
    pub id: &'static str,
    /// Valid kebab-case package name expected in the YAML frontmatter.
    pub package_name: &'static str,
    source: &'static str,
}

impl BuiltinSkill {
    /// Returns the frontmatter description shown in the admin panel/export.
    pub fn description(self) -> &'static str {
        frontmatter_value(self.source, "description").unwrap_or("Built-in skill")
    }

    /// Returns the instruction body without YAML metadata.
    pub fn instructions(self) -> &'static str {
        split_skill(self.source).map_or(self.source, |(_, body)| body)
    }

    /// Returns the package name declared in the embedded frontmatter.
    pub fn metadata_name(self) -> Option<&'static str> {
        frontmatter_value(self.source, "name")
    }
}

fn split_skill(source: &'static str) -> Option<(&'static str, &'static str)> {
    let rest = source.strip_prefix("---\n")?;
    let boundary = rest.find("\n---\n")?;
    Some((&rest[..boundary], rest[boundary + 5..].trim_start()))
}

fn frontmatter_value(source: &'static str, key: &str) -> Option<&'static str> {
    let (frontmatter, _) = split_skill(source)?;
    frontmatter.lines().find_map(|line| {
        let (candidate, value) = line.split_once(':')?;
        (candidate.trim() == key).then_some(value.trim())
    })
}

macro_rules! builtin_skills {
    ($(($id:literal, $name:literal)),+ $(,)?) => {
        pub const BUILTIN_SKILLS: &[BuiltinSkill] = &[
            $(BuiltinSkill {
                id: $id,
                package_name: $name,
                source: include_str!(concat!("../defaults/skills/", $name, "/SKILL.md")),
            }),+
        ];
    };
}

builtin_skills![
    ("search", "web-search"),
    ("web_fetch", "web-fetch"),
    ("text_to_image", "text-to-image"),
    ("image_to_image", "image-to-image"),
    ("text_to_video", "text-to-video"),
    ("image_to_video", "image-to-video"),
    ("video_to_video", "video-to-video"),
    ("text_to_audio", "text-to-audio"),
    ("video_to_audio", "video-to-audio"),
    ("text_to_speech", "text-to-speech"),
    ("text_to_3d", "text-to-3d"),
    ("image_to_3d", "image-to-3d"),
    ("text_to_image_vector", "text-to-image-vector"),
    ("image_to_image_vector", "image-to-image-vector"),
    ("image_understanding", "image-understanding"),
    ("video_understanding", "video-understanding"),
    ("transcription", "media-transcription"),
    ("file", "code-file-delivery"),
    ("model_upgrade", "advanced-model-routing"),
    ("youtube_cc", "youtube-cc"),
    ("prompt_expansion", "prompt-expansion"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_embedded_skill_has_valid_frontmatter_and_a_body() {
        for skill in BUILTIN_SKILLS {
            assert_eq!(skill.metadata_name(), Some(skill.package_name));
            assert!(skill.package_name.bytes().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-'
            }));
            assert!(!skill.description().is_empty());
            assert!(skill.instructions().starts_with('#'));
            assert!(!skill.instructions().starts_with("---"));
        }
    }
}
