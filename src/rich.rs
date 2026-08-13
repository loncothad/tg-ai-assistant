//! Conversion and size-safe delivery for Telegram Rich Markdown.
//!
//! Telegram Rich Markdown deliberately resembles GitHub Flavored Markdown, but
//! model output also commonly contains CommonMark Setext headings, tilde code
//! fences, indented code blocks, and LaTeX `\(...\)` / `\[...\]` delimiters.
//! This module normalizes those forms before any text reaches `InputRichMessage`.

// Telegram permits 32,768 UTF-8 characters. Leave room for fence repair and
// future server-side accounting changes.
const SAFE_RICH_CHARS: usize = 31_900;
const SAFE_BLOCKS: usize = 450;

/// Converts ordinary model-produced Markdown into Telegram Rich Markdown.
///
/// The conversion is intentionally conservative: GFM constructs already
/// supported by Telegram (tables, task lists, footnotes, images, and quotes)
/// pass through unchanged. Code content is never rewritten.
pub fn to_telegram_markdown(input: &str) -> String {
    let normalized = input.replace("\r\n", "\n").replace('\r', "\n");
    let lines = normalized.lines().collect::<Vec<_>>();
    let mut output = String::with_capacity(normalized.len());
    let mut in_fence = false;
    let mut index = 0;
    while index < lines.len() {
        let line = lines[index];
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            let indent = &line[..line.len() - trimmed.len()];
            if trimmed.starts_with("~~~") {
                output.push_str(indent);
                output.push_str("```");
                output.push_str(trimmed.trim_start_matches('~'));
            } else {
                output.push_str(line);
            }
            output.push('\n');
            in_fence = !in_fence;
            index += 1;
            continue;
        }
        if in_fence {
            output.push_str(line);
            output.push('\n');
            index += 1;
            continue;
        }

        // CommonMark Setext headings are not part of Telegram's documented
        // grammar; convert them into equivalent ATX headings.
        if index + 1 < lines.len() && !line.trim().is_empty() {
            let underline = lines[index + 1].trim();
            let heading = if underline.len() >= 3 && underline.chars().all(|c| c == '=') {
                Some("# ")
            } else if underline.len() >= 3 && underline.chars().all(|c| c == '-') {
                Some("## ")
            } else {
                None
            };
            if let Some(prefix) = heading {
                output.push_str(prefix);
                output.push_str(&normalize_inline(line));
                output.push('\n');
                index += 2;
                continue;
            }
        }

        // Convert a run of four-space CommonMark code lines to the explicitly
        // supported fenced representation.
        if line.starts_with("    ")
            && is_indented_code_start(line)
            && (index == 0 || lines[index - 1].trim().is_empty())
        {
            output.push_str("```\n");
            while index < lines.len()
                && (lines[index].starts_with("    ") || lines[index].is_empty())
            {
                output.push_str(lines[index].strip_prefix("    ").unwrap_or(lines[index]));
                output.push('\n');
                index += 1;
            }
            output.push_str("```\n");
            continue;
        }

        output.push_str(&normalize_inline(line));
        output.push('\n');
        index += 1;
    }
    if !normalized.ends_with('\n') {
        output.pop();
    }
    output
}

fn normalize_inline(line: &str) -> String {
    // These are the delimiters most frequently emitted by models trained to
    // target MathJax. Telegram Rich Markdown documents `$` and `$$` instead.
    let mut output = String::with_capacity(line.len());
    let mut characters = line.chars().peekable();
    let mut code_ticks = None;
    while let Some(character) = characters.next() {
        if character == '`' {
            let mut count = 1usize;
            while characters.next_if_eq(&'`').is_some() {
                count += 1;
            }
            output.extend(std::iter::repeat_n('`', count));
            code_ticks = match code_ticks {
                None => Some(count),
                Some(open) if open == count => None,
                current => current,
            };
        } else if character == '\\' && code_ticks.is_none() {
            match characters.peek().copied() {
                Some('(' | ')') => {
                    characters.next();
                    output.push('$');
                }
                Some('[' | ']') => {
                    characters.next();
                    output.push_str("$$");
                }
                _ => output.push(character),
            }
        } else {
            output.push(character);
        }
    }
    output
}

fn is_indented_code_start(line: &str) -> bool {
    let content = line.trim_start();
    let ordered_list = content.split_once('.').is_some_and(|(number, rest)| {
        number.chars().all(|c| c.is_ascii_digit()) && rest.starts_with(' ')
    });
    !content.starts_with("- ")
        && !content.starts_with("* ")
        && !content.starts_with("+ ")
        && !content.starts_with("> ")
        && !ordered_list
}

/// Converts and splits Markdown on line boundaries while preserving UTF-8 and
/// keeping each fenced-code chunk independently parseable.
pub fn chunks(input: &str) -> Vec<String> {
    let converted = to_telegram_markdown(input);
    if converted.chars().count() <= SAFE_RICH_CHARS && converted.lines().count() <= SAFE_BLOCKS {
        return vec![converted];
    }
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut fence: Option<String> = None;
    let mut blocks = 0usize;
    for line in converted.split_inclusive('\n') {
        let trimmed = line.trim_start();
        let fence_line = trimmed.starts_with("```");
        let repair = if fence.is_some() { 4 } else { 0 };
        if !current.is_empty()
            && (current.chars().count() + line.chars().count() + repair > SAFE_RICH_CHARS
                || blocks >= SAFE_BLOCKS)
        {
            if fence.is_some() {
                current.push_str("```\n");
            }
            chunks.push(std::mem::take(&mut current));
            if let Some(opening) = &fence {
                current.push_str(opening);
                current.push('\n');
            }
            blocks = 0;
        }
        for character in line.chars() {
            if current.chars().count() + repair >= SAFE_RICH_CHARS {
                if fence.is_some() {
                    current.push_str("\n```");
                }
                chunks.push(std::mem::take(&mut current));
                if let Some(opening) = &fence {
                    current.push_str(opening);
                    current.push('\n');
                }
            }
            current.push(character);
        }
        blocks += 1;
        if fence_line {
            if fence.is_some() {
                fence = None;
            } else {
                fence = Some(trimmed.trim_end().to_owned());
            }
        }
    }
    if !current.is_empty() {
        if fence.is_some() {
            current.push_str("\n```");
        }
        chunks.push(current);
    }
    chunks
}

pub fn compact_error() -> &'static str {
    "I couldn’t complete that request. Please try again in a moment. If it keeps failing, ask an administrator to check the service logs."
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_common_model_markdown_without_touching_code() {
        let value = "Title\n=====\n\n~~~rust\nlet x = \\(1\\);\n~~~\n\nMath: \\(x + 1\\)";
        let converted = to_telegram_markdown(value);
        assert!(converted.starts_with("# Title\n"));
        assert!(converted.contains("```rust\nlet x = \\(1\\);\n```"));
        assert!(converted.ends_with("Math: $x + 1$"));
        assert_eq!(
            normalize_inline("`\\(literal\\)` and \\(math\\)"),
            "`\\(literal\\)` and $math$"
        );
        assert_eq!(
            to_telegram_markdown("- parent\n    - child"),
            "- parent\n    - child"
        );
    }

    #[test]
    fn split_is_unicode_safe() {
        let value = "🦀".repeat(32_001);
        let pieces = chunks(&value);
        assert_eq!(pieces.concat(), value);
        assert!(pieces.iter().all(|p| p.chars().count() <= SAFE_RICH_CHARS));
    }

    #[test]
    fn long_fenced_code_is_repaired_per_chunk() {
        let value = format!("```text\n{}\n```", "a".repeat(64_000));
        let pieces = chunks(&value);
        assert!(pieces.len() > 1);
        assert!(pieces.iter().all(|piece| piece.starts_with("```text")));
        assert!(pieces.iter().all(|piece| piece.trim_end().ends_with("```")));
    }
}
