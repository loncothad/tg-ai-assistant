//! Telegram Rich Message-compatible Markdown chunking and safe user errors.

// Telegram allows 32,768 characters; leave headroom for parser/accounting changes.
const SAFE_RICH_CHARS: usize = 32_000;

/// Splits Markdown on paragraph/line boundaries while preserving UTF-8.
pub fn chunks(input: &str) -> Vec<String> {
    if input.chars().count() <= SAFE_RICH_CHARS {
        return vec![input.to_owned()];
    }
    let mut chunks = Vec::new();
    let mut current = String::new();
    for segment in input.split_inclusive('\n') {
        if current.chars().count() + segment.chars().count() > SAFE_RICH_CHARS
            && !current.is_empty()
        {
            chunks.push(std::mem::take(&mut current));
        }
        if segment.chars().count() > SAFE_RICH_CHARS {
            for character in segment.chars() {
                current.push(character);
                if current.chars().count() == SAFE_RICH_CHARS {
                    chunks.push(std::mem::take(&mut current));
                }
            }
        } else {
            current.push_str(segment);
        }
    }
    if !current.is_empty() {
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
    fn split_is_unicode_safe() {
        let value = "🦀".repeat(8001);
        let pieces = chunks(&value);
        assert_eq!(pieces.concat(), value);
        assert!(pieces.iter().all(|p| p.chars().count() <= SAFE_RICH_CHARS));
    }
}
