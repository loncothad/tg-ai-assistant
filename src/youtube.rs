//! Captions-only YouTube context retrieval.
//!
//! This module deliberately never downloads or submits YouTube video media to
//! an understanding model. It extracts a public caption-track URL from the
//! watch page, downloads that track, and returns bounded, untrusted text for
//! the normal chat model.

use compact_str::CompactString;
use eyre::{Context, ContextCompat, bail};
use serde::Deserialize;
use smallvec::SmallVec;
use url::Url;

use crate::{Result, http::HttpClient};

const MAX_WATCH_PAGE_BYTES: usize = 6 * 1024 * 1024;
const MAX_CAPTION_BYTES: usize = 4 * 1024 * 1024;
const MAX_TRANSCRIPT_CHARS: usize = 80_000;

/// Caption text and track metadata supplied to the answering model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct YoutubeCaptions {
    pub source_url: CompactString,
    pub language: CompactString,
    pub automatically_generated: bool,
    pub transcript: String,
}

#[derive(Debug, Deserialize)]
struct CaptionTrack {
    #[serde(rename = "baseUrl")]
    base_url: String,
    #[serde(rename = "languageCode", default)]
    language_code: CompactString,
    #[serde(default)]
    kind: Option<CompactString>,
}

#[derive(Debug, Deserialize)]
struct CaptionDocument {
    #[serde(default)]
    events: Vec<CaptionEvent>,
}

#[derive(Debug, Deserialize)]
struct CaptionEvent {
    #[serde(default)]
    segs: Vec<CaptionSegment>,
}

#[derive(Debug, Deserialize)]
struct CaptionSegment {
    #[serde(default)]
    utf8: String,
}

/// Finds the first canonical, HTTPS YouTube URL in arbitrary message text.
pub fn find_url(text: &str) -> Option<Url> {
    text.split_whitespace().find_map(|word| {
        let candidate = word.trim_matches(|character: char| {
            matches!(
                character,
                '(' | ')' | '[' | ']' | '<' | '>' | ',' | '.' | ';' | '!' | '?' | '"' | '\''
            )
        });
        let parsed = Url::parse(candidate).ok()?;
        (parsed.scheme() == "https" && is_youtube_host(parsed.host_str()?)).then_some(parsed)
    })
}

/// Downloads a public caption track without downloading or analyzing video.
pub async fn fetch(client: &HttpClient, source: &Url) -> Result<YoutubeCaptions> {
    let video_id = video_id(source).context("The YouTube URL does not contain a valid video ID")?;
    let watch_url = format!("https://www.youtube.com/watch?v={video_id}&hl=en");
    let response = client
        .get(&watch_url)
        .header(reqwest::header::ACCEPT_LANGUAGE, "en-US,en;q=0.8,*;q=0.2")
        .timeout(std::time::Duration::from_secs(20))
        .send()
        .await
        .context("Failed to load the YouTube caption metadata")?
        .error_for_status()
        .context("YouTube rejected the caption metadata request")?;
    ensure_bounded_length(&response, MAX_WATCH_PAGE_BYTES, "YouTube watch page")?;
    let bytes = response
        .bytes()
        .await
        .context("Failed to read the YouTube watch page")?;
    if bytes.len() > MAX_WATCH_PAGE_BYTES {
        bail!("YouTube watch page exceeded the safe size limit");
    }
    let page = std::str::from_utf8(&bytes).context("YouTube returned a non-UTF-8 watch page")?;
    let tracks_json = json_array_after(page, "\"captionTracks\":")
        .context("This YouTube video has no accessible captions")?;
    let tracks: SmallVec<[CaptionTrack; 8]> =
        serde_json::from_str::<Vec<CaptionTrack>>(tracks_json)
            .context("YouTube returned invalid caption metadata")?
            .into_iter()
            .collect();
    let track = tracks
        .iter()
        .find(|track| track.kind.as_deref() != Some("asr"))
        .or_else(|| tracks.first())
        .context("This YouTube video has no accessible caption track")?;

    let mut caption_url =
        Url::parse(&track.base_url).context("YouTube returned an invalid caption-track URL")?;
    if caption_url.scheme() != "https"
        || !is_caption_host(caption_url.host_str().unwrap_or_default())
    {
        bail!("YouTube returned an untrusted caption-track URL");
    }
    caption_url.query_pairs_mut().append_pair("fmt", "json3");
    let response = client
        .get(caption_url)
        .timeout(std::time::Duration::from_secs(20))
        .send()
        .await
        .context("Failed to download the YouTube captions")?
        .error_for_status()
        .context("YouTube rejected the caption download")?;
    if !is_caption_host(response.url().host_str().unwrap_or_default()) {
        bail!("YouTube redirected the caption request to an untrusted host");
    }
    ensure_bounded_length(&response, MAX_CAPTION_BYTES, "YouTube caption track")?;
    let bytes = response
        .bytes()
        .await
        .context("Failed to read the YouTube captions")?;
    if bytes.len() > MAX_CAPTION_BYTES {
        bail!("YouTube caption track exceeded the safe size limit");
    }
    let document: CaptionDocument =
        serde_json::from_slice(&bytes).context("YouTube returned invalid caption data")?;
    let transcript = normalize_transcript(document);
    if transcript.is_empty() {
        bail!("The selected YouTube caption track is empty");
    }
    Ok(YoutubeCaptions {
        source_url: CompactString::new(source.as_str()),
        language: track.language_code.clone(),
        automatically_generated: track.kind.as_deref() == Some("asr"),
        transcript,
    })
}

fn ensure_bounded_length(response: &reqwest::Response, limit: usize, label: &str) -> Result<()> {
    if response
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        bail!("{label} exceeded the safe size limit");
    }
    Ok(())
}

fn video_id(url: &Url) -> Option<CompactString> {
    let host = url.host_str()?.trim_start_matches("www.");
    let candidate = if host == "youtu.be" {
        url.path_segments()?.next()
    } else if let Some(value) = url
        .query_pairs()
        .find_map(|(key, value)| (key == "v").then_some(value))
    {
        return valid_video_id(&value).then(|| CompactString::new(value.as_ref()));
    } else {
        let mut segments = url.path_segments()?;
        match segments.next()? {
            "shorts" | "live" | "embed" => segments.next(),
            _ => None,
        }
    }?;
    valid_video_id(candidate).then(|| CompactString::new(candidate))
}

fn valid_video_id(value: &str) -> bool {
    value.len() == 11
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn is_youtube_host(host: &str) -> bool {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    matches!(
        host.as_str(),
        "youtube.com"
            | "www.youtube.com"
            | "m.youtube.com"
            | "music.youtube.com"
            | "youtu.be"
            | "youtube-nocookie.com"
            | "www.youtube-nocookie.com"
    )
}

fn is_caption_host(host: &str) -> bool {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    host == "youtube.com"
        || host.ends_with(".youtube.com")
        || host == "googlevideo.com"
        || host.ends_with(".googlevideo.com")
}

fn json_array_after<'a>(document: &'a str, marker: &str) -> Option<&'a str> {
    let tail = document.get(document.find(marker)? + marker.len()..)?;
    let start = tail.find('[')?;
    let bytes = tail.as_bytes();
    let mut depth = 0_u32;
    let mut quoted = false;
    let mut escaped = false;
    for (offset, byte) in bytes.iter().enumerate().skip(start) {
        if quoted {
            if escaped {
                escaped = false;
            } else if *byte == b'\\' {
                escaped = true;
            } else if *byte == b'"' {
                quoted = false;
            }
            continue;
        }
        match *byte {
            b'"' => quoted = true,
            b'[' => depth += 1,
            b']' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return tail.get(start..=offset);
                }
            }
            _ => {}
        }
    }
    None
}

fn normalize_transcript(document: CaptionDocument) -> String {
    let mut raw = String::new();
    for event in document.events {
        for segment in event.segs {
            raw.push_str(&segment.utf8);
        }
        raw.push(' ');
    }
    let mut result = String::with_capacity(raw.len().min(MAX_TRANSCRIPT_CHARS));
    for word in raw.split_whitespace() {
        let needed = word.chars().count() + usize::from(!result.is_empty());
        if result.chars().count().saturating_add(needed) > MAX_TRANSCRIPT_CHARS {
            result.push_str(" … [captions truncated]");
            break;
        }
        if !result.is_empty() {
            result.push(' ');
        }
        result.push_str(word);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_supported_urls_and_rejects_lookalikes() {
        for url in [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://youtube.com/shorts/dQw4w9WgXcQ",
        ] {
            let parsed = find_url(&format!("summarize {url}")).unwrap();
            assert_eq!(video_id(&parsed).as_deref(), Some("dQw4w9WgXcQ"));
        }
        assert!(find_url("https://youtube.com.evil.example/watch?v=dQw4w9WgXcQ").is_none());
        assert!(find_url("http://youtube.com/watch?v=dQw4w9WgXcQ").is_none());
    }

    #[test]
    fn extracts_nested_json_array_without_stopping_inside_strings() {
        let page = r#"prefix "captionTracks":[{"baseUrl":"https://x/[ok]","nested":[1]}],"next":1"#;
        assert_eq!(
            json_array_after(page, "\"captionTracks\":"),
            Some(r#"[{"baseUrl":"https://x/[ok]","nested":[1]}]"#)
        );
    }
}
