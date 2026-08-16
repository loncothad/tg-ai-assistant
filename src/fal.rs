//! Generic fal.ai queue client for schema-configured model endpoints.
//!
//! fal.ai intentionally gives each endpoint its own JSON schema. Teleforge
//! therefore maps common prompt/reference fields in configuration and preserves
//! arbitrary endpoint defaults, while sharing secure queue submission, status
//! polling, result extraction, and media downloading.

use std::{
    net::IpAddr,
    time::{Duration, Instant},
};

use eyre::{Context, ContextCompat, bail};
use reqwest::header::RETRY_AFTER;
use serde_json::{Map, Value};
use tokio::time::sleep;
use url::Url;

use crate::{
    Result,
    config::{FalConfig, FalEndpointConfig},
    http::HttpClient,
};

#[derive(Clone)]
pub struct FalClient {
    client: HttpClient,
    config: FalConfig,
}

impl FalClient {
    pub fn new(client: HttpClient, config: FalConfig) -> Self {
        Self { client, config }
    }

    pub fn endpoint(&self, id: &str, capability: &str) -> Result<&FalEndpointConfig> {
        self.config
            .endpoints
            .iter()
            .find(|endpoint| {
                endpoint.id == id && endpoint.capabilities.iter().any(|item| item == capability)
            })
            .with_context(|| format!("Fal endpoint {id} is not configured for {capability}"))
    }

    pub async fn run(
        &self,
        endpoint: &FalEndpointConfig,
        input: Map<String, Value>,
        api_key: &str,
    ) -> Result<Value> {
        let target = endpoint_url(
            endpoint
                .base_url
                .as_deref()
                .unwrap_or(&self.config.base_url),
            &endpoint.id,
        )?;
        let response = self
            .client
            .post(target)
            .header(reqwest::header::AUTHORIZATION, format!("Key {api_key}"))
            .header(
                "X-Fal-Request-Timeout",
                self.config.timeout_seconds.to_string(),
            )
            .json(&input)
            .send()
            .await
            .context("Fal queue submission failed")?;
        let submitted = checked_json(response, "Fal queue submission").await?;
        let Some(status_url) = submitted.get("status_url").and_then(Value::as_str) else {
            // Direct fal.run endpoints return their result immediately.
            return Ok(submitted);
        };
        let status_url = trusted_fal_url(status_url)?;
        let response_url = submitted
            .get("response_url")
            .and_then(Value::as_str)
            .map(trusted_fal_url)
            .transpose()?;
        let started = Instant::now();
        loop {
            if started.elapsed() > Duration::from_secs(self.config.timeout_seconds) {
                bail!("Fal endpoint {} timed out", endpoint.id);
            }
            let status_response = self
                .client
                .get(status_url.clone())
                .header(reqwest::header::AUTHORIZATION, format!("Key {api_key}"))
                .query(&[("logs", "1")])
                .send()
                .await
                .context("Fal queue status request failed")?;
            let retry_after = status_response
                .headers()
                .get(RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<u64>().ok());
            let state = checked_json(status_response, "Fal queue status").await?;
            match state.get("status").and_then(Value::as_str).unwrap_or("") {
                "COMPLETED" => {
                    let url = state
                        .get("response_url")
                        .and_then(Value::as_str)
                        .map(trusted_fal_url)
                        .transpose()?
                        .or(response_url.clone())
                        .context("Fal completed without a response URL")?;
                    let response = self
                        .client
                        .get(url)
                        .header(reqwest::header::AUTHORIZATION, format!("Key {api_key}"))
                        .send()
                        .await
                        .context("Fal result request failed")?;
                    return checked_json(response, "Fal result").await;
                }
                "FAILED" | "CANCELLED" => {
                    let message = state
                        .pointer("/error/message")
                        .or_else(|| state.get("error"))
                        .and_then(Value::as_str)
                        .unwrap_or("Endpoint execution failed");
                    bail!("Fal endpoint {} failed: {message}", endpoint.id);
                }
                _ => {}
            }
            sleep(Duration::from_secs(
                retry_after
                    .unwrap_or(self.config.poll_interval_seconds)
                    .max(1),
            ))
            .await;
        }
    }

    pub fn media_urls(&self, endpoint: &FalEndpointConfig, result: &Value) -> Vec<String> {
        let mut urls = endpoint
            .output_url_paths
            .iter()
            .filter_map(|path| result.pointer(path).and_then(Value::as_str))
            .filter(|url| is_http_url(url))
            .map(str::to_owned)
            .collect::<Vec<_>>();
        if urls.is_empty() {
            collect_urls(result, &mut urls);
        }
        urls.sort();
        urls.dedup();
        urls
    }

    pub fn text(&self, endpoint: &FalEndpointConfig, result: &Value) -> Option<String> {
        endpoint
            .output_text_paths
            .iter()
            .filter_map(|path| result.pointer(path).and_then(render_text_value))
            .find(|text| !text.trim().is_empty())
            .or_else(|| {
                ["/text", "/transcript", "/output/text", "/data/text"]
                    .into_iter()
                    .filter_map(|path| result.pointer(path).and_then(render_text_value))
                    .find(|text| !text.trim().is_empty())
            })
    }

    pub async fn download(&self, url: &str) -> Result<(Vec<u8>, String)> {
        let parsed = Url::parse(url).context("Fal returned an invalid media URL")?;
        validate_media_url(&parsed)?;
        let response = self
            .client
            .get(parsed)
            .timeout(Duration::from_secs(120))
            .send()
            .await
            .context("Failed to download Fal output")?;
        let status = response.status();
        let media_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("application/octet-stream")
            .split(';')
            .next()
            .unwrap_or("application/octet-stream")
            .to_owned();
        if !status.is_success() {
            bail!("Fal output download returned {status}");
        }
        Ok((
            response
                .bytes()
                .await
                .context("Failed to read Fal output")?
                .to_vec(),
            media_type,
        ))
    }
}

fn render_text_value(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(text) => Some(text.clone()),
        Value::Bool(_) | Value::Number(_) => Some(value.to_string()),
        Value::Array(_) | Value::Object(_) => serde_json::to_string_pretty(value).ok(),
    }
}

fn endpoint_url(base: &str, id: &str) -> Result<Url> {
    if id.trim().is_empty() || id.split('/').any(|part| part.is_empty() || part == "..") {
        bail!("Invalid Fal endpoint ID");
    }
    let base = format!("{}/", base.trim_end_matches('/'));
    let mut url = Url::parse(&base).context("Invalid fal.base_url")?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| eyre::eyre!("Invalid fal.base_url"))?;
        for segment in id.split('/') {
            segments.push(segment);
        }
    }
    Ok(url)
}

fn trusted_fal_url(value: &str) -> Result<Url> {
    let url = Url::parse(value).context("Fal returned an invalid queue URL")?;
    let host = url.host_str().unwrap_or_default();
    if url.scheme() != "https"
        || !(host == "fal.run"
            || host.ends_with(".fal.run")
            || host == "fal.ai"
            || host.ends_with(".fal.ai"))
    {
        bail!("Fal returned an untrusted queue URL");
    }
    Ok(url)
}

async fn checked_json(response: reqwest::Response, operation: &str) -> Result<Value> {
    let status = response.status();
    let bytes = response
        .bytes()
        .await
        .context("Failed to read Fal response")?;
    if !status.is_success() {
        let excerpt = String::from_utf8_lossy(&bytes[..bytes.len().min(1_000)]);
        bail!("{operation} returned {status}: {excerpt}");
    }
    serde_json::from_slice(&bytes).context("Fal returned invalid JSON")
}

fn is_http_url(value: &str) -> bool {
    Url::parse(value)
        .ok()
        .is_some_and(|url| validate_media_url(&url).is_ok())
}

fn validate_media_url(url: &Url) -> Result<()> {
    if url.scheme() != "https" || !url.username().is_empty() || url.password().is_some() {
        bail!("Fal returned an unsafe media URL");
    }
    let host = url.host_str().context("Fal media URL has no host")?;
    if host.eq_ignore_ascii_case("localhost") || host.ends_with(".localhost") {
        bail!("Fal returned a local media URL");
    }
    if let Ok(ip) = host.parse::<IpAddr>()
        && match ip {
            IpAddr::V4(ip) => {
                ip.is_private() || ip.is_loopback() || ip.is_link_local() || ip.is_unspecified()
            }
            IpAddr::V6(ip) => {
                ip.is_loopback()
                    || ip.is_unspecified()
                    || ip.is_unique_local()
                    || ip.is_unicast_link_local()
            }
        }
    {
        bail!("Fal returned a private media URL");
    }
    Ok(())
}

fn collect_urls(value: &Value, output: &mut Vec<String>) {
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                if matches!(key.as_str(), "url" | "file_url" | "audio_url" | "video_url")
                    && let Some(url) = value.as_str()
                    && is_http_url(url)
                {
                    output.push(url.to_owned());
                }
                collect_urls(value, output);
            }
        }
        Value::Array(values) => values.iter().for_each(|value| collect_urls(value, output)),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_ids_are_encoded_as_path_segments() {
        let url = endpoint_url("https://queue.fal.run", "fal-ai/flux/dev").unwrap();
        assert_eq!(url.as_str(), "https://queue.fal.run/fal-ai/flux/dev");
        assert!(endpoint_url("https://queue.fal.run", "fal-ai/../secret").is_err());
    }

    #[test]
    fn recursively_finds_standard_media_urls() {
        let mut urls = Vec::new();
        collect_urls(
            &serde_json::json!({"images":[{"url":"https://cdn.example/a.png"}]}),
            &mut urls,
        );
        assert_eq!(urls, ["https://cdn.example/a.png"]);
    }

    #[test]
    fn vision_outputs_can_be_structured_or_scalar_json() {
        assert_eq!(
            render_text_value(&serde_json::json!(true)).as_deref(),
            Some("true")
        );
        let object = render_text_value(&serde_json::json!({"label":"sfw","score":0.99})).unwrap();
        assert!(object.contains("\"label\": \"sfw\""));
    }
}
