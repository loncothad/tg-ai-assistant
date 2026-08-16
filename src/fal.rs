//! Generic fal.ai queue client with live OpenAPI endpoint-schema discovery.
//!
//! fal.ai intentionally gives each endpoint its own JSON schema. Teleforge
//! derives common prompt/reference/default/output mappings from the Platform
//! API, caches them, and lets configuration override irregular/private models,
//! while sharing secure queue submission, polling, result extraction, and
//! media downloading.

use std::{
    collections::BTreeMap,
    net::IpAddr,
    sync::Arc,
    time::{Duration, Instant},
};

use eyre::{Context, ContextCompat, bail};
use reqwest::header::RETRY_AFTER;
use serde_json::{Map, Value};
use tokio::sync::RwLock;
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
    discovered: Arc<RwLock<BTreeMap<String, FalEndpointConfig>>>,
}

impl FalClient {
    pub fn new(client: HttpClient, config: FalConfig) -> Self {
        Self {
            client,
            config,
            discovered: Arc::new(RwLock::new(BTreeMap::new())),
        }
    }

    /// Resolves a configured override or lazily derives an executable endpoint
    /// mapping from fal.ai's live OpenAPI schema.
    pub async fn endpoint(
        &self,
        id: &str,
        capability: &str,
        api_key: &str,
    ) -> Result<FalEndpointConfig> {
        let endpoint = self.resolve_endpoint(id, api_key).await?;
        if !endpoint_supports(&endpoint, capability) {
            bail!("Fal endpoint {id} does not support {capability}");
        }
        Ok(endpoint)
    }

    /// Resolves the first supported capability from a caller-supplied fallback
    /// sequence without performing duplicate schema requests.
    pub async fn endpoint_any(
        &self,
        id: &str,
        capabilities: &[&str],
        api_key: &str,
    ) -> Result<FalEndpointConfig> {
        if capabilities.is_empty() {
            bail!("Fal endpoint capability list is empty");
        }
        let endpoint = self.resolve_endpoint(id, api_key).await?;
        if !capabilities
            .iter()
            .any(|capability| endpoint_supports(&endpoint, capability))
        {
            bail!("Fal endpoint {id} does not support the requested capability");
        }
        Ok(endpoint)
    }

    async fn resolve_endpoint(&self, id: &str, api_key: &str) -> Result<FalEndpointConfig> {
        if let Some(endpoint) = self
            .config
            .endpoints
            .iter()
            .find(|endpoint| endpoint.id == id)
        {
            return Ok(endpoint.clone());
        }
        if let Some(endpoint) = self.discovered.read().await.get(id).cloned() {
            return Ok(endpoint);
        }
        let endpoint = self.discover_endpoint(id, api_key).await?;
        self.discovered
            .write()
            .await
            .insert(id.to_owned(), endpoint.clone());
        Ok(endpoint)
    }

    async fn discover_endpoint(&self, id: &str, api_key: &str) -> Result<FalEndpointConfig> {
        let response = self
            .client
            .get(format!(
                "{}/models",
                self.config.catalog_url.trim_end_matches('/')
            ))
            .header(reqwest::header::AUTHORIZATION, format!("Key {api_key}"))
            .query(&[("endpoint_id", id), ("expand", "openapi-3.0")])
            .timeout(Duration::from_secs(20))
            .send()
            .await
            .context("Failed to fetch fal.ai endpoint schema")?;
        if !response.status().is_success() {
            bail!(
                "Fal endpoint schema lookup returned HTTP {}",
                response.status()
            );
        }
        let body: Value = response
            .json()
            .await
            .context("Fal returned an invalid endpoint schema response")?;
        let model = body
            .get("models")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .find(|model| model.get("endpoint_id").and_then(Value::as_str) == Some(id))
            .with_context(|| format!("Fal model catalog did not return endpoint {id}"))?;
        endpoint_from_openapi(model)
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
        let mut unique = Vec::with_capacity(urls.len());
        for url in urls {
            if !unique.contains(&url) {
                unique.push(url);
            }
        }
        unique
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

fn endpoint_supports(endpoint: &FalEndpointConfig, requested: &str) -> bool {
    let has = |value: &str| endpoint.capabilities.iter().any(|item| item == value);
    has(requested)
        || matches!(requested, "text_to_image" | "image_to_image") && has("image_generation")
        || matches!(
            requested,
            "text_to_video" | "image_to_video" | "video_to_video"
        ) && has("video_generation")
        || matches!(requested, "text_to_audio" | "video_to_audio") && has("music_generation")
        || requested == "text_to_speech" && (has("speech_generation") || has("audio_generation"))
}

fn endpoint_from_openapi(model: &Value) -> Result<FalEndpointConfig> {
    let id = model
        .get("endpoint_id")
        .and_then(Value::as_str)
        .context("Fal schema record has no endpoint_id")?;
    let metadata = model
        .get("metadata")
        .and_then(Value::as_object)
        .context("Fal schema record has no metadata")?;
    let category = metadata
        .get("category")
        .and_then(Value::as_str)
        .context("Fal schema record has no category")?;
    let openapi = model
        .get("openapi")
        .context("Fal schema record has no OpenAPI expansion")?;
    let input_schema = request_schema(openapi, id).context("Fal OpenAPI has no input schema")?;
    let mut properties = Map::new();
    let mut required = Vec::new();
    collect_schema(openapi, input_schema, 0, &mut properties, &mut required);
    if properties.is_empty() {
        bail!("Fal endpoint {id} has an empty input schema");
    }

    let mut capabilities = crate::catalog::fal_model_capabilities(
        category,
        id,
        metadata
            .get("display_name")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        metadata
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default(),
    );
    let prompt_field = find_field(
        &properties,
        &[
            "prompt",
            "text",
            "lyrics",
            "description",
            "script",
            "query",
            "content",
            "input",
        ],
        &["prompt", "text"],
    )
    .unwrap_or_default();
    let image_field = find_media_field(&properties, "image");
    let video_field = find_media_field(&properties, "video");
    let audio_field = find_media_field(&properties, "audio")
        .or_else(|| find_field(&properties, &["file_url"], &["audio", "file"]));
    capabilities.retain(|capability| match capability.as_str() {
        "image_to_image"
        | "image_to_video"
        | "image_to_3d"
        | "image_to_image_vector"
        | "image_understanding" => image_field.is_some(),
        "video_to_video" | "video_to_audio" | "video_understanding" => video_field.is_some(),
        "transcription" => audio_field.is_some(),
        _ => true,
    });
    if capabilities.is_empty() {
        bail!("Fal endpoint {id} has no executable assistant capability");
    }
    if prompt_field.is_empty()
        && capabilities.iter().any(|capability| {
            matches!(
                capability.as_str(),
                "text_to_image"
                    | "text_to_video"
                    | "text_to_audio"
                    | "text_to_speech"
                    | "text_to_3d"
                    | "text_to_image_vector"
            )
        })
    {
        bail!("Fal endpoint {id} has no discoverable text input field");
    }

    let language_field = find_field(
        &properties,
        &["language", "language_code", "target_language"],
        &["language"],
    );
    let width_field = find_field(&properties, &["width"], &["width"]).or_else(|| {
        properties
            .contains_key("image_size")
            .then(|| "image_size.width".to_owned())
    });
    let height_field = find_field(&properties, &["height"], &["height"]).or_else(|| {
        properties
            .contains_key("image_size")
            .then(|| "image_size.height".to_owned())
    });
    let aspect_ratio_field = find_field(
        &properties,
        &["aspect_ratio", "aspect"],
        &["aspect", "ratio"],
    );

    let mut defaults = Map::new();
    for (name, schema) in &properties {
        if let Some(value) = schema_default(openapi, schema, 0) {
            defaults.insert(name.clone(), value.clone());
        }
    }
    let mapped = [
        Some(prompt_field.as_str()).filter(|value| !value.is_empty()),
        image_field.as_deref(),
        video_field.as_deref(),
        audio_field.as_deref(),
        language_field.as_deref(),
        width_field.as_deref(),
        height_field.as_deref(),
        aspect_ratio_field.as_deref(),
    ];
    for field in required {
        if mapped.into_iter().flatten().any(|mapped| mapped == field)
            || defaults.contains_key(&field)
        {
            continue;
        }
        let schema = properties
            .get(&field)
            .context("Fal required field is absent from its own schema")?;
        if let Some(value) = schema_enum_value(openapi, schema, 0) {
            defaults.insert(field, value.clone());
        } else if field == "num_images" {
            defaults.insert(field, Value::from(1));
        } else {
            bail!(
                "Fal endpoint {id} requires unsupported input field {field}; add a fal.endpoints override with an appropriate default"
            );
        }
    }

    let mut output_url_paths = Vec::new();
    let mut output_text_paths = Vec::new();
    if let Some(output_schema) = response_schema(openapi, id) {
        collect_output_paths(
            openapi,
            output_schema,
            "",
            0,
            &mut output_url_paths,
            &mut output_text_paths,
        );
    }
    if capabilities.iter().any(|capability| {
        matches!(
            capability.as_str(),
            "image_understanding" | "video_understanding" | "transcription"
        )
    }) && output_text_paths.is_empty()
        && let Some(output_schema) = response_schema(openapi, id)
    {
        collect_top_level_output_paths(openapi, output_schema, &mut output_text_paths);
    }

    let created = ["date", "updated_at"]
        .into_iter()
        .filter_map(|key| metadata.get(key).and_then(Value::as_str))
        .find_map(|value| {
            chrono::DateTime::parse_from_rfc3339(value)
                .ok()
                .map(|date| date.timestamp())
        });
    Ok(FalEndpointConfig {
        id: id.to_owned(),
        base_url: None,
        name: metadata
            .get("display_name")
            .and_then(Value::as_str)
            .unwrap_or(id)
            .to_owned(),
        description: metadata
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        capabilities,
        prompt_field,
        image_field,
        video_field,
        audio_field,
        language_field,
        width_field,
        height_field,
        aspect_ratio_field,
        output_url_paths,
        output_text_paths,
        defaults,
        created,
    })
}

fn request_schema<'a>(openapi: &'a Value, id: &str) -> Option<&'a Value> {
    openapi
        .get("paths")?
        .as_object()?
        .iter()
        .find(|(path, value)| path.trim_start_matches('/') == id && value.get("post").is_some())
        .and_then(|(_, value)| value.pointer("/post/requestBody/content/application~1json/schema"))
        .or_else(|| named_schema(openapi, "Input"))
}

fn response_schema<'a>(openapi: &'a Value, id: &str) -> Option<&'a Value> {
    openapi
        .get("paths")?
        .as_object()?
        .iter()
        .find(|(path, value)| {
            path.trim_start_matches('/') == format!("{id}/requests/{{request_id}}")
                && value.get("get").is_some()
        })
        .and_then(|(_, value)| value.pointer("/get/responses/200/content/application~1json/schema"))
        .or_else(|| named_schema(openapi, "Output"))
}

fn named_schema<'a>(openapi: &'a Value, suffix: &str) -> Option<&'a Value> {
    openapi
        .pointer("/components/schemas")?
        .as_object()?
        .iter()
        .find(|(name, _)| name.ends_with(suffix))
        .map(|(_, schema)| schema)
}

fn resolved_schema<'a>(openapi: &'a Value, schema: &'a Value, depth: usize) -> &'a Value {
    if depth >= 12 {
        return schema;
    }
    schema
        .get("$ref")
        .and_then(Value::as_str)
        .and_then(|reference| reference.strip_prefix('#'))
        .and_then(|pointer| openapi.pointer(pointer))
        .map(|resolved| resolved_schema(openapi, resolved, depth + 1))
        .unwrap_or(schema)
}

fn collect_schema(
    openapi: &Value,
    schema: &Value,
    depth: usize,
    properties: &mut Map<String, Value>,
    required: &mut Vec<String>,
) {
    if depth >= 12 {
        return;
    }
    let schema = resolved_schema(openapi, schema, depth);
    if let Some(values) = schema.get("properties").and_then(Value::as_object) {
        properties.extend(values.clone());
    }
    if let Some(values) = schema.get("required").and_then(Value::as_array) {
        for value in values.iter().filter_map(Value::as_str) {
            if !required.iter().any(|item| item == value) {
                required.push(value.to_owned());
            }
        }
    }
    if let Some(values) = schema.get("allOf").and_then(Value::as_array) {
        for value in values {
            collect_schema(openapi, value, depth + 1, properties, required);
        }
    }
    for branch in ["anyOf", "oneOf"] {
        if let Some(value) = schema
            .get(branch)
            .and_then(Value::as_array)
            .and_then(|values| {
                values
                    .iter()
                    .find(|value| schema_has_properties(openapi, value, 0))
            })
        {
            collect_schema(openapi, value, depth + 1, properties, required);
        }
    }
}

fn schema_has_properties(openapi: &Value, schema: &Value, depth: usize) -> bool {
    if depth >= 12 {
        return false;
    }
    let schema = resolved_schema(openapi, schema, depth);
    schema
        .get("properties")
        .and_then(Value::as_object)
        .is_some()
        || ["allOf", "anyOf", "oneOf"].into_iter().any(|branch| {
            schema
                .get(branch)
                .and_then(Value::as_array)
                .is_some_and(|values| {
                    values
                        .iter()
                        .any(|value| schema_has_properties(openapi, value, depth + 1))
                })
        })
}

fn schema_default<'a>(openapi: &'a Value, schema: &'a Value, depth: usize) -> Option<&'a Value> {
    if depth >= 12 {
        return None;
    }
    let schema = resolved_schema(openapi, schema, depth);
    schema.get("default").or_else(|| {
        ["allOf", "anyOf", "oneOf"].into_iter().find_map(|branch| {
            schema
                .get(branch)
                .and_then(Value::as_array)
                .and_then(|values| {
                    values
                        .iter()
                        .find_map(|value| schema_default(openapi, value, depth + 1))
                })
        })
    })
}

fn schema_enum_value<'a>(openapi: &'a Value, schema: &'a Value, depth: usize) -> Option<&'a Value> {
    if depth >= 12 {
        return None;
    }
    let schema = resolved_schema(openapi, schema, depth);
    schema
        .get("enum")
        .and_then(Value::as_array)
        .and_then(|values| values.first())
        .or_else(|| {
            ["allOf", "anyOf", "oneOf"].into_iter().find_map(|branch| {
                schema
                    .get(branch)
                    .and_then(Value::as_array)
                    .and_then(|values| {
                        values
                            .iter()
                            .find_map(|value| schema_enum_value(openapi, value, depth + 1))
                    })
            })
        })
}

fn find_field(properties: &Map<String, Value>, exact: &[&str], tokens: &[&str]) -> Option<String> {
    exact
        .iter()
        .find(|name| properties.contains_key(**name))
        .map(|name| (*name).to_owned())
        .or_else(|| {
            properties.keys().find_map(|name| {
                let normalized = name.to_ascii_lowercase();
                tokens
                    .iter()
                    .all(|token| normalized.contains(token))
                    .then(|| name.clone())
            })
        })
}

fn find_media_field(properties: &Map<String, Value>, kind: &str) -> Option<String> {
    let exact: &[&str] = match kind {
        "image" => &[
            "image_url",
            "image_urls",
            "input_image_url",
            "input_image_urls",
            "reference_image_url",
            "reference_image_urls",
            "source_image_url",
            "image",
            "images",
        ],
        "video" => &[
            "video_url",
            "video_urls",
            "input_video_url",
            "reference_video_url",
            "source_video_url",
            "video",
        ],
        "audio" => &[
            "audio_url",
            "audio_urls",
            "input_audio_url",
            "reference_audio_url",
            "source_audio_url",
            "audio_file",
            "audio",
        ],
        _ => &[],
    };
    exact
        .iter()
        .find(|name| properties.contains_key(**name))
        .map(|name| (*name).to_owned())
        .or_else(|| {
            properties.keys().find_map(|name| {
                let normalized = name.to_ascii_lowercase();
                (normalized.contains(kind)
                    && ["url", "file", "input", "source", "reference"]
                        .iter()
                        .any(|token| normalized.contains(token)))
                .then(|| name.clone())
            })
        })
}

fn collect_output_paths(
    openapi: &Value,
    schema: &Value,
    prefix: &str,
    depth: usize,
    urls: &mut Vec<String>,
    text: &mut Vec<String>,
) {
    if depth >= 10 {
        return;
    }
    let schema = resolved_schema(openapi, schema, depth);
    for branch in ["allOf", "anyOf", "oneOf"] {
        if let Some(values) = schema.get(branch).and_then(Value::as_array) {
            for value in values {
                collect_output_paths(openapi, value, prefix, depth + 1, urls, text);
            }
        }
    }
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return;
    };
    for (name, value) in properties {
        let path = format!("{prefix}/{}", pointer_segment(name));
        let resolved = resolved_schema(openapi, value, depth);
        let normalized = name.to_ascii_lowercase();
        let is_url = resolved.get("format").and_then(Value::as_str) == Some("uri")
            || normalized == "url"
            || normalized.ends_with("_url");
        if is_url {
            if !urls.contains(&path) {
                urls.push(path);
            }
            continue;
        }
        if matches!(
            normalized.as_str(),
            "text" | "transcript" | "transcription" | "caption" | "description"
        ) && !text.contains(&path)
        {
            text.push(path.clone());
        }
        if let Some(items) = resolved.get("items") {
            collect_output_paths(openapi, items, &format!("{path}/0"), depth + 1, urls, text);
        } else {
            collect_output_paths(openapi, resolved, &path, depth + 1, urls, text);
        }
    }
}

fn collect_top_level_output_paths(openapi: &Value, schema: &Value, paths: &mut Vec<String>) {
    let mut properties = Map::new();
    let mut required = Vec::new();
    collect_schema(openapi, schema, 0, &mut properties, &mut required);
    for name in properties.keys() {
        let normalized = name.to_ascii_lowercase();
        if normalized == "url" || normalized.ends_with("_url") {
            continue;
        }
        let path = format!("/{}", pointer_segment(name));
        if !paths.contains(&path) {
            paths.push(path);
        }
    }
}

fn pointer_segment(value: &str) -> String {
    value.replace('~', "~0").replace('/', "~1")
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

    #[test]
    fn live_openapi_schema_becomes_an_executable_endpoint_mapping() {
        let model = serde_json::json!({
            "endpoint_id": "fal-ai/example/image-to-video",
            "metadata": {
                "display_name": "Example image video",
                "description": "Test endpoint",
                "category": "image-to-video",
                "date": "2026-01-02T03:04:05Z"
            },
            "openapi": {
                "paths": {
                    "/fal-ai/example/image-to-video": {
                        "post": {"requestBody": {"content": {"application/json": {
                            "schema": {"$ref": "#/components/schemas/ExampleInput"}
                        }}}}
                    },
                    "/fal-ai/example/image-to-video/requests/{request_id}": {
                        "get": {"responses": {"200": {"content": {"application/json": {
                            "schema": {"$ref": "#/components/schemas/ExampleOutput"}
                        }}}}}
                    }
                },
                "components": {"schemas": {
                    "ExampleInput": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "image_url": {"type": "string", "format": "uri"},
                            "aspect_ratio": {"type": "string", "default": "auto"},
                            "quality": {"type": "string", "enum": ["standard", "high"]}
                        },
                        "required": ["prompt", "image_url", "quality"]
                    },
                    "ExampleOutput": {
                        "type": "object",
                        "properties": {"video": {"$ref": "#/components/schemas/VideoFile"}}
                    },
                    "VideoFile": {
                        "type": "object",
                        "properties": {"url": {"type": "string", "format": "uri"}}
                    }
                }}
            }
        });
        let endpoint = endpoint_from_openapi(&model).unwrap();
        assert_eq!(endpoint.capabilities, ["image_to_video"]);
        assert_eq!(endpoint.prompt_field, "prompt");
        assert_eq!(endpoint.image_field.as_deref(), Some("image_url"));
        assert_eq!(endpoint.aspect_ratio_field.as_deref(), Some("aspect_ratio"));
        assert_eq!(endpoint.defaults["aspect_ratio"], "auto");
        assert_eq!(endpoint.defaults["quality"], "standard");
        assert!(endpoint.output_url_paths.contains(&"/video/url".into()));
    }

    #[test]
    fn live_schema_rejects_unmapped_required_inputs() {
        let model = serde_json::json!({
            "endpoint_id": "fal-ai/example/text-to-image",
            "metadata": {"category": "text-to-image"},
            "openapi": {
                "paths": {"/fal-ai/example/text-to-image": {"post": {"requestBody": {
                    "content": {"application/json": {"schema": {
                        "$ref": "#/components/schemas/Input"
                    }}}
                }}}},
                "components": {"schemas": {"Input": {
                    "properties": {
                        "prompt": {"type": "string"},
                        "account_specific_token": {"type": "string"}
                    },
                    "required": ["prompt", "account_specific_token"]
                }}}
            }
        });
        let error = endpoint_from_openapi(&model).unwrap_err().to_string();
        assert!(error.starts_with("Fal endpoint"));
        assert!(error.contains("account_specific_token"));
    }
}
