//! Cached OpenRouter model discovery and capability-aware catalog metadata.
//!
//! OpenRouter's authenticated, user-scoped catalog is the source of truth for
//! selectable models. It reflects each bot API key's preferences, guardrails,
//! privacy policy, and eligibility without maintaining a hard-coded allowlist.

use crate::Result;
use eyre::{Context, ContextCompat, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::{
    collections::{BTreeMap, btree_map::Entry},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::RwLock;

const CATALOG_TTL: Duration = Duration::from_secs(10 * 60);

/// Browser-safe model metadata used by the administration model picker.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct CatalogModel {
    pub id: String,
    pub name: String,
    pub description: String,
    pub created: Option<i64>,
    pub context_length: Option<u64>,
    pub max_completion_tokens: Option<u64>,
    pub input_modalities: Vec<String>,
    pub output_modalities: Vec<String>,
    pub tokenizer: Option<String>,
    pub instruct_type: Option<String>,
    pub pricing: BTreeMap<String, String>,
    pub supported_parameters: Vec<String>,
    pub supported_voices: Vec<String>,
    pub knowledge_cutoff: Option<String>,
    pub expiration_date: Option<String>,
    pub supported_resolutions: Vec<String>,
    pub supported_aspect_ratios: Vec<String>,
    pub supported_durations: Vec<String>,
    pub supported_sizes: Vec<String>,
    pub supported_frame_images: Vec<String>,
    pub generates_audio: Option<bool>,
}

impl CatalogModel {
    /// Returns whether this model can serve the requested Teleforge capability.
    pub fn supports(&self, capability: &str) -> bool {
        let input = |value: &str| self.input_modalities.iter().any(|item| item == value);
        let output = |value: &str| self.output_modalities.iter().any(|item| item == value);
        match capability {
            "chat" => input("text") && output("text"),
            "image_understanding" => input("image") && output("text"),
            "video_understanding" => input("video") && output("text"),
            "image_generation" => output("image"),
            "audio_generation" => output("speech"),
            "transcription" => output("transcription"),
            "video_generation" => output("video"),
            _ => false,
        }
    }
}

#[derive(Clone)]
struct CachedCatalog {
    loaded: Instant,
    models: Arc<Vec<CatalogModel>>,
}

/// Shared time-bounded cache that avoids blocking each HTMX panel refresh on OpenRouter.
#[derive(Clone, Default)]
pub struct ModelCatalogCache {
    inner: Arc<RwLock<BTreeMap<String, CachedCatalog>>>,
}

impl ModelCatalogCache {
    /// Returns the current catalog, refreshing it at most once every ten minutes.
    pub async fn get(
        &self,
        client: &reqwest::Client,
        base_url: &str,
        bot_id: &str,
        api_key: &str,
    ) -> Result<Arc<Vec<CatalogModel>>> {
        {
            let guard = self.inner.read().await;
            if let Some(cached) = guard.get(bot_id)
                && cached.loaded.elapsed() < CATALOG_TTL
            {
                return Ok(cached.models.clone());
            }
        }

        let mut guard = self.inner.write().await;
        if let Some(cached) = guard.get(bot_id)
            && cached.loaded.elapsed() < CATALOG_TTL
        {
            return Ok(cached.models.clone());
        }

        match fetch_catalog(client, base_url, api_key).await {
            Ok(models) => {
                let models = Arc::new(models);
                guard.insert(
                    bot_id.to_owned(),
                    CachedCatalog {
                        loaded: Instant::now(),
                        models: models.clone(),
                    },
                );
                Ok(models)
            }
            Err(error) => {
                // A stale catalog is preferable to making model administration
                // unavailable during a transient OpenRouter outage.
                if let Some(cached) = guard.get(bot_id) {
                    Ok(cached.models.clone())
                } else {
                    Err(error)
                }
            }
        }
    }
}

async fn fetch_catalog(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
) -> Result<Vec<CatalogModel>> {
    let base = base_url.trim_end_matches('/');
    // This authenticated endpoint excludes models unavailable under the API
    // key's provider preferences, guardrails, privacy policy, and eligibility.
    let general = fetch_data(client, &format!("{base}/models/user?sort=newest"), api_key).await?;

    let mut models = BTreeMap::<String, CatalogModel>::new();
    merge_values(&mut models, general);
    if models.is_empty() {
        bail!("OpenRouter returned an empty model catalog");
    }
    let mut models = models.into_values().collect::<Vec<_>>();
    models.sort_by(|left, right| {
        right
            .created
            .unwrap_or_default()
            .cmp(&left.created.unwrap_or_default())
            .then_with(|| {
                left.name
                    .to_ascii_lowercase()
                    .cmp(&right.name.to_ascii_lowercase())
                    .then_with(|| left.id.cmp(&right.id))
            })
    });
    Ok(models)
}

async fn fetch_data(client: &reqwest::Client, url: &str, api_key: &str) -> Result<Vec<Value>> {
    let response = client
        .get(url)
        .bearer_auth(api_key)
        .timeout(Duration::from_secs(15))
        .send()
        .await
        .wrap_err_with(|| format!("Failed to fetch OpenRouter model catalog from {url}"))?;
    if !response.status().is_success() {
        bail!(
            "OpenRouter model catalog returned HTTP {}",
            response.status()
        );
    }
    let body: Value = response
        .json()
        .await
        .wrap_err("OpenRouter returned an invalid model catalog")?;
    body.get("data")
        .and_then(Value::as_array)
        .cloned()
        .context("OpenRouter model catalog has no data array")
}

fn merge_values(models: &mut BTreeMap<String, CatalogModel>, values: Vec<Value>) {
    for value in values {
        let Some(incoming) = parse_model(&value) else {
            continue;
        };
        match models.entry(incoming.id.clone()) {
            Entry::Vacant(entry) => {
                entry.insert(incoming);
            }
            Entry::Occupied(mut entry) => merge_model(entry.get_mut(), incoming),
        }
    }
}

fn parse_model(value: &Value) -> Option<CatalogModel> {
    let object = value.as_object()?;
    if requires_additional_identification(object) {
        return None;
    }
    let id = string(object, "id")?;
    let architecture = object.get("architecture").and_then(Value::as_object);
    let top_provider = object.get("top_provider").and_then(Value::as_object);
    Some(CatalogModel {
        name: string(object, "name").unwrap_or_else(|| id.clone()),
        description: string(object, "description").unwrap_or_default(),
        created: object.get("created").and_then(Value::as_i64),
        context_length: number(object, "context_length"),
        max_completion_tokens: top_provider.and_then(|item| number(item, "max_completion_tokens")),
        input_modalities: strings(architecture, "input_modalities"),
        output_modalities: strings(architecture, "output_modalities"),
        tokenizer: architecture.and_then(|item| string(item, "tokenizer")),
        instruct_type: architecture.and_then(|item| string(item, "instruct_type")),
        pricing: pricing(object),
        supported_parameters: supported_parameters(object),
        supported_voices: strings(Some(object), "supported_voices"),
        knowledge_cutoff: string(object, "knowledge_cutoff"),
        expiration_date: string(object, "expiration_date"),
        supported_resolutions: strings(Some(object), "supported_resolutions"),
        supported_aspect_ratios: strings(Some(object), "supported_aspect_ratios"),
        supported_durations: strings(Some(object), "supported_durations"),
        supported_sizes: strings(Some(object), "supported_sizes"),
        supported_frame_images: strings(Some(object), "supported_frame_images"),
        generates_audio: object.get("generate_audio").and_then(Value::as_bool),
        id,
    })
}

fn requires_additional_identification(object: &Map<String, Value>) -> bool {
    let non_empty_array = |key: &str| {
        object
            .get(key)
            .and_then(Value::as_array)
            .is_some_and(|items| !items.is_empty())
    };
    let true_value = |key: &str| object.get(key).and_then(Value::as_bool) == Some(true);
    non_empty_array("required_attestation_types")
        || true_value("requires_user_ids")
        || true_value("requiresUserIDs")
        || object
            .get("data_policy")
            .and_then(Value::as_object)
            .is_some_and(|policy| {
                policy.get("requires_user_ids").and_then(Value::as_bool) == Some(true)
                    || policy.get("requiresUserIDs").and_then(Value::as_bool) == Some(true)
            })
}

fn merge_model(current: &mut CatalogModel, incoming: CatalogModel) {
    if current.description.is_empty() {
        current.description = incoming.description;
    }
    current.context_length = current.context_length.or(incoming.context_length);
    current.max_completion_tokens = current
        .max_completion_tokens
        .or(incoming.max_completion_tokens);
    current.created = current.created.or(incoming.created);
    current.tokenizer = current.tokenizer.take().or(incoming.tokenizer);
    current.instruct_type = current.instruct_type.take().or(incoming.instruct_type);
    current.knowledge_cutoff = current
        .knowledge_cutoff
        .take()
        .or(incoming.knowledge_cutoff);
    current.expiration_date = current.expiration_date.take().or(incoming.expiration_date);
    append_unique(&mut current.input_modalities, incoming.input_modalities);
    append_unique(&mut current.output_modalities, incoming.output_modalities);
    append_unique(
        &mut current.supported_parameters,
        incoming.supported_parameters,
    );
    append_unique(&mut current.supported_voices, incoming.supported_voices);
    append_unique(
        &mut current.supported_resolutions,
        incoming.supported_resolutions,
    );
    append_unique(
        &mut current.supported_aspect_ratios,
        incoming.supported_aspect_ratios,
    );
    append_unique(
        &mut current.supported_durations,
        incoming.supported_durations,
    );
    append_unique(&mut current.supported_sizes, incoming.supported_sizes);
    append_unique(
        &mut current.supported_frame_images,
        incoming.supported_frame_images,
    );
    current.generates_audio = current.generates_audio.or(incoming.generates_audio);
    current.pricing.extend(incoming.pricing);
}

fn append_unique(target: &mut Vec<String>, values: Vec<String>) {
    for value in values {
        if !target.contains(&value) {
            target.push(value);
        }
    }
}

fn supported_parameters(object: &Map<String, Value>) -> Vec<String> {
    let mut parameters = match object.get("supported_parameters") {
        Some(Value::Array(values)) => values
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        Some(Value::Object(values)) => values.keys().cloned().collect(),
        _ => Vec::new(),
    };
    append_unique(
        &mut parameters,
        strings(Some(object), "allowed_passthrough_parameters"),
    );
    parameters
}

fn pricing(object: &Map<String, Value>) -> BTreeMap<String, String> {
    let mut result = BTreeMap::new();
    for (prefix, key) in [("", "pricing"), ("sku · ", "pricing_skus")] {
        if let Some(values) = object.get(key).and_then(Value::as_object) {
            result.extend(values.iter().filter_map(|(key, value)| {
                scalar(value).map(|value| (format!("{prefix}{key}"), value))
            }));
        }
    }
    result
}

fn strings(object: Option<&Map<String, Value>>, key: &str) -> Vec<String> {
    object
        .and_then(|object| object.get(key))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(scalar)
        .collect()
}

fn string(object: &Map<String, Value>, key: &str) -> Option<String> {
    object.get(key).and_then(scalar)
}

fn number(object: &Map<String, Value>, key: &str) -> Option<u64> {
    object.get(key).and_then(|value| {
        value
            .as_u64()
            .or_else(|| value.as_str()?.parse::<u64>().ok())
    })
}

fn scalar(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(input: &[&str], output: &[&str]) -> CatalogModel {
        CatalogModel {
            input_modalities: input.iter().map(ToString::to_string).collect(),
            output_modalities: output.iter().map(ToString::to_string).collect(),
            ..CatalogModel::default()
        }
    }

    #[test]
    fn capability_filter_distinguishes_understanding_and_generation() {
        assert!(model(&["text", "image"], &["text"]).supports("image_understanding"));
        assert!(!model(&["text", "image"], &["text"]).supports("image_generation"));
        assert!(model(&["text"], &["image"]).supports("image_generation"));
        assert!(model(&["audio"], &["transcription"]).supports("transcription"));
        assert!(model(&["text"], &["speech"]).supports("audio_generation"));
        assert!(model(&["text", "video"], &["text"]).supports("video_understanding"));
        assert!(model(&["text", "image"], &["video"]).supports("video_generation"));
    }

    #[test]
    fn parses_catalog_metadata() {
        let value = serde_json::json!({
            "id": "vendor/model",
            "name": "Model",
            "description": "Useful model",
            "context_length": 128000,
            "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
            "supported_parameters": ["tools", "temperature"],
            "top_provider": {"max_completion_tokens": 8192}
        });
        let parsed = parse_model(&value).expect("model");
        assert_eq!(parsed.id, "vendor/model");
        assert_eq!(parsed.context_length, Some(128000));
        assert_eq!(parsed.max_completion_tokens, Some(8192));
        assert_eq!(parsed.pricing["prompt"], "0.000001");
        assert!(parsed.supports("image_understanding"));
    }

    #[test]
    fn excludes_models_requiring_additional_identification() {
        let value = serde_json::json!({
            "id": "vendor/restricted",
            "name": "Restricted",
            "required_attestation_types": ["organization"]
        });
        assert!(parse_model(&value).is_none());
    }
}
