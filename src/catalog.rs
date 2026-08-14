//! Cached OpenRouter model discovery and capability-aware catalog metadata.
//!
//! OpenRouter's authenticated, user-scoped catalog is the source of truth for
//! selectable models. It reflects each bot API key's preferences, guardrails,
//! privacy policy, and eligibility without maintaining a hard-coded allowlist.

use crate::{Result, config::ModelProvider};
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
    pub model_provider: ModelProvider,
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
            "chat" | "model_upgrade" | "output_processing" | "error_processing" => {
                input("text") && output("text")
            }
            "intent_planning" | "intent_planning_fallback" => {
                input("text")
                    && output("text")
                    && self.supported_parameters.iter().any(|parameter| {
                        matches!(parameter.as_str(), "response_format" | "structured_outputs")
                    })
            }
            "image_understanding" => input("image") && output("text"),
            "video_understanding" => input("video") && output("text"),
            "image_generation" => output("image"),
            "audio_generation" | "speech_generation" => output("speech"),
            "music_generation" => output("audio"),
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
    /// Returns the current OpenRouter catalog, refreshing it at most once every ten minutes.
    pub async fn get_openrouter(
        &self,
        client: &reqwest::Client,
        base_url: &str,
        bot_id: &str,
        api_key: &str,
    ) -> Result<Arc<Vec<CatalogModel>>> {
        self.get(client, base_url, bot_id, api_key, ModelProvider::Openrouter)
            .await
    }

    /// Returns the current AI Hub catalog, refreshing it at most once every ten minutes.
    pub async fn get_aihub(
        &self,
        client: &reqwest::Client,
        base_url: &str,
        bot_id: &str,
        api_key: &str,
    ) -> Result<Arc<Vec<CatalogModel>>> {
        self.get(client, base_url, bot_id, api_key, ModelProvider::Aihub)
            .await
    }

    async fn get(
        &self,
        client: &reqwest::Client,
        base_url: &str,
        bot_id: &str,
        api_key: &str,
        model_provider: ModelProvider,
    ) -> Result<Arc<Vec<CatalogModel>>> {
        let cache_key = format!("{}:{bot_id}", model_provider.as_str());
        {
            let guard = self.inner.read().await;
            if let Some(cached) = guard.get(&cache_key)
                && cached.loaded.elapsed() < CATALOG_TTL
            {
                return Ok(cached.models.clone());
            }
        }

        let mut guard = self.inner.write().await;
        if let Some(cached) = guard.get(&cache_key)
            && cached.loaded.elapsed() < CATALOG_TTL
        {
            return Ok(cached.models.clone());
        }

        let fetched = match model_provider {
            ModelProvider::Openrouter => fetch_openrouter_catalog(client, base_url, api_key).await,
            ModelProvider::Aihub => fetch_aihub_catalog(client, base_url, api_key).await,
        };
        match fetched {
            Ok(models) => {
                let models = Arc::new(models);
                guard.insert(
                    cache_key.clone(),
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
                if let Some(cached) = guard.get(&cache_key) {
                    Ok(cached.models.clone())
                } else {
                    Err(error)
                }
            }
        }
    }
}

async fn fetch_openrouter_catalog(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
) -> Result<Vec<CatalogModel>> {
    let base = base_url.trim_end_matches('/');
    // This authenticated endpoint excludes models unavailable under the API
    // key's provider preferences, guardrails, privacy policy, and eligibility.
    let general = fetch_data(
        client,
        &format!("{base}/models/user?sort=newest&output_modalities=all"),
        api_key,
        "OpenRouter",
    )
    .await?;

    let mut models = BTreeMap::<String, CatalogModel>::new();
    merge_values(&mut models, general);
    // `/models/user` currently defaults to text-output models even when an
    // `output_modalities=all` query is supplied. Pull the non-text catalogs
    // explicitly so media-only models are not silently omitted from the
    // capability chooser.
    for modality in ["speech", "transcription"] {
        if let Ok(values) = fetch_data(
            client,
            &format!("{base}/models?sort=newest&output_modalities={modality}"),
            api_key,
            "OpenRouter",
        )
        .await
        {
            merge_values(&mut models, values);
        }
    }
    // Dedicated media catalogs expose generation constraints and SKU pricing.
    // Their entries must be inserted as well as enriched: media-only IDs are
    // commonly absent from `/models/user`.
    for (endpoint, output_modality) in [("images/models", "image"), ("videos/models", "video")] {
        if let Ok(values) =
            fetch_data(client, &format!("{base}/{endpoint}"), api_key, "OpenRouter").await
        {
            merge_media_values(&mut models, values, output_modality);
        }
    }
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

async fn fetch_aihub_catalog(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
) -> Result<Vec<CatalogModel>> {
    let values = fetch_data(
        client,
        &format!("{}/models", base_url.trim_end_matches('/')),
        api_key,
        "AI Hub",
    )
    .await?;
    let mut models = values
        .iter()
        .filter_map(parse_aihub_model)
        .collect::<Vec<_>>();
    if models.is_empty() {
        bail!("AI Hub returned an empty model catalog");
    }
    models.sort_by(|left, right| {
        right
            .created
            .unwrap_or_default()
            .cmp(&left.created.unwrap_or_default())
            .then_with(|| left.name.cmp(&right.name))
            .then_with(|| left.id.cmp(&right.id))
    });
    Ok(models)
}

async fn fetch_data(
    client: &reqwest::Client,
    url: &str,
    api_key: &str,
    provider: &str,
) -> Result<Vec<Value>> {
    let response = client
        .get(url)
        .bearer_auth(api_key)
        .timeout(Duration::from_secs(15))
        .send()
        .await
        .wrap_err_with(|| format!("Failed to fetch {provider} model catalog from {url}"))?;
    if !response.status().is_success() {
        bail!(
            "{provider} model catalog returned HTTP {}",
            response.status()
        );
    }
    let body: Value = response
        .json()
        .await
        .wrap_err_with(|| format!("{provider} returned an invalid model catalog"))?;
    body.get("data")
        .and_then(Value::as_array)
        .cloned()
        .wrap_err_with(|| format!("{provider} model catalog has no data array"))
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

fn merge_media_values(
    models: &mut BTreeMap<String, CatalogModel>,
    values: Vec<Value>,
    output_modality: &str,
) {
    for value in values {
        if let Some(mut incoming) = parse_model(&value) {
            append_unique(
                &mut incoming.output_modalities,
                vec![output_modality.to_owned()],
            );
            if incoming.input_modalities.is_empty() {
                incoming.input_modalities.push("text".to_owned());
            }
            match models.entry(incoming.id.clone()) {
                Entry::Vacant(entry) => {
                    entry.insert(incoming);
                }
                Entry::Occupied(mut entry) => merge_model(entry.get_mut(), incoming),
            }
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
        model_provider: ModelProvider::Openrouter,
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

fn parse_aihub_model(value: &Value) -> Option<CatalogModel> {
    let object = value.as_object()?;
    let id = string(object, "id")?;
    let is_image = id.to_ascii_lowercase().contains("image");
    let created = string(object, "created_at").and_then(|value| {
        chrono::DateTime::parse_from_rfc3339(&value)
            .ok()
            .map(|date| date.timestamp())
    });
    Some(CatalogModel {
        model_provider: ModelProvider::Aihub,
        name: string(object, "display_name").unwrap_or_else(|| id.clone()),
        description: "AI Hub model. This catalog does not publish pricing or context metadata."
            .to_owned(),
        created,
        input_modalities: vec!["text".to_owned()],
        output_modalities: vec![if is_image { "image" } else { "text" }.to_owned()],
        id,
        ..CatalogModel::default()
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
        assert!(model(&["text"], &["speech"]).supports("speech_generation"));
        assert!(model(&["text"], &["audio"]).supports("music_generation"));
        assert!(model(&["text", "video"], &["text"]).supports("video_understanding"));
        assert!(model(&["text", "image"], &["video"]).supports("video_generation"));
        assert!(model(&["text"], &["text"]).supports("output_processing"));
        assert!(model(&["text"], &["text"]).supports("error_processing"));
        let mut planner = model(&["text"], &["text"]);
        planner
            .supported_parameters
            .push("response_format".to_owned());
        assert!(planner.supports("intent_planning"));
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
    fn parses_aihub_chat_and_image_models_without_inventing_prices() {
        let chat = parse_aihub_model(&serde_json::json!({
            "id": "gpt-5.4-mini",
            "display_name": "GPT 5.4 Mini",
            "created_at": "2024-01-01T00:00:00Z"
        }))
        .unwrap();
        assert_eq!(chat.model_provider, ModelProvider::Aihub);
        assert!(chat.supports("chat"));
        assert!(chat.pricing.is_empty());

        let image = parse_aihub_model(&serde_json::json!({
            "id": "gpt-image-2",
            "display_name": "GPT Image 2"
        }))
        .unwrap();
        assert!(image.supports("image_generation"));
        assert!(!image.supports("chat"));
    }

    #[test]
    fn preserves_media_sku_pricing() {
        let parsed = parse_model(&serde_json::json!({
            "id": "vendor/video",
            "architecture": {"input_modalities": ["text"], "output_modalities": ["video"]},
            "pricing": {"prompt": "0", "completion": "0"},
            "pricing_skus": {"per-video-second": "0.50"}
        }))
        .unwrap();
        assert_eq!(parsed.pricing["prompt"], "0");
        assert_eq!(parsed.pricing["sku · per-video-second"], "0.50");
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

    #[test]
    fn dedicated_media_catalogs_insert_models_without_architecture() {
        let mut models = BTreeMap::new();
        merge_media_values(
            &mut models,
            vec![serde_json::json!({
                "id": "vendor/video",
                "name": "Video model",
                "supported_durations": [5, 10]
            })],
            "video",
        );
        let model = &models["vendor/video"];
        assert!(model.supports("video_generation"));
        assert_eq!(model.input_modalities, ["text"]);
    }
}
