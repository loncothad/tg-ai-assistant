//! Cached provider model discovery and capability-aware catalog metadata.
//!
//! OpenRouter and AI Hub publish their native catalogs, while fal.ai publishes
//! a paginated endpoint index, account pricing, and on-demand OpenAPI schemas.
//! Provider data is the source of truth; local fal.ai entries are explicit
//! overrides for private or irregular endpoint contracts.

use crate::{
    Result,
    config::{FalConfig, ModelProvider},
    http::HttpClient,
};
use eyre::{Context, ContextCompat, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
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
    /// Exact capabilities declared by schema-configured providers.
    #[serde(default)]
    pub supported_capabilities: Vec<String>,
}

impl CatalogModel {
    /// Returns whether this model can serve the requested Teleforge capability.
    pub fn supports(&self, capability: &str) -> bool {
        if !self.supported_capabilities.is_empty() {
            return self
                .supported_capabilities
                .iter()
                .any(|item| item == capability)
                || legacy_capability_alias(self, capability);
        }
        let input = |value: &str| self.input_modalities.iter().any(|item| item == value);
        let output = |value: &str| self.output_modalities.iter().any(|item| item == value);
        match capability {
            "chat" | "model_upgrade" | "output_processing" | "error_processing" => {
                input("text") && output("text")
            }
            "intent_planning" | "intent_planning_fallback" => {
                input("text")
                    && input("image")
                    && output("text")
                    && self.supported_parameters.iter().any(|parameter| {
                        matches!(parameter.as_str(), "response_format" | "structured_outputs")
                    })
            }
            "image_understanding" => input("image") && output("text"),
            "video_understanding" => input("video") && output("text"),
            "image_generation" | "text_to_image" => input("text") && output("image"),
            "image_to_image" => input("image") && output("image"),
            "audio_generation" | "speech_generation" | "text_to_speech" => {
                input("text") && (output("speech") || output("audio"))
            }
            "music_generation" | "text_to_audio" => input("text") && output("audio"),
            "video_to_audio" => input("video") && output("audio"),
            "transcription" => output("transcription") || (input("audio") && output("text")),
            "video_generation" | "text_to_video" => input("text") && output("video"),
            "image_to_video" => input("image") && output("video"),
            "video_to_video" => input("video") && output("video"),
            "text_to_3d" => input("text") && output("3d"),
            "image_to_3d" => input("image") && output("3d"),
            "text_to_image_vector" => input("text") && (output("svg") || output("vector")),
            "image_to_image_vector" => input("image") && (output("svg") || output("vector")),
            _ => false,
        }
    }
}

fn legacy_capability_alias(model: &CatalogModel, requested: &str) -> bool {
    let has = |value: &str| {
        model
            .supported_capabilities
            .iter()
            .any(|item| item == value)
    };
    let input = |value: &str| model.input_modalities.iter().any(|item| item == value);
    match requested {
        "text_to_image" => input("text") && has("image_generation"),
        "image_to_image" => input("image") && has("image_generation"),
        "text_to_video" => input("text") && has("video_generation"),
        "image_to_video" => input("image") && has("video_generation"),
        "video_to_video" => input("video") && has("video_generation"),
        "text_to_audio" => input("text") && has("music_generation"),
        "video_to_audio" => input("video") && has("music_generation"),
        "text_to_speech" => input("text") && (has("speech_generation") || has("audio_generation")),
        _ => false,
    }
}

/// Whether a capability has its own input/output-specific model selector.
pub fn is_specialized_generation_capability(capability: &str) -> bool {
    matches!(
        capability,
        "text_to_image"
            | "image_to_image"
            | "text_to_video"
            | "image_to_video"
            | "video_to_video"
            | "text_to_audio"
            | "video_to_audio"
            | "text_to_speech"
            | "image_to_3d"
            | "text_to_3d"
            | "text_to_image_vector"
            | "image_to_image_vector"
    )
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
        client: &HttpClient,
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
        client: &HttpClient,
        base_url: &str,
        bot_id: &str,
        api_key: &str,
    ) -> Result<Arc<Vec<CatalogModel>>> {
        self.get(client, base_url, bot_id, api_key, ModelProvider::Aihub)
            .await
    }

    /// Returns fal.ai's live endpoint catalogue merged with administrator
    /// schema overrides. The live list is account-scoped and paginated.
    pub async fn get_fal(
        &self,
        client: &HttpClient,
        config: &FalConfig,
        bot_id: &str,
        api_key: &str,
    ) -> Result<Arc<Vec<CatalogModel>>> {
        let discovered = self
            .get(
                client,
                &config.catalog_url,
                bot_id,
                api_key,
                ModelProvider::Fal,
            )
            .await?;
        Ok(Arc::new(merge_fal_overrides(discovered.as_ref(), config)))
    }

    async fn get(
        &self,
        client: &HttpClient,
        base_url: &str,
        bot_id: &str,
        api_key: &str,
        model_provider: ModelProvider,
    ) -> Result<Arc<Vec<CatalogModel>>> {
        let credential_fingerprint = Sha256::digest(api_key.as_bytes());
        let cache_key = format!(
            "{}:{bot_id}:{}",
            model_provider.as_str(),
            hex::encode(&credential_fingerprint[..8])
        );
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
            ModelProvider::Fal => {
                match tokio::time::timeout(
                    Duration::from_secs(45),
                    fetch_fal_catalog(client, base_url, api_key),
                )
                .await
                {
                    Ok(result) => result,
                    Err(_) => bail!("Fal model discovery timed out"),
                }
            }
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

/// Converts fal.ai endpoint overrides into the common admin catalog shape.
pub fn fal_catalog(config: &FalConfig) -> Vec<CatalogModel> {
    let mut models = config
        .endpoints
        .iter()
        .map(|endpoint| {
            let mut input = vec!["text".to_owned()];
            if endpoint.image_field.is_some() {
                input.push("image".to_owned());
            }
            if endpoint.video_field.is_some() {
                input.push("video".to_owned());
            }
            if endpoint.audio_field.is_some() {
                input.push("audio".to_owned());
            }
            let mut output = Vec::new();
            for capability in &endpoint.capabilities {
                let modality = match capability.as_str() {
                    "image_generation" => "image",
                    "text_to_image" | "image_to_image" => "image",
                    "speech_generation" | "audio_generation" => "speech",
                    "music_generation" | "text_to_audio" | "video_to_audio" => "audio",
                    "transcription" => "transcription",
                    "video_generation" | "text_to_video" | "image_to_video" | "video_to_video" => {
                        "video"
                    }
                    "image_understanding" | "video_understanding" => "text",
                    "text_to_3d" | "image_to_3d" => "3d",
                    "text_to_image_vector" | "image_to_image_vector" => "vector",
                    _ => continue,
                };
                if !output.iter().any(|item| item == modality) {
                    output.push(modality.to_owned());
                }
            }
            CatalogModel {
                model_provider: ModelProvider::Fal,
                id: endpoint.id.clone(),
                name: if endpoint.name.is_empty() {
                    endpoint.id.clone()
                } else {
                    endpoint.name.clone()
                },
                description: if endpoint.description.is_empty() {
                    "Configured fal.ai endpoint".to_owned()
                } else {
                    endpoint.description.clone()
                },
                created: endpoint.created,
                input_modalities: input,
                output_modalities: output,
                supported_capabilities: endpoint.capabilities.clone(),
                ..CatalogModel::default()
            }
        })
        .collect::<Vec<_>>();
    models.sort_by(|left, right| {
        right
            .created
            .unwrap_or_default()
            .cmp(&left.created.unwrap_or_default())
            .then_with(|| left.name.cmp(&right.name))
    });
    models
}

fn merge_fal_overrides(discovered: &[CatalogModel], config: &FalConfig) -> Vec<CatalogModel> {
    let mut models = discovered
        .iter()
        .cloned()
        .map(|model| (model.id.clone(), model))
        .collect::<BTreeMap<_, _>>();
    for endpoint in &config.endpoints {
        let mut configured = fal_catalog(&FalConfig {
            endpoints: vec![endpoint.clone()],
            ..config.clone()
        })
        .into_iter()
        .next()
        .expect("one configured endpoint produces one catalog entry");
        if let Some(live) = models.get(&endpoint.id) {
            if endpoint.name.is_empty() {
                configured.name.clone_from(&live.name);
            }
            if endpoint.description.is_empty() {
                configured.description.clone_from(&live.description);
            }
            configured.created = endpoint.created.or(live.created);
            configured.pricing.clone_from(&live.pricing);
        }
        models.insert(endpoint.id.clone(), configured);
    }
    let mut models = models.into_values().collect::<Vec<_>>();
    sort_catalog(&mut models);
    models
}

async fn fetch_fal_catalog(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
) -> Result<Vec<CatalogModel>> {
    let base = base_url.trim_end_matches('/');
    let mut cursor = None::<String>;
    let mut models = BTreeMap::<String, CatalogModel>::new();
    // The cursor is server-issued, and the hard page bound prevents a faulty
    // upstream from keeping an administration request alive indefinitely.
    for _ in 0..100 {
        let mut request = client
            .get(format!("{base}/models"))
            .header(reqwest::header::AUTHORIZATION, format!("Key {api_key}"))
            .query(&[("limit", "100"), ("status", "active")])
            .timeout(Duration::from_secs(20));
        if let Some(value) = cursor.as_deref() {
            request = request.query(&[("cursor", value)]);
        }
        let response = request
            .send()
            .await
            .context("Failed to fetch fal.ai model catalog")?;
        if !response.status().is_success() {
            bail!("Fal model catalog returned HTTP {}", response.status());
        }
        let body: Value = response
            .json()
            .await
            .context("Fal returned an invalid model catalog")?;
        let values = body
            .get("models")
            .and_then(Value::as_array)
            .context("Fal model catalog has no models array")?;
        for value in values {
            if let Some(model) = parse_fal_model(value) {
                models.insert(model.id.clone(), model);
            }
        }
        let next_cursor = body
            .get("next_cursor")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_owned);
        if next_cursor.is_some() && next_cursor == cursor {
            bail!("Fal model catalog repeated its pagination cursor");
        }
        cursor = next_cursor;
        if cursor.is_none() || body.get("has_more").and_then(Value::as_bool) == Some(false) {
            break;
        }
    }
    if models.is_empty() {
        bail!("Fal returned an empty compatible model catalog");
    }
    // Pricing is useful metadata, not a prerequisite for choosing a model.
    // Keep a slow/rate-limited billing endpoint from holding the catalog UI.
    let _ = tokio::time::timeout(
        Duration::from_secs(8),
        enrich_fal_pricing(client, base, api_key, &mut models),
    )
    .await;
    let mut models = models.into_values().collect::<Vec<_>>();
    sort_catalog(&mut models);
    Ok(models)
}

async fn enrich_fal_pricing(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
    models: &mut BTreeMap<String, CatalogModel>,
) {
    let ids = models.keys().cloned().collect::<Vec<_>>();
    for chunk in ids.chunks(50) {
        let joined = chunk.join(",");
        let Ok(response) = client
            .get(format!("{base_url}/models/pricing"))
            .header(reqwest::header::AUTHORIZATION, format!("Key {api_key}"))
            .query(&[("endpoint_id", joined.as_str())])
            .timeout(Duration::from_secs(5))
            .send()
            .await
        else {
            continue;
        };
        if !response.status().is_success() {
            continue;
        }
        let Ok(body) = response.json::<Value>().await else {
            continue;
        };
        for price in body
            .get("prices")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(id) = price.get("endpoint_id").and_then(Value::as_str) else {
                continue;
            };
            let Some(model) = models.get_mut(id) else {
                continue;
            };
            let unit = price
                .get("unit")
                .and_then(Value::as_str)
                .unwrap_or("billing unit");
            let currency = price
                .get("currency")
                .and_then(Value::as_str)
                .unwrap_or("USD");
            if let Some(value) = price.get("unit_price").and_then(scalar) {
                model
                    .pricing
                    .insert(format!("fal · {currency} per {unit}"), value);
            }
        }
    }
}

fn parse_fal_model(value: &Value) -> Option<CatalogModel> {
    let object = value.as_object()?;
    let id = string(object, "endpoint_id")?;
    let metadata = object.get("metadata")?.as_object()?;
    let category = string(metadata, "category")?;
    let capabilities = fal_model_capabilities(
        &category,
        &id,
        metadata
            .get("display_name")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        metadata
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default(),
    );
    if capabilities.is_empty() {
        return None;
    }
    let (input_modalities, output_modalities) = fal_modalities(&capabilities);
    let created = ["date", "updated_at"]
        .into_iter()
        .filter_map(|key| string(metadata, key))
        .find_map(|value| {
            chrono::DateTime::parse_from_rfc3339(&value)
                .ok()
                .map(|date| date.timestamp())
        });
    Some(CatalogModel {
        model_provider: ModelProvider::Fal,
        name: string(metadata, "display_name").unwrap_or_else(|| id.clone()),
        description: string(metadata, "description").unwrap_or_default(),
        created,
        input_modalities,
        output_modalities,
        supported_capabilities: capabilities,
        id,
        ..CatalogModel::default()
    })
}

/// Maps fal.ai's public model categories to Teleforge's input/output-specific
/// capability names. Unknown operational categories such as training are
/// intentionally omitted from the assistant model picker.
pub(crate) fn fal_category_capabilities(category: &str) -> Vec<String> {
    let normalized = category
        .trim()
        .to_ascii_lowercase()
        .replace(['_', ' '], "-");
    let capabilities: &[&str] = match normalized.as_str() {
        "text-to-image" => &["text_to_image"],
        "image-to-image" => &["image_to_image"],
        "text-to-video" => &["text_to_video"],
        "image-to-video" => &["image_to_video"],
        "video-to-video" => &["video_to_video"],
        "text-to-audio" | "text-to-music" => &["text_to_audio"],
        "video-to-audio" => &["video_to_audio"],
        "text-to-speech" => &["text_to_speech"],
        "speech-to-text" | "audio-to-text" | "transcription" => &["transcription"],
        "text-to-3d" => &["text_to_3d"],
        "image-to-3d" => &["image_to_3d"],
        "text-to-vector" | "text-to-svg" => &["text_to_image_vector"],
        "image-to-vector" | "image-to-svg" => &["image_to_image_vector"],
        "image-to-text" => &["image_understanding"],
        "video-to-text" => &["video_understanding"],
        // fal.ai currently groups narrow classifiers and general visual
        // understanding endpoints together. Schema validation at selection
        // time determines which input kind a particular endpoint accepts.
        "vision" => &["image_understanding", "video_understanding"],
        _ => &[],
    };
    capabilities
        .iter()
        .map(|value| (*value).to_owned())
        .collect()
}

pub(crate) fn fal_model_capabilities(
    category: &str,
    id: &str,
    name: &str,
    description: &str,
) -> Vec<String> {
    let mut capabilities = fal_category_capabilities(category);
    let identity = format!("{id} {name} {description}").to_ascii_lowercase();
    if identity
        .split(|character: char| !character.is_ascii_alphanumeric())
        .any(|token| token == "svg")
    {
        for capability in &mut capabilities {
            *capability = match capability.as_str() {
                "text_to_image" => "text_to_image_vector".to_owned(),
                "image_to_image" => "image_to_image_vector".to_owned(),
                _ => capability.clone(),
            };
        }
    }
    capabilities
}

fn fal_modalities(capabilities: &[String]) -> (Vec<String>, Vec<String>) {
    let mut input = Vec::new();
    let mut output = Vec::new();
    for capability in capabilities {
        let (inputs, result): (&[&str], &str) = match capability.as_str() {
            "text_to_image" => (&["text"], "image"),
            "image_to_image" => (&["text", "image"], "image"),
            "text_to_video" => (&["text"], "video"),
            "image_to_video" => (&["text", "image"], "video"),
            "video_to_video" => (&["text", "video"], "video"),
            "text_to_audio" => (&["text"], "audio"),
            "video_to_audio" => (&["text", "video"], "audio"),
            "text_to_speech" => (&["text"], "speech"),
            "transcription" => (&["audio"], "transcription"),
            "text_to_3d" => (&["text"], "3d"),
            "image_to_3d" => (&["text", "image"], "3d"),
            "text_to_image_vector" => (&["text"], "vector"),
            "image_to_image_vector" => (&["text", "image"], "vector"),
            "image_understanding" => (&["text", "image"], "text"),
            "video_understanding" => (&["text", "video"], "text"),
            _ => continue,
        };
        append_unique(
            &mut input,
            inputs.iter().map(|value| (*value).to_owned()).collect(),
        );
        if !output.iter().any(|value| value == result) {
            output.push(result.to_owned());
        }
    }
    (input, output)
}

fn sort_catalog(models: &mut [CatalogModel]) {
    models.sort_by(|left, right| {
        right
            .created
            .unwrap_or_default()
            .cmp(&left.created.unwrap_or_default())
            .then_with(|| left.name.cmp(&right.name))
            .then_with(|| left.id.cmp(&right.id))
    });
}

async fn fetch_openrouter_catalog(
    client: &HttpClient,
    base_url: &str,
    api_key: &str,
) -> Result<Vec<CatalogModel>> {
    let base = base_url.trim_end_matches('/');
    // Start with the user-scoped catalog, then merge the authenticated public
    // catalog. `/models/user` reflects account policy but can omit otherwise
    // selectable models; request-time routing remains the final eligibility
    // authority. Identity-attestation models are filtered during parsing.
    let general = fetch_data(
        client,
        &format!("{base}/models/user?sort=newest&output_modalities=all"),
        api_key,
        "OpenRouter",
    )
    .await?;

    let mut models = BTreeMap::<String, CatalogModel>::new();
    merge_values(&mut models, general);
    if let Ok(values) = fetch_data(
        client,
        &format!("{base}/models?sort=newest&output_modalities=all"),
        api_key,
        "OpenRouter",
    )
    .await
    {
        merge_values(&mut models, values);
    }
    // `/models/user` currently defaults to text-output models even when an
    // `output_modalities=all` query is supplied. Pull the non-text catalogs
    // explicitly so media-only models are not silently omitted from the
    // capability chooser.
    for modality in ["audio", "speech", "transcription"] {
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
    client: &HttpClient,
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
    client: &HttpClient,
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
        supported_capabilities: Vec::new(),
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
        let mut planner = model(&["text", "image"], &["text"]);
        planner
            .supported_parameters
            .push("response_format".to_owned());
        assert!(planner.supports("intent_planning"));
        assert!(!model(&["text"], &["text"]).supports("intent_planning"));
    }

    #[test]
    fn fal_declared_capabilities_do_not_leak_across_broad_modalities() {
        let model = CatalogModel {
            input_modalities: vec!["text".into(), "image".into()],
            output_modalities: vec!["text".into()],
            supported_capabilities: vec!["image_understanding".into()],
            ..CatalogModel::default()
        };
        assert!(model.supports("image_understanding"));
        assert!(!model.supports("chat"));
        assert!(!model.supports("image_to_image"));
    }

    #[test]
    fn fal_live_categories_map_to_exact_selector_capabilities() {
        let model = parse_fal_model(&serde_json::json!({
            "endpoint_id": "fal-ai/example/image-to-3d",
            "metadata": {
                "display_name": "Example 3D",
                "description": "Makes a mesh",
                "category": "image-to-3d",
                "date": "2026-02-03T04:05:06Z"
            }
        }))
        .unwrap();
        assert_eq!(model.model_provider, ModelProvider::Fal);
        assert!(model.supports("image_to_3d"));
        assert!(!model.supports("text_to_3d"));
        assert_eq!(model.input_modalities, ["text", "image"]);
        assert_eq!(model.output_modalities, ["3d"]);
        assert!(model.created.is_some());
        assert!(
            parse_fal_model(&serde_json::json!({
                "endpoint_id": "fal-ai/example/trainer",
                "metadata": {"category": "training"}
            }))
            .is_none()
        );
        let vector = parse_fal_model(&serde_json::json!({
            "endpoint_id": "fal-ai/image2svg",
            "metadata": {
                "display_name": "Image2SVG",
                "description": "Produces an SVG file",
                "category": "image-to-image"
            }
        }))
        .unwrap();
        assert!(vector.supports("image_to_image_vector"));
        assert!(!vector.supports("image_to_image"));
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
