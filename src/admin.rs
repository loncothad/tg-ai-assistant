//! Telegram Mini App administration UI and authenticated HTMX endpoints.

use crate::{
    Result,
    catalog::ModelCatalogCache,
    config::{Config, ModelProvider},
    db::{ModelRouting, Store},
    fal::FalClient,
    http::HttpClient,
};
use axum::{
    Form, Json, Router,
    extract::{Multipart, Path, Query, State},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use chrono::Utc;
use eyre::{ContextCompat, WrapErr, bail};
use hmac::{Hmac, Mac, digest::KeyInit};
use rand::RngExt;
use serde::Deserialize;
use serde_json::Value;
use sha2::Sha256;
use std::{collections::BTreeMap, sync::Arc};

#[derive(Clone)]
pub struct AdminState {
    pub config: Arc<Config>,
    pub store: Store,
    pub client: HttpClient,
    catalog: ModelCatalogCache,
    fal: FalClient,
}

impl AdminState {
    /// Creates shared state for the authenticated Mini App endpoints.
    pub fn new(config: Arc<Config>, store: Store, client: HttpClient) -> Self {
        let fal = FalClient::new(client.clone(), config.fal.clone());
        Self {
            config,
            store,
            client,
            catalog: ModelCatalogCache::default(),
            fal,
        }
    }
}

pub fn router(state: AdminState) -> Router {
    Router::new()
        .route("/admin/{bot_id}", get(shell))
        .route("/admin/{bot_id}/", get(shell))
        .route("/admin/{bot_id}/panel", get(panel))
        .route("/admin/{bot_id}/models", get(models))
        .route("/admin/{bot_id}/model-details", get(model_details))
        .route("/admin/{bot_id}/model-providers", get(model_providers))
        .route("/admin/{bot_id}/model", post(set_model))
        .route("/admin/{bot_id}/search", post(set_search))
        .route("/admin/{bot_id}/capability", post(set_capability))
        .route("/admin/{bot_id}/credential", post(set_credential))
        .route("/admin/{bot_id}/content", post(set_content))
        .route("/admin/{bot_id}/skills/export", get(export_skills))
        .route("/admin/{bot_id}/access", post(set_access))
        .with_state(state)
}

async fn shell(Path(bot_id): Path<String>, State(state): State<AdminState>) -> Response {
    if state.config.bot(&bot_id).is_none() {
        return StatusCode::NOT_FOUND.into_response();
    }
    let mut random = [0u8; 18];
    rand::rng().fill(&mut random);
    let nonce = base64::Engine::encode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, random);
    let html = format!(
        r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<base href="/admin/{bot_id}/"><title>Teleforge Admin</title>
<script src="https://telegram.org/js/telegram-web-app.js?59"></script>
<script src="https://cdn.jsdelivr.net/npm/htmx.org@2.0.9/dist/htmx.min.js" integrity="sha384-ESlCao+z/oasnu2Uc/5K1LQTI7YCF2KKO4xakCPQCFuiHhCh8Oa/R5NwHY6guZ3m" crossorigin="anonymous"></script>
<style nonce="{nonce}">{css}</style></head><body><main class="app">
<header class="topbar"><div class="logo"><svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="m4 12 5 5L20 6" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg></div><div><h1>Teleforge</h1><div class="subtitle">Secure bot administration · {bot_id}</div></div></header>
<div id="panel"><div class="loading"><div><div class="spinner"></div>Authenticating…</div></div></div>
</main>
<dialog id="model-dialog"><div class="picker-layout"><div class="picker-list"><div class="picker-head"><div><h2>Choose a model</h2><div id="picker-subtitle" class="subtitle">Live provider catalogs</div></div><button id="model-close" class="icon-button" type="button" aria-label="Close">×</button></div><div id="model-provider-tabs" class="provider-tabs" role="tablist" aria-label="Model provider"></div><div class="search-wrap"><svg viewBox="0 0 24 24" fill="none"><circle cx="11" cy="11" r="7" stroke="currentColor" stroke-width="2"/><path d="m16 16 4 4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg><input id="model-search" type="search" autocomplete="off" placeholder="Fuzzy search names, IDs, descriptions…"></div><div id="model-count" class="result-count"></div><div id="model-results" class="model-results"></div></div><div id="model-detail" class="picker-detail"></div></div></dialog>
<template id="model-settings"><div class="openrouter-routing"><label>OpenRouter routing<select name="routing"><option value="auto">Auto · OpenRouter default</option><option value="cheapest">Cheapest provider</option><option value="throughput">Highest throughput</option><option value="latency">Lowest latency</option><option value="exacto">Exacto tool quality</option></select></label><label>OpenRouter endpoint<select name="provider"><option value="">Auto · any compatible provider</option></select></label><small class=provider-help>Loading the providers that currently serve this model…</small></div><div class="direct-provider-note muted">AI Hub models are sent directly to AI Hub; OpenRouter routing controls do not apply.</div><button type="button" data-save-model id="model-save">Use this model</button></template>
<script nonce="{nonce}">{js}</script></body></html>"#,
        css = include_str!("../assets/admin.css"),
        js = include_str!("../assets/admin.js"),
    );
    let mut response = Html(html).into_response();
    let csp = format!(
        "default-src 'none'; script-src 'nonce-{nonce}' https://telegram.org https://cdn.jsdelivr.net; style-src 'nonce-{nonce}'; connect-src 'self'; img-src 'self' data: https:; font-src 'self'; frame-ancestors https://web.telegram.org https://*.telegram.org"
    );
    response.headers_mut().insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_str(&csp).unwrap(),
    );
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response.headers_mut().insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    response
}

async fn panel(
    Path(bot): Path<String>,
    State(state): State<AdminState>,
    headers: HeaderMap,
) -> Response {
    render_authorized(&state, &bot, &headers).await
}

async fn models(
    Path(bot): Path<String>,
    State(state): State<AdminState>,
    headers: HeaderMap,
) -> Response {
    let (_, bot) = match authorize(&state, &bot, &headers) {
        Ok(auth) => auth,
        Err(error) => return auth_error(error),
    };
    let mut models = Vec::new();
    let mut providers = Vec::new();
    let (openrouter, aihub, fal) = tokio::join!(
        load_provider_catalog(&state, &bot, ModelProvider::Openrouter, "OpenRouter"),
        load_provider_catalog(&state, &bot, ModelProvider::Aihub, "AI Hub"),
        load_provider_catalog(&state, &bot, ModelProvider::Fal, "fal.ai"),
    );
    for loaded in [openrouter, aihub, fal] {
        if !loaded.configured {
            providers.push(serde_json::json!({
                "id": loaded.provider.as_str(), "label": loaded.label, "available": false,
                "message": "API key is not configured"
            }));
            continue;
        }
        match loaded.catalog {
            Ok(catalog) => {
                models.extend(catalog.iter().cloned());
                providers.push(serde_json::json!({
                    "id": loaded.provider.as_str(), "label": loaded.label, "available": true,
                    "models": catalog.len()
                }));
            }
            Err(error) => {
                tracing::warn!(bot_id = %bot, provider = loaded.provider.as_str(), error = %format!("{error:#}"), "model catalog unavailable");
                providers.push(serde_json::json!({
                    "id": loaded.provider.as_str(), "label": loaded.label, "available": false,
                    "message": if loaded.provider == ModelProvider::Fal {
                        "Live fal.ai catalog is unavailable; verify the FAL_KEY API scope and service logs"
                    } else {
                        "Catalog is temporarily unavailable"
                    }
                }));
            }
        }
    }
    no_store(Json(serde_json::json!({ "models": models, "providers": providers })).into_response())
}

struct LoadedProviderCatalog {
    provider: ModelProvider,
    label: &'static str,
    configured: bool,
    catalog: Result<Arc<Vec<crate::catalog::CatalogModel>>>,
}

async fn load_provider_catalog(
    state: &AdminState,
    bot: &str,
    provider: ModelProvider,
    label: &'static str,
) -> LoadedProviderCatalog {
    let configured = state
        .store
        .credential_configured(bot, provider.as_str())
        .await
        .unwrap_or(false);
    let catalog = if configured {
        catalog_for(state, bot, provider).await
    } else {
        Err(eyre::eyre!("Provider API key is not configured"))
    };
    LoadedProviderCatalog {
        provider,
        label,
        configured,
        catalog,
    }
}

#[derive(Deserialize)]
struct ModelProvidersQuery {
    model: String,
    model_provider: ModelProvider,
    capability: String,
}

#[derive(Deserialize)]
struct ModelDetailsQuery {
    model: String,
    model_provider: ModelProvider,
}

async fn model_details(
    Path(bot): Path<String>,
    State(state): State<AdminState>,
    headers: HeaderMap,
    Query(query): Query<ModelDetailsQuery>,
) -> Response {
    let (_, bot) = match authorize(&state, &bot, &headers) {
        Ok(auth) => auth,
        Err(error) => return auth_error(error),
    };
    if query.model_provider != ModelProvider::Fal {
        return message(
            StatusCode::BAD_REQUEST,
            "Detailed endpoint schemas are currently available for fal.ai models",
        );
    }
    let catalog = match catalog_for(&state, &bot, ModelProvider::Fal).await {
        Ok(catalog) => catalog,
        Err(error) => return internal(error),
    };
    if !catalog.iter().any(|model| model.id == query.model) {
        return message(StatusCode::BAD_REQUEST, "Unknown fal.ai model");
    }
    let key = match fal_key(&state, &bot).await {
        Ok(key) => key,
        Err(error) => return internal(error),
    };
    match state.fal.model_details(&query.model, &key).await {
        Ok(details) => no_store(Json(details).into_response()),
        Err(error) => {
            tracing::warn!(bot_id = %bot, model = %query.model, error = %format!("{error:#}"), "fal model details unavailable");
            message(
                StatusCode::BAD_GATEWAY,
                "Fal.ai did not return a usable OpenAPI schema for this endpoint",
            )
        }
    }
}

async fn model_providers(
    Path(bot): Path<String>,
    State(state): State<AdminState>,
    headers: HeaderMap,
    Query(query): Query<ModelProvidersQuery>,
) -> Response {
    let (_, bot) = match authorize(&state, &bot, &headers) {
        Ok(auth) => auth,
        Err(error) => return auth_error(error),
    };
    if query.model_provider != ModelProvider::Openrouter {
        return no_store(Json(serde_json::json!({ "providers": [] })).into_response());
    }
    let catalog = match catalog_for(&state, &bot, query.model_provider).await {
        Ok(catalog) => catalog,
        Err(error) => return internal(error),
    };
    if !catalog.iter().any(|model| model.id == query.model) {
        return message(StatusCode::BAD_REQUEST, "Unknown model");
    }
    let endpoint_url = match model_endpoints_url(
        &state.config.openrouter.base_url,
        &query.model,
        &query.capability,
    ) {
        Ok(url) => url,
        Err(error) => return internal(error),
    };
    let response = match state
        .client
        .get(endpoint_url)
        .bearer_auth(match openrouter_key(&state, &bot).await {
            Ok(key) => key,
            Err(error) => return internal(error),
        })
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) => return internal(error.into()),
    };
    if !response.status().is_success() {
        return message(
            StatusCode::BAD_GATEWAY,
            "OpenRouter provider listing failed",
        );
    }
    let value: Value = match response.json().await {
        Ok(value) => value,
        Err(error) => return internal(error.into()),
    };
    let mut providers = value
        .pointer("/data/endpoints")
        .or_else(|| value.get("endpoints"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|endpoint| {
            Some(serde_json::json!({
                "tag": endpoint.get("tag").or_else(|| endpoint.get("provider_tag")).or_else(|| endpoint.get("provider_slug"))?.as_str()?,
                "name": endpoint.get("provider_name")?.as_str()?,
                "context_length": endpoint.get("context_length"),
                "max_completion_tokens": endpoint.get("max_completion_tokens"),
                "pricing": endpoint.get("pricing"),
                "quantization": endpoint.get("quantization"),
                "uptime": endpoint.get("uptime_last_30m"),
                "latency": endpoint.get("latency_last_30m"),
                "throughput": endpoint.get("throughput_last_30m")
            }))
        })
        .collect::<Vec<_>>();
    providers.sort_by(|left, right| {
        left.get("name")
            .and_then(Value::as_str)
            .cmp(&right.get("name").and_then(Value::as_str))
    });
    no_store(Json(serde_json::json!({ "providers": providers })).into_response())
}

fn model_endpoints_url(base_url: &str, model: &str, capability: &str) -> Result<url::Url> {
    let mut url = url::Url::parse(base_url).wrap_err("Invalid OpenRouter base URL")?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| eyre::eyre!("OpenRouter base URL cannot contain path segments"))?;
        segments.pop_if_empty().push("models");
        if matches!(
            capability,
            "image_generation" | "text_to_image" | "image_to_image"
        ) {
            segments.pop().push("images").push("models");
        } else if matches!(
            capability,
            "video_generation" | "text_to_video" | "image_to_video" | "video_to_video"
        ) {
            segments.pop().push("videos").push("models");
        }
        for segment in model.split('/') {
            segments.push(segment);
        }
        segments.push("endpoints");
    }
    Ok(url)
}

async fn openrouter_key(state: &AdminState, bot: &str) -> Result<String> {
    state
        .store
        .credential(bot, "openrouter")
        .await?
        .context("OpenRouter API key is not configured")
}

async fn aihub_key(state: &AdminState, bot: &str) -> Result<String> {
    state
        .store
        .credential(bot, "aihub")
        .await?
        .context("AI Hub API key is not configured")
}

async fn fal_key(state: &AdminState, bot: &str) -> Result<String> {
    state
        .store
        .credential(bot, "fal")
        .await?
        .context("Fal API key is not configured")
}

async fn catalog_for(
    state: &AdminState,
    bot: &str,
    model_provider: ModelProvider,
) -> Result<Arc<Vec<crate::catalog::CatalogModel>>> {
    match model_provider {
        ModelProvider::Openrouter => {
            let key = openrouter_key(state, bot).await?;
            state
                .catalog
                .get_openrouter(&state.client, &state.config.openrouter.base_url, bot, &key)
                .await
        }
        ModelProvider::Aihub => {
            let key = aihub_key(state, bot).await?;
            state
                .catalog
                .get_aihub(&state.client, &state.config.aihub.base_url, bot, &key)
                .await
        }
        ModelProvider::Fal => {
            let key = fal_key(state, bot).await?;
            state
                .catalog
                .get_fal(&state.client, &state.config.fal, bot, &key)
                .await
        }
    }
}

#[derive(Deserialize)]
struct ModelForm {
    capability: String,
    model: String,
    model_provider: ModelProvider,
    routing: String,
    provider: Option<String>,
}
async fn set_model(
    Path(bot): Path<String>,
    State(state): State<AdminState>,
    headers: HeaderMap,
    Form(form): Form<ModelForm>,
) -> Response {
    let (user, bot) = match authorize(&state, &bot, &headers) {
        Ok(auth) => auth,
        Err(e) => return auth_error(e),
    };
    if matches!(
        form.capability.as_str(),
        "intent_planning"
            | "intent_planning_fallback"
            | "chat"
            | "output_processing"
            | "error_processing"
            | "model_upgrade"
    ) && form.model_provider == ModelProvider::Fal
    {
        return message(
            StatusCode::BAD_REQUEST,
            "Fal endpoints are not general chat or intent-processing models",
        );
    }
    if matches!(
        form.capability.as_str(),
        "intent_planning" | "intent_planning_fallback"
    ) && form.model_provider != ModelProvider::Openrouter
    {
        return message(
            StatusCode::BAD_REQUEST,
            "Intent processing currently requires an OpenRouter model",
        );
    }
    let allowed = catalog_for(&state, &bot, form.model_provider)
        .await
        .map(|models| {
            models
                .iter()
                .any(|model| model.id == form.model && model.supports(&form.capability))
        })
        .unwrap_or_else(|_| {
            form.model_provider == ModelProvider::Openrouter
                && model_allowed_fallback(&state.config, &form.capability, &form.model)
        });
    if !allowed {
        return message(
            StatusCode::BAD_REQUEST,
            "Unknown model or model does not support this capability",
        );
    }
    if form.model_provider == ModelProvider::Fal {
        let key = match fal_key(&state, &bot).await {
            Ok(key) => key,
            Err(error) => return message(StatusCode::BAD_REQUEST, &error.to_string()),
        };
        if let Err(error) = state
            .fal
            .endpoint(&form.model, &form.capability, &key)
            .await
        {
            return message(
                StatusCode::BAD_REQUEST,
                &format!("Fal model schema is not executable: {error}"),
            );
        }
    }
    if form.model_provider != ModelProvider::Openrouter
        && (form.routing != "auto"
            || form
                .provider
                .as_deref()
                .is_some_and(|value| !value.is_empty()))
    {
        return message(
            StatusCode::BAD_REQUEST,
            "This provider does not support OpenRouter endpoint routing controls",
        );
    }
    if !matches!(
        form.routing.as_str(),
        "auto" | "cheapest" | "throughput" | "latency" | "exacto"
    ) {
        return message(StatusCode::BAD_REQUEST, "Unknown routing strategy");
    }
    if form.routing == "exacto"
        && !matches!(
            form.capability.as_str(),
            "chat" | "image_understanding" | "video_understanding"
        )
    {
        return message(
            StatusCode::BAD_REQUEST,
            "Exacto routing is available only for chat-style capabilities",
        );
    }
    let provider = form
        .provider
        .filter(|value| !value.is_empty() && value != "auto");
    if provider.as_deref().is_some_and(|value| {
        !value.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.' | '/')
        })
    }) {
        return message(StatusCode::BAD_REQUEST, "Invalid provider slug");
    }
    if let Err(e) = state
        .store
        .set_model(
            &bot,
            &form.capability,
            &form.model,
            ModelRouting {
                model_provider: form.model_provider,
                strategy: form.routing,
                provider,
            },
            user,
        )
        .await
    {
        return internal(e);
    }
    render(&state, &bot).await
}
#[derive(Deserialize)]
struct SearchForm {
    provider: String,
}
async fn set_search(
    Path(bot): Path<String>,
    State(state): State<AdminState>,
    headers: HeaderMap,
    Form(form): Form<SearchForm>,
) -> Response {
    let (user, bot) = match authorize(&state, &bot, &headers) {
        Ok(auth) => auth,
        Err(e) => return auth_error(e),
    };
    if form
        .provider
        .parse::<crate::config::SearchProvider>()
        .is_err()
    {
        return message(StatusCode::BAD_REQUEST, "Unknown provider");
    }
    if !state
        .store
        .credential_configured(&bot, &form.provider)
        .await
        .unwrap_or(false)
    {
        return message(
            StatusCode::BAD_REQUEST,
            "The selected search provider is unavailable because its API key is not configured",
        );
    }
    if let Err(e) = state
        .store
        .set_search_provider(&bot, &form.provider, user)
        .await
    {
        return internal(e);
    }
    render(&state, &bot).await
}
#[derive(Deserialize)]
struct CapabilityForm {
    capability: String,
    enabled: bool,
}
async fn set_capability(
    Path(bot): Path<String>,
    State(state): State<AdminState>,
    headers: HeaderMap,
    Form(form): Form<CapabilityForm>,
) -> Response {
    let (actor, bot) = match authorize(&state, &bot, &headers) {
        Ok(auth) => auth,
        Err(error) => return auth_error(error),
    };
    if form.enabled {
        let settings = match state.store.settings(&bot).await {
            Ok(settings) => settings,
            Err(error) => return internal(error),
        };
        if form.capability == "web_fetch"
            && model_provider_for_capability(&settings, "chat") != ModelProvider::Openrouter
        {
            return message(
                StatusCode::BAD_REQUEST,
                "Web Fetch requires an OpenRouter chat model because it is an OpenRouter server tool",
            );
        }
        let required_provider = if form.capability == "search" {
            settings
                .search_provider
                .unwrap_or_else(|| state.config.search.default_provider.as_str().to_owned())
        } else if form.capability == "web_fetch" {
            "openrouter".to_owned()
        } else {
            model_provider_for_skill(&settings, &form.capability)
                .as_str()
                .to_owned()
        };
        match state
            .store
            .credential_configured(&bot, &required_provider)
            .await
        {
            Ok(true) => {}
            Ok(false) => {
                return message(
                    StatusCode::BAD_REQUEST,
                    &format!(
                        "This capability is unavailable because the {required_provider} API key is not configured"
                    ),
                );
            }
            Err(error) => return internal(error),
        }
    }
    if let Err(e) = state
        .store
        .set_capability(&bot, &form.capability, form.enabled)
        .await
    {
        return internal(e);
    }
    if let Err(error) = state
        .store
        .audit(
            &bot,
            Some(actor),
            "capability.set",
            Some(&format!("{}={}", form.capability, form.enabled)),
        )
        .await
    {
        return internal(error);
    }
    render(&state, &bot).await
}
#[derive(Deserialize)]
struct CredentialForm {
    provider: String,
    secret: Option<String>,
    action: String,
}
async fn set_credential(
    Path(bot): Path<String>,
    State(state): State<AdminState>,
    headers: HeaderMap,
    Form(form): Form<CredentialForm>,
) -> Response {
    let (actor, bot) = match authorize(&state, &bot, &headers) {
        Ok(auth) => auth,
        Err(error) => return auth_error(error),
    };
    let result = if form.action == "remove" {
        state.store.remove_credential(&bot, &form.provider).await
    } else {
        let secret = form.secret.unwrap_or_default();
        if secret.trim().is_empty() {
            return message(StatusCode::BAD_REQUEST, "Secret cannot be empty");
        }
        state
            .store
            .set_credential(&bot, &form.provider, &secret)
            .await
    };
    if let Err(e) = result {
        return internal(e);
    }
    if let Err(error) = state
        .store
        .audit(
            &bot,
            Some(actor),
            "credential.change",
            Some(&format!("{}:{}", form.provider, form.action)),
        )
        .await
    {
        return internal(error);
    }
    render(&state, &bot).await
}
async fn set_content(
    Path(bot): Path<String>,
    State(state): State<AdminState>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Response {
    let (actor, bot) = match authorize(&state, &bot, &headers) {
        Ok(auth) => auth,
        Err(error) => return auth_error(error),
    };
    let mut kind = String::new();
    let mut mode = String::new();
    let mut content = String::new();
    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or_default().to_owned();
        let bytes = match field.bytes().await {
            Ok(v) => v,
            Err(_) => return message(StatusCode::BAD_REQUEST, "Invalid upload"),
        };
        if bytes.len() > 524_288 {
            return message(StatusCode::PAYLOAD_TOO_LARGE, "Import exceeds 512 KiB");
        };
        let text = String::from_utf8_lossy(&bytes).to_string();
        match name.as_str() {
            "kind" => kind = text,
            "mode" => mode = text,
            "content" | "file" if !text.trim().is_empty() => content = text,
            _ => {}
        }
    }
    let result = match mode.as_str() {
        "default" => {
            state
                .store
                .set_custom_content(&bot, &kind, None, false)
                .await
        }
        "disable" => {
            let current = state.store.settings(&bot).await;
            match current {
                Ok(s) => {
                    state
                        .store
                        .set_custom_content(
                            &bot,
                            &kind,
                            if kind == "prompt" {
                                s.custom_system_prompt
                            } else {
                                s.custom_skills
                            },
                            false,
                        )
                        .await
                }
                Err(e) => Err(e),
            }
        }
        _ if kind == "skills" && content.trim_start().starts_with('{') => {
            import_skill_bundle(&state.store, &bot, &content).await
        }
        _ => {
            state
                .store
                .set_custom_content(&bot, &kind, Some(content), true)
                .await
        }
    };
    if let Err(e) = result {
        return internal(e);
    }
    if let Err(error) = state
        .store
        .audit(
            &bot,
            Some(actor),
            "content.change",
            Some(&format!("{kind}:{mode}")),
        )
        .await
    {
        return internal(error);
    }
    render(&state, &bot).await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SkillBundle {
    version: u32,
    #[serde(default)]
    builtins: Vec<ImportedBuiltinSkill>,
    custom: Option<String>,
    #[serde(default)]
    custom_enabled: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImportedBuiltinSkill {
    id: String,
    enabled: bool,
    // These fields make the exported bundle self-documenting. Built-in text is
    // immutable at runtime and therefore intentionally ignored during import.
    description: Option<String>,
    instructions: Option<String>,
}

async fn import_skill_bundle(store: &Store, bot: &str, content: &str) -> Result<()> {
    let bundle: SkillBundle =
        serde_json::from_str(content).wrap_err("Invalid skill bundle JSON")?;
    if bundle.version != 1 {
        bail!("Unsupported skill bundle version")
    }
    let mut settings = store.settings(bot).await?;
    for skill in bundle.builtins {
        let _metadata = (skill.description, skill.instructions);
        settings
            .capabilities
            .set(&skill.id, skill.enabled)
            .wrap_err_with(|| format!("Unknown built-in skill: {}", skill.id))?;
    }
    settings.custom_skills = bundle.custom;
    settings.custom_skills_enabled = bundle.custom_enabled;
    store.save_settings(bot, &settings).await
}
async fn export_skills(
    Path(bot): Path<String>,
    State(state): State<AdminState>,
    headers: HeaderMap,
) -> Response {
    let (_, bot) = match authorize(&state, &bot, &headers) {
        Ok(auth) => auth,
        Err(error) => return auth_error(error),
    };
    let settings = match state.store.settings(&bot).await {
        Ok(v) => v,
        Err(e) => return internal(e),
    };
    let enabled = |id: &str| settings.capabilities.enabled(id);
    let skills=crate::defaults::BUILTIN_SKILLS.iter().map(|s|serde_json::json!({"id":s.id,"description":s.description,"enabled":enabled(s.id),"instructions":s.instructions})).collect::<Vec<_>>();
    let bundle = serde_json::json!({"version":1,"builtins":skills,"custom":settings.custom_skills,"custom_enabled":settings.custom_skills_enabled});
    no_store(
        Html(format!(
            "<textarea rows=18>{}</textarea>",
            esc(&serde_json::to_string_pretty(&bundle).unwrap_or_default())
        ))
        .into_response(),
    )
}
#[derive(Deserialize)]
struct AccessForm {
    user_id: u64,
    allowed: bool,
}
async fn set_access(
    Path(bot): Path<String>,
    State(state): State<AdminState>,
    headers: HeaderMap,
    Form(form): Form<AccessForm>,
) -> Response {
    let (actor, bot) = match authorize(&state, &bot, &headers) {
        Ok(auth) => auth,
        Err(e) => return auth_error(e),
    };
    if state
        .config
        .bot(&bot)
        .is_some_and(|b| b.admin_user_ids.contains(&form.user_id))
        && !form.allowed
    {
        return message(
            StatusCode::BAD_REQUEST,
            "Configured administrators cannot be denied",
        );
    }
    if let Err(e) = state
        .store
        .set_user_allowed(&bot, form.user_id, form.allowed, actor)
        .await
    {
        return internal(e);
    }
    render(&state, &bot).await
}

async fn render_authorized(state: &AdminState, bot: &str, headers: &HeaderMap) -> Response {
    let (_, canonical) = match authorize(state, bot, headers) {
        Ok(auth) => auth,
        Err(error) => return auth_error(error),
    };
    render(state, &canonical).await
}
async fn render(state: &AdminState, bot: &str) -> Response {
    let response = match render_html(state, bot).await {
        Ok(v) => Html(v).into_response(),
        Err(e) => internal(e),
    };
    no_store(response)
}
async fn render_html(state: &AdminState, bot: &str) -> Result<String> {
    let settings = state.store.settings(bot).await?;
    let mut configured = BTreeMap::new();
    for p in ["openrouter", "aihub", "fal", "brave", "exa", "serpapi"] {
        configured.insert(p, state.store.credential_configured(bot, p).await?);
    }
    let openrouter_ready = configured.get("openrouter").copied().unwrap_or(false);
    let aihub_ready = configured.get("aihub").copied().unwrap_or(false);
    let fal_ready = configured.get("fal").copied().unwrap_or(false);
    let chat_provider_ready = openrouter_ready || aihub_ready;
    let any_model_provider_ready = openrouter_ready || aihub_ready || fal_ready;
    let route = |capability: &str| {
        settings
            .model_routing
            .get(capability)
            .cloned()
            .unwrap_or_default()
    };
    let model_forms = [
        model_form(
            "Intent processing",
            "intent_planning",
            &settings.selected_planner_model,
            &route("intent_planning"),
            &configured,
            openrouter_ready,
        ),
        model_form(
            "Intent processing fallback",
            "intent_planning_fallback",
            &settings.selected_planner_fallback_model,
            &route("intent_planning_fallback"),
            &configured,
            openrouter_ready,
        ),
        model_form(
            "General chat",
            "chat",
            &settings.selected_model,
            &route("chat"),
            &configured,
            chat_provider_ready,
        ),
        model_form(
            "Text output processing",
            "output_processing",
            &settings.selected_output_processing_model,
            &route("output_processing"),
            &configured,
            chat_provider_ready,
        ),
        model_form(
            "Error explanation",
            "error_processing",
            &settings.selected_error_processing_model,
            &route("error_processing"),
            &configured,
            chat_provider_ready,
        ),
        model_form(
            "Advanced model",
            "model_upgrade",
            &settings.selected_upgrade_model,
            &route("model_upgrade"),
            &configured,
            chat_provider_ready,
        ),
        model_form(
            "Image understanding",
            "image_understanding",
            &settings.selected_image_understanding_model,
            &route("image_understanding"),
            &configured,
            any_model_provider_ready,
        ),
        model_form(
            "Video understanding",
            "video_understanding",
            &settings.selected_video_understanding_model,
            &route("video_understanding"),
            &configured,
            any_model_provider_ready,
        ),
        model_form(
            "Transcription",
            "transcription",
            &settings.selected_transcription_model,
            &route("transcription"),
            &configured,
            any_model_provider_ready,
        ),
        specialized_model_form(
            &settings,
            "Text → image",
            "text_to_image",
            &settings.selected_image_generation_model,
            &configured,
            any_model_provider_ready,
        ),
        specialized_model_form(
            &settings,
            "Image → image",
            "image_to_image",
            &settings.selected_image_generation_model,
            &configured,
            any_model_provider_ready,
        ),
        specialized_model_form(
            &settings,
            "Text → video",
            "text_to_video",
            &settings.selected_video_generation_model,
            &configured,
            any_model_provider_ready,
        ),
        specialized_model_form(
            &settings,
            "Image → video",
            "image_to_video",
            &settings.selected_video_generation_model,
            &configured,
            any_model_provider_ready,
        ),
        specialized_model_form(
            &settings,
            "Video → video",
            "video_to_video",
            &settings.selected_video_generation_model,
            &configured,
            any_model_provider_ready,
        ),
        specialized_model_form(
            &settings,
            "Text → audio",
            "text_to_audio",
            &settings.selected_music_generation_model,
            &configured,
            any_model_provider_ready,
        ),
        specialized_model_form(
            &settings,
            "Video → audio",
            "video_to_audio",
            &settings.selected_music_generation_model,
            &configured,
            any_model_provider_ready,
        ),
        specialized_model_form(
            &settings,
            "Text → speech",
            "text_to_speech",
            &settings.selected_audio_generation_model,
            &configured,
            any_model_provider_ready,
        ),
        specialized_model_form(
            &settings,
            "Image → 3D",
            "image_to_3d",
            "",
            &configured,
            fal_ready,
        ),
        specialized_model_form(
            &settings,
            "Text → 3D",
            "text_to_3d",
            "",
            &configured,
            fal_ready,
        ),
        specialized_model_form(
            &settings,
            "Text → image (vector HTML)",
            "text_to_image_vector",
            "",
            &configured,
            fal_ready,
        ),
        specialized_model_form(
            &settings,
            "Image → image (vector HTML)",
            "image_to_image_vector",
            "",
            &configured,
            fal_ready,
        ),
    ]
    .join("");
    let provider = settings
        .search_provider
        .as_deref()
        .unwrap_or(state.config.search.default_provider.as_str());
    let search_available = configured.get(provider).copied().unwrap_or(false);
    let enabled = |id: &str| settings.capabilities.enabled(id);
    let caps = crate::defaults::BUILTIN_SKILLS
        .iter()
        .map(|skill| {
            let on = enabled(skill.id);
            let selected_model_provider = model_provider_for_skill(&settings, skill.id);
            let available = match skill.id {
                "search" => search_available,
                "web_fetch" => {
                    model_provider_for_capability(&settings, "chat") == ModelProvider::Openrouter
                        && openrouter_ready
                }
                "youtube" => {
                    settings.capabilities.enabled("video_understanding")
                        && configured
                            .get(model_provider_for_capability(&settings, "video_understanding").as_str())
                            .copied()
                            .unwrap_or(false)
                }
                "prompt_expansion" => openrouter_ready,
                _ => configured
                    .get(selected_model_provider.as_str())
                    .copied()
                    .unwrap_or(false),
            };
            let availability = if available {
                "Available"
            } else if skill.id == "search" {
                "Unavailable: the selected search provider has no configured API key"
            } else if skill.id == "web_fetch" {
                "Unavailable: Web Fetch requires an OpenRouter chat model and API key"
            } else if skill.id == "youtube" {
                "Unavailable: YouTube description requires media understanding and a video-understanding provider key"
            } else if skill.id == "prompt_expansion" {
                "Unavailable: prompt expansion requires the OpenRouter intent processor key"
            } else {
                "Unavailable: the API key for this skill's selected model provider is missing"
            };
            format!(
                "<form hx-post=\"capability\" hx-target=\"#panel\"><input type=hidden name=capability value=\"{}\"><input type=hidden name=enabled value=\"{}\"><button {}>{}: {}</button><small>{}<br>{}</small></form>",
                esc(skill.id),
                !on,
                if available { "" } else { "disabled" },
                esc(skill.id),
                if on { "enabled" } else { "disabled" },
                esc(skill.description),
                availability
            )
        })
        .collect::<String>();
    let creds=configured.iter().map(|(p,on)|format!("<form hx-post=\"credential\" hx-target=\"#panel\"><input type=hidden name=provider value=\"{p}\"><input type=password name=secret autocomplete=off placeholder=\"{}\"><div class=grid><button name=action value=save>Save {p}</button><button name=action value=remove>Remove</button></div><small>{}</small></form>",if *on{"Replace configured key"}else{"Enter API key"},if *on{"Configured and encrypted (write-only)"}else{"Unavailable: corresponding environment variable was not provided at startup and no key has been saved"})).collect::<String>();
    let content = |kind: &str, value: Option<&str>, enabled: bool| {
        format!(
            "<form hx-post=\"content\" hx-target=\"#panel\" hx-encoding=\"multipart/form-data\"><input type=hidden name=kind value=\"{kind}\"><textarea name=content rows=8 placeholder=\"Paste custom {kind}\">{}</textarea><input type=file name=file accept=\".md,.txt,.json,text/plain,text/markdown,application/json\"><div class=grid><button name=mode value=save>Import and enable</button><button name=mode value=disable>Disable custom</button><button name=mode value=default>Reset to build default</button></div><small>Custom {kind}: {}</small></form>",
            esc(value.unwrap_or_default()),
            if enabled { "enabled" } else { "disabled" }
        )
    };
    Ok(format!(
        "<nav class=\"jump-nav\"><button type=button data-jump=\"models\">Models</button><button type=button data-jump=\"skills\">Skills</button><button type=button data-jump=\"credentials\">Credentials</button><button type=button data-jump=\"instructions\">Instructions</button><button type=button data-jump=\"access\">Access</button></nav>
        <section id=\"models\"><div class=\"section-head\"><div><h2>Models by capability</h2><p>OpenRouter, AI Hub, and the live fal.ai endpoint catalog are separated by provider and capability. Intent processing lists only models with text and image input, text output, and structured-output support.</p></div><span class=\"badge\">Live catalogs</span></div><div class=\"grid model-grid\">{model_forms}</div><form hx-post=\"search\" hx-target=\"#panel\" hx-swap=\"innerHTML transition:true\"><label>Web search provider<select name=provider>{}</select></label><button>Save search provider</button></form></section>
        <section id=\"skills\"><div class=\"section-head\"><div><h2>Built-in skills</h2><p>Enable only the callable APIs and instructions this bot should expose.</p></div></div><div class=grid>{caps}</div></section>
        <section id=\"credentials\"><div class=\"section-head\"><div><h2>API credentials</h2><p>Encrypted at rest and never returned to the browser.</p></div></div><div class=grid>{creds}</div></section>
        <section id=\"instructions\"><div class=\"section-head\"><div><h2>System prompt</h2><p>Override or restore the prompt compiled into this binary.</p></div></div>{}</section>
        <section><div class=\"section-head\"><div><h2>Skills import and export</h2><p>Manage custom skill bundles without redeploying.</p></div></div>{}<button class=primary hx-get=\"skills/export\" hx-target=\"#skill-export\">Export current skill bundle</button><div id=skill-export></div></section>
        <section id=\"access\"><div class=\"section-head\"><div><h2>User access</h2><p>Allow or deny a Telegram user ID for this bot only.</p></div></div><form hx-post=\"access\" hx-target=\"#panel\"><input type=number name=user_id min=1 required placeholder=\"Telegram user ID\"><div class=grid><button name=allowed value=true>Allow</button><button name=allowed value=false>Deny</button></div></form></section>",
        provider_options(provider, &configured),
        content(
            "prompt",
            settings.custom_system_prompt.as_deref(),
            settings.custom_system_prompt_enabled
        ),
        content(
            "skills",
            settings.custom_skills.as_deref(),
            settings.custom_skills_enabled
        )
    ))
}

fn specialized_model_form(
    settings: &crate::db::BotSettings,
    label: &str,
    capability: &str,
    legacy_fallback: &str,
    configured: &BTreeMap<&str, bool>,
    available: bool,
) -> String {
    let selected = settings
        .specialized_generation_models
        .get(capability)
        .map(String::as_str)
        .unwrap_or(legacy_fallback);
    let routing = settings
        .model_routing
        .get(capability)
        .cloned()
        .unwrap_or_else(|| {
            if matches!(
                capability,
                "image_to_3d" | "text_to_3d" | "text_to_image_vector" | "image_to_image_vector"
            ) {
                return ModelRouting {
                    model_provider: ModelProvider::Fal,
                    ..ModelRouting::default()
                };
            }
            let legacy = match capability {
                "text_to_image"
                | "image_to_image"
                | "text_to_image_vector"
                | "image_to_image_vector" => "image_generation",
                "text_to_video" | "image_to_video" | "video_to_video" => "video_generation",
                "text_to_speech" => "speech_generation",
                "text_to_audio" | "video_to_audio" => "music_generation",
                _ => capability,
            };
            settings
                .model_routing
                .get(legacy)
                .cloned()
                .unwrap_or_default()
        });
    model_form(label, capability, selected, &routing, configured, available)
}

fn model_form(
    label: &str,
    capability: &str,
    selected: &str,
    routing: &ModelRouting,
    configured: &BTreeMap<&str, bool>,
    chooser_available: bool,
) -> String {
    let selected_provider = routing.model_provider.as_str();
    let selected_available = configured.get(selected_provider).copied().unwrap_or(false);
    let selected_configured = !selected.trim().is_empty();
    let initial_summary = if selected_configured {
        "Loading provider metadata…"
    } else {
        "Choose a model to enable this capability"
    };
    format!(
        "<article class=\"card model-card\" data-capability=\"{}\" data-model=\"{}\" data-model-provider=\"{}\"><h3>{}</h3><div class=model-name>{}</div><div class=model-id>{}</div><p class=model-summary>{}</p><div class=\"chips model-chips\"></div><div class=route-line><span>API: {}</span><span>Routing: {}</span><span>Endpoint: {}</span></div><button type=button class=model-picker data-capability=\"{}\" data-model=\"{}\" data-model-provider=\"{}\" data-routing=\"{}\" data-provider=\"{}\" {disabled}>{button}</button><small class=\"{status_class}\">{status}</small></article>",
        esc(capability),
        esc(selected),
        esc(selected_provider),
        esc(label),
        esc(selected),
        esc(selected),
        esc(initial_summary),
        esc(selected_provider),
        esc(&routing.strategy),
        esc(routing.provider.as_deref().unwrap_or("auto")),
        esc(capability),
        esc(selected),
        esc(selected_provider),
        esc(&routing.strategy),
        esc(routing.provider.as_deref().unwrap_or("")),
        disabled = if chooser_available { "" } else { "disabled" },
        button = if chooser_available {
            "Choose model"
        } else {
            "Provider key required"
        },
        status_class = if selected_available && selected_configured {
            "status-ok"
        } else {
            "status-off"
        },
        status = if !selected_configured {
            "Not configured: choose a model"
        } else if selected_available {
            "Ready"
        } else {
            "Unavailable: selected model provider API key is not configured"
        }
    )
}

fn model_provider_for_capability(
    settings: &crate::db::BotSettings,
    capability: &str,
) -> ModelProvider {
    settings
        .model_routing
        .get(capability)
        .map(|routing| routing.model_provider)
        .unwrap_or_default()
}

fn model_provider_for_skill(settings: &crate::db::BotSettings, skill: &str) -> ModelProvider {
    if let Some(routing) = settings.model_routing.get(skill) {
        return routing.model_provider;
    }
    let capability = match skill {
        "text_to_image"
        | "image_to_image"
        | "text_to_video"
        | "image_to_video"
        | "video_to_video"
        | "text_to_audio"
        | "video_to_audio"
        | "text_to_speech"
        | "text_to_3d"
        | "image_to_3d"
        | "text_to_image_vector"
        | "image_to_image_vector"
        | "image_understanding"
        | "video_understanding" => {
            if matches!(
                skill,
                "text_to_3d" | "image_to_3d" | "text_to_image_vector" | "image_to_image_vector"
            ) {
                return ModelProvider::Fal;
            }
            let legacy = match skill {
                "text_to_image" | "image_to_image" => "image_generation",
                "text_to_video" | "image_to_video" | "video_to_video" => "video_generation",
                "text_to_audio" | "video_to_audio" => "music_generation",
                "text_to_speech" => "speech_generation",
                _ => skill,
            };
            return model_provider_for_capability(settings, legacy);
        }
        "transcription" => "transcription",
        "file" => "chat",
        "model_upgrade" => "model_upgrade",
        "youtube" => "video_understanding",
        "prompt_expansion" => "intent_planning",
        _ => "chat",
    };
    model_provider_for_capability(settings, capability)
}

fn model_allowed_fallback(config: &Config, capability: &str, id: &str) -> bool {
    match capability {
        "chat" | "model_upgrade" | "output_processing" | "error_processing" => {
            config.openrouter.models.iter().any(|model| model.id == id)
        }
        "intent_planning" => config.openrouter.planner.model == id,
        "intent_planning_fallback" => config.openrouter.planner.fallback_model == id,
        "image_understanding" => config
            .openrouter
            .understanding
            .image
            .models
            .iter()
            .any(|model| model.id == id),
        "video_understanding" => config
            .openrouter
            .understanding
            .video
            .models
            .iter()
            .any(|model| model.id == id),
        "image_generation" | "text_to_image" | "image_to_image" => config
            .openrouter
            .image
            .models
            .iter()
            .any(|model| model.id == id),
        "audio_generation" | "speech_generation" | "text_to_speech" => config
            .openrouter
            .audio
            .models
            .iter()
            .any(|model| model.id == id),
        "music_generation" | "text_to_audio" | "video_to_audio" => config
            .openrouter
            .music
            .models
            .iter()
            .any(|model| model.id == id),
        "transcription" => config
            .openrouter
            .transcription
            .models
            .iter()
            .any(|model| model.id == id),
        "video_generation" | "text_to_video" | "image_to_video" | "video_to_video" => config
            .openrouter
            .video
            .models
            .iter()
            .any(|model| model.id == id),
        _ => false,
    }
}
fn provider_options(selected: &str, configured: &BTreeMap<&str, bool>) -> String {
    ["openrouter", "brave", "exa", "serpapi"]
        .into_iter()
        .map(|p| {
            let available = configured.get(p).copied().unwrap_or(false);
            format!(
                "<option value=\"{p}\" {} {}>{p}{}</option>",
                if p == selected { "selected" } else { "" },
                if available { "" } else { "disabled" },
                if available {
                    ""
                } else {
                    " — unavailable (API key missing)"
                }
            )
        })
        .collect()
}

fn authorize(state: &AdminState, bot_id: &str, headers: &HeaderMap) -> Result<(u64, String)> {
    let bot = state.config.bot(bot_id).context("Unknown bot")?;
    let raw = headers
        .get("x-telegram-init-data")
        .and_then(|v| v.to_str().ok())
        .context("Missing Telegram authentication")?;
    let user = authenticate_init_data(bot, state.config.server.admin_init_data_ttl_seconds, raw)?;
    Ok((user, bot.id.clone()))
}

fn authenticate_init_data(
    bot: &crate::config::BotConfig,
    max_age_seconds: i64,
    raw: &str,
) -> Result<u64> {
    let mut pairs = BTreeMap::new();
    for (k, v) in url::form_urlencoded::parse(raw.as_bytes()) {
        pairs.insert(k.into_owned(), v.into_owned());
    }
    let hash = pairs
        .remove("hash")
        .context("Missing Telegram authentication hash")?;
    let check = pairs
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join("\n");
    let mut secret = Hmac::<Sha256>::new_from_slice(b"WebAppData").expect("HMAC key");
    secret.update(bot.token.as_bytes());
    let secret = secret.finalize().into_bytes();
    let mut signature = Hmac::<Sha256>::new_from_slice(&secret).expect("HMAC key");
    signature.update(check.as_bytes());
    signature
        .verify_slice(&hex::decode(hash).wrap_err("Invalid authentication hash")?)
        .map_err(|_| eyre::eyre!("Invalid Telegram authentication signature"))?;
    let auth_date = pairs
        .get("auth_date")
        .and_then(|v| v.parse::<i64>().ok())
        .context("Missing authentication date")?;
    let age = Utc::now().timestamp() - auth_date;
    if age < 0 || age > max_age_seconds {
        bail!("Telegram authentication has expired")
    }
    let user: Value = serde_json::from_str(pairs.get("user").context("Missing Telegram user")?)?;
    let id = user
        .get("id")
        .and_then(Value::as_u64)
        .context("Missing Telegram user ID")?;
    if !bot.admin_user_ids.contains(&id) {
        bail!("Administrator access required")
    }
    Ok(id)
}
fn esc(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
fn auth_error(error: eyre::Report) -> Response {
    message(StatusCode::UNAUTHORIZED, &error.to_string())
}
fn internal(error: eyre::Report) -> Response {
    tracing::error!(error=%format!("{error:#}"),"admin request failed");
    message(StatusCode::INTERNAL_SERVER_ERROR, "Internal server error")
}
fn message(status: StatusCode, text: &str) -> Response {
    (
        status,
        Html(format!("<section><h2>{}</h2></section>", esc(text))),
    )
        .into_response()
}

fn no_store(mut response: Response) -> Response {
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AccessConfig, BotConfig};

    fn signed_init_data(bot_token: &str, user_id: u64, auth_date: i64) -> String {
        let user = serde_json::json!({"id": user_id, "first_name": "Admin"}).to_string();
        let check = format!("auth_date={auth_date}\nuser={user}");
        let mut secret = Hmac::<Sha256>::new_from_slice(b"WebAppData").unwrap();
        secret.update(bot_token.as_bytes());
        let mut signature =
            Hmac::<Sha256>::new_from_slice(&secret.finalize().into_bytes()).unwrap();
        signature.update(check.as_bytes());
        let hash = hex::encode(signature.finalize().into_bytes());
        url::form_urlencoded::Serializer::new(String::new())
            .append_pair("auth_date", &auth_date.to_string())
            .append_pair("user", &user)
            .append_pair("hash", &hash)
            .finish()
    }

    #[test]
    fn telegram_init_data_is_verified_against_immutable_admins() {
        let bot = BotConfig {
            id: "test".into(),
            token: "123:test-token".into(),
            enabled: true,
            default_model: "chat".into(),
            admin_user_ids: vec![42],
            allowed_user_ids: vec![],
            allowed_chat_ids: vec![],
            access: AccessConfig::default(),
        };
        let now = Utc::now().timestamp();
        assert_eq!(
            authenticate_init_data(&bot, 900, &signed_init_data(&bot.token, 42, now)).unwrap(),
            42
        );
        assert!(authenticate_init_data(&bot, 900, &signed_init_data(&bot.token, 7, now)).is_err());
        let mut tampered = signed_init_data(&bot.token, 42, now);
        tampered.push('x');
        assert!(authenticate_init_data(&bot, 900, &tampered).is_err());
    }

    #[test]
    fn model_card_uses_picker_instead_of_a_large_select() {
        let configured = BTreeMap::from([("openrouter", true), ("aihub", false)]);
        let html = model_form(
            "General chat",
            "chat",
            "vendor/model",
            &ModelRouting {
                model_provider: ModelProvider::Openrouter,
                strategy: "cheapest".into(),
                provider: None,
            },
            &configured,
            true,
        );
        assert!(html.contains("class=model-picker"));
        assert!(html.contains("data-model=\"vendor/model\""));
        assert!(!html.contains("<select"));
    }

    #[test]
    fn browser_catalog_filter_includes_processing_capabilities() {
        let javascript = include_str!("../assets/admin.js");
        assert!(javascript.contains("'output_processing', 'error_processing'"));
        assert!(javascript.contains("output_processing: 'Text output processing'"));
        assert!(javascript.contains("error_processing: 'Error explanation'"));
    }

    #[test]
    fn model_endpoint_url_encodes_untrusted_path_segments() {
        let url =
            model_endpoints_url("https://openrouter.ai/api/v1", "vendor/a?b", "chat").unwrap();
        assert_eq!(
            url.as_str(),
            "https://openrouter.ai/api/v1/models/vendor/a%3Fb/endpoints"
        );
        let image = model_endpoints_url(
            "https://openrouter.ai/api/v1",
            "vendor/image",
            "image_generation",
        )
        .unwrap();
        assert_eq!(
            image.as_str(),
            "https://openrouter.ai/api/v1/images/models/vendor/image/endpoints"
        );
    }
}
