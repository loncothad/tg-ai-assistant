//! Telegram Mini App administration UI and authenticated HTMX endpoints.

use crate::{
    Result,
    catalog::ModelCatalogCache,
    config::Config,
    db::{ModelRouting, Store},
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
    pub client: reqwest::Client,
    catalog: ModelCatalogCache,
}

impl AdminState {
    /// Creates shared state for the authenticated Mini App endpoints.
    pub fn new(config: Arc<Config>, store: Store, client: reqwest::Client) -> Self {
        Self {
            config,
            store,
            client,
            catalog: ModelCatalogCache::default(),
        }
    }
}

pub fn router(state: AdminState) -> Router {
    Router::new()
        .route("/admin/{bot_id}", get(shell))
        .route("/admin/{bot_id}/", get(shell))
        .route("/admin/{bot_id}/panel", get(panel))
        .route("/admin/{bot_id}/models", get(models))
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
<dialog id="model-dialog"><div class="picker-layout"><div class="picker-list"><div class="picker-head"><div><h2>Choose a model</h2><div id="picker-subtitle" class="subtitle">Live OpenRouter catalog</div></div><button id="model-close" class="icon-button" type="button" aria-label="Close">×</button></div><div class="search-wrap"><svg viewBox="0 0 24 24" fill="none"><circle cx="11" cy="11" r="7" stroke="currentColor" stroke-width="2"/><path d="m16 16 4 4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg><input id="model-search" type="search" autocomplete="off" placeholder="Fuzzy search names, IDs, descriptions…"></div><div id="model-count" class="result-count"></div><div id="model-results" class="model-results"></div></div><div id="model-detail" class="picker-detail"></div></div></dialog>
<template id="model-settings"><label>Provider routing<select name="routing"><option value="auto">Auto · OpenRouter default</option><option value="cheapest">Cheapest provider</option><option value="throughput">Highest throughput</option><option value="latency">Lowest latency</option><option value="exacto">Exacto tool quality</option></select></label><label>Provider override<select name="provider"><option value="">Auto · any compatible provider</option></select></label><small class=provider-help>Loading the providers that currently serve this model…</small><button type="button" data-save-model id="model-save">Use this model</button></template>
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
    match catalog_for(&state, &bot).await {
        Ok(models) => {
            no_store(Json(serde_json::json!({ "models": models.as_ref() })).into_response())
        }
        Err(error) => internal(error),
    }
}

#[derive(Deserialize)]
struct ModelProvidersQuery {
    model: String,
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
    let catalog = match catalog_for(&state, &bot).await {
        Ok(catalog) => catalog,
        Err(error) => return internal(error),
    };
    if !catalog.iter().any(|model| model.id == query.model) {
        return message(StatusCode::BAD_REQUEST, "Unknown model");
    }
    let endpoint_url = match model_endpoints_url(&state.config.openrouter.base_url, &query.model) {
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
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|endpoint| {
            Some(serde_json::json!({
                "tag": endpoint.get("tag")?.as_str()?,
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

fn model_endpoints_url(base_url: &str, model: &str) -> Result<url::Url> {
    let mut url = url::Url::parse(base_url).wrap_err("Invalid OpenRouter base URL")?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| eyre::eyre!("OpenRouter base URL cannot contain path segments"))?;
        segments.pop_if_empty().push("models");
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

async fn catalog_for(
    state: &AdminState,
    bot: &str,
) -> Result<Arc<Vec<crate::catalog::CatalogModel>>> {
    let key = openrouter_key(state, bot).await?;
    state
        .catalog
        .get(&state.client, &state.config.openrouter.base_url, bot, &key)
        .await
}

#[derive(Deserialize)]
struct ModelForm {
    capability: String,
    model: String,
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
    let allowed = catalog_for(&state, &bot)
        .await
        .map(|models| {
            models
                .iter()
                .any(|model| model.id == form.model && model.supports(&form.capability))
        })
        .unwrap_or_else(|_| model_allowed_fallback(&state.config, &form.capability, &form.model));
    if !allowed {
        return message(
            StatusCode::BAD_REQUEST,
            "Unknown model or model does not support this capability",
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
        let required_provider = if form.capability == "search" {
            match state.store.settings(&bot).await {
                Ok(settings) => settings
                    .search_provider
                    .unwrap_or_else(|| state.config.search.default_provider.as_str().to_owned()),
                Err(error) => return internal(error),
            }
        } else {
            "openrouter".to_owned()
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
        match skill.id.as_str() {
            "search" => settings.capabilities.search = skill.enabled,
            "web_fetch" => settings.capabilities.web_fetch = skill.enabled,
            "image" => settings.capabilities.image = skill.enabled,
            "audio" => settings.capabilities.audio = skill.enabled,
            "video" => settings.capabilities.video = skill.enabled,
            "media" => settings.capabilities.media = skill.enabled,
            "transcription" => settings.capabilities.transcription = skill.enabled,
            "file" => settings.capabilities.file = skill.enabled,
            _ => bail!("Unknown built-in skill: {}", skill.id),
        }
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
    let enabled = |id: &str| match id {
        "search" => settings.capabilities.search,
        "web_fetch" => settings.capabilities.web_fetch,
        "image" => settings.capabilities.image,
        "audio" => settings.capabilities.audio,
        "video" => settings.capabilities.video,
        "media" => settings.capabilities.media,
        "transcription" => settings.capabilities.transcription,
        "file" => settings.capabilities.file,
        _ => false,
    };
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
    for p in ["openrouter", "brave", "exa", "serpapi"] {
        configured.insert(p, state.store.credential_configured(bot, p).await?);
    }
    let openrouter_ready = configured.get("openrouter").copied().unwrap_or(false);
    let route = |capability: &str| {
        settings
            .model_routing
            .get(capability)
            .cloned()
            .unwrap_or_default()
    };
    let model_forms = [
        model_form(
            "General chat",
            "chat",
            &settings.selected_model,
            &route("chat"),
            openrouter_ready,
        ),
        model_form(
            "Image understanding",
            "image_understanding",
            &settings.selected_image_understanding_model,
            &route("image_understanding"),
            openrouter_ready,
        ),
        model_form(
            "Video understanding",
            "video_understanding",
            &settings.selected_video_understanding_model,
            &route("video_understanding"),
            openrouter_ready,
        ),
        model_form(
            "Image generation",
            "image_generation",
            &settings.selected_image_generation_model,
            &route("image_generation"),
            openrouter_ready,
        ),
        model_form(
            "Speech generation",
            "audio_generation",
            &settings.selected_audio_generation_model,
            &route("audio_generation"),
            openrouter_ready,
        ),
        model_form(
            "Transcription",
            "transcription",
            &settings.selected_transcription_model,
            &route("transcription"),
            openrouter_ready,
        ),
        model_form(
            "Video generation",
            "video_generation",
            &settings.selected_video_generation_model,
            &route("video_generation"),
            openrouter_ready,
        ),
    ]
    .join("");
    let provider = settings
        .search_provider
        .as_deref()
        .unwrap_or(state.config.search.default_provider.as_str());
    let search_available = configured.get(provider).copied().unwrap_or(false);
    let enabled = |id: &str| match id {
        "search" => settings.capabilities.search,
        "web_fetch" => settings.capabilities.web_fetch,
        "image" => settings.capabilities.image,
        "audio" => settings.capabilities.audio,
        "video" => settings.capabilities.video,
        "media" => settings.capabilities.media,
        "transcription" => settings.capabilities.transcription,
        "file" => settings.capabilities.file,
        _ => false,
    };
    let caps = crate::defaults::BUILTIN_SKILLS
        .iter()
        .map(|skill| {
            let on = enabled(skill.id);
            let available = if skill.id == "search" {
                search_available
            } else {
                openrouter_ready
            };
            let availability = if available {
                "Available"
            } else if skill.id == "search" {
                "Unavailable: the selected search provider has no configured API key"
            } else {
                "Unavailable: OPENROUTER_API_KEY was not provided and no OpenRouter key has been saved"
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
        <section id=\"models\"><div class=\"section-head\"><div><h2>Models by capability</h2><p>Every compatible model in OpenRouter's live catalog is searchable.</p></div><span class=\"badge\">Live catalog</span></div><div class=\"grid model-grid\">{model_forms}</div><form hx-post=\"search\" hx-target=\"#panel\" hx-swap=\"innerHTML transition:true\"><label>Web search provider<select name=provider>{}</select></label><button>Save search provider</button></form></section>
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

fn model_form(
    label: &str,
    capability: &str,
    selected: &str,
    routing: &ModelRouting,
    available: bool,
) -> String {
    format!(
        "<article class=\"card model-card\" data-capability=\"{}\" data-model=\"{}\"><h3>{}</h3><div class=model-name>{}</div><div class=model-id>{}</div><p class=model-summary>Loading metadata from OpenRouter…</p><div class=\"chips model-chips\"></div><div class=route-line><span>Routing: {}</span><span>Provider: {}</span></div><button type=button class=model-picker data-capability=\"{}\" data-model=\"{}\" data-routing=\"{}\" data-provider=\"{}\" {disabled}>{button}</button><small class=\"{status_class}\">{status}</small></article>",
        esc(capability),
        esc(selected),
        esc(label),
        esc(selected),
        esc(selected),
        esc(&routing.strategy),
        esc(routing.provider.as_deref().unwrap_or("auto")),
        esc(capability),
        esc(selected),
        esc(&routing.strategy),
        esc(routing.provider.as_deref().unwrap_or("")),
        disabled = if available { "" } else { "disabled" },
        button = if available {
            "Choose model"
        } else {
            "OpenRouter key required"
        },
        status_class = if available { "status-ok" } else { "status-off" },
        status = if available {
            "Ready"
        } else {
            "Unavailable: no OpenRouter API key is configured"
        }
    )
}

fn model_allowed_fallback(config: &Config, capability: &str, id: &str) -> bool {
    match capability {
        "chat" => config.openrouter.models.iter().any(|model| model.id == id),
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
        "image_generation" => config
            .openrouter
            .image
            .models
            .iter()
            .any(|model| model.id == id),
        "audio_generation" => config
            .openrouter
            .audio
            .models
            .iter()
            .any(|model| model.id == id),
        "transcription" => config
            .openrouter
            .transcription
            .models
            .iter()
            .any(|model| model.id == id),
        "video_generation" => config
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
        let html = model_form(
            "General chat",
            "chat",
            "vendor/model",
            &ModelRouting {
                strategy: "cheapest".into(),
                provider: None,
            },
            true,
        );
        assert!(html.contains("class=model-picker"));
        assert!(html.contains("data-model=\"vendor/model\""));
        assert!(!html.contains("<select"));
    }

    #[test]
    fn model_endpoint_url_encodes_untrusted_path_segments() {
        let url = model_endpoints_url("https://openrouter.ai/api/v1", "vendor/a?b").unwrap();
        assert_eq!(
            url.as_str(),
            "https://openrouter.ai/api/v1/models/vendor/a%3Fb/endpoints"
        );
    }
}
