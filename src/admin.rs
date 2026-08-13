//! Telegram Mini App administration UI and authenticated HTMX endpoints.

use crate::{
    Result,
    config::Config,
    db::{ModelRouting, Store},
};
use axum::{
    Form, Router,
    extract::{Multipart, Path, State},
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
}

pub fn router(state: AdminState) -> Router {
    Router::new()
        .route("/admin/{bot_id}", get(shell))
        .route("/admin/{bot_id}/panel", get(panel))
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
        r#"<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><base href="/admin/{bot_id}/"><title>Teleforge Admin</title><script src="https://telegram.org/js/telegram-web-app.js?59"></script><script src="https://cdn.jsdelivr.net/npm/htmx.org@2.0.9/dist/htmx.min.js" integrity="sha384-ESlCao+z/oasnu2Uc/5K1LQTI7YCF2KKO4xakCPQCFuiHhCh8Oa/R5NwHY6guZ3m" crossorigin="anonymous"></script><style nonce="{nonce}">:root{{color-scheme:light dark}}body{{font:15px system-ui;margin:0;padding:16px;background:var(--tg-theme-bg-color,#fff);color:var(--tg-theme-text-color,#111)}}main{{max-width:760px;margin:auto}}section{{border:1px solid color-mix(in srgb,currentColor 18%,transparent);border-radius:12px;padding:14px;margin:12px 0}}label{{display:block;margin:8px 0}}input,select,textarea,button{{box-sizing:border-box;width:100%;padding:10px;margin:4px 0;border-radius:8px;border:1px solid #888;background:var(--tg-theme-secondary-bg-color,#eee);color:inherit}}button{{cursor:pointer;background:var(--tg-theme-button-color,#2481cc);color:var(--tg-theme-button-text-color,#fff)}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:8px}}small{{opacity:.75}}.ok{{color:#2a5}}.off{{opacity:.6}}</style></head><body><main><h1>Teleforge Admin</h1><div id="panel">Authenticating…</div></main><script nonce="{nonce}">const app=window.Telegram.WebApp;app.ready();app.expand();document.addEventListener('htmx:configRequest',e=>{{e.detail.headers['X-Telegram-Init-Data']=app.initData;}});htmx.ajax('GET',location.pathname+'/panel',{{target:'#panel',swap:'innerHTML'}});</script></body></html>"#
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
    let user = match authorize(&state, &bot, &headers) {
        Ok(v) => v,
        Err(e) => return auth_error(e),
    };
    if !model_allowed(&state.config, &form.capability, &form.model) {
        return message(StatusCode::BAD_REQUEST, "Unknown model");
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
        !value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
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
    let user = match authorize(&state, &bot, &headers) {
        Ok(v) => v,
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
    let actor = match authorize(&state, &bot, &headers) {
        Ok(value) => value,
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
    let actor = match authorize(&state, &bot, &headers) {
        Ok(value) => value,
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
    let actor = match authorize(&state, &bot, &headers) {
        Ok(value) => value,
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
    if let Err(e) = authorize(&state, &bot, &headers) {
        return auth_error(e);
    }
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
    let actor = match authorize(&state, &bot, &headers) {
        Ok(v) => v,
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
    if let Err(e) = authorize(state, bot, headers) {
        return auth_error(e);
    }
    render(state, bot).await
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
    let openrouter_key = state.store.credential(bot, "openrouter").await?;
    let providers = match openrouter_key.as_deref() {
        Some(key) => list_openrouter_providers(state, key)
            .await
            .unwrap_or_default(),
        None => Vec::new(),
    };
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
            state.config.openrouter.models.iter().map(|model| {
                (
                    model.id.as_str(),
                    model.label.as_deref().unwrap_or(&model.id),
                )
            }),
            &route("chat"),
            &providers,
            openrouter_ready,
        ),
        model_form(
            "Image understanding",
            "image_understanding",
            &settings.selected_image_understanding_model,
            state
                .config
                .openrouter
                .understanding
                .image
                .models
                .iter()
                .map(|model| {
                    (
                        model.id.as_str(),
                        model.label.as_deref().unwrap_or(&model.id),
                    )
                }),
            &route("image_understanding"),
            &providers,
            openrouter_ready,
        ),
        model_form(
            "Video understanding",
            "video_understanding",
            &settings.selected_video_understanding_model,
            state
                .config
                .openrouter
                .understanding
                .video
                .models
                .iter()
                .map(|model| {
                    (
                        model.id.as_str(),
                        model.label.as_deref().unwrap_or(&model.id),
                    )
                }),
            &route("video_understanding"),
            &providers,
            openrouter_ready,
        ),
        model_form(
            "Image generation",
            "image_generation",
            &settings.selected_image_generation_model,
            state.config.openrouter.image.models.iter().map(|model| {
                (
                    model.id.as_str(),
                    model.label.as_deref().unwrap_or(&model.id),
                )
            }),
            &route("image_generation"),
            &providers,
            openrouter_ready,
        ),
        model_form(
            "Speech generation",
            "audio_generation",
            &settings.selected_audio_generation_model,
            state.config.openrouter.audio.models.iter().map(|model| {
                (
                    model.id.as_str(),
                    model.label.as_deref().unwrap_or(&model.id),
                )
            }),
            &route("audio_generation"),
            &providers,
            openrouter_ready,
        ),
        model_form(
            "Transcription",
            "transcription",
            &settings.selected_transcription_model,
            state
                .config
                .openrouter
                .transcription
                .models
                .iter()
                .map(|model| {
                    (
                        model.id.as_str(),
                        model.label.as_deref().unwrap_or(&model.id),
                    )
                }),
            &route("transcription"),
            &providers,
            openrouter_ready,
        ),
        model_form(
            "Video generation",
            "video_generation",
            &settings.selected_video_generation_model,
            state.config.openrouter.video.models.iter().map(|model| {
                (
                    model.id.as_str(),
                    model.label.as_deref().unwrap_or(&model.id),
                )
            }),
            &route("video_generation"),
            &providers,
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
        "<section><h2>Models by capability</h2><div class=grid>{model_forms}</div><form hx-post=\"search\" hx-target=\"#panel\"><label>Web search provider</label><select name=provider>{}</select><button>Save search provider</button></form></section><section><h2>Built-in skills</h2><p>Enabling a skill exposes its instructions and callable API where applicable.</p><div class=grid>{caps}</div></section><section><h2>API credentials</h2>{creds}</section><section><h2>System prompt</h2>{}</section><section><h2>Skills import/export</h2>{}<button hx-get=\"skills/export\" hx-target=\"#skill-export\">Export current skill bundle</button><div id=skill-export></div></section><section><h2>User access</h2><form hx-post=\"access\" hx-target=\"#panel\"><input type=number name=user_id min=1 required><div class=grid><button name=allowed value=true>Allow</button><button name=allowed value=false>Deny</button></div></form></section>",
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

fn model_form<'a>(
    label: &str,
    capability: &str,
    selected: &str,
    models: impl Iterator<Item = (&'a str, &'a str)>,
    routing: &ModelRouting,
    providers: &[(String, String)],
    available: bool,
) -> String {
    let options = models
        .map(|(id, name)| {
            format!(
                "<option value=\"{}\" {}>{}</option>",
                esc(id),
                if id == selected { "selected" } else { "" },
                esc(name)
            )
        })
        .collect::<String>();
    let mut strategy_choices = vec![
        ("auto", "Auto (OpenRouter default/Auto Exacto)"),
        ("cheapest", "Cheapest provider"),
        ("throughput", "Highest throughput"),
        ("latency", "Lowest latency"),
    ];
    if matches!(
        capability,
        "chat" | "image_understanding" | "video_understanding"
    ) {
        strategy_choices.push(("exacto", "Exacto tool quality"));
    }
    let strategies = strategy_choices
        .into_iter()
        .map(|(value, name)| {
            format!(
                "<option value=\"{value}\" {}>{name}</option>",
                if routing.strategy == value {
                    "selected"
                } else {
                    ""
                }
            )
        })
        .collect::<String>();
    let mut provider_choices = providers.to_vec();
    if let Some(selected_provider) = routing.provider.as_deref()
        && !provider_choices
            .iter()
            .any(|(slug, _)| slug == selected_provider)
    {
        provider_choices.push((selected_provider.to_owned(), selected_provider.to_owned()));
    }
    let provider_options = std::iter::once(("auto", "Auto (any provider)"))
        .chain(
            provider_choices
                .iter()
                .map(|(slug, name)| (slug.as_str(), name.as_str())),
        )
        .map(|(slug, name)| {
            format!(
                "<option value=\"{}\" {}>{}</option>",
                esc(slug),
                if routing.provider.as_deref() == Some(slug) {
                    "selected"
                } else {
                    ""
                },
                esc(name)
            )
        })
        .collect::<String>();
    format!(
        "<form hx-post=\"model\" hx-target=\"#panel\"><input type=hidden name=capability value=\"{}\"><label>{}</label><select name=model {disabled}>{options}</select><label>Provider routing</label><select name=routing {disabled}>{strategies}</select><label>Provider</label><select name=provider {disabled}>{provider_options}</select><button {disabled}>Save</button><small>{status}</small></form>",
        esc(capability),
        esc(label),
        disabled = if available { "" } else { "disabled" },
        status = if available {
            "Available through the configured OpenRouter key. OpenRouter rejects providers that do not serve the selected model."
        } else {
            "Unavailable: OPENROUTER_API_KEY was not provided and no key has been saved below"
        }
    )
}

async fn list_openrouter_providers(
    state: &AdminState,
    api_key: &str,
) -> Result<Vec<(String, String)>> {
    let response = state
        .client
        .get(format!(
            "{}/providers",
            state.config.openrouter.base_url.trim_end_matches('/')
        ))
        .bearer_auth(api_key)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .wrap_err("Failed to list OpenRouter providers")?;
    if !response.status().is_success() {
        bail!("OpenRouter provider listing failed");
    }
    let value: Value = response
        .json()
        .await
        .wrap_err("Invalid OpenRouter provider list")?;
    let mut providers = value
        .get("data")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|provider| {
            Some((
                provider.get("slug")?.as_str()?.to_owned(),
                provider.get("name")?.as_str()?.to_owned(),
            ))
        })
        .collect::<Vec<_>>();
    providers.sort_by(|left, right| left.1.cmp(&right.1));
    Ok(providers)
}

fn model_allowed(config: &Config, capability: &str, id: &str) -> bool {
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

fn authorize(state: &AdminState, bot_id: &str, headers: &HeaderMap) -> Result<u64> {
    let bot = state.config.bot(bot_id).context("Unknown bot")?;
    let raw = headers
        .get("x-telegram-init-data")
        .and_then(|v| v.to_str().ok())
        .context("Missing Telegram authentication")?;
    authenticate_init_data(bot, state.config.server.admin_init_data_ttl_seconds, raw)
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
}
