//! Strict YAML configuration and environment-variable expansion.
//!
//! Runtime-editable values live in redb; this module contains deployment settings,
//! curated models, bootstrap secrets, and immutable Telegram administrator IDs.

use std::{collections::HashSet, fs, path::Path, time::Duration};

use eyre::{Context, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Result;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub telegram: TelegramConfig,
    pub openrouter: OpenRouterConfig,
    pub search: SearchConfig,
    pub bots: Vec<BotConfig>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TelegramConfig {
    /// Comma-separated Bot API tokens, assigned to enabled bot entries in order.
    pub tokens: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    #[serde(default = "default_listen")]
    pub listen: String,
    pub public_url: String,
    #[serde(default = "default_request_timeout")]
    pub request_timeout_seconds: u64,
    #[serde(default = "default_concurrency")]
    pub max_concurrent_requests_per_bot: usize,
    #[serde(default = "default_auth_ttl")]
    pub admin_init_data_ttl_seconds: i64,
    #[serde(default = "default_media_bytes")]
    pub max_input_media_bytes: usize,
    #[serde(default)]
    pub json_logs: bool,
}
impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            listen: default_listen(),
            public_url: String::new(),
            request_timeout_seconds: default_request_timeout(),
            max_concurrent_requests_per_bot: default_concurrency(),
            admin_init_data_ttl_seconds: default_auth_ttl(),
            max_input_media_bytes: default_media_bytes(),
            json_logs: false,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DatabaseConfig {
    pub path: String,
    /// Base64-encoded 32-byte key used to encrypt provider credentials at rest.
    pub encryption_key: String,
    #[serde(default = "default_history_limit")]
    pub history_limit: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OpenRouterConfig {
    #[serde(default)]
    pub bootstrap_api_key: String,
    #[serde(default = "default_openrouter_url")]
    pub base_url: String,
    #[serde(default)]
    pub site_url: Option<String>,
    #[serde(default = "default_app_name")]
    pub app_name: String,
    pub models: Vec<ModelConfig>,
    pub understanding: UnderstandingConfig,
    #[serde(default)]
    pub web_search: OpenRouterWebSearchConfig,
    #[serde(default)]
    pub web_fetch: OpenRouterWebFetchConfig,
    #[serde(default)]
    pub defaults: OpenRouterOptions,
    pub image: MediaModelConfig,
    pub audio: AudioModelConfig,
    pub transcription: TranscriptionModelConfig,
    pub video: VideoModelConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OpenRouterWebSearchConfig {
    #[serde(default = "default_openrouter_search_engine")]
    pub engine: String,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default = "default_openrouter_search_results")]
    pub max_results: u8,
    #[serde(default)]
    pub max_uses: Option<u16>,
    #[serde(default = "default_openrouter_total_results")]
    pub max_total_results: u16,
    #[serde(default = "default_search_context_size")]
    pub search_context_size: String,
    #[serde(default)]
    pub max_characters: Option<u32>,
    #[serde(default)]
    pub user_location: Option<Value>,
    #[serde(default)]
    pub allowed_domains: Vec<String>,
    #[serde(default)]
    pub excluded_domains: Vec<String>,
    #[serde(default)]
    pub extra: serde_json::Map<String, Value>,
}

impl Default for OpenRouterWebSearchConfig {
    fn default() -> Self {
        Self {
            engine: default_openrouter_search_engine(),
            mode: None,
            max_results: default_openrouter_search_results(),
            max_uses: Some(5),
            max_total_results: default_openrouter_total_results(),
            search_context_size: default_search_context_size(),
            max_characters: None,
            user_location: None,
            allowed_domains: Vec::new(),
            excluded_domains: Vec::new(),
            extra: serde_json::Map::new(),
        }
    }
}

/// Controls OpenRouter's model-callable URL/PDF content retrieval server tool.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OpenRouterWebFetchConfig {
    #[serde(default = "default_openrouter_fetch_engine")]
    pub engine: String,
    #[serde(default)]
    pub max_uses: Option<u16>,
    #[serde(default)]
    pub max_content_tokens: Option<u32>,
    #[serde(default)]
    pub allowed_domains: Vec<String>,
    #[serde(default)]
    pub blocked_domains: Vec<String>,
    #[serde(default)]
    pub extra: serde_json::Map<String, Value>,
}

impl Default for OpenRouterWebFetchConfig {
    fn default() -> Self {
        Self {
            engine: default_openrouter_fetch_engine(),
            max_uses: Some(5),
            max_content_tokens: Some(50_000),
            allowed_domains: Vec::new(),
            blocked_domains: Vec::new(),
            extra: serde_json::Map::new(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub id: String,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub options: OpenRouterOptions,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelChoice {
    pub id: String,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UnderstandingConfig {
    pub image: ChatCapabilityConfig,
    pub video: ChatCapabilityConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCapabilityConfig {
    pub default_model: String,
    pub models: Vec<ModelConfig>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OpenRouterOptions {
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default)]
    pub provider: Option<Value>,
    #[serde(default)]
    pub plugins: Vec<Value>,
    #[serde(default)]
    pub transforms: Vec<String>,
    #[serde(default)]
    pub reasoning: Option<Value>,
    #[serde(default)]
    pub response_format: Option<Value>,
    #[serde(default)]
    pub tools: Vec<Value>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub cache_control: Option<Value>,
    #[serde(default)]
    pub image_config: Option<Value>,
    #[serde(default)]
    pub logit_bias: Option<Value>,
    #[serde(default)]
    pub max_completion_tokens: Option<u64>,
    #[serde(default)]
    pub max_tool_calls: Option<u8>,
    #[serde(default)]
    pub metadata: Option<Value>,
    #[serde(default)]
    pub min_p: Option<f64>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub repetition_penalty: Option<f64>,
    #[serde(default)]
    pub route: Option<Value>,
    #[serde(default)]
    pub service_tier: Option<String>,
    #[serde(default)]
    pub stop_server_tools_when: Option<Value>,
    #[serde(default)]
    pub top_a: Option<f64>,
    #[serde(default)]
    pub trace: Option<Value>,
    #[serde(default)]
    pub modalities: Vec<String>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<u64>,
    #[serde(default)]
    pub max_tokens: Option<u64>,
    #[serde(default)]
    pub frequency_penalty: Option<f64>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub stop: Option<Value>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<u8>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MediaModelConfig {
    pub model: String,
    pub models: Vec<ModelChoice>,
    #[serde(default = "default_image_size")]
    pub size: String,
    #[serde(default)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AudioModelConfig {
    pub model: String,
    pub models: Vec<ModelChoice>,
    #[serde(default = "default_voice")]
    pub voice: String,
    #[serde(default = "default_audio_format")]
    pub response_format: String,
    #[serde(default = "default_speed")]
    pub speed: f64,
    #[serde(default)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TranscriptionModelConfig {
    pub model: String,
    pub models: Vec<ModelChoice>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VideoModelConfig {
    pub model: String,
    pub models: Vec<ModelChoice>,
    #[serde(default = "default_video_duration")]
    pub duration: u64,
    #[serde(default = "default_aspect_ratio")]
    pub aspect_ratio: String,
    #[serde(default = "default_video_resolution")]
    pub resolution: String,
    #[serde(default = "default_video_poll")]
    pub poll_interval_seconds: u64,
    #[serde(default = "default_video_timeout")]
    pub timeout_seconds: u64,
    #[serde(default)]
    pub generate_audio: bool,
    #[serde(default)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchConfig {
    #[serde(default)]
    pub default_provider: SearchProvider,
    #[serde(default = "default_search_results")]
    pub max_results: usize,
    pub brave: ProviderConfig,
    pub exa: ProviderConfig,
    pub serpapi: ProviderConfig,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SearchProvider {
    Openrouter,
    #[default]
    Brave,
    Exa,
    Serpapi,
}
impl SearchProvider {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Openrouter => "openrouter",
            Self::Brave => "brave",
            Self::Exa => "exa",
            Self::Serpapi => "serpapi",
        }
    }
}
impl std::str::FromStr for SearchProvider {
    type Err = eyre::Report;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "openrouter" => Ok(Self::Openrouter),
            "brave" => Ok(Self::Brave),
            "exa" => Ok(Self::Exa),
            "serpapi" | "google" => Ok(Self::Serpapi),
            _ => bail!("Unknown search provider: {value}"),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderConfig {
    #[serde(default)]
    pub bootstrap_api_key: String,
    pub base_url: String,
    #[serde(default)]
    pub options: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BotConfig {
    pub id: String,
    #[serde(skip)]
    pub token: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    pub default_model: String,
    pub admin_user_ids: Vec<u64>,
    #[serde(default)]
    pub allowed_user_ids: Vec<u64>,
    #[serde(default)]
    pub allowed_chat_ids: Vec<i64>,
    #[serde(default)]
    pub access: AccessConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AccessConfig {
    #[serde(default = "default_true")]
    pub private_messages: bool,
    #[serde(default = "default_true")]
    pub group_chats: bool,
    #[serde(default = "default_true")]
    pub guest_messages: bool,
    #[serde(default = "default_true")]
    pub require_mention_in_groups: bool,
    #[serde(default)]
    pub allow_everyone: bool,
}
impl Default for AccessConfig {
    fn default() -> Self {
        Self {
            private_messages: true,
            group_chats: true,
            guest_messages: true,
            require_mention_in_groups: true,
            allow_everyone: false,
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path)
            .wrap_err_with(|| format!("Failed to read {}", path.display()))?;
        let expanded = expand_env(&raw)?;
        let mut config: Self = serde_yaml::from_str(&expanded)
            .wrap_err_with(|| format!("Invalid configuration in {}", path.display()))?;
        config.assign_telegram_tokens()?;
        config.validate()?;
        Ok(config)
    }
    fn assign_telegram_tokens(&mut self) -> Result<()> {
        let tokens = self
            .telegram
            .tokens
            .split(',')
            .map(str::trim)
            .filter(|token| !token.is_empty())
            .collect::<Vec<_>>();
        let enabled = self
            .bots
            .iter_mut()
            .filter(|bot| bot.enabled)
            .collect::<Vec<_>>();
        if tokens.len() != enabled.len() {
            bail!(
                "Telegram token count ({}) must match enabled bot count ({})",
                tokens.len(),
                enabled.len()
            );
        }
        for (bot, token) in enabled.into_iter().zip(tokens) {
            bot.token = token.to_owned();
        }
        Ok(())
    }
    pub fn validate(&self) -> Result<()> {
        if self.bots.is_empty() {
            bail!("At least one bot must be configured");
        }
        if !self.server.public_url.starts_with("https://") {
            bail!("Server public_url must use HTTPS for Telegram Web Apps");
        }
        if self.database.encryption_key.trim().is_empty() {
            bail!("Database encryption_key cannot be empty");
        }
        if self.openrouter.models.is_empty() {
            bail!("OpenRouter models cannot be empty");
        }
        validate_openrouter_options("defaults", &self.openrouter.defaults)?;
        for model in self
            .openrouter
            .models
            .iter()
            .chain(self.openrouter.understanding.image.models.iter())
            .chain(self.openrouter.understanding.video.models.iter())
        {
            validate_openrouter_options(&model.id, &model.options)?;
        }
        if !matches!(
            self.openrouter.web_search.engine.as_str(),
            "auto" | "native" | "exa" | "firecrawl" | "parallel" | "perplexity"
        ) {
            bail!("OpenRouter web-search engine is invalid");
        }
        if !(1..=25).contains(&self.openrouter.web_search.max_results)
            || self.openrouter.web_search.max_total_results == 0
            || (self.openrouter.web_search.engine == "perplexity"
                && self.openrouter.web_search.max_results > 20)
        {
            bail!("OpenRouter web-search result limits are invalid");
        }
        if self.openrouter.web_search.max_uses == Some(0)
            || self
                .openrouter
                .web_search
                .max_characters
                .is_some_and(|value| !(1..=100_000).contains(&value))
        {
            bail!("OpenRouter web-search usage or character limits are invalid");
        }
        if self
            .openrouter
            .web_search
            .mode
            .as_deref()
            .is_some_and(|mode| {
                !matches!(
                    mode,
                    "instant"
                        | "fast"
                        | "auto"
                        | "deep-lite"
                        | "deep"
                        | "deep-reasoning"
                        | "turbo"
                        | "basic"
                        | "advanced"
                )
            })
        {
            bail!("OpenRouter web-search mode is invalid");
        }
        if !matches!(
            self.openrouter.web_search.search_context_size.as_str(),
            "low" | "medium" | "high"
        ) {
            bail!("OpenRouter web-search context size is invalid");
        }
        if !matches!(
            self.openrouter.web_fetch.engine.as_str(),
            "auto" | "native" | "exa" | "openrouter" | "firecrawl" | "parallel"
        ) {
            bail!("OpenRouter web-fetch engine is invalid");
        }
        if self.openrouter.web_fetch.max_uses == Some(0)
            || self.openrouter.web_fetch.max_content_tokens == Some(0)
        {
            bail!("OpenRouter web-fetch limits must be greater than zero");
        }
        validate_chat_capability("image understanding", &self.openrouter.understanding.image)?;
        validate_chat_capability("video understanding", &self.openrouter.understanding.video)?;
        validate_model_choices(
            "image generation",
            &self.openrouter.image.model,
            &self.openrouter.image.models,
        )?;
        validate_model_choices(
            "audio generation",
            &self.openrouter.audio.model,
            &self.openrouter.audio.models,
        )?;
        validate_model_choices(
            "transcription",
            &self.openrouter.transcription.model,
            &self.openrouter.transcription.models,
        )?;
        validate_model_choices(
            "video generation",
            &self.openrouter.video.model,
            &self.openrouter.video.models,
        )?;
        if [
            self.openrouter.image.model.as_str(),
            self.openrouter.audio.model.as_str(),
            self.openrouter.transcription.model.as_str(),
            self.openrouter.video.model.as_str(),
        ]
        .iter()
        .any(|v| v.is_empty())
        {
            bail!("OpenRouter image, audio, transcription, and video models must be configured");
        }
        let mut model_ids = HashSet::new();
        for model in &self.openrouter.models {
            if model.id.is_empty() || !model_ids.insert(model.id.as_str()) {
                bail!(
                    "OpenRouter model IDs must be non-empty and unique: {}",
                    model.id
                );
            }
        }
        let mut bot_ids = HashSet::new();
        let mut tokens = HashSet::new();
        for bot in &self.bots {
            if !bot_ids.insert(bot.id.as_str()) {
                bail!("Duplicate bot ID: {}", bot.id);
            }
            if bot.enabled && !tokens.insert(bot.token.as_str()) {
                bail!("The same Telegram token is configured more than once");
            }
            if !model_ids.contains(bot.default_model.as_str()) {
                bail!(
                    "Bot {} references unknown model {}",
                    bot.id,
                    bot.default_model
                );
            }
            if bot.admin_user_ids.is_empty() {
                bail!("Bot {} requires an administrator", bot.id);
            }
        }
        if self.server.max_concurrent_requests_per_bot == 0 || self.search.max_results == 0 {
            bail!("Concurrency and search result limits must be greater than zero");
        }
        if self.server.max_input_media_bytes == 0 {
            bail!("Server max_input_media_bytes must be greater than zero");
        }
        Ok(())
    }
    pub fn model(&self, id: &str) -> Option<&ModelConfig> {
        self.openrouter.models.iter().find(|model| model.id == id)
    }
    pub fn understanding_model(&self, capability: &str, id: &str) -> Option<&ModelConfig> {
        let models = match capability {
            "image_understanding" => &self.openrouter.understanding.image.models,
            "video_understanding" => &self.openrouter.understanding.video.models,
            _ => return None,
        };
        models.iter().find(|model| model.id == id)
    }
    pub fn bot(&self, id: &str) -> Option<&BotConfig> {
        self.bots.iter().find(|bot| bot.enabled && bot.id == id)
    }
    pub fn timeout(&self) -> Duration {
        Duration::from_secs(self.server.request_timeout_seconds)
    }
}

fn validate_openrouter_options(name: &str, options: &OpenRouterOptions) -> Result<()> {
    if options
        .max_tool_calls
        .is_some_and(|value| !(1..=30).contains(&value))
    {
        bail!("OpenRouter {name} max_tool_calls must be between 1 and 30");
    }
    Ok(())
}

fn validate_chat_capability(name: &str, config: &ChatCapabilityConfig) -> Result<()> {
    if config.models.is_empty()
        || !config
            .models
            .iter()
            .any(|model| model.id == config.default_model)
    {
        bail!("OpenRouter {name} models must include the default model");
    }
    let mut ids = HashSet::new();
    if config
        .models
        .iter()
        .any(|model| model.id.is_empty() || !ids.insert(model.id.as_str()))
    {
        bail!("OpenRouter {name} model IDs must be non-empty and unique");
    }
    Ok(())
}

fn validate_model_choices(name: &str, default: &str, models: &[ModelChoice]) -> Result<()> {
    if models.is_empty() || !models.iter().any(|model| model.id == default) {
        bail!("OpenRouter {name} models must include the default model");
    }
    let mut ids = HashSet::new();
    if models
        .iter()
        .any(|model| model.id.is_empty() || !ids.insert(model.id.as_str()))
    {
        bail!("OpenRouter {name} model IDs must be non-empty and unique");
    }
    Ok(())
}

fn expand_env(input: &str) -> Result<String> {
    let mut out = String::new();
    let mut rest = input;
    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let tail = &rest[start + 2..];
        let Some(end) = tail.find('}') else {
            bail!("Unclosed environment expression")
        };
        let expression = &tail[..end];
        let (name, default) = expression
            .split_once(":-")
            .map_or((expression, None), |(n, d)| (n, Some(d)));
        if name.is_empty() || !name.chars().all(|c| c == '_' || c.is_ascii_alphanumeric()) {
            bail!("Invalid environment variable name: {name}");
        }
        match std::env::var(name) {
            Ok(value) => out.push_str(&value),
            Err(_) => {
                if let Some(default) = default {
                    out.push_str(default)
                } else {
                    bail!("Missing environment variable: {name}")
                }
            }
        }
        rest = &tail[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

fn default_listen() -> String {
    "0.0.0.0:8080".into()
}
fn default_request_timeout() -> u64 {
    180
}
fn default_concurrency() -> usize {
    8
}
fn default_auth_ttl() -> i64 {
    900
}
fn default_media_bytes() -> usize {
    20 * 1024 * 1024
}
fn default_history_limit() -> usize {
    30
}
fn default_openrouter_url() -> String {
    "https://openrouter.ai/api/v1".into()
}
fn default_openrouter_search_engine() -> String {
    "auto".into()
}
fn default_openrouter_fetch_engine() -> String {
    "auto".into()
}
fn default_openrouter_search_results() -> u8 {
    5
}
fn default_openrouter_total_results() -> u16 {
    15
}
fn default_search_context_size() -> String {
    "medium".into()
}
fn default_app_name() -> String {
    "Teleforge".into()
}
fn default_image_size() -> String {
    "1024x1024".into()
}
fn default_voice() -> String {
    "alloy".into()
}
fn default_audio_format() -> String {
    "mp3".into()
}
fn default_speed() -> f64 {
    1.0
}
fn default_video_duration() -> u64 {
    8
}
fn default_aspect_ratio() -> String {
    "16:9".into()
}
fn default_video_resolution() -> String {
    "720p".into()
}
fn default_video_poll() -> u64 {
    15
}
fn default_video_timeout() -> u64 {
    600
}
fn default_search_results() -> usize {
    8
}
fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::{Engine, engine::general_purpose::STANDARD};
    #[test]
    fn environment_defaults_expand() {
        assert_eq!(expand_env("A=${UNSET_TELEFORGE_987:-B}").unwrap(), "A=B");
    }

    #[test]
    fn exhaustive_example_is_valid() {
        let yaml = include_str!("../config.example.yaml")
            .replace("${TELEFORGE_PUBLIC_URL}", "https://admin.example.test")
            .replace("${TELEFORGE_MASTER_KEY}", &STANDARD.encode([7_u8; 32]))
            .replace("${TELEGRAM_BOT_TOKENS}", "1:primary-token,2:team-token");
        let mut config: Config = serde_yaml::from_str(&expand_env(&yaml).unwrap()).unwrap();
        config.assign_telegram_tokens().unwrap();
        config.validate().unwrap();
        assert_eq!(config.bots.len(), 2);
        assert_eq!(config.openrouter.image.models.len(), 2);
    }
}
