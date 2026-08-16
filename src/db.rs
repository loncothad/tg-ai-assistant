//! Local redb persistence with per-bot isolation and encrypted credentials.
//!
//! Every key begins with the configured bot ID. Provider keys are encrypted with
//! ChaCha20-Poly1305 before redb sees them and are never returned by admin views.

use std::{collections::BTreeMap, path::Path, sync::Arc};

use base64::{Engine, engine::general_purpose::STANDARD};
use chacha20poly1305::{ChaCha20Poly1305, KeyInit, aead::Aead};
use chrono::Utc;
use compact_str::CompactString;
use eyre::{Context, bail};
use rand::RngExt;
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{
    Result,
    config::{BotConfig, Config, DatabaseConfig, ModelProvider},
};

const STATE: TableDefinition<&str, &[u8]> = TableDefinition::new("state");
const ACCESS: TableDefinition<&str, u8> = TableDefinition::new("access");
const SECRETS: TableDefinition<&str, &[u8]> = TableDefinition::new("secrets");

const SPECIALIZED_GENERATION_CAPABILITIES: [&str; 12] = [
    "text_to_image",
    "image_to_image",
    "text_to_video",
    "image_to_video",
    "video_to_video",
    "text_to_audio",
    "video_to_audio",
    "text_to_speech",
    "image_to_3d",
    "text_to_3d",
    "text_to_image_vector",
    "image_to_image_vector",
];

fn default_generation_models(config: &Config) -> BTreeMap<String, String> {
    SPECIALIZED_GENERATION_CAPABILITIES
        .into_iter()
        .map(|capability| {
            let configured_fal = config
                .fal
                .endpoints
                .iter()
                .find(|endpoint| endpoint.capabilities.iter().any(|item| item == capability))
                .map(|endpoint| endpoint.id.clone());
            let fallback = match capability {
                "text_to_image" | "image_to_image" => config.openrouter.image.model.clone(),
                "text_to_video" | "image_to_video" | "video_to_video" => {
                    config.openrouter.video.model.clone()
                }
                "text_to_speech" => config.openrouter.audio.model.clone(),
                "text_to_audio" | "video_to_audio" => config.openrouter.music.model.clone(),
                // 3D/vector are schema-specific fal endpoints. An empty value
                // makes missing setup explicit instead of choosing a raster model.
                "image_to_3d" | "text_to_3d" | "text_to_image_vector" | "image_to_image_vector" => {
                    configured_fal.unwrap_or_default()
                }
                _ => unreachable!(),
            };
            (capability.to_owned(), fallback)
        })
        .collect()
}

fn default_generation_routing(config: &Config) -> BTreeMap<String, ModelRouting> {
    SPECIALIZED_GENERATION_CAPABILITIES
        .into_iter()
        .filter(|capability| {
            matches!(
                *capability,
                "image_to_3d" | "text_to_3d" | "text_to_image_vector" | "image_to_image_vector"
            ) && config
                .fal
                .endpoints
                .iter()
                .any(|endpoint| endpoint.capabilities.iter().any(|item| item == capability))
        })
        .map(|capability| {
            (
                capability.to_owned(),
                ModelRouting {
                    model_provider: ModelProvider::Fal,
                    ..ModelRouting::default()
                },
            )
        })
        .collect()
}

#[derive(Clone)]
pub struct Store {
    db: Arc<Database>,
    cipher: Arc<ChaCha20Poly1305>,
    history_limit: usize,
    mutation_lock: Arc<tokio::sync::Mutex<()>>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ChatMessage {
    /// Short OpenAI-compatible role name, stored inline in the common case.
    pub role: CompactString,
    pub content: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Capabilities {
    pub search: bool,
    #[serde(default = "enabled_by_default")]
    pub web_fetch: bool,
    pub image: bool,
    pub audio: bool,
    #[serde(default = "enabled_by_default")]
    pub music: bool,
    pub video: bool,
    #[serde(default = "enabled_by_default")]
    pub media: bool,
    #[serde(default = "enabled_by_default")]
    pub transcription: bool,
    #[serde(default = "enabled_by_default")]
    pub file: bool,
    #[serde(default = "enabled_by_default")]
    pub model_upgrade: bool,
    /// Allows public YouTube URLs to be supplied as video-model inputs.
    #[serde(default = "enabled_by_default")]
    pub youtube: bool,
    /// Allows explicitly requested prompt expansion after planner approval.
    #[serde(default = "enabled_by_default")]
    pub prompt_expansion: bool,
    /// Capability-specific overrides. `None` inherits the corresponding
    /// legacy group switch, preserving existing redb settings.
    #[serde(default)]
    pub text_to_image: Option<bool>,
    #[serde(default)]
    pub image_to_image: Option<bool>,
    #[serde(default)]
    pub text_to_video: Option<bool>,
    #[serde(default)]
    pub image_to_video: Option<bool>,
    #[serde(default)]
    pub video_to_video: Option<bool>,
    #[serde(default)]
    pub text_to_audio: Option<bool>,
    #[serde(default)]
    pub video_to_audio: Option<bool>,
    #[serde(default)]
    pub text_to_speech: Option<bool>,
    #[serde(default)]
    pub text_to_3d: Option<bool>,
    #[serde(default)]
    pub image_to_3d: Option<bool>,
    #[serde(default)]
    pub text_to_image_vector: Option<bool>,
    #[serde(default)]
    pub image_to_image_vector: Option<bool>,
    #[serde(default)]
    pub image_understanding: Option<bool>,
    #[serde(default)]
    pub video_understanding: Option<bool>,
    /// Request-specific aggregates used by the tool layer, never persisted.
    #[serde(skip, default = "enabled_by_default")]
    pub three_d: bool,
    #[serde(skip, default = "enabled_by_default")]
    pub vector: bool,
}
impl Default for Capabilities {
    fn default() -> Self {
        Self {
            search: true,
            web_fetch: true,
            image: true,
            audio: true,
            music: true,
            video: true,
            media: true,
            transcription: true,
            file: true,
            model_upgrade: true,
            youtube: true,
            prompt_expansion: true,
            text_to_image: None,
            image_to_image: None,
            text_to_video: None,
            image_to_video: None,
            video_to_video: None,
            text_to_audio: None,
            video_to_audio: None,
            text_to_speech: None,
            text_to_3d: None,
            image_to_3d: None,
            text_to_image_vector: None,
            image_to_image_vector: None,
            image_understanding: None,
            video_understanding: None,
            three_d: true,
            vector: true,
        }
    }
}

impl Capabilities {
    /// Returns an exact skill switch, falling back to the legacy group switch
    /// for settings saved before capability-specific media skills existed.
    pub fn enabled(&self, capability: &str) -> bool {
        match capability {
            "search" => self.search,
            "web_fetch" => self.web_fetch,
            "text_to_image" => self.image && self.text_to_image.unwrap_or(true),
            "image_to_image" => self.image && self.image_to_image.unwrap_or(true),
            "text_to_video" => self.video && self.text_to_video.unwrap_or(true),
            "image_to_video" => self.video && self.image_to_video.unwrap_or(true),
            "video_to_video" => self.video && self.video_to_video.unwrap_or(true),
            "text_to_audio" => self.music && self.text_to_audio.unwrap_or(true),
            "video_to_audio" => self.music && self.video_to_audio.unwrap_or(true),
            "text_to_speech" => self.audio && self.text_to_speech.unwrap_or(true),
            "text_to_3d" => self.image && self.text_to_3d.unwrap_or(true),
            "image_to_3d" => self.image && self.image_to_3d.unwrap_or(true),
            "text_to_image_vector" => self.image && self.text_to_image_vector.unwrap_or(true),
            "image_to_image_vector" => self.image && self.image_to_image_vector.unwrap_or(true),
            "image_understanding" => self.media && self.image_understanding.unwrap_or(true),
            "video_understanding" => self.media && self.video_understanding.unwrap_or(true),
            "transcription" => self.transcription,
            "file" => self.file,
            "model_upgrade" => self.model_upgrade,
            "youtube" => self.youtube,
            "prompt_expansion" => self.prompt_expansion,
            _ => false,
        }
    }

    fn enable_image_overrides(&mut self) {
        if !self.image {
            for value in [
                &mut self.text_to_image,
                &mut self.image_to_image,
                &mut self.text_to_3d,
                &mut self.image_to_3d,
                &mut self.text_to_image_vector,
                &mut self.image_to_image_vector,
            ] {
                value.get_or_insert(false);
            }
        }
        self.image = true;
    }

    fn enable_video_overrides(&mut self) {
        if !self.video {
            for value in [
                &mut self.text_to_video,
                &mut self.image_to_video,
                &mut self.video_to_video,
            ] {
                value.get_or_insert(false);
            }
        }
        self.video = true;
    }

    fn enable_music_overrides(&mut self) {
        if !self.music {
            for value in [&mut self.text_to_audio, &mut self.video_to_audio] {
                value.get_or_insert(false);
            }
        }
        self.music = true;
    }

    fn enable_media_overrides(&mut self) {
        if !self.media {
            for value in [&mut self.image_understanding, &mut self.video_understanding] {
                value.get_or_insert(false);
            }
        }
        self.media = true;
    }

    /// Changes one switch while accepting legacy exported skill identifiers.
    pub fn set(&mut self, capability: &str, enabled: bool) -> Result<()> {
        match capability {
            "search" => self.search = enabled,
            "web_fetch" => self.web_fetch = enabled,
            "image" => self.image = enabled,
            "audio" | "speech" => self.audio = enabled,
            "music" => self.music = enabled,
            "video" => self.video = enabled,
            "media" => self.media = enabled,
            "text_to_image" => {
                self.enable_image_overrides();
                self.text_to_image = Some(enabled);
            }
            "image_to_image" => {
                self.enable_image_overrides();
                self.image_to_image = Some(enabled);
            }
            "text_to_video" => {
                self.enable_video_overrides();
                self.text_to_video = Some(enabled);
            }
            "image_to_video" => {
                self.enable_video_overrides();
                self.image_to_video = Some(enabled);
            }
            "video_to_video" => {
                self.enable_video_overrides();
                self.video_to_video = Some(enabled);
            }
            "text_to_audio" => {
                self.enable_music_overrides();
                self.text_to_audio = Some(enabled);
            }
            "video_to_audio" => {
                self.enable_music_overrides();
                self.video_to_audio = Some(enabled);
            }
            "text_to_speech" => {
                self.audio = true;
                self.text_to_speech = Some(enabled);
            }
            "text_to_3d" => {
                self.enable_image_overrides();
                self.text_to_3d = Some(enabled);
            }
            "image_to_3d" => {
                self.enable_image_overrides();
                self.image_to_3d = Some(enabled);
            }
            "text_to_image_vector" => {
                self.enable_image_overrides();
                self.text_to_image_vector = Some(enabled);
            }
            "image_to_image_vector" => {
                self.enable_image_overrides();
                self.image_to_image_vector = Some(enabled);
            }
            "image_understanding" => {
                self.enable_media_overrides();
                self.image_understanding = Some(enabled);
            }
            "video_understanding" => {
                self.enable_media_overrides();
                self.video_understanding = Some(enabled);
            }
            "transcription" => self.transcription = enabled,
            "file" => self.file = enabled,
            "model_upgrade" => self.model_upgrade = enabled,
            "youtube" => self.youtube = enabled,
            "prompt_expansion" => self.prompt_expansion = enabled,
            _ => bail!("Unknown capability: {capability}"),
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BotSettings {
    pub selected_model: String,
    #[serde(default)]
    pub selected_output_processing_model: String,
    #[serde(default)]
    pub selected_error_processing_model: String,
    #[serde(default)]
    pub selected_planner_model: String,
    #[serde(default)]
    pub selected_planner_fallback_model: String,
    #[serde(default)]
    pub selected_upgrade_model: String,
    #[serde(default)]
    pub selected_image_understanding_model: String,
    #[serde(default)]
    pub selected_video_understanding_model: String,
    #[serde(default)]
    pub selected_image_generation_model: String,
    #[serde(default)]
    pub selected_audio_generation_model: String,
    #[serde(default)]
    pub selected_music_generation_model: String,
    #[serde(default)]
    pub selected_transcription_model: String,
    #[serde(default)]
    pub selected_video_generation_model: String,
    /// Per-input/output generation models.  Kept as a map so newly introduced
    /// provider modalities do not require a redb schema migration.
    #[serde(default)]
    pub specialized_generation_models: BTreeMap<String, String>,
    #[serde(default)]
    pub model_routing: BTreeMap<String, ModelRouting>,
    pub search_provider: Option<String>,
    pub capabilities: Capabilities,
    pub custom_system_prompt: Option<String>,
    pub custom_system_prompt_enabled: bool,
    pub custom_skills: Option<String>,
    pub custom_skills_enabled: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ModelRouting {
    #[serde(default)]
    pub model_provider: ModelProvider,
    pub strategy: String,
    pub provider: Option<String>,
}

impl Default for ModelRouting {
    fn default() -> Self {
        Self {
            model_provider: ModelProvider::Openrouter,
            strategy: "auto".into(),
            provider: None,
        }
    }
}

impl Store {
    pub async fn connect(config: &DatabaseConfig) -> Result<Self> {
        let key = STANDARD
            .decode(&config.encryption_key)
            .wrap_err("Database encryption_key must be base64")?;
        if key.len() != 32 {
            bail!("Database encryption_key must decode to exactly 32 bytes");
        }
        if let Some(parent) = Path::new(&config.path).parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .wrap_err("Failed to create database directory")?;
        }
        let db = Database::create(&config.path).wrap_err("Failed to open local redb database")?;
        let store = Self {
            db: Arc::new(db),
            cipher: Arc::new(ChaCha20Poly1305::new_from_slice(&key).expect("validated key")),
            history_limit: config.history_limit,
            mutation_lock: Arc::new(tokio::sync::Mutex::new(())),
        };
        let write = store.db.begin_write()?;
        {
            write.open_table(STATE)?;
            write.open_table(ACCESS)?;
            write.open_table(SECRETS)?;
        }
        write.commit()?;
        Ok(store)
    }

    pub async fn seed_bot(&self, bot: &BotConfig, config: &Config) -> Result<()> {
        if self.get_settings(&bot.id).await?.is_none() {
            self.put_json(
                &settings_key(&bot.id),
                &BotSettings {
                    selected_model: bot.default_model.clone(),
                    selected_output_processing_model: bot.default_model.clone(),
                    selected_error_processing_model: bot.default_model.clone(),
                    selected_planner_model: config.openrouter.planner.model.clone(),
                    selected_planner_fallback_model: config
                        .openrouter
                        .planner
                        .fallback_model
                        .clone(),
                    selected_upgrade_model: bot.default_model.clone(),
                    selected_image_understanding_model: config
                        .openrouter
                        .understanding
                        .image
                        .default_model
                        .clone(),
                    selected_video_understanding_model: config
                        .openrouter
                        .understanding
                        .video
                        .default_model
                        .clone(),
                    selected_image_generation_model: config.openrouter.image.model.clone(),
                    selected_audio_generation_model: config.openrouter.audio.model.clone(),
                    selected_music_generation_model: config.openrouter.music.model.clone(),
                    selected_transcription_model: config.openrouter.transcription.model.clone(),
                    selected_video_generation_model: config.openrouter.video.model.clone(),
                    specialized_generation_models: default_generation_models(config),
                    model_routing: default_generation_routing(config),
                    search_provider: None,
                    capabilities: Capabilities::default(),
                    custom_system_prompt: None,
                    custom_system_prompt_enabled: false,
                    custom_skills: None,
                    custom_skills_enabled: false,
                },
            )
            .await?;
        }
        let mut settings = self.settings(&bot.id).await?;
        let mut changed = false;
        for (value, fallback) in [
            (
                &mut settings.selected_output_processing_model,
                &bot.default_model,
            ),
            (
                &mut settings.selected_error_processing_model,
                &bot.default_model,
            ),
            (
                &mut settings.selected_planner_model,
                &config.openrouter.planner.model,
            ),
            (
                &mut settings.selected_planner_fallback_model,
                &config.openrouter.planner.fallback_model,
            ),
            (&mut settings.selected_upgrade_model, &bot.default_model),
            (
                &mut settings.selected_image_understanding_model,
                &config.openrouter.understanding.image.default_model,
            ),
            (
                &mut settings.selected_video_understanding_model,
                &config.openrouter.understanding.video.default_model,
            ),
            (
                &mut settings.selected_image_generation_model,
                &config.openrouter.image.model,
            ),
            (
                &mut settings.selected_audio_generation_model,
                &config.openrouter.audio.model,
            ),
            (
                &mut settings.selected_music_generation_model,
                &config.openrouter.music.model,
            ),
            (
                &mut settings.selected_transcription_model,
                &config.openrouter.transcription.model,
            ),
            (
                &mut settings.selected_video_generation_model,
                &config.openrouter.video.model,
            ),
        ] {
            if value.is_empty() {
                value.clone_from(fallback);
                changed = true;
            }
        }
        if settings.selected_model.is_empty() {
            settings.selected_model.clone_from(&bot.default_model);
            changed = true;
        }
        for (capability, fallback) in default_generation_models(config) {
            if fallback.is_empty() {
                continue;
            }
            if settings
                .specialized_generation_models
                .get(&capability)
                .is_none_or(String::is_empty)
            {
                settings
                    .specialized_generation_models
                    .insert(capability, fallback);
                changed = true;
            }
        }
        for (capability, routing) in default_generation_routing(config) {
            if let std::collections::btree_map::Entry::Vacant(entry) =
                settings.model_routing.entry(capability)
            {
                entry.insert(routing);
                changed = true;
            }
        }
        if changed {
            self.save_settings(&bot.id, &settings).await?;
        }
        for user in &bot.allowed_user_ids {
            if self.user_allowed(&bot.id, *user).await?.is_none() {
                self.set_user_allowed(&bot.id, *user, true, 0).await?;
            }
        }
        for (provider, value) in [
            ("openrouter", config.openrouter.bootstrap_api_key.as_str()),
            ("aihub", config.aihub.bootstrap_api_key.as_str()),
            ("fal", config.fal.bootstrap_api_key.as_str()),
            ("brave", config.search.brave.bootstrap_api_key.as_str()),
            ("exa", config.search.exa.bootstrap_api_key.as_str()),
            ("serpapi", config.search.serpapi.bootstrap_api_key.as_str()),
        ] {
            if !value.is_empty() && !self.credential_configured(&bot.id, provider).await? {
                self.set_credential(&bot.id, provider, value).await?;
            }
        }
        Ok(())
    }

    pub async fn settings(&self, bot_id: &str) -> Result<BotSettings> {
        self.get_settings(bot_id)
            .await?
            .ok_or_else(|| eyre::eyre!("Bot settings are not initialized"))
    }
    async fn get_settings(&self, bot_id: &str) -> Result<Option<BotSettings>> {
        self.get_json(&settings_key(bot_id)).await
    }
    pub async fn save_settings(&self, bot_id: &str, settings: &BotSettings) -> Result<()> {
        let _guard = self.mutation_lock.lock().await;
        self.save_settings_unlocked(bot_id, settings).await
    }
    async fn save_settings_unlocked(&self, bot_id: &str, settings: &BotSettings) -> Result<()> {
        self.put_json(&settings_key(bot_id), settings).await
    }
    pub async fn selected_model(&self, bot_id: &str) -> Result<String> {
        Ok(self.settings(bot_id).await?.selected_model)
    }
    pub async fn set_model(
        &self,
        bot_id: &str,
        capability: &str,
        model: &str,
        routing: ModelRouting,
        actor: u64,
    ) -> Result<()> {
        let _guard = self.mutation_lock.lock().await;
        let mut settings = self.settings(bot_id).await?;
        match capability {
            "chat" => settings.selected_model = model.into(),
            "output_processing" => settings.selected_output_processing_model = model.into(),
            "error_processing" => settings.selected_error_processing_model = model.into(),
            "intent_planning" => settings.selected_planner_model = model.into(),
            "intent_planning_fallback" => settings.selected_planner_fallback_model = model.into(),
            "model_upgrade" => settings.selected_upgrade_model = model.into(),
            "image_understanding" => settings.selected_image_understanding_model = model.into(),
            "video_understanding" => settings.selected_video_understanding_model = model.into(),
            "image_generation" => settings.selected_image_generation_model = model.into(),
            "audio_generation" | "speech_generation" => {
                settings.selected_audio_generation_model = model.into()
            }
            "music_generation" => settings.selected_music_generation_model = model.into(),
            "transcription" => settings.selected_transcription_model = model.into(),
            "video_generation" => settings.selected_video_generation_model = model.into(),
            capability if crate::catalog::is_specialized_generation_capability(capability) => {
                settings
                    .specialized_generation_models
                    .insert(capability.into(), model.into());
            }
            _ => bail!("Unknown model capability: {capability}"),
        }
        settings.model_routing.insert(capability.into(), routing);
        self.save_settings_unlocked(bot_id, &settings).await?;
        self.audit(bot_id, Some(actor), "model.set", Some(capability))
            .await
    }
    pub async fn selected_search_provider(&self, bot_id: &str) -> Result<Option<String>> {
        Ok(self.settings(bot_id).await?.search_provider)
    }
    pub async fn set_search_provider(
        &self,
        bot_id: &str,
        provider: &str,
        actor: u64,
    ) -> Result<()> {
        let _guard = self.mutation_lock.lock().await;
        let mut settings = self.settings(bot_id).await?;
        settings.search_provider = Some(provider.into());
        self.save_settings_unlocked(bot_id, &settings).await?;
        self.audit(bot_id, Some(actor), "search_provider.set", Some(provider))
            .await
    }

    pub async fn set_capability(
        &self,
        bot_id: &str,
        capability: &str,
        enabled: bool,
    ) -> Result<()> {
        let _guard = self.mutation_lock.lock().await;
        let mut settings = self.settings(bot_id).await?;
        settings.capabilities.set(capability, enabled)?;
        self.save_settings_unlocked(bot_id, &settings).await
    }

    pub async fn set_custom_content(
        &self,
        bot_id: &str,
        kind: &str,
        content: Option<String>,
        enabled: bool,
    ) -> Result<()> {
        let _guard = self.mutation_lock.lock().await;
        let mut settings = self.settings(bot_id).await?;
        match kind {
            "prompt" => {
                settings.custom_system_prompt = content;
                settings.custom_system_prompt_enabled = enabled;
            }
            "skills" => {
                settings.custom_skills = content;
                settings.custom_skills_enabled = enabled;
            }
            _ => bail!("Unknown custom content type"),
        }
        self.save_settings_unlocked(bot_id, &settings).await
    }

    pub async fn effective_instructions(&self, bot_id: &str) -> Result<String> {
        let settings = self.settings(bot_id).await?;
        let prompt = if settings.custom_system_prompt_enabled {
            settings
                .custom_system_prompt
                .as_deref()
                .unwrap_or(crate::defaults::SYSTEM_PROMPT)
        } else {
            crate::defaults::SYSTEM_PROMPT
        };
        let enabled = |id: &str| settings.capabilities.enabled(id);
        let mut skills = crate::defaults::BUILTIN_SKILLS
            .iter()
            .filter(|skill| enabled(skill.id))
            .map(|skill| skill.instructions)
            .collect::<Vec<_>>()
            .join("\n\n");
        if settings.custom_skills_enabled {
            if let Some(custom) = &settings.custom_skills {
                skills.push_str("\n\n# Imported custom skills\n");
                skills.push_str(custom);
            }
        }
        Ok(format!("{prompt}\n\n# Enabled skills\n{skills}"))
    }

    pub async fn credential_configured(&self, bot_id: &str, provider: &str) -> Result<bool> {
        Ok(self
            .get_raw(SECRETS, &secret_key(bot_id, provider))?
            .is_some())
    }
    pub async fn set_credential(&self, bot_id: &str, provider: &str, secret: &str) -> Result<()> {
        validate_provider(provider)?;
        let encrypted = self.encrypt(secret.as_bytes())?;
        self.put_raw(SECRETS, &secret_key(bot_id, provider), &encrypted)
    }
    pub async fn remove_credential(&self, bot_id: &str, provider: &str) -> Result<()> {
        validate_provider(provider)?;
        self.remove_raw(SECRETS, &secret_key(bot_id, provider))
    }
    pub async fn credential(&self, bot_id: &str, provider: &str) -> Result<Option<String>> {
        validate_provider(provider)?;
        self.get_raw(SECRETS, &secret_key(bot_id, provider))?
            .map(|value| {
                String::from_utf8(self.decrypt(&value)?)
                    .wrap_err("Decrypted credential is invalid UTF-8")
            })
            .transpose()
    }

    pub async fn user_allowed(&self, bot_id: &str, user_id: u64) -> Result<Option<bool>> {
        Ok(self.get_u8(&access_key(bot_id, user_id))?.map(|v| v != 0))
    }
    pub async fn set_user_allowed(
        &self,
        bot_id: &str,
        user_id: u64,
        allowed: bool,
        actor: u64,
    ) -> Result<()> {
        self.put_u8(&access_key(bot_id, user_id), u8::from(allowed))?;
        self.audit(
            bot_id,
            Some(actor),
            if allowed {
                "access.allow"
            } else {
                "access.deny"
            },
            Some(&user_id.to_string()),
        )
        .await
    }

    /// Removes a user entry from this bot's allowlist entirely. This differs
    /// from denying a user: the next configured YAML allowlist seed can add
    /// the user again if the entry is absent.
    pub async fn remove_user_allowed(
        &self,
        bot_id: &str,
        user_id: u64,
        actor: u64,
    ) -> Result<bool> {
        let _guard = self.mutation_lock.lock().await;
        let key = access_key(bot_id, user_id);
        if self.get_u8(&key)?.is_none() {
            return Ok(false);
        }
        self.remove_u8(&key)?;
        self.audit(
            bot_id,
            Some(actor),
            "access.remove",
            Some(&user_id.to_string()),
        )
        .await?;
        Ok(true)
    }

    pub async fn list_allowed_users(&self, bot_id: &str) -> Result<Vec<u64>> {
        let prefix = format!("{bot_id}|");
        let read = self.db.begin_read()?;
        let table = read.open_table(ACCESS)?;
        let mut users = Vec::new();
        for entry in table.iter()? {
            let (key, value) = entry?;
            if value.value() != 0 && key.value().starts_with(&prefix) {
                if let Ok(id) = key.value()[prefix.len()..].parse() {
                    users.push(id);
                }
            }
        }
        users.sort_unstable();
        Ok(users)
    }

    pub async fn history(&self, bot_id: &str, scope: &str) -> Result<Vec<ChatMessage>> {
        Ok(self
            .get_json(&history_key(bot_id, scope))
            .await?
            .unwrap_or_default())
    }
    pub async fn append_message(
        &self,
        bot_id: &str,
        scope: &str,
        role: &str,
        content: &str,
    ) -> Result<()> {
        let _guard = self.mutation_lock.lock().await;
        let mut values: Vec<ChatMessage> = self.history(bot_id, scope).await?;
        values.push(ChatMessage {
            role: role.into(),
            content: content.into(),
        });
        if values.len() > self.history_limit {
            values.drain(..values.len() - self.history_limit);
        }
        self.put_json(&history_key(bot_id, scope), &values).await
    }
    pub async fn clear_history(&self, bot_id: &str, scope: &str) -> Result<()> {
        self.remove_raw(STATE, &history_key(bot_id, scope))
    }
    pub async fn offset(&self, bot_id: &str) -> Result<Option<i64>> {
        self.get_json(&format!("offset|{bot_id}")).await
    }
    pub async fn set_offset(&self, bot_id: &str, offset: i64) -> Result<()> {
        self.put_json(&format!("offset|{bot_id}"), &offset).await
    }
    pub async fn audit(
        &self,
        bot_id: &str,
        actor: Option<u64>,
        action: &str,
        details: Option<&str>,
    ) -> Result<()> {
        let key = format!(
            "audit|{bot_id}|{}|{}",
            Utc::now().timestamp_nanos_opt().unwrap_or_default(),
            actor.unwrap_or_default()
        );
        self.put_json(&key, &(action, details)).await
    }

    async fn get_json<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        self.get_raw(STATE, key)?
            .map(|v| serde_json::from_slice(&v).wrap_err("Stored JSON is invalid"))
            .transpose()
    }
    async fn put_json<T: Serialize>(&self, key: &str, value: &T) -> Result<()> {
        self.put_raw(STATE, key, &serde_json::to_vec(value)?)
    }
    fn get_raw(
        &self,
        definition: TableDefinition<&str, &[u8]>,
        key: &str,
    ) -> Result<Option<Vec<u8>>> {
        let read = self.db.begin_read()?;
        let table = read.open_table(definition)?;
        Ok(table.get(key)?.map(|v| v.value().to_vec()))
    }
    fn put_raw(
        &self,
        definition: TableDefinition<&str, &[u8]>,
        key: &str,
        value: &[u8],
    ) -> Result<()> {
        let write = self.db.begin_write()?;
        {
            let mut table = write.open_table(definition)?;
            table.insert(key, value)?;
        }
        write.commit()?;
        Ok(())
    }
    fn remove_raw(&self, definition: TableDefinition<&str, &[u8]>, key: &str) -> Result<()> {
        let write = self.db.begin_write()?;
        {
            let mut table = write.open_table(definition)?;
            table.remove(key)?;
        }
        write.commit()?;
        Ok(())
    }
    fn get_u8(&self, key: &str) -> Result<Option<u8>> {
        let read = self.db.begin_read()?;
        let table = read.open_table(ACCESS)?;
        Ok(table.get(key)?.map(|v| v.value()))
    }
    fn put_u8(&self, key: &str, value: u8) -> Result<()> {
        let write = self.db.begin_write()?;
        {
            write.open_table(ACCESS)?.insert(key, value)?;
        }
        write.commit()?;
        Ok(())
    }
    fn remove_u8(&self, key: &str) -> Result<()> {
        let write = self.db.begin_write()?;
        {
            write.open_table(ACCESS)?.remove(key)?;
        }
        write.commit()?;
        Ok(())
    }
    fn encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>> {
        let mut nonce = [0u8; 12];
        rand::rng().fill(&mut nonce);
        let mut out = nonce.to_vec();
        out.extend(
            self.cipher
                .encrypt((&nonce).into(), plaintext)
                .map_err(|_| eyre::eyre!("Credential encryption failed"))?,
        );
        Ok(out)
    }
    fn decrypt(&self, value: &[u8]) -> Result<Vec<u8>> {
        if value.len() < 13 {
            bail!("Encrypted credential is truncated");
        }
        let nonce: [u8; 12] = value[..12]
            .try_into()
            .wrap_err("Encrypted credential nonce is invalid")?;
        self.cipher
            .decrypt((&nonce).into(), &value[12..])
            .map_err(|_| eyre::eyre!("Credential decryption failed"))
    }
}

fn settings_key(bot: &str) -> String {
    format!("settings|{bot}")
}
fn history_key(bot: &str, scope: &str) -> String {
    format!("history|{bot}|{scope}")
}
fn access_key(bot: &str, user: u64) -> String {
    format!("{bot}|{user}")
}
fn secret_key(bot: &str, provider: &str) -> String {
    format!("{bot}|{provider}")
}
fn validate_provider(value: &str) -> Result<()> {
    if matches!(
        value,
        "openrouter" | "aihub" | "fal" | "brave" | "exa" | "serpapi"
    ) {
        Ok(())
    } else {
        bail!("Unknown credential provider")
    }
}

fn enabled_by_default() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn test_store() -> (tempfile::TempDir, Store) {
        let directory = tempfile::tempdir().unwrap();
        let store = Store::connect(&DatabaseConfig {
            path: directory.path().join("state.redb").display().to_string(),
            encryption_key: STANDARD.encode([9_u8; 32]),
            history_limit: 3,
        })
        .await
        .unwrap();
        for bot in ["a", "b"] {
            store
                .put_json(
                    &settings_key(bot),
                    &BotSettings {
                        selected_model: "chat".into(),
                        selected_output_processing_model: "output".into(),
                        selected_error_processing_model: "errors".into(),
                        selected_planner_model: "planner".into(),
                        selected_planner_fallback_model: "planner-fallback".into(),
                        selected_upgrade_model: "advanced".into(),
                        selected_image_understanding_model: "vision".into(),
                        selected_video_understanding_model: "video-vision".into(),
                        selected_image_generation_model: "image".into(),
                        selected_audio_generation_model: "speech".into(),
                        selected_music_generation_model: "music".into(),
                        selected_transcription_model: "stt".into(),
                        selected_video_generation_model: "video".into(),
                        specialized_generation_models: BTreeMap::new(),
                        model_routing: BTreeMap::new(),
                        search_provider: None,
                        capabilities: Capabilities::default(),
                        custom_system_prompt: None,
                        custom_system_prompt_enabled: false,
                        custom_skills: None,
                        custom_skills_enabled: false,
                    },
                )
                .await
                .unwrap();
        }
        (directory, store)
    }

    #[tokio::test]
    async fn bot_state_is_isolated_and_credentials_are_encrypted() {
        let (_directory, store) = test_store().await;
        store
            .append_message("a", "pm:1", "user", "Only bot A")
            .await
            .unwrap();
        assert_eq!(store.history("a", "pm:1").await.unwrap().len(), 1);
        assert!(store.history("b", "pm:1").await.unwrap().is_empty());

        store
            .set_credential("a", "openrouter", "super-secret-value")
            .await
            .unwrap();
        assert_eq!(
            store
                .credential("a", "openrouter")
                .await
                .unwrap()
                .as_deref(),
            Some("super-secret-value")
        );
        assert!(store.credential("b", "openrouter").await.unwrap().is_none());
        let ciphertext = store
            .get_raw(SECRETS, &secret_key("a", "openrouter"))
            .unwrap()
            .unwrap();
        assert!(
            !ciphertext
                .windows(b"super-secret-value".len())
                .any(|window| window == b"super-secret-value")
        );
    }

    #[test]
    fn legacy_model_routing_defaults_to_openrouter() {
        let routing: ModelRouting =
            serde_json::from_str(r#"{"strategy":"auto","provider":null}"#).unwrap();
        assert_eq!(routing.model_provider, ModelProvider::Openrouter);
    }

    #[tokio::test]
    async fn disabling_a_skill_removes_its_embedded_instructions() {
        let (_directory, store) = test_store().await;
        store
            .set_capability("a", "text_to_image", false)
            .await
            .unwrap();
        let instructions = store.effective_instructions("a").await.unwrap();
        assert!(!instructions.contains("# Text-to-image generation"));
        assert!(instructions.contains("# Image-to-image generation"));
        assert!(instructions.contains("# Web research skill"));
    }

    #[test]
    fn capability_specific_media_switches_preserve_legacy_group_state() {
        let mut capabilities = Capabilities {
            image: false,
            ..Capabilities::default()
        };
        assert!(!capabilities.enabled("text_to_image"));
        assert!(!capabilities.enabled("image_to_image"));
        capabilities.set("text_to_image", true).unwrap();
        assert!(capabilities.enabled("text_to_image"));
        assert!(!capabilities.enabled("image_to_image"));
    }

    #[tokio::test]
    async fn removing_allowlist_entry_is_idempotent_and_deletes_the_entry() {
        let (_directory, store) = test_store().await;
        store.set_user_allowed("a", 42, true, 1).await.unwrap();
        assert_eq!(store.user_allowed("a", 42).await.unwrap(), Some(true));
        assert!(store.remove_user_allowed("a", 42, 1).await.unwrap());
        assert_eq!(store.user_allowed("a", 42).await.unwrap(), None);
        assert!(!store.remove_user_allowed("a", 42, 1).await.unwrap());
    }
}
