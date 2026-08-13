//! Multi-token Frankenstein long-polling runtime and rich-message delivery.

use std::{sync::Arc, time::Duration};

use base64::{Engine, engine::general_purpose::STANDARD};
use eyre::{Context, ContextCompat, bail};
use frankenstein::{
    AsyncTelegramApi,
    client_reqwest::Bot,
    inline_mode::{
        InlineQueryResult, InlineQueryResultArticle, InputMessageContent, InputRichMessageContent,
    },
    input_file::FileUpload,
    input_media::{
        InputMedia, InputMediaAudio, InputMediaDocument, InputMediaPhoto, InputMediaVideo,
    },
    methods::{
        AnswerGuestQueryParams, EditMessageMediaParams, EditMessageTextParams, GetFileParams,
        GetUpdatesParams, SendAudioParams, SendChatActionParams, SendDocumentParams,
        SendPhotoParams, SendRichMessageParams, SendVideoParams,
    },
    rich_message::InputRichMessage,
    types::{
        AllowedUpdate, ChatAction, ChatType, InlineKeyboardButton, InlineKeyboardMarkup, Message,
        ReplyMarkup, ReplyParameters, WebAppInfo,
    },
    updates::{Update, UpdateContent},
};
use tempfile::Builder as TempFileBuilder;
use tokio::{
    sync::{mpsc, watch},
    task::JoinSet,
    time::sleep,
};
use tracing::{error, info, warn};

use crate::{
    Result,
    catalog::ModelCatalogCache,
    config::{BotConfig, Config, ModelProvider, SearchProvider},
    db::{ModelRouting, Store},
    openrouter::{
        ChatRequest, GenerationPromptContext, MediaInput, OpenRouter, PlannedAction, PlannedSkill,
        PlanningRequest, ProgressUpdate, RequestPlan, ToolModel, ToolModels,
    },
    rich,
    search::SearchService,
};

#[derive(Clone)]
pub struct BotRunner {
    telegram: Bot,
    bot: BotConfig,
    config: Arc<Config>,
    store: Store,
    openrouter: OpenRouter,
    search: SearchService,
    client: reqwest::Client,
    catalog: ModelCatalogCache,
    bot_user_id: u64,
    username: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MessageMode {
    Private,
    Group,
    Guest,
}

#[derive(Clone, Copy, Default)]
struct CommandOptions<'a> {
    model_override: Option<&'a MessageModelOverride>,
    guest_pending_id: Option<&'a str>,
    reply_context: Option<&'a str>,
}

impl BotRunner {
    pub async fn new(
        bot: BotConfig,
        config: Arc<Config>,
        store: Store,
        client: reqwest::Client,
    ) -> Result<Self> {
        store.seed_bot(&bot, &config).await?;
        let telegram = Bot::builder()
            .api_url(format!("{}{}", frankenstein::BASE_API_URL, bot.token))
            .client(client.clone())
            .build();
        let identity = telegram
            .get_me()
            .await
            .context("Telegram getMe failed")?
            .result;
        let username = identity
            .username
            .clone()
            .context("Configured Telegram bot has no username")?;
        info!(bot_id = %bot.id, telegram_username = %username, "Telegram bot authenticated");
        let openrouter = OpenRouter::new(
            client.clone(),
            config.openrouter.clone(),
            config.aihub.clone(),
        );
        let search = SearchService::new(client.clone(), config.search.clone());
        Ok(Self {
            telegram,
            bot,
            config,
            store,
            openrouter,
            search,
            client,
            catalog: ModelCatalogCache::default(),
            bot_user_id: identity.id,
            username,
        })
    }

    pub async fn run(self, mut shutdown: watch::Receiver<bool>) -> Result<()> {
        let mut offset = self.store.offset(&self.bot.id).await?;
        let mut backoff = 1u64;
        let mut jobs = JoinSet::new();
        loop {
            if *shutdown.borrow() {
                break;
            }
            let params = GetUpdatesParams::builder()
                .maybe_offset(offset)
                .timeout(30)
                .allowed_updates(vec![AllowedUpdate::Message, AllowedUpdate::GuestMessage])
                .build();
            let response = tokio::select! {
                _ = shutdown.changed() => break,
                result = jobs.join_next(), if !jobs.is_empty() => {
                    if let Some(result) = result {
                        self.log_update_result(result);
                    }
                    continue;
                }
                response = self.telegram.get_updates(&params) => response,
            };
            let updates = match response {
                Ok(response) => {
                    backoff = 1;
                    response.result
                }
                Err(error) => {
                    warn!(bot_id = %self.bot.id, %error, retry_seconds = backoff, "Telegram polling failed");
                    tokio::select! { _ = shutdown.changed() => break, _ = sleep(Duration::from_secs(backoff)) => {} }
                    backoff = (backoff * 2).min(30);
                    continue;
                }
            };
            if updates.is_empty() {
                continue;
            }
            let next_offset = updates
                .iter()
                .map(|u| i64::from(u.update_id) + 1)
                .max()
                .expect("non-empty updates");
            for update in updates {
                let runner = self.clone();
                jobs.spawn(async move { runner.process(update).await });
            }
            self.store.set_offset(&self.bot.id, next_offset).await?;
            offset = Some(next_offset);
        }
        while let Some(result) = jobs.join_next().await {
            self.log_update_result(result);
        }
        info!(bot_id = %self.bot.id, "Telegram bot stopped");
        Ok(())
    }

    fn log_update_result(&self, result: std::result::Result<Result<()>, tokio::task::JoinError>) {
        match result {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                error!(bot_id = %self.bot.id, error = %format!("{error:#}"), "update processing failed")
            }
            Err(error) => error!(bot_id = %self.bot.id, %error, "update task panicked"),
        }
    }

    async fn process(&self, update: Update) -> Result<()> {
        match update.content {
            UpdateContent::Message(message) => {
                self.process_message(*message, MessageMode::Private).await
            }
            UpdateContent::GuestMessage(message) => {
                self.process_message(*message, MessageMode::Guest).await
            }
            _ => Ok(()),
        }
    }

    async fn process_message(&self, message: Message, hinted_mode: MessageMode) -> Result<()> {
        let mode = if hinted_mode == MessageMode::Guest {
            MessageMode::Guest
        } else if message.chat.type_field == ChatType::Private {
            MessageMode::Private
        } else {
            MessageMode::Group
        };
        let user_id = message
            .guest_bot_caller_user
            .as_ref()
            .or(message.from.as_ref())
            .map(|u| u.id)
            .context("Message has no identifiable caller")?;
        if !self.mode_enabled(mode) {
            return Ok(());
        }
        if !self.is_allowed(user_id, message.chat.id).await? {
            if mode != MessageMode::Group {
                self.respond(&message, mode, "This bot is restricted. Ask an administrator to add your Telegram user ID to the allowlist.", None).await?;
            }
            return Ok(());
        }
        let raw_text = message
            .text
            .as_deref()
            .or(message.caption.as_deref())
            .unwrap_or_default()
            .trim();
        let has_media = message_has_media(&message)
            || message
                .reply_to_message
                .as_deref()
                .is_some_and(message_has_media)
            || message
                .external_reply
                .as_deref()
                .is_some_and(external_reply_has_media);
        if raw_text.is_empty() && !has_media {
            return Ok(());
        }
        let command_without_mention = parse_command(raw_text).is_some();
        if mode == MessageMode::Group
            && self.bot.access.require_mention_in_groups
            && !self.addressed_to_bot(&message, raw_text)
            && !command_without_mention
        {
            return Ok(());
        }
        let mut text = if raw_text.is_empty() {
            default_media_prompt(&message).to_owned()
        } else {
            self.strip_address(raw_text)
        };
        let reply_context = reply_context(&message);
        let replied_text = replied_message_text(&message);
        let telegram_quote = telegram_quote_text(&message);
        let scope = scope_id(mode, &message, user_id);

        let model_override = if let Some((command, arguments)) = parse_command(&text)
            && command == "model"
            && !arguments.is_empty()
        {
            if !self.is_admin(user_id) {
                self.respond(
                    &message,
                    mode,
                    "Administrator access is required for per-message model overrides.",
                    None,
                )
                .await?;
                return Ok(());
            }
            let (model_override, prompt) = parse_model_override(arguments)?;
            text = prompt.to_owned();
            Some(model_override)
        } else {
            None
        };

        if model_override.is_none()
            && let Some((command, arguments)) = parse_command(&text)
        {
            if let Err(error) = self
                .command(
                    &message,
                    mode,
                    user_id,
                    &command,
                    arguments,
                    CommandOptions {
                        // Explicit commands bypass the natural-language
                        // planner. Pass only Telegram's selected quote (or the
                        // raw replied text), never the verbose context wrapper.
                        reply_context: telegram_quote.or(replied_text),
                        ..CommandOptions::default()
                    },
                )
                .await
            {
                error!(bot_id = %self.bot.id, user_id, error = %format!("{error:#}"), "command failed");
                self.respond(&message, mode, &rich::detailed_error(&error), None)
                    .await?;
            }
            return Ok(());
        }

        // Telegram does not support sendChatAction for guest-bot messages and
        // answers those attempts with PEER_ID_INVALID. Guest replies still use
        // answerGuestQuery through `respond` below.
        if mode != MessageMode::Guest && message.guest_bot_caller_user.is_none() {
            self.send_action(
                message.chat.id,
                message.message_thread_id,
                ChatAction::Typing,
            )
            .await;
        }
        let guest_pending_id = if mode == MessageMode::Guest {
            Some(
                self.answer_guest_pending(&message, "Reading the request")
                    .await?,
            )
        } else {
            None
        };
        let (sender, receiver) = mpsc::unbounded_channel();
        let runner = self.clone();
        let progress_task = if let Some(inline_id) = guest_pending_id.clone() {
            tokio::spawn(async move {
                runner.update_guest_progress(inline_id, receiver).await;
            })
        } else {
            let progress_message_id = self
                .send_progress_message(&message, "Reading the request")
                .await?;
            let chat_id = message.chat.id;
            tokio::spawn(async move {
                runner
                    .update_progress_message(chat_id, progress_message_id, receiver)
                    .await;
            })
        };
        let _ = sender.send(ProgressUpdate::step("Read request"));
        if reply_context.is_some() {
            let _ = sender.send(ProgressUpdate::step("Loaded replied-message context"));
        }
        let _ = sender.send(ProgressUpdate::step("Classifying request"));
        let settings = self.store.settings(&self.bot.id).await?;
        let mut capabilities = settings.capabilities.clone();
        let attachment_flags = attachment_flags(&message);
        let planner_key = self.store.credential(&self.bot.id, "openrouter").await?;
        let request_text = with_reply_context(&text, reply_context.as_deref());
        let plan = if let Some(key) = planner_key.as_deref() {
            match self
                .openrouter
                .plan_request(PlanningRequest {
                    text: &text,
                    replied_message: replied_text,
                    telegram_quote,
                    model: &settings.selected_planner_model,
                    fallback_model: &settings.selected_planner_fallback_model,
                    capabilities: &capabilities,
                    has_image: attachment_flags.0,
                    has_video: attachment_flags.1,
                    has_audio: attachment_flags.2,
                    api_key: key,
                })
                .await
            {
                Ok(plan) => Some(plan),
                Err(error) => {
                    warn!(bot_id = %self.bot.id, user_id, error = %format!("{error:#}"), "request planner failed; continuing with normal chat");
                    None
                }
            }
        } else {
            None
        };
        let _ = sender.send(ProgressUpdate::step(if plan.is_some() {
            "Classified request"
        } else {
            "Applied standard routing"
        }));
        let override_generation = if let Some(override_) = model_override.as_ref() {
            match self.media_only_override_action(override_).await {
                Ok(action) => action,
                Err(error) => {
                    warn!(bot_id = %self.bot.id, user_id, error = %format!("{error:#}"), "failed to inspect overridden model capabilities");
                    None
                }
            }
        } else {
            None
        };
        let generation_command = match override_generation
            .or_else(|| plan.as_ref().and_then(RequestPlan::direct_generation))
        {
            Some(PlannedAction::GenerateImage) => Some("image"),
            Some(PlannedAction::GenerateSpeech | PlannedAction::GenerateAudio) => Some("speech"),
            Some(PlannedAction::GenerateMusic) => Some("music"),
            Some(PlannedAction::GenerateVideo) => Some("video"),
            _ => None,
        };
        if let Some(command) = generation_command {
            let generation_prompt = plan.as_ref().map_or_else(
                || text.trim().to_owned(),
                |plan| plan.effective_generation_prompt(&text, replied_text, telegram_quote),
            );
            drop(sender);
            let _ = progress_task.await;
            if let Err(error) = self
                .command(
                    &message,
                    mode,
                    user_id,
                    command,
                    &generation_prompt,
                    CommandOptions {
                        model_override: model_override.as_ref(),
                        guest_pending_id: guest_pending_id.as_deref(),
                        reply_context: None,
                    },
                )
                .await
            {
                error!(bot_id = %self.bot.id, user_id, error = %format!("{error:#}"), "planned generation command failed");
                if let Some(inline_id) = guest_pending_id.as_deref() {
                    self.edit_guest_error(inline_id, &error).await;
                } else {
                    self.respond(&message, mode, &rich::detailed_error(&error), None)
                        .await?;
                }
            }
            return Ok(());
        }
        if let Some(plan) = plan.as_ref() {
            if plan.action == PlannedAction::Refuse {
                drop(sender);
                let _ = progress_task.await;
                if let Some(inline_id) = guest_pending_id.as_deref() {
                    self.edit_guest_text(inline_id, &plan.refusal_message)
                        .await?;
                } else {
                    self.respond(&message, mode, &plan.refusal_message, None)
                        .await?;
                }
                return Ok(());
            }
        }
        let provider = self.search_provider().await?;
        let search_key = if capabilities.search {
            if provider == SearchProvider::Openrouter {
                self.store.credential(&self.bot.id, "openrouter").await?
            } else {
                self.store
                    .credential(&self.bot.id, provider.as_str())
                    .await?
            }
        } else {
            None
        };
        capabilities.search &= search_key.is_some();
        let history = self.store.history(&self.bot.id, &scope).await?;
        let media = if capabilities.media
            || capabilities.transcription
            || capabilities.image
            || capabilities.audio
            || capabilities.music
            || capabilities.video
        {
            self.collect_media(&message, &request_text).await?
        } else {
            Vec::new()
        };
        let (mut model_capability, mut selected) = if capabilities.media
            && media
                .iter()
                .any(|item| matches!(item, MediaInput::Video { .. }))
        {
            (
                "video_understanding",
                &settings.selected_video_understanding_model,
            )
        } else if capabilities.media
            && media
                .iter()
                .any(|item| matches!(item, MediaInput::Image { .. }))
        {
            (
                "image_understanding",
                &settings.selected_image_understanding_model,
            )
        } else {
            ("chat", &settings.selected_model)
        };
        if model_override.is_none()
            && model_capability == "chat"
            && capabilities.model_upgrade
            && plan
                .as_ref()
                .is_some_and(|plan| plan.skills.contains(&PlannedSkill::ModelUpgrade))
        {
            model_capability = "model_upgrade";
            selected = &settings.selected_upgrade_model;
        }
        let default_routing = ModelRouting::default();
        let override_routing = model_override.as_ref().map(|override_| ModelRouting {
            model_provider: override_.model_provider,
            ..ModelRouting::default()
        });
        let routing = override_routing.as_ref().unwrap_or_else(|| {
            settings
                .model_routing
                .get(model_capability)
                .unwrap_or(&default_routing)
        });
        let selected = model_override
            .as_ref()
            .map_or(selected.as_str(), |override_| override_.model.as_str());
        let model = match routing.model_provider {
            ModelProvider::Openrouter => self.config.resolved_model(model_capability, selected),
            ModelProvider::Aihub => self.config.resolved_aihub_model(selected),
        };
        if routing.model_provider == ModelProvider::Aihub {
            capabilities.web_fetch = false;
        }
        let api_key = self.model_api_key(routing.model_provider).await?;
        let image_routing = settings
            .model_routing
            .get("image_generation")
            .unwrap_or(&default_routing);
        let audio_routing = settings
            .model_routing
            .get("speech_generation")
            .or_else(|| settings.model_routing.get("audio_generation"))
            .unwrap_or(&default_routing);
        let music_routing = settings
            .model_routing
            .get("music_generation")
            .unwrap_or(&default_routing);
        let transcription_routing = settings
            .model_routing
            .get("transcription")
            .unwrap_or(&default_routing);
        let video_routing = settings
            .model_routing
            .get("video_generation")
            .unwrap_or(&default_routing);
        let image_key = self
            .optional_model_api_key(image_routing.model_provider, capabilities.image)
            .await?;
        let audio_key = self
            .optional_model_api_key(audio_routing.model_provider, capabilities.audio)
            .await?;
        let music_key = self
            .optional_model_api_key(music_routing.model_provider, capabilities.music)
            .await?;
        let transcription_key = self
            .optional_model_api_key(
                transcription_routing.model_provider,
                capabilities.transcription,
            )
            .await?;
        let video_key = self
            .optional_model_api_key(video_routing.model_provider, capabilities.video)
            .await?;
        capabilities.image &= !image_key.is_empty();
        capabilities.audio &= !audio_key.is_empty();
        capabilities.music &= !music_key.is_empty();
        capabilities.transcription &= !transcription_key.is_empty();
        capabilities.video &= !video_key.is_empty();
        let author = caller_name(&message);
        let contextual_text = format!("{author}: {request_text}{}", media_summary(&media));
        let mut instructions = self.store.effective_instructions(&self.bot.id).await?;
        instructions.push_str(&format!("\n\n# Current request context\nCurrent UTC date and time: {}\nBot: @{} (backend ID {})\nConversation mode: {:?}\nChat: {} (ID {})\nCaller: {} (ID {}, language {})\nTelegram message Unix time: {}\nEnabled capabilities and callable tools for this request: search={}, web_fetch={}, image_generation={}, speech_generation={}, music_generation={}, video_generation={}, media_understanding={}, transcription={}, file_delivery={}, model_upgrade={}. If a capability is true, never claim it is disabled; call its tool when the user explicitly requests that operation. Preserve the user's exact request; do not substitute a planner-written prompt. Format non-trivial responses for Telegram Rich Messages with descriptive headings, bold key terms and labels, and sparing italics for qualifications.", chrono::Utc::now().to_rfc3339(), self.username, self.bot.id, mode, message.chat.title.as_deref().unwrap_or("Private or untitled chat"), message.chat.id, author, user_id, message.guest_bot_caller_user.as_ref().or(message.from.as_ref()).and_then(|u| u.language_code.as_deref()).unwrap_or("unknown"), message.date, capabilities.search, capabilities.web_fetch, capabilities.image, capabilities.audio, capabilities.music, capabilities.video, capabilities.media, capabilities.transcription, capabilities.file, capabilities.model_upgrade));
        self.store
            .append_message(&self.bot.id, &scope, "user", &contextual_text)
            .await?;
        let session_id = format!("{}:{scope}", self.bot.id);
        let _ = sender.send(ProgressUpdate::step("Prepared conversation context"));
        let _ = sender.send(ProgressUpdate::step(format!(
            "Selected model: {}",
            model.id
        )));
        let mut result = self
            .openrouter
            .chat(ChatRequest {
                model: &model,
                system_prompt: &instructions,
                history: &history,
                user_message: &contextual_text,
                session_id: &session_id,
                media: &media,
                search: &self.search,
                search_provider: provider,
                search_api_key: search_key.as_deref(),
                api_key: &api_key,
                model_provider: routing.model_provider,
                capabilities: &capabilities,
                routing,
                tool_models: ToolModels {
                    image_generation: ToolModel {
                        model: &settings.selected_image_generation_model,
                        routing: image_routing,
                        api_key: &image_key,
                    },
                    audio_generation: ToolModel {
                        model: &settings.selected_audio_generation_model,
                        routing: audio_routing,
                        api_key: &audio_key,
                    },
                    music_generation: ToolModel {
                        model: &settings.selected_music_generation_model,
                        routing: music_routing,
                        api_key: &music_key,
                    },
                    transcription: ToolModel {
                        model: &settings.selected_transcription_model,
                        routing: transcription_routing,
                        api_key: &transcription_key,
                    },
                    video_generation: ToolModel {
                        model: &settings.selected_video_generation_model,
                        routing: video_routing,
                        api_key: &video_key,
                    },
                },
                generation_prompt: GenerationPromptContext {
                    current_request: &text,
                    replied_message: replied_text,
                    telegram_quote,
                    model: &settings.selected_planner_model,
                    fallback_model: &settings.selected_planner_fallback_model,
                    api_key: planner_key.as_deref(),
                    has_image: attachment_flags.0,
                    has_video: attachment_flags.1,
                    has_audio: attachment_flags.2,
                },
                progress: Some(sender.clone()),
            })
            .await;
        if let (Ok(answer), Some(plan)) = (&mut result, plan.as_ref()) {
            answer.apply_planned_delivery(plan, capabilities.file);
        }
        let _ = sender.send(ProgressUpdate::step(if result.is_ok() {
            "Request completed"
        } else {
            "Request failed"
        }));
        drop(sender);
        let _ = progress_task.await;
        match result {
            Ok(answer) => {
                self.store
                    .append_message(&self.bot.id, &scope, "assistant", &answer.text)
                    .await?;
                if mode == MessageMode::Guest {
                    let inline_id = guest_pending_id
                        .as_deref()
                        .context("Guest request has no pending result")?;
                    let delivery = if let Some(image) = answer.generated_images.into_iter().next() {
                        self.edit_guest_image(
                            inline_id,
                            image.bytes,
                            &image.media_type,
                            &image.model,
                            &image.prompt,
                        )
                        .await
                    } else if let Some(audio) = answer.generated_audio.into_iter().next() {
                        self.edit_guest_audio(
                            inline_id,
                            audio.bytes,
                            &audio.media_type,
                            &audio.model,
                            &audio.prompt,
                        )
                        .await
                    } else if let Some(video) = answer.generated_videos.first() {
                        self.edit_guest_video(inline_id, &video.url, &video.model, &video.prompt)
                            .await
                    } else if let Some(file) = answer.generated_files.first() {
                        self.edit_guest_document(inline_id, file.bytes.clone(), &file.filename)
                            .await
                    } else {
                        self.edit_guest_text(inline_id, &answer.text).await
                    };
                    if let Err(error) = delivery {
                        error!(bot_id = %self.bot.id, user_id, error = %format!("{error:#}"), "guest result delivery failed");
                        self.edit_guest_error(inline_id, &error).await;
                    }
                    if let Some(id) = answer.generation_id {
                        self.store
                            .audit(&self.bot.id, Some(user_id), "chat.complete", Some(&id))
                            .await?;
                    }
                    return Ok(());
                }
                self.respond(&message, mode, &answer.text, None).await?;
                for image in answer.generated_images {
                    self.send_photo_bytes(
                        message.chat.id,
                        message.message_thread_id,
                        &image.bytes,
                        &image.media_type,
                        Some(&generation_caption(&image.model, &image.prompt)),
                        Some(message.message_id),
                    )
                    .await?;
                }
                for audio in answer.generated_audio {
                    self.send_audio_bytes(
                        message.chat.id,
                        message.message_thread_id,
                        &audio.bytes,
                        &audio.media_type,
                        Some(&generation_caption(&audio.model, &audio.prompt)),
                        Some(message.message_id),
                    )
                    .await?;
                }
                for video in answer.generated_videos {
                    self.send_video_url(
                        message.chat.id,
                        message.message_thread_id,
                        &video.url,
                        Some(&generation_caption(&video.model, &video.prompt)),
                        Some(message.message_id),
                    )
                    .await?;
                }
                for file in answer.generated_files {
                    self.send_document_bytes(
                        message.chat.id,
                        message.message_thread_id,
                        &file.filename,
                        &file.bytes,
                        Some(message.message_id),
                    )
                    .await?;
                }
                for url in answer.media_urls {
                    self.send_photo_url(
                        message.chat.id,
                        message.message_thread_id,
                        &url,
                        None,
                        Some(message.message_id),
                    )
                    .await?;
                }
                if let Some(id) = answer.generation_id {
                    self.store
                        .audit(&self.bot.id, Some(user_id), "chat.complete", Some(&id))
                        .await?;
                }
            }
            Err(error) => {
                error!(bot_id = %self.bot.id, user_id, error = %format!("{error:#}"), "assistant request failed");
                if let Some(inline_id) = guest_pending_id.as_deref() {
                    self.edit_guest_error(inline_id, &error).await;
                } else {
                    self.respond(&message, mode, &rich::detailed_error(&error), None)
                        .await?;
                }
            }
        }
        Ok(())
    }

    async fn media_only_override_action(
        &self,
        override_: &MessageModelOverride,
    ) -> Result<Option<PlannedAction>> {
        let key = self.model_api_key(override_.model_provider).await?;
        let models = match override_.model_provider {
            ModelProvider::Openrouter => {
                self.catalog
                    .get_openrouter(
                        &self.client,
                        &self.config.openrouter.base_url,
                        &self.bot.id,
                        &key,
                    )
                    .await?
            }
            ModelProvider::Aihub => {
                self.catalog
                    .get_aihub(
                        &self.client,
                        &self.config.aihub.base_url,
                        &self.bot.id,
                        &key,
                    )
                    .await?
            }
        };
        let Some(model) = models.iter().find(|model| model.id == override_.model) else {
            return Ok(None);
        };
        if model.supports("chat") {
            return Ok(None);
        }
        let actions = [
            ("image_generation", PlannedAction::GenerateImage),
            ("speech_generation", PlannedAction::GenerateSpeech),
            ("music_generation", PlannedAction::GenerateMusic),
            ("video_generation", PlannedAction::GenerateVideo),
        ]
        .into_iter()
        .filter_map(|(capability, action)| model.supports(capability).then_some(action))
        .collect::<Vec<_>>();
        if actions.len() == 1 {
            Ok(actions.first().copied())
        } else {
            Ok(None)
        }
    }

    async fn command(
        &self,
        message: &Message,
        mode: MessageMode,
        user_id: u64,
        command: &str,
        arguments: &str,
        options: CommandOptions<'_>,
    ) -> Result<()> {
        let CommandOptions {
            model_override: outer_model_override,
            guest_pending_id,
            reply_context,
        } = options;
        let mut arguments = arguments;
        let inline_model_override = if outer_model_override.is_none()
            && let Some((nested_command, nested_arguments)) = parse_command(arguments)
            && nested_command == "model"
        {
            if !self.is_admin(user_id) {
                bail!("Administrator access is required for per-message model overrides");
            }
            let (override_, prompt) = parse_model_override(nested_arguments)?;
            arguments = prompt;
            Some(override_)
        } else {
            None
        };
        let model_override = outer_model_override.or(inline_model_override.as_ref());
        let contextual_arguments = reply_context
            .filter(|_| matches!(command, "search" | "image" | "music" | "video"))
            .map(|reply| with_reply_context(arguments, Some(reply)));
        let arguments = contextual_arguments.as_deref().unwrap_or(arguments);
        match command {
            "start" | "help" => self.respond(message, mode, HELP, None).await?,
            "new" => {
                let scope = scope_id(mode, message, user_id);
                self.store.clear_history(&self.bot.id, &scope).await?;
                self.respond(message, mode, "Conversation history cleared.", None)
                    .await?;
            }
            "model" => {
                let settings = self.store.settings(&self.bot.id).await?;
                let provider = settings
                    .model_routing
                    .get("chat")
                    .map(|routing| routing.model_provider)
                    .unwrap_or_default();
                self.respond(
                    message,
                    mode,
                    &format!(
                        "Current model: `{}` via `{}`",
                        settings.selected_model,
                        provider.as_str()
                    ),
                    None,
                )
                .await?;
            }
            "searchprovider" => {
                let provider = self.search_provider().await?;
                self.respond(
                    message,
                    mode,
                    &format!("Current web-search provider: `{}`", provider.as_str()),
                    None,
                )
                .await?;
            }
            "search" => {
                require_arguments(arguments, "-search <query>")?;
                let settings = self.store.settings(&self.bot.id).await?;
                if !settings.capabilities.search {
                    bail!("Search is disabled by an administrator");
                }
                let provider = self.search_provider().await?;
                let body = if provider == SearchProvider::Openrouter {
                    let key = self
                        .store
                        .credential(&self.bot.id, "openrouter")
                        .await?
                        .context("OpenRouter API key is not configured")?;
                    let selected_routing = settings
                        .model_routing
                        .get("chat")
                        .cloned()
                        .unwrap_or_default();
                    let (model, routing) =
                        if selected_routing.model_provider == ModelProvider::Openrouter {
                            (
                                self.config.resolved_model("chat", &settings.selected_model),
                                selected_routing,
                            )
                        } else {
                            (
                                self.config.resolved_model("chat", &self.bot.default_model),
                                ModelRouting::default(),
                            )
                        };
                    self.openrouter
                        .search(arguments, &model, &routing, &key)
                        .await?
                } else {
                    let key = self
                        .store
                        .credential(&self.bot.id, provider.as_str())
                        .await?
                        .context("Search API key is not configured")?;
                    let results = self.search.search(provider, arguments, &key).await?;
                    if results.is_empty() {
                        "No results found.".to_owned()
                    } else {
                        results
                            .iter()
                            .enumerate()
                            .map(|(i, r)| {
                                format!("{}. [{}]({})\n{}", i + 1, r.title, r.url, r.snippet)
                            })
                            .collect::<Vec<_>>()
                            .join("\n\n")
                    }
                };
                self.respond(message, mode, &body, None).await?;
            }
            "image" => {
                let settings = self.store.settings(&self.bot.id).await?;
                if !settings.capabilities.image {
                    bail!("Image generation is disabled by an administrator");
                }
                require_arguments(arguments, "/image <prompt>")?;
                let generation_model = model_override.map_or(
                    settings.selected_image_generation_model.as_str(),
                    |override_| override_.model.as_str(),
                );
                let routing = model_override.map_or_else(
                    || {
                        settings
                            .model_routing
                            .get("image_generation")
                            .cloned()
                            .unwrap_or_default()
                    },
                    |override_| ModelRouting {
                        model_provider: override_.model_provider,
                        ..ModelRouting::default()
                    },
                );
                if mode == MessageMode::Guest {
                    let inline_id = match guest_pending_id {
                        Some(id) => id.to_owned(),
                        None => {
                            self.answer_guest_pending(message, "Generating the requested image")
                                .await?
                        }
                    };
                    let (progress, progress_task) =
                        self.begin_guest_progress(&inline_id, "Parsed image-generation request");
                    let result = async {
                        let references = self.collect_media(message, arguments).await?;
                        let effective_prompt = self
                            .prepare_generation_prompt(arguments, &references, true)
                            .await?;
                        let _ = progress.send(ProgressUpdate::generation(
                            "image",
                            generation_model,
                            &effective_prompt,
                        ));
                        let key = self.model_api_key(routing.model_provider).await?;
                        self.openrouter
                            .generate_image_with_references(
                                &effective_prompt,
                                &references,
                                generation_model,
                                &routing,
                                &key,
                            )
                            .await
                    }
                    .await;
                    drop(progress);
                    let _ = progress_task.await;
                    match result {
                        Ok(mut images) if !images.is_empty() => {
                            let image = images.remove(0);
                            if let Err(error) = self
                                .edit_guest_image(
                                    &inline_id,
                                    image.bytes,
                                    &image.media_type,
                                    &image.model,
                                    &image.prompt,
                                )
                                .await
                            {
                                error!(bot_id = %self.bot.id, user_id, error = %format!("{error:#}"), "guest image delivery failed");
                                self.edit_guest_error(&inline_id, &error).await;
                            }
                        }
                        Ok(_) => {
                            self.edit_guest_error(
                                &inline_id,
                                &"OpenRouter returned no generated images",
                            )
                            .await
                        }
                        Err(error) => {
                            error!(bot_id = %self.bot.id, user_id, error = %format!("{error:#}"), "guest image generation failed");
                            self.edit_guest_error(&inline_id, &error).await;
                        }
                    }
                    return Ok(());
                }
                let (progress, progress_task) = self
                    .begin_chat_progress(message, "Parsed image-generation request")
                    .await?;
                self.send_action(
                    message.chat.id,
                    message.message_thread_id,
                    ChatAction::UploadPhoto,
                )
                .await;
                let _ = progress.send(ProgressUpdate::step("Read reference media"));
                let references = self.collect_media(message, arguments).await?;
                let effective_prompt = self
                    .prepare_generation_prompt(arguments, &references, true)
                    .await?;
                let key = self.model_api_key(routing.model_provider).await?;
                let _ = progress.send(ProgressUpdate::generation(
                    "image",
                    generation_model,
                    &effective_prompt,
                ));
                let result = self
                    .openrouter
                    .generate_image_with_references(
                        &effective_prompt,
                        &references,
                        generation_model,
                        &routing,
                        &key,
                    )
                    .await;
                let _ = progress.send(ProgressUpdate::step(if result.is_ok() {
                    "Image generation completed"
                } else {
                    "Image generation failed"
                }));
                drop(progress);
                let _ = progress_task.await;
                match result {
                    Ok(images) => {
                        for image in images {
                            self.send_photo_bytes(
                                message.chat.id,
                                message.message_thread_id,
                                &image.bytes,
                                &image.media_type,
                                Some(&generation_caption(&image.model, &image.prompt)),
                                Some(message.message_id),
                            )
                            .await?;
                        }
                    }
                    Err(error) => {
                        error!(bot_id = %self.bot.id, user_id, error = %format!("{error:#}"), "image generation failed");
                        self.respond(message, mode, &rich::detailed_error(&error), None)
                            .await?;
                    }
                }
            }
            "audio" | "speech" | "music" => {
                let settings = self.store.settings(&self.bot.id).await?;
                let is_music = command == "music";
                if is_music && !settings.capabilities.music {
                    bail!("Music generation is disabled by an administrator");
                }
                if !is_music && !settings.capabilities.audio {
                    bail!("Speech generation is disabled by an administrator");
                }
                require_arguments(
                    arguments,
                    if is_music {
                        "/music <prompt>"
                    } else {
                        "/speech <text>"
                    },
                )?;
                let generation_model = model_override.map_or(
                    if is_music {
                        settings.selected_music_generation_model.as_str()
                    } else {
                        settings.selected_audio_generation_model.as_str()
                    },
                    |override_| override_.model.as_str(),
                );
                let routing = model_override.map_or_else(
                    || {
                        settings
                            .model_routing
                            .get(if is_music {
                                "music_generation"
                            } else {
                                "speech_generation"
                            })
                            .or_else(|| {
                                (!is_music)
                                    .then(|| settings.model_routing.get("audio_generation"))
                                    .flatten()
                            })
                            .cloned()
                            .unwrap_or_default()
                    },
                    |override_| ModelRouting {
                        model_provider: override_.model_provider,
                        ..ModelRouting::default()
                    },
                );
                if mode == MessageMode::Guest {
                    let inline_id = match guest_pending_id {
                        Some(id) => id.to_owned(),
                        None => {
                            self.answer_guest_pending(
                                message,
                                if is_music {
                                    "Generating the requested music"
                                } else {
                                    "Generating the requested speech"
                                },
                            )
                            .await?
                        }
                    };
                    let (progress, progress_task) = self.begin_guest_progress(
                        &inline_id,
                        if is_music {
                            "Parsed music-generation request"
                        } else {
                            "Parsed speech-generation request"
                        },
                    );
                    let result = async {
                        let references = self.collect_media(message, arguments).await?;
                        let effective_prompt = if is_music {
                            self.prepare_generation_prompt(arguments, &references, false)
                                .await?
                        } else {
                            arguments.to_owned()
                        };
                        let _ = progress.send(ProgressUpdate::generation(
                            if is_music { "music" } else { "speech" },
                            generation_model,
                            &effective_prompt,
                        ));
                        let key = self.model_api_key(routing.model_provider).await?;
                        if is_music {
                            self.openrouter
                                .generate_music_with_references(
                                    &effective_prompt,
                                    &references,
                                    generation_model,
                                    &routing,
                                    &key,
                                )
                                .await
                        } else {
                            self.openrouter
                                .generate_speech_with_references(
                                    &effective_prompt,
                                    &references,
                                    generation_model,
                                    &routing,
                                    &key,
                                )
                                .await
                        }
                    }
                    .await;
                    drop(progress);
                    let _ = progress_task.await;
                    match result {
                        Ok(audio) => {
                            if let Err(error) = self
                                .edit_guest_audio(
                                    &inline_id,
                                    audio.bytes,
                                    &audio.media_type,
                                    &audio.model,
                                    &audio.prompt,
                                )
                                .await
                            {
                                error!(bot_id = %self.bot.id, user_id, error = %format!("{error:#}"), "guest audio delivery failed");
                                self.edit_guest_error(&inline_id, &error).await;
                            }
                        }
                        Err(error) => {
                            error!(bot_id = %self.bot.id, user_id, error = %format!("{error:#}"), "guest audio generation failed");
                            self.edit_guest_error(&inline_id, &error).await;
                        }
                    }
                    return Ok(());
                }
                let (progress, progress_task) = self
                    .begin_chat_progress(
                        message,
                        if is_music {
                            "Parsed music-generation request"
                        } else {
                            "Parsed speech-generation request"
                        },
                    )
                    .await?;
                self.send_action(
                    message.chat.id,
                    message.message_thread_id,
                    ChatAction::UploadVoice,
                )
                .await;
                let references = self.collect_media(message, arguments).await?;
                let effective_prompt = if is_music {
                    self.prepare_generation_prompt(arguments, &references, false)
                        .await?
                } else {
                    arguments.to_owned()
                };
                let key = self.model_api_key(routing.model_provider).await?;
                let _ = progress.send(ProgressUpdate::generation(
                    if is_music { "music" } else { "speech" },
                    generation_model,
                    &effective_prompt,
                ));
                let result = if is_music {
                    self.openrouter
                        .generate_music_with_references(
                            &effective_prompt,
                            &references,
                            generation_model,
                            &routing,
                            &key,
                        )
                        .await
                } else {
                    self.openrouter
                        .generate_speech_with_references(
                            &effective_prompt,
                            &references,
                            generation_model,
                            &routing,
                            &key,
                        )
                        .await
                };
                let _ = progress.send(ProgressUpdate::step(if result.is_ok() {
                    if is_music {
                        "Music generation completed"
                    } else {
                        "Speech generation completed"
                    }
                } else {
                    if is_music {
                        "Music generation failed"
                    } else {
                        "Speech generation failed"
                    }
                }));
                drop(progress);
                let _ = progress_task.await;
                match result {
                    Ok(audio) => {
                        self.send_audio_bytes(
                            message.chat.id,
                            message.message_thread_id,
                            &audio.bytes,
                            &audio.media_type,
                            Some(&generation_caption(&audio.model, &audio.prompt)),
                            Some(message.message_id),
                        )
                        .await?
                    }
                    Err(error) => {
                        error!(bot_id = %self.bot.id, user_id, error = %format!("{error:#}"), "audio generation failed");
                        self.respond(message, mode, &rich::detailed_error(&error), None)
                            .await?;
                    }
                }
            }
            "transcribe" => {
                let settings = self.store.settings(&self.bot.id).await?;
                if !settings.capabilities.transcription {
                    bail!("Transcription is disabled by an administrator");
                }
                if mode == MessageMode::Guest {
                    let inline_id = match guest_pending_id {
                        Some(id) => id.to_owned(),
                        None => {
                            self.answer_guest_pending(message, "Reading the audio")
                                .await?
                        }
                    };
                    let (progress, progress_task) =
                        self.begin_guest_progress(&inline_id, "Reading attached audio");
                    let result = async {
                        let media = self.collect_media(message, arguments).await?;
                        let _ = progress.send(ProgressUpdate::step("Transcribing audio"));
                        self.transcribe_media_inputs(&media).await
                    }
                    .await;
                    drop(progress);
                    let _ = progress_task.await;
                    match result {
                        Ok(transcript) => {
                            self.edit_guest_text(
                                &inline_id,
                                &format!("# Transcript\n\n{transcript}"),
                            )
                            .await?;
                        }
                        Err(error) => {
                            error!(bot_id = %self.bot.id, user_id, error = %format!("{error:#}"), "guest transcription failed");
                            self.edit_guest_error(&inline_id, &error).await;
                        }
                    }
                    return Ok(());
                }
                let media = self.collect_media(message, arguments).await?;
                let transcript = self.transcribe_media_inputs(&media).await?;
                self.respond(
                    message,
                    mode,
                    &format!("# Transcript\n\n{transcript}"),
                    None,
                )
                .await?;
            }
            "video" => {
                let settings = self.store.settings(&self.bot.id).await?;
                if !settings.capabilities.video {
                    bail!("Video generation is disabled by an administrator");
                }
                require_arguments(arguments, "/video <prompt>")?;
                let generation_model = model_override.map_or(
                    settings.selected_video_generation_model.as_str(),
                    |override_| override_.model.as_str(),
                );
                let routing = model_override.map_or_else(
                    || {
                        settings
                            .model_routing
                            .get("video_generation")
                            .cloned()
                            .unwrap_or_default()
                    },
                    |override_| ModelRouting {
                        model_provider: override_.model_provider,
                        ..ModelRouting::default()
                    },
                );
                if mode == MessageMode::Guest {
                    let inline_id = match guest_pending_id {
                        Some(id) => id.to_owned(),
                        None => {
                            self.answer_guest_pending(message, "Generating the requested video")
                                .await?
                        }
                    };
                    let (progress, progress_task) =
                        self.begin_guest_progress(&inline_id, "Parsed video-generation request");
                    let result = async {
                        let _ = progress.send(ProgressUpdate::generation(
                            "video",
                            generation_model,
                            arguments,
                        ));
                        let key = self.model_api_key(routing.model_provider).await?;
                        let references = self.collect_media(message, arguments).await?;
                        self.openrouter
                            .generate_video_with_references(
                                arguments,
                                &references,
                                generation_model,
                                &routing,
                                &key,
                            )
                            .await
                    }
                    .await;
                    drop(progress);
                    let _ = progress_task.await;
                    match result {
                        Ok(url) => {
                            if let Err(error) = self
                                .edit_guest_video(&inline_id, &url, generation_model, arguments)
                                .await
                            {
                                error!(bot_id = %self.bot.id, user_id, error = %format!("{error:#}"), "guest video delivery failed");
                                self.edit_guest_error(&inline_id, &error).await;
                            }
                        }
                        Err(error) => {
                            error!(bot_id = %self.bot.id, user_id, error = %format!("{error:#}"), "guest video generation failed");
                            self.edit_guest_error(&inline_id, &error).await;
                        }
                    }
                    return Ok(());
                }
                let (progress, progress_task) = self
                    .begin_chat_progress(message, "Parsed video-generation request")
                    .await?;
                self.send_action(
                    message.chat.id,
                    message.message_thread_id,
                    ChatAction::UploadVideo,
                )
                .await;
                let _ = progress.send(ProgressUpdate::step("Read reference media"));
                let references = self.collect_media(message, arguments).await?;
                let key = self.model_api_key(routing.model_provider).await?;
                let _ = progress.send(ProgressUpdate::generation(
                    "video",
                    generation_model,
                    arguments,
                ));
                let result = self
                    .openrouter
                    .generate_video_with_references(
                        arguments,
                        &references,
                        generation_model,
                        &routing,
                        &key,
                    )
                    .await;
                let _ = progress.send(ProgressUpdate::step(if result.is_ok() {
                    "Video generation completed"
                } else {
                    "Video generation failed"
                }));
                drop(progress);
                let _ = progress_task.await;
                match result {
                    Ok(url) => {
                        self.send_video_url(
                            message.chat.id,
                            message.message_thread_id,
                            &url,
                            Some(&generation_caption(generation_model, arguments)),
                            Some(message.message_id),
                        )
                        .await?
                    }
                    Err(error) => {
                        error!(bot_id = %self.bot.id, user_id, error = %format!("{error:#}"), "video generation failed");
                        self.respond(message, mode, &rich::detailed_error(&error), None)
                            .await?;
                    }
                }
            }
            "admin" if self.is_admin(user_id) => {
                if mode != MessageMode::Private {
                    self.respond(
                        message,
                        mode,
                        "For security, open the admin panel from this bot's private chat.",
                        None,
                    )
                    .await?;
                    return Ok(());
                }
                let keyboard = self.admin_keyboard().await?;
                self.respond(message, mode, "# Secure admin panel\n\nOpen the Telegram Mini App below. Every panel request is authenticated by Telegram and checked against this bot's immutable administrator list.", Some(keyboard)).await?;
            }
            "allow" if self.is_admin(user_id) => {
                let target = parse_user_id(arguments)?;
                self.store
                    .set_user_allowed(&self.bot.id, target, true, user_id)
                    .await?;
                self.respond(
                    message,
                    mode,
                    &format!("User `{target}` is now allowed."),
                    None,
                )
                .await?;
            }
            "deny" if self.is_admin(user_id) => {
                let target = parse_user_id(arguments)?;
                if self.is_admin(target) {
                    bail!("Configured administrators cannot be denied");
                }
                self.store
                    .set_user_allowed(&self.bot.id, target, false, user_id)
                    .await?;
                self.respond(
                    message,
                    mode,
                    &format!("User `{target}` is now denied."),
                    None,
                )
                .await?;
            }
            "allowed" if self.is_admin(user_id) => {
                let users = self.store.list_allowed_users(&self.bot.id).await?;
                self.respond(
                    message,
                    mode,
                    &format!(
                        "Allowed user IDs:\n{}",
                        users
                            .iter()
                            .map(|v| format!("- `{v}`"))
                            .collect::<Vec<_>>()
                            .join("\n")
                    ),
                    None,
                )
                .await?;
            }
            "admin" | "allow" | "deny" | "allowed" => {
                self.respond(message, mode, "Administrator access required.", None)
                    .await?
            }
            _ => {
                self.respond(
                    message,
                    mode,
                    "Unknown command. Use `/help` to see available commands.",
                    None,
                )
                .await?
            }
        }
        Ok(())
    }

    async fn admin_keyboard(&self) -> Result<InlineKeyboardMarkup> {
        let url = format!(
            "{}/admin/{}",
            self.config.server.public_url.trim_end_matches('/'),
            self.bot_user_id
        );
        let app = WebAppInfo::builder().url(url).build();
        let button = InlineKeyboardButton::builder()
            .text("Open secure admin panel")
            .web_app(app)
            .build();
        Ok(InlineKeyboardMarkup::builder()
            .inline_keyboard(vec![vec![button]])
            .build())
    }

    async fn respond(
        &self,
        message: &Message,
        mode: MessageMode,
        markdown: &str,
        keyboard: Option<InlineKeyboardMarkup>,
    ) -> Result<()> {
        if mode == MessageMode::Guest {
            let query_id = message
                .guest_query_id
                .as_ref()
                .context("Guest message has no guest_query_id")?;
            let content = rich::chunks(markdown)
                .into_iter()
                .next()
                .unwrap_or_default();
            let rich_message = InputRichMessage::builder().markdown(content).build();
            let input = InputRichMessageContent::builder()
                .rich_message(rich_message)
                .build();
            let article = InlineQueryResultArticle::builder()
                .id("answer")
                .title("AI assistant answer")
                .input_message_content(InputMessageContent::Rich(input))
                .build();
            let params = AnswerGuestQueryParams::builder()
                .guest_query_id(query_id.clone())
                .result(InlineQueryResult::Article(article))
                .build();
            self.telegram.answer_guest_query(&params).await?;
            return Ok(());
        }
        self.send_rich(
            message.chat.id,
            message.message_thread_id,
            markdown,
            Some(message.message_id),
            keyboard,
        )
        .await
    }

    async fn answer_guest_pending(&self, message: &Message, status: &str) -> Result<String> {
        let query_id = message
            .guest_query_id
            .as_ref()
            .context("Guest message has no guest_query_id")?;
        let rich_message = InputRichMessage::builder()
            .markdown(rich::to_telegram_markdown(&format!(
                "⏳ **{status}…**\n\nThis result will update automatically."
            )))
            .build();
        let input = InputRichMessageContent::builder()
            .rich_message(rich_message)
            .build();
        let article = InlineQueryResultArticle::builder()
            .id("generation")
            .title(status)
            .input_message_content(InputMessageContent::Rich(input))
            .build();
        let params = AnswerGuestQueryParams::builder()
            .guest_query_id(query_id.clone())
            .result(InlineQueryResult::Article(article))
            .build();
        Ok(self
            .telegram
            .answer_guest_query(&params)
            .await?
            .result
            .inline_message_id)
    }

    async fn edit_guest_image(
        &self,
        inline_message_id: &str,
        bytes: Vec<u8>,
        media_type: &str,
        model: &str,
        prompt: &str,
    ) -> Result<()> {
        let url = self.host_guest_media(bytes, media_type, None).await?;
        let media = InputMediaPhoto::builder()
            .media(FileUpload::from(url.clone()))
            .caption(generation_caption(model, prompt))
            .build();
        let params = EditMessageMediaParams::builder()
            .inline_message_id(inline_message_id)
            .media(InputMedia::Photo(media))
            .build();
        if let Err(error) = self.telegram.edit_message_media(&params).await {
            warn!(bot_id = %self.bot.id, %error, "photo delivery failed; retrying generated image as a document");
            let media = InputMediaDocument::builder()
                .media(FileUpload::from(url))
                .caption(generation_caption(model, prompt))
                .build();
            let params = EditMessageMediaParams::builder()
                .inline_message_id(inline_message_id)
                .media(InputMedia::Document(media))
                .build();
            self.telegram.edit_message_media(&params).await?;
        }
        Ok(())
    }

    async fn edit_guest_audio(
        &self,
        inline_message_id: &str,
        bytes: Vec<u8>,
        media_type: &str,
        model: &str,
        prompt: &str,
    ) -> Result<()> {
        let url = self.host_guest_media(bytes, media_type, None).await?;
        let media = InputMediaAudio::builder()
            .media(FileUpload::from(url))
            .caption(generation_caption(model, prompt))
            .build();
        let params = EditMessageMediaParams::builder()
            .inline_message_id(inline_message_id)
            .media(InputMedia::Audio(media))
            .build();
        self.telegram.edit_message_media(&params).await?;
        Ok(())
    }

    async fn edit_guest_video(
        &self,
        inline_message_id: &str,
        url: &str,
        model: &str,
        prompt: &str,
    ) -> Result<()> {
        let (bytes, media_type) = self.download_generated_asset(url, "video/mp4").await?;
        let url = self.host_guest_media(bytes, &media_type, None).await?;
        let media = InputMediaVideo::builder()
            .media(FileUpload::from(url.clone()))
            .caption(generation_caption(model, prompt))
            .supports_streaming(true)
            .build();
        let params = EditMessageMediaParams::builder()
            .inline_message_id(inline_message_id)
            .media(InputMedia::Video(media))
            .build();
        if let Err(error) = self.telegram.edit_message_media(&params).await {
            warn!(bot_id = %self.bot.id, %error, "video delivery failed; retrying generated video as a document");
            let media = InputMediaDocument::builder()
                .media(FileUpload::from(url))
                .caption(generation_caption(model, prompt))
                .build();
            let params = EditMessageMediaParams::builder()
                .inline_message_id(inline_message_id)
                .media(InputMedia::Document(media))
                .build();
            self.telegram.edit_message_media(&params).await?;
        }
        Ok(())
    }

    async fn edit_guest_document(
        &self,
        inline_message_id: &str,
        bytes: Vec<u8>,
        filename: &str,
    ) -> Result<()> {
        let url = self
            .host_guest_media(bytes, "application/octet-stream", Some(filename))
            .await?;
        let media = InputMediaDocument::builder()
            .media(FileUpload::from(url))
            .caption(format!("Generated file: {filename}"))
            .build();
        let params = EditMessageMediaParams::builder()
            .inline_message_id(inline_message_id)
            .media(InputMedia::Document(media))
            .build();
        self.telegram.edit_message_media(&params).await?;
        Ok(())
    }

    async fn host_guest_media(
        &self,
        bytes: Vec<u8>,
        media_type: &str,
        filename: Option<&str>,
    ) -> Result<String> {
        let token = if let Some(filename) = filename {
            crate::ephemeral_media::publish_named(bytes, media_type, filename)?
        } else {
            crate::ephemeral_media::publish(bytes, media_type)?
        };
        let url = format!(
            "{}/generated/{token}",
            self.config.server.public_url.trim_end_matches('/')
        );
        let response = self
            .client
            .get(&url)
            .header(reqwest::header::RANGE, "bytes=0-0")
            .timeout(Duration::from_secs(15))
            .send()
            .await
            .wrap_err("Generated-media public URL is unreachable from the backend")?;
        if !response.status().is_success() {
            bail!(
                "Generated-media public URL returned {}; check the reverse proxy for /generated/*",
                response.status()
            );
        }
        Ok(url)
    }

    async fn download_generated_asset(
        &self,
        raw_url: &str,
        fallback_media_type: &str,
    ) -> Result<(Vec<u8>, String)> {
        const MAX_BYTES: usize = 50 * 1024 * 1024;
        let parsed = url::Url::parse(raw_url).context("Generated media returned an invalid URL")?;
        if parsed.scheme() != "https" {
            bail!("Generated media URL must use HTTPS");
        }
        if parsed.host_str().is_none_or(generated_media_host_blocked) {
            bail!("Generated media URL has an invalid host");
        }
        let openrouter_host = url::Url::parse(&self.config.openrouter.base_url)
            .ok()
            .and_then(|url| url.host_str().map(str::to_owned));
        let is_openrouter_content = parsed.host_str() == openrouter_host.as_deref();
        let mut request = self.client.get(parsed);
        if is_openrouter_content {
            let api_key = self
                .store
                .credential(&self.bot.id, "openrouter")
                .await?
                .context("OpenRouter API key is required to download generated video content")?;
            request = request
                .bearer_auth(api_key)
                .header("X-OpenRouter-Title", &self.config.openrouter.app_name)
                .header("X-OpenRouter-Metadata", "enabled");
            if let Some(site_url) = &self.config.openrouter.site_url {
                request = request.header("HTTP-Referer", site_url);
            }
        }
        let mut response = request
            .send()
            .await
            .context("Failed to download generated media")?
            .error_for_status()
            .wrap_err_with(|| {
                if is_openrouter_content {
                    "Authenticated OpenRouter media download returned an error status"
                } else {
                    "Generated media download returned an error status"
                }
            })?;
        if response
            .content_length()
            .is_some_and(|length| length > MAX_BYTES as u64)
        {
            bail!("Generated media exceeds the 50 MiB Telegram delivery limit");
        }
        let mut media_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(|value| value.split(';').next().unwrap_or(value).trim().to_owned())
            .filter(|value| !value.is_empty() && value != "application/octet-stream")
            .unwrap_or_else(|| fallback_media_type.to_owned());
        let mut bytes = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .context("Generated media download failed")?
        {
            if bytes.len() + chunk.len() > MAX_BYTES {
                bail!("Generated media exceeds the 50 MiB Telegram delivery limit");
            }
            bytes.extend_from_slice(&chunk);
        }
        if bytes.is_empty() {
            bail!("Generated media download was empty");
        }
        if fallback_media_type.starts_with("video/") && !media_type.starts_with("video/") {
            media_type = if bytes.get(4..8) == Some(b"ftyp") {
                "video/mp4".to_owned()
            } else if bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
                "video/webm".to_owned()
            } else {
                bail!("Generated video URL returned unsupported content type {media_type}");
            };
        }
        Ok((bytes, media_type))
    }

    async fn edit_guest_text(&self, inline_message_id: &str, markdown: &str) -> Result<()> {
        let content = rich::chunks(markdown)
            .into_iter()
            .next()
            .unwrap_or_default();
        let rich_message = InputRichMessage::builder().markdown(content).build();
        let params = EditMessageTextParams::builder()
            .inline_message_id(inline_message_id)
            .rich_message(rich_message)
            .build();
        self.telegram.edit_message_text(&params).await?;
        Ok(())
    }

    async fn edit_guest_error(
        &self,
        inline_message_id: &str,
        error: &(dyn std::fmt::Display + Sync),
    ) {
        let detail = rich::detailed_error(error);
        let mut bounded = detail.chars().take(3_750).collect::<String>();
        if detail.chars().count() > 3_750 {
            bounded.push_str("\n…\n```");
        }
        let rich_message = InputRichMessage::builder()
            .markdown(rich::to_telegram_markdown(&bounded))
            .build();
        let params = EditMessageTextParams::builder()
            .inline_message_id(inline_message_id)
            .rich_message(rich_message)
            .build();
        if let Err(error) = self.telegram.edit_message_text(&params).await {
            warn!(bot_id = %self.bot.id, %error, "failed to update guest generation error");
        }
    }

    async fn send_rich(
        &self,
        chat_id: i64,
        thread_id: Option<i32>,
        markdown: &str,
        reply_to: Option<i32>,
        keyboard: Option<InlineKeyboardMarkup>,
    ) -> Result<()> {
        let chunks = rich::chunks(markdown);
        for (index, chunk) in chunks.iter().enumerate() {
            let rich_message = InputRichMessage::builder().markdown(chunk.clone()).build();
            let mut params = SendRichMessageParams::builder()
                .chat_id(chat_id)
                .rich_message(rich_message)
                .build();
            params.message_thread_id = thread_id;
            if index == 0 {
                params.reply_parameters = reply_to
                    .map(|message_id| ReplyParameters::builder().message_id(message_id).build());
                params.reply_markup = keyboard.clone().map(ReplyMarkup::InlineKeyboardMarkup);
            }
            self.telegram.send_rich_message(&params).await?;
        }
        Ok(())
    }

    async fn send_progress_message(&self, message: &Message, status: &str) -> Result<i32> {
        let rich_message = progress_rich_message(&[ProgressUpdate::step(status)]);
        let mut params = SendRichMessageParams::builder()
            .chat_id(message.chat.id)
            .rich_message(rich_message)
            .build();
        params.message_thread_id = message.message_thread_id;
        params.reply_parameters = Some(
            ReplyParameters::builder()
                .message_id(message.message_id)
                .build(),
        );
        Ok(self
            .telegram
            .send_rich_message(&params)
            .await?
            .result
            .message_id)
    }

    async fn begin_chat_progress(
        &self,
        message: &Message,
        status: &str,
    ) -> Result<(
        mpsc::UnboundedSender<ProgressUpdate>,
        tokio::task::JoinHandle<()>,
    )> {
        let message_id = self.send_progress_message(message, status).await?;
        let (sender, receiver) = mpsc::unbounded_channel();
        let runner = self.clone();
        let chat_id = message.chat.id;
        let task = tokio::spawn(async move {
            runner
                .update_progress_message(chat_id, message_id, receiver)
                .await;
        });
        let _ = sender.send(ProgressUpdate::step(status));
        Ok((sender, task))
    }

    fn begin_guest_progress(
        &self,
        inline_message_id: &str,
        status: &str,
    ) -> (
        mpsc::UnboundedSender<ProgressUpdate>,
        tokio::task::JoinHandle<()>,
    ) {
        let (sender, receiver) = mpsc::unbounded_channel();
        let runner = self.clone();
        let inline_message_id = inline_message_id.to_owned();
        let task = tokio::spawn(async move {
            runner
                .update_guest_progress(inline_message_id, receiver)
                .await;
        });
        let _ = sender.send(ProgressUpdate::step(status));
        (sender, task)
    }

    async fn update_progress_message(
        &self,
        chat_id: i64,
        message_id: i32,
        mut updates: mpsc::UnboundedReceiver<ProgressUpdate>,
    ) {
        let mut steps = Vec::new();
        let mut dirty = false;
        let mut ticker = tokio::time::interval(Duration::from_secs(1));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                update = updates.recv() => match update {
                    Some(update) => {
                        push_progress_step(&mut steps, update);
                        dirty = true;
                    },
                    None => break,
                },
                _ = ticker.tick(), if dirty => {
                    let params = EditMessageTextParams::builder()
                        .chat_id(chat_id)
                        .message_id(message_id)
                        .rich_message(progress_rich_message(&steps))
                        .build();
                    if let Err(error) = self.telegram.edit_message_text(&params).await {
                        if !message_not_modified(&error) {
                            warn!(bot_id = %self.bot.id, %error, "failed to update progress message");
                            break;
                        }
                    }
                    dirty = false;
                }
            }
        }
        if dirty {
            let params = EditMessageTextParams::builder()
                .chat_id(chat_id)
                .message_id(message_id)
                .rich_message(progress_rich_message(&steps))
                .build();
            if let Err(error) = self.telegram.edit_message_text(&params).await
                && !message_not_modified(&error)
            {
                warn!(bot_id = %self.bot.id, %error, "failed to complete progress message");
            }
        }
    }

    async fn update_guest_progress(
        &self,
        inline_message_id: String,
        mut updates: mpsc::UnboundedReceiver<ProgressUpdate>,
    ) {
        let mut steps = Vec::new();
        let mut dirty = false;
        let mut ticker = tokio::time::interval(Duration::from_secs(1));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                update = updates.recv() => match update {
                    Some(update) => {
                        push_progress_step(&mut steps, update);
                        dirty = true;
                    },
                    None => break,
                },
                _ = ticker.tick(), if dirty => {
                    let params = EditMessageTextParams::builder()
                        .inline_message_id(inline_message_id.clone())
                        .rich_message(progress_rich_message(&steps))
                        .build();
                    if let Err(error) = self.telegram.edit_message_text(&params).await {
                        if !message_not_modified(&error) {
                            warn!(bot_id = %self.bot.id, %error, "failed to update guest progress message");
                            break;
                        }
                    }
                    dirty = false;
                }
            }
        }
        if dirty {
            let params = EditMessageTextParams::builder()
                .inline_message_id(inline_message_id)
                .rich_message(progress_rich_message(&steps))
                .build();
            if let Err(error) = self.telegram.edit_message_text(&params).await {
                if !message_not_modified(&error) {
                    warn!(bot_id = %self.bot.id, %error, "failed to complete guest progress message");
                }
            }
        }
    }

    async fn send_action(&self, chat_id: i64, thread_id: Option<i32>, action: ChatAction) {
        let mut params = SendChatActionParams::builder()
            .chat_id(chat_id)
            .action(action)
            .build();
        params.message_thread_id = thread_id;
        if let Err(error) = self.telegram.send_chat_action(&params).await {
            warn!(bot_id = %self.bot.id, %error, "failed to send chat action");
        }
    }

    async fn send_photo_bytes(
        &self,
        chat_id: i64,
        thread_id: Option<i32>,
        bytes: &[u8],
        media_type: &str,
        caption: Option<&str>,
        reply_to: Option<i32>,
    ) -> Result<()> {
        let suffix = if media_type.contains("jpeg") {
            ".jpg"
        } else if media_type.contains("webp") {
            ".webp"
        } else if media_type.contains("svg") {
            ".svg"
        } else {
            ".png"
        };
        let file = TempFileBuilder::new()
            .prefix("teleforge-")
            .suffix(suffix)
            .tempfile()?;
        tokio::fs::write(file.path(), bytes).await?;
        if media_type.contains("svg") || media_type.contains("avif") {
            let mut params = SendDocumentParams::builder()
                .chat_id(chat_id)
                .document(FileUpload::from(file.path().to_path_buf()))
                .build();
            params.message_thread_id = thread_id;
            params.caption = caption.map(str::to_owned);
            params.reply_parameters = reply_to
                .map(|message_id| ReplyParameters::builder().message_id(message_id).build());
            self.telegram.send_document(&params).await?;
            return Ok(());
        }
        let mut params = SendPhotoParams::builder()
            .chat_id(chat_id)
            .photo(FileUpload::from(file.path().to_path_buf()))
            .build();
        params.message_thread_id = thread_id;
        params.caption = caption.map(str::to_owned);
        params.reply_parameters =
            reply_to.map(|message_id| ReplyParameters::builder().message_id(message_id).build());
        self.telegram.send_photo(&params).await?;
        Ok(())
    }

    async fn send_document_bytes(
        &self,
        chat_id: i64,
        thread_id: Option<i32>,
        filename: &str,
        bytes: &[u8],
        reply_to: Option<i32>,
    ) -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join(filename);
        tokio::fs::write(&path, bytes).await?;
        let mut params = SendDocumentParams::builder()
            .chat_id(chat_id)
            .document(FileUpload::from(path))
            .caption(format!("Generated file · {filename}"))
            .build();
        params.message_thread_id = thread_id;
        params.reply_parameters =
            reply_to.map(|message_id| ReplyParameters::builder().message_id(message_id).build());
        self.telegram.send_document(&params).await?;
        Ok(())
    }

    async fn send_photo_url(
        &self,
        chat_id: i64,
        thread_id: Option<i32>,
        url: &str,
        caption: Option<&str>,
        reply_to: Option<i32>,
    ) -> Result<()> {
        if url.starts_with("data:") {
            warn!(bot_id = %self.bot.id, "skipping data-URL image from chat completion; use /image for generated files");
            return Ok(());
        }
        let mut params = SendPhotoParams::builder()
            .chat_id(chat_id)
            .photo(FileUpload::from(url.to_owned()))
            .build();
        params.message_thread_id = thread_id;
        params.caption = caption.map(str::to_owned);
        params.reply_parameters =
            reply_to.map(|message_id| ReplyParameters::builder().message_id(message_id).build());
        self.telegram.send_photo(&params).await?;
        Ok(())
    }

    async fn send_audio_bytes(
        &self,
        chat_id: i64,
        thread_id: Option<i32>,
        bytes: &[u8],
        media_type: &str,
        caption: Option<&str>,
        reply_to: Option<i32>,
    ) -> Result<()> {
        let suffix = if media_type.contains("wav") {
            ".wav"
        } else if media_type.contains("pcm") {
            ".pcm"
        } else {
            ".mp3"
        };
        let file = TempFileBuilder::new()
            .prefix("teleforge-audio-")
            .suffix(suffix)
            .tempfile()?;
        tokio::fs::write(file.path(), bytes).await?;
        let mut params = SendAudioParams::builder()
            .chat_id(chat_id)
            .audio(FileUpload::from(file.path().to_path_buf()))
            .build();
        params.message_thread_id = thread_id;
        params.caption = caption.map(str::to_owned);
        params.reply_parameters =
            reply_to.map(|message_id| ReplyParameters::builder().message_id(message_id).build());
        self.telegram.send_audio(&params).await?;
        Ok(())
    }

    async fn send_video_url(
        &self,
        chat_id: i64,
        thread_id: Option<i32>,
        url: &str,
        caption: Option<&str>,
        reply_to: Option<i32>,
    ) -> Result<()> {
        let (bytes, media_type) = self.download_generated_asset(url, "video/mp4").await?;
        let suffix = match media_type.as_str() {
            "video/webm" => ".webm",
            "video/quicktime" => ".mov",
            _ => ".mp4",
        };
        let file = TempFileBuilder::new()
            .prefix("teleforge-video-")
            .suffix(suffix)
            .tempfile()?;
        tokio::fs::write(file.path(), bytes).await?;
        let mut params = SendVideoParams::builder()
            .chat_id(chat_id)
            .video(FileUpload::from(file.path().to_path_buf()))
            .build();
        params.message_thread_id = thread_id;
        params.caption = caption.map(str::to_owned);
        params.supports_streaming = Some(true);
        params.reply_parameters =
            reply_to.map(|message_id| ReplyParameters::builder().message_id(message_id).build());
        if let Err(error) = self.telegram.send_video(&params).await {
            warn!(bot_id = %self.bot.id, %error, "video upload failed; retrying generated video as a document");
            let mut params = SendDocumentParams::builder()
                .chat_id(chat_id)
                .document(FileUpload::from(file.path().to_path_buf()))
                .build();
            params.message_thread_id = thread_id;
            params.caption = caption.map(str::to_owned);
            params.reply_parameters = reply_to
                .map(|message_id| ReplyParameters::builder().message_id(message_id).build());
            self.telegram.send_document(&params).await?;
        }
        Ok(())
    }

    async fn collect_media(&self, message: &Message, text: &str) -> Result<Vec<MediaInput>> {
        let mut inputs = Vec::new();
        for source in [Some(message), message.reply_to_message.as_deref()]
            .into_iter()
            .flatten()
        {
            if let Some(photo) = source.photo.as_ref().and_then(|items| {
                items.iter().max_by_key(|item| {
                    item.file_size
                        .unwrap_or(u64::from(item.width) * u64::from(item.height))
                })
            }) {
                let data = self.download_telegram_file(&photo.file_id).await?;
                inputs.push(MediaInput::Image {
                    url: data_url("image/jpeg", &data),
                });
            }
            if let Some(video) = &source.video {
                let data = self.download_telegram_file(&video.file_id).await?;
                inputs.push(MediaInput::Video {
                    url: data_url(video.mime_type.as_deref().unwrap_or("video/mp4"), &data),
                });
            }
            if let Some(video) = &source.video_note {
                let data = self.download_telegram_file(&video.file_id).await?;
                inputs.push(MediaInput::Video {
                    url: data_url("video/mp4", &data),
                });
            }
            if let Some(voice) = &source.voice {
                let data = self.download_telegram_file(&voice.file_id).await?;
                inputs.push(MediaInput::Audio {
                    data: STANDARD.encode(data),
                    format: audio_format(voice.mime_type.as_deref(), None).into(),
                });
            }
            if let Some(audio) = &source.audio {
                let data = self.download_telegram_file(&audio.file_id).await?;
                inputs.push(MediaInput::Audio {
                    data: STANDARD.encode(data),
                    format: audio_format(audio.mime_type.as_deref(), audio.file_name.as_deref())
                        .into(),
                });
            }
            if let Some(document) = &source.document {
                let mime = document.mime_type.as_deref().unwrap_or_default();
                if mime.starts_with("image/")
                    || mime.starts_with("video/")
                    || mime.starts_with("audio/")
                {
                    let data = self.download_telegram_file(&document.file_id).await?;
                    if mime.starts_with("image/") {
                        inputs.push(MediaInput::Image {
                            url: data_url(mime, &data),
                        });
                    } else if mime.starts_with("video/") {
                        inputs.push(MediaInput::Video {
                            url: data_url(mime, &data),
                        });
                    } else {
                        inputs.push(MediaInput::Audio {
                            data: STANDARD.encode(data),
                            format: audio_format(Some(mime), document.file_name.as_deref()).into(),
                        });
                    }
                }
            }
        }
        if let Some(source) = message.external_reply.as_deref() {
            if let Some(photo) = source.photo.as_ref().and_then(|items| {
                items.iter().max_by_key(|item| {
                    item.file_size
                        .unwrap_or(u64::from(item.width) * u64::from(item.height))
                })
            }) {
                let data = self.download_telegram_file(&photo.file_id).await?;
                inputs.push(MediaInput::Image {
                    url: data_url("image/jpeg", &data),
                });
            }
            if let Some(video) = &source.video {
                let data = self.download_telegram_file(&video.file_id).await?;
                inputs.push(MediaInput::Video {
                    url: data_url(video.mime_type.as_deref().unwrap_or("video/mp4"), &data),
                });
            }
            if let Some(video) = &source.video_note {
                let data = self.download_telegram_file(&video.file_id).await?;
                inputs.push(MediaInput::Video {
                    url: data_url("video/mp4", &data),
                });
            }
            if let Some(voice) = &source.voice {
                let data = self.download_telegram_file(&voice.file_id).await?;
                inputs.push(MediaInput::Audio {
                    data: STANDARD.encode(data),
                    format: audio_format(voice.mime_type.as_deref(), None).into(),
                });
            }
            if let Some(audio) = &source.audio {
                let data = self.download_telegram_file(&audio.file_id).await?;
                inputs.push(MediaInput::Audio {
                    data: STANDARD.encode(data),
                    format: audio_format(audio.mime_type.as_deref(), audio.file_name.as_deref())
                        .into(),
                });
            }
            if let Some(document) = &source.document {
                let mime = document.mime_type.as_deref().unwrap_or_default();
                if mime.starts_with("image/")
                    || mime.starts_with("video/")
                    || mime.starts_with("audio/")
                {
                    let data = self.download_telegram_file(&document.file_id).await?;
                    if mime.starts_with("image/") {
                        inputs.push(MediaInput::Image {
                            url: data_url(mime, &data),
                        });
                    } else if mime.starts_with("video/") {
                        inputs.push(MediaInput::Video {
                            url: data_url(mime, &data),
                        });
                    } else {
                        inputs.push(MediaInput::Audio {
                            data: STANDARD.encode(data),
                            format: audio_format(Some(mime), document.file_name.as_deref()).into(),
                        });
                    }
                }
            }
        }
        if let Some(url) = youtube_url(text) {
            inputs.push(MediaInput::Video { url });
        }
        Ok(inputs)
    }

    async fn prepare_generation_prompt(
        &self,
        prompt: &str,
        media: &[MediaInput],
        require_non_image_description: bool,
    ) -> Result<String> {
        let settings = self.store.settings(&self.bot.id).await?;
        let mut supplements = Vec::new();
        let videos = media
            .iter()
            .filter(|item| matches!(item, MediaInput::Video { .. }))
            .cloned()
            .collect::<Vec<_>>();
        if !videos.is_empty() {
            if !settings.capabilities.media {
                if require_non_image_description {
                    bail!(
                        "Video understanding is disabled; it is required for video-to-image transformation"
                    );
                }
            } else {
                let routing = settings
                    .model_routing
                    .get("video_understanding")
                    .cloned()
                    .unwrap_or_default();
                let model_id = &settings.selected_video_understanding_model;
                let model = match routing.model_provider {
                    ModelProvider::Openrouter => {
                        self.config.resolved_model("video_understanding", model_id)
                    }
                    ModelProvider::Aihub => self.config.resolved_aihub_model(model_id),
                };
                let key = self.model_api_key(routing.model_provider).await?;
                let description = self
                    .openrouter
                    .describe_generation_media(&model, &videos, &routing, &key)
                    .await?;
                supplements.push(format!("Reference video description:\n{description}"));
            }
        }

        let audio = media
            .iter()
            .filter_map(|item| match item {
                MediaInput::Audio { data, format } => Some((data, format)),
                _ => None,
            })
            .collect::<Vec<_>>();
        if !audio.is_empty() {
            if !settings.capabilities.transcription {
                if require_non_image_description {
                    bail!(
                        "Transcription is disabled; it is required for audio-to-image transformation"
                    );
                }
            } else {
                let routing = settings
                    .model_routing
                    .get("transcription")
                    .cloned()
                    .unwrap_or_default();
                let key = self.model_api_key(routing.model_provider).await?;
                for (index, (data, format)) in audio.into_iter().enumerate() {
                    let transcript = self
                        .openrouter
                        .transcribe_audio(
                            data,
                            format,
                            None,
                            &settings.selected_transcription_model,
                            &routing,
                            &key,
                        )
                        .await?;
                    supplements.push(format!(
                        "Reference audio {} transcript:\n{transcript}",
                        index + 1
                    ));
                }
            }
        }
        if supplements.is_empty() {
            return Ok(prompt.to_owned());
        }
        Ok(format!(
            "Original user instruction (preserve exactly):\n{prompt}\n\n{}",
            supplements.join("\n\n")
        ))
    }

    async fn transcribe_media_inputs(&self, media: &[MediaInput]) -> Result<String> {
        let settings = self.store.settings(&self.bot.id).await?;
        let routing = settings
            .model_routing
            .get("transcription")
            .cloned()
            .unwrap_or_default();
        let key = self.model_api_key(routing.model_provider).await?;
        let mut transcripts = Vec::new();
        for input in media {
            if let MediaInput::Audio { data, format } = input {
                transcripts.push(
                    self.openrouter
                        .transcribe_audio(
                            data,
                            format,
                            None,
                            &settings.selected_transcription_model,
                            &routing,
                            &key,
                        )
                        .await?,
                );
            }
        }
        if transcripts.is_empty() {
            bail!("Reply to or attach a Telegram voice note or audio file");
        }
        Ok(transcripts.join("\n\n"))
    }

    async fn download_telegram_file(&self, file_id: &str) -> Result<Vec<u8>> {
        let file = self
            .telegram
            .get_file(&GetFileParams::builder().file_id(file_id).build())
            .await
            .context("Telegram getFile failed")?
            .result;
        let limit = self.config.server.max_input_media_bytes;
        if file.file_size.is_some_and(|size| size > limit as u64) {
            bail!("Telegram attachment exceeds the configured media limit");
        }
        let path = file.file_path.context("Telegram returned no file path")?;
        let url = format!(
            "https://api.telegram.org/file/bot{}/{}",
            self.bot.token, path
        );
        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|_| eyre::eyre!("Telegram file download failed"))?;
        if !response.status().is_success() {
            bail!("Telegram file download failed");
        }
        if response
            .content_length()
            .is_some_and(|size| size > limit as u64)
        {
            bail!("Telegram attachment exceeds the configured media limit");
        }
        let data = response
            .bytes()
            .await
            .map_err(|_| eyre::eyre!("Telegram file download failed"))?;
        if data.len() > limit {
            bail!("Telegram attachment exceeds the configured media limit");
        }
        Ok(data.to_vec())
    }

    async fn is_allowed(&self, user_id: u64, chat_id: i64) -> Result<bool> {
        if self.is_admin(user_id)
            || self.bot.access.allow_everyone
            || self.bot.allowed_chat_ids.contains(&chat_id)
        {
            return Ok(true);
        }
        Ok(self
            .store
            .user_allowed(&self.bot.id, user_id)
            .await?
            .unwrap_or(false))
    }
    fn is_admin(&self, user_id: u64) -> bool {
        self.bot.admin_user_ids.contains(&user_id)
    }
    fn mode_enabled(&self, mode: MessageMode) -> bool {
        match mode {
            MessageMode::Private => self.bot.access.private_messages,
            MessageMode::Group => self.bot.access.group_chats,
            MessageMode::Guest => self.bot.access.guest_messages,
        }
    }
    fn addressed_to_bot(&self, message: &Message, text: &str) -> bool {
        text.to_ascii_lowercase()
            .contains(&format!("@{}", self.username.to_ascii_lowercase()))
            || message
                .reply_to_message
                .as_ref()
                .and_then(|m| m.from.as_ref())
                .is_some_and(|u| u.id == self.bot_user_id)
    }
    fn strip_address(&self, text: &str) -> String {
        text.replace(&format!("@{}", self.username), "")
            .replace(&format!("@{}", self.username.to_ascii_lowercase()), "")
            .trim()
            .to_owned()
    }
    async fn search_provider(&self) -> Result<SearchProvider> {
        Ok(self
            .store
            .selected_search_provider(&self.bot.id)
            .await?
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| self.search.default_provider()))
    }

    async fn model_api_key(&self, provider: ModelProvider) -> Result<String> {
        let name = match provider {
            ModelProvider::Openrouter => "OpenRouter",
            ModelProvider::Aihub => "AI Hub",
        };
        self.store
            .credential(&self.bot.id, provider.as_str())
            .await?
            .wrap_err_with(|| format!("{name} API key is not configured"))
    }

    async fn optional_model_api_key(
        &self,
        provider: ModelProvider,
        required: bool,
    ) -> Result<String> {
        if required {
            Ok(self
                .store
                .credential(&self.bot.id, provider.as_str())
                .await?
                .unwrap_or_default())
        } else {
            Ok(String::new())
        }
    }
}

fn parse_command(text: &str) -> Option<(String, &str)> {
    let text = text.strip_prefix('/').or_else(|| text.strip_prefix('-'))?;
    let (head, rest) = text.split_once(char::is_whitespace).unwrap_or((text, ""));
    Some((
        head.split('@').next().unwrap_or(head).to_ascii_lowercase(),
        rest.trim(),
    ))
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MessageModelOverride {
    model_provider: ModelProvider,
    model: String,
}

fn parse_model_override(arguments: &str) -> Result<(MessageModelOverride, &str)> {
    let (specification, prompt) = arguments
        .split_once(char::is_whitespace)
        .context("Expected `-model [openrouter:|aihub:]<model-id> <request>`")?;
    let prompt = prompt.trim();
    if prompt.is_empty() {
        bail!("Expected a request after the model ID");
    }
    let (model_provider, model) = if let Some(model) = specification.strip_prefix("openrouter:") {
        (ModelProvider::Openrouter, model)
    } else if let Some(model) = specification.strip_prefix("aihub:") {
        (ModelProvider::Aihub, model)
    } else {
        (ModelProvider::Openrouter, specification)
    };
    if model.is_empty()
        || model.len() > 200
        || !model.chars().all(|character| {
            character.is_ascii_alphanumeric()
                || matches!(character, '/' | ':' | '.' | '-' | '_' | '~')
        })
    {
        bail!("Invalid model ID");
    }
    Ok((
        MessageModelOverride {
            model_provider,
            model: model.to_owned(),
        },
        prompt,
    ))
}

fn generation_caption(model: &str, prompt: &str) -> String {
    const MAX_PROMPT_CHARS: usize = 850;
    let mut prompt = prompt
        .trim()
        .chars()
        .take(MAX_PROMPT_CHARS)
        .collect::<String>();
    if prompt.chars().count() == MAX_PROMPT_CHARS {
        prompt.push('…');
    }
    format!("Model: {model}\nPrompt: {prompt}")
}

fn push_progress_step(steps: &mut Vec<ProgressUpdate>, update: ProgressUpdate) {
    if steps.last() == Some(&update) {
        return;
    }
    steps.push(update);
    if steps.len() > 12 {
        steps.remove(0);
    }
}

fn message_not_modified(error: &impl std::fmt::Display) -> bool {
    error.to_string().contains("message is not modified")
}

fn generated_media_host_blocked(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost")
        || host.ends_with(".localhost")
        || host.ends_with(".local")
    {
        return true;
    }
    host.parse::<std::net::IpAddr>()
        .is_ok_and(|address| match address {
            std::net::IpAddr::V4(address) => {
                address.is_private()
                    || address.is_loopback()
                    || address.is_link_local()
                    || address.is_unspecified()
                    || address.is_broadcast()
            }
            std::net::IpAddr::V6(address) => {
                address.is_loopback()
                    || address.is_unspecified()
                    || address.is_unique_local()
                    || address.is_unicast_link_local()
            }
        })
}

fn progress_rich_message(steps: &[ProgressUpdate]) -> InputRichMessage {
    let mut markdown = String::from("## Processing\n\n");
    if steps.is_empty() {
        markdown.push_str("- **Current:** Starting");
    }
    for (index, update) in steps.iter().enumerate() {
        let is_current = index + 1 == steps.len();
        let marker = if is_current { "Current" } else { "Done" };
        match update {
            ProgressUpdate::Step(status) => {
                markdown.push_str(&format!("- **{marker}:** {}\n", rich::escape_text(status)));
            }
            ProgressUpdate::Generation {
                kind,
                model,
                prompt,
            } => {
                let prompt = prompt
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
                    .chars()
                    .take(1_500)
                    .collect::<String>();
                markdown.push_str(&format!(
                    "- **{marker}:** Generating {}\n  - **Model:** `{}`\n  - **Prompt:** *{}*\n",
                    rich::escape_text(kind),
                    rich::escape_text(model),
                    rich::escape_text(&prompt)
                ));
            }
        }
    }
    InputRichMessage::builder()
        .markdown(rich::to_telegram_markdown(&markdown))
        .build()
}

fn parse_user_id(value: &str) -> Result<u64> {
    value
        .trim()
        .parse()
        .context("Expected a numeric Telegram user ID")
}
fn require_arguments<'a>(value: &'a str, usage: &str) -> Result<&'a str> {
    if value.trim().is_empty() {
        bail!("Usage: {usage}");
    }
    Ok(value)
}
fn scope_id(mode: MessageMode, message: &Message, user_id: u64) -> String {
    match mode {
        MessageMode::Private => format!("pm:{user_id}"),
        MessageMode::Group => format!(
            "chat:{}:thread:{}",
            message.chat.id,
            message.message_thread_id.unwrap_or_default()
        ),
        MessageMode::Guest => format!("guest:{user_id}:chat:{}", message.chat.id),
    }
}

fn caller_name(message: &Message) -> String {
    let user = message
        .guest_bot_caller_user
        .as_ref()
        .or(message.from.as_ref());
    match user {
        Some(user) => format!(
            "{}{}",
            user.first_name,
            user.username
                .as_ref()
                .map(|name| format!(" (@{name})"))
                .unwrap_or_default()
        ),
        None => "Unknown caller".into(),
    }
}

fn reply_context(message: &Message) -> Option<String> {
    let replied = message.reply_to_message.as_deref();
    let quoted = message
        .quote
        .as_deref()
        .map(|quote| quote.text.trim())
        .filter(|text| !text.is_empty());
    if replied.is_none() && quoted.is_none() && message.external_reply.is_none() {
        return None;
    }

    let mut context = String::from("[Replied message context]");
    if let Some(replied) = replied {
        context.push_str("\nAuthor: ");
        context.push_str(&caller_name(replied));
        if let Some(content) = replied
            .text
            .as_deref()
            .or(replied.caption.as_deref())
            .map(str::trim)
            .filter(|content| !content.is_empty())
        {
            context.push_str("\nContent:\n");
            context.push_str(content);
        }
        let media = message_media_labels(replied);
        if !media.is_empty() {
            context.push_str("\nAttachments: ");
            context.push_str(&media.join(", "));
        }
    } else {
        context.push_str("\nAuthor: unavailable from Telegram");
    }
    if let Some(quoted) = quoted {
        let duplicate = replied.is_some_and(|replied| {
            replied.text.as_deref() == Some(quoted) || replied.caption.as_deref() == Some(quoted)
        });
        if !duplicate {
            if message
                .quote
                .as_deref()
                .and_then(|quote| quote.is_manual)
                .unwrap_or(false)
            {
                context.push_str("\nQuoted excerpt selected by the caller (primary focus):\n");
            } else {
                context.push_str("\nTelegram quoted excerpt (primary focus):\n");
            }
            context.push_str(quoted);
        }
    }
    if replied.is_none()
        && let Some(external) = message.external_reply.as_deref()
    {
        let media = external_reply_media_labels(external);
        if !media.is_empty() {
            context.push_str("\nAttachments: ");
            context.push_str(&media.join(", "));
        } else if quoted.is_none() {
            context.push_str("\nContent: Telegram supplied only external-reply metadata.");
        }
    }
    Some(context)
}

fn replied_message_text(message: &Message) -> Option<&str> {
    message
        .reply_to_message
        .as_deref()
        .and_then(|reply| reply.text.as_deref().or(reply.caption.as_deref()))
        .map(str::trim)
        .filter(|text| !text.is_empty())
}

fn telegram_quote_text(message: &Message) -> Option<&str> {
    message
        .quote
        .as_deref()
        .map(|quote| quote.text.trim())
        .filter(|text| !text.is_empty())
}

fn with_reply_context(text: &str, reply: Option<&str>) -> String {
    reply.map_or_else(|| text.to_owned(), |reply| format!("{text}\n\n{reply}"))
}

fn message_media_labels(message: &Message) -> Vec<&'static str> {
    let mut labels = Vec::new();
    if message.photo.is_some() {
        labels.push("image");
    }
    if message.video.is_some() || message.video_note.is_some() {
        labels.push("video");
    }
    if message.voice.is_some() || message.audio.is_some() {
        labels.push("audio");
    }
    if let Some(mime) = message
        .document
        .as_ref()
        .and_then(|document| document.mime_type.as_deref())
    {
        if mime.starts_with("image/") {
            labels.push("image document");
        } else if mime.starts_with("video/") {
            labels.push("video document");
        } else if mime.starts_with("audio/") {
            labels.push("audio document");
        } else {
            labels.push("document");
        }
    }
    labels
}

fn external_reply_media_labels(
    reply: &frankenstein::types::ExternalReplyInfo,
) -> Vec<&'static str> {
    let mut labels = Vec::new();
    if reply.photo.is_some() {
        labels.push("image");
    }
    if reply.video.is_some() || reply.video_note.is_some() {
        labels.push("video");
    }
    if reply.voice.is_some() || reply.audio.is_some() {
        labels.push("audio");
    }
    if let Some(mime) = reply
        .document
        .as_ref()
        .and_then(|document| document.mime_type.as_deref())
    {
        if mime.starts_with("image/") {
            labels.push("image document");
        } else if mime.starts_with("video/") {
            labels.push("video document");
        } else if mime.starts_with("audio/") {
            labels.push("audio document");
        } else {
            labels.push("document");
        }
    }
    labels
}

fn message_has_media(message: &Message) -> bool {
    message.photo.is_some()
        || message.video.is_some()
        || message.video_note.is_some()
        || message.voice.is_some()
        || message.audio.is_some()
        || message.document.as_ref().is_some_and(|document| {
            document.mime_type.as_deref().is_some_and(|mime| {
                mime.starts_with("image/")
                    || mime.starts_with("video/")
                    || mime.starts_with("audio/")
            })
        })
}

fn external_reply_has_media(reply: &frankenstein::types::ExternalReplyInfo) -> bool {
    reply.photo.is_some()
        || reply.video.is_some()
        || reply.video_note.is_some()
        || reply.voice.is_some()
        || reply.audio.is_some()
        || reply.document.as_ref().is_some_and(|document| {
            document.mime_type.as_deref().is_some_and(|mime| {
                mime.starts_with("image/")
                    || mime.starts_with("video/")
                    || mime.starts_with("audio/")
            })
        })
}

fn attachment_flags(message: &Message) -> (bool, bool, bool) {
    let sources = [Some(message), message.reply_to_message.as_deref()];
    let external = message.external_reply.as_deref();
    let image = sources.iter().flatten().any(|item| {
        item.photo.is_some()
            || item.document.as_ref().is_some_and(|document| {
                document
                    .mime_type
                    .as_deref()
                    .is_some_and(|mime| mime.starts_with("image/"))
            })
    }) || external.is_some_and(|item| {
        item.photo.is_some()
            || item.document.as_ref().is_some_and(|document| {
                document
                    .mime_type
                    .as_deref()
                    .is_some_and(|mime| mime.starts_with("image/"))
            })
    });
    let video = sources.iter().flatten().any(|item| {
        item.video.is_some()
            || item.video_note.is_some()
            || item.document.as_ref().is_some_and(|document| {
                document
                    .mime_type
                    .as_deref()
                    .is_some_and(|mime| mime.starts_with("video/"))
            })
    }) || external.is_some_and(|item| {
        item.video.is_some()
            || item.video_note.is_some()
            || item.document.as_ref().is_some_and(|document| {
                document
                    .mime_type
                    .as_deref()
                    .is_some_and(|mime| mime.starts_with("video/"))
            })
    });
    let audio = sources.iter().flatten().any(|item| {
        item.voice.is_some()
            || item.audio.is_some()
            || item.document.as_ref().is_some_and(|document| {
                document
                    .mime_type
                    .as_deref()
                    .is_some_and(|mime| mime.starts_with("audio/"))
            })
    }) || external.is_some_and(|item| {
        item.voice.is_some()
            || item.audio.is_some()
            || item.document.as_ref().is_some_and(|document| {
                document
                    .mime_type
                    .as_deref()
                    .is_some_and(|mime| mime.starts_with("audio/"))
            })
    });
    (image, video, audio)
}

fn default_media_prompt(message: &Message) -> &'static str {
    let sources = [Some(message), message.reply_to_message.as_deref()];
    if sources.iter().flatten().any(|item| {
        item.voice.is_some()
            || item.audio.is_some()
            || item
                .document
                .as_ref()
                .and_then(|document| document.mime_type.as_deref())
                .is_some_and(|mime| mime.starts_with("audio/"))
    }) || message.external_reply.as_deref().is_some_and(|item| {
        item.voice.is_some()
            || item.audio.is_some()
            || item.document.as_ref().is_some_and(|document| {
                document
                    .mime_type
                    .as_deref()
                    .is_some_and(|mime| mime.starts_with("audio/"))
            })
    }) {
        "Transcribe and summarize the attached audio."
    } else {
        "Describe and analyze the attached image or video."
    }
}

fn media_summary(media: &[MediaInput]) -> String {
    if media.is_empty() {
        return String::new();
    }
    let labels = media
        .iter()
        .map(|item| match item {
            MediaInput::Image { .. } => "image",
            MediaInput::Video { .. } => "video",
            MediaInput::Audio { .. } => "audio",
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("\n[Attached media for this turn: {labels}]")
}

fn data_url(mime: &str, bytes: &[u8]) -> String {
    format!("data:{mime};base64,{}", STANDARD.encode(bytes))
}

fn audio_format<'a>(mime: Option<&'a str>, file_name: Option<&'a str>) -> &'a str {
    let value = mime.or(file_name).unwrap_or("ogg").to_ascii_lowercase();
    if value.contains("wav") {
        "wav"
    } else if value.contains("flac") {
        "flac"
    } else if value.contains("mp3") || value.contains("mpeg") {
        "mp3"
    } else if value.contains("m4a") || value.contains("mp4") {
        "m4a"
    } else if value.contains("aac") {
        "aac"
    } else {
        "ogg"
    }
}

fn youtube_url(text: &str) -> Option<String> {
    text.split_whitespace().find_map(|word| {
        let candidate = word.trim_matches(|character: char| {
            matches!(
                character,
                '(' | ')' | '[' | ']' | '<' | '>' | ',' | '.' | ';'
            )
        });
        let parsed = url::Url::parse(candidate).ok()?;
        let host = parsed.host_str()?.trim_start_matches("www.");
        matches!(host, "youtube.com" | "m.youtube.com" | "youtu.be").then(|| parsed.into())
    })
}

const HELP: &str = r#"# AI Assistant

Ask me a question directly. I can use live web search when the selected model decides it is needed.

- `/new` — clear this conversation
- `/model` — show the active model
- `-model [openrouter:|aihub:]<model-id> <request>` — use a model for one request (administrators only)
- `/search <query>` — force a live web search
- `/searchprovider` — show the active search provider
- `/image <prompt>` — generate an image
- `/speech <text>` — generate spoken audio (`/audio` is an alias)
- `/music <prompt>` — generate music or other non-speech audio
- `/transcribe` — transcribe an attached/replied-to voice note or audio file
- `/video <prompt>` — generate a video
- `/admin` — open the admin panel (administrators only)

In groups, mention the bot or reply to one of its messages unless mention-only mode is disabled."#;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn command_parsing_handles_bot_suffix() {
        assert_eq!(
            parse_command("/image@mybot red panda"),
            Some(("image".to_owned(), "red panda"))
        );
        assert_eq!(
            parse_command("-SEARCH current news"),
            Some(("search".to_owned(), "current news"))
        );
    }

    #[test]
    fn parses_admin_message_model_override() {
        let (override_, prompt) = parse_model_override("aihub:gpt-5.4-mini explain this").unwrap();
        assert_eq!(override_.model_provider, ModelProvider::Aihub);
        assert_eq!(override_.model, "gpt-5.4-mini");
        assert_eq!(prompt, "explain this");

        let (override_, prompt) = parse_model_override("openai/gpt-5.4:free write code").unwrap();
        assert_eq!(override_.model_provider, ModelProvider::Openrouter);
        assert_eq!(override_.model, "openai/gpt-5.4:free");
        assert_eq!(prompt, "write code");
    }

    #[test]
    fn quote_only_guest_replies_are_added_to_request_context() {
        let message: Message = serde_json::from_value(serde_json::json!({
            "message_id": 2,
            "date": 1,
            "chat": {"id": -100, "type": "supergroup"},
            "guest_bot_caller_user": {
                "id": 42,
                "is_bot": false,
                "first_name": "Caller"
            },
            "text": "What does the replied message say?",
            "quote": {"text": "The exact replied text", "position": 0}
        }))
        .unwrap();
        let context = reply_context(&message).unwrap();
        assert!(context.contains("The exact replied text"));
        assert!(
            with_reply_context("Question", Some(&context)).contains("[Replied message context]")
        );
    }

    #[test]
    fn full_replies_include_author_text_and_media_labels() {
        let message: Message = serde_json::from_value(serde_json::json!({
            "message_id": 2,
            "date": 1,
            "chat": {"id": 10, "type": "private"},
            "from": {"id": 42, "is_bot": false, "first_name": "Caller"},
            "text": "Transform it",
            "reply_to_message": {
                "message_id": 1,
                "date": 1,
                "chat": {"id": 10, "type": "private"},
                "from": {"id": 7, "is_bot": false, "first_name": "Alice", "username": "alice"},
                "caption": "Reference caption",
                "photo": [{"file_id":"id", "file_unique_id":"unique", "width":64, "height":64}]
            }
        }))
        .unwrap();
        let context = reply_context(&message).unwrap();
        assert!(context.contains("Alice (@alice)"));
        assert!(context.contains("Reference caption"));
        assert!(context.contains("Attachments: image"));
    }

    #[test]
    fn partial_reply_quotes_are_distinguished_from_the_full_message() {
        let message: Message = serde_json::from_value(serde_json::json!({
            "message_id": 2,
            "date": 1,
            "chat": {"id": 10, "type": "private"},
            "from": {"id": 42, "is_bot": false, "first_name": "Caller"},
            "text": "Explain this part",
            "quote": {
                "text": "selected middle sentence",
                "position": 18,
                "is_manual": true
            },
            "reply_to_message": {
                "message_id": 1,
                "date": 1,
                "chat": {"id": 10, "type": "private"},
                "from": {"id": 7, "is_bot": false, "first_name": "Alice"},
                "text": "Opening sentence. selected middle sentence. Closing sentence."
            }
        }))
        .unwrap();
        let context = reply_context(&message).unwrap();
        assert!(context.contains("Opening sentence"));
        assert!(context.contains(
            "Quoted excerpt selected by the caller (primary focus):\nselected middle sentence"
        ));
    }
}
