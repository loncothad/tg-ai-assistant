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
    methods::{
        AnswerGuestQueryParams, GetFileParams, GetUpdatesParams, SendAudioParams,
        SendChatActionParams, SendDocumentParams, SendPhotoParams, SendRichMessageParams,
        SendVideoParams,
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
    sync::{Semaphore, watch},
    task::JoinSet,
    time::sleep,
};
use tracing::{error, info, warn};

use crate::{
    Result,
    config::{BotConfig, Config, SearchProvider},
    db::{ModelRouting, Store},
    openrouter::{ChatRequest, MediaInput, OpenRouter, ToolModels},
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
    semaphore: Arc<Semaphore>,
    bot_user_id: u64,
    username: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MessageMode {
    Private,
    Group,
    Guest,
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
        let openrouter = OpenRouter::new(client.clone(), config.openrouter.clone());
        let search = SearchService::new(client.clone(), config.search.clone());
        let concurrency = config.server.max_concurrent_requests_per_bot;
        Ok(Self {
            telegram,
            bot,
            config,
            store,
            openrouter,
            search,
            client,
            semaphore: Arc::new(Semaphore::new(concurrency)),
            bot_user_id: identity.id,
            username,
        })
    }

    pub async fn run(self, mut shutdown: watch::Receiver<bool>) -> Result<()> {
        let mut offset = self.store.offset(&self.bot.id).await?;
        let mut backoff = 1u64;
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
            let mut jobs = JoinSet::new();
            for update in updates {
                let runner = self.clone();
                jobs.spawn(async move {
                    let permit = runner
                        .semaphore
                        .clone()
                        .acquire_owned()
                        .await
                        .context("Bot concurrency semaphore closed")?;
                    let result = runner.process(update).await;
                    drop(permit);
                    result
                });
            }
            while let Some(result) = jobs.join_next().await {
                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        error!(bot_id = %self.bot.id, error = %format!("{error:#}"), "update processing failed")
                    }
                    Err(error) => error!(bot_id = %self.bot.id, %error, "update task panicked"),
                }
            }
            self.store.set_offset(&self.bot.id, next_offset).await?;
            offset = Some(next_offset);
        }
        info!(bot_id = %self.bot.id, "Telegram bot stopped");
        Ok(())
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
                .is_some_and(message_has_media);
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
        let text = if raw_text.is_empty() {
            default_media_prompt(&message).to_owned()
        } else {
            self.strip_address(raw_text)
        };
        let scope = scope_id(mode, &message, user_id);

        if let Some((command, arguments)) = parse_command(&text) {
            if let Err(error) = self
                .command(&message, mode, user_id, &scope, &command, arguments)
                .await
            {
                error!(bot_id = %self.bot.id, user_id, error = %format!("{error:#}"), "command failed");
                self.respond(&message, mode, rich::compact_error(), None)
                    .await?;
            }
            return Ok(());
        }

        self.send_action(
            message.chat.id,
            message.message_thread_id,
            ChatAction::Typing,
        )
        .await;
        let settings = self.store.settings(&self.bot.id).await?;
        let mut capabilities = settings.capabilities.clone();
        if mode == MessageMode::Guest {
            capabilities.image = false;
            capabilities.audio = false;
            capabilities.video = false;
            capabilities.media = false;
            capabilities.transcription = false;
        }
        let provider = self.search_provider().await?;
        let openrouter_key = self
            .store
            .credential(&self.bot.id, "openrouter")
            .await?
            .context("OpenRouter API key is not configured")?;
        let search_key = if capabilities.search {
            if provider == SearchProvider::Openrouter {
                Some(openrouter_key.clone())
            } else {
                self.store
                    .credential(&self.bot.id, provider.as_str())
                    .await?
            }
        } else {
            None
        };
        let history = self.store.history(&self.bot.id, &scope).await?;
        let media = if capabilities.media
            || capabilities.transcription
            || capabilities.image
            || capabilities.video
        {
            self.collect_media(&message, &text).await?
        } else {
            Vec::new()
        };
        let (model_capability, selected) = if capabilities.media
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
        let model = self.config.resolved_model(model_capability, selected);
        let author = caller_name(&message);
        let contextual_text = format!("{author}: {text}{}", media_summary(&media));
        let mut instructions = self.store.effective_instructions(&self.bot.id).await?;
        instructions.push_str(&format!("\n\n# Current request context\nCurrent UTC date and time: {}\nBot: @{} (backend ID {})\nConversation mode: {:?}\nChat: {} (ID {})\nCaller: {} (ID {}, language {})\nTelegram message Unix time: {}\nEnabled capabilities for this request: search={}, web_fetch={}, image={}, audio={}, video={}, media={}, transcription={}", chrono::Utc::now().to_rfc3339(), self.username, self.bot.id, mode, message.chat.title.as_deref().unwrap_or("Private or untitled chat"), message.chat.id, author, user_id, message.guest_bot_caller_user.as_ref().or(message.from.as_ref()).and_then(|u| u.language_code.as_deref()).unwrap_or("unknown"), message.date, capabilities.search, capabilities.web_fetch, capabilities.image, capabilities.audio, capabilities.video, capabilities.media, capabilities.transcription));
        self.store
            .append_message(&self.bot.id, &scope, "user", &contextual_text)
            .await?;
        let default_routing = ModelRouting::default();
        let routing = settings
            .model_routing
            .get(model_capability)
            .unwrap_or(&default_routing);
        let session_id = format!("{}:{scope}", self.bot.id);
        match self
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
                api_key: &openrouter_key,
                capabilities: &capabilities,
                routing,
                tool_models: ToolModels {
                    image_generation: &settings.selected_image_generation_model,
                    image_routing: settings
                        .model_routing
                        .get("image_generation")
                        .unwrap_or(&default_routing),
                    audio_generation: &settings.selected_audio_generation_model,
                    audio_routing: settings
                        .model_routing
                        .get("audio_generation")
                        .unwrap_or(&default_routing),
                    transcription: &settings.selected_transcription_model,
                    transcription_routing: settings
                        .model_routing
                        .get("transcription")
                        .unwrap_or(&default_routing),
                    video_generation: &settings.selected_video_generation_model,
                    video_routing: settings
                        .model_routing
                        .get("video_generation")
                        .unwrap_or(&default_routing),
                },
            })
            .await
        {
            Ok(answer) => {
                self.store
                    .append_message(&self.bot.id, &scope, "assistant", &answer.text)
                    .await?;
                self.respond(&message, mode, &answer.text, None).await?;
                if mode != MessageMode::Guest {
                    for image in answer.generated_images {
                        self.send_photo_bytes(
                            message.chat.id,
                            message.message_thread_id,
                            &image.bytes,
                            &image.media_type,
                            Some("Generated with OpenRouter"),
                        )
                        .await?;
                    }
                    for audio in answer.generated_audio {
                        self.send_audio_bytes(
                            message.chat.id,
                            message.message_thread_id,
                            &audio.bytes,
                            &audio.media_type,
                        )
                        .await?;
                    }
                    for video in answer.generated_videos {
                        self.send_video_url(
                            message.chat.id,
                            message.message_thread_id,
                            &video,
                            Some("Generated with OpenRouter"),
                        )
                        .await?;
                    }
                    for url in answer.media_urls {
                        self.send_photo_url(message.chat.id, message.message_thread_id, &url, None)
                            .await?;
                    }
                }
                if let Some(id) = answer.generation_id {
                    self.store
                        .audit(&self.bot.id, Some(user_id), "chat.complete", Some(&id))
                        .await?;
                }
            }
            Err(error) => {
                error!(bot_id = %self.bot.id, user_id, error = %format!("{error:#}"), "assistant request failed");
                self.respond(&message, mode, rich::compact_error(), None)
                    .await?;
            }
        }
        Ok(())
    }

    async fn command(
        &self,
        message: &Message,
        mode: MessageMode,
        user_id: u64,
        scope: &str,
        command: &str,
        arguments: &str,
    ) -> Result<()> {
        match command {
            "start" | "help" => self.respond(message, mode, HELP, None).await?,
            "new" => {
                self.store.clear_history(&self.bot.id, scope).await?;
                self.respond(message, mode, "Conversation history cleared.", None)
                    .await?;
            }
            "model" => {
                let model = self.store.selected_model(&self.bot.id).await?;
                self.respond(message, mode, &format!("Current model: `{model}`"), None)
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
                    let model = self.config.resolved_model("chat", &settings.selected_model);
                    let routing = settings
                        .model_routing
                        .get("chat")
                        .cloned()
                        .unwrap_or_default();
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
                if mode == MessageMode::Guest {
                    self.respond(
                        message,
                        mode,
                        "Image generation is not available in guest mode.",
                        None,
                    )
                    .await?;
                    return Ok(());
                }
                require_arguments(arguments, "/image <prompt>")?;
                self.send_action(
                    message.chat.id,
                    message.message_thread_id,
                    ChatAction::UploadPhoto,
                )
                .await;
                let key = self
                    .store
                    .credential(&self.bot.id, "openrouter")
                    .await?
                    .context("OpenRouter API key is not configured")?;
                let references = self.collect_media(message, arguments).await?;
                let routing = settings
                    .model_routing
                    .get("image_generation")
                    .cloned()
                    .unwrap_or_default();
                match self
                    .openrouter
                    .generate_image_with_references(
                        arguments,
                        &references,
                        &settings.selected_image_generation_model,
                        &routing,
                        &key,
                    )
                    .await
                {
                    Ok(images) => {
                        for image in images {
                            self.send_photo_bytes(
                                message.chat.id,
                                message.message_thread_id,
                                &image.bytes,
                                &image.media_type,
                                Some("Generated with OpenRouter"),
                            )
                            .await?;
                        }
                    }
                    Err(error) => {
                        error!(bot_id = %self.bot.id, user_id, error = %format!("{error:#}"), "image generation failed");
                        self.respond(message, mode, rich::compact_error(), None)
                            .await?;
                    }
                }
            }
            "audio" => {
                if mode == MessageMode::Guest {
                    self.respond(
                        message,
                        mode,
                        "Audio generation is not available in guest mode.",
                        None,
                    )
                    .await?;
                    return Ok(());
                }
                let settings = self.store.settings(&self.bot.id).await?;
                if !settings.capabilities.audio {
                    bail!("Audio generation is disabled by an administrator");
                }
                require_arguments(arguments, "/audio <text>")?;
                let key = self
                    .store
                    .credential(&self.bot.id, "openrouter")
                    .await?
                    .context("OpenRouter API key is not configured")?;
                self.send_action(
                    message.chat.id,
                    message.message_thread_id,
                    ChatAction::UploadVoice,
                )
                .await;
                let routing = settings
                    .model_routing
                    .get("audio_generation")
                    .cloned()
                    .unwrap_or_default();
                match self
                    .openrouter
                    .generate_audio(
                        arguments,
                        &settings.selected_audio_generation_model,
                        &routing,
                        &key,
                    )
                    .await
                {
                    Ok(audio) => {
                        self.send_audio_bytes(
                            message.chat.id,
                            message.message_thread_id,
                            &audio.bytes,
                            &audio.media_type,
                        )
                        .await?
                    }
                    Err(error) => {
                        error!(bot_id = %self.bot.id, user_id, error = %format!("{error:#}"), "audio generation failed");
                        self.respond(message, mode, rich::compact_error(), None)
                            .await?;
                    }
                }
            }
            "transcribe" => {
                if mode == MessageMode::Guest {
                    self.respond(
                        message,
                        mode,
                        "Transcription is not available in guest mode.",
                        None,
                    )
                    .await?;
                    return Ok(());
                }
                let settings = self.store.settings(&self.bot.id).await?;
                if !settings.capabilities.transcription {
                    bail!("Transcription is disabled by an administrator");
                }
                let media = self.collect_media(message, arguments).await?;
                let key = self
                    .store
                    .credential(&self.bot.id, "openrouter")
                    .await?
                    .context("OpenRouter API key is not configured")?;
                let mut transcripts = Vec::new();
                let routing = settings
                    .model_routing
                    .get("transcription")
                    .cloned()
                    .unwrap_or_default();
                for input in &media {
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
                self.respond(
                    message,
                    mode,
                    &format!("# Transcript\n\n{}", transcripts.join("\n\n")),
                    None,
                )
                .await?;
            }
            "video" => {
                let settings = self.store.settings(&self.bot.id).await?;
                if !settings.capabilities.video {
                    bail!("Video generation is disabled by an administrator");
                }
                if mode == MessageMode::Guest {
                    self.respond(
                        message,
                        mode,
                        "Video generation is not available in guest mode.",
                        None,
                    )
                    .await?;
                    return Ok(());
                }
                require_arguments(arguments, "/video <prompt>")?;
                self.send_action(
                    message.chat.id,
                    message.message_thread_id,
                    ChatAction::UploadVideo,
                )
                .await;
                let key = self
                    .store
                    .credential(&self.bot.id, "openrouter")
                    .await?
                    .context("OpenRouter API key is not configured")?;
                let references = self.collect_media(message, arguments).await?;
                let routing = settings
                    .model_routing
                    .get("video_generation")
                    .cloned()
                    .unwrap_or_default();
                match self
                    .openrouter
                    .generate_video_with_references(
                        arguments,
                        &references,
                        &settings.selected_video_generation_model,
                        &routing,
                        &key,
                    )
                    .await
                {
                    Ok(url) => {
                        self.send_video_url(
                            message.chat.id,
                            message.message_thread_id,
                            &url,
                            Some("Generated with OpenRouter"),
                        )
                        .await?
                    }
                    Err(error) => {
                        error!(bot_id = %self.bot.id, user_id, error = %format!("{error:#}"), "video generation failed");
                        self.respond(message, mode, rich::compact_error(), None)
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
            self.bot.id
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
            self.telegram.send_document(&params).await?;
            return Ok(());
        }
        let mut params = SendPhotoParams::builder()
            .chat_id(chat_id)
            .photo(FileUpload::from(file.path().to_path_buf()))
            .build();
        params.message_thread_id = thread_id;
        params.caption = caption.map(str::to_owned);
        self.telegram.send_photo(&params).await?;
        Ok(())
    }

    async fn send_photo_url(
        &self,
        chat_id: i64,
        thread_id: Option<i32>,
        url: &str,
        caption: Option<&str>,
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
        self.telegram.send_photo(&params).await?;
        Ok(())
    }

    async fn send_audio_bytes(
        &self,
        chat_id: i64,
        thread_id: Option<i32>,
        bytes: &[u8],
        media_type: &str,
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
            .caption("Generated with OpenRouter")
            .build();
        params.message_thread_id = thread_id;
        self.telegram.send_audio(&params).await?;
        Ok(())
    }

    async fn send_video_url(
        &self,
        chat_id: i64,
        thread_id: Option<i32>,
        url: &str,
        caption: Option<&str>,
    ) -> Result<()> {
        let mut params = SendVideoParams::builder()
            .chat_id(chat_id)
            .video(FileUpload::from(url.to_owned()))
            .build();
        params.message_thread_id = thread_id;
        params.caption = caption.map(str::to_owned);
        params.supports_streaming = Some(true);
        self.telegram.send_video(&params).await?;
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
        if let Some(url) = youtube_url(text) {
            inputs.push(MediaInput::Video { url });
        }
        Ok(inputs)
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
}

fn parse_command(text: &str) -> Option<(String, &str)> {
    let text = text.strip_prefix('/').or_else(|| text.strip_prefix('-'))?;
    let (head, rest) = text.split_once(char::is_whitespace).unwrap_or((text, ""));
    Some((
        head.split('@').next().unwrap_or(head).to_ascii_lowercase(),
        rest.trim(),
    ))
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
- `/search <query>` — force a live web search
- `/searchprovider` — show the active search provider
- `/image <prompt>` — generate an image
- `/audio <text>` — generate spoken audio
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
}
