//! OpenRouter and AI Hub chat/tool and media API clients.

use std::time::{Duration, Instant};

use base64::{Engine, engine::general_purpose::STANDARD};
use eyre::{Context, ContextCompat, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use tokio::sync::mpsc::UnboundedSender;
use tokio::time::sleep;

use crate::{
    Result,
    config::{
        AiHubConfig, ModelConfig, ModelProvider, OpenRouterConfig, OpenRouterOptions,
        SearchProvider,
    },
    db::{Capabilities, ChatMessage, ModelRouting},
    search::SearchService,
};

const MAX_TOOL_ROUNDS: usize = 6;

#[derive(Clone)]
pub struct OpenRouter {
    client: reqwest::Client,
    config: OpenRouterConfig,
    aihub: AiHubConfig,
}

#[derive(Clone, Debug)]
pub struct AssistantResponse {
    pub text: String,
    pub media_urls: Vec<String>,
    pub generation_id: Option<String>,
    pub usage: Option<Value>,
    pub generated_images: Vec<GeneratedImage>,
    pub generated_audio: Vec<GeneratedImage>,
    pub generated_videos: Vec<GeneratedVideo>,
    pub generated_files: Vec<GeneratedFile>,
}

#[derive(Clone, Debug)]
pub struct GeneratedImage {
    pub bytes: Vec<u8>,
    pub media_type: String,
    pub model: String,
    pub prompt: String,
}

/// A generated video URL with the exact model and prompt used to create it.
#[derive(Clone, Debug)]
pub struct GeneratedVideo {
    pub url: String,
    pub model: String,
    pub prompt: String,
}

/// A UTF-8 file requested by the model for delivery through Telegram.
#[derive(Clone, Debug)]
pub struct GeneratedFile {
    pub filename: String,
    pub bytes: Vec<u8>,
}

/// Schema-validated routing decision produced by the inexpensive planner.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct RequestPlan {
    pub action: PlannedAction,
    pub skills: Vec<PlannedSkill>,
    pub delivery: PlannedDelivery,
    pub filename: String,
    pub refusal_message: String,
}

impl RequestPlan {
    /// Resolves a direct media action from either the primary action field or
    /// the planner's selected generation skill. This tolerates inexpensive
    /// models that emit a correct skill but leave `action` set to `chat`.
    pub fn direct_generation(&self) -> Option<PlannedAction> {
        match self.action {
            PlannedAction::GenerateImage
            | PlannedAction::GenerateAudio
            | PlannedAction::GenerateVideo => Some(self.action),
            PlannedAction::Chat | PlannedAction::GenerateCode => {
                let generations = self
                    .skills
                    .iter()
                    .filter_map(|skill| match skill {
                        PlannedSkill::ImageGeneration => Some(PlannedAction::GenerateImage),
                        PlannedSkill::AudioGeneration => Some(PlannedAction::GenerateAudio),
                        PlannedSkill::VideoGeneration => Some(PlannedAction::GenerateVideo),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                if generations.len() == 1 {
                    generations.first().copied()
                } else {
                    None
                }
            }
            PlannedAction::Refuse => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlannedDelivery {
    #[default]
    Inline,
    File,
}

/// A user-visible milestone recorded in the live Telegram progress message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProgressUpdate {
    /// A short processing milestone.
    Step(String),
    /// A media-generation call, including its unmodified effective input.
    Generation {
        kind: &'static str,
        model: String,
        prompt: String,
    },
}

impl ProgressUpdate {
    pub fn step(value: impl Into<String>) -> Self {
        Self::Step(value.into())
    }

    pub fn generation(kind: &'static str, model: &str, prompt: &str) -> Self {
        Self::Generation {
            kind,
            model: model.to_owned(),
            prompt: prompt.to_owned(),
        }
    }
}

impl AssistantResponse {
    /// Enforces the intent planner's requested delivery when the model did not
    /// call `send_file` itself. Size and fenced-code safeguards run separately.
    pub fn apply_planned_delivery(&mut self, plan: &RequestPlan, file_enabled: bool) {
        if !file_enabled
            || plan.delivery != PlannedDelivery::File
            || !self.generated_files.is_empty()
            || self.text.trim().is_empty()
        {
            return;
        }
        let filename = safe_filename(if plan.filename.trim().is_empty() {
            "answer.md"
        } else {
            plan.filename.trim()
        })
        .unwrap_or_else(|| "answer.md".to_owned());
        self.generated_files.push(GeneratedFile {
            filename: filename.clone(),
            bytes: self.text.as_bytes().to_vec(),
        });
        self.text = format!("The requested output is attached as `{filename}`.");
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlannedAction {
    Chat,
    GenerateCode,
    GenerateImage,
    GenerateAudio,
    GenerateVideo,
    Refuse,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlannedSkill {
    GenerateCode,
    Search,
    WebFetch,
    ImageGeneration,
    AudioGeneration,
    VideoGeneration,
    ImageUnderstanding,
    VideoUnderstanding,
    Transcription,
    FileDelivery,
    ModelUpgrade,
}

impl PlannedSkill {
    fn as_str(self) -> &'static str {
        match self {
            Self::GenerateCode => "generate_code",
            Self::Search => "search",
            Self::WebFetch => "web_fetch",
            Self::ImageGeneration => "image_generation",
            Self::AudioGeneration => "audio_generation",
            Self::VideoGeneration => "video_generation",
            Self::ImageUnderstanding => "image_understanding",
            Self::VideoUnderstanding => "video_understanding",
            Self::Transcription => "transcription",
            Self::FileDelivery => "file_delivery",
            Self::ModelUpgrade => "model_upgrade",
        }
    }
}

/// Minimal context passed to the planner; attachment bytes and secrets are never
/// included in the classification request.
pub struct PlanningRequest<'a> {
    pub text: &'a str,
    pub model: &'a str,
    pub fallback_model: &'a str,
    pub capabilities: &'a Capabilities,
    pub has_image: bool,
    pub has_video: bool,
    pub has_audio: bool,
    pub api_key: &'a str,
}

/// Private media downloaded from Telegram or a public video URL supplied by a user.
#[derive(Clone, Debug)]
pub enum MediaInput {
    Image { url: String },
    Video { url: String },
    Audio { data: String, format: String },
}

/// Complete, explicitly scoped input for one model-driven assistant turn.
pub struct ChatRequest<'a> {
    pub model: &'a ModelConfig,
    pub system_prompt: &'a str,
    pub history: &'a [ChatMessage],
    pub user_message: &'a str,
    pub session_id: &'a str,
    pub media: &'a [MediaInput],
    pub search: &'a SearchService,
    pub search_provider: SearchProvider,
    pub search_api_key: Option<&'a str>,
    pub api_key: &'a str,
    pub model_provider: ModelProvider,
    pub capabilities: &'a Capabilities,
    pub routing: &'a ModelRouting,
    pub tool_models: ToolModels<'a>,
    pub progress: Option<UnboundedSender<ProgressUpdate>>,
}

#[derive(Clone, Copy)]
pub struct ToolModels<'a> {
    pub image_generation: ToolModel<'a>,
    pub audio_generation: ToolModel<'a>,
    pub transcription: ToolModel<'a>,
    pub video_generation: ToolModel<'a>,
}

/// Provider-qualified model and credential used by a model-callable media tool.
#[derive(Clone, Copy)]
pub struct ToolModel<'a> {
    pub model: &'a str,
    pub routing: &'a ModelRouting,
    pub api_key: &'a str,
}

impl OpenRouter {
    pub fn new(client: reqwest::Client, config: OpenRouterConfig, aihub: AiHubConfig) -> Self {
        Self {
            client,
            config,
            aihub,
        }
    }

    /// Classifies a natural-language request through OpenRouter Structured
    /// Outputs. Callers should fall back to ordinary chat if this bounded call
    /// fails, since planning is an optimization rather than a dependency.
    pub async fn plan_request(&self, request: PlanningRequest<'_>) -> Result<RequestPlan> {
        let enabled = enabled_planner_skills(request.capabilities);
        let schema = json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["chat", "generate_code", "generate_image", "generate_audio", "generate_video", "refuse"],
                    "description": "Use generate_code for source code, configuration, scripts, patches, or complete software artifacts. Use generate_image, generate_audio, or generate_video whenever the user explicitly asks to create that media artifact, in any language. Use chat for ordinary prose."
                },
                "skills": {
                    "type": "array",
                    "items": {"type":"string", "enum": ["generate_code", "search", "web_fetch", "image_generation", "audio_generation", "video_generation", "image_understanding", "video_understanding", "transcription", "file_delivery", "model_upgrade"]},
                    "uniqueItems": true
                },
                "delivery": {
                    "type": "string",
                    "enum": ["inline", "file"],
                    "description": "Choose file when the user requests a downloadable file, a complete code/configuration artifact, or an answer expected to be too large for convenient chat reading. Otherwise choose inline."
                },
                "filename": {
                    "type": "string",
                    "description": "Safe filename with extension when delivery is file; empty when delivery is inline."
                },
                "refusal_message": {
                    "type": "string",
                    "description": "Empty unless action is refuse; then a concise, respectful response in the user's language."
                }
            },
            "required": ["action", "skills", "delivery", "filename", "refusal_message"],
            "additionalProperties": false
        });
        let system = format!(
            "You are a request classifier for a Telegram AI assistant. Return only the required schema. Enabled skills: {}. Attachments: image={}, video={}, audio={}. Classify the user's original text; never rewrite, expand, translate, improve, or execute it. Use generate_code for source code, configuration, scripts, patches, and complete software artifacts; it is handled by the normal assistant pipeline. An explicit request to draw, generate, create, synthesize, or make new image/audio/video media MUST use the corresponding generate_* action and generation skill, regardless of language. Describing or understanding existing media, transcribing, researching, opening URLs, answering, and transforming text use chat with suitable skills. Select model_upgrade only for a genuinely difficult request whose complexity, ambiguity, reasoning depth, or accuracy requirements materially benefit from the configured advanced model; do not select it for routine requests. Choose file for a complete source-code/configuration artifact when file_delivery is enabled, a requested downloadable artifact, or output expected to be unwieldy in chat; provide a safe filename and include file_delivery when enabled. Use inline for normal prose. Never select a disabled skill. Select refuse only when fulfilling the request itself is disallowed; do not refuse merely because a skill is unavailable. For refusal, write a concise localized explanation and safe alternative. The original user text is the sole downstream request and is not a field in your output.",
            enabled.join(", "),
            request.has_image,
            request.has_video,
            request.has_audio
        );
        let models = [request.model, request.fallback_model];
        let primary_model = models[0];
        let mut failures = Vec::new();
        let mut parsed = None;
        for (attempt, model) in models.into_iter().enumerate() {
            if attempt == 1 && model == primary_model {
                continue;
            }
            let body = json!({
                "model": model,
                "messages": [
                    {"role":"system", "content":system},
                    {"role":"user", "content":request.text}
                ],
                "temperature": 0,
                "max_tokens": self.config.planner.max_tokens,
                "plugins": [{"id":"response-healing"}],
                "provider": {
                    "require_parameters": true,
                    "allow_fallbacks": true,
                    "data_collection": "allow",
                    "zdr": false
                },
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "telegram_request_plan",
                        "strict": true,
                        "schema": schema
                    }
                }
            });
            let result = tokio::time::timeout(
                Duration::from_secs(self.config.planner.timeout_seconds),
                self.post_json_for(
                    ModelProvider::Openrouter,
                    "chat/completions",
                    body,
                    request.api_key,
                ),
            )
            .await;
            let value = match result {
                Ok(Ok(value)) => value,
                Ok(Err(error)) => {
                    failures.push(format!("{model}: {error:#}"));
                    continue;
                }
                Err(_) => {
                    failures.push(format!("{model}: Timed out"));
                    continue;
                }
            };
            match parse_planner_response(&value) {
                Ok(plan) => {
                    parsed = Some(plan);
                    break;
                }
                Err(error) => failures.push(format!("{model}: {error:#}")),
            }
        }
        let mut plan = parsed.wrap_err_with(|| {
            format!(
                "OpenRouter request planner exhausted its models: {}",
                failures.join("; ")
            )
        })?;
        plan.refusal_message.truncate(2_000);
        plan.skills
            .retain(|skill| enabled.contains(&skill.as_str()));
        if !planned_action_enabled(plan.action, request.capabilities) {
            plan.action = PlannedAction::Chat;
            plan.refusal_message.clear();
        }
        if plan.action == PlannedAction::Refuse && plan.refusal_message.trim().is_empty() {
            plan.refusal_message =
                "I can’t help fulfill that request, but I can help with a safe alternative."
                    .to_owned();
        }
        Ok(plan)
    }

    pub async fn chat(&self, request: ChatRequest<'_>) -> Result<AssistantResponse> {
        let ChatRequest {
            model,
            system_prompt,
            history,
            user_message,
            session_id,
            media,
            search,
            search_provider,
            search_api_key,
            api_key,
            model_provider,
            capabilities,
            routing,
            tool_models,
            progress,
        } = request;
        report_progress(&progress, "Preparing conversation context");
        let mut messages = vec![json!({ "role": "system", "content": system_prompt })];
        messages.extend(
            history
                .iter()
                .map(|m| json!({ "role": m.role, "content": m.content })),
        );
        messages.push(json!({ "role": "user", "content": user_content(user_message, media, capabilities.media) }));

        let mut generated_images = Vec::new();
        let mut generated_audio = Vec::new();
        let mut generated_videos = Vec::new();
        let mut generated_files = Vec::new();
        for _ in 0..MAX_TOOL_ROUNDS {
            let mut body = Map::new();
            body.insert("messages".into(), Value::Array(messages.clone()));
            if model_provider == ModelProvider::Openrouter {
                body.insert("session_id".into(), json!(session_id));
                apply_options(&mut body, &self.config.defaults);
            }
            apply_options(&mut body, &model.options);
            let routed_model = if model_provider == ModelProvider::Openrouter {
                apply_routing(&mut body, &model.id, routing, true)
            } else {
                model.id.clone()
            };
            body.insert("model".into(), Value::String(routed_model));
            add_tools(
                &mut body,
                capabilities,
                ToolContext {
                    search_provider,
                    search_ready: search_api_key.is_some(),
                    web_search: &self.config.web_search,
                    web_fetch: &self.config.web_fetch,
                    audio_attached: media
                        .iter()
                        .any(|item| matches!(item, MediaInput::Audio { .. })),
                    openrouter_server_tools: model_provider == ModelProvider::Openrouter,
                },
            );
            report_progress(&progress, "Waiting for the selected AI model");

            let value = self
                .post_json_for(
                    model_provider,
                    "chat/completions",
                    Value::Object(body),
                    api_key,
                )
                .await?;
            let message = value
                .pointer("/choices/0/message")
                .cloned()
                .wrap_err_with(|| {
                    format!(
                        "{} returned no assistant message",
                        provider_name(model_provider)
                    )
                })?;
            let calls = message
                .get("tool_calls")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            if calls.is_empty() {
                let (mut text, media_urls) = extract_content(&message);
                if text.is_empty() {
                    text = extract_refusal(&message).unwrap_or_default();
                }
                if text.is_empty() && media_urls.is_empty() {
                    bail!(
                        "{} returned an empty response",
                        provider_name(model_provider)
                    );
                }
                let text = materialize_file_answer(text, capabilities.file, &mut generated_files);
                return Ok(AssistantResponse {
                    text,
                    media_urls,
                    generation_id: value.get("id").and_then(Value::as_str).map(str::to_owned),
                    usage: value.get("usage").cloned(),
                    generated_images,
                    generated_audio,
                    generated_videos,
                    generated_files,
                });
            }

            messages.push(message);
            for call in calls {
                let call_id = call.get("id").and_then(Value::as_str).unwrap_or("unknown");
                let name = call
                    .pointer("/function/name")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let arguments = call
                    .pointer("/function/arguments")
                    .and_then(Value::as_str)
                    .and_then(|s| serde_json::from_str::<Value>(s).ok())
                    .unwrap_or(Value::Null);
                let output = match name {
                    "web_search" => {
                        report_progress(&progress, "Searching the web");
                        let query = arguments
                            .get("query")
                            .and_then(Value::as_str)
                            .unwrap_or(user_message);
                        if search_provider == SearchProvider::Openrouter {
                            match self
                                .config
                                .models
                                .first()
                                .context("OpenRouter search has no configured model")
                            {
                                Ok(search_model) => match self
                                    .search(
                                        query,
                                        search_model,
                                        &ModelRouting::default(),
                                        search_api_key.unwrap_or_default(),
                                    )
                                    .await
                                {
                                    Ok(answer) => json!({
                                        "query": query,
                                        "provider": "openrouter",
                                        "answer": answer
                                    })
                                    .to_string(),
                                    Err(error) => json!({"error": error.to_string()}).to_string(),
                                },
                                Err(error) => json!({"error": error.to_string()}).to_string(),
                            }
                        } else {
                            search
                                .tool_output(
                                    search_provider,
                                    query,
                                    search_api_key.unwrap_or_default(),
                                )
                                .await
                        }
                    }
                    "generate_image" if capabilities.image => {
                        let prompt = arguments
                            .get("prompt")
                            .and_then(Value::as_str)
                            .unwrap_or(user_message);
                        report_generation_progress(
                            &progress,
                            "image",
                            tool_models.image_generation.model,
                            prompt,
                        );
                        match self
                            .generate_image_with_references(
                                prompt,
                                media,
                                tool_models.image_generation.model,
                                tool_models.image_generation.routing,
                                tool_models.image_generation.api_key,
                            )
                            .await
                        {
                            Ok(values) => {
                                let count = values.len();
                                generated_images.extend(values);
                                json!({"status":"completed","images":count}).to_string()
                            }
                            Err(error) => json!({"error":error.to_string()}).to_string(),
                        }
                    }
                    "generate_audio" if capabilities.audio => {
                        let input = arguments
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or(user_message);
                        report_generation_progress(
                            &progress,
                            "audio",
                            tool_models.audio_generation.model,
                            input,
                        );
                        match self
                            .generate_audio(
                                input,
                                tool_models.audio_generation.model,
                                tool_models.audio_generation.routing,
                                tool_models.audio_generation.api_key,
                            )
                            .await
                        {
                            Ok(value) => {
                                generated_audio.push(value);
                                json!({"status":"completed","audio_files":1}).to_string()
                            }
                            Err(error) => json!({"error":error.to_string()}).to_string(),
                        }
                    }
                    "generate_video" if capabilities.video => {
                        let prompt = arguments
                            .get("prompt")
                            .and_then(Value::as_str)
                            .unwrap_or(user_message);
                        report_generation_progress(
                            &progress,
                            "video",
                            tool_models.video_generation.model,
                            prompt,
                        );
                        match self
                            .generate_video_with_references(
                                prompt,
                                media,
                                tool_models.video_generation.model,
                                tool_models.video_generation.routing,
                                tool_models.video_generation.api_key,
                            )
                            .await
                        {
                            Ok(value) => {
                                generated_videos.push(GeneratedVideo {
                                    url: value,
                                    model: tool_models.video_generation.model.to_owned(),
                                    prompt: prompt.to_owned(),
                                });
                                json!({"status":"completed","videos":1}).to_string()
                            }
                            Err(error) => json!({"error":error.to_string()}).to_string(),
                        }
                    }
                    "transcribe_audio" if capabilities.transcription => {
                        report_progress(&progress, "Transcribing the audio");
                        let language = arguments.get("language").and_then(Value::as_str);
                        let mut transcripts = Vec::new();
                        for item in media {
                            if let MediaInput::Audio { data, format } = item {
                                match self
                                    .transcribe_audio(
                                        data,
                                        format,
                                        language,
                                        tool_models.transcription.model,
                                        tool_models.transcription.routing,
                                        tool_models.transcription.api_key,
                                    )
                                    .await
                                {
                                    Ok(value) => transcripts.push(value),
                                    Err(error) => {
                                        transcripts.push(format!("[Transcription failed: {error}]"))
                                    }
                                }
                            }
                        }
                        if transcripts.is_empty() {
                            json!({"error":"No audio attachment is available"}).to_string()
                        } else {
                            json!({"transcripts":transcripts}).to_string()
                        }
                    }
                    "send_file" if capabilities.file => {
                        report_progress(&progress, "Preparing a downloadable file");
                        match file_from_arguments(&arguments) {
                            Ok(file) => {
                                let filename = file.filename.clone();
                                let bytes = file.bytes.len();
                                generated_files.push(file);
                                json!({"status":"ready","filename":filename,"bytes":bytes})
                                    .to_string()
                            }
                            Err(error) => json!({"error":error.to_string()}).to_string(),
                        }
                    }
                    _ => {
                        json!({ "error": format!("Unknown or disabled tool: {name}") }).to_string()
                    }
                };
                messages
                    .push(json!({ "role": "tool", "tool_call_id": call_id, "content": output }));
            }
        }
        bail!(
            "{} exceeded the maximum tool-call rounds",
            provider_name(model_provider)
        )
    }

    pub async fn search(
        &self,
        query: &str,
        model: &ModelConfig,
        routing: &ModelRouting,
        api_key: &str,
    ) -> Result<String> {
        let mut body = Map::new();
        body.insert(
            "messages".into(),
            json!([{"role":"user","content":format!("Search the web and answer with citations: {query}")}]),
        );
        apply_options(&mut body, &self.config.defaults);
        apply_options(&mut body, &model.options);
        let routed_model = apply_routing(&mut body, &model.id, routing, true);
        body.insert("model".into(), json!(routed_model));
        let tool = openrouter_web_search_tool(&self.config.web_search);
        match body.get_mut("tools") {
            Some(Value::Array(tools)) => tools.push(tool),
            _ => {
                body.insert("tools".into(), json!([tool]));
            }
        }
        let value = self
            .post_json("chat/completions", Value::Object(body), api_key)
            .await?;
        let message = value
            .pointer("/choices/0/message")
            .context("OpenRouter search returned no assistant message")?;
        let (text, _) = extract_content(message);
        if text.is_empty() {
            bail!("OpenRouter search returned an empty answer");
        }
        Ok(text)
    }

    pub async fn generate_image(&self, prompt: &str, api_key: &str) -> Result<Vec<GeneratedImage>> {
        self.generate_image_with_references(
            prompt,
            &[],
            &self.config.image.model,
            &ModelRouting::default(),
            api_key,
        )
        .await
    }

    pub async fn generate_image_with_references(
        &self,
        prompt: &str,
        media: &[MediaInput],
        model: &str,
        routing: &ModelRouting,
        api_key: &str,
    ) -> Result<Vec<GeneratedImage>> {
        if model.is_empty() {
            bail!("Image generation model is not configured");
        }
        let mut body = self.config.image.extra.clone();
        if !self.config.image.size.is_empty() {
            body.insert("size".into(), json!(self.config.image.size));
        }
        if let Some(choice) = self
            .config
            .image
            .models
            .iter()
            .find(|choice| choice.id == model)
        {
            body.extend(choice.extra.clone());
        }
        let routed_model = if routing.model_provider == ModelProvider::Openrouter {
            apply_routing(&mut body, model, routing, false)
        } else {
            model.to_owned()
        };
        body.insert("model".into(), json!(routed_model));
        body.insert("prompt".into(), json!(prompt));
        let references = media
            .iter()
            .filter_map(|item| match item {
                MediaInput::Image { url } => Some(reference("image_url", url)),
                _ => None,
            })
            .collect::<Vec<_>>();
        if !references.is_empty() {
            if routing.model_provider == ModelProvider::Aihub {
                bail!(
                    "AI Hub image generation does not support reference media through the OpenAI generations endpoint"
                );
            }
            body.insert("input_references".into(), Value::Array(references));
        }
        let endpoint = if routing.model_provider == ModelProvider::Aihub {
            "images/generations"
        } else {
            "images"
        };
        let value = self
            .post_json_for(
                routing.model_provider,
                endpoint,
                Value::Object(body),
                api_key,
            )
            .await?;
        let mut images = Vec::new();
        for item in value
            .get("data")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(encoded) = item.get("b64_json").and_then(Value::as_str) {
                images.push(GeneratedImage {
                    bytes: STANDARD
                        .decode(encoded)
                        .context("OpenRouter returned invalid base64 image")?,
                    media_type: item
                        .get("media_type")
                        .and_then(Value::as_str)
                        .unwrap_or("image/png")
                        .to_owned(),
                    model: model.to_owned(),
                    prompt: prompt.to_owned(),
                });
            } else if let Some(url) = item.get("url").and_then(Value::as_str) {
                let response = self
                    .client
                    .get(url)
                    .timeout(Duration::from_secs(60))
                    .send()
                    .await
                    .context("Failed to download generated image")?;
                let status = response.status();
                let media_type = response
                    .headers()
                    .get(reqwest::header::CONTENT_TYPE)
                    .and_then(|value| value.to_str().ok())
                    .unwrap_or("image/png")
                    .to_owned();
                if !status.is_success() {
                    bail!("Generated image download returned {status}");
                }
                images.push(GeneratedImage {
                    bytes: response
                        .bytes()
                        .await
                        .context("Failed to read generated image")?
                        .to_vec(),
                    media_type,
                    model: model.to_owned(),
                    prompt: prompt.to_owned(),
                });
            }
        }
        if images.is_empty() {
            bail!("OpenRouter returned no generated images");
        }
        Ok(images)
    }

    pub async fn generate_audio(
        &self,
        input: &str,
        model: &str,
        routing: &ModelRouting,
        api_key: &str,
    ) -> Result<GeneratedImage> {
        let mut body = self.config.audio.extra.clone();
        body.insert(
            "response_format".into(),
            json!(self.config.audio.response_format),
        );
        body.insert("speed".into(), json!(self.config.audio.speed));
        if let Some(choice) = self
            .config
            .audio
            .models
            .iter()
            .find(|choice| choice.id == model)
        {
            body.extend(choice.extra.clone());
        }
        let configured_voice = body
            .get("voice")
            .and_then(Value::as_str)
            .unwrap_or(&self.config.audio.voice);
        let voice = if routing.model_provider == ModelProvider::Openrouter {
            self.resolve_speech_voice(model, configured_voice, api_key)
                .await
        } else {
            configured_voice.to_owned()
        };
        body.insert("voice".into(), json!(voice));
        let routed_model = if routing.model_provider == ModelProvider::Openrouter {
            apply_routing(&mut body, model, routing, false)
        } else {
            model.to_owned()
        };
        body.insert("model".into(), json!(routed_model));
        body.insert("input".into(), json!(input));
        let response = self
            .request_for(
                routing.model_provider,
                self.client
                    .post(format!(
                        "{}/audio/speech",
                        self.base_url(routing.model_provider)
                    ))
                    .json(&body),
                api_key,
            )
            .send()
            .await
            .context("OpenRouter speech request failed")?;
        let status = response.status();
        let media_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("audio/mpeg")
            .to_owned();
        let bytes = response
            .bytes()
            .await
            .context("Failed to read OpenRouter speech response")?;
        if !status.is_success() {
            bail!(
                "OpenRouter speech request failed for model {model}, voice {voice}, format {}: {status}: {}",
                self.config.audio.response_format,
                String::from_utf8_lossy(&bytes[..bytes.len().min(2000)])
            );
        }
        Ok(GeneratedImage {
            bytes: bytes.to_vec(),
            media_type,
            model: model.to_owned(),
            prompt: input.to_owned(),
        })
    }

    async fn resolve_speech_voice(
        &self,
        model: &str,
        configured_voice: &str,
        api_key: &str,
    ) -> String {
        let request = self.request(
            self.client
                .get(format!(
                    "{}/models",
                    self.config.base_url.trim_end_matches('/')
                ))
                .query(&[("output_modalities", "speech")]),
            api_key,
        );
        let Ok(response) = request.send().await else {
            return configured_voice.to_owned();
        };
        let Ok(response) = response.error_for_status() else {
            return configured_voice.to_owned();
        };
        let Ok(value) = response.json::<Value>().await else {
            return configured_voice.to_owned();
        };
        let voices = value
            .get("data")
            .and_then(Value::as_array)
            .and_then(|models| {
                models
                    .iter()
                    .find(|candidate| candidate.get("id").and_then(Value::as_str) == Some(model))
            })
            .and_then(|model| model.get("supported_voices"))
            .and_then(Value::as_array)
            .map(|voices| {
                voices
                    .iter()
                    .filter_map(Value::as_str)
                    .filter(|voice| !voice.trim().is_empty())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        select_speech_voice(configured_voice, &voices)
    }

    pub async fn transcribe_audio(
        &self,
        data: &str,
        format: &str,
        language: Option<&str>,
        model: &str,
        routing: &ModelRouting,
        api_key: &str,
    ) -> Result<String> {
        let mut body = self.config.transcription.extra.clone();
        if let Some(language) = language.or(self.config.transcription.language.as_deref()) {
            body.insert("language".into(), json!(language));
        }
        if let Some(temperature) = self.config.transcription.temperature {
            body.insert("temperature".into(), json!(temperature));
        }
        if let Some(choice) = self
            .config
            .transcription
            .models
            .iter()
            .find(|choice| choice.id == model)
        {
            body.extend(choice.extra.clone());
        }
        let routed_model = if routing.model_provider == ModelProvider::Openrouter {
            apply_routing(&mut body, model, routing, false)
        } else {
            model.to_owned()
        };
        body.insert("model".into(), json!(routed_model));
        body.insert(
            "input_audio".into(),
            json!({"data": data, "format": format}),
        );
        self.post_json_for(
            routing.model_provider,
            "audio/transcriptions",
            Value::Object(body),
            api_key,
        )
        .await?
        .get("text")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .context("OpenRouter transcription returned no text")
    }

    pub async fn generate_video(&self, prompt: &str, api_key: &str) -> Result<String> {
        self.generate_video_with_references(
            prompt,
            &[],
            &self.config.video.model,
            &ModelRouting::default(),
            api_key,
        )
        .await
    }

    pub async fn generate_video_with_references(
        &self,
        prompt: &str,
        media: &[MediaInput],
        model: &str,
        routing: &ModelRouting,
        api_key: &str,
    ) -> Result<String> {
        if routing.model_provider != ModelProvider::Openrouter {
            bail!("AI Hub does not expose an OpenAI-compatible video generation endpoint");
        }
        if model.is_empty() {
            bail!("Video generation model is not configured");
        }
        let mut body = self.config.video.extra.clone();
        body.insert("duration".into(), json!(self.config.video.duration));
        body.insert("aspect_ratio".into(), json!(self.config.video.aspect_ratio));
        body.insert("resolution".into(), json!(self.config.video.resolution));
        body.insert(
            "generate_audio".into(),
            json!(self.config.video.generate_audio),
        );
        if let Some(choice) = self
            .config
            .video
            .models
            .iter()
            .find(|choice| choice.id == model)
        {
            body.extend(choice.extra.clone());
        }
        let routed_model = apply_routing(&mut body, model, routing, false);
        body.insert("model".into(), json!(routed_model));
        body.insert("prompt".into(), json!(prompt));
        let references = media
            .iter()
            .filter_map(|item| match item {
                MediaInput::Image { url } => Some(reference("image_url", url)),
                MediaInput::Video { url } => Some(reference("video_url", url)),
                MediaInput::Audio { .. } => None,
            })
            .collect::<Vec<_>>();
        if !references.is_empty() {
            body.insert("input_references".into(), Value::Array(references));
        }
        let submitted = self
            .post_json("videos", Value::Object(body), api_key)
            .await?;
        let id = submitted
            .get("id")
            .and_then(Value::as_str)
            .context("Video response did not contain a job ID")?;
        let started = Instant::now();
        loop {
            if started.elapsed() > Duration::from_secs(self.config.video.timeout_seconds) {
                bail!("Video generation timed out; OpenRouter job ID: {id}");
            }
            sleep(Duration::from_secs(
                self.config.video.poll_interval_seconds.max(1),
            ))
            .await;
            let state = self.get_json(&format!("videos/{id}"), api_key).await?;
            match state
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
            {
                "completed" => {
                    return state
                        .get("unsigned_urls")
                        .and_then(Value::as_array)
                        .and_then(|v| v.first())
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                        .context("Completed video job returned no URL");
                }
                "failed" | "cancelled" | "expired" => bail!(
                    "Video generation {}: {}",
                    state
                        .get("status")
                        .and_then(Value::as_str)
                        .unwrap_or("failed"),
                    state
                        .get("error")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown error")
                ),
                _ => {}
            }
        }
    }

    async fn post_json(&self, endpoint: &str, body: Value, api_key: &str) -> Result<Value> {
        self.post_json_for(ModelProvider::Openrouter, endpoint, body, api_key)
            .await
    }

    async fn post_json_for(
        &self,
        model_provider: ModelProvider,
        endpoint: &str,
        body: Value,
        api_key: &str,
    ) -> Result<Value> {
        let response = self
            .request_for(
                model_provider,
                self.client
                    .post(format!("{}/{}", self.base_url(model_provider), endpoint))
                    .json(&body),
                api_key,
            )
            .send()
            .await
            .wrap_err_with(|| format!("{} request failed", provider_name(model_provider)))?;
        checked_json(response, provider_name(model_provider)).await
    }

    async fn get_json(&self, endpoint: &str, api_key: &str) -> Result<Value> {
        let response = self
            .request(
                self.client.get(format!(
                    "{}/{}",
                    self.config.base_url.trim_end_matches('/'),
                    endpoint
                )),
                api_key,
            )
            .send()
            .await
            .context("OpenRouter request failed")?;
        checked_json(response, "OpenRouter").await
    }

    fn request(&self, builder: reqwest::RequestBuilder, api_key: &str) -> reqwest::RequestBuilder {
        self.request_for(ModelProvider::Openrouter, builder, api_key)
    }

    fn request_for(
        &self,
        model_provider: ModelProvider,
        builder: reqwest::RequestBuilder,
        api_key: &str,
    ) -> reqwest::RequestBuilder {
        if model_provider == ModelProvider::Aihub {
            return builder.bearer_auth(api_key);
        }
        let builder = builder
            .bearer_auth(api_key)
            .header("X-OpenRouter-Title", &self.config.app_name)
            .header("X-OpenRouter-Metadata", "enabled");
        match &self.config.site_url {
            Some(site) => builder.header("HTTP-Referer", site),
            None => builder,
        }
    }

    fn base_url(&self, model_provider: ModelProvider) -> &str {
        match model_provider {
            ModelProvider::Openrouter => self.config.base_url.trim_end_matches('/'),
            ModelProvider::Aihub => self.aihub.base_url.trim_end_matches('/'),
        }
    }
}

fn select_speech_voice(configured: &str, supported: &[&str]) -> String {
    if supported.is_empty() || supported.contains(&configured) {
        configured.to_owned()
    } else {
        supported[0].to_owned()
    }
}

fn provider_name(model_provider: ModelProvider) -> &'static str {
    match model_provider {
        ModelProvider::Openrouter => "OpenRouter",
        ModelProvider::Aihub => "AI Hub",
    }
}

fn apply_options(body: &mut Map<String, Value>, options: &OpenRouterOptions) {
    macro_rules! optional {
        ($name:literal, $value:expr) => {
            if let Some(value) = $value {
                body.insert($name.into(), json!(value));
            }
        };
    }
    if !options.models.is_empty() {
        body.insert("models".into(), json!(options.models));
    }
    optional!("provider", &options.provider);
    if !options.plugins.is_empty() {
        body.insert("plugins".into(), json!(options.plugins));
    }
    if !options.transforms.is_empty() {
        body.insert("transforms".into(), json!(options.transforms));
    }
    optional!("reasoning", &options.reasoning);
    optional!("response_format", &options.response_format);
    if !options.tools.is_empty() {
        body.insert("tools".into(), json!(options.tools));
    }
    optional!("tool_choice", &options.tool_choice);
    optional!("parallel_tool_calls", options.parallel_tool_calls);
    optional!("cache_control", &options.cache_control);
    optional!("image_config", &options.image_config);
    optional!("logit_bias", &options.logit_bias);
    optional!("max_completion_tokens", options.max_completion_tokens);
    optional!("max_tool_calls", options.max_tool_calls);
    optional!("metadata", &options.metadata);
    optional!("min_p", options.min_p);
    optional!("reasoning_effort", &options.reasoning_effort);
    optional!("repetition_penalty", options.repetition_penalty);
    optional!("route", &options.route);
    optional!("service_tier", &options.service_tier);
    optional!("stop_server_tools_when", &options.stop_server_tools_when);
    optional!("top_a", options.top_a);
    optional!("trace", &options.trace);
    if !options.modalities.is_empty() {
        body.insert("modalities".into(), json!(options.modalities));
    }
    optional!("temperature", options.temperature);
    optional!("top_p", options.top_p);
    optional!("top_k", options.top_k);
    optional!("max_tokens", options.max_tokens);
    optional!("frequency_penalty", options.frequency_penalty);
    optional!("presence_penalty", options.presence_penalty);
    optional!("seed", options.seed);
    optional!("stop", &options.stop);
    optional!("logprobs", options.logprobs);
    optional!("top_logprobs", options.top_logprobs);
    optional!("user", &options.user);
    body.extend(options.extra.clone());
}

fn apply_routing(
    body: &mut Map<String, Value>,
    model: &str,
    routing: &ModelRouting,
    exacto_supported: bool,
) -> String {
    let mut provider = body
        .remove("provider")
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    match routing.strategy.as_str() {
        "cheapest" => {
            provider.insert("sort".into(), json!("price"));
        }
        "throughput" => {
            provider.insert("sort".into(), json!("throughput"));
        }
        "latency" => {
            provider.insert("sort".into(), json!("latency"));
        }
        _ => {}
    }
    if let Some(slug) = routing.provider.as_deref() {
        provider.insert("only".into(), json!([slug]));
    }
    if !provider.is_empty() {
        body.insert("provider".into(), Value::Object(provider));
    }
    if exacto_supported && routing.strategy == "exacto" && !model.ends_with(":exacto") {
        format!("{model}:exacto")
    } else {
        model.to_owned()
    }
}

struct ToolContext<'a> {
    search_provider: SearchProvider,
    search_ready: bool,
    web_search: &'a crate::config::OpenRouterWebSearchConfig,
    web_fetch: &'a crate::config::OpenRouterWebFetchConfig,
    audio_attached: bool,
    openrouter_server_tools: bool,
}

fn add_tools(body: &mut Map<String, Value>, capabilities: &Capabilities, context: ToolContext<'_>) {
    let mut additions = Vec::new();
    if capabilities.search && context.search_ready {
        if context.search_provider == SearchProvider::Openrouter && context.openrouter_server_tools
        {
            additions.push(openrouter_web_search_tool(context.web_search));
        } else {
            additions.push(json!({
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the live web when current or sourced information is needed.",
                    "parameters": {
                        "type": "object",
                        "properties": { "query": { "type": "string" } },
                        "required": ["query"]
                    }
                }
            }));
        }
    }
    if capabilities.web_fetch && context.openrouter_server_tools {
        additions.push(openrouter_web_fetch_tool(context.web_fetch));
    }
    if capabilities.image {
        additions.push(function_tool(
            "generate_image",
            "Generate an image and deliver it to Telegram.",
            "prompt",
        ));
    }
    if capabilities.audio {
        additions.push(function_tool(
            "generate_audio",
            "Generate spoken audio and deliver it to Telegram.",
            "text",
        ));
    }
    if capabilities.video {
        additions.push(function_tool(
            "generate_video",
            "Generate a video and deliver it to Telegram.",
            "prompt",
        ));
    }
    if capabilities.transcription && context.audio_attached {
        additions.push(json!({
            "type":"function",
            "function": {
                "name":"transcribe_audio",
                "description":"Transcribe the attached Telegram voice note or audio file.",
                "parameters": {
                    "type":"object",
                    "properties": {"language":{"type":"string","description":"Optional ISO-639-1 language hint"}}
                }
            }
        }));
    }
    if capabilities.file {
        additions.push(json!({
            "type":"function",
            "function": {
                "name":"send_file",
                "description":"Deliver a long answer, source code, configuration, or structured text as a downloadable Telegram file.",
                "parameters": {
                    "type":"object",
                    "properties": {
                        "filename":{"type":"string","description":"Safe filename including an appropriate extension"},
                        "content":{"type":"string","description":"Complete UTF-8 file content"}
                    },
                    "required":["filename","content"]
                }
            }
        }));
    }
    if additions.is_empty() {
        return;
    }
    match body.get_mut("tools") {
        Some(Value::Array(tools)) => tools.extend(additions),
        _ => {
            body.insert("tools".into(), Value::Array(additions));
        }
    }
    body.entry("tool_choice").or_insert(json!("auto"));
}

fn report_progress(progress: &Option<UnboundedSender<ProgressUpdate>>, status: &str) {
    if let Some(progress) = progress {
        let _ = progress.send(ProgressUpdate::step(status));
    }
}

fn report_generation_progress(
    progress: &Option<UnboundedSender<ProgressUpdate>>,
    kind: &'static str,
    model: &str,
    prompt: &str,
) {
    if let Some(progress) = progress {
        let _ = progress.send(ProgressUpdate::generation(kind, model, prompt));
    }
}

fn enabled_planner_skills(capabilities: &Capabilities) -> Vec<&'static str> {
    [
        (true, "generate_code"),
        (capabilities.search, "search"),
        (capabilities.web_fetch, "web_fetch"),
        (capabilities.image, "image_generation"),
        (capabilities.audio, "audio_generation"),
        (capabilities.video, "video_generation"),
        (capabilities.media, "image_understanding"),
        (capabilities.media, "video_understanding"),
        (capabilities.transcription, "transcription"),
        (capabilities.file, "file_delivery"),
        (capabilities.model_upgrade, "model_upgrade"),
    ]
    .into_iter()
    .filter_map(|(enabled, name)| enabled.then_some(name))
    .collect()
}

fn planned_action_enabled(action: PlannedAction, capabilities: &Capabilities) -> bool {
    match action {
        PlannedAction::GenerateImage => capabilities.image,
        PlannedAction::GenerateAudio => capabilities.audio,
        PlannedAction::GenerateVideo => capabilities.video,
        PlannedAction::Chat | PlannedAction::GenerateCode | PlannedAction::Refuse => true,
    }
}

fn file_from_arguments(arguments: &Value) -> Result<GeneratedFile> {
    const MAX_FILE_BYTES: usize = 20 * 1024 * 1024;
    let raw_name = arguments
        .get("filename")
        .and_then(Value::as_str)
        .context("File tool requires a filename")?;
    let filename = safe_filename(raw_name).context("File name must contain safe characters")?;
    let bytes = arguments
        .get("content")
        .and_then(Value::as_str)
        .context("File tool requires content")?
        .as_bytes()
        .to_vec();
    if bytes.is_empty() || bytes.len() > MAX_FILE_BYTES {
        bail!("File content must be between 1 byte and 20 MiB");
    }
    Ok(GeneratedFile { filename, bytes })
}

fn safe_filename(raw_name: &str) -> Option<String> {
    let filename = raw_name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('.')
        .to_owned();
    (!filename.is_empty() && filename.len() <= 128).then_some(filename)
}

fn materialize_file_answer(
    text: String,
    file_enabled: bool,
    generated_files: &mut Vec<GeneratedFile>,
) -> String {
    const LARGE_ANSWER_CHARS: usize = 8_000;
    const CODE_FILE_CHARS: usize = 512;
    if !file_enabled || !generated_files.is_empty() {
        return text;
    }
    if text.chars().count() > LARGE_ANSWER_CHARS {
        generated_files.push(GeneratedFile {
            filename: "answer.md".to_owned(),
            bytes: text.as_bytes().to_vec(),
        });
        return "The complete answer is attached as `answer.md`.".to_owned();
    }
    let Some((language, code)) = largest_fenced_code(&text) else {
        return text;
    };
    if code.chars().count() < CODE_FILE_CHARS {
        return text;
    }
    let extension = match language.to_ascii_lowercase().as_str() {
        "html" | "htm" => "html",
        "css" => "css",
        "javascript" | "js" => "js",
        "typescript" | "ts" => "ts",
        "rust" | "rs" => "rs",
        "python" | "py" => "py",
        "bash" | "sh" | "shell" => "sh",
        "json" => "json",
        "yaml" | "yml" => "yaml",
        "toml" => "toml",
        "sql" => "sql",
        "markdown" | "md" => "md",
        _ => "txt",
    };
    let filename = format!("answer.{extension}");
    generated_files.push(GeneratedFile {
        filename: filename.clone(),
        bytes: code.into_bytes(),
    });
    format!("The generated code is attached as `{filename}`.")
}

fn largest_fenced_code(text: &str) -> Option<(String, String)> {
    let mut current: Option<(String, String)> = None;
    let mut largest: Option<(String, String)> = None;
    for line in text.lines() {
        if let Some((language, code)) = current.as_mut() {
            if line.trim() == "```" {
                let completed = current.take().expect("current fence exists");
                if largest
                    .as_ref()
                    .is_none_or(|(_, largest_code)| completed.1.len() > largest_code.len())
                {
                    largest = Some(completed);
                }
            } else {
                code.push_str(line);
                code.push('\n');
                let _ = language;
            }
        } else if let Some(language) = line.trim_start().strip_prefix("```") {
            current = Some((language.trim().to_owned(), String::new()));
        }
    }
    largest
}

fn openrouter_web_search_tool(config: &crate::config::OpenRouterWebSearchConfig) -> Value {
    let mut parameters = config.extra.clone();
    parameters.insert("engine".into(), json!(config.engine));
    if let Some(value) = config.mode.as_deref() {
        parameters.insert("mode".into(), json!(value));
    }
    parameters.insert("max_results".into(), json!(config.max_results));
    if let Some(value) = config.max_uses {
        parameters.insert("max_uses".into(), json!(value));
    }
    parameters.insert("max_total_results".into(), json!(config.max_total_results));
    parameters.insert(
        "search_context_size".into(),
        json!(config.search_context_size),
    );
    if let Some(value) = config.max_characters {
        parameters.insert("max_characters".into(), json!(value));
    }
    if let Some(value) = config.user_location.as_ref() {
        parameters.insert("user_location".into(), value.clone());
    }
    if !config.allowed_domains.is_empty() {
        parameters.insert("allowed_domains".into(), json!(config.allowed_domains));
    }
    if !config.excluded_domains.is_empty() {
        parameters.insert("excluded_domains".into(), json!(config.excluded_domains));
    }
    json!({"type":"openrouter:web_search","parameters":parameters})
}

fn openrouter_web_fetch_tool(config: &crate::config::OpenRouterWebFetchConfig) -> Value {
    let mut parameters = config.extra.clone();
    parameters.insert("engine".into(), json!(config.engine));
    if let Some(value) = config.max_uses {
        parameters.insert("max_uses".into(), json!(value));
    }
    if let Some(value) = config.max_content_tokens {
        parameters.insert("max_content_tokens".into(), json!(value));
    }
    if !config.allowed_domains.is_empty() {
        parameters.insert("allowed_domains".into(), json!(config.allowed_domains));
    }
    if !config.blocked_domains.is_empty() {
        parameters.insert("blocked_domains".into(), json!(config.blocked_domains));
    }
    json!({"type":"openrouter:web_fetch","parameters":parameters})
}

fn function_tool(name: &str, description: &str, argument: &str) -> Value {
    let mut properties = Map::new();
    properties.insert(argument.into(), json!({"type":"string"}));
    json!({"type":"function","function":{"name":name,"description":description,"parameters":{"type":"object","properties":properties,"required":[argument]}}})
}

fn user_content(text: &str, media: &[MediaInput], media_enabled: bool) -> Value {
    if media.is_empty() {
        return Value::String(text.to_owned());
    }
    let mut content = vec![json!({"type":"text","text":text})];
    for item in media {
        match item {
            MediaInput::Image { url } if media_enabled => {
                content.push(json!({"type":"image_url","image_url":{"url":url}}));
            }
            MediaInput::Video { url } if media_enabled => {
                content.push(json!({"type":"video_url","video_url":{"url":url}}));
            }
            MediaInput::Audio { .. } => content.push(json!({
                "type":"text",
                "text":"A private audio attachment is available through the transcribe_audio tool."
            })),
            _ => {}
        }
    }
    Value::Array(content)
}

fn reference(kind: &str, url: &str) -> Value {
    let mut value = Map::new();
    value.insert("type".into(), Value::String(kind.to_owned()));
    value.insert(kind.into(), json!({"url":url}));
    Value::Object(value)
}

fn extract_content(message: &Value) -> (String, Vec<String>) {
    let mut text = String::new();
    let mut urls = Vec::new();
    match message.get("content") {
        Some(Value::String(value)) => text = value.clone(),
        Some(Value::Array(items)) => {
            for item in items {
                if let Some(value) = item.get("text").and_then(Value::as_str) {
                    text.push_str(value);
                }
                if let Some(url) = item.pointer("/image_url/url").and_then(Value::as_str) {
                    urls.push(url.to_owned());
                }
            }
        }
        _ => {}
    }
    if let Some(images) = message.get("images").and_then(Value::as_array) {
        for image in images {
            if let Some(url) = image
                .pointer("/image_url/url")
                .or_else(|| image.get("url"))
                .and_then(Value::as_str)
            {
                urls.push(url.to_owned());
            }
        }
    }
    let citations = message
        .get("annotations")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|annotation| {
            let citation = annotation.get("url_citation")?;
            Some((
                citation
                    .get("title")
                    .and_then(Value::as_str)
                    .unwrap_or("Source"),
                citation.get("url")?.as_str()?,
            ))
        })
        .collect::<std::collections::BTreeSet<_>>();
    if !citations.is_empty() {
        text.push_str("\n\n### Sources\n");
        for (title, url) in citations {
            text.push_str(&format!("- [{title}]({url})\n"));
        }
    }
    (text, urls)
}

fn extract_refusal(message: &Value) -> Option<String> {
    message
        .get("refusal")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .or_else(|| {
            message
                .get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .find(|part| part.get("type").and_then(Value::as_str) == Some("refusal"))
                .and_then(|part| part.get("refusal").or_else(|| part.get("text")))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
        })
}

fn parse_planner_response(value: &Value) -> Result<RequestPlan> {
    if let Some(message) = value
        .pointer("/error/message")
        .and_then(Value::as_str)
        .filter(|message| !message.trim().is_empty())
    {
        bail!("OpenRouter request planner returned an error: {message}");
    }
    let message = value
        .pointer("/choices/0/message")
        .context("OpenRouter request planner returned no message")?;
    if let Some(refusal) = extract_refusal(message) {
        return Ok(RequestPlan {
            action: PlannedAction::Refuse,
            skills: Vec::new(),
            delivery: PlannedDelivery::Inline,
            filename: String::new(),
            refusal_message: refusal,
        });
    }

    for field in ["parsed", "structured_output"] {
        if let Some(document @ Value::Object(_)) = message.get(field) {
            return serde_json::from_value(document.clone()).wrap_err_with(|| {
                format!("OpenRouter request planner returned invalid {field} content")
            });
        }
    }

    let content = message.get("content").unwrap_or(&Value::Null);
    if let Value::Object(document) = content {
        return serde_json::from_value(Value::Object(document.clone()))
            .wrap_err("OpenRouter request planner returned invalid object content");
    }
    let text = match content {
        Value::String(text) => text.clone(),
        Value::Array(_) => extract_content(message).0,
        _ => String::new(),
    };
    if text.trim().is_empty() {
        let finish_reason = value
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let model = value
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let reasoning_present = message
            .get("reasoning")
            .is_some_and(|reasoning| !reasoning.is_null());
        bail!(
            "OpenRouter request planner returned no structured content (response_model={model}, finish_reason={finish_reason}, content_type={}, reasoning_present={reasoning_present})",
            json_type(content)
        );
    }
    parse_plan_json(&text)
}

fn parse_plan_json(text: &str) -> Result<RequestPlan> {
    let trimmed = text.trim();
    if let Ok(plan) = serde_json::from_str(trimmed) {
        return Ok(plan);
    }
    let unwrapped = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```JSON"))
        .or_else(|| trimmed.strip_prefix("```"))
        .and_then(|body| body.strip_suffix("```"))
        .map(str::trim);
    if let Some(document) = unwrapped {
        if let Ok(plan) = serde_json::from_str(document) {
            return Ok(plan);
        }
    }
    let document = trimmed
        .find('{')
        .zip(trimmed.rfind('}'))
        .filter(|(start, end)| start < end)
        .map(|(start, end)| &trimmed[start..=end])
        .context("OpenRouter request planner returned no JSON object")?;
    serde_json::from_str(document)
        .wrap_err("OpenRouter request planner returned invalid structured content")
}

fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

async fn checked_json(response: reqwest::Response, provider: &str) -> Result<Value> {
    let status = response.status();
    let bytes = response
        .bytes()
        .await
        .wrap_err_with(|| format!("Failed to read {provider} response"))?;
    if !status.is_success() {
        bail!(
            "{provider} returned {status}: {}",
            String::from_utf8_lossy(&bytes[..bytes.len().min(2000)])
        );
    }
    serde_json::from_slice(&bytes).wrap_err_with(|| format!("{provider} returned invalid JSON"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_options_override_defaults_and_keep_passthrough() {
        let mut body = Map::new();
        let mut first = OpenRouterOptions {
            temperature: Some(0.1),
            ..Default::default()
        };
        first.extra.insert("usage".into(), json!({"include": true}));
        apply_options(&mut body, &first);
        apply_options(
            &mut body,
            &OpenRouterOptions {
                temperature: Some(0.8),
                ..Default::default()
            },
        );
        assert_eq!(body["temperature"], json!(0.8));
        assert_eq!(body["usage"], json!({"include": true}));
    }

    #[test]
    fn capability_switches_control_tool_schemas() {
        let mut body = Map::new();
        let capabilities = Capabilities {
            image: false,
            video: false,
            ..Default::default()
        };
        add_tools(
            &mut body,
            &capabilities,
            ToolContext {
                search_provider: SearchProvider::Brave,
                search_ready: true,
                web_search: &crate::config::OpenRouterWebSearchConfig::default(),
                web_fetch: &crate::config::OpenRouterWebFetchConfig::default(),
                audio_attached: true,
                openrouter_server_tools: true,
            },
        );
        let names = body["tools"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|tool| tool.pointer("/function/name").and_then(Value::as_str))
            .collect::<Vec<_>>();
        assert!(names.contains(&"web_search"));
        assert!(
            body["tools"]
                .as_array()
                .unwrap()
                .iter()
                .any(|tool| tool["type"] == "openrouter:web_fetch")
        );
        assert!(names.contains(&"generate_audio"));
        assert!(names.contains(&"transcribe_audio"));
        assert!(!names.contains(&"generate_image"));
        assert!(!names.contains(&"generate_video"));
    }

    #[test]
    fn non_openrouter_chat_omits_openrouter_server_tools() {
        let mut body = Map::new();
        add_tools(
            &mut body,
            &Capabilities::default(),
            ToolContext {
                search_provider: SearchProvider::Openrouter,
                search_ready: true,
                web_search: &crate::config::OpenRouterWebSearchConfig::default(),
                web_fetch: &crate::config::OpenRouterWebFetchConfig::default(),
                audio_attached: false,
                openrouter_server_tools: false,
            },
        );
        assert!(body["tools"].as_array().unwrap().iter().all(|tool| {
            !tool["type"]
                .as_str()
                .is_some_and(|kind| kind.starts_with("openrouter:"))
        }));
    }

    #[test]
    fn private_media_uses_multimodal_content() {
        let content = user_content(
            "Describe this",
            &[MediaInput::Image {
                url: "data:image/png;base64,AA==".into(),
            }],
            true,
        );
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image_url");
    }

    #[test]
    fn planner_actions_are_bounded_by_admin_capabilities() {
        let capabilities = Capabilities {
            image: false,
            ..Default::default()
        };
        assert!(!planned_action_enabled(
            PlannedAction::GenerateImage,
            &capabilities
        ));
        assert!(planned_action_enabled(PlannedAction::Chat, &capabilities));
        assert!(!enabled_planner_skills(&capabilities).contains(&"image_generation"));
    }

    #[test]
    fn provider_refusal_is_extracted_without_phrase_guessing() {
        let message = json!({"role":"assistant", "content":null, "refusal":"I can’t do that."});
        assert_eq!(
            extract_refusal(&message).as_deref(),
            Some("I can’t do that.")
        );
    }

    #[test]
    fn planner_accepts_string_array_object_and_fenced_responses() {
        let document = json!({
            "action":"generate_image",
            "skills":["image_generation"],
            "delivery":"inline",
            "filename":"",
            "refusal_message":""
        });
        let string_response = json!({
            "choices":[{"message":{"content":document.to_string()},"finish_reason":"stop"}]
        });
        let array_response = json!({
            "choices":[{"message":{"content":[{"type":"output_text","text":document.to_string()}]},"finish_reason":"stop"}]
        });
        let object_response = json!({
            "choices":[{"message":{"parsed":document},"finish_reason":"stop"}]
        });
        let fenced_response = json!({
            "choices":[{"message":{"content":format!("```json\n{}\n```", document)},"finish_reason":"stop"}]
        });
        for response in [
            string_response,
            array_response,
            object_response,
            fenced_response,
        ] {
            assert_eq!(
                parse_planner_response(&response).unwrap().action,
                PlannedAction::GenerateImage
            );
        }
    }

    #[test]
    fn planner_empty_content_error_has_safe_shape_diagnostics() {
        let response = json!({
            "model":"free-model",
            "choices":[{"message":{"content":null,"reasoning":"hidden"},"finish_reason":"length"}]
        });
        let error = parse_planner_response(&response).unwrap_err().to_string();
        assert!(error.contains("finish_reason=length"));
        assert!(error.contains("content_type=null"));
        assert!(!error.contains("hidden"));
    }

    #[test]
    fn planner_generation_skill_recovers_a_chat_action() {
        let plan = RequestPlan {
            action: PlannedAction::Chat,
            skills: vec![PlannedSkill::ImageGeneration],
            delivery: PlannedDelivery::Inline,
            filename: String::new(),
            refusal_message: String::new(),
        };
        assert_eq!(plan.direct_generation(), Some(PlannedAction::GenerateImage));
    }

    #[test]
    fn planner_accepts_generate_code_as_normal_assistant_work() {
        let response = json!({
            "choices":[{"message":{"content":json!({
                "action":"generate_code",
                "skills":["generate_code", "file_delivery"],
                "delivery":"file",
                "filename":"main.rs",
                "refusal_message":""
            }).to_string()},"finish_reason":"stop"}]
        });
        let plan = parse_planner_response(&response).unwrap();
        assert_eq!(plan.action, PlannedAction::GenerateCode);
        assert!(plan.skills.contains(&PlannedSkill::GenerateCode));
        assert_eq!(plan.direct_generation(), None);
        assert!(planned_action_enabled(
            plan.action,
            &Capabilities::default()
        ));
    }

    #[test]
    fn speech_voice_prefers_configured_when_supported_and_falls_back_safely() {
        assert_eq!(select_speech_voice("nova", &["alloy", "nova"]), "nova");
        assert_eq!(
            select_speech_voice("nova", &["flux-alexis-en", "flux-bree-en"]),
            "flux-alexis-en"
        );
        assert_eq!(select_speech_voice("custom", &[]), "custom");
    }

    #[test]
    fn planner_generation_recovery_never_indexes_an_empty_or_ambiguous_list() {
        let mut plan = RequestPlan {
            action: PlannedAction::Chat,
            skills: Vec::new(),
            delivery: PlannedDelivery::Inline,
            filename: String::new(),
            refusal_message: String::new(),
        };
        assert_eq!(plan.direct_generation(), None);
        plan.skills = vec![PlannedSkill::ImageGeneration, PlannedSkill::VideoGeneration];
        assert_eq!(plan.direct_generation(), None);
    }

    #[test]
    fn file_tool_sanitizes_names_and_limits_content() {
        let file = file_from_arguments(&json!({
            "filename": "../answer file.rs",
            "content": "fn main() {}"
        }))
        .unwrap();
        assert_eq!(file.filename, "_answer_file.rs");
        assert_eq!(file.bytes, b"fn main() {}");
    }

    #[test]
    fn substantial_code_answers_are_materialized_as_files() {
        let code = format!("<!doctype html>\n{}", "<main>content</main>\n".repeat(40));
        let answer = format!("Here is the page:\n\n```html\n{code}```");
        let mut files = Vec::new();
        let summary = materialize_file_answer(answer, true, &mut files);
        assert_eq!(summary, "The generated code is attached as `answer.html`.");
        assert_eq!(files[0].filename, "answer.html");
        assert_eq!(files[0].bytes, code.as_bytes());
    }

    #[test]
    fn large_answers_are_materialized_as_markdown_files() {
        let mut files = Vec::new();
        let summary = materialize_file_answer("x".repeat(8_001), true, &mut files);
        assert_eq!(summary, "The complete answer is attached as `answer.md`.");
        assert_eq!(files[0].filename, "answer.md");
    }
}
