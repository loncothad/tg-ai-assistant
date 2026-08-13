//! OpenRouter chat/tool, image, speech, and asynchronous video clients.

use std::time::{Duration, Instant};

use base64::{Engine, engine::general_purpose::STANDARD};
use eyre::{Context, ContextCompat, bail};
use serde_json::{Map, Value, json};
use tokio::time::sleep;

use crate::{
    Result,
    config::{ModelConfig, OpenRouterConfig, OpenRouterOptions, SearchProvider},
    db::{Capabilities, ChatMessage, ModelRouting},
    search::SearchService,
};

const MAX_TOOL_ROUNDS: usize = 6;

#[derive(Clone)]
pub struct OpenRouter {
    client: reqwest::Client,
    config: OpenRouterConfig,
}

#[derive(Clone, Debug)]
pub struct AssistantResponse {
    pub text: String,
    pub media_urls: Vec<String>,
    pub generation_id: Option<String>,
    pub usage: Option<Value>,
    pub generated_images: Vec<GeneratedImage>,
    pub generated_audio: Vec<GeneratedImage>,
    pub generated_videos: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct GeneratedImage {
    pub bytes: Vec<u8>,
    pub media_type: String,
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
    pub capabilities: &'a Capabilities,
    pub routing: &'a ModelRouting,
    pub tool_models: ToolModels<'a>,
}

#[derive(Clone, Copy)]
pub struct ToolModels<'a> {
    pub image_generation: &'a str,
    pub image_routing: &'a ModelRouting,
    pub audio_generation: &'a str,
    pub audio_routing: &'a ModelRouting,
    pub transcription: &'a str,
    pub transcription_routing: &'a ModelRouting,
    pub video_generation: &'a str,
    pub video_routing: &'a ModelRouting,
}

impl OpenRouter {
    pub fn new(client: reqwest::Client, config: OpenRouterConfig) -> Self {
        Self { client, config }
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
            capabilities,
            routing,
            tool_models,
        } = request;
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
        for _ in 0..MAX_TOOL_ROUNDS {
            let mut body = Map::new();
            body.insert("messages".into(), Value::Array(messages.clone()));
            body.insert("session_id".into(), json!(session_id));
            apply_options(&mut body, &self.config.defaults);
            apply_options(&mut body, &model.options);
            let routed_model = apply_routing(&mut body, &model.id, routing, true);
            body.insert("model".into(), Value::String(routed_model));
            add_tools(
                &mut body,
                capabilities,
                search_provider,
                search_api_key.is_some(),
                &self.config.web_search,
                &self.config.web_fetch,
                media
                    .iter()
                    .any(|item| matches!(item, MediaInput::Audio { .. })),
            );

            let value = self
                .post_json("chat/completions", Value::Object(body), api_key)
                .await?;
            let message = value
                .pointer("/choices/0/message")
                .cloned()
                .context("OpenRouter returned no assistant message")?;
            let calls = message
                .get("tool_calls")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            if calls.is_empty() {
                let (text, media_urls) = extract_content(&message);
                if text.is_empty() && media_urls.is_empty() {
                    bail!("OpenRouter returned an empty response");
                }
                return Ok(AssistantResponse {
                    text,
                    media_urls,
                    generation_id: value.get("id").and_then(Value::as_str).map(str::to_owned),
                    usage: value.get("usage").cloned(),
                    generated_images,
                    generated_audio,
                    generated_videos,
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
                        let query = arguments
                            .get("query")
                            .and_then(Value::as_str)
                            .unwrap_or(user_message);
                        search
                            .tool_output(search_provider, query, search_api_key.unwrap_or_default())
                            .await
                    }
                    "generate_image" if capabilities.image => {
                        let prompt = arguments
                            .get("prompt")
                            .and_then(Value::as_str)
                            .unwrap_or(user_message);
                        match self
                            .generate_image_with_references(
                                prompt,
                                media,
                                tool_models.image_generation,
                                tool_models.image_routing,
                                api_key,
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
                        match self
                            .generate_audio(
                                input,
                                tool_models.audio_generation,
                                tool_models.audio_routing,
                                api_key,
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
                        match self
                            .generate_video_with_references(
                                prompt,
                                media,
                                tool_models.video_generation,
                                tool_models.video_routing,
                                api_key,
                            )
                            .await
                        {
                            Ok(value) => {
                                generated_videos.push(value);
                                json!({"status":"completed","videos":1}).to_string()
                            }
                            Err(error) => json!({"error":error.to_string()}).to_string(),
                        }
                    }
                    "transcribe_audio" if capabilities.transcription => {
                        let language = arguments.get("language").and_then(Value::as_str);
                        let mut transcripts = Vec::new();
                        for item in media {
                            if let MediaInput::Audio { data, format } = item {
                                match self
                                    .transcribe_audio(
                                        data,
                                        format,
                                        language,
                                        tool_models.transcription,
                                        tool_models.transcription_routing,
                                        api_key,
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
                    _ => {
                        json!({ "error": format!("Unknown or disabled tool: {name}") }).to_string()
                    }
                };
                messages
                    .push(json!({ "role": "tool", "tool_call_id": call_id, "content": output }));
            }
        }
        bail!("OpenRouter exceeded the maximum tool-call rounds")
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
        if self.config.image.model.is_empty() {
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
        let routed_model = apply_routing(&mut body, model, routing, false);
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
            body.insert("input_references".into(), Value::Array(references));
        }
        let value = self
            .post_json("images", Value::Object(body), api_key)
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
        body.insert("voice".into(), json!(self.config.audio.voice));
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
        let routed_model = apply_routing(&mut body, model, routing, false);
        body.insert("model".into(), json!(routed_model));
        body.insert("input".into(), json!(input));
        let response = self
            .request(
                self.client
                    .post(format!(
                        "{}/audio/speech",
                        self.config.base_url.trim_end_matches('/')
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
                "OpenRouter returned {status}: {}",
                String::from_utf8_lossy(&bytes[..bytes.len().min(2000)])
            );
        }
        Ok(GeneratedImage {
            bytes: bytes.to_vec(),
            media_type,
        })
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
        let routed_model = apply_routing(&mut body, model, routing, false);
        body.insert("model".into(), json!(routed_model));
        body.insert(
            "input_audio".into(),
            json!({"data": data, "format": format}),
        );
        self.post_json("audio/transcriptions", Value::Object(body), api_key)
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
        if self.config.video.model.is_empty() {
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
        let response = self
            .request(
                self.client
                    .post(format!(
                        "{}/{}",
                        self.config.base_url.trim_end_matches('/'),
                        endpoint
                    ))
                    .json(&body),
                api_key,
            )
            .send()
            .await
            .context("OpenRouter request failed")?;
        checked_json(response).await
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
        checked_json(response).await
    }

    fn request(&self, builder: reqwest::RequestBuilder, api_key: &str) -> reqwest::RequestBuilder {
        let builder = builder
            .bearer_auth(api_key)
            .header("X-OpenRouter-Title", &self.config.app_name)
            .header("X-OpenRouter-Metadata", "enabled");
        match &self.config.site_url {
            Some(site) => builder.header("HTTP-Referer", site),
            None => builder,
        }
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

fn add_tools(
    body: &mut Map<String, Value>,
    capabilities: &Capabilities,
    search_provider: SearchProvider,
    search_ready: bool,
    web_search: &crate::config::OpenRouterWebSearchConfig,
    web_fetch: &crate::config::OpenRouterWebFetchConfig,
    audio_attached: bool,
) {
    let mut additions = Vec::new();
    if capabilities.search && search_ready {
        if search_provider == SearchProvider::Openrouter {
            additions.push(openrouter_web_search_tool(web_search));
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
    if capabilities.web_fetch {
        additions.push(openrouter_web_fetch_tool(web_fetch));
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
    if capabilities.transcription && audio_attached {
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

async fn checked_json(response: reqwest::Response) -> Result<Value> {
    let status = response.status();
    let bytes = response
        .bytes()
        .await
        .context("Failed to read OpenRouter response")?;
    if !status.is_success() {
        bail!(
            "OpenRouter returned {status}: {}",
            String::from_utf8_lossy(&bytes[..bytes.len().min(2000)])
        );
    }
    serde_json::from_slice(&bytes).context("OpenRouter returned invalid JSON")
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
            SearchProvider::Brave,
            true,
            &crate::config::OpenRouterWebSearchConfig::default(),
            &crate::config::OpenRouterWebFetchConfig::default(),
            true,
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
}
