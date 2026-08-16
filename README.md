# Teleforge

Teleforge is a production-oriented, multi-bot Telegram AI assistant written in Rust with [`frankenstein`](https://crates.io/crates/frankenstein). It sends Telegram Rich Messages, supports OpenRouter, the OpenAI-compatible AI Hub API, and live-discovered fal.ai endpoints, supports Brave Search, Exa, and Google results through SerpAPI, and stores all runtime state in a local redb database.

## Capabilities

- Any number of Telegram bot tokens in one process. Each configured bot ID has isolated settings, encrypted API credentials, history, access rules, audit records, and update offsets.
- Private-message, group/thread, and Telegram guest-bot modes. Group handling can require a mention or reply.
- OpenRouter chat options including fallback model lists, provider routing, server tools, plugins, transforms, reasoning, structured responses, modalities, sampling controls, usage, and forward-compatible fields under `extra`.
- AI Hub chat/tool calling and image generation through its OpenAI-compatible endpoints. Its authenticated `/v1/models` catalog and encrypted per-bot credential are independent from OpenRouter.
- fal.ai queue and direct endpoints for generation, transcription, 3D/vector work, and image/video vision models. The Platform API supplies a paginated live catalog, capability categories, account pricing, and per-model OpenAPI schemas. Teleforge lazily derives executable prompt/media/default/output mappings from those schemas and caches them; optional YAML entries override unusual or private endpoints.
- OpenRouter server-side Web Search and Web Fetch. Fetch is a separately toggleable built-in skill that can retrieve page and PDF text, with engine, usage/content limits, and domain policy configured in YAML.
- Real model tools named `web_search`, `generate_image`, `generate_speech`, `generate_music`, `generate_video`, and attachment-scoped `transcribe_audio`. The AI can invoke them during an ordinary conversation; the backend executes the selected API and delivers generated media to Telegram.
- Continuous long polling dispatches received updates independently behind a configurable per-bot semaphore. A long video job does not delay polling, while bursts cannot saturate every CPU core; immediately returning empty polls receive a short delay to prevent idle busy loops.
- Each in-flight request is cancellation-tracked. Deleting either the caller's message or the bot's live processing message cancels its planner/provider future within five seconds and suppresses final delivery. Telegram business-message deletion updates cancel immediately by message ID; ordinary messages are checked with a no-op empty-reaction probe because the standard Bot API does not emit their deletion events.
- Live reply-linked Rich Messages are created before intent classification and retain the completed processing steps while assistant and media-generation requests run; generation steps disclose the exact effective prompt and model, and guest mode immediately creates and updates its pending inline result. A bounded OpenRouter structured-output classifier selects actions without producing or rewriting downstream prompts; explicit commands remain deterministic.
- A toggleable `send_file` tool lets the model deliver source code, configuration, structured text, or an unwieldy answer as a named Telegram document. Answers above 8,000 characters automatically become `answer.md`, and substantial fenced code becomes a language-appropriate file even when the model forgets to call the tool.
- Direct `/search`, `/image`, `/speech`, `/music`, `/transcribe`, and `/video` commands. `/audio` remains an alias for `/speech`. Every command also accepts the `-COMMAND` form (case-insensitive), including query and guest requests.
- Reply context is preserved in private, group, and guest modes, including Telegram quote-only and external-reply updates. Current/replied images, videos, and audio can drive supported transformations: image-to-image, image/video/audio-to-video, image/video/audio-to-music, audio-reference speech/voice cloning, and video/audio-to-image through faithful understanding/transcription preprocessing.
- An authenticated Telegram Mini App admin panel built with HTMX for model/provider selection, per-skill switches, API-key management, custom prompt/skill import, skill-bundle export/import, and user allowlisting. Its responsive blue interface uses a local fuzzy-search picker instead of rendering enormous model dropdowns.
- Native Telegram Rich Message responses and rich guest-query results with CommonMark-to-Rich-Markdown normalization, Unicode-safe limits, block limits, and code-fence repair across chunks.
- Bounded per-bot concurrency, timeouts, polling backoff, graceful shutdown, structured logs, and health endpoints.

## Build-time defaults and runtime customization

The normal repository files [`defaults/system.md`](defaults/system.md) and [`defaults/skills/`](defaults/skills/) are embedded into the binary with `include_str!`. Each built-in skill has its own file and a Rust-side description. Edit those files and rebuild to change the shipped defaults.

Built-in skills are enabled by default. Disabling one in the admin panel removes its instructions and, where applicable, its callable tool schema from subsequent AI requests. Media understanding and transcription have independent switches from media generation. Runtime custom prompts and skill instructions live only in redb, can be enabled/disabled/reset independently, and never alter the embedded defaults. Imported custom skills may direct the model to the enabled built-in tool names; importing arbitrary executable code or arbitrary HTTP endpoints is deliberately unsupported.

Skill export produces a versioned JSON bundle containing the built-in descriptions/instructions, enabled states, and custom skill text. Uploading that JSON in the skill import form restores the toggles and custom text. Built-in instruction text remains build-time immutable; edit the files and rebuild to replace it.

## Quick start

Requirements: Rust 1.85+, one or more BotFather tokens, an HTTPS public origin for the Telegram Mini App, and at least one OpenRouter or AI Hub key. Search keys are optional.

```sh
cp config.example.yaml config.yaml
cp .env.example .env
openssl rand -base64 32  # place this once in TELEFORGE_MASTER_KEY
set -a
. ./.env
set +a
cargo test
cargo run --release -- --config config.yaml
```

Replace the example Telegram user IDs and bot tokens before starting. `TELEGRAM_BOT_TOKENS` is one comma-separated value; tokens map in order to enabled `bots` entries, while disabled entries consume no token. The application does not parse `.env` itself. [`config.example.yaml`](config.example.yaml) is exhaustive and supports `${NAME}` and `${NAME:-default}` expansion.

When using Compose, keep `.env` private (`chmod 600 .env`) but make the bind-mounted
`config.yaml` readable by the container's unprivileged UID 10001 (`chmod 644
config.yaml`). The configuration contains environment placeholders rather than the
provider secrets; Compose injects those secrets from `.env`. The Compose mount uses
the `:Z` SELinux label required by enforcing AlmaLinux/RHEL-family hosts.

`server.public_url` must be an HTTPS origin routed to the HTTP listener. `GET /healthz` is liveness; `GET /readyz` reports readiness. Open `/admin` in the bot's private chat to launch its Mini App.

## Administration and access

`admin_user_ids` in YAML is the immutable administrator boundary for each bot. The server validates Telegram Mini App `initData` on every HTMX request using that bot's token, enforces a short authentication age, and checks the authenticated user against this list. It does not trust IDs supplied by the browser.

Admin links generated by `/admin` use Telegram's numeric bot ID, for example `/admin/123456789`. Internal YAML bot keys remain accepted as backward-compatible aliases, while database access is normalized back to the internal key so bot state stays isolated.

Generation model selection is input/output-specific: text→image, image→image, text→video, image→video, video→video, text→audio, video→audio, text→speech, text→3D, image→3D, text→vector image, and image→vector image. Vector results are delivered as `.html` documents with SVG isolated under a restrictive content-security policy. An explicit resolution or aspect ratio wins; otherwise visual transformations inherit reference dimensions and ratio, and requests without either leave resolution selection to the provider.

The answer allowlist is separate and editable from the panel or with `/allow`, `/deny`, and `/allowed`. `allowed_user_ids` only seeds missing entries, so later admin changes survive restarts. `allowed_chat_ids` grants access in selected chats. `allow_everyone` should be enabled only for intentionally public bots.

The model chooser is per capability: intent processing, its fallback, general chat, output/error processing, advanced model upgrade, image/video understanding, image/speech/music/video generation, and transcription each have an independent per-bot selection. Intent models must accept text and images, produce text, and advertise structured outputs; the planner receives the request text and attached/replied images while deliberately omitting expensive audio/video bytes. The toggleable model-upgrade skill routes difficult or explicitly upgraded chat requests and can also upgrade image/video understanding when the advanced model advertises the required input modality. OpenRouter's user-scoped and authenticated public all-modality catalogs are merged with its dedicated image/video catalogs, so valid public models are not lost merely because `/models/user` omitted them. OpenRouter, AI Hub, and the live fal.ai catalog have separate tabs; equal IDs cannot be confused across providers. Models requiring additional identity attestation remain omitted. The fuzzy picker searches compatible models by name, ID, description, and modality, ranks equal matches newest-first, and renders only the best 60 results.

OpenRouter's all-modality catalog is enriched from its dedicated image and video model endpoints. Token prices are displayed per million tokens, while request, image, video-second, and other SKU prices are shown per published unit. OpenRouter overloads the top-level `prompt` field for some speech/transcription models with minute- or hour-based rates, so media-capability cards preserve the numeric rate and label it as a provider-published billing unit instead of incorrectly multiplying it into a token price. Zero prompt/completion fields are labeled “Not billed on this field”; applicable non-token/SKU rates remain visible. AI Hub currently publishes only IDs, display names, types, and creation timestamps, so the panel explicitly reports unavailable pricing/context metadata rather than inventing it.

With `FAL_KEY` configured, Teleforge follows every page of `fal.catalog_url/models`, maps active model categories to the exact capability selectors, and best-effort enriches entries from the account-aware pricing endpoint. Saving a fal.ai model fetches its `openapi-3.0` expansion and verifies that all required inputs can be supplied before changing redb state. Derived schemas are cached in memory. `fal.endpoints` is an override layer: a matching entry replaces discovery and is useful when a private endpoint has no public schema or requires an administrator-chosen default.

Each selection also has OpenRouter routing controls: Auto uses normal OpenRouter routing (including Auto Exacto behavior for tool requests), Cheapest sorts by price, Highest throughput and Lowest latency use their corresponding provider sorts, Exacto selects the tool-quality model variant where applicable, and a provider can be pinned with `provider.only`. The provider chooser comes from OpenRouter's endpoint list for the selected model and shows provider name, tag, context, and recent uptime; it does not offer unrelated global providers. Catalog data is refreshed at most every ten minutes and a stale cached copy remains usable during transient catalog failures. Runtime selections are stored in redb and survive restarts even when a model is not present in the YAML override list.

AI Hub selections are sent directly to `aihub.base_url`; fal.ai selections use `fal.base_url` and their discovered or overridden endpoint schema. OpenRouter routing controls and server tools do not apply to either. fal.ai is offered for categorized generation, transcription, 3D/vector, and media-understanding endpoints, not general chat, intent, or output processing. Local function tools continue to work when their independently selected provider differs from the chat provider. OpenRouter Web Fetch requires the chat capability itself to use OpenRouter.

The YAML `defaults` and configured chat-model `options` expose the current OpenRouter request surface, including server `tools`, `tool_choice`, parallel tool calls, cache control, image configuration, reasoning, response formats, provider policy, plugins, routing/service tier, tracing, sampling, and an `extra` map for newly introduced fields. Model entries in YAML are defaults and optional per-model overrides; they do not restrict the live chooser. A dynamically selected model inherits `defaults`. Runtime routing is merged last so the administrator's selection is authoritative.

Leave `openrouter.defaults.provider.require_parameters` set to `false` when using the live model chooser. If it is `true`, OpenRouter rejects every endpoint that does not advertise every optional request field (for example `parallel_tool_calls`), even when the selected model itself supports tool calls. The shipped configuration permits provider retention and collection with `data_collection: "allow"` and `zdr: false`; deployments with stricter privacy requirements can override those independently.

| Command | Purpose |
| --- | --- |
| `/help`, `/start` | Show usage |
| `/new` | Clear the current isolated conversation |
| `/model`, `/searchprovider` | Show current selections |
| `-model [openrouter:\|aihub:\|fal:]<model-id> <request>` | Use that model for one request (administrators only) |
| `-smart <request>` | Force the configured advanced model (`-upgrade` is an alias) |
| `/search <query>` | Force live web search |
| `/image <prompt>` | Generate and upload an image |
| `/speech <text>` | Generate and upload spoken audio (`/audio` is an alias) |
| `/music <prompt>` | Generate and upload music or other non-speech audio |
| `/transcribe` | Transcribe attached or replied-to voice/audio |
| `/video <prompt>` | Generate and send a video |
| `/admin` | Open the Mini App (immutable administrators only) |
| `/allow <id>`, `/deny <id>`, `/remove <id>`, `/allowed` | Manage this bot's answer allowlist (`/removeallow` and `/unallow` alias `/remove`) |

Commands can start with either `/` or `-`, and command names are case-insensitive—for example, `-SEARCH current Istanbul transit news`. In groups, a dash command is recognized without requiring a bot mention. The one-request `-model` override accepts an optional provider prefix, applies to direct media generation when the planner selects it, never changes stored settings, and is rejected for non-admin users. Media-only override models are also identified from the provider catalog, preventing them from being sent incorrectly to a text-output chat endpoint if planning fails. Guest image, audio, video, and generated-document delivery immediately posts a pending Rich Message and then edits that same guest result with the completed artifact. Generated media captions state the exact model and prompt, and normal-chat media replies to the triggering Telegram message. Provider video URLs are downloaded, type-checked, size-bounded, and uploaded through Teleforge rather than delegated to Telegram. Generated bytes are exposed at an unguessable, extension-bearing HTTPS URL for at most ten minutes with content length and byte-range support; the cache is bounded to 50 MiB per item and 128 MiB total. Telegram-incompatible images and videos fall back to document delivery. Attachment-dependent guest operations remain subject to the media made available by Telegram in the guest update.

Application-generated HTTP request, guest-result, and ephemeral-media identifiers
use UUIDv7. Hosted documents preserve their safe filename extension and use a
matching MIME type. If Telegram cannot ingest a hosted guest-result URL, Teleforge
keeps the successful generation and edits the result into a temporary direct-download
link instead of replacing it with a failure. Cryptographic encryption and
authentication nonces remain random bytes because they are security primitives,
not application identifiers.

Conversation requests include the stored context plus current UTC time, Telegram message time, caller/display name and ID, language, bot identity, chat/title/thread scope, access mode, and effective capabilities.

One application-wide HTTP client and connection pool is shared by every bot,
Telegram transport, provider, search adapter, catalog request, and media download.
Bounded planner/progress collections use inline small-vector storage, while short
roles, status labels, and progress model identifiers use compact inline strings; large prompts,
responses, URLs, and binary payloads retain their natural heap-backed representations.

Natural-language routing uses `openrouter.planner`, which defaults to the zero-cost
`openrouter/free` router. It requests a strict JSON Schema and forces routing only to
endpoints that advertise structured outputs. The classifier contains one bounded action,
an allowlisted skill list, and an optional ordered workflow. Explicit compound requests
can therefore feed one result into the next operation—for example
`compose_text -> music_generation` writes complete lyrics before passing them to the
music tool. A compound plan always runs through the tool-capable assistant loop instead
of collapsing into direct generation, and the live Rich Message shows its ordered
stages. For ordinary single-stage media requests, a separate structured extraction pass
selects only verbatim prompt excerpts from the current request, replied message, or
Telegram quote; chat-model generation tool arguments are not trusted as replacement
prompts. Only a planner-authorized workflow may pass a non-empty, size-bounded
intermediate result as the downstream tool argument.
The backend intersects that list with admin-enabled capabilities. The same small-model prompt selects inline or
file delivery and a safe filename; the backend enforces that choice when the main
model does not call `send_file`. A structured refusal is returned directly in
the user's language. Provider-native `message.refusal` responses are also handled as
valid assistant responses instead of being misreported as empty output. Planner
responses may use string, content-part array, parsed-object, or fenced JSON forms.
Response healing is enabled, and reasoning is disabled so a small completion budget
cannot be consumed before the JSON is emitted. If the free router fails, the request
is retried with the configured very-cheap `planner.fallback_model`. Timeouts, rate
limits, unsupported endpoints, and malformed JSON are logged and fall back to the
ordinary assistant/tool loop, so the planner cannot take the bot offline. In guest
mode that loop can still call enabled generation tools and replace its pending Rich
Message with the first generated image, audio file, or video.

Telegram Rich Markdown is close to GFM but is not identical to arbitrary Markdown
emitted by models. Before delivery, Teleforge converts Setext headings, tilde and
indented code blocks, and common MathJax delimiters into Telegram's documented forms.
Already-compatible headings, tables, task lists, footnotes, links, remote media,
quotes, and fenced code remain intact. Long output is split below Telegram's 32,768
character and 500-block limits, with open code fences closed and reopened so every
chunk is independently parseable.

The admin model panel also selects independent text-output and error-explanation
models per bot. Normal assistant prose passes through the text-output model for
localized, readable Markdown without changing facts, code, links, or citations.
Errors are redacted locally before the error model sees them; that model returns at
most three short sentences and one small diagnostic excerpt. If it is unavailable,
Teleforge deterministically extracts a short provider message instead of exposing a
raw JSON response or internal routing metadata.

Photos, videos, video notes, voice notes, audio files, and matching Telegram documents can be supplied directly or by replying to the media message. Private Telegram media is size-checked and encoded only for the current provider request. Images and videos can guide image/video generation where the selected provider endpoint supports references. YouTube URLs are sent as video inputs only while the independent YouTube skill toggle is enabled. Prompt expansion is also independently toggleable and is authorized only when the user explicitly asks for it and the intent processor selects that skill; media-generation prompts are then expanded before the generation call.

## Local persistence and secrets

redb writes to `database.path`; there is no remote database mode. Only one Teleforge process should open a database file. Back up the file before deployments or schema-affecting upgrades.

API keys entered through the panel are write-only in the UI and encrypted at rest with ChaCha20-Poly1305 using `database.encryption_key`. Bootstrap keys from the environment are copied into each bot's isolated encrypted state only when that provider is not yet configured. Protect the master key separately from the database backup: losing it makes credentials unrecoverable, while disclosure of both defeats at-rest protection.

Provider environment variables are optional. If one is absent and no encrypted per-bot key has been saved, its provider options and dependent capability controls are disabled; the panel explicitly names the missing key. Saving a key activates those controls on the next HTMX refresh. Removing it makes them unavailable again without silently falling back to a different credential.

## Deployment and security

### Native systemd deployment (recommended)

Build on the server, or copy a Linux binary built for the server's architecture
and glibc version. Build-time prompts, skills, and Mini App assets are embedded;
configuration, secrets, and redb remain external.

```bash
cargo build --locked --release
sudo install -o root -g root -m 0755 target/release/teleforge /usr/local/bin/teleforge
sudo useradd --system --home-dir /var/lib/teleforge --shell /usr/sbin/nologin teleforge
sudo install -o root -g root -m 0644 deploy/teleforge.tmpfiles /etc/tmpfiles.d/teleforge.conf
sudo systemd-tmpfiles --create /etc/tmpfiles.d/teleforge.conf
sudo install -o root -g teleforge -m 0640 config.example.yaml /etc/teleforge/config.yaml
sudo install -o root -g teleforge -m 0600 deploy/teleforge.env.example /etc/teleforge/teleforge.env
sudo install -o root -g root -m 0644 deploy/teleforge.service /etc/systemd/system/teleforge.service
sudoedit /etc/teleforge/teleforge.env
sudoedit /etc/teleforge/config.yaml
sudo systemctl daemon-reload
sudo systemctl enable --now teleforge
```

Before starting, replace every placeholder and generate
`TELEFORGE_MASTER_KEY` with `openssl rand -base64 32`. Monitor with
`journalctl -u teleforge -f` and probe `http://127.0.0.1:8080/readyz`.

For upgrades, atomically install a compatible prebuilt binary and restart:

```bash
sudo install -o root -g root -m 0755 teleforge-linux-amd64 /usr/local/bin/teleforge.new
sudo mv /usr/local/bin/teleforge.new /usr/local/bin/teleforge
sudo systemctl restart teleforge
curl http://127.0.0.1:8080/readyz
```

Keep Caddy proxying HTTPS to `127.0.0.1:8080`; never expose the redb file,
environment file, or backend port publicly.

### Docker deployment

```sh
docker compose up --build -d
```

To update an existing Compose deployment, run this from the checkout that contains
`compose.yaml`, `config.yaml`, and `.env`. `docker compose down` does not remove the
named `teleforge-data` volume, but stopping first is required for a consistent redb backup:

```sh
cd /opt/teleforge/tg-ai-assistant
umask 077
OLD_REV="$(git rev-parse HEAD)"
mkdir -p backups
docker compose down

DATA_VOLUME="$(docker volume ls -q --filter label=com.docker.compose.volume=teleforge-data | head -n1)"
test -n "$DATA_VOLUME"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
cp -p config.yaml "backups/config-$STAMP.yaml"
cp -p .env "backups/env-$STAMP"
docker run --rm -v "$DATA_VOLUME":/data:ro -v "$PWD/backups":/backup alpine:3.20 \
  tar -czf "/backup/redb-$STAMP.tar.gz" -C /data .

git fetch --prune
git pull --ff-only
docker compose up -d --build --remove-orphans

docker compose ps
curl --fail http://127.0.0.1:8080/healthz
curl --fail http://127.0.0.1:8080/readyz
docker compose logs --tail=100 teleforge
```

Keep `config.yaml`, `.env`, and the redb volume; do not replace them with the example
files during an update. The new Docker build includes `src/`, `defaults/`, and the
admin Web App assets. Caddy normally needs no change or reload because the upstream
address remains `127.0.0.1:8080`. If the new version must be rolled back, stop the
stack, return to the revision printed by the earlier `OLD_REV` command, rebuild, and
start it again:

```sh
docker compose down
git switch --detach "$OLD_REV"
docker compose up -d --build --remove-orphans
```

Only restore the redb archive if a rollback explicitly requires restoring state; keep
the master encryption key together with the database backup or encrypted credentials
cannot be decrypted.

The container runs as an unprivileged user and persists `/app/data`. Its Docker healthcheck is built into the binary, so the runtime image does not need `curl`; only CA certificates are installed for outbound HTTPS. Terminate any old instance before moving a Telegram token because long polling must have one active consumer per token. Put an HTTPS reverse proxy in front of port 8080 and do not expose the redb file.

- Never commit real bot tokens, provider keys, the master key, or `config.yaml`.
- Keep immutable admin IDs restrictive. Every admin mutation is re-authenticated server-side; credentials are never rendered back to the browser.
- The Mini App response uses a nonce-based Content Security Policy, pinned HTMX integrity metadata, no-store caching, and a short Telegram authentication TTL.
- Search output is treated as untrusted content by the embedded prompt. User-facing provider failures are generic while details stay in service logs.
- Generated media can spend meaningful credits. Review models, limits, provider routing/data-retention options, and access lists before enabling a public bot. A model appearing in a provider catalog may still be denied by that account's group or entitlement policy; those failures remain in service logs.
- This backend uses long polling. If adding webhooks, retain per-token authenticated routes and bot-scoped persistence.

## Development

```sh
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo build --release --locked
```

All Rust modules carry module-level documentation. Tests cover catalog parsing and capability filtering, environment expansion, Rich Message chunking, option/tool composition, local database isolation/encryption, and Mini App authentication.
