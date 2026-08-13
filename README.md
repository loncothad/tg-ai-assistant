# Teleforge

Teleforge is a production-oriented, multi-bot Telegram AI assistant written in Rust with [`frankenstein`](https://crates.io/crates/frankenstein). It sends Telegram Rich Messages, uses OpenRouter for chat and media, supports Brave Search, Exa, and Google results through SerpAPI, and stores all runtime state in a local redb database.

## Capabilities

- Any number of Telegram bot tokens in one process. Each configured bot ID has isolated settings, encrypted API credentials, history, access rules, audit records, and update offsets.
- Private-message, group/thread, and Telegram guest-bot modes. Group handling can require a mention or reply.
- OpenRouter chat options including fallback model lists, provider routing, server tools, plugins, transforms, reasoning, structured responses, modalities, sampling controls, usage, and forward-compatible fields under `extra`.
- OpenRouter server-side Web Search and Web Fetch. Fetch is a separately toggleable built-in skill that can retrieve page and PDF text, with engine, usage/content limits, and domain policy configured in YAML.
- Real model tools named `web_search`, `generate_image`, `generate_audio`, `generate_video`, and attachment-scoped `transcribe_audio`. The AI can invoke them during an ordinary conversation; the backend executes the selected API and delivers generated media to Telegram.
- Live Rich Message drafts show the current request phase and elapsed time while ordinary assistant requests run. Media and file generation also uploads a small, local in-progress stub document immediately and removes it when the completed file is ready. Explicit image/video requests are recognized server-side and dispatched directly so a model cannot incorrectly claim that an enabled generator is unavailable.
- A toggleable `send_file` tool lets the model deliver source code, configuration, structured text, or an unwieldy answer as a named Telegram document. Text answers above 12,000 characters automatically fall back to `answer.md`.
- Direct `/search`, `/image`, `/audio`, `/transcribe`, and `/video` commands. Every command also accepts the `-COMMAND` form (case-insensitive), including query and guest requests.
- An authenticated Telegram Mini App admin panel built with HTMX for model/provider selection, per-skill switches, API-key management, custom prompt/skill import, skill-bundle export/import, and user allowlisting. Its responsive blue interface uses a local fuzzy-search picker instead of rendering enormous model dropdowns.
- Native Telegram Rich Message responses and rich guest-query results with Unicode-safe chunking.
- Bounded per-bot concurrency, timeouts, polling backoff, graceful shutdown, structured logs, and health endpoints.

## Build-time defaults and runtime customization

The normal repository files [`defaults/system.md`](defaults/system.md) and [`defaults/skills/`](defaults/skills/) are embedded into the binary with `include_str!`. Each built-in skill has its own file and a Rust-side description. Edit those files and rebuild to change the shipped defaults.

Built-in skills are enabled by default. Disabling one in the admin panel removes its instructions and, where applicable, its callable tool schema from subsequent AI requests. Media understanding and transcription have independent switches from media generation. Runtime custom prompts and skill instructions live only in redb, can be enabled/disabled/reset independently, and never alter the embedded defaults. Imported custom skills may direct the model to the enabled built-in tool names; importing arbitrary executable code or arbitrary HTTP endpoints is deliberately unsupported.

Skill export produces a versioned JSON bundle containing the built-in descriptions/instructions, enabled states, and custom skill text. Uploading that JSON in the skill import form restores the toggles and custom text. Built-in instruction text remains build-time immutable; edit the files and rebuild to replace it.

## Quick start

Requirements: Rust 1.85+, one or more BotFather tokens, an HTTPS public origin for the Telegram Mini App, and an OpenRouter key. Search keys are optional.

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

`server.public_url` must be an HTTPS origin routed to the HTTP listener. `GET /healthz` is liveness; `GET /readyz` reports readiness. Open `/admin` in the bot's private chat to launch its Mini App.

## Administration and access

`admin_user_ids` in YAML is the immutable administrator boundary for each bot. The server validates Telegram Mini App `initData` on every HTMX request using that bot's token, enforces a short authentication age, and checks the authenticated user against this list. It does not trust IDs supplied by the browser.

Admin links generated by `/admin` use Telegram's numeric bot ID, for example `/admin/123456789`. Internal YAML bot keys remain accepted as backward-compatible aliases, while database access is normalized back to the internal key so bot state stays isolated.

The answer allowlist is separate and editable from the panel or with `/allow`, `/deny`, and `/allowed`. `allowed_user_ids` only seeds missing entries, so later admin changes survive restarts. `allowed_chat_ids` grants access in selected chats. `allow_everyone` should be enabled only for intentionally public bots.

The model chooser is per capability: general chat, image understanding, video understanding, image generation, speech generation, transcription, and video generation each have an independent per-bot selection. Teleforge retrieves and caches OpenRouter's authenticated `/models/user` catalog separately for every bot, so provider preferences, guardrails, privacy settings, and account eligibility are respected. Models marked as requiring additional identity attestation are omitted. The fuzzy picker searches compatible models by name, ID, description, and modality, ranks equal matches newest-first, and renders only the best 60 results so the Web App stays responsive. It displays context/output limits, token pricing, modalities, knowledge cutoff, tokenizer, supported parameters, voices, resolutions, aspect ratios, and durations when OpenRouter publishes them.

Each selection also has OpenRouter routing controls: Auto uses normal OpenRouter routing (including Auto Exacto behavior for tool requests), Cheapest sorts by price, Highest throughput and Lowest latency use their corresponding provider sorts, Exacto selects the tool-quality model variant where applicable, and a provider can be pinned with `provider.only`. The provider chooser comes from OpenRouter's endpoint list for the selected model and shows provider name, tag, context, and recent uptime; it does not offer unrelated global providers. Catalog data is refreshed at most every ten minutes and a stale cached copy remains usable during transient catalog failures. Runtime selections are stored in redb and survive restarts even when a model is not present in the YAML override list.

The YAML `defaults` and configured chat-model `options` expose the current OpenRouter request surface, including server `tools`, `tool_choice`, parallel tool calls, cache control, image configuration, reasoning, response formats, provider policy, plugins, routing/service tier, tracing, sampling, and an `extra` map for newly introduced fields. Model entries in YAML are defaults and optional per-model overrides; they do not restrict the live chooser. A dynamically selected model inherits `defaults`. Runtime routing is merged last so the administrator's selection is authoritative.

Leave `openrouter.defaults.provider.require_parameters` set to `false` when using the live model chooser. If it is `true`, OpenRouter rejects every endpoint that does not advertise every optional request field (for example `parallel_tool_calls`), even when the selected model itself supports tool calls. The shipped configuration permits provider retention and collection with `data_collection: "allow"` and `zdr: false`; deployments with stricter privacy requirements can override those independently.

| Command | Purpose |
| --- | --- |
| `/help`, `/start` | Show usage |
| `/new` | Clear the current isolated conversation |
| `/model`, `/searchprovider` | Show current selections |
| `/search <query>` | Force live web search |
| `/image <prompt>` | Generate and upload an image |
| `/audio <text>` | Generate and upload spoken audio |
| `/transcribe` | Transcribe attached or replied-to voice/audio |
| `/video <prompt>` | Generate and send a video |
| `/admin` | Open the Mini App (immutable administrators only) |
| `/allow <id>`, `/deny <id>`, `/allowed` | Manage this bot's answer allowlist |

Commands can start with either `/` or `-`, and command names are case-insensitive—for example, `-SEARCH current Istanbul transit news`. In groups, a dash command is recognized without requiring a bot mention. Guest replies remain a single Rich Message result, so text commands and OpenRouter Web Search/Web Fetch work there, while media-producing and attachment commands remain unavailable.

Conversation requests include the stored context plus current UTC time, Telegram message time, caller/display name and ID, language, bot identity, chat/title/thread scope, access mode, and effective capabilities.

Photos, videos, video notes, voice notes, audio files, and matching Telegram documents can be supplied directly or by replying to the media message. Private Telegram media is size-checked and encoded only for the current OpenRouter request. Images and videos can guide image/video generation. YouTube URLs are sent as video inputs, subject to the selected model/provider's video support.

## Local persistence and secrets

redb writes to `database.path`; there is no remote database mode. Only one Teleforge process should open a database file. Back up the file before deployments or schema-affecting upgrades.

API keys entered through the panel are write-only in the UI and encrypted at rest with ChaCha20-Poly1305 using `database.encryption_key`. Bootstrap keys from the environment are copied into each bot's isolated encrypted state only when that provider is not yet configured. Protect the master key separately from the database backup: losing it makes credentials unrecoverable, while disclosure of both defeats at-rest protection.

Provider environment variables are optional. If one is absent and no encrypted per-bot key has been saved, its provider options and dependent capability controls are disabled; the panel explicitly names the missing key. Saving a key activates those controls on the next HTMX refresh. Removing it makes them unavailable again without silently falling back to a different credential.

## Deployment and security

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

The container runs as an unprivileged user and persists `/app/data`. Terminate any old instance before moving a Telegram token because long polling must have one active consumer per token. Put an HTTPS reverse proxy in front of port 8080 and do not expose the redb file.

- Never commit real bot tokens, provider keys, the master key, or `config.yaml`.
- Keep immutable admin IDs restrictive. Every admin mutation is re-authenticated server-side; credentials are never rendered back to the browser.
- The Mini App response uses a nonce-based Content Security Policy, pinned HTMX integrity metadata, no-store caching, and a short Telegram authentication TTL.
- Search output is treated as untrusted content by the embedded prompt. User-facing provider failures are generic while details stay in service logs.
- Generated media can spend meaningful credits. Review models, limits, OpenRouter routing/data-retention options, and access lists before enabling a public bot.
- This backend uses long polling. If adding webhooks, retain per-token authenticated routes and bot-scoped persistence.

## Development

```sh
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo build --release --locked
```

All Rust modules carry module-level documentation. Tests cover catalog parsing and capability filtering, environment expansion, Rich Message chunking, option/tool composition, local database isolation/encryption, and Mini App authentication.
