//! Teleforge process entry point, health server, Mini App, and graceful shutdown.

use std::{net::SocketAddr, path::PathBuf, sync::Arc, time::Duration};

use axum::{
    Json, Router,
    http::{HeaderValue, Request},
    routing::get,
};
use clap::Parser;
use eyre::{Context, bail};
use serde_json::json;
use teleforge::{
    Result,
    admin::{self, AdminState},
    config::Config,
    db::Store,
    ephemeral_media,
    http::HttpClient,
    telegram::BotRunner,
};
use tokio::{
    sync::watch,
    task::{JoinHandle, JoinSet},
};
use tower_http::{
    catch_panic::CatchPanicLayer,
    request_id::{MakeRequestId, PropagateRequestIdLayer, RequestId, SetRequestIdLayer},
    trace::TraceLayer,
};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    #[arg(short, long, env = "TELEFORGE_CONFIG", default_value = "config.yaml")]
    config: PathBuf,
    /// Probe the local readiness endpoint and exit without starting workers.
    #[arg(long)]
    healthcheck: bool,
}

#[derive(Clone, Copy)]
struct MakeRequestUuidV7;

impl MakeRequestId for MakeRequestUuidV7 {
    fn make_request_id<B>(&mut self, _request: &Request<B>) -> Option<RequestId> {
        HeaderValue::from_str(&Uuid::now_v7().to_string())
            .ok()
            .map(RequestId::new)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if args.healthcheck {
        let client = HttpClient::build(Duration::from_secs(15))?;
        let response = client
            .get("http://127.0.0.1:8080/readyz")
            .send()
            .await
            .context("Readiness probe failed")?;
        if !response.status().is_success() {
            bail!("Readiness probe returned {}", response.status());
        }
        return Ok(());
    }
    let config = Arc::new(Config::load(&args.config)?);
    init_tracing(config.server.json_logs);
    let store = Store::connect(&config.database).await?;
    let client = HttpClient::build(config.timeout())?;

    let enabled = config
        .bots
        .iter()
        .filter(|bot| bot.enabled)
        .cloned()
        .collect::<Vec<_>>();
    if enabled.is_empty() {
        bail!("No enabled bots are configured");
    }
    let mut runners = Vec::with_capacity(enabled.len());
    for bot in enabled {
        runners.push(BotRunner::new(bot, config.clone(), store.clone(), client.clone()).await?);
    }

    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let server = spawn_health_server(
        &config.server.listen,
        runners.len(),
        AdminState::new(config.clone(), store.clone(), client.clone()),
        shutdown_rx.clone(),
    )
    .await?;
    let mut bots = JoinSet::new();
    for runner in runners {
        bots.spawn(runner.run(shutdown_rx.clone()));
    }
    info!(bot_count = bots.len(), listen = %config.server.listen, "Teleforge started");

    let reason = tokio::select! {
        signal = shutdown_signal() => { signal?; "shutdown signal".to_owned() },
        result = bots.join_next() => match result {
            Some(Ok(Ok(()))) => "bot worker stopped".to_owned(),
            Some(Ok(Err(error))) => format!("bot worker failed: {error:#}"),
            Some(Err(error)) => format!("bot worker panicked: {error}"),
            None => "all bot workers stopped".to_owned(),
        },
    };
    info!(%reason, "shutting down");
    let _ = shutdown_tx.send(true);
    let _ = tokio::time::timeout(Duration::from_secs(35), async {
        while bots.join_next().await.is_some() {}
    })
    .await;
    let _ = tokio::time::timeout(Duration::from_secs(5), server).await;
    Ok(())
}

async fn spawn_health_server(
    listen: &str,
    bot_count: usize,
    admin_state: AdminState,
    mut shutdown: watch::Receiver<bool>,
) -> Result<JoinHandle<()>> {
    let address: SocketAddr = listen
        .parse()
        .with_context(|| format!("Invalid server.listen address: {listen}"))?;
    let app = Router::new()
        .route("/healthz", get(|| async { Json(json!({ "status": "ok" })) }))
        .route("/readyz", get(move || async move { Json(json!({ "status": "ready", "bots": bot_count, "version": env!("CARGO_PKG_VERSION") })) }))
        .route("/generated/{token}", get(ephemeral_media::serve))
        .merge(admin::router(admin_state))
        .layer(PropagateRequestIdLayer::x_request_id())
        .layer(SetRequestIdLayer::x_request_id(MakeRequestUuidV7))
        .layer(TraceLayer::new_for_http())
        .layer(CatchPanicLayer::new());
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .with_context(|| format!("Failed to bind health server to {address}"))?;
    Ok(tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                while !*shutdown.borrow() {
                    if shutdown.changed().await.is_err() {
                        break;
                    }
                }
            })
            .await
        {
            error!(%error, "health server failed");
        }
    }))
}

fn init_tracing(json_logs: bool) {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("teleforge=info,tower_http=info"));
    if json_logs {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .json()
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .compact()
            .init();
    }
}

async fn shutdown_signal() -> Result<()> {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        tokio::select! { result = tokio::signal::ctrl_c() => result?, _ = terminate.recv() => {} }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await?;
    Ok(())
}
