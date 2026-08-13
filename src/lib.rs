//! Teleforge's reusable configuration, persistence, AI-provider, search, and Telegram runtime.

pub mod admin;
pub mod config;
pub mod db;
pub mod defaults;
pub mod openrouter;
pub mod rich;
pub mod search;
pub mod telegram;

pub type Result<T> = eyre::Result<T>;
