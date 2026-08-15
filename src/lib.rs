//! Teleforge's reusable configuration, persistence, AI-provider, search, and Telegram runtime.

pub mod admin;
pub mod catalog;
pub mod config;
pub mod db;
pub mod defaults;
pub mod ephemeral_media;
pub mod fal;
pub mod http;
pub mod openrouter;
pub mod rich;
pub mod search;
pub mod telegram;

pub type Result<T> = eyre::Result<T>;
