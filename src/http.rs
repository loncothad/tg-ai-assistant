//! Application-wide HTTP transport shared by Telegram and every provider.
//!
//! `reqwest::Client` clones already share their connector pool internally. This
//! wrapper makes that ownership model explicit across Teleforge and keeps the
//! sole client-construction site in one module.

use std::{ops::Deref, sync::Arc, time::Duration};

use eyre::Context;

use crate::Result;

/// Cloneable handle to the process-wide HTTP connection pool.
#[derive(Clone, Debug)]
pub struct HttpClient(Arc<reqwest::Client>);

impl HttpClient {
    /// Builds the single transport used by one Teleforge process.
    pub fn build(timeout: Duration) -> Result<Self> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(timeout)
            .user_agent(format!("teleforge/{}", env!("CARGO_PKG_VERSION")))
            .pool_max_idle_per_host(16)
            .build()
            .context("Failed to build shared HTTP client")?;
        Ok(Self(Arc::new(client)))
    }

    /// Returns a lightweight handle for APIs that require an owned
    /// `reqwest::Client`. It points at the same connector and connection pool.
    pub fn reqwest_handle(&self) -> reqwest::Client {
        self.0.as_ref().clone()
    }
}

impl Deref for HttpClient {
    type Target = reqwest::Client;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloned_handles_retain_the_same_client_instance() {
        let client = HttpClient::build(Duration::from_secs(1)).unwrap();
        let cloned = client.clone();
        assert!(Arc::ptr_eq(&client.0, &cloned.0));
    }
}
