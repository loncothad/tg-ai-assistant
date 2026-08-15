//! Live web-search adapters for Brave, Exa, and Google through SerpAPI.

use crate::{
    Result,
    config::{ProviderConfig, SearchConfig, SearchProvider},
    http::HttpClient,
};
use eyre::{Context, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

#[derive(Clone)]
pub struct SearchService {
    client: HttpClient,
    config: SearchConfig,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

impl SearchService {
    pub fn new(client: HttpClient, config: SearchConfig) -> Self {
        Self { client, config }
    }
    pub fn default_provider(&self) -> SearchProvider {
        self.config.default_provider
    }
    pub async fn search(
        &self,
        provider: SearchProvider,
        query: &str,
        api_key: &str,
    ) -> Result<Vec<SearchResult>> {
        if query.trim().is_empty() {
            bail!("Search query cannot be empty");
        }
        if api_key.is_empty() {
            bail!("The {} API key is not configured", provider.as_str());
        }
        match provider {
            SearchProvider::Openrouter => bail!("OpenRouter search is executed by its server tool"),
            SearchProvider::Brave => self.brave(query, api_key).await,
            SearchProvider::Exa => self.exa(query, api_key).await,
            SearchProvider::Serpapi => self.serpapi(query, api_key).await,
        }
    }
    pub async fn tool_output(
        &self,
        provider: SearchProvider,
        query: &str,
        api_key: &str,
    ) -> String {
        match self.search(provider, query, api_key).await {
            Ok(results) => {
                json!({"query": query, "provider": provider.as_str(), "results": results})
                    .to_string()
            }
            Err(error) => {
                json!({"query": query, "provider": provider.as_str(), "error": error.to_string()})
                    .to_string()
            }
        }
    }
    fn provider(&self, value: SearchProvider) -> &ProviderConfig {
        match value {
            SearchProvider::Openrouter => &self.config.brave,
            SearchProvider::Brave => &self.config.brave,
            SearchProvider::Exa => &self.config.exa,
            SearchProvider::Serpapi => &self.config.serpapi,
        }
    }
    async fn brave(&self, query: &str, key: &str) -> Result<Vec<SearchResult>> {
        let cfg = self.provider(SearchProvider::Brave);
        let mut params = vec![
            ("q".into(), query.into()),
            ("count".into(), self.config.max_results.min(20).to_string()),
        ];
        append_options(&mut params, &cfg.options);
        let value = checked(
            self.client
                .get(&cfg.base_url)
                .header("X-Subscription-Token", key)
                .query(&params)
                .send()
                .await?,
        )
        .await?;
        Ok(parse_results(
            value.pointer("/web/results"),
            "url",
            "description",
            self.config.max_results,
        ))
    }
    async fn exa(&self, query: &str, key: &str) -> Result<Vec<SearchResult>> {
        let cfg = self.provider(SearchProvider::Exa);
        let mut body = Map::from_iter([
            ("query".into(), json!(query)),
            ("numResults".into(), json!(self.config.max_results)),
            ("contents".into(), json!({"text":{"maxCharacters":2000}})),
        ]);
        body.extend(cfg.options.clone());
        let value = checked(
            self.client
                .post(&cfg.base_url)
                .header("x-api-key", key)
                .json(&body)
                .send()
                .await?,
        )
        .await?;
        Ok(parse_results(
            value.get("results"),
            "url",
            "text",
            self.config.max_results,
        ))
    }
    async fn serpapi(&self, query: &str, key: &str) -> Result<Vec<SearchResult>> {
        let cfg = self.provider(SearchProvider::Serpapi);
        let mut params = vec![
            ("q".into(), query.into()),
            ("api_key".into(), key.into()),
            ("engine".into(), "google".into()),
            ("num".into(), self.config.max_results.to_string()),
        ];
        append_options(&mut params, &cfg.options);
        let value = checked(self.client.get(&cfg.base_url).query(&params).send().await?).await?;
        Ok(parse_results(
            value.get("organic_results"),
            "link",
            "snippet",
            self.config.max_results,
        ))
    }
}
fn parse_results(
    value: Option<&Value>,
    url_key: &str,
    snippet_key: &str,
    limit: usize,
) -> Vec<SearchResult> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .take(limit)
        .filter_map(|item| {
            Some(SearchResult {
                title: item
                    .get("title")
                    .and_then(Value::as_str)
                    .unwrap_or("Untitled")
                    .into(),
                url: item.get(url_key)?.as_str()?.into(),
                snippet: item
                    .get(snippet_key)
                    .or_else(|| item.get("summary"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .into(),
            })
        })
        .collect()
}
fn append_options(params: &mut Vec<(String, String)>, options: &Map<String, Value>) {
    for (key, value) in options {
        if let Some(value) = value.as_str() {
            params.push((key.clone(), value.into()));
        } else if value.is_number() || value.is_boolean() {
            params.push((key.clone(), value.to_string()));
        }
    }
}
async fn checked(response: reqwest::Response) -> Result<Value> {
    let status = response.status();
    let bytes = response
        .bytes()
        .await
        .wrap_err("Failed to read search response")?;
    if !status.is_success() {
        bail!(
            "Search API returned {status}: {}",
            String::from_utf8_lossy(&bytes[..bytes.len().min(1000)])
        );
    }
    serde_json::from_slice(&bytes).wrap_err("Search API returned invalid JSON")
}
