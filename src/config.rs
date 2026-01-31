//! Configuration management for rec

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HistoryEntry {
    pub timestamp: String,
    pub original: String,
    pub corrected: String,
    pub model: String,
    pub custom_words: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    pub custom_words: Vec<String>,
    pub claude_model: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            custom_words: vec![],
            claude_model: "claude-haiku-4-5".to_string(),
        }
    }
}

impl Config {
    /// Get the config file path
    fn config_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
        let config_dir = dirs::config_dir()
            .ok_or("Could not find config directory")?
            .join("rec");

        fs::create_dir_all(&config_dir)?;
        Ok(config_dir.join("config.json"))
    }

    /// Load config from disk, creating with defaults if it doesn't exist
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let path = Self::config_path()?;

        if !path.exists() {
            let config = Self::default();
            config.save()?;
            return Ok(config);
        }

        let content = fs::read_to_string(&path)?;

        match serde_json::from_str(&content) {
            Ok(config) => Ok(config),
            Err(e) => {
                // Config is corrupted - make a backup and recreate
                let backup_path = path.with_extension("json.bak");
                fs::copy(&path, &backup_path)?;

                eprintln!("⚠️  Config file was corrupted and has been reset to defaults");
                eprintln!("   Backup saved to: {}", backup_path.display());
                eprintln!("   Error was: {}\n", e);

                let config = Self::default();
                config.save()?;
                Ok(config)
            }
        }
    }

    /// Save config to disk
    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let path = Self::config_path()?;
        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Add a custom word to the list (deduplicated)
    pub fn add_custom_word(&mut self, word: String) {
        if !self.custom_words.contains(&word) {
            self.custom_words.push(word);
        }
    }

    /// Get the history file path
    fn history_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
        let config_dir = dirs::config_dir()
            .ok_or("Could not find config directory")?
            .join("rec");

        fs::create_dir_all(&config_dir)?;
        Ok(config_dir.join("history.json"))
    }

    /// Load history from disk
    pub fn load_history() -> Result<Vec<HistoryEntry>, Box<dyn std::error::Error>> {
        let path = Self::history_path()?;

        if !path.exists() {
            return Ok(vec![]);
        }

        let content = fs::read_to_string(&path)?;

        match serde_json::from_str(&content) {
            Ok(history) => Ok(history),
            Err(e) => {
                // History is corrupted - make a backup and start fresh
                let backup_path = path.with_extension("json.bak");
                fs::copy(&path, &backup_path)?;

                eprintln!("⚠️  History file was corrupted and has been reset");
                eprintln!("   Backup saved to: {}", backup_path.display());
                eprintln!("   Error was: {}\n", e);

                Ok(vec![])
            }
        }
    }

    /// Add entry to history
    pub fn add_to_history(
        original: &str,
        corrected: &str,
        model: &str,
        custom_words: &[String],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut history = Self::load_history()?;

        let timestamp = chrono::Utc::now().to_rfc3339();

        history.push(HistoryEntry {
            timestamp,
            original: original.to_string(),
            corrected: corrected.to_string(),
            model: model.to_string(),
            custom_words: custom_words.to_vec(),
        });

        let path = Self::history_path()?;
        let content = serde_json::to_string_pretty(&history)?;
        fs::write(path, content)?;

        Ok(())
    }
}
