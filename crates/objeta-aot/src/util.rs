use serde::Serialize;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::types::ResolvedModelFiles;

pub fn write_json(path: PathBuf, value: &impl Serialize) -> Result<(), Box<dyn Error>> {
    let json = serde_json::to_string_pretty(value)?;
    fs::write(path, json)?;
    Ok(())
}

pub fn now_rfc3339ish() -> String {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(dur) => format!("unix:{}", dur.as_secs()),
        Err(_) => "unix:0".to_string(),
    }
}

pub fn extract_number_after(s: &str, needle: &str) -> Option<u32> {
    let start = s.find(needle)? + needle.len();
    let digits: String = s[start..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}

pub fn resolve_model_files(model: &Path) -> Result<ResolvedModelFiles, Box<dyn Error>> {
    if model.is_file() {
        let file_name = model.file_name().and_then(|s| s.to_str()).unwrap_or_default();
        if file_name.ends_with(".index.json") {
            let dir = model
                .parent()
                .ok_or("index json path has no parent directory")?;
            let config_path = dir.join("config.json");
            if !config_path.exists() {
                return Err(format!("missing config.json next to {}", model.display()).into());
            }
            return Ok(ResolvedModelFiles {
                index_path: model.to_path_buf(),
                config_path,
            });
        }
        return Err(format!(
            "expected a model directory or model.safetensors.index.json, got {}",
            model.display()
        )
        .into());
    }

    let index_path = model.join("model.safetensors.index.json");
    let config_path = model.join("config.json");
    if !index_path.exists() {
        return Err(format!("missing {}", index_path.display()).into());
    }
    if !config_path.exists() {
        return Err(format!("missing {}", config_path.display()).into());
    }
    Ok(ResolvedModelFiles {
        index_path,
        config_path,
    })
}

pub fn infer_model_name(model: &Path, config: &crate::types::ModelConfig) -> String {
    config
        .model_type
        .clone()
        .or_else(|| model.file_name().and_then(|s| s.to_str()).map(|s| s.to_string()))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "qwen36".to_string())
}

/// Parse human-readable byte strings like "3GB", "512MB", "8192" (raw bytes).
pub fn parse_bytes_human(s: &str) -> Option<u64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let upper = s.to_ascii_uppercase();
    if let Some(num) = upper.strip_suffix("GB") {
        let n: f64 = num.trim().parse().ok()?;
        Some((n * 1024.0 * 1024.0 * 1024.0) as u64)
    } else if let Some(num) = upper.strip_suffix("MB") {
        let n: f64 = num.trim().parse().ok()?;
        Some((n * 1024.0 * 1024.0) as u64)
    } else if let Some(num) = upper.strip_suffix("KB") {
        let n: f64 = num.trim().parse().ok()?;
        Some((n * 1024.0) as u64)
    } else {
        s.parse::<u64>().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_human_bytes_gb() {
        assert_eq!(parse_bytes_human("3GB"), Some(3 * 1024 * 1024 * 1024));
    }

    #[test]
    fn parse_human_bytes_mb() {
        assert_eq!(parse_bytes_human("512MB"), Some(512 * 1024 * 1024));
    }

    #[test]
    fn parse_human_bytes_raw() {
        assert_eq!(parse_bytes_human("1024"), Some(1024));
    }

    #[test]
    fn parse_human_bytes_empty() {
        assert_eq!(parse_bytes_human(""), None);
    }
}
