use serde::{Deserialize, Serialize};
use std::process::Command;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OsMemoryPressureState {
    Unknown,
    Normal,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OsTelemetrySample {
    pub rss_mb: f32,
    pub memory_pressure_state: String,
    pub pageouts_delta: u64,
    pub swapouts_delta: u64,
    pub confirmed_pageouts_delta: u64,
    pub confirmed_swapouts_delta: u64,
}

#[derive(Debug, Clone, Default)]
pub struct MacOsMemoryTelemetryCollector {
    last_pageouts: Option<u64>,
    last_swap_used_bytes: Option<u64>,
}

impl MacOsMemoryTelemetryCollector {
    pub fn sample(&mut self) -> OsTelemetrySample {
        let rss_mb = current_rss_mb().unwrap_or(0.0);
        let memory_pressure_state = current_memory_pressure_state()
            .map(|s| match s {
                OsMemoryPressureState::Unknown => "unknown",
                OsMemoryPressureState::Normal => "normal",
                OsMemoryPressureState::Warning => "warning",
                OsMemoryPressureState::Critical => "critical",
            })
            .unwrap_or("unknown")
            .to_string();

        let pageouts_now = current_pageouts().ok();
        let swap_used_now = current_swap_used_bytes().ok();
        let pageouts_delta = match (self.last_pageouts, pageouts_now) {
            (Some(prev), Some(now)) => now.saturating_sub(prev),
            _ => 0,
        };
        let swapouts_delta = match (self.last_swap_used_bytes, swap_used_now) {
            (Some(prev), Some(now)) => now.saturating_sub(prev),
            _ => 0,
        };
        let confirmed_pageouts_delta = if self.last_pageouts.is_some() {
            pageouts_delta
        } else {
            0
        };
        let confirmed_swapouts_delta = if self.last_swap_used_bytes.is_some() {
            swapouts_delta
        } else {
            0
        };
        self.last_pageouts = pageouts_now;
        self.last_swap_used_bytes = swap_used_now;

        OsTelemetrySample {
            rss_mb,
            memory_pressure_state,
            pageouts_delta,
            swapouts_delta,
            confirmed_pageouts_delta,
            confirmed_swapouts_delta,
        }
    }

    pub fn reset(&mut self) {
        self.last_pageouts = None;
        self.last_swap_used_bytes = None;
    }
}

fn current_rss_mb() -> Result<f32, ()> {
    let pid = std::process::id().to_string();
    let out = Command::new("/bin/ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .map_err(|_| ())?;
    let txt = String::from_utf8_lossy(&out.stdout);
    let kb: f32 = txt.trim().parse().map_err(|_| ())?;
    Ok(kb / 1024.0)
}

fn current_pageouts() -> Result<u64, ()> {
    let out = Command::new("/usr/bin/vm_stat")
        .output()
        .map_err(|_| ())?;
    let txt = String::from_utf8_lossy(&out.stdout);
    let page_size = parse_page_size(&txt).unwrap_or(4096);
    for line in txt.lines() {
        let lower = line.to_ascii_lowercase();
        if lower.contains("pageouts") {
            let count = parse_last_u64(line)?;
            return Ok(count.saturating_mul(page_size));
        }
    }
    Err(())
}

fn current_swap_used_bytes() -> Result<u64, ()> {
    let out = Command::new("/usr/sbin/sysctl")
        .args(["vm.swapusage"])
        .output()
        .map_err(|_| ())?;
    let txt = String::from_utf8_lossy(&out.stdout);
    let lower = txt.to_ascii_lowercase();
    let used_pos = lower.find("used =").ok_or(())?;
    let rest = &txt[used_pos + 6..];
    let token = rest.split_whitespace().next().ok_or(())?;
    parse_size_token_to_bytes(token).ok_or(())
}

fn current_memory_pressure_state() -> Result<OsMemoryPressureState, ()> {
    let out = Command::new("/usr/bin/memory_pressure")
        .arg("-Q")
        .output()
        .map_err(|_| ())?;
    let txt = String::from_utf8_lossy(&out.stdout).to_ascii_lowercase();
    if txt.contains("critical") {
        Ok(OsMemoryPressureState::Critical)
    } else if txt.contains("warning") {
        Ok(OsMemoryPressureState::Warning)
    } else if txt.contains("normal") || txt.contains("system-wide memory free percentage") {
        Ok(OsMemoryPressureState::Normal)
    } else {
        Ok(OsMemoryPressureState::Unknown)
    }
}

fn parse_page_size(vm_stat_output: &str) -> Option<u64> {
    let start = vm_stat_output.find("page size of ")?;
    let rest = &vm_stat_output[start + "page size of ".len()..];
    let n = rest.split_whitespace().next()?;
    n.parse().ok()
}

fn parse_last_u64(s: &str) -> Result<u64, ()> {
    let cleaned = s.replace('.', "").replace(':', " ");
    cleaned
        .split_whitespace()
        .last()
        .ok_or(())?
        .parse::<u64>()
        .map_err(|_| ())
}

fn parse_size_token_to_bytes(token: &str) -> Option<u64> {
    let lower = token.trim().trim_end_matches('.').to_ascii_lowercase();
    let split_at = lower
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(lower.len());
    let (num, unit) = lower.split_at(split_at);
    let value: f64 = num.parse().ok()?;
    let mult = match unit {
        "k" | "kb" => 1024.0,
        "m" | "mb" => 1024.0 * 1024.0,
        "g" | "gb" => 1024.0 * 1024.0 * 1024.0,
        "t" | "tb" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        _ => 1.0,
    };
    Some((value * mult) as u64)
}
