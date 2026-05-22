//! objeta-parser — Safetensors mmap loader with lazy tensor access.
//!
//! Design principles:
//! - mmap the entire safetensors file (zero-copy where possible)
//! - Parse the JSON header to build a tensor index
//! - Each tensor view is a (dtype, shape, offset) triple — data is read on demand
//! - Support for bf16 → f32 conversion on access

use memmap2::Mmap;
use objeta_core::{ObjetaError, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

pub mod deepseek;
pub mod sanity;

// ── Tensor index entry ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub offset: usize,   // byte offset from data_start
    pub nbytes: usize,   // total bytes of raw data
    pub nelem: usize,    // number of elements
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    BF16,
    I64,
    I32,
    I8,
    U8,
    F8_E8M0,
    F8_E4M3,
    BOOL,
}

impl Dtype {
    pub fn element_size(&self) -> usize {
        match self {
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
            Dtype::I64 => 8,
            Dtype::I8 | Dtype::U8 | Dtype::F8_E8M0 | Dtype::BOOL => 1,
            Dtype::F8_E4M3 => 1,
        }
    }

    pub fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s {
            "F32" | "FLOAT32" => Ok(Dtype::F32),
            "F16" | "FLOAT16" => Ok(Dtype::F16),
            "BF16" | "BFLOAT16" => Ok(Dtype::BF16),
            "I64" | "INT64" => Ok(Dtype::I64),
            "I32" | "INT32" => Ok(Dtype::I32),
            "I8" | "INT8" => Ok(Dtype::I8),
            "U8" | "UINT8" => Ok(Dtype::U8),
            "F8_E8M0" => Ok(Dtype::F8_E8M0),
            "F8_E4M3" => Ok(Dtype::F8_E4M3),
            "BOOL" => Ok(Dtype::BOOL),
            other => Err(format!("Unknown dtype: {}", other)),
        }
    }
}

// ── Safetensors header (JSON) ─────────────────────────────────────────────

/// Parsed safetensors header. The format is a flat JSON object where
/// each key is either `__metadata__` (ignored) or a tensor name pointing
/// to its dtype/shape/data_offsets.
struct SafetensorsHeader {
    tensors: HashMap<String, TensorEntry>,
}

#[derive(Deserialize)]
struct TensorEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

// ── Model Weights ─────────────────────────────────────────────────────────

/// Lazy mmap-based access to model weights.
///
/// Usage:
/// ```ignore
/// let mut loader = ModelWeights::open("model.safetensors")?;
/// let gate_w: &[f32] = loader.load_f32("model.layers.0.mlp.gate_proj.weight")?;
/// ```
pub struct ModelWeights {
    /// All safetensors files loaded as mmaps
    buffers: Vec<Mmap>,
    /// Tensor name → (mmap_index, info)
    index: HashMap<String, (usize, TensorInfo)>,
    /// Base directory for sharded models
    base_dir: PathBuf,
}

impl ModelWeights {
    /// Open a single safetensors file or a directory of shards.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut buffers = Vec::new();
        let mut index = HashMap::new();

        if path.is_dir() {
            // Sharded model: load all .safetensors files
            let mut sf_files: Vec<_> = std::fs::read_dir(path)?
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.file_name()
                        .to_str()
                        .is_some_and(|n| n.ends_with(".safetensors"))
                })
                .collect();
            sf_files.sort_by_key(|e| e.file_name());

            let base_dir = path.to_path_buf();

            for entry in sf_files {
                let file_path = entry.path();
                let file = File::open(&file_path)?;
                let mmap = unsafe { Mmap::map(&file)? };
                let (header, header_len) = Self::parse_header(&mmap)?;

                let data_start = 8 + header_len;
                let buffer_idx = buffers.len();
                buffers.push(mmap);

                for (name, entry) in header.tensors {
                    let nelem: usize = entry.shape.iter().product();
                    let nbytes = entry.data_offsets.1 - entry.data_offsets.0;
                    let info = TensorInfo {
                        dtype: Dtype::from_str(&entry.dtype)
                            .map_err(ObjetaError::Parse)?,
                        shape: entry.shape,
                        offset: data_start + entry.data_offsets.0,
                        nbytes,
                        nelem,
                    };
                    index.insert(name, (buffer_idx, info));
                }
            }

            Ok(ModelWeights { buffers, index, base_dir })
        } else {
            // Single safetensors file
            let file = File::open(path)?;
            let mmap = unsafe { Mmap::map(&file)? };
            let (header, header_len) = Self::parse_header(&mmap)?;

            let data_start = 8 + header_len;
            let base_dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();

            for (name, entry) in header.tensors {
                let nelem: usize = entry.shape.iter().product();
                let nbytes = entry.data_offsets.1 - entry.data_offsets.0;
                let info = TensorInfo {
                    dtype: Dtype::from_str(&entry.dtype)
                        .map_err(ObjetaError::Parse)?,
                    shape: entry.shape,
                    offset: data_start + entry.data_offsets.0,
                    nbytes,
                    nelem,
                };
                index.insert(name, (0, info));
            }

            buffers.push(mmap);
            Ok(ModelWeights { buffers, index, base_dir })
        }
    }

    fn parse_header(mmap: &Mmap) -> Result<(SafetensorsHeader, usize)> {
        let header_len = u64::from_le_bytes(
            mmap[..8].try_into().map_err(|_| ObjetaError::Parse("file too short".into()))?,
        ) as usize;
        let header_start = 8;
        let header_end = header_start + header_len;
        if header_end > mmap.len() {
            return Err(ObjetaError::Parse("header length exceeds file size".into()));
        }
        let header_json = std::str::from_utf8(&mmap[header_start..header_end])
            .map_err(|e| ObjetaError::Parse(format!("invalid UTF-8 in header: {}", e)))?;

        // Parse as raw JSON object, filtering out __metadata__
        let raw: serde_json::Value = serde_json::from_str(header_json)
            .map_err(|e| ObjetaError::Parse(format!("invalid header JSON: {}", e)))?;

        let mut tensors = HashMap::new();
        if let Some(obj) = raw.as_object() {
            for (key, val) in obj {
                if key == "__metadata__" {
                    continue;
                }
                let entry: TensorEntry = serde_json::from_value(val.clone())
                    .map_err(|e| ObjetaError::Parse(format!(
                        "invalid tensor entry for '{}': {}", key, e
                    )))?;
                tensors.insert(key.clone(), entry);
            }
        }

        let header = SafetensorsHeader { tensors };
        Ok((header, header_len))
    }

    /// Get tensor info by name.
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|(_, info)| info)
    }

    /// List all tensor names.
    pub fn keys(&self) -> Vec<&String> {
        self.index.keys().collect()
    }

    /// Check if a tensor exists.
    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Get tensor as raw bytes slice.
    pub fn get_raw(&self, name: &str) -> Result<&[u8]> {
        let (buf_idx, info) = self
            .index
            .get(name)
            .ok_or_else(|| ObjetaError::MissingTensor(name.to_string()))?;
        let buf = &self.buffers[*buf_idx];
        Ok(&buf[info.offset..info.offset + info.nbytes])
    }

    /// Get tensor as f32 slice. Converts bf16/f16 on the fly if needed.
    pub fn get_f32(&self, name: &str, out: &mut Vec<f32>) -> Result<()> {
        let (buf_idx, info) = self
            .index
            .get(name)
            .ok_or_else(|| ObjetaError::MissingTensor(name.to_string()))?;
        let raw = &self.buffers[*buf_idx][info.offset..info.offset + info.nbytes];

        out.clear();
        out.reserve(info.nelem);

        match info.dtype {
            Dtype::F32 => {
                let ptr = raw.as_ptr() as *const f32;
                let len = raw.len() / 4;
                let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                out.extend_from_slice(slice);
            }
            Dtype::F16 => {
                let ptr = raw.as_ptr() as *const u16;
                let len = raw.len() / 2;
                let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                out.extend(slice.iter().map(|&v| half_to_f32(v)));
            }
            Dtype::BF16 => {
                let ptr = raw.as_ptr() as *const u16;
                let len = raw.len() / 2;
                let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                out.extend(slice.iter().map(|&v| bf16_to_f32(v)));
            }
            _ => return Err(ObjetaError::Parse(format!("cannot convert {:?} to f32", info.dtype))),
        }
        Ok(())
    }

    /// Get a 2D weight matrix as row-major f32.
    /// Returns (rows, cols, data) where data is in row-major order.
    pub fn get_matrix(&self, name: &str) -> Result<(usize, usize, Vec<f32>)> {
        let info = self.info(name).ok_or_else(|| ObjetaError::MissingTensor(name.to_string()))?;
        let mut data = Vec::new();
        self.get_f32(name, &mut data)?;

        let (rows, cols) = if info.shape.len() == 2 {
            (info.shape[0], info.shape[1])
        } else {
            return Err(ObjetaError::Parse(format!(
                "expected 2D tensor for {}, got shape {:?}",
                name, info.shape
            )));
        };
        Ok((rows, cols, data))
    }

    /// Number of tensors loaded.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }
}

// ── FP conversion helpers ─────────────────────────────────────────────────

#[inline]
fn half_to_f32(h: u16) -> f32 {
    // IEEE 754-2008 binary16 → binary32
    let sign = (h >> 15) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mantissa = (h & 0x3ff) as u32;

    if exp == 0 {
        if mantissa == 0 {
            f32::from_bits(sign << 31)
        } else {
            // Subnormal
            let m2 = mantissa;
            let e2 = 1 - 15 - 10;
            (m2 as f32) * 2f32.powi(e2) * if sign == 0 { 1.0 } else { -1.0 }
        }
    } else if exp == 31 {
        if mantissa == 0 {
            f32::from_bits((sign << 31) | 0x7f80_0000)
        } else {
            f32::NAN
        }
    } else {
        let exp32 = exp + (127 - 15);
        f32::from_bits((sign << 31) | (exp32 << 23) | (mantissa << 13))
    }
}

#[inline]
fn bf16_to_f32(h: u16) -> f32 {
    // bfloat16 → float32: just shift left by 16 bits
    f32::from_bits((h as u32) << 16)
}

// ── Model configuration loader ────────────────────────────────────────────

/// Parse model config.json for architectural parameters.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,
    #[serde(default)]
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: usize,
    pub vocab_size: usize,
    #[serde(default)]
    pub model_type: String,
}

impl ModelConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let config_path = if path.is_dir() {
            path.join("config.json")
        } else {
            path.parent()
                .unwrap_or(Path::new("."))
                .join("config.json")
        };

        let content = std::fs::read_to_string(&config_path)
            .map_err(ObjetaError::Io)?;
        let mut config: ModelConfig = serde_json::from_str(&content)
            .map_err(|e| ObjetaError::Parse(format!("invalid config.json: {}", e)))?;

        // Apply defaults for missing fields
        if config.head_dim == 0 {
            config.head_dim = config.hidden_size / config.num_attention_heads;
        }
        if config.num_key_value_heads == 0 {
            config.num_key_value_heads = config.num_attention_heads;
        }
        if config.intermediate_size == 0 {
            // Common fallbacks: some models use different field names
            config.intermediate_size = config.hidden_size * 4;
        }
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_element_size() {
        assert_eq!(Dtype::F32.element_size(), 4);
        assert_eq!(Dtype::F16.element_size(), 2);
        assert_eq!(Dtype::BF16.element_size(), 2);
    }

    #[test]
    fn test_bf16_to_f32() {
        // bf16 1.0 = 0x3f80
        assert_eq!(bf16_to_f32(0x3f80u16), 1.0f32);
        // bf16 0.0 = 0x0000
        assert_eq!(bf16_to_f32(0x0000u16), 0.0f32);
    }

    #[test]
    fn test_half_to_f32() {
        // f16 1.0 = 0x3c00
        assert_eq!(half_to_f32(0x3c00u16), 1.0f32);
        assert_eq!(half_to_f32(0x0000u16), 0.0f32);
    }
}