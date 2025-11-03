pub type Result<T> = core::result::Result<T, RunnerError>;

#[derive(thiserror::Error, Debug)]
pub enum RunnerError {
    #[error("not implemented")] 
    NotImplemented,
    #[error("{0}")]
    Message(String),
}

pub mod config {
    use serde::Deserialize;
    use std::env;
    use std::path::PathBuf;

    #[derive(Debug, Clone, Deserialize)]
    pub struct RunnerConfig {
        pub model_dir: PathBuf,
        pub context_size: Option<usize>,
        pub gpu_layers: Option<usize>,
        pub scheduler_tick_ms: Option<u64>,
        pub max_batch_tokens: Option<usize>,
    }

    impl Default for RunnerConfig {
        fn default() -> Self {
            Self {
                model_dir: PathBuf::from("models"),
                context_size: Some(2048),
                gpu_layers: None,
                scheduler_tick_ms: Some(2),
                max_batch_tokens: Some(1024),
            }
        }
    }

    impl RunnerConfig {
        pub fn load() -> Self {
            if let Ok(path) = env::var("RUNNER_CONFIG") {
                let Ok(text) = std::fs::read_to_string(path) else { return Self::default() };
                let Ok(cfg) = serde_yaml::from_str::<RunnerConfig>(&text) else { return Self::default() };
                return cfg;
            }
            let mut cfg = Self::default();
            if let Ok(dir) = env::var("RUNNER_MODEL_DIR") {
                cfg.model_dir = PathBuf::from(dir);
            }
            if let Some(v) = env::var("RUNNER_CONTEXT_SIZE").ok().and_then(|v| v.parse().ok()) { cfg.context_size = Some(v); }
            if let Some(v) = env::var("RUNNER_GPU_LAYERS").ok().and_then(|v| v.parse().ok()) { cfg.gpu_layers = Some(v); }
            if let Some(v) = env::var("RUNNER_TICK_MS").ok().and_then(|v| v.parse().ok()) { cfg.scheduler_tick_ms = Some(v); }
            if let Some(v) = env::var("RUNNER_MAX_BATCH_TOKENS").ok().and_then(|v| v.parse().ok()) { cfg.max_batch_tokens = Some(v); }
            cfg
        }
    }
}

