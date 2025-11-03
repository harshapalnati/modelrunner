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
    }

    impl Default for RunnerConfig {
        fn default() -> Self {
            Self { model_dir: PathBuf::from("models") }
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
            cfg
        }
    }
}

