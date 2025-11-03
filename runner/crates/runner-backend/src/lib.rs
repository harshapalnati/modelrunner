use runner_common::Result;

#[derive(Debug, Clone, Default)]
pub struct LoadParams;

#[derive(Debug, Clone, Default)]
pub struct ModelHandle;

#[derive(Debug, Clone, Default)]
pub struct SequenceState;

#[derive(Debug, Clone, Default)]
pub struct ForwardOutput;

#[derive(Debug, Clone, Default)]
pub struct KvStats;

pub trait InferenceBackend: Send + Sync {
    fn load_model(&self, path: &str, params: LoadParams) -> Result<ModelHandle>;
    fn tokenize(&self, text: &str) -> Result<Vec<u32>>;
    fn detokenize(&self, tokens: &[u32]) -> Result<String>;
    fn forward(&self, requests: &mut [SequenceState]) -> Result<ForwardOutput>;
    fn kv_usage(&self) -> KvStats;
}

#[cfg(feature = "mock")]
pub mod mock {
    use super::*;
    use runner_common::RunnerError;

    #[derive(Default)]
    pub struct MockBackend;

    impl MockBackend { pub fn new() -> Self { Self } }

    impl InferenceBackend for MockBackend {
        fn load_model(&self, _path: &str, _params: LoadParams) -> Result<ModelHandle> {
            Ok(ModelHandle::default())
        }
        fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
            // very naive: bytes as tokens
            Ok(text.as_bytes().iter().map(|b| *b as u32).collect())
        }
        fn detokenize(&self, tokens: &[u32]) -> Result<String> {
            let bytes: Vec<u8> = tokens.iter().map(|t| *t as u8).collect();
            Ok(String::from_utf8_lossy(&bytes).to_string())
        }
        fn forward(&self, _requests: &mut [SequenceState]) -> Result<ForwardOutput> {
            Ok(ForwardOutput::default())
        }
        fn kv_usage(&self) -> KvStats { KvStats::default() }
    }
}

