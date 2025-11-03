use runner_backend::InferenceBackend;
use runner_common::Result;
use crate::sampler::sample_top_k_top_p;

pub fn generate_once(
    backend: &dyn InferenceBackend,
    prompt: &str,
    max_tokens: usize,
) -> Result<String> {
    let _ = max_tokens; // TODO: use with real step loop
    let tokens = backend.tokenize(prompt).unwrap_or_default();
    let _ = sample_top_k_top_p::<rand::rngs::StdRng>(&[0.0_f32; 1], 0, 1.0, 1.0, None);
    let text = backend.detokenize(&tokens).unwrap_or_else(|_| prompt.to_string());
    Ok(text)
}

