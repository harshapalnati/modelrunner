use runner_backend::{ForwardOutput, InferenceBackend, KvStats, LoadParams, ModelHandle, SequenceState};
use runner_common::{Result, RunnerError};
use std::sync::{Arc, Mutex};

#[cfg(llama_ffi)]
mod ffi {
    // Prefer generated bindings if present
    include!(concat!(env!("OUT_DIR"), "/llama_bindings.rs"));
}

#[derive(Default, Clone)]
pub struct LlamaCppBackend { state: Arc<Mutex<State>> }

#[cfg(llama_ffi)]
#[derive(Default)]
struct State {
    model_loaded: bool,
    model_path: Option<String>,
    n_ctx: i32,
}

#[cfg(not(llama_ffi))]
#[derive(Default)]
struct State {
    model_loaded: bool,
    model_path: Option<String>,
    n_ctx: i32,
}

impl LlamaCppBackend {
    pub fn new() -> Self { Self { state: Arc::new(Mutex::new(State::default())) } }

    #[cfg(llama_ffi)]
    pub fn generate_with_callback<F: FnMut(String)>(
        &self,
        prompt: &str,
        max_tokens: usize,
        mut emit: F,
    ) -> Result<String> {
        self.generate_with_callback_params(prompt, max_tokens, 1.0, 1.0, 0, &mut emit)
    }

    #[cfg(llama_ffi)]
    pub fn generate_with_callback_params<F: FnMut(String)>(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        mut emit: F,
    ) -> Result<String> {
        unsafe {
            // Initialize backend (older APIs return void)
            ffi::llama_backend_init();
            let st = self.state.lock().unwrap();
            let Some(ref model_path) = st.model_path else { return Err(RunnerError::Message("model not loaded".into())) };
            let cpath = std::ffi::CString::new(model_path.as_str()).unwrap();
            let mut mparams = ffi::llama_model_default_params();
            // For generation we need full model; keep defaults
            let model = ffi::llama_load_model_from_file(cpath.as_ptr(), mparams);
            if model.is_null() { return Err(RunnerError::Message("llama_load_model_from_file failed".into())); }
            let mut cparams = ffi::llama_context_default_params();
            // reasonable default context
            cparams.n_ctx = 2048u32;
            let ctx = ffi::llama_new_context_with_model(model, cparams);
            if ctx.is_null() { ffi::llama_free_model(model); return Err(RunnerError::Message("llama_new_context_with_model failed".into())) }

            // tokenize prompt
            let cprompt = std::ffi::CString::new(prompt).unwrap();
            // API expects model*, text ptr, text_len (i32), tokens*, n_max_tokens (i32), special flags
            let n = ffi::llama_tokenize(model, cprompt.as_ptr(), 0i32, std::ptr::null_mut(), 0i32, true, false);
            let mut ptoks: Vec<i32> = vec![0; n as usize];
            let n2 = ffi::llama_tokenize(model, cprompt.as_ptr(), 0i32, ptoks.as_mut_ptr(), ptoks.len() as i32, true, false);
            let ptoks = &ptoks[..(n2 as usize)];

            let mut n_past: i32 = 0;
            // evaluate all prompt tokens first using decode+batch
            if !ptoks.is_empty() {
                let mut toks: Vec<ffi::llama_token> = ptoks.iter().map(|&t| t as ffi::llama_token).collect();
                let batch = ffi::llama_batch_get_one(toks.as_mut_ptr(), toks.len() as i32, 0, 0);
                let rc = ffi::llama_decode(ctx, batch);
                ffi::llama_batch_free(batch);
                if rc != 0 { ffi::llama_free(ctx); ffi::llama_free_model(model); return Err(RunnerError::Message("llama_decode prompt failed".into())) }
                n_past += ptoks.len() as i32;
            }

            let mut generated = String::new();
            let vocab = ffi::llama_n_vocab(model);
            let eos = ffi::llama_token_eos(model);
            let mut cur: i32 = -1; // -1 indicates take logits after prompt
            for _step in 0..max_tokens {
                if cur >= 0 {
                    let mut one: [ffi::llama_token; 1] = [cur as ffi::llama_token];
                    let batch = ffi::llama_batch_get_one(one.as_mut_ptr(), 1, n_past, 0);
                    let rc = ffi::llama_decode(ctx, batch);
                    ffi::llama_batch_free(batch);
                    if rc != 0 { break; }
                    n_past += 1;
                }
                let logits = ffi::llama_get_logits(ctx);
                if logits.is_null() { break; }
                let slice = std::slice::from_raw_parts(logits, vocab as usize);
                // Greedy pick (avoid external sampler dependency)
                let mut best_id: i32 = 0;
                let mut best_val = f32::MIN;
                for (i, &v) in slice.iter().enumerate() { if v > best_val { best_val = v; best_id = i as i32; } }
                if best_id == eos { break; }

                // detokenize this piece
                let needed = ffi::llama_token_to_piece(model, best_id, std::ptr::null_mut(), 0);
                if needed > 0 {
                    let mut buf: Vec<i8> = vec![0; needed as usize + 1];
                    let written = ffi::llama_token_to_piece(model, best_id, buf.as_mut_ptr(), buf.len() as i32);
                    if written > 0 {
                        let bytes = std::slice::from_raw_parts(buf.as_ptr() as *const u8, written as usize);
                        let piece = String::from_utf8_lossy(bytes).to_string();
                        emit(piece.clone());
                        generated.push_str(&piece);
                    }
                }
                cur = best_id;
            }

            ffi::llama_free(ctx);
            ffi::llama_free_model(model);
            Ok(generated)
        }
    }
}

impl InferenceBackend for LlamaCppBackend {
    fn load_model(&self, path: &str, params: LoadParams) -> Result<ModelHandle> {
        #[cfg(llama_ffi)]
        unsafe {
            // Initialize backend (older APIs return void)
            ffi::llama_backend_init();
            let cpath = std::ffi::CString::new(path).unwrap();
            let mut mparams = ffi::llama_model_default_params();
            mparams.vocab_only = false;
            let model = ffi::llama_load_model_from_file(cpath.as_ptr(), mparams);
            if model.is_null() { return Err(RunnerError::Message("llama_load_model_from_file failed".into())); }
            let mut cparams = ffi::llama_context_default_params();
            cparams.n_ctx = params.n_ctx as u32;
            let ctx = ffi::llama_new_context_with_model(model, cparams);
            if ctx.is_null() { ffi::llama_free_model(model); return Err(RunnerError::Message("llama_new_context_with_model failed".into())); }
            // Do not retain raw pointers across threads; free immediately, store only path
            ffi::llama_free(ctx);
            ffi::llama_free_model(model);
            if let Ok(mut st) = self.state.lock() { st.model_loaded = true; st.model_path = Some(path.to_string()); st.n_ctx = cparams.n_ctx as i32; }
            return Ok(ModelHandle::default());
        }
        #[allow(unreachable_code)]
        Err(RunnerError::NotImplemented)
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        #[cfg(llama_ffi)]
        {
            // Without persistent model context, fall back to naive tokenization for this method
            return Ok(text.as_bytes().iter().map(|b| *b as u32).collect());
        }
        #[allow(unreachable_code)]
        Ok(text.as_bytes().iter().map(|b| *b as u32).collect())
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        #[cfg(llama_ffi)]
        {
            // Naive fallback
            let bytes: Vec<u8> = tokens.iter().map(|t| *t as u8).collect();
            return Ok(String::from_utf8_lossy(&bytes).to_string());
        }
        #[allow(unreachable_code)]
        {
            let bytes: Vec<u8> = tokens.iter().map(|t| *t as u8).collect();
            Ok(String::from_utf8_lossy(&bytes).to_string())
        }
    }

    fn forward(&self, _requests: &mut [SequenceState]) -> Result<ForwardOutput> {
        #[cfg(llama_ffi)]
        {
            // Not used in current flow; return default
            return Ok(ForwardOutput::default());
        }
        #[allow(unreachable_code)]
        { Ok(ForwardOutput::default()) }
    }

    fn kv_usage(&self) -> KvStats { KvStats::default() }
}

