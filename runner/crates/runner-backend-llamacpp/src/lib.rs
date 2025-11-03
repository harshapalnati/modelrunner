use runner_backend::{ForwardOutput, InferenceBackend, KvStats, LoadParams, ModelHandle, SequenceState};
use runner_common::{Result, RunnerError};
use std::sync::{Arc, Mutex};

#[cfg(llama_ffi)]
mod ffi {
    // Prefer generated bindings if present
    include!(concat!(env!("OUT_DIR"), "/llama_bindings.rs"));
}

#[derive(Default, Clone)]
pub struct LlamaCppBackend {
    state: Arc<Mutex<State>>,    
}

#[derive(Default)]
struct State {
    model_loaded: bool,
    model_path: Option<String>,
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
        unsafe {
            if ffi::llama_backend_init() != 0 {
                return Err(RunnerError::Message("llama_backend_init failed".into()));
            }
            let st = self.state.lock().unwrap();
            let Some(ref model_path) = st.model_path else { return Err(RunnerError::Message("model not loaded".into())) };
            let cpath = std::ffi::CString::new(model_path.as_str()).unwrap();
            let mut mparams = ffi::llama_model_default_params();
            // For generation we need full model; keep defaults
            let model = ffi::llama_load_model_from_file(cpath.as_ptr(), mparams);
            if model.is_null() { return Err(RunnerError::Message("llama_load_model_from_file failed".into())); }
            let mut cparams = ffi::llama_context_default_params();
            // reasonable default context
            cparams.n_ctx = 2048;
            let ctx = ffi::llama_new_context_with_model(model, cparams);
            if ctx.is_null() { ffi::llama_free_model(model); return Err(RunnerError::Message("llama_new_context_with_model failed".into())) }

            // tokenize prompt
            let cprompt = std::ffi::CString::new(prompt).unwrap();
            let n = ffi::llama_tokenize(ctx, cprompt.as_ptr(), std::ptr::null_mut(), 0, true, false);
            let mut ptoks: Vec<i32> = vec![0; n as usize];
            let n2 = ffi::llama_tokenize(ctx, cprompt.as_ptr(), ptoks.as_mut_ptr(), ptoks.len() as i32, true, false);
            let ptoks = &ptoks[..(n2 as usize)];

            let mut n_past: i32 = 0;
            // evaluate all prompt tokens first
            if !ptoks.is_empty() {
                let rc = ffi::llama_eval(ctx, ptoks.as_ptr(), ptoks.len() as i32, n_past, 0);
                if rc != 0 { ffi::llama_free(ctx); ffi::llama_free_model(model); return Err(RunnerError::Message("llama_eval prompt failed".into())) }
                n_past += ptoks.len() as i32;
            }

            let mut generated = String::new();
            let vocab = ffi::llama_n_vocab(model);
            let eos = ffi::llama_token_eos(model);
            let mut cur: i32 = -1; // -1 indicates take logits after prompt
            for _step in 0..max_tokens {
                if cur >= 0 {
                    let rc = ffi::llama_eval(ctx, &cur as *const i32, 1, n_past, 0);
                    if rc != 0 { break; }
                    n_past += 1;
                }
                let logits = ffi::llama_get_logits(ctx);
                if logits.is_null() { break; }
                let slice = std::slice::from_raw_parts(logits, vocab as usize);
                // greedy argmax
                let mut best_id: i32 = 0;
                let mut best_val = f32::MIN;
                for (i, &v) in slice.iter().enumerate() {
                    if v > best_val { best_val = v; best_id = i as i32; }
                }
                if best_id == eos { break; }

                // detokenize this piece
                let needed = ffi::llama_token_to_piece(ctx, best_id, std::ptr::null_mut(), 0, false);
                if needed > 0 {
                    let mut buf: Vec<i8> = vec![0; needed as usize + 1];
                    let written = ffi::llama_token_to_piece(ctx, best_id, buf.as_mut_ptr(), buf.len() as i32, false);
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
    fn load_model(&self, path: &str, _params: LoadParams) -> Result<ModelHandle> {
        #[cfg(llama_ffi)]
        unsafe {
            if ffi::llama_backend_init() != 0 { return Err(RunnerError::Message("llama_backend_init failed".into())); }

            // Try loading model to ensure headers/libs are coherent
            let cpath = std::ffi::CString::new(path).unwrap();
            let mut params = ffi::llama_model_default_params();
            // vocab_only = true allows tokenization/detokenization without full memory
            params.vocab_only = true;
            let model = ffi::llama_load_model_from_file(cpath.as_ptr(), params);
            if model.is_null() {
                return Err(RunnerError::Message("llama_load_model_from_file failed".into()));
            }
            // Free immediately; we only validate availability here.
            ffi::llama_free_model(model);
            if let Ok(mut st) = self.state.lock() {
                st.model_loaded = true;
                st.model_path = Some(path.to_string());
            }
            return Ok(ModelHandle::default());
        }
        #[allow(unreachable_code)]
        Err(RunnerError::NotImplemented)
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        #[cfg(llama_ffi)]
        unsafe {
            let st = self.state.lock().unwrap();
            let Some(ref model_path) = st.model_path else { return Ok(vec![]) };
            let cpath = std::ffi::CString::new(model_path.as_str()).unwrap();
            let mut params = ffi::llama_model_default_params();
            params.vocab_only = true;
            let model = ffi::llama_load_model_from_file(cpath.as_ptr(), params);
            if model.is_null() { return Ok(vec![]) }
            let ctx_params = ffi::llama_context_default_params();
            let ctx = ffi::llama_new_context_with_model(model, ctx_params);
            if ctx.is_null() { ffi::llama_free_model(model); return Ok(vec![]) }

            let ctext = std::ffi::CString::new(text).unwrap();
            // First call to get length
            let n = ffi::llama_tokenize(
                ctx,
                ctext.as_ptr(),
                std::ptr::null_mut(),
                0,
                true,
                false,
            );
            let mut buf: Vec<i32> = vec![0; n as usize];
            let n2 = ffi::llama_tokenize(
                ctx,
                ctext.as_ptr(),
                buf.as_mut_ptr(),
                buf.len() as i32,
                true,
                false,
            );
            ffi::llama_free(ctx);
            ffi::llama_free_model(model);
            let toks: Vec<u32> = buf[..(n2 as usize)].iter().map(|t| *t as u32).collect();
            return Ok(toks);
        }
        #[allow(unreachable_code)]
        Ok(text.as_bytes().iter().map(|b| *b as u32).collect())
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        #[cfg(llama_ffi)]
        unsafe {
            let st = self.state.lock().unwrap();
            let Some(ref model_path) = st.model_path else { return Ok(String::new()) };
            let cpath = std::ffi::CString::new(model_path.as_str()).unwrap();
            let mut params = ffi::llama_model_default_params();
            params.vocab_only = true;
            let model = ffi::llama_load_model_from_file(cpath.as_ptr(), params);
            if model.is_null() { return Ok(String::new()) }
            let ctx_params = ffi::llama_context_default_params();
            let ctx = ffi::llama_new_context_with_model(model, ctx_params);
            if ctx.is_null() { ffi::llama_free_model(model); return Ok(String::new()) }

            let mut out = String::new();
            for &t in tokens {
                // Query piece size first
                let needed = ffi::llama_token_to_piece(
                    ctx,
                    t as i32,
                    std::ptr::null_mut(),
                    0,
                    false,
                );
                if needed > 0 {
                    let mut buf: Vec<i8> = vec![0; needed as usize + 1];
                    let written = ffi::llama_token_to_piece(
                        ctx,
                        t as i32,
                        buf.as_mut_ptr(),
                        buf.len() as i32,
                        false,
                    );
                    if written > 0 {
                        let slice = std::slice::from_raw_parts(buf.as_ptr() as *const u8, written as usize);
                        out.push_str(&String::from_utf8_lossy(slice));
                    }
                }
            }
            ffi::llama_free(ctx);
            ffi::llama_free_model(model);
            return Ok(out);
        }
        #[allow(unreachable_code)]
        {
            let bytes: Vec<u8> = tokens.iter().map(|t| *t as u8).collect();
            Ok(String::from_utf8_lossy(&bytes).to_string())
        }
    }

    fn forward(&self, _requests: &mut [SequenceState]) -> Result<ForwardOutput> {
        // Placeholder; once wired, this will call into llama_decode
        Ok(ForwardOutput::default())
    }

    fn kv_usage(&self) -> KvStats { KvStats::default() }
}

