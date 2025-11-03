fn main() {
    // Declare custom cfg to silence unexpected_cfg warnings when using llama_ffi gating.
    println!("cargo:rustc-check-cfg=cfg(llama_ffi)");
}

