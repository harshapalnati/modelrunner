fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Resolve library directory: env override or vendored third_party/llama.cpp
    let mut lib_dirs: Vec<String> = Vec::new();
    let env_dir = std::env::var("LLAMA_CPP_DIR").ok();
    let vendored_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../third_party/llama.cpp");
    if let Some(d) = env_dir.clone() { lib_dirs.push(d); }
    if vendored_dir.exists() { lib_dirs.push(vendored_dir.to_string_lossy().to_string()); }

    if let Some(dir) = lib_dirs.first() {
        // Allow downstream to gate code on presence of FFI
        println!("cargo:rustc-cfg=llama_ffi");
        // Search path can include bin/lib
        println!("cargo:rustc-link-search=native={dir}");
        if let Ok(extra) = std::env::var("LLAMA_CPP_LIB") { println!("cargo:rustc-link-search=native={extra}"); }

        // OS-specific linking
        let target = std::env::var("TARGET").unwrap_or_default();
        // Prefer static if available, fall back to dynamic
        if target.contains("windows") {
            // MSVC libs often built as llama.lib
            println!("cargo:rustc-link-lib=static=llama");
        } else if target.contains("apple-darwin") {
            println!("cargo:rustc-link-lib=static=llama");
            println!("cargo:rustc-link-lib=c++");
        } else {
            // Linux
            println!("cargo:rustc-link-lib=static=llama");
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=dl");
            println!("cargo:rustc-link-lib=pthread");
        }

        // Generate bindings to llama.h if available
        let headers = [
            std::path::Path::new(dir).join("llama.h"),
            std::path::Path::new(dir).join("include/llama.h"),
            vendored_dir.join("llama.h"),
            vendored_dir.join("include/llama.h"),
        ];
        if let Some(h) = headers.iter().find(|p| p.exists()) {
            let bindings = bindgen::Builder::default()
                .header(h.to_string_lossy())
                .allowlist_function("llama_.*")
                .allowlist_type("llama_.*")
                .allowlist_var("LLAMA_.*")
                .clang_arg(format!("-I{}", dir))
                .clang_arg(format!("-I{}", vendored_dir.to_string_lossy()))
                .generate()
                .expect("bindgen failed for llama.h");
            let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("llama_bindings.rs");
            bindings.write_to_file(out_path).expect("write bindings");
            println!("cargo:rerun-if-changed={}", h.to_string_lossy());
        } else {
            println!("cargo:warning=llama.h not found; using manual externs");
        }
    } else {
        println!("cargo:warning=LLAMA_CPP_DIR not set; llama.cpp FFI will be disabled (stub backend)");
    }
}

