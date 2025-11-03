fn main() {
    // Declare custom cfg and make build script rerun when changed
    println!("cargo:rustc-check-cfg=cfg(llama_ffi)");
    println!("cargo:rerun-if-changed=build.rs");

    // Candidate roots: explicit env var or vendored directory
    let env_dir = std::env::var("LLAMA_CPP_DIR").ok();
    let vendored_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../third_party/llama.cpp");

    fn canon(p: &std::path::Path) -> std::path::PathBuf {
        std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
    }

    let mut header_roots: Vec<std::path::PathBuf> = Vec::new();
    if let Some(ref d) = env_dir { header_roots.push(canon(std::path::Path::new(d))); }
    if vendored_dir.exists() { header_roots.push(canon(&vendored_dir)); }

    if header_roots.is_empty() {
        println!("cargo:warning=LLAMA_CPP_DIR not set and vendored llama.cpp not found; FFI disabled");
        return;
    }

    // Enable FFI
    println!("cargo:rustc-cfg=llama_ffi");

    // Link search paths: explicit LLAMA_CPP_LIB + common build dirs under roots
    if let Ok(extra) = std::env::var("LLAMA_CPP_LIB") {
        // Allow multiple entries separated by ';' (Windows) or ':' (Unix)
        let parts: Vec<&str> = extra.split(|c| c == ';' || c == ':').filter(|s| !s.is_empty()).collect();
        if parts.is_empty() {
            println!("cargo:rustc-link-search=native={}", canon(std::path::Path::new(&extra)).to_string_lossy());
        } else {
            for p in parts { println!("cargo:rustc-link-search=native={}", canon(std::path::Path::new(p)).to_string_lossy()); }
        }
    }
    for root in &header_roots {
        for p in [
            root.join("build/src/Release"),
            root.join("build/Release"),
            root.join("build/bin/Release"),
            root.join("build/ggml/src/Release"),
            root.join("build/ggml.dir/Release"),
            root.join("build/common/Release"),
            root.join("src/Release"),
        ] {
            if p.exists() { println!("cargo:rustc-link-search=native={}", canon(&p).to_string_lossy()); }
        }
    }

    // OS-specific link libraries
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("windows") {
        println!("cargo:rustc-link-lib=static=llama");
    } else if target.contains("apple-darwin") {
        println!("cargo:rustc-link-lib=static=llama");
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=static=llama");
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=pthread");
    }

    // Generate bindings to llama.h
    let mut header_found = false;
    for root in &header_roots {
        for h in [root.join("llama.h"), root.join("include/llama.h")] {
            if h.exists() {
                let bindings = bindgen::Builder::default()
                    .header(h.to_string_lossy())
                    .allowlist_function("llama_.*")
                    .allowlist_type("llama_.*")
                    .allowlist_var("LLAMA_.*")
                    // include roots: llama headers and ggml headers
                    .clang_arg(format!("-I{}", root.to_string_lossy()))
                    .clang_arg(format!("-I{}", root.join("include").to_string_lossy()))
                    .clang_arg(format!("-I{}", root.join("ggml/include").to_string_lossy()))
                    .clang_arg(format!("-I{}", root.join("ggml/src").to_string_lossy()))
                    .generate()
                    .expect("bindgen failed for llama.h");
                let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("llama_bindings.rs");
                bindings.write_to_file(out_path).expect("write bindings");
                println!("cargo:rerun-if-changed={}", h.to_string_lossy());
                header_found = true;
                break;
            }
        }
        if header_found { break; }
    }
    if !header_found {
        println!("cargo:warning=llama.h not found; using manual externs");
    }
}

