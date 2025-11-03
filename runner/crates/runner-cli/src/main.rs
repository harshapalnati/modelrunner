use axum::Router;
use clap::{Parser, Subcommand, Args};
use runner_api::app;
use runner_backend::mock::MockBackend;
use runner_core::decode::generate_once;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use tracing_opentelemetry::OpenTelemetryLayer;

#[derive(Parser, Debug)]
#[command(name = "runner", version, about = "Next Inference CLI (skeleton)")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Serve,
    Run(RunArgs),
    List,
    Stats,
    Pull(PullArgs),
    Version,
}

#[derive(Args, Debug)]
struct RunArgs {
    #[arg(short, long)]
    prompt: String,
    #[arg(short = 'n', long, default_value_t = 128)]
    max_tokens: usize,
}

#[derive(Args, Debug)]
struct PullArgs {
    /// Source URL (hf://org/repo/file or https URL)
    source: String,
    /// Optional model name to save under models/<name>.gguf
    #[arg(short, long)]
    name: Option<String>,
}

#[tokio::main]
async fn main() {
    init_tracing();

    let cli = Cli::parse();
    match cli.command {
        Commands::Serve => serve().await,
        Commands::Run(args) => run_local(args).await,
        Commands::List => list_models().await,
        Commands::Pull(args) => pull_model(args).await,
        Commands::Stats => stats().await,
        Commands::Version => println!("{}", env!("CARGO_PKG_VERSION")),
    }
}

async fn serve() {
    let app: Router = app();
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", 8080)).await.unwrap();
    tracing::info!("listening on http://0.0.0.0:8080");
    let shutdown = async {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("shutdown signal received");
    };
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
        .unwrap();
}

async fn run_local(args: RunArgs) {
    let backend = MockBackend::new();
    let text = generate_once(&backend, &args.prompt, args.max_tokens).unwrap_or_default();
    println!("{}", text);
}

async fn list_models() {
    let cfg = runner_common::config::RunnerConfig::load();
    let path = cfg.model_dir;
    match std::fs::read_dir(&path) {
        Ok(read_dir) => {
            println!("models dir: {}", path.display());
            for entry in read_dir.flatten() {
                println!("- {}", entry.path().display());
            }
        }
        Err(_) => println!("no models directory at {}", path.display()),
    }
}

async fn stats() {
    use sysinfo::{System, SystemExt, CpuExt};
    let mut sys = System::new_all();
    sys.refresh_all();
    let total_mem = sys.total_memory();
    let used_mem = sys.used_memory();
    let cpu_avg: f32 = sys.cpus().iter().map(|c| c.cpu_usage()).sum::<f32>() / (sys.cpus().len() as f32);
    println!("CPU: {:.1}%", cpu_avg);
    println!("Memory: {} / {} MiB", used_mem / 1024 / 1024, total_mem / 1024 / 1024);
    println!("GPU: see /metrics for NVML-based GPU stats if NVIDIA is present");
}

fn init_tracing() {
    let env_filter = tracing_subscriber::EnvFilter::new(
        std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
    );

    if let Ok(endpoint) = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT") {
        let tracer = opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(opentelemetry_otlp::new_exporter().tonic().with_endpoint(endpoint))
            .install_simple()
            .ok();
        if let Some(tracer) = tracer {
            tracing_subscriber::registry()
                .with(env_filter)
                .with(tracing_subscriber::fmt::layer())
                .with(OpenTelemetryLayer::new(tracer))
                .init();
            return;
        }
    }

    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer())
        .init();
}

async fn pull_model(args: PullArgs) {
    let cfg = runner_common::config::RunnerConfig::load();
    let models_dir = cfg.model_dir;
    let _ = std::fs::create_dir_all(&models_dir);

    let (url, filename) = if let Some(rest) = args.source.strip_prefix("hf://") {
        // naive hf://org/repo/file mapping to https
        let parts: Vec<&str> = rest.split('/').collect();
        if parts.len() < 3 {
            eprintln!("invalid hf:// URL; expected hf://org/repo/file");
            return;
        }
        let org = parts[0];
        let repo = parts[1];
        let file = parts[2..].join("/");
        (format!("https://huggingface.co/{}/{}/resolve/main/{}", org, repo, file), file)
    } else {
        let fname = args.source.split('/').last().unwrap_or("model.gguf").to_string();
        (args.source, fname)
    };

    let name = args.name.unwrap_or_else(|| filename.clone());
    let target_path = models_dir.join(name);
    println!("Downloading to {}", target_path.display());

    match reqwest::get(&url).await {
        Ok(resp) => {
            if !resp.status().is_success() {
                eprintln!("download failed: status {}", resp.status());
                return;
            }
            let bytes = match resp.bytes().await { Ok(b) => b, Err(e) => { eprintln!("download error: {}", e); return; } };
            if let Err(e) = std::fs::write(&target_path, &bytes) {
                eprintln!("write error: {}", e);
                return;
            }
            println!("Saved {} bytes", bytes.len());
        }
        Err(e) => eprintln!("request error: {}", e),
    }
}

