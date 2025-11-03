use std::time::Instant;

#[tokio::main]
async fn main() {
    let n: usize = std::env::var("N").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let prompt = std::env::var("PROMPT").unwrap_or_else(|_| "Hello".into());
    let url = std::env::var("URL").unwrap_or_else(|_| "http://127.0.0.1:8080/generate".into());
    let client = reqwest::Client::new();
    let start = Instant::now();
    let mut tasks = Vec::new();
    for _ in 0..n {
        let c = client.clone();
        let p = prompt.clone();
        let u = url.clone();
        tasks.push(tokio::spawn(async move {
            let body = serde_json::json!({"prompt": p});
            let _ = c.post(&u).json(&body).send().await.ok();
        }));
    }
    for t in tasks { let _ = t.await; }
    println!("completed {} requests in {:.2}s", n, start.elapsed().as_secs_f32());
}

