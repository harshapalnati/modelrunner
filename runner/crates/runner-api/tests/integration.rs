use axum::Router;
use runner_api::app;

#[tokio::test]
async fn metrics_and_generate_and_sse() {
    let app: Router = app();
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let addr = listener.local_addr().unwrap();
    let srv = tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });

    let base = format!("http://{}:{}", addr.ip(), addr.port());
    let client = reqwest::Client::new();

    // metrics
    let r = client.get(format!("{}/metrics", base)).send().await.unwrap();
    assert!(r.status().is_success());

    // generate
    let body = serde_json::json!({"prompt":"Hello"});
    let r = client.post(format!("{}/generate", base)).json(&body).send().await.unwrap();
    assert!(r.status().is_success());

    // sse demo
    let r = client.get(format!("{}/sse/generate", base)).send().await.unwrap();
    assert!(r.status().is_success());

    drop(srv);
}

