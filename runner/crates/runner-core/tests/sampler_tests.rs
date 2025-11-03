use runner_core::sampler::sample_top_k_top_p;

#[test]
fn sample_is_deterministic_with_seed() {
    let logits = vec![0.1, 0.2, 0.3, 0.4];
    let a = sample_top_k_top_p::<rand::rngs::StdRng>(&logits, 0, 1.0, 1.0, Some(42));
    let b = sample_top_k_top_p::<rand::rngs::StdRng>(&logits, 0, 1.0, 1.0, Some(42));
    assert_eq!(a, b);
}

