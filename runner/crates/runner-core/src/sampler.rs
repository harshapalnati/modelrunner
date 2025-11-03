use rand::prelude::*;

pub fn sample_top_k_top_p<R: Rng + ?Sized>(
    logits: &[f32],
    top_k: usize,
    top_p: f32,
    temperature: f32,
    seed: Option<u64>,
) -> usize {
    let mut rng: StdRng = match seed { Some(s) => SeedableRng::seed_from_u64(s), None => StdRng::from_entropy() };
    if logits.is_empty() { return 0; }
    let mut pairs: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &l)| (i, l / temperature.max(1e-4))).collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut cutoff = pairs.len();
    if top_k > 0 { cutoff = cutoff.min(top_k); }
    let mut sum = 0.0_f32;
    let mut probs: Vec<(usize, f32)> = Vec::with_capacity(cutoff);
    for &(i, l) in &pairs[..cutoff] {
        let p = (l).exp();
        probs.push((i, p));
        sum += p;
    }
    probs.iter_mut().for_each(|p| p.1 /= sum.max(1e-9));
    if top_p < 1.0 {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut acc = 0.0_f32;
        let mut keep = 0;
        for &(_, p) in &probs { acc += p; keep += 1; if acc >= top_p { break; } }
        probs.truncate(keep);
        let z: f32 = probs.iter().map(|p| p.1).sum();
        for p in &mut probs { p.1 /= z.max(1e-9); }
    }
    let r: f32 = rng.gen();
    let mut acc = 0.0_f32;
    for (i, p) in probs { acc += p; if r <= acc { return i; } }
    pairs[0].0
}

