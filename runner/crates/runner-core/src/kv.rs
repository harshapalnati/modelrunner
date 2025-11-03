use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};

pub struct NaiveKvCache { pub capacity_bytes: usize }
impl NaiveKvCache { pub fn new(capacity_bytes: usize) -> Self { Self { capacity_bytes } } }

pub struct PagedKvManager {
    capacity_blocks: usize,
    used_blocks: AtomicUsize,
    enable_spill: bool,
    free_list: Mutex<Vec<usize>>, // simplistic free list for defrag demo
}

impl PagedKvManager {
    pub const TOKENS_PER_BLOCK: usize = 32;
    pub fn new(capacity_bytes: usize) -> Arc<Self> {
        let capacity_blocks = capacity_bytes / 4096;
        Arc::new(Self { capacity_blocks, used_blocks: AtomicUsize::new(0), enable_spill: false, free_list: Mutex::new(Vec::new()) })
    }
    pub fn tokens_to_blocks(&self, tokens: usize) -> usize {
        (tokens + Self::TOKENS_PER_BLOCK - 1) / Self::TOKENS_PER_BLOCK
    }
    pub fn try_reserve(self: &Arc<Self>, blocks: usize) -> Option<Reservation> {
        loop {
            let used = self.used_blocks.load(Ordering::Relaxed);
            if used + blocks > self.capacity_blocks { return None; }
            if self.used_blocks.compare_exchange(used, used + blocks, Ordering::SeqCst, Ordering::Relaxed).is_ok() {
                return Some(Reservation { manager: self.clone(), blocks });
            }
        }
    }
    pub fn used_blocks(&self) -> usize { self.used_blocks.load(Ordering::Relaxed) }
    pub fn capacity_blocks(&self) -> usize { self.capacity_blocks }
    fn release(&self, blocks: usize) {
        self.used_blocks.fetch_sub(blocks, Ordering::SeqCst);
        let mut fl = self.free_list.lock().unwrap();
        fl.push(blocks);
    }
    pub fn defragment(&self) { let mut fl = self.free_list.lock().unwrap(); fl.clear(); }
    pub fn enable_spill_to_host(&mut self, enable: bool) { self.enable_spill = enable; }
}

pub struct Reservation { pub(crate) manager: Arc<PagedKvManager>, pub(crate) blocks: usize }
impl Drop for Reservation { fn drop(&mut self) { self.manager.release(self.blocks) } }

#[derive(Default)]
pub struct PrefixCache { counts: Mutex<HashMap<u64, usize>>, tokens: Mutex<HashMap<u64, Vec<u32>>> }
impl PrefixCache {
    pub fn new() -> Arc<Self> { Arc::new(Self::default()) }
    pub fn hash_prefix(&self, text: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        let slice = if text.len() > 256 { &text[..256] } else { text };
        slice.hash(&mut hasher);
        hasher.finish()
    }
    pub fn note(&self, h: u64) { let mut g = self.counts.lock().unwrap(); *g.entry(h).or_insert(0) += 1; }
    pub fn is_common(&self, h: u64) -> bool { let g = self.counts.lock().unwrap(); g.get(&h).copied().unwrap_or(0) >= 2 }
    pub fn put_tokens(&self, h: u64, toks: Vec<u32>) { let mut t = self.tokens.lock().unwrap(); t.insert(h, toks); }
    pub fn get_tokens(&self, h: u64) -> Option<Vec<u32>> { let t = self.tokens.lock().unwrap(); t.get(&h).cloned() }
}

