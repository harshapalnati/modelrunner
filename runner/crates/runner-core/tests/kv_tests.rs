use runner_core::kv::PagedKvManager;

#[test]
fn reservation_releases_on_drop() {
    let kv = PagedKvManager::new(4096 * 10);
    let used0 = kv.used_blocks();
    {
        let r = kv.try_reserve(2).expect("reserve");
        assert!(kv.used_blocks() >= used0 + 2);
        drop(r);
    }
    assert_eq!(kv.used_blocks(), used0);
}

