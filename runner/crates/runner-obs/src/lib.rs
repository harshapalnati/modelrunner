//! Observability utilities: GPU and system metrics

use once_cell::sync::Lazy;
use prometheus::{Gauge, IntGauge};

static GPU_UTIL: Lazy<Gauge> = Lazy::new(|| prometheus::register_gauge!("runner_gpu_utilization", "GPU utilization percent").unwrap());
static GPU_MEM_USED: Lazy<IntGauge> = Lazy::new(|| prometheus::register_int_gauge!("runner_gpu_memory_bytes", "GPU memory used (bytes)").unwrap());
static GPU_TEMP: Lazy<Gauge> = Lazy::new(|| prometheus::register_gauge!("runner_gpu_temperature_celsius", "GPU temperature in C").unwrap());

pub fn init() {
    // Touch statics to ensure registration and avoid dead_code warnings when NVML is disabled.
    let _ = &*GPU_UTIL;
    let _ = &*GPU_MEM_USED;
    let _ = &*GPU_TEMP;
}

pub fn spawn_gpu_polling() {
    #[cfg(feature = "nvidia")]
    tokio::spawn(async move {
        let nvml = match nvml_wrapper::NVML::init() { Ok(n) => n, Err(_) => return };
        let device = match nvml.device_by_index(0) { Ok(d) => d, Err(_) => return };
        loop {
            if let Ok(util) = device.utilization_rates() { GPU_UTIL.set(util.gpu as f64); }
            if let Ok(mem) = device.memory_info() { GPU_MEM_USED.set(mem.used as i64); }
            if let Ok(temp) = device.temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu) { GPU_TEMP.set(temp as f64); }
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        }
    });
}

