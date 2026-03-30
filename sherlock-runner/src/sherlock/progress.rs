/// No-op progress emission for Lambda (no Tauri AppHandle).
pub fn emit_progress(_progress: f64, _stage: &str) {
    log::info!("[progress] {:.0}% — {}", _progress * 100.0, _stage);
}
