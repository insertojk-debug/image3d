//! The game, as a library: `src/main.rs` runs it on desktop, and on Android / iOS the
//! `#[bevy_main]` entry below is what the phone starts.
//!
//! `nav` and `world` are the engine-free parts (walking routes, terrain) that `benches/core.rs`
//! benchmarks with Criterion; `touch` is the touch-screen controls.

pub mod nav;
pub mod touch;
pub mod world;

mod game;

pub use game::run;

/// The entry point on phones (Bevy turns this into `android_main` / `main_rs`).
#[cfg(any(target_os = "android", target_os = "ios"))]
#[bevy::prelude::bevy_main]
pub fn main() {
    run();
}