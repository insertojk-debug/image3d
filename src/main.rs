//! Desktop entry point. The game itself lives in the library (`src/game.rs`) so that Android and
//! iOS can load it too (see `src/lib.rs`).

fn main() {
    image3d_house::run();
}