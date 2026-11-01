# Playing on a phone or tablet

The game now has touch controls and can be built for the browser (works on any phone), Android
and iOS. **None of this has been compiled or run yet** - treat it as a first draft.

## Controls

The touch screen pretends to be the mouse and keyboard, so everything in the game works as before.

| Touch | Same as |
|---|---|
| Drag on the left half | W / A / S / D (thumb stick appears where you touch) |
| Drag on the right | Hold the right mouse button and move: look around |
| Tap on the right | Left click (talk, pick up, equip a tool, click a fish or can ...) |
| Press, wait a moment, then drag | Hold the left mouse button and move (drag objects, swing tools, paint) |
| **Jump** | Space |
| **Kneel** (toggle) | Shift |
| **Drop** | Q |
| **Use** | E (get off the horse) |
| **Fire** | C (feed the embers with bark) |
| **Clear** | X |
| **Hungry** | H (make Tom hungry) |
| **Throw** | Middle mouse button (throw an apple) |
| **Poss** | F (possess the animal or person you look at) |
| **+** / **-** | Mouse wheel (push / pull what you hold) |
| **X** (top right) | Esc (close the chat) |

On Android / iOS the buttons are always visible. In a browser they appear after the first touch.
Phones and browsers also get lighter graphics (no shadows, no MSAA): see `LOW_QUALITY` in
`src/touch.rs`.

## Build for the browser (easiest)

```
rustup target add wasm32-unknown-unknown
cargo install trunk
trunk serve --release --address 0.0.0.0
```

Open `http://<your computer's IP>:8080` on the phone (same Wi-Fi). Turn the phone sideways.
`trunk build --release` makes a `dist/` folder you can put on any web host.

## Build for Android

```
rustup target add aarch64-linux-android
cargo install cargo-apk
cargo apk run --lib --release
```

Needs the Android SDK and NDK (`ANDROID_HOME`, `ANDROID_NDK_ROOT`) and a phone with USB debugging.
The package name and orientation are in `[package.metadata.android]` in `Cargo.toml`.

## Build for iOS

Needs a Mac with Xcode. Add `"staticlib"` to `crate-type` under `[lib]` in `Cargo.toml`, then:

```
rustup target add aarch64-apple-ios
cargo build --lib --release --target aarch64-apple-ios
```

Link the resulting library from the Xcode project in Bevy's `examples/mobile` (copy that folder and
point its build script at this crate).

## Things to know

- **Speed.** The game is heavy (physics, lots of small objects). Expect it to be slow on older phones.
  If it is, the biggest wins are fewer trees/plants and a smaller ground.
- **The focus effect on dialogue** uses a custom shader. It may not work in browsers that only have
  WebGL2; if the screen is black in a browser, comment out `.add_plugins(DialogueFocusPlugin)`.
- **Not on phones:** dropping an image or model file onto the window (to make a 3D model) and the
  Blender conversion - they need a desktop.
- The `cdylib` in `[lib]` makes desktop builds link one extra file; remove it if you only target desktop.