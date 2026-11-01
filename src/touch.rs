//! Touch controls, so the game can be played on a phone or tablet.
//!
//! Instead of changing every system that reads the mouse and keyboard, the touch screen
//! *pretends* to be them: a left-hand thumb stick presses W/A/S/D, dragging on the right looks
//! around (the same as holding the right mouse button and moving), a tap clicks, a press-and-hold
//! holds the left mouse button (to grab, drag and swing tools), and on-screen buttons press the
//! other keys. Everything else in the game keeps working exactly as it does with a mouse.
//!
//! | Touch                                   | Same as                                        |
//! |-----------------------------------------|------------------------------------------------|
//! | drag on the left half                   | W / A / S / D (thumb stick)                    |
//! | drag on the right                       | hold the right mouse button + move (look)      |
//! | tap on the right                        | left click                                     |
//! | press, wait a moment, then drag         | hold the left mouse button + move              |
//! | Jump / Kneel / Drop / Use / Clear / ... | Space / Shift (toggle) / Q / E / X / ...       |
//! | Throw                                   | middle mouse button (throw an apple)           |
//! | Poss                                    | F (possess the animal or person you look at)   |
//! | + / -                                   | mouse wheel (push / pull what you hold)        |

use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::input::touch::Touches;
use bevy::input::InputSystem;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use std::collections::HashMap;
use std::sync::Mutex;

/// Phones and browsers get lighter graphics (no shadows, no MSAA).
pub const LOW_QUALITY: bool = cfg!(any(target_os = "android", target_os = "ios", target_arch = "wasm32"));
/// On Android and iOS the touch buttons are always shown; elsewhere only once a finger touches.
pub const ALWAYS_SHOW_HUD: bool = cfg!(any(target_os = "android", target_os = "ios"));

const BTN: f32 = 58.0; // size of an on-screen button (logical pixels)
const GAP: f32 = 8.0;
const MARGIN: f32 = 14.0;
const STICK_RADIUS: f32 = 70.0;
const LOOK_SPEED: f32 = 1.6; // finger pixels -> "mouse" pixels when looking
const DRAG_SPEED: f32 = 1.5; // the same while holding (swinging a tool)
const TAP_DRIFT: f32 = 14.0; // a finger that moves further than this is dragging, not tapping
const HOLD_AFTER: f32 = 0.22; // seconds before a still finger counts as press-and-hold

// ---------------------------------------------------------------------------
// Where the finger is pointing
// ---------------------------------------------------------------------------

/// (position in logical pixels, frames since a finger last used it)
static POINTER: Mutex<Option<(Vec2, u32)>> = Mutex::new(None);

fn set_pointer(pos: Vec2) {
    if let Ok(mut p) = POINTER.lock() {
        *p = Some((pos, 0));
    }
}

/// Where the player is pointing: the finger while one is being used, otherwise the mouse cursor.
/// Use this instead of `window.cursor_position()`, which is always `None` on a touch screen.
pub fn pointer_position(window: &Window) -> Option<Vec2> {
    let touch = POINTER.lock().ok().and_then(|g| (*g).map(|(p, _)| p));
    touch.or_else(|| window.cursor_position())
}

// ---------------------------------------------------------------------------
// The on-screen buttons
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TouchButton {
    Jump,
    Kneel,
    Drop,
    Throw,
    Use,
    Fire,
    Clear,
    Hungry,
    Possess,
    Push,
    Pull,
    Close,
}

impl TouchButton {
    const ALL: [TouchButton; 12] = [
        TouchButton::Jump,
        TouchButton::Kneel,
        TouchButton::Drop,
        TouchButton::Throw,
        TouchButton::Use,
        TouchButton::Fire,
        TouchButton::Clear,
        TouchButton::Hungry,
        TouchButton::Possess,
        TouchButton::Push,
        TouchButton::Pull,
        TouchButton::Close,
    ];

    fn label(self) -> &'static str {
        match self {
            TouchButton::Jump => "Jump",
            TouchButton::Kneel => "Kneel",
            TouchButton::Drop => "Drop",
            TouchButton::Throw => "Throw",
            TouchButton::Use => "Use",
            TouchButton::Fire => "Fire",
            TouchButton::Clear => "Clear",
            TouchButton::Hungry => "Hungry",
            TouchButton::Possess => "Poss",
            TouchButton::Push => "+",
            TouchButton::Pull => "-",
            TouchButton::Close => "X",
        }
    }

    /// (left/top corner, right/bottom corner) in logical pixels for a window of size `win`.
    /// Eight buttons sit in the bottom-right corner; push / pull / close run down the right edge.
    fn rect(self, win: Vec2) -> (Vec2, Vec2) {
        // (column from the right, row from the bottom) in the corner cluster, or a slot on the side
        let (cluster, side): (Option<(f32, f32)>, f32) = match self {
            TouchButton::Jump => (Some((0.0, 0.0)), 0.0),
            TouchButton::Kneel => (Some((1.0, 0.0)), 0.0),
            TouchButton::Drop => (Some((2.0, 0.0)), 0.0),
            TouchButton::Throw => (Some((3.0, 0.0)), 0.0),
            TouchButton::Use => (Some((0.0, 1.0)), 0.0),
            TouchButton::Fire => (Some((1.0, 1.0)), 0.0),
            TouchButton::Clear => (Some((2.0, 1.0)), 0.0),
            TouchButton::Hungry => (Some((3.0, 1.0)), 0.0),
            TouchButton::Possess => (Some((4.0, 0.0)), 0.0),
            TouchButton::Push => (None, 0.0),
            TouchButton::Pull => (None, 1.0),
            TouchButton::Close => (None, 2.0),
        };
        let step = BTN + GAP;
        match cluster {
            Some((col, row)) => {
                let max = Vec2::new(win.x - MARGIN - col * step, win.y - MARGIN - row * step);
                (max - Vec2::splat(BTN), max)
            }
            None => {
                let min = Vec2::new(win.x - MARGIN - BTN, MARGIN + side * step);
                (min, min + Vec2::splat(BTN))
            }
        }
    }

    fn contains(self, win: Vec2, p: Vec2) -> bool {
        let (lo, hi) = self.rect(win);
        p.x >= lo.x && p.x <= hi.x && p.y >= lo.y && p.y <= hi.y
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct StickTouch {
    id: u64,
    origin: Vec2,
    pos: Vec2,
}

#[derive(Clone, Copy, PartialEq)]
enum LookMode {
    /// finger just went down: a tap, a drag (look) or a press-and-hold, we do not know yet
    Undecided,
    /// dragging: looking around
    Look,
    /// press-and-hold: the left mouse button is held
    Hold,
}

#[derive(Clone, Copy)]
struct LookTouch {
    id: u64,
    start: Vec2,
    last: Vec2,
    held: f32,
    mode: LookMode,
}

#[derive(Clone, Copy)]
enum Tap {
    Key(KeyCode),
    Mouse(MouseButton),
}

#[derive(Resource, Default)]
pub struct TouchControls {
    /// the touch HUD is shown
    pub enabled: bool,
    stick: Option<StickTouch>,
    look: Option<LookTouch>,
    /// which finger is holding which button
    buttons: HashMap<u64, TouchButton>,
    kneeling: bool,
    /// W, A, S, D as pressed by the thumb stick
    walk: [bool; 4],
    /// taps waiting to be released (a tap is held for two frames)
    taps: Vec<(Tap, u8)>,
    repeat: f32,
}

fn tap(ctl: &mut TouchControls, keys: &mut ButtonInput<KeyCode>, mouse: &mut ButtonInput<MouseButton>, what: Tap) {
    match what {
        Tap::Key(k) => keys.press(k),
        Tap::Mouse(b) => mouse.press(b),
    }
    ctl.taps.push((what, 2));
}

fn press_button(
    b: TouchButton,
    ctl: &mut TouchControls,
    keys: &mut ButtonInput<KeyCode>,
    mouse: &mut ButtonInput<MouseButton>,
    win: Vec2,
) {
    match b {
        TouchButton::Jump => keys.press(KeyCode::Space),
        TouchButton::Kneel => {
            ctl.kneeling = !ctl.kneeling;
            if ctl.kneeling {
                keys.press(KeyCode::ShiftLeft);
            } else {
                keys.release(KeyCode::ShiftLeft);
            }
        }
        TouchButton::Drop => tap(ctl, keys, mouse, Tap::Key(KeyCode::KeyQ)),
        TouchButton::Use => tap(ctl, keys, mouse, Tap::Key(KeyCode::KeyE)),
        TouchButton::Clear => tap(ctl, keys, mouse, Tap::Key(KeyCode::KeyX)),
        TouchButton::Fire => tap(ctl, keys, mouse, Tap::Key(KeyCode::KeyC)),
        TouchButton::Hungry => tap(ctl, keys, mouse, Tap::Key(KeyCode::KeyH)),
        TouchButton::Possess => tap(ctl, keys, mouse, Tap::Key(KeyCode::KeyF)),
        TouchButton::Close => tap(ctl, keys, mouse, Tap::Key(KeyCode::Escape)),
        TouchButton::Throw => {
            // an apple is thrown where the "cursor" points: the middle of the screen
            set_pointer(win * 0.5);
            tap(ctl, keys, mouse, Tap::Mouse(MouseButton::Middle));
        }
        TouchButton::Push | TouchButton::Pull => ctl.repeat = 0.0, // handled while held
    }
}

fn release_button(b: TouchButton, keys: &mut ButtonInput<KeyCode>) {
    if b == TouchButton::Jump {
        keys.release(KeyCode::Space);
    }
}

// ---------------------------------------------------------------------------
// The system that turns touches into mouse and keyboard input
// ---------------------------------------------------------------------------

pub fn touch_input_system(
    time: Res<Time>,
    touches: Res<Touches>,
    windows: Query<(Entity, &Window), With<PrimaryWindow>>,
    mut ctl: ResMut<TouchControls>,
    mut keys: ResMut<ButtonInput<KeyCode>>,
    mut mouse: ResMut<ButtonInput<MouseButton>>,
    mut motion: EventWriter<MouseMotion>,
    mut wheel: EventWriter<MouseWheel>,
) {
    let Ok((window_entity, window)) = windows.get_single() else {
        return;
    };
    let win = Vec2::new(window.width(), window.height());
    let dt = time.delta_seconds().min(0.1);

    // the finger position is only remembered for a little while after it lets go
    if let Ok(mut p) = POINTER.lock() {
        if let Some((pos, age)) = *p {
            *p = if age >= 40 { None } else { Some((pos, age + 1)) };
        }
    }
    if touches.iter().next().is_some() {
        ctl.enabled = true;
    }

    // taps from earlier frames are let go of
    for (what, n) in std::mem::take(&mut ctl.taps) {
        if n <= 1 {
            match what {
                Tap::Key(k) => keys.release(k),
                Tap::Mouse(b) => mouse.release(b),
            }
        } else {
            ctl.taps.push((what, n - 1));
        }
    }

    // --- fingers that just went down ---
    for t in touches.iter_just_pressed() {
        let (id, pos) = (t.id(), t.position());
        let hit = TouchButton::ALL.iter().copied().find(|b| b.contains(win, pos));
        if let Some(b) = hit {
            ctl.buttons.insert(id, b);
            press_button(b, &mut ctl, &mut keys, &mut mouse, win);
            continue;
        }
        if pos.x < win.x * 0.4 && ctl.stick.is_none() {
            ctl.stick = Some(StickTouch { id, origin: pos, pos });
            continue;
        }
        if ctl.look.is_none() {
            ctl.look = Some(LookTouch {
                id,
                start: pos,
                last: pos,
                held: 0.0,
                mode: LookMode::Undecided,
            });
        }
    }

    // --- the thumb stick: W A S D ---
    if let Some(mut st) = ctl.stick {
        if let Some(t) = touches.get_pressed(st.id) {
            st.pos = t.position();
            ctl.stick = Some(st);
        }
    }
    let mut want = [false; 4]; // W A S D
    if let Some(st) = ctl.stick {
        let v = (st.pos - st.origin) / STICK_RADIUS;
        want = [v.y < -0.3, v.x < -0.3, v.y > 0.3, v.x > 0.3];
    }
    let walk_keys = [KeyCode::KeyW, KeyCode::KeyA, KeyCode::KeyS, KeyCode::KeyD];
    for i in 0..4 {
        if want[i] && !ctl.walk[i] {
            keys.press(walk_keys[i]);
        } else if !want[i] && ctl.walk[i] {
            keys.release(walk_keys[i]);
        }
        ctl.walk[i] = want[i];
    }

    // --- the right-hand finger: look, tap or press-and-hold ---
    if let Some(mut lk) = ctl.look {
        if let Some(t) = touches.get_pressed(lk.id) {
            let pos = t.position();
            let delta = pos - lk.last;
            lk.last = pos;
            lk.held += dt;
            match lk.mode {
                LookMode::Undecided => {
                    if pos.distance(lk.start) > TAP_DRIFT {
                        lk.mode = LookMode::Look;
                        mouse.press(MouseButton::Right); // looking = right mouse button held
                    } else if lk.held > HOLD_AFTER {
                        lk.mode = LookMode::Hold;
                        set_pointer(pos);
                        mouse.press(MouseButton::Left);
                    }
                }
                LookMode::Look => {
                    if delta != Vec2::ZERO {
                        motion.send(MouseMotion { delta: delta * LOOK_SPEED });
                    }
                }
                LookMode::Hold => {
                    set_pointer(pos);
                    if delta != Vec2::ZERO {
                        motion.send(MouseMotion { delta: delta * DRAG_SPEED });
                    }
                }
            }
            ctl.look = Some(lk);
        }
    }

    // --- fingers that just lifted ---
    let mut ended: Vec<(u64, bool)> = Vec::new(); // (finger, was cancelled)
    for t in touches.iter_just_released() {
        ended.push((t.id(), false));
    }
    for t in touches.iter_just_canceled() {
        ended.push((t.id(), true));
    }
    for (id, cancelled) in ended {
        if let Some(b) = ctl.buttons.remove(&id) {
            release_button(b, &mut keys);
        }
        if ctl.stick.map_or(false, |s| s.id == id) {
            ctl.stick = None;
        }
        if let Some(lk) = ctl.look {
            if lk.id == id {
                match lk.mode {
                    LookMode::Look => mouse.release(MouseButton::Right),
                    LookMode::Hold => mouse.release(MouseButton::Left),
                    LookMode::Undecided => {
                        if !cancelled {
                            // a quick tap: a left click where the finger was
                            set_pointer(lk.start);
                            tap(&mut ctl, &mut keys, &mut mouse, Tap::Mouse(MouseButton::Left));
                        }
                    }
                }
                ctl.look = None;
            }
        }
    }

    // --- + and - scroll the mouse wheel while they are held ---
    let dir: f32 = ctl
        .buttons
        .values()
        .map(|b| match b {
            TouchButton::Push => 1.0,
            TouchButton::Pull => -1.0,
            _ => 0.0,
        })
        .sum();
    if dir != 0.0 {
        ctl.repeat -= dt;
        if ctl.repeat <= 0.0 {
            ctl.repeat = 0.08;
            wheel.send(MouseWheel {
                unit: MouseScrollUnit::Line,
                x: 0.0,
                y: dir,
                window: window_entity,
            });
        }
    } else {
        ctl.repeat = 0.0;
    }
}

// ---------------------------------------------------------------------------
// What the touch controls look like
// ---------------------------------------------------------------------------

#[derive(Component)]
struct TouchHudRoot;

#[derive(Component)]
struct TouchButtonNode(TouchButton);

#[derive(Component)]
struct StickBase;

#[derive(Component)]
struct StickKnob;

fn touch_hud_setup(mut commands: Commands) {
    commands
        .spawn((
            NodeBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    width: Val::Percent(100.0),
                    height: Val::Percent(100.0),
                    ..default()
                },
                visibility: Visibility::Hidden,
                z_index: ZIndex::Global(50),
                ..default()
            },
            TouchHudRoot,
        ))
        .with_children(|root| {
            for b in TouchButton::ALL {
                root.spawn((
                    NodeBundle {
                        style: Style {
                            position_type: PositionType::Absolute,
                            width: Val::Px(BTN),
                            height: Val::Px(BTN),
                            justify_content: JustifyContent::Center,
                            align_items: AlignItems::Center,
                            ..default()
                        },
                        background_color: BackgroundColor(Color::srgba(0.1, 0.1, 0.12, 0.45)),
                        border_radius: BorderRadius::all(Val::Px(12.0)),
                        ..default()
                    },
                    TouchButtonNode(b),
                ))
                .with_children(|n| {
                    n.spawn(TextBundle::from_section(
                        b.label(),
                        TextStyle {
                            font_size: 16.0,
                            color: Color::WHITE,
                            ..default()
                        },
                    ));
                });
            }
            root.spawn((
                NodeBundle {
                    style: Style {
                        position_type: PositionType::Absolute,
                        width: Val::Px(STICK_RADIUS * 2.0),
                        height: Val::Px(STICK_RADIUS * 2.0),
                        ..default()
                    },
                    background_color: BackgroundColor(Color::srgba(1.0, 1.0, 1.0, 0.12)),
                    border_radius: BorderRadius::MAX,
                    visibility: Visibility::Hidden,
                    ..default()
                },
                StickBase,
            ));
            root.spawn((
                NodeBundle {
                    style: Style {
                        position_type: PositionType::Absolute,
                        width: Val::Px(50.0),
                        height: Val::Px(50.0),
                        ..default()
                    },
                    background_color: BackgroundColor(Color::srgba(1.0, 1.0, 1.0, 0.35)),
                    border_radius: BorderRadius::MAX,
                    visibility: Visibility::Hidden,
                    ..default()
                },
                StickKnob,
            ));
        });
}

fn touch_hud_system(
    ctl: Res<TouchControls>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut root: Query<&mut Visibility, (With<TouchHudRoot>, Without<StickBase>, Without<StickKnob>)>,
    mut buttons: Query<(&TouchButtonNode, &mut Style, &mut BackgroundColor)>,
    mut base: Query<(&mut Style, &mut Visibility), (With<StickBase>, Without<TouchButtonNode>)>,
    mut knob: Query<
        (&mut Style, &mut Visibility),
        (With<StickKnob>, Without<StickBase>, Without<TouchButtonNode>),
    >,
) {
    let Ok(window) = windows.get_single() else {
        return;
    };
    let win = Vec2::new(window.width(), window.height());
    for mut v in &mut root {
        *v = if ctl.enabled { Visibility::Inherited } else { Visibility::Hidden };
    }
    for (node, mut style, mut bg) in &mut buttons {
        let (lo, _) = node.0.rect(win);
        style.left = Val::Px(lo.x);
        style.top = Val::Px(lo.y);
        let held = ctl.buttons.values().any(|b| *b == node.0)
            || (node.0 == TouchButton::Kneel && ctl.kneeling);
        *bg = BackgroundColor(if held {
            Color::srgba(1.0, 1.0, 1.0, 0.45)
        } else {
            Color::srgba(0.1, 0.1, 0.12, 0.45)
        });
    }
    match ctl.stick {
        Some(st) => {
            let offset = (st.pos - st.origin).clamp_length_max(STICK_RADIUS);
            for (mut style, mut vis) in &mut base {
                style.left = Val::Px(st.origin.x - STICK_RADIUS);
                style.top = Val::Px(st.origin.y - STICK_RADIUS);
                *vis = Visibility::Inherited;
            }
            for (mut style, mut vis) in &mut knob {
                style.left = Val::Px(st.origin.x + offset.x - 25.0);
                style.top = Val::Px(st.origin.y + offset.y - 25.0);
                *vis = Visibility::Inherited;
            }
        }
        None => {
            for (_, mut vis) in &mut base {
                *vis = Visibility::Hidden;
            }
            for (_, mut vis) in &mut knob {
                *vis = Visibility::Hidden;
            }
        }
    }
}

/// Phones and browsers cannot afford shadows: turn them off on every light as it appears.
fn lighter_graphics_system(
    mut points: Query<&mut PointLight, Added<PointLight>>,
    mut suns: Query<&mut DirectionalLight, Added<DirectionalLight>>,
    mut spots: Query<&mut SpotLight, Added<SpotLight>>,
) {
    for mut l in &mut points {
        l.shadows_enabled = false;
    }
    for mut l in &mut suns {
        l.shadows_enabled = false;
    }
    for mut l in &mut spots {
        l.shadows_enabled = false;
    }
}

pub struct TouchControlsPlugin;

impl Plugin for TouchControlsPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(TouchControls {
            enabled: ALWAYS_SHOW_HUD,
            ..default()
        })
        .add_systems(Startup, touch_hud_setup)
        .add_systems(PreUpdate, touch_input_system.after(InputSystem))
        .add_systems(Update, touch_hud_system);
        if LOW_QUALITY {
            app.insert_resource(Msaa::Off)
                .add_systems(Update, lighter_graphics_system);
        }
    }
}