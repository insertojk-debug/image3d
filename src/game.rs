//! Image -> 3D physics object inside a house, with a "hand" you can grab it with.
//!
//! Run:  cargo run --release
//! Then DRAG & DROP onto the window:
//!   * a Blender export  (.glb / .gltf)  -> real 3D model with physics
//!   * an .fbx file (converted on the fly with Blender or FBX2glTF, see convert_fbx)
//!   * any image file    (png, jpg, ...) -> image turned into a 3D relief
//! (Optionally: cargo run --release -- path/to/image.png to start with one.)
//!
//! A two-story wooden house with a river, changing day and weather, roaming pets,
//! and one person (Tom) you can talk to, who gets hungry and feeds himself.
//!
//! Controls
//!   Drop a file       : it becomes a 3D physics object in front of you
//!   X                 : remove all dropped objects
//!   Left mouse (hold)  : grab a physics object at the exact point you click (cursor becomes a
//!                        hand). Only that point is pulled; the rest swings, tilts and trails
//!                        behind it under real physics
//!   Left mouse (click) : click a person to talk; click again to hear their next line
//!   Esc                : close the conversation
//!   Sword (on the table): grab its handle to take it; hold the left mouse button and
//!                         move the mouse to swing it; Q drops it
//!   Axe (on the table) : grab its handle; hold the left mouse button and swing at
//!                         a tree to chop it; Q drops it. The trunk is built from layers of
//!                         wood blocks: the first swing makes a small straight cut, and
//!                         swinging at the same spot again cuts deeper, layer by layer. Wood
//!                         that is cut off from the rest falls away as chunks, and once enough
//!                         is gone the tree goes over. A felled trunk and its branches are
//!                         chopped the very same way: every swing chips the outer layer of
//!                         wood off the spot you hit, repeated swings go deeper layer by
//!                         layer, and once too little wood is left there it snaps: the part
//!                         beyond the cut falls off as its own physics object, the part
//!                         nearer the trunk stays attached to the felled trunk
//!   Shovel (on the table): grab its handle; hold LMB to dig, aim at the ground, and
//!                          swing the mouse; Q drops it
//!   Spear (on the table) : grab its handle; click while aiming to thrust; Q drops it.
//!                          Click a fish in the river (within reach) to stab it: the fish stays
//!                          skewered on the spear. Carry it to a burning fire and click the fire
//!                          to roast the fish over the flames until it is cooked
//!   Knife (on the table) : grab its handle; hold the left mouse button and flick the mouse to
//!                          slash grass and plants; click a wood chip (a piece knocked off a
//!                          tree or log) to cut a V notch into the middle of it; Q drops it
//!   Bucket (on the table): grab it by the handle, rim or side, dip it in the river and
//!                          tilt it to scoop water; tilt it back over to pour it out
//!                        Mud you let go of above the bucket stays in it; tip the bucket and the
//!                        mud slops out in globs that splat where they land
//!                        The bail (wire handle) swings on its ears as you move or tip the bucket;
//!                        grab the bail itself and the bucket hangs from it
//!                        Water washes mud away: splats in the river, rain or poured water
//!                        dissolve, a glob dunked in the river melts, and water in the bucket
//!                        washes its mud out (and turns the water brown)
//!   Sticks and fire    : double-click a stick on the ground to hold it. Hold C to kneel, click
//!                        another stick lying beside you and keep the left button down (move the
//!                        mouse to go faster): your stick rubs against it until it smokes and a
//!                        fire starts right where that stick lay. Drop sticks or branches into the
//!                        fire to make it bigger; rain and water weaken it. Q puts the stick down
//!   Kneel              : hold C (slow, no jumping)
//!   Your body          : you can see your hands (they reach for and grip what you hold, and hold
//!                        your tools) and your legs and boots when you look down
//!   Leaf mantises      : look like leaves; they watch you and rear up with raised front legs if
//!                        you come close
//!   Grasshoppers       : a few live in the meadow; walk up to one and it hops away
//!   River water        : hold the left mouse button on the river to pull out a glob of water.
//!                        It drips as you carry it; let go over a bucket to fill it, or anywhere
//!                        else to splash it (which washes mud off)
//!   Mouse wheel        : push / pull the grabbed object
//!   River mud          : the banks (and the riverbed) are muddy. Hold the left mouse button on
//!                        the mud to pull out a glob of it. It drips while you carry it. Let go
//!                        and it splats wherever it lands (ground, walls, furniture, things you
//!                        dropped in). Click a splat you made to pick it up and move it again
//!   Right mouse (hold) : look around
//!   W A S D            : walk      Left Shift : run      Space : jump
//!   Release the mouse while moving to THROW the object.
//!   Possession: look straight at an animal or a person and press F to become it (WASD moves it,
//!                          Space / Shift fly a bird up / down); F again returns to your own body.
//!   Paint brush + paint cans (on the floor by the table): click the brush to take it, click a
//!                          paint can to dip it, then click (or drag) on anything to paint that
//!                          spot with the can's colour. Dip again when it runs dry. Q drops it
//!   Apples lie around the yard: hold one with the left button (or just point at it) and
//!   press the MIDDLE mouse button to throw it where you are aiming.

use bevy::core_pipeline::{
    core_3d::graph::{Core3d, Node3d},
    fullscreen_vertex_shader::fullscreen_shader_vertex_state,
};
use bevy::ecs::query::QueryItem;
use bevy::ecs::system::SystemParam;
use bevy::gltf::GltfAssetLabel;
use bevy::hierarchy::HierarchyQueryExt;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;
use bevy::render::{
    extract_component::{
        ComponentUniforms, DynamicUniformIndex, ExtractComponent, ExtractComponentPlugin,
        UniformComponentPlugin,
    },
    render_graph::{
        NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel, ViewNode, ViewNodeRunner,
    },
    render_resource::{
        binding_types::{sampler, texture_2d, uniform_buffer},
        *,
    },
    renderer::{RenderContext, RenderDevice},
    texture::BevyDefault,
    view::ViewTarget,
    RenderApp,
};
use bevy::render::{mesh::PrimitiveTopology, render_asset::RenderAssetUsages};
use bevy::scene::{SceneInstance, SceneSpawner};
use bevy::window::{CursorIcon, FileDragAndDrop, PrimaryWindow};
use bevy_rapier3d::prelude::*;
use image::{imageops::FilterType, Rgba, RgbaImage};
use crate::nav::*;
use crate::touch::{pointer_position, TouchControlsPlugin, ALWAYS_SHOW_HUD};
use crate::world::*;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Mutex};

const MAX_CELLS: u32 = 56; // image is downsampled so its longest side has this many columns
const OBJECT_SIZE: f32 = 1.2; // world size (meters) of the object's longest side
const OBJECT_DEPTH: f32 = 0.25; // extra extrusion height from brightness
const OBJECT_BASE: f32 = 0.05; // minimum thickness
const MODEL_SIZE: f32 = 1.5; // dropped 3D models are scaled so their longest side is this big (m)

// House dimensions (meters)
const FOOT: f32 = 8.4; // full footprint, outer wall to outer wall
const WALL_T: f32 = 0.2;
const STORY_H: f32 = 3.0;
const SLAB_T: f32 = 0.2;
const F1_FLOOR_Y: f32 = 0.0; // floor 1 walking height
const F1_WALL_CY: f32 = F1_FLOOR_Y + STORY_H / 2.0;
const F2_FLOOR_Y: f32 = F1_FLOOR_Y + STORY_H + SLAB_T; // floor 2 walking height
const F2_WALL_CY: f32 = F2_FLOOR_Y + STORY_H / 2.0;
const ROOF_BASE_Y: f32 = F2_FLOOR_Y + STORY_H; // top of 2nd-story walls, eave height
const DOOR_W: f32 = 1.2;
const DOOR_H: f32 = 2.2;
const NPC_CENTER_Y: f32 = 0.57; // capsule center above the feet (0.7/2 + 0.22)
const GROUND_HALF_SIZE: f32 = 40.0;
const GROUND_CELL_SIZE: f32 = 0.5;
const DIG_MAX_DEPTH: f32 = 0.55;
const RIVER_DEPTH: f32 = 0.32;
const RIVER_WATER_Y: f32 = GROUND_SURFACE_Y - 0.025;
const RIVER_FLOOD_MAX: f32 = 1.25;
const RIVER_FLOOD_HEIGHT: f32 = 0.18;
const MUD_BAND: f32 = 2.4; // how far the mud reaches past the water's edge (m)
const MUD_BLOB_RADIUS: f32 = 0.14; // a full glob of mud
const MUD_GRAB_REACH: f32 = 8.0; // how far away you can pull mud out of the bank (m)
const STICK_RADIUS: f32 = 0.036;
const STICK_HALF: f32 = 0.23; // half the straight part of a stick
const STICK_LENGTH: f32 = STICK_HALF * 2.0 + STICK_RADIUS * 2.0;
const FIRE_MAX_FUEL: f32 = 4.0;
const BUCKET_WATER_PER_BLOB: f32 = 0.5; // a full glob of water fills this much of the bucket
const BUCKET_MUD_PER_BLOB: f32 = 0.55; // a full glob of mud fills this much of the bucket
const BUCKET_MUD_CHUNK: f32 = 0.12; // mud leaves a tipped bucket in globs of this much
const BUCKET_MUD_MASS_FACTOR: f32 = 1.6; // mud is heavier than the same amount of water
const MUD_MAX_PATCHES: usize = 800; // oldest splats disappear past this many
const DAY_PHASE_SECONDS: f32 = 3_600.0;
const WEATHER_PHASE_SECONDS: f32 = 3_600.0;

// Player
const GRAVITY: f32 = 20.0;
const JUMP_V: f32 = 6.3; // v^2 / 2g = ~1 m jump
const WALK_SPEED: f32 = 3.5;
const RUN_SPEED: f32 = 6.0;
const EYE_OFFSET: f32 = 0.7; // camera above the capsule center (1.6 m eye height)
const PLAYER_SPAWN: Vec3 = Vec3::new(0.0, 0.92, 3.0);

// Sword "ready" stance, in camera space: blade tilted up and a bit forward/right
const SWORD_REST_PSI: f32 = 0.35; // sweep angle to the right of straight ahead (rad)
const SWORD_REST_PHI: f32 = 1.0; // elevation above horizontal (rad)

// The "hand": a critically-damped spring pulling on the exact point you grabbed.
const HAND_STIFFNESS: f32 = 140.0; // 1/s^2, per unit mass
const HAND_DAMPING: f32 = 20.0; // 1/s, per unit mass
const HAND_MAX_ACCEL: f32 = 90.0; // m/s^2 the hand can add on top of holding the weight up
const HAND_MAX_REACH_ERROR: f32 = 3.0; // m, caps how hard a far-away cursor can yank
const HELD_LINEAR_DAMPING: f32 = 0.3;
const HELD_ANGULAR_DAMPING: f32 = 2.0;
const APPLE_RADIUS: f32 = 0.045; // m
const APPLE_THROW_SPEED: f32 = 15.0; // m/s when thrown with the middle mouse button
const APPLE_REACH: f32 = 5.0; // m, how far away a pointed-at apple can still be thrown
const WORLD_GRAVITY: f32 = 9.81; // Rapier's default gravity, used to hold the weight up

// Bucket (a small pail). Origin = middle of the underside of the base, +Y = up.
const BUCKET_SCALE: f32 = 0.4;
const BUCKET_HEIGHT: f32 = 0.74 * BUCKET_SCALE; // underside of base -> rim
const BUCKET_BASE_T: f32 = 0.06 * BUCKET_SCALE;
const BUCKET_BOTTOM_R: f32 = 0.28 * BUCKET_SCALE; // outer radius at the base
const BUCKET_TOP_R: f32 = 0.37 * BUCKET_SCALE; // outer radius at the rim
const BUCKET_WALL_T: f32 = 0.05 * BUCKET_SCALE;
const BUCKET_EMPTY_MASS: f32 = 0.8; // kg added to the (very light) collider mass
const BUCKET_WATER_MASS: f32 = 2.4; // kg of water in a full bucket
const BUCKET_WATER_TOLERANCE: f32 = 0.025; // ripples / meniscus: water reaches a bit above its level

// The colour of a freshly cut trunk / branch face (base for the ring pattern).
const INNER_WOOD_COLOR: Color = Color::srgb(0.68, 0.45, 0.22);

/// Marks every physics object the user dropped in (image relief or 3D model).
#[derive(Component)]
struct ImageObject;

/// A dropped 3D model that is still loading; holds seconds waited so far.
#[derive(Component)]
struct PendingModel(f32);

#[derive(Component)]
struct StatusText;

#[derive(Component)]
struct DialogueBox;

/// The person in the house (Tom). Click him to talk; click again for the next line.
#[derive(Component)]
struct Npc {
    name: String,
    lines: Vec<String>,
    index: usize,
}

/// Who you're currently talking to, if anyone.
#[derive(Resource, Default)]
struct DialogueState {
    entity: Option<Entity>,
    focus_active: bool,
}

#[derive(Component, Default, Clone, Copy, ExtractComponent, ShaderType)]
struct DialogueFocusSettings {
    amount: f32,
    blur_amount: f32,
    center: Vec2,
    radius: f32,
    aspect: f32,
    magnification: f32,
}

const DIALOGUE_FOCUS_SHADER_ID: u128 = 0x6d8f2bc1_471e_4a62_9c03_8af1702e5d91;

const DIALOGUE_FOCUS_SHADER: &str = r#"
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;

struct DialogueFocusSettings {
    amount: f32,
    blur_amount: f32,
    center: vec2<f32>,
    radius: f32,
    aspect: f32,
    magnification: f32,
#ifdef SIXTEEN_BYTE_ALIGNMENT
    _webgl2_padding: vec2<f32>,
#endif
}

@group(0) @binding(2) var<uniform> settings: DialogueFocusSettings;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let focus_delta = (in.uv - settings.center) * vec2<f32>(settings.aspect, 1.0);
    let focus_distance = length(focus_delta);
    let original = textureSample(screen_texture, texture_sampler, in.uv);
    let strength = settings.amount;
    let lens = (1.0 - smoothstep(settings.radius * 0.82, settings.radius, focus_distance))
        * strength;
    let zoom_uv = settings.center + (in.uv - settings.center) / settings.magnification;
    let magnified = textureSample(screen_texture, texture_sampler, zoom_uv);
    let focused = mix(original, magnified, lens);
    let outside_focus = smoothstep(settings.radius + 0.025, settings.radius + 0.18, focus_distance);
    let blur = outside_focus * settings.blur_amount;
    if blur < 0.001 {
        return focused;
    }

    let spread_y = 0.002 + blur * 0.006;
    let spread_x = spread_y / max(settings.aspect, 0.5);
    let blurred = (
        textureSample(screen_texture, texture_sampler, in.uv + vec2<f32>(-spread_x, -spread_y))
        + textureSample(screen_texture, texture_sampler, in.uv + vec2<f32>(0.0, -spread_y))
        + textureSample(screen_texture, texture_sampler, in.uv + vec2<f32>(spread_x, -spread_y))
        + textureSample(screen_texture, texture_sampler, in.uv + vec2<f32>(-spread_x, 0.0))
        + textureSample(screen_texture, texture_sampler, in.uv + vec2<f32>(spread_x, 0.0))
        + textureSample(screen_texture, texture_sampler, in.uv + vec2<f32>(-spread_x, spread_y))
        + textureSample(screen_texture, texture_sampler, in.uv + vec2<f32>(0.0, spread_y))
        + textureSample(screen_texture, texture_sampler, in.uv + vec2<f32>(spread_x, spread_y))
    ) / 8.0;
    return mix(focused, blurred, blur);
}
"#;

struct DialogueFocusPlugin;

impl Plugin for DialogueFocusPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractComponentPlugin::<DialogueFocusSettings>::default(),
            UniformComponentPlugin::<DialogueFocusSettings>::default(),
        ));

        app.world_mut().resource_mut::<Assets<Shader>>().insert(
            Handle::<Shader>::weak_from_u128(DIALOGUE_FOCUS_SHADER_ID).id(),
            Shader::from_wgsl(DIALOGUE_FOCUS_SHADER, "dialogue_focus.wgsl"),
        );

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_graph_node::<ViewNodeRunner<DialogueFocusNode>>(
                    Core3d,
                    DialogueFocusLabel,
                )
                .add_render_graph_edges(
                    Core3d,
                    (
                        Node3d::Tonemapping,
                        DialogueFocusLabel,
                        Node3d::EndMainPassPostProcessing,
                    ),
                );
        }
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<DialogueFocusPipeline>();
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct DialogueFocusLabel;

#[derive(Default)]
struct DialogueFocusNode;

impl ViewNode for DialogueFocusNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static DialogueFocusSettings,
        &'static DynamicUniformIndex<DialogueFocusSettings>,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_target, _, settings_index): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let focus_pipeline = world.resource::<DialogueFocusPipeline>();
        let Some(pipeline) = world
            .resource::<PipelineCache>()
            .get_render_pipeline(focus_pipeline.pipeline_id)
        else {
            return Ok(());
        };
        let Some(settings_binding) = world
            .resource::<ComponentUniforms<DialogueFocusSettings>>()
            .uniforms()
            .binding()
        else {
            return Ok(());
        };

        let post_process = view_target.post_process_write();
        let bind_group = render_context.render_device().create_bind_group(
            "dialogue_focus_bind_group",
            &focus_pipeline.layout,
            &BindGroupEntries::sequential((
                post_process.source,
                &focus_pipeline.sampler,
                settings_binding,
            )),
        );
        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("dialogue_focus_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: post_process.destination,
                resolve_target: None,
                ops: Operations::default(),
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[settings_index.index()]);
        render_pass.draw(0..3, 0..1);
        Ok(())
    }
}

#[derive(Resource)]
struct DialogueFocusPipeline {
    layout: BindGroupLayout,
    sampler: Sampler,
    pipeline_id: CachedRenderPipelineId,
}

impl FromWorld for DialogueFocusPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layout = render_device.create_bind_group_layout(
            "dialogue_focus_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    sampler(SamplerBindingType::Filtering),
                    uniform_buffer::<DialogueFocusSettings>(true),
                ),
            ),
        );
        let sampler = render_device.create_sampler(&SamplerDescriptor::default());
        let pipeline_id =
            world
                .resource_mut::<PipelineCache>()
                .queue_render_pipeline(RenderPipelineDescriptor {
                    label: Some("dialogue_focus_pipeline".into()),
                    layout: vec![layout.clone()],
                    vertex: fullscreen_shader_vertex_state(),
                    fragment: Some(FragmentState {
                        shader: Handle::<Shader>::weak_from_u128(DIALOGUE_FOCUS_SHADER_ID),
                        shader_defs: vec![],
                        entry_point: "fragment".into(),
                        targets: vec![Some(ColorTargetState {
                            format: TextureFormat::bevy_default(),
                            blend: None,
                            write_mask: ColorWrites::ALL,
                        })],
                    }),
                    primitive: PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: MultisampleState::default(),
                    push_constant_ranges: vec![],
                });
        Self {
            layout,
            sampler,
            pipeline_id,
        }
    }
}

/// Set for one frame when a click was spent talking to an NPC, so grab_system
/// doesn't also try to grab whatever is behind them.
#[derive(Resource, Default)]
struct ClickConsumed(bool);

/// Marks the outdoor rain streaks.
#[derive(Component)]
struct Raindrop {
    speed: f32,
}

#[derive(Component)]
struct Puddle {
    remaining: f32,
    lifetime: f32,
    size: f32,
}

/// Marks the directional light used for lightning flashes.
#[derive(Component)]
struct Lightning;

#[derive(Component)]
struct Daylight;

#[derive(Component)]
struct Bird {
    phase: f32,
    radius: f32,
    altitude: f32,
    speed: f32,
}

#[derive(Component)]
struct BirdWing {
    side: f32,
    phase: f32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HopState {
    Resting,
    Crouching,
    Hopping,
}

/// The entities of a grasshopper's animated parts.
#[derive(Clone, Copy)]
struct GrasshopperParts {
    hips: [Entity; 2],
    knees: [Entity; 2],
    antennae: [Entity; 2],
    wings: [Entity; 2],
}

/// A grasshopper that sits in the grass, hops about, and hops away when you get close.
#[derive(Component)]
struct Grasshopper {
    state: HopState,
    timer: f32,   // time left in Resting / Crouching
    crouch: f32,  // how long this crouch lasts
    t: f32,       // 0..1 progress through the current hop
    duration: f32,
    height: f32,
    start: Vec3,
    target: Vec3,
    pos: Vec3,    // where it sits on the ground
    yaw: f32,
    home: Vec3,
    seed: u32,
    phase: f32,
    parts: GrasshopperParts,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MantisState {
    Hold, // sits still, swaying like a leaf in the wind
    Walk,
    Rear, // rears up with its front legs raised
}

#[derive(Clone, Copy)]
struct MantisParts {
    head: Entity,
    shoulders: [Entity; 2],
    elbows: [Entity; 2],
    legs: [Entity; 4], // middle left/right, hind left/right
    wings: [Entity; 2],
}

/// A leaf mantis: sways in the grass like a leaf, watches you, and rears up with its
/// front legs if you come close. It snaps at grasshoppers that hop too near.
#[derive(Component)]
struct LeafMantis {
    state: MantisState,
    timer: f32,
    pos: Vec3,
    yaw: f32,
    target: Vec3,
    home: Vec3,
    seed: u32,
    phase: f32,
    strike: f32,   // < 0: not striking, else 0..1 progress of the strike
    cooldown: f32, // until it may strike again
    head_yaw: f32,
    rear: f32,     // 0..1 blend toward the rearing pose
    parts: MantisParts,
}

/// One of your two arms in view (child of the camera): sleeve, wrist, palm and fingers.
#[derive(Component)]
struct FpArm {
    side: usize, // 0 = left, 1 = right
}

/// A finger / thumb joint of a viewmodel hand, curled by fp_body_system.
#[derive(Component)]
struct FpFinger {
    side: usize,
    level: u8, // which joint along the finger
    thumb: bool,
    bias: f32, // slight per-finger difference in curl
}

/// Your legs and torso, which you see when you look down.
#[derive(Component)]
struct FpLegsRoot;

#[derive(Component)]
struct FpLegPart {
    side: usize,
    kind: u8, // 0 hip, 1 knee, 2 ankle
}

#[derive(Default)]
struct FpState {
    phase: f32,
    last: Option<Vec3>,
    speed: f32,
    grips: [f32; 2],
}

/// A dry stick lying in the grass. Double-click to hold it, then click another stick and
/// hold the button to rub them together until they catch fire.
#[derive(Component)]
struct Stick {
    equipped: bool,
}

/// How far down you are kneeling (0 standing .. 1 kneeling). Hold C.
#[derive(Resource, Default)]
struct Kneel {
    amount: f32,
}

#[derive(Resource, Default)]
struct StickState {
    last_click: Option<(Entity, f32)>, // for double-click detection
    rubbing: Option<Entity>,           // the stick held in the other hand while rubbing
    rubbing_block: Option<Entity>,
    rubbing_local: Vec3,
    burned_pile: Option<(Entity, Entity)>,   // <-- new
    heat: f32,                         // 0..1, a fire starts at 1
    phase: f32,
    smoke_timer: f32,
    msg_timer: f32,
    seed: u32,
    /// world positions of the two grips (left, right) for the hands
    grips: [Option<Vec3>; 2],
}

/// Tracks consecutive clicks on a fish skewered by the spear.
#[derive(Resource, Default)]
struct FishClickState {
    last_click: Option<(Entity, f32)>,
    /// Fish detached from the spear and currently being dragged with the mouse.
    holding: Option<(Entity, f32)>, // entity and distance from the camera ray
}

#[derive(SystemParam)]
struct ToolsHeld<'w, 's> {
    swords: Query<'w, 's, &'static Sword>,
    axes: Query<'w, 's, &'static Axe>,
    shovels: Query<'w, 's, &'static Shovel>,
    spears: Query<'w, 's, &'static Spear>,
    machetes: Query<'w, 's, &'static Machete>,
    knives: Query<'w, 's, &'static Knife>,
    brushes: Query<'w, 's, &'static Brush>,
}

#[derive(SystemParam)]
struct ToolUis<'w> {
    sword: Res<'w, SwordUi>,
    axe: Res<'w, AxeUi>,
    shovel: Res<'w, ShovelUi>,
    spear: Res<'w, SpearUi>,
    machete: Res<'w, MacheteUi>,
    knife: Res<'w, KnifeUi>,
    brush: Res<'w, BrushUi>,
}

impl ToolsHeld<'_, '_> {
    fn any(&self) -> bool {
        self.swords.iter().any(|t| t.equipped)
            || self.axes.iter().any(|t| t.equipped)
            || self.shovels.iter().any(|t| t.equipped)
            || self.spears.iter().any(|t| t.equipped)
            || self.machetes.iter().any(|t| t.equipped)
            || self.knives.iter().any(|t| t.equipped)
            || self.brushes.iter().any(|t| t.equipped)
    }
}

#[derive(Clone, Copy)]
struct Tongue {
    entity: Entity,
    angle: f32,
    phase: f32,
    speed: f32,
    layer: usize, // 0 outer orange, 1 yellow, 2 pale core
}

/// A campfire. Fuel is what is left to burn; flames grow with it. Wood that lands in the
/// fire is burnt for more fuel, water and rain take it away.
#[derive(Component)]
struct Fire {
    fuel: f32,
    size: f32,
    flare: f32, // brief extra brightness when wood is added
    age: f32,
    smoke_timer: f32,
    spark_timer: f32,
    seed: u32,
    light: Entity,
    glow: Entity,
    tongues: Vec<Tongue>,
}

/// Smoke puffs and sparks.
#[derive(Component)]
struct FireParticle {
    velocity: Vec3,
    life: f32,
    max_life: f32,
    size0: f32,
    grow: f32,
    buoyancy: f32,
}

/// A little pile of charred wood dust sitting in the V notch. It grows while you rub,
/// and it smokes as long as it exists.
#[derive(Component)]
struct BurnedDustPile {
    growth: f32,       // 0..1, tracks how far the rub got
    smoke_timer: f32,
    seed: u32,
    /// seconds of smoke left; refilled while you rub it or while bark lies near it
    smoke_left: f32,
    /// when it was made (for removing the oldest when there are too many)
    born: f32,
}

const PILE_SMOKE_SECONDS: f32 = 25.0; // how long a pile keeps smoking with no bark near it
const PILE_BARK_RADIUS: f32 = 2.5; // bark this close keeps the pile smoking
const PILE_REUSE_RADIUS: f32 = 0.12; // rubbing within this of a pile adds to it, else a new pile
const MAX_BURNED_PILES: usize = 10; // the oldest pile goes away beyond this

#[derive(Resource)]
struct FireAssets {
    ball: Handle<Mesh>,
    cylinder: Handle<Mesh>,
    outer: Handle<StandardMaterial>,
    mid: Handle<StandardMaterial>,
    core: Handle<StandardMaterial>,
    ember: Handle<StandardMaterial>,
    smoke: Handle<StandardMaterial>,
    spark: Handle<StandardMaterial>,
    charred: Handle<StandardMaterial>,
    ash: Handle<StandardMaterial>,
    dust: Handle<StandardMaterial>,
    burned_dust: Handle<StandardMaterial>,
}

#[derive(Resource)]
struct Climate {
    day_phase: f32,
    storm: bool,
    weather_timer: f32,
    storm_remaining: f32,
}

#[derive(Component)]
struct RiverWater;

/// A glob of mud you are carrying (or that is flying through the air). It drips and
/// shrinks while it exists; when it hits something after you let go it becomes a MudPatch.
#[derive(Component)]
struct MudBlob {
    amount: f32, // 1.0 = a full glob, drips take it down
    drip_timer: f32,
    free_time: f32, // seconds since it was let go
    seed: u32,
    no_absorb: f32, // seconds left during which it can't fall back into a bucket / stick to it
}

/// The mud pile visible inside a bucket.
#[derive(Component)]
struct BucketMudPart;

/// A falling drop of mud.
#[derive(Component)]
struct MudDrip {
    velocity: Vec3,
    size: f32,
    water: bool, // a drop of clean water instead of mud
}

/// A glob of river water you are carrying (or that is flying through the air). It drips and
/// shrinks; let go and it splashes where it lands, or fills a bucket if it drops in.
#[derive(Component)]
struct WaterBlob {
    amount: f32,
    drip_timer: f32,
    free_time: f32,
    seed: u32,
}

#[derive(Resource)]
struct WaterAssets {
    blob: Handle<Mesh>,
    drop: Handle<Mesh>, // unit sphere, scaled per use
    material: Handle<StandardMaterial>,
}

/// Mud stuck to a surface. Bigger ones can be picked up again.
#[derive(Component)]
struct MudPatch {
    amount: f32,
    radius: f32,
    base_scale: Vec3, // the patch's own scale when fresh; water shrinks it away
    wash: f32,        // 0 = fresh, 1 = completely washed away
}

/// A little brown cloud of mud drifting off in water.
#[derive(Component)]
struct MudPuff {
    velocity: Vec3,
    life: f32,
}

#[derive(Resource)]
struct MudAssets {
    blob: Handle<Mesh>,
    drip: Handle<Mesh>,
    splat: Handle<Mesh>,
    cylinder: Handle<Mesh>,
    material: Handle<StandardMaterial>,
}

#[derive(Resource, Default)]
struct MudState {
    patches: VecDeque<Entity>,
    seed: u32,
}

#[derive(Component)]
struct Fish {
    phase: f32,
    speed: f32,
    offset: f32,
    impaled: bool,
    /// True once pulled out of the river; dropped fish use physics instead of swimming.
    dropped: bool,
    /// 0 = raw, 1 = fully cooked
    cooked: f32,
    body_mat: Handle<StandardMaterial>,
    fin_mat: Handle<StandardMaterial>,
    raw_body: [f32; 3],
    raw_fin: [f32; 3],
}

const SPEAR_FISH_REACH: f32 = 3.6; // how far you can reach out to stab a fish (m)
const SPEAR_FISH_Y: f32 = 1.3; // where along the shaft a stabbed fish sits (spear-local)
const FISH_COOK_SECONDS: f32 = 4.0; // time held over a fire to cook a raw fish
const FIRE_COOK_REACH: f32 = 3.2; // how far from the fire you can roast a fish (m)
const COOKED_BODY: [f32; 3] = [0.46, 0.27, 0.12];
const COOKED_FIN: [f32; 3] = [0.30, 0.17, 0.08];

#[derive(Resource, Default)]
struct RiverState {
    water_level: f32,
    overflow: f32,
}

#[derive(Resource, Default)]
struct RiverContact {
    was_in_water: bool,
    splash_cooldown: f32,
    last_position: Option<Vec3>,
}

#[derive(Component)]
struct SplashParticle {
    velocity: Vec3,
    lifetime: f32,
}

impl Default for Climate {
    fn default() -> Self {
        Self {
            day_phase: 0.32,
            storm: false,
            weather_timer: WEATHER_PHASE_SECONDS,
            storm_remaining: 0.0,
        }
    }
}

/// The front door (on the +X wall). `open` is the target state, `angle` animates.
#[derive(Resource, Default)]
struct Door {
    open: bool,
    angle: f32,
}

#[derive(Component)]
struct DoorPivot;

/// You: a capsule with real collision. The camera follows its eyes.
#[derive(Component)]
struct Player {
    yaw: f32,
    pitch: f32,
    vy: f32,      // vertical velocity (m/s)
    last_dy: f32, // vertical move we asked for last frame (to detect bumping your head)
    mounted: Option<Entity>,
    climbing_tree: Option<Entity>,
}

#[derive(Component)]
struct Horse {
    home: Vec3,
    target: Vec3,
    pause: f32,
    seed: u32,
}

#[derive(Component)]
struct HorseBack;

/// A throwable apple (a physics body like a pebble; middle mouse button throws it).
#[derive(Component)]
struct Apple;

#[derive(Component)]
struct Pebble {
    last_position: Vec3,
    radius: f32,
    thrown_velocity: Vec3,
}

#[derive(Component)]
struct PullablePlant {
    anchor: Vec3,
    uprooted: bool,
}

/// A pail you can scoop river water with. `fill` is 0..=1 of the interior.
#[derive(Component, Default)]
struct Bucket {
    fill: f32,
    in_water: bool,
    state: u8,       // 0 empty, 1 some water, 2 full (only used for status messages)
    drip_debt: f32,  // fractional water drops still owed while pouring
    drip_count: u32, // seeds the drop randomness
    mass_fill: f32,  // load (water + mud) the body's mass was last updated for
    mud: f32,        // 0..=1 of the interior filled with mud
    mud_debt: f32,   // mud that has left the mud pile but is not yet a glob
    mud_state: u8,   // 0 empty, 1 some mud, 2 full (only used for status messages)
    mud_visual: Option<(Entity, Entity)>, // the mud pile inside: body and domed top
    muddy: f32,       // 0..=1: how much mud the water inside has washed out of the pile
    muddy_shown: f32, // the value the water's colour was last set for
}

#[derive(Component)]
struct BucketWater;

/// The bucket's bail (wire handle). It hangs on the two ears at the rim and swings about the
/// line through them like a pendulum, driven by gravity and by how the bucket is moved.
#[derive(Component, Default)]
struct BailPivot {
    angle: f32, // 0 = standing straight up; +/- = leaning toward the bucket's +Z / -Z side
    vel: f32,
    last_linvel: Vec3,
}

#[derive(Component)]
struct TreeBranch {
    tree: Entity,
    anchor: Vec3,
    broken: bool,
    chops_remaining: u8,
}

/// The sword. Local axes: +Y runs up the blade (toward the tip), the blade's
/// flat faces point along +/-X and its cutting edges along +/-Z.
#[derive(Component)]
struct Sword {
    equipped: bool,
    psi: f32,     // sweep: how far the blade points right of straight ahead
    phi: f32,     // how high the blade points above horizontal
    psi_vel: f32, // smoothed sweep speed, used to turn the edge into the cut
    roll: f32,    // wrist roll
}

/// The grip. Grab this part to pick the sword up.
#[derive(Component)]
struct SwordHandle;

#[derive(Resource, Default)]
struct SwordUi {
    hover: bool,
    swinging: bool,
}

/// An axe held in front of the camera. Its swings can damage nearby trees.
#[derive(Component)]
struct Axe {
    equipped: bool,
    psi: f32,
    phi: f32,
    psi_vel: f32,
    roll: f32,
    chop_cooldown: f32,
}

#[derive(Component)]
struct AxeHandle;

#[derive(Resource, Default)]
struct AxeUi {
    hover: bool,
    swinging: bool,
}

#[derive(Component)]
struct Shovel {
    equipped: bool,
    swing: f32,
    dig_cooldown: f32,
}

#[derive(Component)]
struct ShovelHandle;

#[derive(Resource, Default)]
struct ShovelUi {
    hover: bool,
    swinging: bool,
}

#[derive(Component)]
struct Spear {
    equipped: bool,
    impaled_fish: Option<Entity>,
    attack_left: f32,
    attack_duration: f32,
    push_distance: f32,
    attack_direction: Vec3,
    cooldown: f32,
    /// seconds left of the current roasting; > 0 while the fish is held over a fire
    cook_left: f32,
    /// world point the fish is held at while roasting
    cook_point: Vec3,
}

#[derive(Component)]
struct SpearHandle;

#[derive(Resource, Default)]
struct SpearUi {
    hover: bool,
    attacking: bool,
}


/// A machete held in front of the camera. Its swings can cut grass, plants and
/// sticks, and it behaves like the other tools (Q drops it).
#[derive(Component)]
struct Machete {
    equipped: bool,
    psi: f32,
    phi: f32,
    psi_vel: f32,
    roll: f32,
    slash_cooldown: f32,
    strip_cooldown: f32,
}

/// Grab this part of the machete (its grip) to pick it up.
#[derive(Component)]
struct MacheteHandle;

#[derive(Resource, Default)]
struct MacheteUi {
    hover: bool,
    swinging: bool,
}

/// State of the machete's bark-peeling interaction. Set while the player holds LMB on a
/// chopped trunk; cleared when they let go or walk away.
#[derive(Resource, Default)]
struct MachetePeel {
    /// The trunk we're peeling, if any.
    trunk: Option<Entity>,
    /// Which sector around the trunk axis we're peeling.
    sector: i32,
    /// The next row to strip.
    current_row: i32,
    /// Signed vertical mouse motion accumulated since the last strip.
    accumulated: f32,
    /// How long LMB has been pressed with the cursor over a chopped trunk.
    hold_time: f32,
}

/// A piece of bark that was scraped off a trunk. It's a normal physics object you can grab,
/// and it's excluded from the wood-debris cap so it doesn't get culled.
#[derive(Component)]
struct BarkStrip;

/// A knife held in front of the camera: short blade, quick slashes. Cuts grass,
/// plants and sticks like the machete. Q drops it.
#[derive(Component)]
struct Knife {
    equipped: bool,
    psi: f32,
    phi: f32,
    psi_vel: f32,
    roll: f32,
    slash_cooldown: f32,
}

/// Grab this part of the knife (its grip) to pick it up.
#[derive(Component)]
struct KnifeHandle;

#[derive(Resource, Default)]
struct KnifeUi {
    hover: bool,
    swinging: bool,
}


#[derive(Component)]
struct DigSite {
    radius: f32,
}

#[derive(Component)]
struct DiggableGround;

/// A tree that can be felled with repeated axe swings.
/// Physics-driven: it has no "health". It falls when the notch has eaten
/// enough of the trunk's load-bearing cross-section that the remaining
/// wood can no longer resist the tree's own weight and lean.
#[derive(Component)]
struct Tree {
    notch_depth: f32,
    // Real notch parameters, in trunk-local coordinates.
    // y_bottom = Y (0 at tree base) of the deepest point of the V.
    // height   = vertical extent of the V.
    // phi      = angle around the trunk's Y axis (0 = +X, pi/2 = +Z).
    notch_y_bottom: f32,
    notch_height: f32,
    notch_phi: f32,
    height: f32,
    far_lod: bool,
    base_radius: f32,
    top_radius: f32,
    lean: Vec2,
    bark: Handle<StandardMaterial>,
    inner_wood: Handle<StandardMaterial>,
    trunk_mesh: Handle<Mesh>,
    cavity: Option<Entity>,
    notch_center: Vec3,
    notch_normal: Vec3,
    cut_height: f32,
    fall_dir: Vec3,
    /// The trunk as a grid of wood blocks. Created on the first axe hit.
    wood: Option<TrunkWood>,
    /// Collider that replaces the full-height one once the upper trunk has fallen.
    stump_collider: Option<Entity>,
}

#[derive(Component)]
struct LeafSway {
    phase: f32,
    rest_rotation: Quat,
}

#[derive(Component)]
struct TreeLodPiece {
    tree: Entity,
    high_detail: bool,
}

#[derive(Component)]
struct TreeCutMark;

#[derive(Component)]
struct TreeChip {
    remaining: f32,
}

/// Wood chips / blocks knocked off a tree or log. Capped so they don't pile up forever.
#[derive(Component)]
struct WoodDebris {
    born: f32,
}

/// A wood chip that already has a V notch cut into it with the knife. Its mesh and collider
/// were replaced by ones with the wood really removed. These chips no longer time out or get
/// culled with the other debris.
#[derive(Component)]
struct KnifeNotched;

/// How many wood-debris entities may exist at once. Oldest ones are despawned when exceeded.
const MAX_WOOD_DEBRIS: usize = 400;

/// A felled trunk (or a piece of one). Its wood is the same layered block grid the standing
/// tree is made of, so it is chopped the same way: each swing chips the outer layer off the
/// spot that was hit, and when too little wood is left in one place the log snaps there.
#[derive(Component, Clone)]
struct ChoppableTrunk {
    radius: f32,
    center_offset: Vec3,
    bark: Handle<StandardMaterial>,
    /// The wood blocks this log still has.
    wood: TrunkWood,
    /// Add this to a point in the wood grid's coordinates to get the body-local point.
    offset: Vec3,
    /// The log's mesh (rebuilt in place after every swing).
    mesh: Handle<Mesh>,
    /// Child entity that carries the log's collision shape.
    collider: Entity,
}

/// A branch of a felled tree. Its limbs turn into layered wood where the axe hits them (see
/// `BranchShape::carved`), and when too little wood is left in one place the branch snaps
/// there: the part beyond the cut drops off as its own physics object, the part nearer the
/// trunk stays attached.
#[derive(Component)]
struct FelledBranch {
    /// True once the branch is a free physics body (not a child of the felled trunk).
    detached: bool,
}

/// Where along a parent limb its two child limbs sprout (0 = start, 1 = end).
const OAK_BRANCH_SPLIT: f32 = 0.62;

/// The limb geometry of one branch (segment tree + leaf clusters). Kept on the branch so the
/// meshes and colliders can be rebuilt when it is cut. All positions are branch-local.
#[derive(Component, Clone)]
struct BranchShape {
    segments: Vec<OakBranchSegment>,
    /// (leaf cluster position, seed, index of the segment that carries it)
    tips: Vec<(Vec3, u32, usize)>,
    wood: Handle<StandardMaterial>,
    foliage: [Handle<StandardMaterial>; 3],
    /// Per limb: its layered wood grid once an axe has hit it (None = still a plain cylinder).
    /// The grid lives in the limb's own frame: +Y runs from `start` to `end`.
    carved: Vec<Option<TrunkWood>>,
    /// Colour multipliers that turn the limb's wood material into the colour of each cut layer.
    cut_ratio: [[f32; 3]; WOOD_LAYERS],
    /// Leaves that have been pulled off this branch (see `OakLeaf::id`).
    plucked: Vec<u64>,
}

fn add_branch_cut_mark(
    branch_entity: Entity,
    hit_world: Vec3,
    ray_direction: Vec3,
    inner_wood: Handle<StandardMaterial>,
    transforms: &Query<&GlobalTransform>,
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
) {
    let Ok(branch_transform) = transforms.get(branch_entity) else {
        return;
    };
    let branch_world = branch_transform.compute_transform();
    let world_normal = -ray_direction.normalize_or_zero();
    if world_normal.length_squared() < 0.5 {
        return;
    }
    let local_point = branch_transform
        .compute_matrix()
        .inverse()
        .transform_point3(hit_world + world_normal * 0.015);
    let local_normal = branch_world.rotation.inverse() * world_normal;
    let local_rotation = Quat::from_rotation_arc(Vec3::Y, local_normal);
    let cut_disc = meshes.add(Cylinder::new(0.085, 0.018));
    commands.entity(branch_entity).with_children(|parent| {
        parent.spawn(PbrBundle {
            mesh: cut_disc,
            material: inner_wood,
            transform: Transform::from_translation(local_point).with_rotation(local_rotation),
            ..default()
        });
    });
}

#[derive(SystemParam)]
struct TreeHitQueries<'w, 's> {
    trees: Query<'w, 's, (Entity, &'static mut Tree, &'static GlobalTransform)>,
    parents: Query<'w, 's, &'static Parent>,
    branches: Query<'w, 's, (Entity, &'static mut TreeBranch)>,
    choppable_trunks: Query<'w, 's, (Entity, &'static mut ChoppableTrunk)>,
    felled_branches: Query<'w, 's, &'static mut FelledBranch>,
    children: Query<'w, 's, &'static Children>,
    shapes: Query<'w, 's, &'static BranchShape>,
    velocities: Query<'w, 's, &'static Velocity>,
    lod_pieces: Query<'w, 's, &'static TreeLodPiece>,
}

/// The brass knobs on the door; grab one to pull the door open.
#[derive(Component)]
struct DoorHandle;

/// `hover`: cursor is over the handle. `active`: you're holding it.
#[derive(Resource, Default)]
struct DoorGrab {
    hover: bool,
    active: bool,
}

/// Lets a person wander between random spots on their floor.
#[derive(Component)]
struct Walker {
    floor_y: f32,
    target: Vec3,
    speed: f32,
    wait: f32,
    seed: u32,
}

#[derive(Clone, Copy)]
enum PetKind {
    Cat,
    Dog,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PetActivity {
    Walking,
    Sitting,
    Running,
}

#[derive(Component)]
struct Pet {
    kind: PetKind,
    activity: PetActivity,
    target: Vec3,
    timer: f32,
    phase: f32,
    seed: u32,
}

#[derive(Component)]
struct PetBody;

#[derive(Component)]
struct PetLeg(u8);

#[derive(Component)]
struct PetTail;

/// Result of a background .fbx -> .glb conversion.
struct Converted {
    result: Result<String, String>, // Ok = asset path of the .glb
    transform: Transform,
    name: String,
}

#[derive(Resource)]
struct ConvertChannel {
    tx: mpsc::Sender<Converted>,
    rx: Mutex<mpsc::Receiver<Converted>>,
}

impl ConvertChannel {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            tx,
            rx: Mutex::new(rx),
        }
    }
}

#[derive(Resource, Default)]
struct Status(String);

#[derive(Resource)]
struct ImagePath(Option<String>);

#[derive(Resource, Default)]
struct Grab {
    entity: Option<Entity>,
    local_anchor: Vec3,
    distance: f32,
    /// > 0 while a bucket is held by its bail: how far (m) from the ears the grabbed spot is
    bail_reach: f32,
}

pub fn run() {
    let path = std::env::args().nth(1);
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Image -> 3D House".into(),
                // in a web page: fill the browser window (phones too)
                fit_canvas_to_parent: true,
                ..default()
            }),
            ..default()
        }))
        .add_plugins(DialogueFocusPlugin)
        .add_plugins(TouchControlsPlugin)
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .insert_resource(ImagePath(path))
        .insert_resource(Grab::default())
        .insert_resource(Status::default())
        .insert_resource(ConvertChannel::new())
        .insert_resource(DialogueState::default())
        .insert_resource(ClickConsumed::default())
        .insert_resource(Door::default())
        .insert_resource(DoorGrab::default())
        .insert_resource(SwordUi::default())
        .insert_resource(AxeUi::default())
        .insert_resource(ShovelUi::default())
        .insert_resource(SpearUi::default())
        .insert_resource(MacheteUi::default())
        .insert_resource(MachetePeel::default())
        .insert_resource(KnifeUi::default()) 
        .insert_resource(NpcSwing::default())
        .insert_resource(Possession::default())
        .insert_resource(HunterGear::default())
        .insert_resource(HunterActs::default())
        .insert_resource(BrushUi::default())
        .insert_resource(PaintDabs::default())
        .insert_resource(Climate::default())
        .insert_resource(RiverState {
            water_level: RIVER_WATER_Y,
            overflow: 0.0,
        })
        .insert_resource(RiverContact::default())
        .insert_resource(MudState::default())
        .insert_resource(StickState::default())
        .insert_resource(FishClickState::default())
        .insert_resource(Kneel::default())
        .add_systems(Update, kneel_system)
        .insert_resource(ClearColor(Color::srgb(0.1, 0.12, 0.16)))
        .insert_resource(AmbientLight {
            // Updated through the day/night and weather cycle.
            color: Color::srgb(0.55, 0.6, 0.7),
            brightness: 90.0,
        })
        .add_systems(Startup, (setup, setup_mud_assets, setup_water_assets))
        .add_systems(Startup, setup_fp_body.after(setup))
        .add_systems(Startup, setup_fire_assets)
        .add_systems(
            Update,
            (
                (
                    player_system,
                    possession_control_system,
                    horse_wander_system,
                    camera_follow,
                    sword_pose_system,
                    axe_pose_system,
                    shovel_pose_system,
                    spear_pose_system,
                    machete_peel_system,
                    machete_pose_system,
                    knife_pose_system, 
                    brush_pose_system,
                    bucket_scoop_system,
                    axe_chop_system,
                    shovel_dig_system,
                    machete_slash_system,
                    knife_slash_system,  
                )
                    .chain(),
                (
                    reset_click_consumed,
                    sword_input_system,
                    axe_input_system,
                    shovel_input_system,
                    spear_cook_system,
                    spear_input_system,
                    machete_input_system,
                    knife_input_system,
                    brush_input_system,
                    brush_paint_system,
                    npc_interact_system,
                    update_dialogue_focus,
                    horse_interact_system,
                    door_grab_system,
                    stick_input_system,
                    water_pick_system,
                    mud_pick_system,
                    bail_grab_system,
                    leaf_pluck_system,
                    grab_system,
                )
                    .chain(),
                update_dialogue_ui,
                handle_dropped_files,
                poll_conversions,
                finalize_models,
                clear_objects,
                update_status,
                climate_system,
                lightning_flicker,
                update_rain,
                update_puddles,
                sway_oak_leaves,
                pet_system,
                animate_river,
                swim_fish,
                water_splash_system,
                bird_flight_system,
                door_update,
                npc_walk_system,
            ),
        )
        .add_systems(
            Update,
            (
                pullable_nature_system,
                tree_lod_system,
                tree_chip_system,
                cap_wood_debris_system, 
                pebble_break_system,
                ensure_grab_components,
            ),
        )
        .add_systems(
            Update,
            (
                mud_blob_system,
                mud_drip_system,
                bucket_mud_system,
                bail_swing_system,
                mud_wash_system,
                water_blob_system,
                grasshopper_system,
                leaf_mantis_system,
                fp_body_system,
                stick_pose_system,
                burned_dust_smoke_system,   
                bark_feed_fire_system,
                fire_system,
                fire_particle_system,
            )
                .after(grab_system),
        )
        .add_systems(
            Update,
            (
                knife_carve_system
            )
                .after(grab_system),
        )
        .add_systems(
            PostUpdate,
            possession_apply_system
                .after(PhysicsSet::Writeback)
                .before(bevy::transform::TransformSystem::TransformPropagate),
        )
        .add_systems(
            Update,
            (
                hunter_setup_system,
                hunter_gear_system,
                hunter_guard_system,
                hunter_brain_system.before(axe_chop_system),
                hunter_actions_system,
                hunter_pose_system,
                hunter_ui_system,
            )
                .chain(),
        )
        .run();
}

// ---------------------------------------------------------------------------
// A hungry person: Tom gets hungry, then hunts a fish, chops a tree, peels bark, carves a
// V notch, rubs a stick into it until burned dust appears, feeds the embers with bark to make
// a fire, and cooks and eats his fish. Press H to make him hungry right now.
// ---------------------------------------------------------------------------

const HUNGER_RATE: f32 = 1.0 / 240.0; // per second: full to starving in four minutes
const HUNT_AT: f32 = 0.7; // he starts feeding himself at this hunger
const HUNTER_SPEED: f32 = 1.7; // m/s

#[derive(Clone, Copy, PartialEq, Eq)]
enum Held {
    Spear,
    Axe,
    Machete,
    Knife,
    Stick,
    Bark,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HState {
    Idle,
    FetchSpear,
    FindFish,
    Stab,
    FetchAxe,
    GoTree,
    Chop,
    FetchMachete,
    GoTrunk,
    Peel,
    FetchKnife,
    GoChip,
    Carve,
    FetchStick,
    GoNotch,
    Rub,
    FetchBark,
    CarryBark,
    Light,
    FetchSpearBack,
    GoFire,
    Cook,
    Eat,
}

/// Tom's hunger and what he is doing about it.
#[derive(Component)]
struct Hunter {
    /// 0 = full .. 1 = starving
    hunger: f32,
    state: HState,
    /// seconds spent in the current state
    t: f32,
    /// idle time before he may start feeding himself again
    cooldown: f32,
    holding: Option<(Entity, Held)>,
    target: Option<Entity>,
    fish: Option<Entity>,
    chip: Option<Entity>,
    pile_pos: Option<Vec3>,
    fire: Option<Entity>,
    /// the world point he is working on (the tool is posed towards it)
    aim: Option<Vec3>,
    swing_t: f32,
    hit_done: bool,
    swings: u32,
    steps: u32,
    attempts: u32,
    bark_placed: u32,
    trunk_count0: usize,
    msg: &'static str,
}

impl Hunter {
    fn new() -> Self {
        Self {
            hunger: 0.45,
            state: HState::Idle,
            t: 0.0,
            cooldown: 0.0,
            holding: None,
            target: None,
            fish: None,
            chip: None,
            pile_pos: None,
            fire: None,
            aim: None,
            swing_t: 0.0,
            hit_done: false,
            swings: 0,
            steps: 0,
            attempts: 0,
            bark_placed: 0,
            trunk_count0: 0,
            msg: "",
        }
    }
}

/// A tool, stick or bark strip Tom is carrying (it is posed in his hand every frame).
#[derive(Component)]
struct HeldByHunter;

/// A wood chip the knife could not turn into a V notch.
#[derive(Component)]
struct HunterSkip;

#[derive(Component)]
struct HungerText;

/// An axe swing made by Tom: the ray the axe is swung along. The axe system uses it.
#[derive(Resource, Default)]
struct NpcSwing {
    axe: Option<(Vec3, Vec3)>,
}

#[derive(Resource, Default)]
struct HunterGear {
    spear: Option<Entity>,
    axe: Option<Entity>,
    machete: Option<Entity>,
    knife: Option<Entity>,
}

enum HAct {
    PickUp(Entity),
    Drop { entity: Entity, pos: Vec3, rot: Quat },
    Impale { spear: Entity, fish: Entity },
    Peel { trunk: Entity, from: Vec3 },
    Carve { chip: Entity },
    Pile { pos: Vec3 },
    Fire { pos: Vec3 },
    CookStep { fish: Entity, dt: f32 },
    Eat { fish: Entity, spear: Entity },
}

#[derive(Resource, Default)]
struct HunterActs {
    list: Vec<HAct>,
}

#[derive(SystemParam)]
struct HunterWorld<'w, 's> {
    globals: Query<'w, 's, &'static GlobalTransform>,
    fish: Query<'w, 's, (Entity, &'static Fish)>,
    trees: Query<'w, 's, (Entity, &'static Tree)>,
    trunks: Query<'w, 's, (Entity, &'static ChoppableTrunk)>,
    chips: Query<
        'w,
        's,
        Entity,
        (
            Or<(With<WoodDebris>, With<TreeChip>)>,
            Without<KnifeNotched>,
            Without<HunterSkip>,
        ),
    >,
    notched: Query<'w, 's, (), With<KnifeNotched>>,
    skipped: Query<'w, 's, (), With<HunterSkip>>,
    sticks: Query<'w, 's, (Entity, &'static Stick), Without<HeldByHunter>>,
    bark: Query<'w, 's, Entity, (With<BarkStrip>, Without<HeldByHunter>)>,
    fires: Query<'w, 's, (Entity, &'static Fire)>,
    spears: Query<'w, 's, &'static Spear>,
    axes: Query<'w, 's, &'static Axe>,
    machetes: Query<'w, 's, &'static Machete>,
    knives: Query<'w, 's, &'static Knife>,
    grab: Res<'w, Grab>,
}

/// Walks one step towards `dest` (ground point). True once he has arrived.
fn hunter_walk(tf: &mut Transform, dest: Vec3, dt: f32, speed: f32, tol: f32) -> bool {
    let pos = tf.translation;
    let (step, is_final) = hunter_step_target(pos, dest);
    let mut to = step - pos;
    to.y = 0.0;
    let dist = to.length();
    if dist > 0.02 {
        let dir = to / dist;
        tf.translation += dir * (speed * dt).min(dist);
        tf.rotation = Quat::from_rotation_y((-dir.x).atan2(-dir.z));
    }
    let ground = if hunter_inside(tf.translation) {
        F1_FLOOR_Y
    } else {
        land_surface_height(tf.translation.x, tf.translation.z)
    } + NPC_CENTER_Y;
    tf.translation.y += (ground - tf.translation.y) * (1.0 - (-12.0 * dt).exp());
    let mut left = dest - tf.translation;
    left.y = 0.0;
    is_final && left.length() < tol
}

fn hunter_face(tf: &mut Transform, point: Vec3) {
    let mut d = point - tf.translation;
    d.y = 0.0;
    if d.length() > 0.01 {
        let d = d.normalize();
        tf.rotation = Quat::from_rotation_y((-d.x).atan2(-d.z));
    }
}

fn hunter_next(h: &mut Hunter, state: HState) {
    h.state = state;
    h.t = 0.0;
    h.swing_t = 0.0;
    h.hit_done = false;
    h.steps = 0;
    h.msg = hunter_state_text(state);
}

fn hunter_state_text(state: HState) -> &'static str {
    match state {
        HState::Idle => "",
        HState::FetchSpear => "Hungry. Going for the spear to catch a fish.",
        HState::FindFish => "Looking for a fish in the river...",
        HState::Stab => "Got you!",
        HState::FetchAxe => "Now I need the axe to chop a tree.",
        HState::GoTree | HState::Chop => "Chopping down a tree for wood.",
        HState::FetchMachete => "I need the machete to cut the bark.",
        HState::GoTrunk | HState::Peel => "Cutting bark off the log.",
        HState::FetchKnife => "Time to get the knife.",
        HState::GoChip | HState::Carve => "Cutting a V notch into a piece of wood.",
        HState::FetchStick => "I need a stick.",
        HState::GoNotch | HState::Rub => "Rubbing the stick in the notch until burned dust appears.",
        HState::FetchBark | HState::CarryBark => "Carrying bark to the burned dust.",
        HState::Light => "Feeding the embers with bark to make a fire.",
        HState::FetchSpearBack => "Going back for my spear with the fish on it.",
        HState::GoFire | HState::Cook => "Cooking the fish over the fire.",
        HState::Eat => "Mmm, that is good.",
    }
}

/// Puts down whatever he carries and goes back to being idle for a while.
fn hunter_abort(h: &mut Hunter, acts: &mut HunterActs, tf: &Transform, why: &'static str) {
    if let Some((e, _)) = h.holding.take() {
        acts.list.push(HAct::Drop {
            entity: e,
            pos: hunter_feet(tf),
            rot: Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
        });
    }
    h.state = HState::Idle;
    h.t = 0.0;
    h.cooldown = 20.0;
    h.target = None;
    h.msg = why;
}

/// A spot on the ground just in front of him, where he lays things down.
fn hunter_feet(tf: &Transform) -> Vec3 {
    let fwd = tf.rotation * Vec3::NEG_Z;
    tf.translation - Vec3::Y * (NPC_CENTER_Y - 0.3) + fwd * 0.6
}

/// Where to stand to pick something up: in front of the table for things lying on it.
fn hunter_approach_point(tool: Vec3, from: Vec3) -> Vec3 {
    if hunter_inside(tool) && tool.z < -1.6 && tool.z > -3.6 && tool.x < -0.2 {
        return Vec3::new(tool.x.clamp(-3.4, -0.5), 0.0, -1.35);
    }
    let mut d = Vec3::new(from.x - tool.x, 0.0, from.z - tool.z);
    d = d.normalize_or_zero();
    Vec3::new(tool.x, 0.0, tool.z) + d * 1.0
}

enum Fetch {
    Walking,
    Got,
    Blocked,
}

fn hunter_fetch(
    h: &mut Hunter,
    tf: &mut Transform,
    dt: f32,
    tool: Option<Entity>,
    kind: Held,
    free: bool,
    globals: &Query<&GlobalTransform>,
    acts: &mut HunterActs,
) -> Fetch {
    let Some(tool) = tool else {
        return Fetch::Blocked;
    };
    if !free {
        return Fetch::Blocked;
    }
    let Ok(g) = globals.get(tool) else {
        return Fetch::Blocked;
    };
    let tp = g.translation();
    let dest = hunter_approach_point(tp, tf.translation);
    let arrived = hunter_walk(tf, dest, dt, HUNTER_SPEED, 0.35);
    let mut flat = tp - tf.translation;
    flat.y = 0.0;
    if arrived || flat.length() < 0.9 {
        hunter_face(tf, tp);
        acts.list.push(HAct::PickUp(tool));
        h.holding = Some((tool, kind));
        return Fetch::Got;
    }
    Fetch::Walking
}

fn hunter_orient(dir: Vec3, side: Vec3) -> Quat {
    let y = dir.normalize_or_zero();
    let y = if y == Vec3::ZERO { Vec3::Y } else { y };
    let mut z = side - y * side.dot(y);
    if z.length_squared() < 1e-4 {
        z = Vec3::X - y * Vec3::X.dot(y);
    }
    if z.length_squared() < 1e-4 {
        z = Vec3::Z - y * Vec3::Z.dot(y);
    }
    let z = z.normalize();
    let x = y.cross(z);
    Quat::from_mat3(&Mat3::from_cols(x, y, z))
}

/// Like `hunter_orient`, but `across` is where the tool's local X axis points. (An axe head is
/// wide along X, so its cutting edge runs along `across`.)
fn hunter_orient_across(dir: Vec3, across: Vec3) -> Quat {
    let y = dir.normalize_or_zero();
    let y = if y == Vec3::ZERO { Vec3::Y } else { y };
    let mut x = across - y * across.dot(y);
    if x.length_squared() < 1e-4 {
        x = Vec3::X - y * Vec3::X.dot(y);
    }
    if x.length_squared() < 1e-4 {
        x = Vec3::Z - y * Vec3::Z.dot(y);
    }
    let x = x.normalize();
    let z = x.cross(y);
    Quat::from_mat3(&Mat3::from_cols(x, y, z))
}

/// Gives Tom his hunger (and a status line), once.
fn hunter_setup_system(
    mut commands: Commands,
    npcs: Query<Entity, (With<Npc>, Without<Hunter>)>,
    mut ui_done: Local<bool>,
) {
    for e in &npcs {
        commands.entity(e).insert(Hunter::new());
    }
    if !*ui_done {
        *ui_done = true;
        commands.spawn((
            TextBundle::from_section(
                "",
                TextStyle {
                    font_size: 20.0,
                    color: Color::srgb(1.0, 0.8, 0.5),
                    ..default()
                },
            )
            .with_style(Style {
                position_type: PositionType::Absolute,
                top: Val::Px(8.0),
                // on a phone the touch buttons run down the right edge, so keep clear of them
                right: Val::Px(if ALWAYS_SHOW_HUD { 96.0 } else { 8.0 }),
                ..default()
            }),
            HungerText,
        ));
    }
}

fn hunter_gear_system(
    mut gear: ResMut<HunterGear>,
    spears: Query<Entity, With<Spear>>,
    axes: Query<Entity, With<Axe>>,
    machetes: Query<Entity, With<Machete>>,
    knives: Query<Entity, With<Knife>>,
) {
    if gear.spear.is_none() {
        gear.spear = spears.iter().next();
    }
    if gear.axe.is_none() {
        gear.axe = axes.iter().next();
    }
    if gear.machete.is_none() {
        gear.machete = machetes.iter().next();
    }
    if gear.knife.is_none() {
        gear.knife = knives.iter().next();
    }
}

/// If the player takes a tool out of Tom's hand, he gives up on the current job for a while.
fn hunter_guard_system(
    mut commands: Commands,
    mut hunters: Query<&mut Hunter>,
    spears: Query<(Entity, &Spear), With<HeldByHunter>>,
    axes: Query<(Entity, &Axe), With<HeldByHunter>>,
    machetes: Query<(Entity, &Machete), With<HeldByHunter>>,
    knives: Query<(Entity, &Knife), With<HeldByHunter>>,
    sticks: Query<(Entity, &Stick), With<HeldByHunter>>,
) {
    let mut taken: Vec<Entity> = Vec::new();
    for (e, t) in &spears {
        if t.equipped {
            taken.push(e);
        }
    }
    for (e, t) in &axes {
        if t.equipped {
            taken.push(e);
        }
    }
    for (e, t) in &machetes {
        if t.equipped {
            taken.push(e);
        }
    }
    for (e, t) in &knives {
        if t.equipped {
            taken.push(e);
        }
    }
    for (e, t) in &sticks {
        if t.equipped {
            taken.push(e);
        }
    }
    for e in taken {
        commands.entity(e).remove::<HeldByHunter>();
        for mut h in &mut hunters {
            if h.holding.map(|x| x.0) == Some(e) {
                h.holding = None;
                h.state = HState::Idle;
                h.t = 0.0;
                h.cooldown = 25.0;
                h.target = None;
                h.msg = "Hey, I was using that!";
            }
        }
    }
}

/// Tom's decisions and walking. The actual changes to the world are queued in `HunterActs`.
fn hunter_brain_system(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut door: ResMut<Door>,
    dialogue: Res<DialogueState>,
    poss: Res<Possession>,
    gear: Res<HunterGear>,
    mut acts: ResMut<HunterActs>,
    mut swing: ResMut<NpcSwing>,
    world: HunterWorld,
    mut hunters: Query<(Entity, &mut Transform, &mut Hunter)>,
) {
    let dt = time.delta_seconds().min(0.05);
    swing.axe = None;
    for (e, mut tf, mut h) in &mut hunters {
        h.hunger = (h.hunger + dt * HUNGER_RATE).min(1.0);
        if keys.just_pressed(KeyCode::KeyH) {
            h.hunger = 1.0;
            h.cooldown = 0.0;
        }
        if h.cooldown > 0.0 {
            h.cooldown -= dt;
        }
        if dialogue.entity == Some(e) || poss.target == Some(e) {
            continue; // he stops and talks to you, or you are him
        }
        h.aim = None;
        let state = h.state;

        // He opens the front door when he is going through it.
        if state != HState::Idle
            && tf.translation.distance(Vec3::new(4.2, tf.translation.y, 0.0)) < 4.5
        {
            door.angle += (1.55 - door.angle).clamp(-0.06, 0.06);
        }

        let pos = tf.translation;
        match state {
            HState::Idle => {
                h.t += dt;
                if !hunter_inside(pos) {
                    // he is outside with nothing to do: go back in
                    let _ = hunter_walk(&mut tf, Vec3::new(0.0, 0.0, 1.0), dt, HUNTER_SPEED, 0.5);
                } else if h.hunger >= HUNT_AT && h.cooldown <= 0.0 {
                    let fish_on_spear = gear
                        .spear
                        .and_then(|s| world.spears.get(s).ok())
                        .map_or(false, |s| s.impaled_fish.is_some());
                    let fire_burning = world
                        .fires
                        .iter()
                        .any(|(_, f)| f.fuel > 0.4 && f.size > 0.1);
                    let first = if fish_on_spear && fire_burning {
                        HState::FetchSpearBack
                    } else if fish_on_spear {
                        HState::FetchAxe
                    } else {
                        HState::FetchSpear
                    };
                    h.attempts = 0;
                    h.bark_placed = 0;
                    h.fish = None;
                    h.chip = None;
                    h.pile_pos = None;
                    hunter_next(&mut h, first);
                }
            }

            HState::FetchSpear => {
                h.t += dt;
                let free = gear
                    .spear
                    .and_then(|s| world.spears.get(s).ok())
                    .map_or(false, |s| !s.equipped);
                match hunter_fetch(&mut h, &mut tf, dt, gear.spear, Held::Spear, free, &world.globals, &mut acts) {
                    Fetch::Got => hunter_next(&mut h, HState::FindFish),
                    Fetch::Blocked => {
                        h.msg = "Waiting for the spear...";
                        if h.t > 45.0 {
                            hunter_abort(&mut h, &mut acts, &tf, "I can't get the spear.");
                        }
                    }
                    Fetch::Walking => {}
                }
            }

            HState::FindFish => {
                h.t += dt;
                let mut best: Option<(Entity, Vec3, f32)> = None;
                for (fe, f) in world.fish.iter() {
                    if f.impaled {
                        continue;
                    }
                    if let Ok(g) = world.globals.get(fe) {
                        let fp = g.translation();
                        let d = Vec2::new(fp.x - pos.x, fp.z - pos.z).length();
                        if best.map_or(true, |b| d < b.2) {
                            best = Some((fe, fp, d));
                        }
                    }
                }
                match best {
                    None => hunter_abort(&mut h, &mut acts, &tf, "There are no fish to catch."),
                    Some((fe, fp, d)) => {
                        h.target = Some(fe);
                        h.aim = Some(fp);
                        let dir = Vec3::new(fp.x - pos.x, 0.0, fp.z - pos.z).normalize_or_zero();
                        let stand = Vec3::new(fp.x, 0.0, fp.z) - dir * 2.1;
                        if d < 2.7 {
                            h.fish = Some(fe);
                            hunter_next(&mut h, HState::Stab);
                        } else {
                            let _ = hunter_walk(&mut tf, stand, dt, HUNTER_SPEED * 1.2, 0.2);
                        }
                        if h.t > 150.0 {
                            hunter_abort(&mut h, &mut acts, &tf, "The fish got away.");
                        }
                    }
                }
            }

            HState::Stab => {
                h.t += dt;
                let Some(fe) = h.fish else {
                    hunter_next(&mut h, HState::FindFish);
                    continue;
                };
                let fish_pos = world.globals.get(fe).map(|g| g.translation()).ok();
                h.aim = fish_pos;
                if let Some(fp) = fish_pos {
                    hunter_face(&mut tf, fp);
                }
                if h.t >= 0.4 && !h.hit_done {
                    h.hit_done = true;
                    let close = fish_pos
                        .map_or(false, |fp| Vec2::new(fp.x - pos.x, fp.z - pos.z).length() < 3.4)
                        && world.fish.get(fe).map_or(false, |(_, f)| !f.impaled);
                    if close {
                        if let Some(spear) = gear.spear {
                            acts.list.push(HAct::Impale { spear, fish: fe });
                            // he lays the spear (with the fish on it) down and gets on with the rest
                            h.holding = None;
                            acts.list.push(HAct::Drop {
                                entity: spear,
                                pos: hunter_feet(&tf) - (tf.rotation * Vec3::NEG_Z) * 0.8,
                                rot: Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
                            });
                            hunter_next(&mut h, HState::FetchAxe);
                            continue;
                        }
                    }
                }
                if h.t >= 1.0 {
                    h.attempts += 1;
                    if h.attempts > 10 {
                        hunter_abort(&mut h, &mut acts, &tf, "I keep missing the fish.");
                    } else {
                        h.fish = None;
                        hunter_next(&mut h, HState::FindFish);
                    }
                }
            }

            HState::FetchAxe => {
                h.t += dt;
                let free = gear
                    .axe
                    .and_then(|s| world.axes.get(s).ok())
                    .map_or(false, |s| !s.equipped);
                match hunter_fetch(&mut h, &mut tf, dt, gear.axe, Held::Axe, free, &world.globals, &mut acts) {
                    Fetch::Got => {
                        h.target = None;
                        hunter_next(&mut h, HState::GoTree);
                    }
                    Fetch::Blocked => {
                        h.msg = "Waiting for the axe...";
                        if h.t > 45.0 {
                            hunter_abort(&mut h, &mut acts, &tf, "I can't get the axe.");
                        }
                    }
                    Fetch::Walking => {}
                }
            }

            HState::GoTree => {
                h.t += dt;
                let valid = h
                    .target
                    .map_or(false, |t| world.trees.get(t).map_or(false, |(_, tr)| tr.stump_collider.is_none()));
                if !valid {
                    let mut best: Option<(Entity, f32)> = None;
                    for (te, tr) in world.trees.iter() {
                        if tr.stump_collider.is_some() {
                            continue;
                        }
                        if let Ok(g) = world.globals.get(te) {
                            let tp = g.translation();
                            let d = Vec2::new(tp.x - pos.x, tp.z - pos.z).length();
                            if best.map_or(true, |b| d < b.1) {
                                best = Some((te, d));
                            }
                        }
                    }
                    h.target = best.map(|b| b.0);
                }
                let Some(te) = h.target else {
                    hunter_abort(&mut h, &mut acts, &tf, "I can't find a tree.");
                    continue;
                };
                let (Ok((_, tree)), Ok(g)) = (world.trees.get(te), world.globals.get(te)) else {
                    h.target = None;
                    continue;
                };
                let base = g.translation() - Vec3::Y * (tree.height * 0.5);
                let to_me = Vec3::new(pos.x - base.x, 0.0, pos.z - base.z).normalize_or_zero();
                let stand = Vec3::new(base.x, 0.0, base.z) + to_me * (tree.base_radius + 0.85);
                h.aim = Some(base + Vec3::Y * 0.9);
                if hunter_walk(&mut tf, stand, dt, HUNTER_SPEED, 0.3) {
                    h.trunk_count0 = world.trunks.iter().count();
                    h.swings = 0;
                    hunter_next(&mut h, HState::Chop);
                } else if h.t > 150.0 {
                    hunter_abort(&mut h, &mut acts, &tf, "That tree is too far.");
                }
            }

            HState::Chop => {
                h.t += dt;
                let Some(te) = h.target else {
                    hunter_abort(&mut h, &mut acts, &tf, "Lost my tree.");
                    continue;
                };
                let tree_info = world.trees.get(te).ok().and_then(|(_, tr)| {
                    world
                        .globals
                        .get(te)
                        .ok()
                        .map(|g| (g.translation() - Vec3::Y * (tr.height * 0.5), tr.stump_collider.is_some()))
                });
                let felled = match tree_info {
                    Some((_, stump)) => stump,
                    None => true,
                } || world.trunks.iter().count() > h.trunk_count0;
                if felled {
                    if let Some((axe, _)) = h.holding.take() {
                        acts.list.push(HAct::Drop {
                            entity: axe,
                            pos: hunter_feet(&tf),
                            rot: Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
                        });
                    }
                    h.target = None;
                    hunter_next(&mut h, HState::FetchMachete);
                    continue;
                }
                let Some((base, _)) = tree_info else { continue };
                let aim = base + Vec3::Y * 0.9;
                h.aim = Some(aim);
                hunter_face(&mut tf, aim);
                // One overhead chop per second: the axe goes up (0 - 0.55), comes down (0.55 - 0.67)
                // and bites into the trunk at 0.67 (the pose is in `hunter_pose_system`).
                h.swing_t += dt;
                if !h.hit_done && h.swing_t >= 0.67 {
                    h.hit_done = true;
                    // The ray must start OUTSIDE his own body, or it just hits Tom himself
                    // (and nothing ever gets chopped).
                    let flat = Vec3::new(aim.x - pos.x, 0.0, aim.z - pos.z).normalize_or_zero();
                    let origin = pos + flat * 0.5 + Vec3::Y * 0.25;
                    let dir = (aim - origin).normalize_or_zero();
                    swing.axe = Some((origin, dir));
                    h.swings += 1;
                }
                if h.swing_t >= 1.0 {
                    h.swing_t = 0.0;
                    h.hit_done = false;
                }
                if h.swings > 220 {
                    hunter_abort(&mut h, &mut acts, &tf, "This tree is too tough.");
                }
            }

            HState::FetchMachete => {
                h.t += dt;
                let free = gear
                    .machete
                    .and_then(|s| world.machetes.get(s).ok())
                    .map_or(false, |s| !s.equipped);
                match hunter_fetch(&mut h, &mut tf, dt, gear.machete, Held::Machete, free, &world.globals, &mut acts) {
                    Fetch::Got => {
                        h.target = None;
                        hunter_next(&mut h, HState::GoTrunk);
                    }
                    Fetch::Blocked => {
                        h.msg = "Waiting for the machete...";
                        if h.t > 45.0 {
                            hunter_abort(&mut h, &mut acts, &tf, "I can't get the machete.");
                        }
                    }
                    Fetch::Walking => {}
                }
            }

            HState::GoTrunk => {
                h.t += dt;
                if h.target.map_or(true, |t| world.trunks.get(t).is_err()) {
                    let mut best: Option<(Entity, f32)> = None;
                    for (te, _) in world.trunks.iter() {
                        if let Ok(g) = world.globals.get(te) {
                            let tp = g.translation();
                            let d = Vec2::new(tp.x - pos.x, tp.z - pos.z).length();
                            if best.map_or(true, |b| d < b.1) {
                                best = Some((te, d));
                            }
                        }
                    }
                    h.target = best.map(|b| b.0);
                }
                let Some(te) = h.target else {
                    hunter_abort(&mut h, &mut acts, &tf, "There is no log to peel.");
                    continue;
                };
                let (Ok((_, tr)), Ok(g)) = (world.trunks.get(te), world.globals.get(te)) else {
                    h.target = None;
                    continue;
                };
                let len = tr.wood.rows as f32 * WOOD_ROW_H;
                let center = g.affine().transform_point3(tr.offset + Vec3::Y * len * 0.5);
                let axis = g.affine().transform_vector3(Vec3::Y).normalize_or_zero();
                let mut side = axis.cross(Vec3::Y);
                side.y = 0.0;
                let mut side = side.normalize_or_zero();
                if side == Vec3::ZERO {
                    side = Vec3::X;
                }
                if side.dot(pos - center) < 0.0 {
                    side = -side;
                }
                let stand = Vec3::new(center.x, 0.0, center.z) + side * (tr.radius + 0.9);
                h.aim = Some(center);
                if hunter_walk(&mut tf, stand, dt, HUNTER_SPEED, 0.3) {
                    hunter_next(&mut h, HState::Peel);
                } else if h.t > 150.0 {
                    hunter_abort(&mut h, &mut acts, &tf, "I can't reach the log.");
                }
            }

            HState::Peel => {
                h.t += dt;
                let Some(te) = h.target else {
                    hunter_abort(&mut h, &mut acts, &tf, "Lost the log.");
                    continue;
                };
                let center = match (world.trunks.get(te), world.globals.get(te)) {
                    (Ok((_, tr)), Ok(g)) => {
                        let len = tr.wood.rows as f32 * WOOD_ROW_H;
                        g.affine().transform_point3(tr.offset + Vec3::Y * len * 0.5)
                    }
                    _ => {
                        hunter_abort(&mut h, &mut acts, &tf, "The log is gone.");
                        continue;
                    }
                };
                h.aim = Some(center);
                hunter_face(&mut tf, center);
                h.swing_t += dt;
                if h.swing_t >= 0.32 {
                    h.swing_t = 0.0;
                    acts.list.push(HAct::Peel { trunk: te, from: pos });
                    h.steps += 1;
                }
                let bark_n = world.bark.iter().count();
                if (h.steps >= 8 && bark_n >= 3) || h.steps >= 30 {
                    if let Some((m, _)) = h.holding.take() {
                        acts.list.push(HAct::Drop {
                            entity: m,
                            pos: hunter_feet(&tf),
                            rot: Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
                        });
                    }
                    h.target = None;
                    hunter_next(&mut h, HState::FetchKnife);
                }
            }

            HState::FetchKnife => {
                h.t += dt;
                let free = gear
                    .knife
                    .and_then(|s| world.knives.get(s).ok())
                    .map_or(false, |s| !s.equipped);
                match hunter_fetch(&mut h, &mut tf, dt, gear.knife, Held::Knife, free, &world.globals, &mut acts) {
                    Fetch::Got => {
                        h.target = None;
                        hunter_next(&mut h, HState::GoChip);
                    }
                    Fetch::Blocked => {
                        h.msg = "Waiting for the knife...";
                        if h.t > 45.0 {
                            hunter_abort(&mut h, &mut acts, &tf, "I can't get the knife.");
                        }
                    }
                    Fetch::Walking => {}
                }
            }

            HState::GoChip => {
                h.t += dt;
                if h.target.map_or(true, |t| world.chips.get(t).is_err()) {
                    let mut best: Option<(Entity, f32)> = None;
                    for ce in world.chips.iter() {
                        if let Ok(g) = world.globals.get(ce) {
                            let cp = g.translation();
                            let d = Vec2::new(cp.x - pos.x, cp.z - pos.z).length();
                            if best.map_or(true, |b| d < b.1) {
                                best = Some((ce, d));
                            }
                        }
                    }
                    h.target = best.map(|b| b.0);
                }
                let Some(ce) = h.target else {
                    hunter_abort(&mut h, &mut acts, &tf, "I can't find a piece of wood to carve.");
                    continue;
                };
                let Ok(g) = world.globals.get(ce) else {
                    h.target = None;
                    continue;
                };
                let cp = g.translation();
                let d = Vec3::new(pos.x - cp.x, 0.0, pos.z - cp.z).normalize_or_zero();
                let stand = Vec3::new(cp.x, 0.0, cp.z) + d * 0.8;
                h.aim = Some(cp);
                if hunter_walk(&mut tf, stand, dt, HUNTER_SPEED, 0.3) {
                    hunter_next(&mut h, HState::Carve);
                } else if h.t > 150.0 {
                    hunter_abort(&mut h, &mut acts, &tf, "The wood is too far.");
                }
            }

            HState::Carve => {
                h.t += dt;
                let Some(ce) = h.target else {
                    hunter_next(&mut h, HState::GoChip);
                    continue;
                };
                if let Ok(g) = world.globals.get(ce) {
                    let cp = g.translation();
                    h.aim = Some(cp);
                    hunter_face(&mut tf, cp);
                }
                if h.t >= 0.7 && !h.hit_done {
                    h.hit_done = true;
                    acts.list.push(HAct::Carve { chip: ce });
                }
                if h.hit_done {
                    if world.notched.get(ce).is_ok() {
                        h.chip = Some(ce);
                        if let Some((k, _)) = h.holding.take() {
                            acts.list.push(HAct::Drop {
                                entity: k,
                                pos: hunter_feet(&tf),
                                rot: Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
                            });
                        }
                        h.target = None;
                        hunter_next(&mut h, HState::FetchStick);
                    } else if world.skipped.get(ce).is_ok() || h.t > 3.0 {
                        h.attempts += 1;
                        h.target = None;
                        if h.attempts > 12 {
                            hunter_abort(&mut h, &mut acts, &tf, "None of this wood can be notched.");
                        } else {
                            hunter_next(&mut h, HState::GoChip);
                        }
                    }
                }
            }

            HState::FetchStick => {
                h.t += dt;
                if h.target.map_or(true, |t| world.sticks.get(t).map_or(true, |(_, s)| s.equipped)) {
                    let mut best: Option<(Entity, f32)> = None;
                    for (se, st) in world.sticks.iter() {
                        if st.equipped || world.grab.entity == Some(se) {
                            continue;
                        }
                        if let Ok(g) = world.globals.get(se) {
                            let sp = g.translation();
                            let d = Vec2::new(sp.x - pos.x, sp.z - pos.z).length();
                            if best.map_or(true, |b| d < b.1) {
                                best = Some((se, d));
                            }
                        }
                    }
                    h.target = best.map(|b| b.0);
                }
                if h.target.is_none() {
                    hunter_abort(&mut h, &mut acts, &tf, "I can't find a stick.");
                    continue;
                }
                let tool = h.target;
                match hunter_fetch(&mut h, &mut tf, dt, tool, Held::Stick, true, &world.globals, &mut acts) {
                    Fetch::Got => {
                        h.target = None;
                        hunter_next(&mut h, HState::GoNotch);
                    }
                    Fetch::Blocked => h.target = None,
                    Fetch::Walking => {
                        if h.t > 150.0 {
                            hunter_abort(&mut h, &mut acts, &tf, "The stick is too far.");
                        }
                    }
                }
            }

            HState::GoNotch => {
                h.t += dt;
                let Some(ce) = h.chip else {
                    hunter_abort(&mut h, &mut acts, &tf, "Lost the notched wood.");
                    continue;
                };
                let Ok(g) = world.globals.get(ce) else {
                    hunter_abort(&mut h, &mut acts, &tf, "The notched wood is gone.");
                    continue;
                };
                let cp = g.translation();
                let d = Vec3::new(pos.x - cp.x, 0.0, pos.z - cp.z).normalize_or_zero();
                let stand = Vec3::new(cp.x, 0.0, cp.z) + d * 0.8;
                h.aim = Some(cp);
                if hunter_walk(&mut tf, stand, dt, HUNTER_SPEED, 0.3) {
                    hunter_next(&mut h, HState::Rub);
                } else if h.t > 150.0 {
                    hunter_abort(&mut h, &mut acts, &tf, "The notched wood is too far.");
                }
            }

            HState::Rub => {
                h.t += dt;
                let Some(ce) = h.chip else {
                    hunter_abort(&mut h, &mut acts, &tf, "Lost the notched wood.");
                    continue;
                };
                let Ok(g) = world.globals.get(ce) else {
                    hunter_abort(&mut h, &mut acts, &tf, "The notched wood is gone.");
                    continue;
                };
                let cp = g.translation();
                h.aim = Some(cp);
                hunter_face(&mut tf, cp);
                if h.t >= 6.0 {
                    let pile = cp + Vec3::Y * 0.03;
                    acts.list.push(HAct::Pile { pos: pile });
                    h.pile_pos = Some(pile);
                    h.bark_placed = 0;
                    if let Some((st, _)) = h.holding.take() {
                        acts.list.push(HAct::Drop {
                            entity: st,
                            pos: hunter_feet(&tf),
                            rot: Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
                        });
                    }
                    h.target = None;
                    hunter_next(&mut h, HState::FetchBark);
                }
            }

            HState::FetchBark => {
                h.t += dt;
                if h.target.map_or(true, |t| world.bark.get(t).is_err()) {
                    let mut best: Option<(Entity, f32)> = None;
                    for be in world.bark.iter() {
                        if world.grab.entity == Some(be) {
                            continue;
                        }
                        if let Ok(g) = world.globals.get(be) {
                            let bp = g.translation();
                            let d = Vec2::new(bp.x - pos.x, bp.z - pos.z).length();
                            if best.map_or(true, |b| d < b.1) {
                                best = Some((be, d));
                            }
                        }
                    }
                    h.target = best.map(|b| b.0);
                }
                if h.target.is_none() {
                    if h.bark_placed > 0 {
                        hunter_next(&mut h, HState::Light);
                    } else {
                        hunter_abort(&mut h, &mut acts, &tf, "I can't find any bark.");
                    }
                    continue;
                }
                let tool = h.target;
                match hunter_fetch(&mut h, &mut tf, dt, tool, Held::Bark, true, &world.globals, &mut acts) {
                    Fetch::Got => {
                        h.target = None;
                        hunter_next(&mut h, HState::CarryBark);
                    }
                    Fetch::Blocked => h.target = None,
                    Fetch::Walking => {
                        if h.t > 150.0 {
                            hunter_abort(&mut h, &mut acts, &tf, "The bark is too far.");
                        }
                    }
                }
            }

            HState::CarryBark => {
                h.t += dt;
                let Some(pile) = h.pile_pos else {
                    hunter_abort(&mut h, &mut acts, &tf, "Lost the burned dust.");
                    continue;
                };
                let d = Vec3::new(pos.x - pile.x, 0.0, pos.z - pile.z).normalize_or_zero();
                let stand = Vec3::new(pile.x, 0.0, pile.z) + d * 0.9;
                h.aim = Some(pile);
                if hunter_walk(&mut tf, stand, dt, HUNTER_SPEED, 0.3) {
                    hunter_face(&mut tf, pile);
                    if let Some((b, _)) = h.holding.take() {
                        acts.list.push(HAct::Drop {
                            entity: b,
                            pos: pile + Vec3::new(0.18, 0.15, 0.08 * h.bark_placed as f32),
                            rot: Quat::IDENTITY,
                        });
                    }
                    h.bark_placed += 1;
                    if h.bark_placed < 3 && world.bark.iter().count() > 0 {
                        hunter_next(&mut h, HState::FetchBark);
                    } else {
                        hunter_next(&mut h, HState::Light);
                    }
                } else if h.t > 150.0 {
                    hunter_abort(&mut h, &mut acts, &tf, "The burned dust is too far.");
                }
            }

            HState::Light => {
                h.t += dt;
                let Some(pile) = h.pile_pos else {
                    hunter_abort(&mut h, &mut acts, &tf, "Lost the burned dust.");
                    continue;
                };
                h.aim = Some(pile);
                hunter_face(&mut tf, pile);
                if h.t >= 0.8 && !h.hit_done {
                    h.hit_done = true;
                    acts.list.push(HAct::Fire { pos: pile });
                }
                if h.t >= 1.8 {
                    let mut best: Option<(Entity, f32)> = None;
                    for (fe, f) in world.fires.iter() {
                        if f.fuel <= 0.05 {
                            continue;
                        }
                        if let Ok(g) = world.globals.get(fe) {
                            let d = g.translation().distance(pile);
                            if best.map_or(true, |b| d < b.1) {
                                best = Some((fe, d));
                            }
                        }
                    }
                    h.fire = best.map(|b| b.0);
                    if h.fire.is_some() {
                        hunter_next(&mut h, HState::FetchSpearBack);
                    } else {
                        hunter_abort(&mut h, &mut acts, &tf, "The fire would not catch.");
                    }
                }
            }

            HState::FetchSpearBack => {
                h.t += dt;
                let free = gear
                    .spear
                    .and_then(|s| world.spears.get(s).ok())
                    .map_or(false, |s| !s.equipped);
                match hunter_fetch(&mut h, &mut tf, dt, gear.spear, Held::Spear, free, &world.globals, &mut acts) {
                    Fetch::Got => hunter_next(&mut h, HState::GoFire),
                    Fetch::Blocked => {
                        h.msg = "Waiting for the spear...";
                        if h.t > 45.0 {
                            hunter_abort(&mut h, &mut acts, &tf, "I can't get my spear back.");
                        }
                    }
                    Fetch::Walking => {
                        if h.t > 150.0 {
                            hunter_abort(&mut h, &mut acts, &tf, "My spear is too far.");
                        }
                    }
                }
            }

            HState::GoFire | HState::Cook | HState::Eat => {
                h.t += dt;
                // the nearest fire that is still burning
                let mut best: Option<(Entity, Vec3, f32)> = None;
                for (fe, f) in world.fires.iter() {
                    if f.fuel <= 0.05 {
                        continue;
                    }
                    if let Ok(g) = world.globals.get(fe) {
                        let fp = g.translation();
                        let d = Vec2::new(fp.x - pos.x, fp.z - pos.z).length();
                        if best.map_or(true, |b| d < b.2) {
                            best = Some((fe, fp, d));
                        }
                    }
                }
                let Some((fe, fp, _)) = best else {
                    hunter_abort(&mut h, &mut acts, &tf, "The fire went out.");
                    continue;
                };
                h.fire = Some(fe);
                let fish = gear
                    .spear
                    .and_then(|s| world.spears.get(s).ok())
                    .and_then(|s| s.impaled_fish);
                let Some(fish) = fish else {
                    hunter_abort(&mut h, &mut acts, &tf, "My fish is gone.");
                    continue;
                };
                h.aim = Some(fp + Vec3::Y * 0.42);
                if state == HState::GoFire {
                    let d = Vec3::new(pos.x - fp.x, 0.0, pos.z - fp.z).normalize_or_zero();
                    let stand = Vec3::new(fp.x, 0.0, fp.z) + d * 1.7;
                    if hunter_walk(&mut tf, stand, dt, HUNTER_SPEED, 0.3) {
                        hunter_next(&mut h, HState::Cook);
                    } else if h.t > 150.0 {
                        hunter_abort(&mut h, &mut acts, &tf, "The fire is too far.");
                    }
                } else if state == HState::Cook {
                    hunter_face(&mut tf, fp);
                    acts.list.push(HAct::CookStep { fish, dt });
                    let cooked = world.fish.get(fish).map_or(0.0, |(_, f)| f.cooked);
                    if cooked >= 1.0 {
                        hunter_next(&mut h, HState::Eat);
                    } else if h.t > 40.0 {
                        hunter_abort(&mut h, &mut acts, &tf, "This fish will not cook.");
                    }
                } else {
                    hunter_face(&mut tf, fp);
                    if h.t >= 1.8 && !h.hit_done {
                        h.hit_done = true;
                        if let Some(spear) = gear.spear {
                            acts.list.push(HAct::Eat { fish, spear });
                        }
                        h.hunger = 0.0;
                        if let Some((sp, _)) = h.holding.take() {
                            acts.list.push(HAct::Drop {
                                entity: sp,
                                pos: hunter_feet(&tf),
                                rot: Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
                            });
                        }
                        h.cooldown = 10.0;
                        h.state = HState::Idle;
                        h.t = 0.0;
                        h.target = None;
                        h.msg = "That was a good meal.";
                    }
                }
            }
        }
    }
}

/// Carries out what the brain asked for: picking up and putting down things, stabbing the
/// fish, peeling bark, carving the notch, making the burned dust, lighting and cooking.
fn hunter_actions_system(
    mut commands: Commands,
    time: Res<Time>,
    mut acts: ResMut<HunterActs>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
    fire_assets: Res<FireAssets>,
    globals: Query<&GlobalTransform>,
    mut transforms: Query<&mut Transform>,
    mut spears: Query<&mut Spear>,
    mut fish: Query<&mut Fish>,
    mut trunks: Query<(&mut ChoppableTrunk, &GlobalTransform)>,
    chips: Query<&Handle<Mesh>, (Or<(With<WoodDebris>, With<TreeChip>)>, Without<KnifeNotched>)>,
    bark: Query<(Entity, &GlobalTransform), With<BarkStrip>>,
    mut fires: Query<(&GlobalTransform, &mut Fire)>,
    mut peel: Local<Option<(Entity, i32, i32, u32)>>,
) {
    let list = std::mem::take(&mut acts.list);
    for act in list {
        match act {
            HAct::PickUp(e) => {
                commands
                    .entity(e)
                    .remove::<Velocity>()
                    .insert((RigidBody::KinematicPositionBased, HeldByHunter));
            }
            HAct::Drop { entity, pos, rot } => {
                if let Ok(mut t) = transforms.get_mut(entity) {
                    t.translation = pos;
                    t.rotation = rot;
                }
                commands.entity(entity).remove::<HeldByHunter>().insert((
                    RigidBody::Dynamic,
                    Velocity::default(),
                    Sleeping::disabled(),
                ));
            }
            HAct::Impale { spear, fish: fish_e } => {
                if let (Ok(mut f), Ok(mut tf)) = (fish.get_mut(fish_e), transforms.get_mut(fish_e)) {
                    f.impaled = true;
                    attach_fish_to_spear(&mut commands, spear, fish_e, &mut tf);
                    if let Ok(mut sp) = spears.get_mut(spear) {
                        sp.impaled_fish = Some(fish_e);
                    }
                }
            }
            HAct::Peel { trunk, from } => {
                if let Ok((mut tr, gt)) = trunks.get_mut(trunk) {
                    let (sector, row0, n) = match *peel {
                        Some((e, s, r, n)) if e == trunk => (s, r, n),
                        _ => {
                            let local = gt.affine().inverse().transform_point3(from);
                            let (s, r) = tr.wood.locate_free(local - tr.offset);
                            (s, r, 0)
                        }
                    };
                    // work outwards from the row next to him: 0, +1, -1, +2, -2, ...
                    let step = ((n + 1) / 2) as i32;
                    let row = if n % 2 == 1 { row0 + step } else { row0 - step };
                    let rows = tr.wood.rows as i32;
                    let row = row.clamp(0, rows - 1);
                    let _ = strip_bark_row(&mut tr, gt, sector, row, &mut commands, &mut meshes, &mats);
                    *peel = Some((trunk, sector, row0, n + 1));
                }
            }
            HAct::Carve { chip } => {
                let done = match (chips.get(chip), globals.get(chip)) {
                    (Ok(handle), Ok(gt)) => {
                        let handle = handle.clone();
                        carve_chip_for_hunter(&mut commands, &mut meshes, chip, &handle, gt, Vec3::Y)
                    }
                    _ => false,
                };
                if !done {
                    commands.entity(chip).insert(HunterSkip);
                }
            }
            HAct::Pile { pos } => {
                let seed = pos.x.to_bits() ^ pos.z.to_bits().rotate_left(7);
                let _ = spawn_burned_pile(&mut commands, &fire_assets, pos, seed, time.elapsed_seconds());
            }
            HAct::Fire { pos } => {
                // the same thing that happens when bark is fed to the embers by hand
                let mut count = 0u32;
                for (be, gt) in &bark {
                    if gt.translation().distance(pos) < 2.5 {
                        commands.entity(be).despawn_recursive();
                        count += 1;
                    }
                }
                let fuel_added = 0.65 * count.max(1) as f32;
                let mut boosted = false;
                for (gt, mut f) in &mut fires {
                    if gt.translation().distance(pos) < 3.5 {
                        f.fuel = (f.fuel + fuel_added).min(FIRE_MAX_FUEL);
                        f.flare = 1.0;
                        boosted = true;
                        break;
                    }
                }
                if !boosted {
                    let seed = pos.x.to_bits() ^ pos.z.to_bits().rotate_left(13);
                    spawn_fire(&mut commands, &fire_assets, pos + Vec3::Y * 0.02, 1.2 + fuel_added, seed);
                }
            }
            HAct::CookStep { fish: fish_e, dt } => {
                if let Ok(mut f) = fish.get_mut(fish_e) {
                    f.cooked = (f.cooked + dt / FISH_COOK_SECONDS).min(1.0);
                    paint_fish(&f, &mut mats);
                }
            }
            HAct::Eat { fish: fish_e, spear } => {
                commands.entity(fish_e).despawn_recursive();
                if let Ok(mut sp) = spears.get_mut(spear) {
                    sp.impaled_fish = None;
                }
            }
        }
    }
}

/// The same V-notch cut the knife makes when you click a wood chip (`knife_carve_system`).
fn carve_chip_for_hunter(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    chip: Entity,
    mesh_handle: &Handle<Mesh>,
    chip_gt: &GlobalTransform,
    normal: Vec3,
) -> bool {
    let Some(mesh) = meshes.get(mesh_handle) else {
        return false;
    };
    let Some(positions) = mesh
        .attribute(Mesh::ATTRIBUTE_POSITION)
        .and_then(|a| a.as_float3())
    else {
        return false;
    };
    let normals = mesh
        .attribute(Mesh::ATTRIBUTE_NORMAL)
        .and_then(|a| a.as_float3());
    let colors = match mesh.attribute(Mesh::ATTRIBUTE_COLOR) {
        Some(bevy::render::mesh::VertexAttributeValues::Float32x4(values)) => {
            Some(values.as_slice())
        }
        _ => None,
    };
    let (_, rotation, _) = chip_gt.to_scale_rotation_translation();
    let face_dir = rotation.inverse() * normal;
    let Some(cut) = notch_chip(positions, normals, colors, face_dir) else {
        return false;
    };
    let shapes: Vec<(Vec3, Quat, Collider)> = cut
        .halves
        .iter()
        .filter_map(|points| Collider::convex_hull(points))
        .map(|hull| (Vec3::ZERO, Quat::IDENTITY, hull))
        .collect();
    if shapes.is_empty() {
        return false;
    }
    let mut notched = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    notched.insert_attribute(Mesh::ATTRIBUTE_POSITION, cut.positions);
    notched.insert_attribute(Mesh::ATTRIBUTE_NORMAL, cut.normals);
    notched.insert_attribute(Mesh::ATTRIBUTE_UV_0, cut.uvs);
    notched.insert_attribute(Mesh::ATTRIBUTE_COLOR, cut.colors);
    notched.insert_indices(bevy::render::mesh::Indices::U32(cut.indices));
    let new_mesh = meshes.add(notched);
    commands
        .entity(chip)
        .insert((new_mesh, Collider::compound(shapes), KnifeNotched))
        .remove::<TreeChip>()
        .remove::<bevy::render::primitives::Aabb>();
    true
}

/// Poses whatever Tom carries in his hand, and while he works, swings it at what he works on.
fn hunter_pose_system(
    hunters: Query<(&Transform, &Hunter)>,
    mut tools: Query<&mut Transform, (With<HeldByHunter>, Without<Hunter>)>,
) {
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t.clamp(0.0, 1.0);
    for (tf, h) in &hunters {
        let Some((held, kind)) = h.holding else {
            continue;
        };
        let Ok(mut t) = tools.get_mut(held) else {
            continue;
        };
        let fwd = tf.rotation * Vec3::NEG_Z;
        let right = tf.rotation * Vec3::X;
        let hand = tf.translation + fwd * 0.32 + right * 0.26 + Vec3::Y * 0.08;
        let aim_flat = h
            .aim
            .map(|a| Vec3::new(a.x - tf.translation.x, 0.0, a.z - tf.translation.z).normalize_or_zero())
            .filter(|v| *v != Vec3::ZERO)
            .unwrap_or(fwd);

        let (pos, rot) = match kind {
            Held::Spear => {
                let (dir, pos) = match h.state {
                    HState::Stab => {
                        let target = h.aim.unwrap_or(hand + fwd * 2.0);
                        let d = (target - hand).normalize_or_zero();
                        let dir = if d == Vec3::ZERO { fwd } else { d };
                        let p = (h.t * std::f32::consts::PI / 0.9).sin().max(0.0);
                        (dir, hand - dir * 0.5 + dir * (1.1 * p))
                    }
                    HState::Cook => {
                        let target = h.aim.unwrap_or(hand + fwd * 1.5);
                        let d = (target - hand).normalize_or_zero();
                        let dir = if d == Vec3::ZERO { fwd } else { d };
                        (dir, target - dir * SPEAR_FISH_Y)
                    }
                    _ => {
                        let dir = (fwd * 0.8 + Vec3::Y * 0.55).normalize();
                        (dir, hand - dir * 0.5)
                    }
                };
                (pos, hunter_orient(dir, right))
            }
            Held::Axe => {
                if h.state == HState::Chop {
                    // A vertical, overhead chop: the axe goes up over his head, then comes
                    // straight down in front of him into the trunk.
                    let smooth = |t: f32| {
                        let t = t.clamp(0.0, 1.0);
                        t * t * (3.0 - 2.0 * t)
                    };
                    let ph = h.swing_t.clamp(0.0, 1.0);
                    // handle angle above the horizontal, towards the tree
                    let a = if ph < 0.55 {
                        lerp(-0.2, 1.75, smooth(ph / 0.55))
                    } else if ph < 0.67 {
                        lerp(1.75, -0.55, (ph - 0.55) / 0.12)
                    } else {
                        lerp(-0.55, -0.2, smooth((ph - 0.67) / 0.33))
                    };
                    let u = (a / 1.75).clamp(0.0, 1.0); // 1 = overhead, 0 = at the strike
                    let hand = tf.translation
                        + aim_flat * lerp(0.62, 0.12, u)
                        + right * 0.2
                        + Vec3::Y * lerp(-0.02, 0.6, u);
                    let dir = aim_flat * a.cos() + Vec3::Y * a.sin();
                    // the head is wide along its local X: keep that across the swing
                    (hand - dir * 0.2, hunter_orient_across(dir, right))
                } else {
                    let dir = (fwd * 0.5 + Vec3::Y * 0.85).normalize();
                    (hand - dir * 0.2, hunter_orient_across(dir, right))
                }
            }
            Held::Machete => {
                let (dir, pos) = if h.state == HState::Peel {
                    let s = (h.t * 6.0).sin();
                    let dir = (aim_flat * 0.8 - Vec3::Y * 0.35 + right * s * 0.5).normalize();
                    (dir, hand + aim_flat * 0.15)
                } else {
                    ((fwd * 0.7 - Vec3::Y * 0.5).normalize(), hand)
                };
                (pos, hunter_orient(dir, right))
            }
            Held::Knife => {
                let (dir, pos) = if h.state == HState::Carve {
                    let s = (h.t * 10.0).sin();
                    let dir = (aim_flat * 0.6 - Vec3::Y * 0.7 + right * s * 0.25).normalize();
                    (dir, hand + aim_flat * 0.2 - Vec3::Y * 0.1)
                } else {
                    ((fwd * 0.5 - Vec3::Y * 0.8).normalize(), hand)
                };
                (pos, hunter_orient(dir, right))
            }
            Held::Stick => {
                if h.state == HState::Rub {
                    let target = h.aim.unwrap_or(hand + fwd * 0.5);
                    let d = (target - hand).normalize_or_zero();
                    let dir = if d == Vec3::ZERO { fwd } else { d };
                    let pos = target + Vec3::Y * 0.04 - dir * STICK_HALF + right * ((h.t * 14.0).sin() * 0.09);
                    (pos, hunter_orient(dir, right))
                } else {
                    let dir = (fwd * 0.9 + Vec3::Y * 0.4).normalize();
                    (hand, hunter_orient(dir, right))
                }
            }
            Held::Bark => (hand + Vec3::Y * 0.1 + fwd * 0.1, tf.rotation),
        };
        t.translation = pos;
        t.rotation = rot;
    }
}

/// The hunger meter, and what Tom says when you click him.
fn hunter_ui_system(
    hunters: Query<&Hunter>,
    mut npcs: Query<(&mut Npc, &Hunter)>,
    mut texts: Query<&mut Text, With<HungerText>>,
) {
    let Ok(h) = hunters.get_single() else {
        return;
    };
    let pct = (h.hunger * 100.0).round() as i32;
    let doing = if h.state == HState::Idle {
        if h.hunger >= HUNT_AT {
            "starving"
        } else if h.hunger >= 0.45 {
            "getting hungry"
        } else {
            "full"
        }
    } else {
        "feeding himself"
    };
    for mut t in &mut texts {
        t.sections[0].value = format!(
            "Tom - hunger {pct}% ({doing})  [H: make him hungry]\n{}",
            h.msg
        );
    }
    for (mut npc, hu) in &mut npcs {
        let line = if hu.state != HState::Idle {
            hu.msg.to_string()
        } else if hu.msg.is_empty() {
            if hu.hunger >= HUNT_AT {
                "I'm starving! I need to find food.".to_string()
            } else if hu.hunger >= 0.45 {
                "I'm getting hungry...".to_string()
            } else {
                "I'm doing fine for now.".to_string()
            }
        } else {
            hu.msg.to_string()
        };
        if npc.lines.len() != 1 || npc.lines[0] != line {
            npc.lines = vec![line];
            npc.index = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Possession: look at an animal or a person and press F to become it. You see through its
// eyes and walk (or fly, or swim) it around with WASD; F again returns you to your own body.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum PossessMode {
    Walk,
    Fly,
    Swim,
}

#[derive(Clone, Copy)]
struct PossessKind {
    mode: PossessMode,
    /// eye height above the creature's origin
    eye: f32,
    speed: f32,
    /// how big it looks when you aim at it
    radius: f32,
    /// how far away you can possess it
    reach: f32,
    /// height of its middle above its origin
    centre: f32,
}

const PK_PERSON: PossessKind = PossessKind { mode: PossessMode::Walk, eye: 0.45, speed: 3.0, radius: 0.75, reach: 14.0, centre: 0.0 };
const PK_CAT: PossessKind = PossessKind { mode: PossessMode::Walk, eye: 0.28, speed: 2.6, radius: 0.55, reach: 12.0, centre: 0.22 };
const PK_DOG: PossessKind = PossessKind { mode: PossessMode::Walk, eye: 0.38, speed: 3.2, radius: 0.6, reach: 12.0, centre: 0.28 };
const PK_HORSE: PossessKind = PossessKind { mode: PossessMode::Walk, eye: 1.9, speed: 6.0, radius: 1.5, reach: 22.0, centre: 1.0 };
const PK_BIRD: PossessKind = PossessKind { mode: PossessMode::Fly, eye: 0.05, speed: 7.0, radius: 0.9, reach: 40.0, centre: 0.0 };
const PK_HOPPER: PossessKind = PossessKind { mode: PossessMode::Walk, eye: 0.08, speed: 1.3, radius: 0.35, reach: 8.0, centre: 0.05 };
const PK_MANTIS: PossessKind = PossessKind { mode: PossessMode::Walk, eye: 0.28, speed: 1.1, radius: 0.5, reach: 10.0, centre: 0.2 };
const PK_FISH: PossessKind = PossessKind { mode: PossessMode::Swim, eye: 0.05, speed: 2.6, radius: 0.45, reach: 14.0, centre: 0.0 };

/// Who you are possessing, if anyone. While you do, your own body stays where it was.
#[derive(Resource, Default)]
struct Possession {
    target: Option<Entity>,
    kind: Option<PossessKind>,
    name: String,
    /// walkers keep this height above the ground; flyers and swimmers keep their altitude
    height: f32,
    /// where the creature is (the creature itself is moved there after the other systems ran)
    pos: Vec3,
}

#[derive(SystemParam)]
struct PossessCandidates<'w, 's> {
    people: Query<'w, 's, (Entity, &'static GlobalTransform, &'static Npc)>,
    pets: Query<'w, 's, (Entity, &'static GlobalTransform, &'static Pet)>,
    horses: Query<'w, 's, (Entity, &'static GlobalTransform), With<Horse>>,
    birds: Query<'w, 's, (Entity, &'static GlobalTransform), With<Bird>>,
    hoppers: Query<'w, 's, (Entity, &'static GlobalTransform), With<Grasshopper>>,
    mantises: Query<'w, 's, (Entity, &'static GlobalTransform), With<LeafMantis>>,
    fish: Query<'w, 's, (Entity, &'static GlobalTransform, &'static Fish)>,
}

fn possess_ground(p: Vec3) -> f32 {
    if hunter_inside(p) {
        F1_FLOOR_Y
    } else {
        land_surface_height(p.x, p.z)
    }
}

/// The walls of the house (a square ring), with the doorway in the +X wall when the door is open.
fn possess_blocked(p: Vec3, door_open: bool) -> bool {
    if p.y > 6.5 {
        return false; // over the roof
    }
    let m = p.x.abs().max(p.z.abs());
    if m < 4.05 || m > 4.35 {
        return false;
    }
    !(door_open && p.x > 0.0 && p.z.abs() < 0.5)
}

/// Switches the colliders of a creature and everything on it off (so your own eyes are not
/// inside its body when you aim) or back on.
fn possess_set_colliders(
    commands: &mut Commands,
    root: Entity,
    disabled: bool,
    children: &Query<&Children>,
    colliders: &Query<(), With<Collider>>,
) {
    let mut stack = vec![root];
    while let Some(e) = stack.pop() {
        if colliders.get(e).is_ok() {
            if disabled {
                commands.entity(e).insert(ColliderDisabled);
            } else {
                commands.entity(e).remove::<ColliderDisabled>();
            }
        }
        if let Ok(kids) = children.get(e) {
            stack.extend(kids.iter().copied());
        }
    }
}

/// F: possess what you are looking at (or go back to your own body). While possessed, WASD
/// moves the creature (Space / Shift fly up / down) and the mouse looks, exactly as for you.
fn possession_control_system(
    mut commands: Commands,
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    door: Res<Door>,
    rapier: Res<RapierContext>,
    mut poss: ResMut<Possession>,
    mut status: ResMut<Status>,
    grab: Res<Grab>,
    candidates: PossessCandidates,
    mut players: Query<(Entity, &mut Player)>,
    things: Query<&Transform, Without<Camera3d>>,
    cams: Query<&Transform, With<Camera3d>>,
    children: Query<&Children>,
    colliders: Query<(), With<Collider>>,
    mut arms: Query<&mut Visibility, With<FpArm>>,
) {
    let Ok((player_e, mut p)) = players.get_single_mut() else {
        return;
    };
    let Ok(cam_tf) = cams.get_single() else {
        return;
    };
    let dt = time.delta_seconds().min(0.05);

    // the creature disappeared (eaten, despawned ...): you are yourself again
    if let Some(target) = poss.target {
        if things.get(target).is_err() {
            poss.target = None;
            poss.kind = None;
            for mut v in &mut arms {
                *v = Visibility::Inherited;
            }
            status.0 = "It is gone. You are yourself again.".into();
        }
    }

    if keys.just_pressed(KeyCode::KeyF) {
        if let Some(target) = poss.target {
            // back to your own body
            possess_set_colliders(&mut commands, target, false, &children, &colliders);
            poss.target = None;
            poss.kind = None;
            for mut v in &mut arms {
                *v = Visibility::Inherited;
            }
            status.0 = "You are yourself again.".into();
        } else if p.mounted.is_some() {
            status.0 = "Get off the horse first (E).".into();
        } else if grab.entity.is_some() {
            status.0 = "Let go of what you are holding first.".into();
        } else {
            // what are you looking at?
            let origin = cam_tf.translation;
            let dir: Vec3 = *cam_tf.forward();
            let mut best: Option<(f32, Entity, PossessKind, String)> = None;
            {
                let mut consider = |e: Entity, gt: &GlobalTransform, kind: PossessKind, name: String| {
                    let centre = gt.translation() + Vec3::Y * kind.centre;
                    let to = centre - origin;
                    let along = to.dot(dir);
                    if along < 0.3 || along > kind.reach {
                        return;
                    }
                    if (to - dir * along).length() > kind.radius {
                        return;
                    }
                    if best.as_ref().map_or(false, |b| b.0 <= along) {
                        return;
                    }
                    // something solid in the way (a wall, a tree)?
                    let not_it = |c: Entity| c != e && c != player_e;
                    let filter = QueryFilter::default().exclude_sensors().predicate(&not_it);
                    if let Some((_, toi)) = rapier.cast_ray(origin, dir, along, true, filter) {
                        if toi < along - kind.radius - 0.6 {
                            return;
                        }
                    }
                    best = Some((along, e, kind, name));
                };
                for (e, gt, npc) in &candidates.people {
                    consider(e, gt, PK_PERSON, npc.name.clone());
                }
                for (e, gt, pet) in &candidates.pets {
                    match pet.kind {
                        PetKind::Cat => consider(e, gt, PK_CAT, "a cat".into()),
                        PetKind::Dog => consider(e, gt, PK_DOG, "a dog".into()),
                    }
                }
                for (e, gt) in &candidates.horses {
                    consider(e, gt, PK_HORSE, "the horse".into());
                }
                for (e, gt) in &candidates.birds {
                    consider(e, gt, PK_BIRD, "a bird".into());
                }
                for (e, gt) in &candidates.hoppers {
                    consider(e, gt, PK_HOPPER, "a grasshopper".into());
                }
                for (e, gt) in &candidates.mantises {
                    consider(e, gt, PK_MANTIS, "a leaf mantis".into());
                }
                for (e, gt, fish) in &candidates.fish {
                    if !fish.impaled && !fish.dropped {
                        consider(e, gt, PK_FISH, "a fish".into());
                    }
                }
            }
            match best {
                None => {
                    status.0 = "Nothing to possess: look straight at an animal or a person and press F.".into();
                }
                Some((_, e, kind, name)) => {
                    if let Ok(t) = things.get(e) {
                        let fwd = t.rotation * Vec3::NEG_Z;
                        p.yaw = (-fwd.x).atan2(-fwd.z);
                        p.pitch = 0.0;
                        poss.pos = t.translation;
                        poss.height = if kind.mode == PossessMode::Walk {
                            t.translation.y - possess_ground(t.translation)
                        } else {
                            t.translation.y
                        };
                    }
                    possess_set_colliders(&mut commands, e, true, &children, &colliders);
                    for mut v in &mut arms {
                        *v = Visibility::Hidden;
                    }
                    poss.target = Some(e);
                    poss.kind = Some(kind);
                    status.0 = format!(
                        "You are now {name}. WASD moves{}, hold the right mouse button to look, F returns to your body.",
                        if kind.mode == PossessMode::Fly { " (Space up, Shift down)" } else { "" }
                    );
                    poss.name = name;
                }
            }
        }
    }

    // --- moving the creature ---
    let (Some(_), Some(kind)) = (poss.target, poss.kind) else {
        return;
    };
    let (sin, cos) = p.yaw.sin_cos();
    let fwd = Vec3::new(-sin, 0.0, -cos);
    let right = Vec3::new(cos, 0.0, -sin);
    let mut dir = Vec3::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        dir += fwd;
    }
    if keys.pressed(KeyCode::KeyS) {
        dir -= fwd;
    }
    if keys.pressed(KeyCode::KeyD) {
        dir += right;
    }
    if keys.pressed(KeyCode::KeyA) {
        dir -= right;
    }
    let step = dir.normalize_or_zero() * kind.speed * dt;
    let from = poss.pos;
    let mut pos = from;
    // straight on, or (if a wall is in the way) sliding along it
    for candidate in [
        from + step,
        from + Vec3::new(step.x, 0.0, 0.0),
        from + Vec3::new(0.0, 0.0, step.z),
    ] {
        if !possess_blocked(candidate, door.open) {
            pos = candidate;
            break;
        }
    }
    match kind.mode {
        PossessMode::Walk => {
            pos.y = possess_ground(pos) + poss.height;
        }
        PossessMode::Fly => {
            let up = (keys.pressed(KeyCode::Space) as i32 - keys.pressed(KeyCode::ShiftLeft) as i32) as f32;
            pos.y = (pos.y + up * 3.0 * dt).clamp(possess_ground(pos) + 0.4, 30.0);
        }
        PossessMode::Swim => {
            let centre = river_center_x(pos.z);
            let half = RIVER_WIDTH * 0.5 - 0.35;
            pos.x = pos.x.clamp(centre - half, centre + half);
            pos.y = poss.height;
        }
    }
    poss.pos = pos;
}

/// Puts the possessed creature where you moved it, after every system that moves creatures
/// (their own wandering) has run, so you are the one steering it.
fn possession_apply_system(
    poss: Res<Possession>,
    players: Query<&Player>,
    mut things: Query<&mut Transform, Without<Camera3d>>,
) {
    let (Some(target), Some(_)) = (poss.target, poss.kind) else {
        return;
    };
    let Ok(p) = players.get_single() else {
        return;
    };
    if let Ok(mut t) = things.get_mut(target) {
        t.translation = poss.pos;
        t.rotation = Quat::from_rotation_y(p.yaw);
    }
}

// ---------------------------------------------------------------------------
// Image -> 3D mesh
// ---------------------------------------------------------------------------

fn open_image(path: &str) -> Option<RgbaImage> {
    match image::open(path) {
        Ok(img) => Some(
            img.resize(MAX_CELLS, MAX_CELLS, FilterType::Triangle)
                .to_rgba8(),
        ),
        Err(e) => {
            eprintln!("Could not load '{path}': {e}");
            None
        }
    }
}

fn load_image(path: &Option<String>) -> RgbaImage {
    path.as_deref()
        .and_then(open_image)
        .unwrap_or_else(default_image)
}

/// A procedural smiley so the program works with no input file.
fn default_image() -> RgbaImage {
    let n = 48u32;
    RgbaImage::from_fn(n, n, |x, y| {
        let (fx, fy) = (
            x as f32 / n as f32 * 2.0 - 1.0,
            1.0 - y as f32 / n as f32 * 2.0,
        );
        let r = (fx * fx + fy * fy).sqrt();
        if r > 0.95 {
            return Rgba([0, 0, 0, 0]); // transparent outside
        }
        let eye = |cx: f32| ((fx - cx).powi(2) + (fy - 0.3).powi(2)).sqrt() < 0.14;
        let mouth = fy < -0.2 && fy > -0.6 && (r > 0.55 && r < 0.72);
        if eye(-0.35) || eye(0.35) || mouth {
            Rgba([40, 30, 20, 255])
        } else if r > 0.85 {
            Rgba([220, 150, 20, 255])
        } else {
            Rgba([255, 214, 51, 255])
        }
    })
}

fn push_quad(pos: &mut Vec<[f32; 3]>, col: &mut Vec<[f32; 4]>, q: [[f32; 3]; 4], c: [f32; 4]) {
    for i in [0, 1, 2, 0, 2, 3] {
        pos.push(q[i]);
        col.push(c);
    }
}

/// Every pixel becomes a column; height = brightness. Transparent pixels are skipped.
fn relief_mesh(img: &RgbaImage, cell: f32) -> (Mesh, Vec<Vec3>) {
    let (cols, rows) = (img.width() as i32, img.height() as i32);
    let height_at = |x: i32, y: i32| -> Option<f32> {
        if x < 0 || y < 0 || x >= cols || y >= rows {
            return None;
        }
        let p = img.get_pixel(x as u32, y as u32).0;
        if p[3] < 128 {
            return None;
        }
        let lum = (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) / 255.0;
        Some(OBJECT_BASE + OBJECT_DEPTH * lum)
    };

    let (w, h) = (cols as f32 * cell, rows as f32 * cell);
    let zc = (OBJECT_BASE + OBJECT_DEPTH) / 2.0; // center the mesh on its origin
    let mut pos = Vec::new();
    let mut col = Vec::new();

    for j in 0..rows {
        for i in 0..cols {
            let Some(ht) = height_at(i, j) else { continue };
            let p = img.get_pixel(i as u32, j as u32).0;
            let c = Color::srgb_u8(p[0], p[1], p[2]).to_linear().to_f32_array();
            let dark = [c[0] * 0.6, c[1] * 0.6, c[2] * 0.6, 1.0]; // sides a bit darker

            let x0 = i as f32 * cell - w / 2.0;
            let x1 = x0 + cell;
            let y0 = (rows - j - 1) as f32 * cell - h / 2.0;
            let y1 = y0 + cell;
            let z = |v: f32| v - zc;

            // front (+Z) and back (-Z)
            push_quad(
                &mut pos,
                &mut col,
                [
                    [x0, y0, z(ht)],
                    [x1, y0, z(ht)],
                    [x1, y1, z(ht)],
                    [x0, y1, z(ht)],
                ],
                c,
            );
            push_quad(
                &mut pos,
                &mut col,
                [
                    [x0, y0, z(0.0)],
                    [x0, y1, z(0.0)],
                    [x1, y1, z(0.0)],
                    [x1, y0, z(0.0)],
                ],
                dark,
            );

            // side walls only where the neighbour is lower / missing
            let nh = |dx: i32, dy: i32| height_at(i + dx, j + dy).unwrap_or(0.0);
            let n = nh(1, 0);
            if ht > n {
                push_quad(
                    &mut pos,
                    &mut col,
                    [
                        [x1, y0, z(n)],
                        [x1, y1, z(n)],
                        [x1, y1, z(ht)],
                        [x1, y0, z(ht)],
                    ],
                    dark,
                );
            }
            let n = nh(-1, 0);
            if ht > n {
                push_quad(
                    &mut pos,
                    &mut col,
                    [
                        [x0, y0, z(n)],
                        [x0, y0, z(ht)],
                        [x0, y1, z(ht)],
                        [x0, y1, z(n)],
                    ],
                    dark,
                );
            }
            let n = nh(0, -1); // row above = +Y
            if ht > n {
                push_quad(
                    &mut pos,
                    &mut col,
                    [
                        [x1, y1, z(n)],
                        [x0, y1, z(n)],
                        [x0, y1, z(ht)],
                        [x1, y1, z(ht)],
                    ],
                    dark,
                );
            }
            let n = nh(0, 1); // row below = -Y
            if ht > n {
                push_quad(
                    &mut pos,
                    &mut col,
                    [
                        [x0, y0, z(n)],
                        [x1, y0, z(n)],
                        [x1, y0, z(ht)],
                        [x0, y0, z(ht)],
                    ],
                    dark,
                );
            }
        }
    }

    let points: Vec<Vec3> = pos.iter().map(|p| Vec3::from(*p)).collect();
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, pos);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, col);
    mesh.compute_flat_normals();
    (mesh, points)
}

// ---------------------------------------------------------------------------
// Scene: house + object
// ---------------------------------------------------------------------------

fn static_box(
    commands: &mut Commands,
    batch: &mut StaticBoxBatch,
    pos: Vec3,
    size: Vec3,
    color: Color,
) {
    batch.add_box(pos, size, color);
    commands.spawn((
        RigidBody::Fixed,
        Collider::cuboid(size.x / 2.0, size.y / 2.0, size.z / 2.0),
        TransformBundle::from_transform(Transform::from_translation(pos)),
    ));
}

#[derive(Default)]
struct StaticBoxBatch {
    groups: Vec<StaticBoxGroup>,
}

struct StaticBoxGroup {
    color_key: [u32; 4],
    color: Color,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
}

impl StaticBoxBatch {
    fn add_box(&mut self, center: Vec3, size: Vec3, color: Color) {
        let color_key = color.to_linear().to_f32_array().map(f32::to_bits);
        let group_index = self
            .groups
            .iter()
            .position(|group| group.color_key == color_key)
            .unwrap_or_else(|| {
                self.groups.push(StaticBoxGroup {
                    color_key,
                    color,
                    positions: Vec::new(),
                    normals: Vec::new(),
                    uvs: Vec::new(),
                    indices: Vec::new(),
                });
                self.groups.len() - 1
            });
        let group = &mut self.groups[group_index];
        let half = size * 0.5;
        let faces = [
            (
                Vec3::X,
                [
                    Vec3::new(1.0, -1.0, -1.0),
                    Vec3::new(1.0, 1.0, -1.0),
                    Vec3::new(1.0, 1.0, 1.0),
                    Vec3::new(1.0, -1.0, 1.0),
                ],
            ),
            (
                Vec3::NEG_X,
                [
                    Vec3::new(-1.0, -1.0, 1.0),
                    Vec3::new(-1.0, 1.0, 1.0),
                    Vec3::new(-1.0, 1.0, -1.0),
                    Vec3::new(-1.0, -1.0, -1.0),
                ],
            ),
            (
                Vec3::Y,
                [
                    Vec3::new(-1.0, 1.0, 1.0),
                    Vec3::new(1.0, 1.0, 1.0),
                    Vec3::new(1.0, 1.0, -1.0),
                    Vec3::new(-1.0, 1.0, -1.0),
                ],
            ),
            (
                Vec3::NEG_Y,
                [
                    Vec3::new(-1.0, -1.0, -1.0),
                    Vec3::new(1.0, -1.0, -1.0),
                    Vec3::new(1.0, -1.0, 1.0),
                    Vec3::new(-1.0, -1.0, 1.0),
                ],
            ),
            (
                Vec3::Z,
                [
                    Vec3::new(-1.0, -1.0, 1.0),
                    Vec3::new(1.0, -1.0, 1.0),
                    Vec3::new(1.0, 1.0, 1.0),
                    Vec3::new(-1.0, 1.0, 1.0),
                ],
            ),
            (
                Vec3::NEG_Z,
                [
                    Vec3::new(1.0, -1.0, -1.0),
                    Vec3::new(-1.0, -1.0, -1.0),
                    Vec3::new(-1.0, 1.0, -1.0),
                    Vec3::new(1.0, 1.0, -1.0),
                ],
            ),
        ];
        const FACE_UVS: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

        for (normal, corners) in faces {
            let base = group.positions.len() as u32;
            for (corner, uv) in corners.into_iter().zip(FACE_UVS) {
                let point = center + corner * half;
                group.positions.push(point.to_array());
                group.normals.push(normal.to_array());
                group.uvs.push(uv);
            }
            group
                .indices
                .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }

    fn spawn(
        self,
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        mats: &mut Assets<StandardMaterial>,
    ) {
        for group in self.groups {
            let mut mesh = Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::default(),
            );
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, group.positions);
            mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, group.normals);
            mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, group.uvs);
            mesh.insert_indices(bevy::render::mesh::Indices::U32(group.indices));
            commands.spawn(PbrBundle {
                mesh: meshes.add(mesh),
                material: mats.add(group.color),
                ..default()
            });
        }
    }
}

fn near_mountain(x: f32, z: f32) -> bool {
    let point = Vec2::new(x, z);
    [
        (Vec2::new(18.0, -35.0), 26.0),
        (Vec2::new(0.0, -33.0), 17.0),
        (Vec2::new(36.0, -34.0), 18.0),
    ]
    .iter()
    .any(|(center, radius)| point.distance(*center) < *radius)
}

fn riverbed_height(x: f32, z: f32) -> f32 {
    let edge_t = (x - river_center_x(z)).abs() / (RIVER_WIDTH * 0.5);
    let edge_t = edge_t.clamp(0.0, 1.0);
    let edge = edge_t * edge_t * (3.0 - 2.0 * edge_t);
    land_surface_height(x, z) - RIVER_DEPTH * (1.0 - edge)
}

/// 0..1: how muddy the ground is at (x, z). 1 in the river and at its edge, fading out
/// over MUD_BAND metres with a wobbly outline. Used for both the colour and the grabbing.
fn mud_strength(x: f32, z: f32) -> f32 {
    let d = (x - river_center_x(z)).abs() - RIVER_WIDTH * 0.5;
    if d <= 0.0 {
        return 1.0;
    }
    let wobble = 0.72 + 0.2 * (z * 0.9 + 1.3).sin() + 0.1 * (z * 2.3).sin();
    let t = (1.0 - d / (MUD_BAND * wobble)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn terrain_normal(x: f32, z: f32) -> Vec3 {
    let e = 0.25;
    let dx = land_surface_height(x + e, z) - land_surface_height(x - e, z);
    let dz = land_surface_height(x, z + e) - land_surface_height(x, z - e);
    Vec3::new(-dx, 2.0 * e, -dz).normalize()
}

fn ground_geometry(sites: &[(Vec2, f32)]) -> (Mesh, Vec<Vec3>, Vec<[u32; 3]>) {
    let cells = (GROUND_HALF_SIZE * 2.0 / GROUND_CELL_SIZE) as u32;
    let row = cells + 1;
    let mut vertices = Vec::with_capacity((row * row) as usize);
    let mut positions = Vec::with_capacity((row * row) as usize);
    let mut colors = Vec::with_capacity((row * row) as usize);

    for z in 0..=cells {
        for x in 0..=cells {
            let world_x = -GROUND_HALF_SIZE + x as f32 * GROUND_CELL_SIZE;
            let world_z = -GROUND_HALF_SIZE + z as f32 * GROUND_CELL_SIZE;
            let base_height = riverbed_height(world_x, world_z);
            let land_height = land_surface_height(world_x, world_z);
            let mut y = base_height;
            for (center, radius) in sites {
                let distance = Vec2::new(world_x, world_z).distance(*center);
                if distance < *radius {
                    let flat_radius = radius * 0.48;
                    let edge_t =
                        ((distance - flat_radius) / (radius - flat_radius)).clamp(0.0, 1.0);
                    let smooth = edge_t * edge_t * (3.0 - 2.0 * edge_t);
                    y = y.min(land_height - DIG_MAX_DEPTH * (1.0 - smooth));
                }
            }
            let point = Vec3::new(world_x, y, world_z);
            positions.push(point);
            vertices.push(point.to_array());
            let depth = ((land_height - y) / (RIVER_DEPTH + DIG_MAX_DEPTH)).clamp(0.0, 1.0);
            let mut color = Color::srgb(1.0 - depth * 0.5, 1.0 - depth * 0.62, 1.0 - depth * 0.76)
                .to_linear()
                .to_f32_array();
            // wet brown mud along the river (the ground material is green, so push it brown)
            let mud = mud_strength(world_x, world_z);
            color[0] *= 1.0 + mud * 1.1;
            color[1] *= 1.0 - mud * 0.28;
            color[2] *= 1.0 - mud * 0.4;
            colors.push(color);
        }
    }

    let mut triangles = Vec::with_capacity((cells * cells * 2) as usize);
    for z in 0..cells {
        for x in 0..cells {
            let p00 = z * row + x;
            let p10 = p00 + 1;
            let p01 = p00 + row;
            let p11 = p01 + 1;
            triangles.push([p00, p01, p10]);
            triangles.push([p10, p01, p11]);
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    let flat_indices: Vec<u32> = triangles
        .iter()
        .flat_map(|triangle| triangle.iter().copied())
        .collect();
    mesh.insert_indices(bevy::render::mesh::Indices::U32(flat_indices));
    let mut normals = Vec::with_capacity(positions.len());
    for z in 0..=cells {
        for x in 0..=cells {
            let index = (z * row + x) as usize;
            let center_y = positions[index].y;
            let left_y = if x > 0 {
                positions[index - 1].y
            } else {
                center_y
            };
            let right_y = if x < cells {
                positions[index + 1].y
            } else {
                center_y
            };
            let back_y = if z > 0 {
                positions[index - row as usize].y
            } else {
                center_y
            };
            let front_y = if z < cells {
                positions[index + row as usize].y
            } else {
                center_y
            };
            let normal =
                Vec3::new(left_y - right_y, GROUND_CELL_SIZE * 2.0, back_y - front_y).normalize();
            normals.push(normal.to_array());
        }
    }
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    (mesh, positions, triangles)
}

fn spawn_diggable_ground(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &mut Assets<StandardMaterial>,
) {
    let (mesh, vertices, triangles) = ground_geometry(&[]);
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(mesh),
            material: mats.add(Color::srgb(0.16, 0.2, 0.13)),
            ..default()
        },
        RigidBody::Fixed,
        Collider::trimesh(vertices, triangles),
        Friction::coefficient(1.0),
        Restitution::coefficient(0.0),
        DiggableGround,
    ));
}

fn spawn_shoreline(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &mut Assets<StandardMaterial>,
) {
    let water = mats.add(StandardMaterial {
        base_color: Color::srgba(0.025, 0.42, 0.82, 0.86),
        metallic: 0.04,
        perceptual_roughness: 0.12,
        unlit: true,
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    let bank = mats.add(StandardMaterial {
        base_color: Color::srgb(0.2, 0.12, 0.07),
        perceptual_roughness: 0.3,
        reflectance: 0.5,
        ..default()
    });

    // The visible water follows the same winding path as the riverbed.
    let water_mesh = meshes.add(river_water_mesh(0.0, RIVER_WATER_Y, 0.0));
    commands.spawn((
        PbrBundle {
            mesh: water_mesh,
            material: water,
            ..default()
        },
        RiverWater,
    ));
    commands.spawn(PbrBundle {
        mesh: meshes.add(river_bank_mesh()),
        material: bank,
        transform: Transform::from_xyz(0.0, GROUND_SURFACE_Y + 0.006, 0.0),
        ..default()
    });
    commands.spawn(PbrBundle {
        mesh: meshes.add(river_wheat_mesh()),
        material: mats.add(StandardMaterial {
            base_color: Color::srgb(0.92, 0.68, 0.12),
            perceptual_roughness: 0.86,
            ..default()
        }),
        ..default()
    });

    spawn_fish(commands, meshes, mats, -23.0, -1.0, 0);
    spawn_fish(commands, meshes, mats, -10.0, 0.8, 1);
    spawn_fish(commands, meshes, mats, 2.0, 0.6, 2);
    spawn_fish(commands, meshes, mats, 12.0, -1.2, 3);
    spawn_fish(commands, meshes, mats, 25.0, 1.1, 4);
    spawn_fish(commands, meshes, mats, 34.0, -0.5, 5);
}

fn river_water_mesh(elapsed: f32, water_level: f32, overflow: f32) -> Mesh {
    let segments = (GROUND_HALF_SIZE * 2.0 / 0.5) as u32;
    let mut vertices = Vec::with_capacity(((segments + 1) * 2) as usize);
    let mut indices = Vec::with_capacity((segments * 6) as usize);
    let half_width = RIVER_WIDTH * 0.5 + overflow;
    for row in 0..=segments {
        let z = -GROUND_HALF_SIZE + row as f32 * 0.5;
        let center = river_center_x(z);
        let ripple =
            (z * 0.55 + elapsed * 2.4).sin() * 0.012 + (z * 0.23 - elapsed * 1.5).sin() * 0.009;
        for side in [-1.0_f32, 1.0] {
            vertices.push([center + side * half_width, water_level + ripple, z]);
        }
        if row < segments {
            let a = row * 2;
            let b = a + 1;
            let c = a + 2;
            let d = a + 3;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
    mesh.insert_indices(bevy::render::mesh::Indices::U32(indices));
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        vec![[0.0, 1.0, 0.0]; ((segments + 1) * 2) as usize],
    );
    mesh
}

fn river_bank_mesh() -> Mesh {
    let segments = (GROUND_HALF_SIZE * 2.0 / 0.5) as u32;
    let mut vertices = Vec::with_capacity(((segments + 1) * 4) as usize);
    let mut indices = Vec::with_capacity((segments * 12) as usize);
    for side in [-1.0_f32, 1.0] {
        let vertex_offset = vertices.len() as u32;
        for row in 0..=segments {
            let z = -GROUND_HALF_SIZE + row as f32 * 0.5;
            let center = river_center_x(z) + side * (RIVER_WIDTH * 0.5 + 0.25);
            vertices.push([center, 0.0, z]);
            vertices.push([center + side * 0.55, 0.0, z]);
            if row < segments {
                let a = vertex_offset + row * 2;
                let b = a + 1;
                let c = a + 2;
                let d = a + 3;
                if side > 0.0 {
                    indices.extend_from_slice(&[a, c, b, b, c, d]);
                } else {
                    indices.extend_from_slice(&[a, b, c, b, d, c]);
                }
            }
        }
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
    mesh.insert_indices(bevy::render::mesh::Indices::U32(indices));
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        vec![[0.0, 1.0, 0.0]; ((segments + 1) * 4) as usize],
    );
    mesh
}

fn river_wheat_mesh() -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();

    for (side_index, side) in [-1.0_f32, 1.0].into_iter().enumerate() {
        let mut row = 0_u32;
        let mut z = -GROUND_HALF_SIZE + 0.12;
        while z < GROUND_HALF_SIZE {
            let seed = row
                .wrapping_mul(37)
                .wrapping_add(side_index as u32 * 7_919)
                .wrapping_add(5_317);
            let jitter_z = (pseudo_rand(seed) - 0.5) * 0.18;
            for layer in 0..5_u32 {
                let layer_seed = seed.wrapping_add(layer * 1_013);
                let offset = 0.34 + layer as f32 * 0.34 + pseudo_rand(layer_seed) * 0.15;
                let world_z =
                    z + jitter_z + (pseudo_rand(layer_seed.wrapping_add(19)) - 0.5) * 0.18;
                let world_x = river_center_x(world_z) + side * (RIVER_WIDTH * 0.5 + offset);
                let base = Vec3::new(world_x, GROUND_SURFACE_Y + 0.015, world_z);
                let height = 0.92 + pseudo_rand(layer_seed.wrapping_add(31)) * 0.5;
                let lean = Vec3::new(
                    (pseudo_rand(layer_seed.wrapping_add(43)) - 0.5) * 0.2,
                    0.0,
                    (pseudo_rand(layer_seed.wrapping_add(59)) - 0.5) * 0.16,
                );
                let top = base + Vec3::Y * height + lean;
                let across = Vec3::new(0.027, 0.0, 0.0);
                let diagonal = Vec3::new(0.0, 0.0, 0.027);
                push_wheat_quad(
                    &mut positions,
                    &mut normals,
                    &mut indices,
                    base - across,
                    base + across,
                    top + across * 0.22,
                    top - across * 0.22,
                );
                push_wheat_quad(
                    &mut positions,
                    &mut normals,
                    &mut indices,
                    base - diagonal,
                    base + diagonal,
                    top + diagonal * 0.22,
                    top - diagonal * 0.22,
                );

                for leaf_side in [-1.0_f32, 1.0] {
                    let leaf_base = base + Vec3::Y * (height * 0.34);
                    let leaf_tip =
                        leaf_base + Vec3::new(leaf_side * 0.14, height * 0.15, leaf_side * 0.035);
                    let leaf_width = Vec3::new(0.013, 0.0, 0.013);
                    push_wheat_quad(
                        &mut positions,
                        &mut normals,
                        &mut indices,
                        leaf_base - leaf_width,
                        leaf_base + leaf_width,
                        leaf_tip,
                        leaf_tip,
                    );
                }

                let head_base = top - Vec3::Y * 0.04;
                for grain in 0..5_u32 {
                    let grain_side = if grain % 2 == 0 { -1.0 } else { 1.0 };
                    let grain_y = grain as f32 * 0.042;
                    let grain_center =
                        head_base + Vec3::Y * grain_y + across * (grain_side * 0.028);
                    let grain_width = Vec3::new(0.016, 0.0, 0.014);
                    push_wheat_quad(
                        &mut positions,
                        &mut normals,
                        &mut indices,
                        grain_center - grain_width,
                        grain_center + grain_width,
                        grain_center + Vec3::Y * 0.07,
                        grain_center + Vec3::Y * 0.07,
                    );
                }
            }
            row = row.wrapping_add(1);
            z += 0.3;
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(bevy::render::mesh::Indices::U32(indices));
    mesh
}

fn push_wheat_quad(
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
    a: Vec3,
    b: Vec3,
    c: Vec3,
    d: Vec3,
) {
    let normal = (b - a).cross(c - a).normalize_or_zero();
    let start = positions.len() as u32;
    for point in [a, b, c, d] {
        positions.push(point.to_array());
        normals.push(normal.to_array());
    }
    indices.extend_from_slice(&[
        start,
        start + 1,
        start + 2,
        start,
        start + 2,
        start + 3,
        start + 2,
        start + 1,
        start,
        start + 3,
        start + 2,
        start,
    ]);
}

fn spawn_fish(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &mut Assets<StandardMaterial>,
    z: f32,
    offset: f32,
    index: u32,
) {
    let raw_body = [
        0.72 - index as f32 * 0.06,
        0.62 + index as f32 * 0.04,
        0.34 + index as f32 * 0.09,
    ];
    let raw_fin = [0.9, 0.57, 0.25];
    let color = mats.add(StandardMaterial {
        base_color: Color::srgb(raw_body[0], raw_body[1], raw_body[2]),
        unlit: true,
        ..default()
    });
    let fin = mats.add(StandardMaterial {
        base_color: Color::srgb(raw_fin[0], raw_fin[1], raw_fin[2]),
        unlit: true,
        ..default()
    });
    commands
        .spawn((
            SpatialBundle::from_transform(Transform::from_xyz(
                river_center_x(z) + offset,
                RIVER_WATER_Y - 0.12,
                z,
            )),
            Fish {
                phase: z + GROUND_HALF_SIZE,
                speed: 1.0 + index as f32 * 0.27,
                offset,
                impaled: false,
                dropped: false,
                cooked: 0.0,
                body_mat: color.clone(),
                fin_mat: fin.clone(),
                raw_body,
                raw_fin,
            },
        ))
        .with_children(|parent| {
            parent.spawn(PbrBundle {
                mesh: meshes.add(Sphere::new(0.16)),
                material: color.clone(),
                transform: Transform {
                    scale: Vec3::new(0.68, 0.42, 1.5),
                    ..default()
                },
                ..default()
            });
            parent.spawn(PbrBundle {
                mesh: meshes.add(Cone {
                    radius: 0.1,
                    height: 0.28,
                }),
                material: fin.clone(),
                transform: Transform::from_xyz(0.0, 0.0, 0.27)
                    .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
                ..default()
            });
            for side in [-1.0_f32, 1.0] {
                parent.spawn(PbrBundle {
                    mesh: meshes.add(Cone {
                        radius: 0.055,
                        height: 0.2,
                    }),
                    material: fin.clone(),
                    transform: Transform::from_xyz(side * 0.09, -0.015, 0.0)
                        .with_rotation(Quat::from_rotation_z(side * std::f32::consts::FRAC_PI_2)),
                    ..default()
                });
            }
        });
}

fn animate_river(
    time: Res<Time>,
    mut meshes: ResMut<Assets<Mesh>>,
    water: Query<&Handle<Mesh>, With<RiverWater>>,
    mut river: ResMut<RiverState>,
) {
    let elapsed = time.elapsed_seconds();
    // Most of the cycle is calm; each slow swell briefly spills water onto the banks.
    let tide = (elapsed * 0.055 - 1.1).sin().max(0.0);
    let flood = tide * tide * tide;
    river.overflow = RIVER_FLOOD_MAX * flood;
    river.water_level = RIVER_WATER_Y + RIVER_FLOOD_HEIGHT * flood;
    let Ok(handle) = water.get_single() else {
        return;
    };
    if let Some(mesh) = meshes.get_mut(handle) {
        *mesh = river_water_mesh(elapsed, river.water_level, river.overflow);
    }
}

fn swim_fish(
    time: Res<Time>,
    river: Res<RiverState>,
    mut fish: Query<(&mut Fish, &mut Transform)>,
) {
    let dt = time.delta_seconds();
    let span = GROUND_HALF_SIZE * 2.0;
    for (mut fish, mut transform) in &mut fish {
        if fish.impaled || fish.dropped {
            continue;
        }
        fish.phase = (fish.phase + fish.speed * dt).rem_euclid(span);
        let z = fish.phase - GROUND_HALF_SIZE;
        let next_x = river_center_x(z + 0.1);
        let prev_x = river_center_x(z - 0.1);
        let slope = (next_x - prev_x) / 0.2;
        let direction = Vec3::new(slope, 0.0, 1.0).normalize();
        transform.translation = Vec3::new(
            river_center_x(z) + fish.offset,
            river.water_level - 0.16 + (time.elapsed_seconds() * 2.0 + fish.offset).sin() * 0.025,
            z,
        );
        transform.rotation = Quat::from_rotation_y((-direction.x).atan2(-direction.z));
    }
}

fn water_drop_assets(
    slot: &mut Option<(Handle<Mesh>, Handle<StandardMaterial>)>,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) -> (Handle<Mesh>, Handle<StandardMaterial>) {
    slot.get_or_insert_with(|| {
        (
            meshes.add(Sphere::new(0.075)),
            materials.add(StandardMaterial {
                base_color: Color::srgba(0.42, 0.82, 1.0, 0.9),
                emissive: Color::srgb(0.08, 0.3, 0.55).into(),
                alpha_mode: AlphaMode::Blend,
                unlit: true,
                ..default()
            }),
        )
    })
    .clone()
}

fn spawn_water_splash(
    position: Vec3,
    water_level: f32,
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    splash_assets: &mut Option<(Handle<Mesh>, Handle<StandardMaterial>)>,
) {
    let assets = water_drop_assets(splash_assets, meshes, materials);
    for index in 0..8u32 {
        let angle = index as f32 * std::f32::consts::TAU / 8.0;
        let direction = Vec3::new(angle.cos(), 0.0, angle.sin());
        commands.spawn((
            PbrBundle {
                mesh: assets.0.clone(),
                material: assets.1.clone(),
                transform: Transform::from_translation(
                    Vec3::new(position.x, water_level, position.z) + direction * 0.16,
                )
                .with_scale(Vec3::splat(0.65)),
                ..default()
            },
            SplashParticle {
                velocity: direction * (0.7 + (index % 3) as f32 * 0.12)
                    + Vec3::Y * (1.1 + (index % 4) as f32 * 0.16),
                lifetime: 0.65,
            },
        ));
    }
}

fn water_splash_system(
    time: Res<Time>,
    river: Res<RiverState>,
    mut contact: ResMut<RiverContact>,
    player: Query<&Transform, (With<Player>, Without<Pebble>, Without<SplashParticle>)>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut splash_assets: Local<Option<(Handle<Mesh>, Handle<StandardMaterial>)>>,
    mut particles: Query<
        (Entity, &mut SplashParticle, &mut Transform),
        (With<SplashParticle>, Without<Player>, Without<Pebble>),
    >,
    mut pebbles: Query<
        (&mut Pebble, &Transform, &Velocity),
        (With<Pebble>, Without<Player>, Without<SplashParticle>),
    >,
) {
    let dt = time.delta_seconds();
    contact.splash_cooldown = (contact.splash_cooldown - dt).max(0.0);

    if let Ok(player) = player.get_single() {
        let position = player.translation;
        let in_water = (position.x - river_center_x(position.z)).abs()
            < RIVER_WIDTH * 0.5 + river.overflow
            && position.y - 0.9 < river.water_level
            && position.y > river.water_level - 2.0;
        let moved = contact
            .last_position
            .map_or(false, |last| last.distance(position) > 0.12);
        if in_water && (!contact.was_in_water || (moved && contact.splash_cooldown == 0.0)) {
            spawn_water_splash(
                position,
                river.water_level,
                &mut commands,
                &mut meshes,
                &mut materials,
                &mut splash_assets,
            );
            contact.splash_cooldown = 0.38;
        }
        contact.was_in_water = in_water;
        contact.last_position = Some(position);
    }

    for (mut pebble, transform, velocity) in &mut pebbles {
        let position = transform.translation;
        let entered_water = (position.x - river_center_x(position.z)).abs()
            < RIVER_WIDTH * 0.5 + river.overflow
            && pebble.last_position.y > river.water_level + pebble.radius
            && position.y <= river.water_level + pebble.radius
            && velocity.linvel.y < -0.5;
        if entered_water {
            spawn_water_splash(
                position,
                river.water_level,
                &mut commands,
                &mut meshes,
                &mut materials,
                &mut splash_assets,
            );
        }
        pebble.last_position = position;
    }

    for (entity, mut particle, mut transform) in &mut particles {
        particle.lifetime -= dt;
        particle.velocity.y -= 2.8 * dt;
        transform.translation += particle.velocity * dt;
        transform.scale *= (1.0 - dt * 1.5).max(0.0);
        if particle.lifetime <= 0.0 {
            commands.entity(entity).despawn();
        }
    }
}

fn pebble_break_system(
    mut collisions: EventReader<CollisionEvent>,
    mut pebbles: Query<(Entity, &mut Pebble, &Transform)>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut fragment_assets: Local<Option<(Handle<Mesh>, Handle<StandardMaterial>)>>,
) {
    for event in collisions.read() {
        let CollisionEvent::Started(first, second, _) = event else {
            continue;
        };
        let pebble_entity = if pebbles.get(*first).is_ok() {
            *first
        } else if pebbles.get(*second).is_ok() {
            *second
        } else {
            continue;
        };
        let Ok((_, mut pebble, transform)) = pebbles.get_mut(pebble_entity) else {
            continue;
        };
        let impact_velocity = pebble.thrown_velocity;
        if impact_velocity.length() < 13.0 {
            continue;
        }

        let position = transform.translation;
        let radius = pebble.radius;
        pebble.thrown_velocity = Vec3::ZERO;
        let (fragment_mesh, fragment_material) = fragment_assets.get_or_insert_with(|| {
            (
                meshes.add(Sphere::new(1.0)),
                materials.add(Color::srgb(0.43, 0.44, 0.43)),
            )
        });
        let incoming = impact_velocity.normalize_or_zero();
        let seed_base = position.x.to_bits() ^ position.y.to_bits().rotate_left(11);
        for index in 0..6 {
            let seed = seed_base.wrapping_add(index * 37);
            let spread = Vec3::new(
                pseudo_rand(seed) * 2.0 - 1.0,
                pseudo_rand(seed.wrapping_add(1)) * 2.0 - 0.4,
                pseudo_rand(seed.wrapping_add(2)) * 2.0 - 1.0,
            )
            .normalize_or_zero();
            let shard_radius = radius * (0.22 + pseudo_rand(seed.wrapping_add(3)) * 0.12);
            commands.spawn((
                PbrBundle {
                    mesh: fragment_mesh.clone(),
                    material: fragment_material.clone(),
                    transform: Transform::from_translation(position + spread * radius * 0.45)
                        .with_scale(Vec3::splat(shard_radius)),
                    ..default()
                },
                RigidBody::Dynamic,
                Collider::ball(shard_radius),
                Velocity {
                    linvel: incoming * 2.0
                        + spread * (1.5 + pseudo_rand(seed.wrapping_add(4)) * 2.0),
                    angvel: spread * 8.0,
                },
                Restitution::coefficient(0.2),
                Friction::coefficient(0.9),
                TreeChip { remaining: 5.0 },
            ));
        }
        commands.entity(pebble_entity).despawn_recursive();
    }
}

fn update_diggable_ground(
    sites: &[(Vec2, f32)],
    terrain: &mut Query<(Entity, &Handle<Mesh>), With<DiggableGround>>,
    meshes: &mut Assets<Mesh>,
    commands: &mut Commands,
) {
    let Ok((entity, handle)) = terrain.get_single_mut() else {
        return;
    };
    let (mesh, vertices, triangles) = ground_geometry(sites);
    if let Some(ground_mesh) = meshes.get_mut(handle) {
        *ground_mesh = mesh;
    }
    commands
        .entity(entity)
        .insert(Collider::trimesh(vertices, triangles));
}

fn dynamic_box(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &mut Assets<StandardMaterial>,
    pos: Vec3,
    size: Vec3,
    color: Color,
) {
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Cuboid::new(size.x, size.y, size.z)),
            material: mats.add(color),
            transform: Transform::from_translation(pos),
            ..default()
        },
        RigidBody::Dynamic,
        Collider::cuboid(size.x / 2.0, size.y / 2.0, size.z / 2.0),
        Velocity::default(),
        Restitution::coefficient(0.25),
        Friction::coefficient(0.8),
        Sleeping::disabled(),
    ));
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
    path: Res<ImagePath>,
) {
    // Camera
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(0.0, 1.6, 3.3)
                .looking_at(Vec3::new(0.0, 0.8, 0.0), Vec3::Y),
            ..default()
        },
        DialogueFocusSettings {
            center: Vec2::splat(0.5),
            radius: 0.95,
            aspect: 16.0 / 9.0,
            magnification: 1.0,
            ..default()
        },
    ));

    // The player: a 1.8 m tall capsule driven by Rapier's character controller
    // (walks, slides along walls, steps up stairs, falls, jumps).
    commands.spawn((
        SpatialBundle::from_transform(Transform::from_translation(PLAYER_SPAWN)),
        RigidBody::KinematicPositionBased,
        Collider::capsule_y(0.6, 0.3),
        KinematicCharacterController {
            offset: CharacterLength::Absolute(0.01),
            slide: true,
            autostep: Some(CharacterAutostep {
                max_height: CharacterLength::Absolute(0.4),
                min_width: CharacterLength::Absolute(0.2),
                include_dynamic_bodies: false,
            }),
            max_slope_climb_angle: 50.0_f32.to_radians(),
            min_slope_slide_angle: 60.0_f32.to_radians(),
            snap_to_ground: Some(CharacterLength::Absolute(0.2)),
            apply_impulse_to_dynamic_bodies: true,
            filter_groups: Some(CollisionGroups::new(
                Group::ALL,
                Group::ALL & !Group::GROUP_2,
            )),
            ..default()
        },
        Player {
            yaw: 0.0,
            pitch: -0.1,
            vy: 0.0,
            last_dy: 0.0,
            mounted: None,
            climbing_tree: None,
        },
    ));

    // On-screen help + status line
    let style = TextStyle {
        font_size: 16.0,
        color: Color::WHITE,
        ..default()
    };
    commands.spawn((
        TextBundle::from_sections([
            TextSection::new(
                "Drop a .glb / .gltf / .fbx or an image onto the window\n\
                 F look at an animal or person to possess it (F again to return) - LMB grab/talk (hold/release to throw pebbles; pull grass to uproot, branches to break and leaves to pluck) - wheel push/pull - RMB look - WASD walk - Shift kneel - Space jump - click horse back to ride, E dismount - sword: grab handle, hold LMB & swing - axe: grab handle & swing at trees (a felled trunk and its branches are chopped layer by layer too) - shovel: aim at ground & swing to dig - spear: grab handle, aim & click to thrust - bucket: dip/tilt it in the river to scoop, tilt to pour - machete: grab handle, hold LMB & sweep to cut grass, plants and sticks - brush: click it, click a paint can to dip, then click/drag on anything to paint - Tom gets hungry and feeds himself (H makes him hungry now) - MMB throw an apple (holding it, or pointing at it) - X clear - Esc close chat\n",
                style.clone(),
            ),
            TextSection::new("", TextStyle { color: Color::srgb(1.0, 0.85, 0.3), ..style }),
        ])
        .with_style(Style {
            position_type: PositionType::Absolute,
            top: Val::Px(8.0),
            left: Val::Px(8.0),
            ..default()
        }),
        StatusText,
    ));

    // Conversation line, bottom of the screen
    commands.spawn((
        TextBundle::from_section(
            "",
            TextStyle {
                font_size: 22.0,
                color: Color::WHITE,
                ..default()
            },
        )
        .with_style(Style {
            position_type: PositionType::Absolute,
            bottom: Val::Px(24.0),
            left: Val::Px(8.0),
            right: Val::Px(8.0),
            ..default()
        }),
        DialogueBox,
    ));

    // Warm lamps keep the rooms lit at night.
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 900_000.0,
            range: 25.0,
            shadows_enabled: true,
            color: Color::srgb(1.0, 0.92, 0.75),
            ..default()
        },
        transform: Transform::from_xyz(0.0, F1_WALL_CY + 1.0, 0.0),
        ..default()
    });
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 700_000.0,
            range: 25.0,
            shadows_enabled: true,
            color: Color::srgb(1.0, 0.92, 0.75),
            ..default()
        },
        transform: Transform::from_xyz(0.0, F2_WALL_CY + 1.0, 0.0),
        ..default()
    });
    // Daylight changes throughout the day; storms add separate lightning flashes.
    commands.spawn((
        DirectionalLightBundle {
            directional_light: DirectionalLight {
                illuminance: 55_000.0,
                shadows_enabled: true,
                color: Color::srgb(1.0, 0.92, 0.78),
                ..default()
            },
            transform: Transform::from_xyz(-10.0, 18.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        Daylight,
    ));
    // Lightning: an off-by-default flash light.
    commands.spawn((
        DirectionalLightBundle {
            directional_light: DirectionalLight {
                illuminance: 0.0,
                shadows_enabled: false,
                color: Color::srgb(0.85, 0.88, 1.0),
                ..default()
            },
            transform: Transform::from_xyz(0.0, 10.0, 0.0)
                .looking_at(Vec3::new(0.3, 0.0, 0.5), Vec3::Y),
            ..default()
        },
        Lightning,
    ));

    // ----- Two-story wooden house (8.4 x 8.4 m footprint), storm outside -----
    let wood = Color::srgb(0.47, 0.3, 0.17); // outer walls: weathered wood siding
    let floor_wood = Color::srgb(0.55, 0.38, 0.24);
    let roof_color = Color::srgb(0.3, 0.12, 0.08);
    let (m, a) = (&mut *meshes, &mut *mats);
    let mut static_boxes = StaticBoxBatch::default();

    static_box(
        &mut commands,
        &mut static_boxes,
        Vec3::new(0.0, F1_FLOOR_Y - SLAB_T / 2.0, 0.0),
        Vec3::new(FOOT, SLAB_T, FOOT),
        floor_wood,
    );
    // 2nd-floor slab with a stairwell opening over the stairs (left side of the house)
    let slab_y = F2_FLOOR_Y - SLAB_T / 2.0;
    let half = FOOT / 2.0;
    let (hx0, hx1, hz0, hz1) = (-3.9_f32, -2.7_f32, -0.3_f32, 3.95_f32);
    for (x0, x1, z0, z1) in [
        (-half, hx0, -half, half),
        (hx1, half, -half, half),
        (hx0, hx1, -half, hz0),
        (hx0, hx1, hz1, half),
    ] {
        static_box(
            &mut commands,
            &mut static_boxes,
            Vec3::new((x0 + x1) / 2.0, slab_y, (z0 + z1) / 2.0),
            Vec3::new(x1 - x0, SLAB_T, z1 - z0),
            floor_wood,
        );
    }
    static_box(
        &mut commands,
        &mut static_boxes,
        Vec3::new(0.0, ROOF_BASE_Y + SLAB_T / 2.0, 0.0),
        Vec3::new(FOOT, SLAB_T, FOOT),
        floor_wood,
    ); // 2nd-story ceiling

    // Front/back walls get a window so you can see the storm; side walls are solid.
    for &cy in &[F1_WALL_CY, F2_WALL_CY] {
        wall_with_window(
            &mut commands,
            m,
            a,
            &mut static_boxes,
            Vec3::new(0.0, cy, -FOOT / 2.0),
            FOOT,
            STORY_H,
            WALL_T,
            2.4,
            1.3,
            1.1,
            wood,
        );
        wall_with_window(
            &mut commands,
            m,
            a,
            &mut static_boxes,
            Vec3::new(0.0, cy, FOOT / 2.0),
            FOOT,
            STORY_H,
            WALL_T,
            2.4,
            1.3,
            1.1,
            wood,
        );
        static_box(
            &mut commands,
            &mut static_boxes,
            Vec3::new(-FOOT / 2.0, cy, 0.0),
            Vec3::new(WALL_T, STORY_H, FOOT),
            wood,
        );
        if cy == F1_WALL_CY {
            wall_with_door(
                &mut commands,
                &mut static_boxes,
                Vec3::new(FOOT / 2.0, cy, 0.0),
                FOOT,
                STORY_H,
                WALL_T,
                DOOR_W,
                DOOR_H,
                wood,
            );
        } else {
            static_box(
                &mut commands,
                &mut static_boxes,
                Vec3::new(FOOT / 2.0, cy, 0.0),
                Vec3::new(WALL_T, STORY_H, FOOT),
                wood,
            );
        }
    }

    // Pitched roof: two slanted slabs meeting at a ridge along X.
    let half_span = FOOT / 2.0 + 0.2; // small overhang
    let rise = 1.4;
    let slope_len = (half_span * half_span + rise * rise).sqrt();
    let angle = rise.atan2(half_span);
    for side in [-1.0_f32, 1.0] {
        let rot = Quat::from_rotation_x(side * angle);
        commands.spawn((
            PbrBundle {
                mesh: m.add(Cuboid::new(FOOT + 0.4, 0.15, slope_len)),
                material: a.add(roof_color),
                transform: Transform::from_xyz(
                    0.0,
                    ROOF_BASE_Y + SLAB_T + rise / 2.0,
                    side * half_span / 2.0,
                )
                .with_rotation(rot),
                ..default()
            },
            RigidBody::Fixed,
            Collider::cuboid((FOOT + 0.4) / 2.0, 0.075, slope_len / 2.0),
        ));
    }

    // Real stairs along the left wall: 14 steps climbing toward -Z, then you step
    // up onto the 2nd floor. (Each step is a solid block down to the floor.)
    let stair_rise = F2_FLOOR_Y / 15.0;
    for i in 0..14 {
        let top = (i as f32 + 1.0) * stair_rise;
        let z_front = 3.9 - i as f32 * 0.3;
        static_box(
            &mut commands,
            &mut static_boxes,
            Vec3::new((-3.9 + -2.7) / 2.0, top / 2.0, z_front - 0.15),
            Vec3::new(1.2, top, 0.3),
            floor_wood,
        );
    }

    // Triangulated ground can be reshaped when the player digs.
    spawn_diggable_ground(&mut commands, m, a);
    spawn_shoreline(&mut commands, m, a);
    spawn_mountain(&mut commands, m, a);
    spawn_pebbles(&mut commands, m, a);
    spawn_ground_plants(&mut commands, m, a);
    spawn_twigs(&mut commands, m, a);
    spawn_apples(&mut commands, m, a);

    // A larger grove outside the house.
    let mut tree_index = 0u32;
    for x in [8.0_f32, 12.5, 17.0, 21.5, 26.0, 30.5, 35.0] {
        for z in [
            -16.0_f32, -11.5, -7.0, -2.5, 2.0, 6.5, 11.0, 15.5, 20.0, 24.5, 29.0,
        ] {
            let seed = tree_index.wrapping_add(81);
            let jitter_x = (pseudo_rand(seed * 2) - 0.5) * 1.4;
            let jitter_z = (pseudo_rand(seed * 2 + 1) - 0.5) * 1.4;
            let world_x = x + jitter_x;
            let world_z = z + jitter_z;
            if world_x < 6.0
                || (world_x < 10.5 && world_z.abs() < 6.5)
                || near_mountain(world_x, world_z)
                || (world_x > 20.0
                    && (world_x - river_center_x(world_z)).abs() < RIVER_WIDTH * 0.5 + 1.8)
            {
                continue;
            }
            spawn_tree(
                &mut commands,
                m,
                a,
                Vec3::new(world_x, land_surface_height(world_x, world_z), world_z),
                tree_index,
            );
            tree_index += 1;
        }
    }
    for bird_index in 0..8u32 {
        spawn_bird(&mut commands, m, a, bird_index);
    }
    // Leaf mantises, camouflaged in the grass.
    for (i, (x, z)) in [(-5.8_f32, 5.5_f32), (-10.0, -6.0), (8.2, 6.5)]
        .into_iter()
        .enumerate()
    {
        spawn_leaf_mantis(
            &mut commands,
            m,
            a,
            Vec3::new(x, land_surface_height(x, z), z),
            2100 + i as u32 * 29,
        );
    }
    // Grasshoppers in the meadow between the house and the river, and a couple on the far side.
    for (i, (x, z)) in [
        (-8.0_f32, 3.0_f32),
        (-9.5, -3.0),
        (-11.5, 6.0),
        (-7.0, -6.5),
        (7.5, 3.5),
        (9.0, -4.0),
    ]
    .into_iter()
    .enumerate()
    {
        spawn_grasshopper(
            &mut commands,
            m,
            a,
            Vec3::new(x, land_surface_height(x, z), z),
            1300 + i as u32 * 37,
        );
    }

    // Door leaf, hinged on its -Z edge; swings outward (+X). Grab a handle and drag.
    let door_len = DOOR_W - 0.05;
    commands
        .spawn((
            SpatialBundle::from_transform(Transform::from_xyz(
                FOOT / 2.0,
                F1_FLOOR_Y,
                -DOOR_W / 2.0,
            )),
            RigidBody::Fixed,
            DoorPivot,
        ))
        .with_children(|p| {
            p.spawn((
                PbrBundle {
                    mesh: m.add(Cuboid::new(0.06, DOOR_H, door_len)),
                    material: a.add(Color::srgb(0.32, 0.18, 0.1)),
                    transform: Transform::from_xyz(0.0, DOOR_H / 2.0, door_len / 2.0),
                    ..default()
                },
                Collider::cuboid(0.03, DOOR_H / 2.0, door_len / 2.0),
            ));
            for dx in [-0.08_f32, 0.08] {
                p.spawn((
                    PbrBundle {
                        mesh: m.add(Sphere::new(0.07)),
                        material: a.add(Color::srgb(0.9, 0.72, 0.2)),
                        transform: Transform::from_xyz(dx, 1.0, door_len - 0.15),
                        ..default()
                    },
                    Collider::ball(0.12), // a bit bigger than the knob so it's easy to grab
                    DoorHandle,
                ));
            }
        });

    // Porch lamp so the ground outside the door is visible
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 600_000.0,
            range: 14.0,
            color: Color::srgb(1.0, 0.85, 0.6),
            ..default()
        },
        transform: Transform::from_xyz(FOOT / 2.0 + 0.8, 2.6, 0.0),
        ..default()
    });
    // very faint cold "moonlight" so the outdoors isn't pitch black
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 1500.0,
            shadows_enabled: false,
            color: Color::srgb(0.6, 0.7, 1.0),
            ..default()
        },
        transform: Transform::from_xyz(5.0, 20.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });

    // Table
    let wood = Color::srgb(0.4, 0.26, 0.15);
    let (tx, tz) = (-2.5, -2.5);
    static_box(
        &mut commands,
        &mut static_boxes,
        Vec3::new(tx, 0.75, tz),
        Vec3::new(4.0, 0.1, 1.8),
        wood,
    );
    for (dx, dz) in [(-1.7, -0.75), (1.7, -0.75), (-1.7, 0.75), (1.7, 0.75)] {
        static_box(
            &mut commands,
            &mut static_boxes,
            Vec3::new(tx + dx, 0.35, tz + dz),
            Vec3::new(0.1, 0.7, 0.1),
            wood,
        );
    }

    // Grabbable props
    dynamic_box(
        &mut commands,
        m,
        a,
        Vec3::new(2.5, 0.3, -2.0),
        Vec3::splat(0.6),
        Color::srgb(0.7, 0.5, 0.25),
    );
    dynamic_box(
        &mut commands,
        m,
        a,
        Vec3::new(2.5, 0.9, -2.0),
        Vec3::splat(0.5),
        Color::srgb(0.3, 0.55, 0.8),
    );
    dynamic_box(
        &mut commands,
        m,
        a,
        Vec3::new(tx, 1.0, tz),
        Vec3::new(0.3, 0.3, 0.3),
        Color::srgb(0.8, 0.3, 0.3),
    );
    commands.spawn((
        PbrBundle {
            mesh: m.add(Sphere::new(0.25)),
            material: a.add(Color::srgb(0.2, 0.75, 0.35)),
            transform: Transform::from_xyz(1.5, 1.0, 1.0),
            ..default()
        },
        RigidBody::Dynamic,
        Collider::ball(0.25),
        Velocity::default(),
        Restitution::coefficient(0.75),
        Friction::coefficient(0.6),
        Sleeping::disabled(),
    ));

    // ----- A sword lying on the table -----
    spawn_sword(&mut commands, m, a, Vec3::new(-4.0, 0.86, -3.05));

    // ----- An axe lying beside the sword -----
    spawn_axe(&mut commands, m, a, Vec3::new(-2.0, 0.86, -2.05));

    // ----- A shovel lying beside the other tools -----
    spawn_shovel(&mut commands, m, a, Vec3::new(-4.0, 0.86, -2.05));

    // ----- A spear lying with the other tools -----
    spawn_spear(&mut commands, m, a, Vec3::new(-2.6, 0.86, -3.05));

    // ----- A bucket for scooping water -----
    spawn_bucket(&mut commands, m, a, Vec3::new(-1.0, 0.86, -2.05));

    // ----- A machete lying with the other tools -----
    spawn_machete(&mut commands, m, a, Vec3::new(-1.5, 0.86, -3.05));

    // ----- A knife lying with the other tools -----
    spawn_knife(&mut commands, m, a, Vec3::new(-0.6, 0.86, -3.05));

    // ----- A paint brush and cans of paint in different colours, on the floor by the table -----
    spawn_brush(&mut commands, m, a, Vec3::new(-3.5, F1_FLOOR_Y + 0.03, -1.15));
    spawn_paint_cans(&mut commands, m, a, Vec3::new(-2.9, F1_FLOOR_Y + 0.1, -1.15));

    // Friendly pets wander, sit, and occasionally run around the house.
    spawn_pet(
        &mut commands,
        m,
        a,
        Vec3::new(1.7, F1_FLOOR_Y, 1.7),
        PetKind::Cat,
        711,
    );
    spawn_pet(
        &mut commands,
        m,
        a,
        Vec3::new(-1.9, F1_FLOOR_Y, 2.0),
        PetKind::Dog,
        927,
    );
    spawn_horse(
        &mut commands,
        m,
        a,
        Vec3::new(34.0, land_surface_height(34.0, -23.0), -23.0),
    );

    // ----- One person sheltering from the storm: Tom. He gets hungry, and then he hunts,
    // chops, carves, lights a fire and cooks his own meal. Click him to talk. -----
    spawn_npc(
        &mut commands,
        m,
        a,
        Vec3::new(3.2, F1_FLOOR_Y, -3.2),
        "Tom",
        Color::srgb(0.25, 0.35, 0.6),
        &["I'm doing fine for now."],
    );

    // ----- Storm -----
    spawn_rain(&mut commands, m, a);

    // ----- Your image as a 3D physics object -----
    let img = load_image(&path.0);
    let start = Transform::from_xyz(0.0, 2.0, 0.0).with_rotation(Quat::from_euler(
        EulerRot::XYZ,
        0.3,
        0.4,
        0.1,
    ));
    spawn_relief(&mut commands, m, a, &img, start);
    static_boxes.spawn(&mut commands, m, a);
}

/// Builds a wall (spanning local X, normal along Z) with a rectangular window
/// cut into it, plus a faint glass pane so the storm is visible through it.
#[allow(clippy::too_many_arguments)]
fn wall_with_window(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &mut Assets<StandardMaterial>,
    batch: &mut StaticBoxBatch,
    center: Vec3,
    full_w: f32,
    full_h: f32,
    thickness: f32,
    win_w: f32,
    win_h: f32,
    win_bottom: f32, // height of the window's sill above the wall's bottom
    color: Color,
) {
    let side_w = (full_w - win_w) / 2.0;
    static_box(
        commands,
        batch,
        center + Vec3::new(-win_w / 2.0 - side_w / 2.0, 0.0, 0.0),
        Vec3::new(side_w, full_h, thickness),
        color,
    );
    static_box(
        commands,
        batch,
        center + Vec3::new(win_w / 2.0 + side_w / 2.0, 0.0, 0.0),
        Vec3::new(side_w, full_h, thickness),
        color,
    );

    let wall_bottom = center.y - full_h / 2.0;
    let wall_top = center.y + full_h / 2.0;
    let win_bottom_y = wall_bottom + win_bottom;
    let win_top_y = win_bottom_y + win_h;

    if win_bottom > 0.01 {
        static_box(
            commands,
            batch,
            Vec3::new(center.x, wall_bottom + win_bottom / 2.0, center.z),
            Vec3::new(win_w, win_bottom, thickness),
            color,
        );
    }
    let top_h = wall_top - win_top_y;
    if top_h > 0.01 {
        static_box(
            commands,
            batch,
            Vec3::new(center.x, win_top_y + top_h / 2.0, center.z),
            Vec3::new(win_w, top_h, thickness),
            color,
        );
    }

    // Glass: thin, slightly tinted, no collider so rain sound/sight reads through it visually.
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Cuboid::new(win_w, win_h, 0.03)),
            material: mats.add(StandardMaterial {
                base_color: Color::srgba(0.55, 0.7, 0.85, 0.25),
                alpha_mode: AlphaMode::Blend,
                perceptual_roughness: 0.05,
                ..default()
            }),
            transform: Transform::from_xyz(center.x, (win_bottom_y + win_top_y) / 2.0, center.z),
            ..default()
        },
        RigidBody::Fixed,
        Collider::cuboid(win_w / 2.0, win_h / 2.0, 0.015),
    ));
}

/// Spawns one clickable, wandering person standing at `pos` (feet position).
fn spawn_npc(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &mut Assets<StandardMaterial>,
    pos: Vec3,
    name: &str,
    shirt: Color,
    lines: &[&str],
) {
    let (radius, cyl) = (0.22, 0.7);
    let seed = ((pos.x * 100.0) as i32 as u32) ^ (name.len() as u32).wrapping_mul(7919);
    let center = pos + Vec3::Y * NPC_CENTER_Y;
    let body_mesh = meshes.add(Capsule3d::new(radius, cyl));
    let body_mat = mats.add(shirt);
    let head_mesh = meshes.add(Sphere::new(0.17));
    let skin = mats.add(Color::srgb(0.85, 0.7, 0.55));
    let nose_mesh = meshes.add(Cuboid::new(0.06, 0.06, 0.1));
    commands
        .spawn((
            PbrBundle {
                mesh: body_mesh,
                material: body_mat,
                transform: Transform::from_translation(center),
                ..default()
            },
            RigidBody::KinematicPositionBased,
            Collider::capsule_y(cyl / 2.0, radius),
            Npc {
                name: name.to_string(),
                lines: lines.iter().map(|s| s.to_string()).collect(),
                index: 0,
            },
            Walker {
                floor_y: pos.y,
                target: center,
                speed: 0.6 + pseudo_rand(seed) * 0.4,
                wait: pseudo_rand(seed.wrapping_add(1)) * 3.0,
                seed,
            },
        ))
        .with_children(|p| {
            // head, and a little nose so you can tell which way they face (-Z is forward)
            p.spawn(PbrBundle {
                mesh: head_mesh,
                material: skin.clone(),
                transform: Transform::from_xyz(0.0, 0.67, 0.0),
                ..default()
            });
            p.spawn(PbrBundle {
                mesh: nose_mesh,
                material: skin,
                transform: Transform::from_xyz(0.0, 0.67, -0.19),
                ..default()
            });
        });
}

fn spawn_horse(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    position: Vec3,
) {
    let coat = materials.add(Color::srgb(0.42, 0.19, 0.09));
    let dark_coat = materials.add(Color::srgb(0.16, 0.075, 0.04));
    let saddle = materials.add(Color::srgb(0.12, 0.055, 0.035));
    let hoof = materials.add(Color::srgb(0.09, 0.07, 0.06));

    commands
        .spawn((
            SpatialBundle::from_transform(Transform::from_translation(position)),
            RigidBody::Fixed,
            Horse {
                home: position,
                target: position,
                pause: 2.0,
                seed: 1847,
            },
        ))
        .with_children(|horse| {
            horse.spawn((
                SpatialBundle::from_transform(Transform::from_xyz(0.0, 1.25, 0.0)),
                Collider::capsule_y(0.55, 0.72),
                CollisionGroups::new(Group::GROUP_3, Group::ALL),
                HorseBack,
            ));
            horse.spawn((
                SpatialBundle::from_transform(Transform::from_xyz(0.0, 1.91, 0.0)),
                Collider::cuboid(0.58, 0.18, 0.7),
                CollisionGroups::new(Group::GROUP_3, Group::ALL),
                HorseBack,
            ));
            horse.spawn(PbrBundle {
                mesh: meshes.add(Sphere::new(0.7)),
                material: coat.clone(),
                transform: Transform::from_xyz(0.0, 1.35, 0.0)
                    .with_scale(Vec3::new(0.82, 0.86, 1.45)),
                ..default()
            });
            horse.spawn(PbrBundle {
                mesh: meshes.add(Capsule3d::new(0.32, 0.9)),
                material: coat.clone(),
                transform: Transform::from_xyz(0.0, 1.9, -0.47)
                    .with_rotation(Quat::from_rotation_x(-0.48))
                    .with_scale(Vec3::new(0.85, 1.0, 0.85)),
                ..default()
            });
            horse.spawn(PbrBundle {
                mesh: meshes.add(Sphere::new(0.39)),
                material: coat.clone(),
                transform: Transform::from_xyz(0.0, 2.35, -0.9)
                    .with_scale(Vec3::new(0.8, 1.0, 1.25)),
                ..default()
            });
            horse.spawn(PbrBundle {
                mesh: meshes.add(Sphere::new(0.21)),
                material: dark_coat.clone(),
                transform: Transform::from_xyz(0.0, 2.23, -1.17)
                    .with_scale(Vec3::new(0.85, 0.55, 1.1)),
                ..default()
            });
            horse.spawn(PbrBundle {
                mesh: meshes.add(Capsule3d::new(0.16, 0.72)),
                material: dark_coat.clone(),
                transform: Transform::from_xyz(0.0, 2.42, -0.52)
                    .with_rotation(Quat::from_rotation_x(-0.58)),
                ..default()
            });
            horse.spawn(PbrBundle {
                mesh: meshes.add(Capsule3d::new(0.11, 0.52)),
                material: dark_coat.clone(),
                transform: Transform::from_xyz(0.0, 1.65, 0.78)
                    .with_rotation(Quat::from_rotation_x(0.62)),
                ..default()
            });
            for side in [-1.0_f32, 1.0] {
                horse.spawn(PbrBundle {
                    mesh: meshes.add(Cone {
                        radius: 0.1,
                        height: 0.28,
                    }),
                    material: coat.clone(),
                    transform: Transform::from_xyz(side * 0.2, 2.68, -0.92)
                        .with_rotation(Quat::from_rotation_x(-0.12)),
                    ..default()
                });
                horse.spawn(PbrBundle {
                    mesh: meshes.add(Sphere::new(0.045)),
                    material: hoof.clone(),
                    transform: Transform::from_xyz(side * 0.25, 2.4, -1.17),
                    ..default()
                });
            }
            horse.spawn(PbrBundle {
                mesh: meshes.add(Cuboid::new(0.9, 0.2, 1.12)),
                material: saddle,
                transform: Transform::from_xyz(0.0, 1.89, 0.03),
                ..default()
            });
            for x in [-0.38_f32, 0.38] {
                for z in [-0.48_f32, 0.48] {
                    horse.spawn(PbrBundle {
                        mesh: meshes.add(Capsule3d::new(0.13, 0.92)),
                        material: coat.clone(),
                        transform: Transform::from_xyz(x, 0.55, z)
                            .with_scale(Vec3::new(0.78, 1.0, 0.82)),
                        ..default()
                    });
                    horse.spawn(PbrBundle {
                        mesh: meshes.add(Cuboid::new(0.27, 0.18, 0.33)),
                        material: hoof.clone(),
                        transform: Transform::from_xyz(x, 0.12, z),
                        ..default()
                    });
                }
            }
        });
}

fn spawn_mountain(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let stone = materials.add(Color::srgb(0.34, 0.38, 0.4));
    let shadow_stone = materials.add(Color::srgb(0.25, 0.29, 0.32));
    let snow = materials.add(Color::srgb(0.86, 0.9, 0.92));
    let peaks = [
        (Vec3::new(18.0, 18.0, -35.0), 23.0, 36.0, stone.clone()),
        (
            Vec3::new(0.0, 11.5, -33.0),
            14.0,
            23.0,
            shadow_stone.clone(),
        ),
        (Vec3::new(36.0, 13.0, -34.0), 15.0, 26.0, stone.clone()),
    ];
    for (position, radius, height, material) in peaks {
        commands
            .spawn(PbrBundle {
                mesh: meshes.add(Cone { radius, height }),
                material,
                transform: Transform::from_translation(position),
                ..default()
            })
            .insert((
                RigidBody::Fixed,
                Collider::cone(height * 0.5, radius),
                Friction::coefficient(0.9),
            ));
    }
    commands.spawn(PbrBundle {
        mesh: meshes.add(Cone {
            radius: 8.2,
            height: 11.0,
        }),
        material: snow,
        transform: Transform::from_xyz(18.0, 30.5, -35.0),
        ..default()
    });
}

fn spawn_pebbles(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let colors = [
        Color::srgb(0.39, 0.4, 0.38),
        Color::srgb(0.48, 0.45, 0.39),
        Color::srgb(0.31, 0.34, 0.35),
        Color::srgb(0.55, 0.51, 0.44),
    ];
    let mut spawned = 0;
    for seed in 0..400_u32 {
        let x = -37.0 + pseudo_rand(seed.wrapping_mul(17) + 301) * 74.0;
        let z = -37.0 + pseudo_rand(seed.wrapping_mul(31) + 907) * 74.0;
        if (x.abs() < 6.5 && z.abs() < 6.5)
            || (x < -20.0 && z < -26.0)
            || ((x - river_center_x(z)).abs() < RIVER_WIDTH * 0.5 + 1.1)
        {
            continue;
        }

        let radius = if spawned < 42 {
            0.09 + pseudo_rand(seed.wrapping_add(1201)) * 0.11
        } else {
            0.035 + pseudo_rand(seed.wrapping_add(1201)) * 0.045
        };
        let color = colors[(seed as usize) % colors.len()];
        let material = materials.add(StandardMaterial {
            base_color: color,
            perceptual_roughness: 0.96,
            ..default()
        });
        commands.spawn((
            PbrBundle {
                mesh: meshes.add(Sphere::new(radius)),
                material,
                transform: Transform::from_xyz(x, land_surface_height(x, z) + radius * 0.78, z)
                    .with_scale(Vec3::new(1.35, 0.82, 1.15))
                    .with_rotation(Quat::from_rotation_y(
                        pseudo_rand(seed.wrapping_add(1777)) * std::f32::consts::TAU,
                    )),
                ..default()
            },
            RigidBody::Dynamic,
            Collider::ball(radius),
            Velocity::default(),
            Restitution::coefficient(0.32),
            Friction::coefficient(0.9),
            Pebble {
                last_position: Vec3::new(x, land_surface_height(x, z) + radius * 0.78, z),
                radius,
                thrown_velocity: Vec3::ZERO,
            },
            ActiveEvents::COLLISION_EVENTS,
        ));
        spawned += 1;
        if spawned >= 142 {
            break;
        }
    }
}

fn spawn_ground_plants(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let grass_mesh = meshes.add(Cuboid::new(0.045, 0.52, 0.035));
    let grass_materials = [
        materials.add(Color::srgb(0.16, 0.38, 0.12)),
        materials.add(Color::srgb(0.24, 0.46, 0.14)),
        materials.add(Color::srgb(0.34, 0.48, 0.16)),
    ];
    let mut clumps = 0;
    for seed in 0..700_u32 {
        let x = -37.0 + pseudo_rand(seed.wrapping_mul(43) + 2801) * 74.0;
        let z = -37.0 + pseudo_rand(seed.wrapping_mul(59) + 3701) * 74.0;
        if (x.abs() < 7.0 && z.abs() < 7.0)
            || (z < -26.0 && x > -15.0)
            || ((x - river_center_x(z)).abs() < RIVER_WIDTH * 0.5 + 1.5)
        {
            continue;
        }

        let ground_y = land_surface_height(x, z);
        let anchor = Vec3::new(x, ground_y, z);
        commands
            .spawn((
                SpatialBundle::from_transform(Transform::from_translation(anchor)),
                RigidBody::Dynamic,
                Collider::ball(0.17),
                Sensor,
                LockedAxes::ROTATION_LOCKED,
                GravityScale(0.0),
                Velocity::default(),
                PullablePlant {
                    anchor,
                    uprooted: false,
                },
            ))
            .with_children(|plant| {
                for blade in 0..3_u32 {
                    let blade_seed = seed.wrapping_mul(7).wrapping_add(blade);
                    let height = 0.65 + pseudo_rand(blade_seed.wrapping_add(401)) * 0.55;
                    let lean = (pseudo_rand(blade_seed.wrapping_add(503)) - 0.5) * 0.6;
                    plant.spawn(PbrBundle {
                        mesh: grass_mesh.clone(),
                        material: grass_materials[(blade_seed as usize) % grass_materials.len()]
                            .clone(),
                        transform: Transform::from_xyz(
                            (pseudo_rand(blade_seed.wrapping_add(607)) - 0.5) * 0.3,
                            height * 0.26,
                            (pseudo_rand(blade_seed.wrapping_add(709)) - 0.5) * 0.3,
                        )
                        .with_scale(Vec3::new(0.75, height, 1.0))
                        .with_rotation(
                            Quat::from_rotation_y(pseudo_rand(blade_seed.wrapping_add(811)) * 3.14)
                                * Quat::from_rotation_z(lean),
                        ),
                        ..default()
                    });
                }
            });
        clumps += 1;
        if clumps >= 125 {
            break;
        }
    }

    let stem_mesh = meshes.add(Capsule3d::new(0.025, 0.55));
    let leaf_mesh = meshes.add(Sphere::new(0.11));
    let flower_mesh = meshes.add(Sphere::new(0.13));
    let stem_material = materials.add(Color::srgb(0.18, 0.38, 0.12));
    let leaf_material = materials.add(Color::srgb(0.23, 0.48, 0.15));
    let flower_materials = [
        materials.add(Color::srgb(0.95, 0.78, 0.22)),
        materials.add(Color::srgb(0.92, 0.88, 0.82)),
        materials.add(Color::srgb(0.78, 0.36, 0.62)),
    ];

    let mut flowers = 0;
    for seed in 0..400_u32 {
        let x = -36.0 + pseudo_rand(seed.wrapping_mul(71) + 4703) * 72.0;
        let z = -36.0 + pseudo_rand(seed.wrapping_mul(83) + 5701) * 72.0;
        if (x.abs() < 7.0 && z.abs() < 7.0)
            || (z < -26.0 && x > -15.0)
            || ((x - river_center_x(z)).abs() < RIVER_WIDTH * 0.5 + 1.5)
        {
            continue;
        }

        let ground_y = land_surface_height(x, z);
        commands.spawn(PbrBundle {
            mesh: stem_mesh.clone(),
            material: stem_material.clone(),
            transform: Transform::from_xyz(x, ground_y + 0.36, z)
                .with_scale(Vec3::new(0.8, 1.0, 0.8))
                .with_rotation(Quat::from_rotation_z(
                    (pseudo_rand(seed.wrapping_add(6011)) - 0.5) * 0.2,
                )),
            ..default()
        });
        for side in [-1.0_f32, 1.0] {
            commands.spawn(PbrBundle {
                mesh: leaf_mesh.clone(),
                material: leaf_material.clone(),
                transform: Transform::from_xyz(x + side * 0.1, ground_y + 0.24, z)
                    .with_scale(Vec3::new(1.4, 0.45, 0.65))
                    .with_rotation(Quat::from_rotation_y(side * 0.4)),
                ..default()
            });
        }
        commands.spawn(PbrBundle {
            mesh: flower_mesh.clone(),
            material: flower_materials[(seed as usize) % flower_materials.len()].clone(),
            transform: Transform::from_xyz(x, ground_y + 0.72, z)
                .with_scale(Vec3::new(1.0, 0.72, 1.0)),
            ..default()
        });
        flowers += 1;
        if flowers >= 32 {
            break;
        }
    }
}

/// A handful of apples lying in the yard. Each is a small physics body: pick one up with the
/// hand (left button) and press the middle mouse button to throw it.
fn spawn_apples(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let body_mesh = meshes.add(Sphere::new(APPLE_RADIUS));
    let stem_mesh = meshes.add(Cylinder::new(0.0035, 0.03));
    let leaf_mesh = meshes.add(Sphere::new(0.5));
    let stem_mat = materials.add(Color::srgb(0.26, 0.15, 0.07));
    let leaf_mat = materials.add(Color::srgb(0.2, 0.5, 0.14));
    let reds = [
        Color::srgb(0.78, 0.08, 0.07),
        Color::srgb(0.66, 0.05, 0.06),
        Color::srgb(0.85, 0.22, 0.08),
    ];
    for (i, (x, z)) in [
        (5.6_f32, 1.0_f32),
        (6.1, 2.2),
        (5.9, -0.6),
        (6.8, 0.4),
        (7.1, 3.1),
        (6.4, 4.2),
    ]
    .into_iter()
    .enumerate()
    {
        let body_mat = materials.add(StandardMaterial {
            base_color: reds[i % reds.len()],
            perceptual_roughness: 0.35,
            ..default()
        });
        let y = land_surface_height(x, z) + APPLE_RADIUS + 0.02;
        commands
            .spawn((
                SpatialBundle::from_transform(
                    Transform::from_xyz(x, y, z)
                        .with_rotation(Quat::from_rotation_y(i as f32 * 1.7)),
                ),
                RigidBody::Dynamic,
                Collider::ball(APPLE_RADIUS * 0.97),
                ColliderMassProperties::Mass(0.2),
                Velocity::default(),
                Restitution::coefficient(0.35),
                Friction::coefficient(0.8),
                Sleeping::disabled(),
                Apple,
            ))
            .with_children(|apple| {
                // slightly squashed body
                apple.spawn(PbrBundle {
                    mesh: body_mesh.clone(),
                    material: body_mat,
                    transform: Transform::from_scale(Vec3::new(1.0, 0.92, 1.0)),
                    ..default()
                });
                // stem
                apple.spawn(PbrBundle {
                    mesh: stem_mesh.clone(),
                    material: stem_mat.clone(),
                    transform: Transform::from_xyz(0.0, APPLE_RADIUS * 0.92 + 0.008, 0.0),
                    ..default()
                });
                // leaf
                apple.spawn(PbrBundle {
                    mesh: leaf_mesh.clone(),
                    material: leaf_mat.clone(),
                    transform: Transform::from_xyz(0.014, APPLE_RADIUS * 0.92 + 0.016, 0.0)
                        .with_rotation(Quat::from_rotation_z(-0.4))
                        .with_scale(Vec3::new(0.034, 0.005, 0.02)),
                    ..default()
                });
            });
    }
}

fn spawn_twigs(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let mesh = meshes.add(Capsule3d::new(STICK_RADIUS, STICK_HALF * 2.0));
    let wood = [
        materials.add(Color::srgb(0.29, 0.16, 0.075)),
        materials.add(Color::srgb(0.39, 0.23, 0.12)),
        materials.add(Color::srgb(0.34, 0.19, 0.09)),
    ];
    let mut spawned = 0;
    for seed in 0..500_u32 {
        let x = -37.0 + pseudo_rand(seed.wrapping_mul(97) + 6101) * 74.0;
        let z = -37.0 + pseudo_rand(seed.wrapping_mul(113) + 7103) * 74.0;
        if (x.abs() < 7.0 && z.abs() < 7.0)
            || (z < -26.0 && x > -15.0)
            || ((x - river_center_x(z)).abs() < RIVER_WIDTH * 0.5 + 1.5)
        {
            continue;
        }

        spawn_stick(
            commands,
            &mesh,
            &wood[(seed as usize) % wood.len()],
            Vec3::new(x, land_surface_height(x, z) + STICK_RADIUS + 0.01, z),
            pseudo_rand(seed.wrapping_add(8111)) * std::f32::consts::TAU,
        );
        spawned += 1;
        if spawned >= 44 {
            break;
        }
    }
    // a few close to the house, so they are easy to find
    for (i, (x, z)) in [(8.5_f32, 1.5_f32), (9.4, -0.8), (7.6, -2.2), (10.2, 2.6), (8.0, 4.0)]
        .into_iter()
        .enumerate()
    {
        spawn_stick(
            commands,
            &mesh,
            &wood[i % wood.len()],
            Vec3::new(x, land_surface_height(x, z) + STICK_RADIUS + 0.01, z),
            i as f32 * 1.3,
        );
    }
}

/// A stick: a light physics object you can pick up like a pebble, hold, rub for fire, or burn.
fn spawn_stick(
    commands: &mut Commands,
    mesh: &Handle<Mesh>,
    material: &Handle<StandardMaterial>,
    position: Vec3,
    yaw: f32,
) {
    commands.spawn((
        PbrBundle {
            mesh: mesh.clone(),
            material: material.clone(),
            transform: Transform::from_translation(position).with_rotation(
                Quat::from_rotation_y(yaw) * Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
            ),
            ..default()
        },
        RigidBody::Dynamic,
        Collider::capsule_y(STICK_HALF, STICK_RADIUS),
        ColliderMassProperties::Density(500.0),
        Velocity::default(),
        Restitution::coefficient(0.1),
        Friction::coefficient(0.9),
        Sleeping::disabled(),
        Stick { equipped: false },
    ));
}

fn spawn_pet(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &mut Assets<StandardMaterial>,
    pos: Vec3,
    kind: PetKind,
    seed: u32,
) {
    let cat = matches!(kind, PetKind::Cat);
    let coat = mats.add(if cat {
        Color::srgb(0.72, 0.43, 0.2)
    } else {
        Color::srgb(0.58, 0.34, 0.16)
    });
    let muzzle = mats.add(if cat {
        Color::srgb(0.9, 0.75, 0.62)
    } else {
        Color::srgb(0.82, 0.68, 0.49)
    });
    let dark = mats.add(Color::srgb(0.12, 0.09, 0.07));
    let body_mesh = meshes.add(Sphere::new(0.5));
    let head_mesh = meshes.add(Sphere::new(if cat { 0.19 } else { 0.23 }));
    let leg_mesh = meshes.add(Cuboid::new(0.09, if cat { 0.2 } else { 0.25 }, 0.1));
    let tail_mesh = meshes.add(Capsule3d::new(0.035, if cat { 0.42 } else { 0.2 }));
    let body_scale = if cat {
        Vec3::new(0.42, 0.25, 0.62)
    } else {
        Vec3::new(0.5, 0.34, 0.68)
    };
    let body_height = if cat { 0.27 } else { 0.34 };
    let head_z = if cat { -0.34 } else { -0.37 };
    commands
        .spawn((
            SpatialBundle::from_transform(Transform::from_translation(pos)),
            Pet {
                kind,
                activity: PetActivity::Walking,
                target: pos,
                timer: 1.0 + pseudo_rand(seed) * 2.0,
                phase: pseudo_rand(seed.wrapping_add(4)) * std::f32::consts::TAU,
                seed,
            },
        ))
        .with_children(|parent| {
            parent.spawn((
                PbrBundle {
                    mesh: body_mesh.clone(),
                    material: coat.clone(),
                    transform: Transform {
                        translation: Vec3::Y * body_height,
                        scale: body_scale,
                        ..default()
                    },
                    ..default()
                },
                PetBody,
            ));
            parent.spawn(PbrBundle {
                mesh: head_mesh.clone(),
                material: coat.clone(),
                transform: Transform::from_xyz(0.0, body_height + 0.08, head_z),
                ..default()
            });
            parent.spawn(PbrBundle {
                mesh: meshes.add(Sphere::new(0.07)),
                material: muzzle.clone(),
                transform: Transform::from_xyz(0.0, body_height + 0.015, head_z - 0.16),
                ..default()
            });
            for x in [-0.12_f32, 0.12] {
                parent.spawn(PbrBundle {
                    mesh: meshes.add(Sphere::new(0.035)),
                    material: dark.clone(),
                    transform: Transform::from_xyz(x, body_height + 0.1, head_z - 0.16),
                    ..default()
                });
            }

            if cat {
                for x in [-0.115_f32, 0.115] {
                    parent.spawn(PbrBundle {
                        mesh: meshes.add(Cone {
                            radius: 0.075,
                            height: 0.19,
                        }),
                        material: coat.clone(),
                        transform: Transform::from_xyz(x, body_height + 0.29, head_z),
                        ..default()
                    });
                }
            } else {
                for x in [-0.19_f32, 0.19] {
                    parent.spawn(PbrBundle {
                        mesh: meshes.add(Sphere::new(0.1)),
                        material: muzzle.clone(),
                        transform: Transform::from_xyz(x, body_height + 0.02, head_z + 0.04),
                        ..default()
                    });
                }
            }
            for (index, x, z) in [
                (0, -0.13, -0.19),
                (1, 0.13, -0.19),
                (2, -0.13, 0.2),
                (3, 0.13, 0.2),
            ] {
                parent.spawn((
                    PbrBundle {
                        mesh: leg_mesh.clone(),
                        material: coat.clone(),
                        transform: Transform::from_xyz(x, 0.13, z),
                        ..default()
                    },
                    PetLeg(index),
                ));
            }
            parent.spawn((
                PbrBundle {
                    mesh: tail_mesh.clone(),
                    material: coat.clone(),
                    transform: Transform::from_xyz(0.0, body_height + 0.03, 0.37)
                        .with_rotation(Quat::from_rotation_x(1.0)),
                    ..default()
                },
                PetTail,
            ));
        });
}


/// Scatters thin falling streaks outside the house footprint for the storm.
fn spawn_rain(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &mut Assets<StandardMaterial>,
) {
    let mesh = meshes.add(Cuboid::new(0.02, 0.5, 0.02));
    let mat = mats.add(StandardMaterial {
        base_color: Color::srgba(0.75, 0.82, 0.9, 0.5),
        unlit: true,
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    let half = FOOT / 2.0 + 0.3; // keep rain outside the walls
    for i in 0..450u32 {
        let mut x = pseudo_rand(i * 3 + 1) * 32.0 - 16.0;
        let mut z = pseudo_rand(i * 3 + 2) * 32.0 - 16.0;
        if x.abs() < half && z.abs() < half {
            if x.abs() < z.abs() {
                x = if x < 0.0 { -half - 1.5 } else { half + 1.5 };
            } else {
                z = if z < 0.0 { -half - 1.5 } else { half + 1.5 };
            }
        }
        let y = pseudo_rand(i * 3 + 3) * 14.0 + 2.0;
        let speed = 9.0 + pseudo_rand(i * 7 + 5) * 4.0;
        commands.spawn((
            PbrBundle {
                mesh: mesh.clone(),
                material: mat.clone(),
                transform: Transform::from_xyz(x, y, z),
                visibility: Visibility::Hidden,
                ..default()
            },
            Raindrop { speed },
        ));
    }
}

/// Cycles through daylight and night, with occasional timed rainstorms.
fn climate_system(
    time: Res<Time>,
    mut climate: ResMut<Climate>,
    mut clear: ResMut<ClearColor>,
    mut ambient: ResMut<AmbientLight>,
    mut daylight: Query<&mut DirectionalLight, With<Daylight>>,
    mut rain: Query<&mut Visibility, With<Raindrop>>,
) {
    let dt = time.delta_seconds();
    climate.day_phase = (climate.day_phase + dt / (DAY_PHASE_SECONDS * 2.0)).fract();
    if climate.storm {
        climate.storm_remaining -= dt;
        if climate.storm_remaining <= 0.0 {
            climate.storm = false;
            climate.weather_timer = WEATHER_PHASE_SECONDS;
        }
    } else {
        climate.weather_timer -= dt;
        if climate.weather_timer <= 0.0 {
            climate.storm = true;
            climate.storm_remaining = WEATHER_PHASE_SECONDS;
        }
    }

    let daylight_t =
        ((std::f32::consts::TAU * climate.day_phase - std::f32::consts::FRAC_PI_2).sin() + 1.0)
            * 0.5;
    let daylight_t = daylight_t * daylight_t * (3.0 - 2.0 * daylight_t);
    let light_factor = if climate.storm { 0.32 } else { 1.0 };
    for mut light in &mut daylight {
        light.illuminance = (1500.0 + daylight_t * 68_000.0) * light_factor;
        light.color = Color::srgb(
            0.66 + daylight_t * 0.34,
            0.73 + daylight_t * 0.23,
            0.9 + daylight_t * 0.1,
        );
    }

    let storm_dim = if climate.storm { 0.48 } else { 1.0 };
    ambient.brightness = (35.0 + daylight_t * 260.0) * storm_dim;
    ambient.color = Color::srgb(
        0.24 + daylight_t * 0.48,
        0.3 + daylight_t * 0.42,
        0.48 + daylight_t * 0.4,
    );
    let sky_day = daylight_t * if climate.storm { 0.35 } else { 1.0 };
    let sky = Vec3::new(
        0.018 + sky_day * 0.34,
        0.025 + sky_day * 0.55,
        0.08 + sky_day * 0.78,
    );
    clear.0 = Color::srgb(sky.x, sky.y, sky.z);

    for mut visibility in &mut rain {
        *visibility = if climate.storm {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
}

/// Lightning flashes briefly at random intervals during storms.
fn lightning_flicker(
    time: Res<Time>,
    climate: Res<Climate>,
    mut local: Local<(f32, f32)>, // (seconds until next flash, seconds of flash left)
    mut lights: Query<&mut DirectionalLight, With<Lightning>>,
) {
    let dt = time.delta_seconds();
    let (ref mut until_next, ref mut flash_left) = *local;
    if !climate.storm {
        *flash_left = 0.0;
        *until_next = 0.0;
        for mut light in &mut lights {
            light.illuminance = 0.0;
        }
        return;
    }
    if *until_next <= 0.0 && *flash_left <= 0.0 {
        *until_next = 3.0; // first flash a few seconds after launch
    }
    for mut light in &mut lights {
        if *flash_left > 0.0 {
            *flash_left -= dt;
            if *flash_left <= 0.0 {
                light.illuminance = 0.0;
            }
        } else {
            *until_next -= dt;
            if *until_next <= 0.0 {
                let r = pseudo_rand((time.elapsed_seconds() * 1000.0) as u32);
                light.illuminance = 60_000.0 + r * 40_000.0;
                *flash_left = 0.06 + r * 0.08;
                *until_next = 2.5 + r * 7.0;
            }
        }
    }
}

/// Raindrops recycle at ground contact and leave small, temporary puddles.
fn update_rain(
    time: Res<Time>,
    climate: Res<Climate>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
    mut puddle_assets: Local<Option<(Handle<Mesh>, Handle<StandardMaterial>)>>,
    mut drops: Query<(&mut Transform, &Raindrop)>,
) {
    if !climate.storm {
        return;
    }
    let dt = time.delta_seconds();
    let (puddle_mesh, puddle_material) = puddle_assets.get_or_insert_with(|| {
        (
            meshes.add(Circle::new(1.0)),
            mats.add(StandardMaterial {
                base_color: Color::srgba(0.2, 0.38, 0.48, 0.72),
                unlit: true,
                alpha_mode: AlphaMode::Blend,
                ..default()
            }),
        )
    });

    for (mut t, d) in &mut drops {
        t.translation.y -= d.speed * dt;
        if t.translation.y <= GROUND_SURFACE_Y + 0.25 {
            let seed = (time.elapsed_seconds() * 1000.0) as u32
                ^ t.translation.x.to_bits().rotate_left(13)
                ^ t.translation.z.to_bits();
            let size = 0.055 + pseudo_rand(seed) * 0.09;
            let lifetime = 4.0 + pseudo_rand(seed.wrapping_add(1)) * 4.0;

            commands.spawn((
                PbrBundle {
                    mesh: puddle_mesh.clone(),
                    material: puddle_material.clone(),
                    transform: Transform {
                        translation: Vec3::new(
                            t.translation.x,
                            GROUND_SURFACE_Y + 0.006,
                            t.translation.z,
                        ),
                        rotation: Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
                        scale: Vec3::new(size, size, 1.0),
                    },
                    ..default()
                },
                Puddle {
                    remaining: lifetime,
                    lifetime,
                    size,
                },
            ));

            t.translation.y = 15.0;
        }
    }
}

/// Shrinks puddles as they evaporate, then removes them.
fn update_puddles(
    time: Res<Time>,
    mut commands: Commands,
    mut puddles: Query<(Entity, &mut Transform, &mut Puddle)>,
) {
    let dt = time.delta_seconds();
    for (entity, mut transform, mut puddle) in &mut puddles {
        puddle.remaining -= dt;
        if puddle.remaining <= 0.0 {
            commands.entity(entity).despawn();
            continue;
        }

        let scale = puddle.size * (puddle.remaining / puddle.lifetime).clamp(0.0, 1.0);
        transform.scale = Vec3::new(scale, scale, 1.0);
    }
}

/// Small gusts move individual leaves while leaving the branches still.
fn sway_oak_leaves(time: Res<Time>, mut leaves: Query<(&mut Transform, &LeafSway)>) {
    let elapsed = time.elapsed_seconds();
    for (mut transform, leaf) in &mut leaves {
        let wind = (elapsed * 1.25 + leaf.phase).sin() * 0.11
            + (elapsed * 0.7 + leaf.phase * 1.7).sin() * 0.045;
        transform.rotation = leaf.rest_rotation
            * Quat::from_rotation_z(wind)
            * Quat::from_rotation_x((elapsed * 1.05 + leaf.phase).cos() * 0.055);
    }
}

fn tree_lod_system(
    camera: Query<&GlobalTransform, With<Camera3d>>,
    mut trees: Query<(Entity, &GlobalTransform, &mut Tree)>,
    mut pieces: Query<(&TreeLodPiece, &mut Visibility)>,
) {
    let Ok(camera) = camera.get_single() else {
        return;
    };
    let camera_position = camera.translation();
    let mut changed = Vec::new();
    for (entity, transform, mut tree) in &mut trees {
        let distance = transform.translation().distance(camera_position);
        let far = if tree.far_lod {
            distance > 24.0
        } else {
            distance > 30.0
        };
        if far != tree.far_lod {
            tree.far_lod = far;
            changed.push((entity, far));
        }
    }
    for (piece, mut visibility) in &mut pieces {
        if let Some((_, far)) = changed.iter().find(|(tree, _)| *tree == piece.tree) {
            let should_show = piece.high_detail != *far;
            let next = if should_show {
                Visibility::Visible
            } else {
                Visibility::Hidden
            };
            if *visibility != next {
                *visibility = next;
            }
        }
    }
}

fn tree_chip_system(
    time: Res<Time>,
    mut commands: Commands,
    mut chips: Query<(Entity, &mut TreeChip)>,
) {
    let dt = time.delta_seconds();
    for (entity, mut chip) in &mut chips {
        chip.remaining -= dt;
        if chip.remaining <= 0.0 {
            commands.entity(entity).despawn_recursive();
        }
    }
}

fn pullable_nature_system(
    mut commands: Commands,
    children: Query<&Children>,
    grab: Res<Grab>,
    mut objects: Query<(
        Entity,
        &mut Transform,
        &mut Velocity,
        &mut GravityScale,
        Option<&mut PullablePlant>,
        Option<&mut TreeBranch>,
    )>,
) {
    for (entity, mut transform, mut velocity, mut gravity, plant, branch) in &mut objects {
        if let Some(mut plant) = plant {
            if !plant.uprooted && transform.translation.distance(plant.anchor) > 0.45 {
                plant.uprooted = true;
            }
            if plant.uprooted && grab.entity != Some(entity) {
                transform.translation.y =
                    land_surface_height(transform.translation.x, transform.translation.z);
                transform.rotation = Quat::IDENTITY;
                velocity.linvel = Vec3::ZERO;
                velocity.angvel = Vec3::ZERO;
                gravity.0 = 0.0;
            }
        }
        if let Some(mut branch) = branch {
            if !branch.broken && transform.translation.distance(branch.anchor) > 0.7 {
                branch.broken = true;
                gravity.0 = 1.0;
                commands.entity(entity).remove::<Sensor>();
                if let Ok(children) = children.get(entity) {
                    for child in children.iter() {
                        commands.entity(*child).remove::<Sensor>();
                    }
                }
            }
        }
    }
}

/// A wall running along Z (normal along X) with a door-shaped opening in its middle.
#[allow(clippy::too_many_arguments)]
fn wall_with_door(
    commands: &mut Commands,
    batch: &mut StaticBoxBatch,
    center: Vec3,
    full_len: f32,
    full_h: f32,
    thickness: f32,
    door_w: f32,
    door_h: f32,
    color: Color,
) {
    let side_len = (full_len - door_w) / 2.0;
    for sign in [-1.0_f32, 1.0] {
        static_box(
            commands,
            batch,
            center + Vec3::new(0.0, 0.0, sign * (door_w / 2.0 + side_len / 2.0)),
            Vec3::new(thickness, full_h, side_len),
            color,
        );
    }
    let top_h = full_h - door_h;
    let wall_bottom = center.y - full_h / 2.0;
    static_box(
        commands,
        batch,
        Vec3::new(center.x, wall_bottom + door_h + top_h / 2.0, center.z),
        Vec3::new(thickness, top_h, door_w),
        color,
    );
}

/// Hold the left mouse button on a door handle and drag: the door swings so its
/// free edge follows your cursor (outward only).
fn door_grab_system(
    player: Query<Entity, With<Player>>,
    mut consumed: ResMut<ClickConsumed>,
    mut dg: ResMut<DoorGrab>,
    mut door: ResMut<Door>,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    rapier: Res<RapierContext>,
    handles: Query<(), With<DoorHandle>>,
) {
    let Ok(window) = windows.get_single() else {
        return;
    };
    let Ok((camera, cam_tf)) = cameras.get_single() else {
        return;
    };
    let ray = pointer_position(&window)
        .and_then(|c| camera.viewport_to_world(cam_tf, c));

    // Is the cursor on a handle (and close enough to reach it)?
    let filter = world_ray_filter(player.get_single().ok());
    dg.hover = ray.map_or(false, move |r| {
        rapier
            .cast_ray(r.origin, *r.direction, 100.0, true, filter)
            .map_or(false, |(e, toi)| toi < 6.0 && handles.contains(e))
    });

    if buttons.just_pressed(MouseButton::Left) && dg.hover && !consumed.0 {
        dg.active = true;
    }
    if !buttons.pressed(MouseButton::Left) {
        dg.active = false;
    }

    if dg.active {
        consumed.0 = true; // don't let grab_system grab something behind the door
        if let Some(r) = ray {
            let d = *r.direction;
            // Where does the cursor ray cross the handle's height (y = 1 m)?
            if d.y.abs() > 1e-4 {
                let t = (1.0 - r.origin.y) / d.y;
                if t > 0.0 {
                    let p = r.origin + d * t;
                    // Direction from the hinge to that point = where the door should point.
                    let v = Vec2::new(p.x - FOOT / 2.0, p.z + DOOR_W / 2.0);
                    if v.length() > 0.15 {
                        let target = v.x.atan2(v.y).clamp(0.0, 1.75);
                        door.angle += (target - door.angle).clamp(-0.3, 0.3);
                    }
                }
            }
        }
    }
}

/// Applies the door angle to the leaf and works out whether you can walk through.
fn door_update(
    mut door: ResMut<Door>,
    cam: Query<&Transform, (With<Camera3d>, Without<DoorPivot>)>,
    mut pivot: Query<&mut Transform, With<DoorPivot>>,
) {
    // The door won't swing shut on you while you're standing in the doorway.
    if let Ok(c) = cam.get_single() {
        let p = c.translation;
        let in_band = p.z.abs() < 0.5 && p.x > 3.78 && p.x < 4.82 && p.y < DOOR_H - 0.1;
        if in_band && door.angle < 1.45 {
            door.angle = 1.45;
        }
    }
    door.open = door.angle > 1.4; // roughly 80 degrees: the leaf is clear of the opening
    for mut t in &mut pivot {
        t.rotation = Quat::from_rotation_y(door.angle);
    }
}

fn pick_waypoint(floor_y: f32, seed: u32) -> Vec3 {
    for k in 0..12u32 {
        let x = pseudo_rand(seed.wrapping_add(k * 2 + 1)) * 6.8 - 3.4;
        let z = pseudo_rand(seed.wrapping_add(k * 2 + 2).wrapping_mul(31)) * 6.8 - 3.4;
        let blocked = (x < -1.5 && z > -3.1 && z < -1.9)      // table
            || (x > 1.8 && x < 3.2 && z > -2.6 && z < -1.4)   // crates
            || (x < -2.5 && z > -0.5); // stairs / stairwell
        if !blocked {
            return Vec3::new(x, floor_y + NPC_CENTER_Y, z);
        }
    }
    Vec3::new(0.0, floor_y + NPC_CENTER_Y, 0.0)
}

/// People pause, pick a spot, walk to it, repeat. They stop and face you when you talk to them.
fn npc_walk_system(
    time: Res<Time>,
    dialogue: Res<DialogueState>,
    poss: Res<Possession>,
    cam: Query<&Transform, (With<Camera3d>, Without<Npc>)>,
    mut npcs: Query<(Entity, &mut Transform, &mut Walker, Option<&Hunter>), With<Npc>>,
) {
    let dt = time.delta_seconds();
    let cam_pos = cam.get_single().map(|t| t.translation).ok();
    let yaw_to = |dir: Vec3| Quat::from_rotation_y((-dir.x).atan2(-dir.z));

    for (e, mut tf, mut w, hunter) in &mut npcs {
        if poss.target == Some(e) {
            continue; // you are him now
        }
        if dialogue.entity == Some(e) {
            if let Some(c) = cam_pos {
                let mut d = c - tf.translation;
                d.y = 0.0;
                if d.length() > 0.01 {
                    tf.rotation = yaw_to(d.normalize());
                }
            }
            continue;
        }
        // While he is busy feeding himself (or still outside), the hunter brain moves him.
        if let Some(hu) = hunter {
            if hu.state != HState::Idle || !hunter_inside(tf.translation) {
                continue;
            }
        }
        if w.wait > 0.0 {
            w.wait -= dt;
            continue;
        }
        let mut to = w.target - tf.translation;
        to.y = 0.0;
        let dist = to.length();
        if dist < 0.1 {
            w.seed = w.seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            w.target = pick_waypoint(w.floor_y, w.seed);
            w.wait = 1.0 + pseudo_rand(w.seed) * 3.5;
            continue;
        }
        let dir = to / dist;
        tf.translation += dir * (w.speed * dt).min(dist);
        tf.rotation = yaw_to(dir);
    }
}

fn pet_system(
    time: Res<Time>,
    mut pets: Query<
        (Entity, &mut Pet, &mut Transform),
        (Without<PetBody>, Without<PetLeg>, Without<PetTail>),
    >,
    mut bodies: Query<
        (&Parent, &mut Transform),
        (
            With<PetBody>,
            Without<Pet>,
            Without<PetLeg>,
            Without<PetTail>,
        ),
    >,
    mut legs: Query<
        (&Parent, &PetLeg, &mut Transform),
        (
            With<PetLeg>,
            Without<Pet>,
            Without<PetBody>,
            Without<PetTail>,
        ),
    >,
    mut tails: Query<
        (&Parent, &mut Transform),
        (
            With<PetTail>,
            Without<Pet>,
            Without<PetBody>,
            Without<PetLeg>,
        ),
    >,
) {
    let dt = time.delta_seconds();
    let mut states = Vec::new();

    for (entity, mut pet, mut transform) in &mut pets {
        pet.phase += dt;
        pet.timer -= dt;
        if pet.timer <= 0.0 {
            pet.seed = pet.seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let choice = pseudo_rand(pet.seed);
            if choice < 0.34 {
                pet.activity = PetActivity::Sitting;
                pet.timer = 2.0 + pseudo_rand(pet.seed.wrapping_add(1)) * 4.0;
            } else {
                pet.activity = if choice > 0.83 {
                    PetActivity::Running
                } else {
                    PetActivity::Walking
                };
                pet.target = Vec3::new(
                    pseudo_rand(pet.seed.wrapping_add(2)) * 5.8 - 2.9,
                    transform.translation.y,
                    pseudo_rand(pet.seed.wrapping_add(3)) * 5.8 - 2.9,
                );
                pet.timer = 2.0 + pseudo_rand(pet.seed.wrapping_add(4)) * 4.0;
            }
        }

        if pet.activity != PetActivity::Sitting {
            let mut direction = pet.target - transform.translation;
            direction.y = 0.0;
            let distance = direction.length();
            if distance < 0.12 {
                pet.activity = PetActivity::Sitting;
                pet.timer = 1.5 + pseudo_rand(pet.seed.wrapping_add(5)) * 2.0;
            } else {
                let speed = match (pet.kind, pet.activity) {
                    (PetKind::Cat, PetActivity::Running) => 2.3,
                    (PetKind::Cat, _) => 1.15,
                    (PetKind::Dog, PetActivity::Running) => 3.0,
                    (PetKind::Dog, _) => 1.55,
                };
                direction /= distance;
                transform.translation += direction * (speed * dt).min(distance);
                transform.rotation = Quat::from_rotation_y((-direction.x).atan2(-direction.z));
            }
        }

        let moving = pet.activity != PetActivity::Sitting;
        transform.translation.y = if moving {
            (pet.phase * 11.0).sin().abs() * 0.035
        } else {
            0.0
        };
        states.push((
            entity,
            pet.activity,
            pet.phase,
            matches!(pet.kind, PetKind::Cat),
        ));
    }

    for (parent, mut transform) in &mut bodies {
        if let Some((_, activity, _, cat)) = states.iter().find(|state| state.0 == parent.get()) {
            let height = if *cat { 0.27 } else { 0.34 };
            transform.translation.y = if *activity == PetActivity::Sitting {
                height * 0.72
            } else {
                height
            };
            transform.scale.y = if *activity == PetActivity::Sitting {
                if *cat {
                    0.21
                } else {
                    0.28
                }
            } else if *cat {
                0.25
            } else {
                0.34
            };
        }
    }
    for (parent, leg, mut transform) in &mut legs {
        if let Some((_, activity, phase, _)) = states.iter().find(|state| state.0 == parent.get()) {
            let stride = match activity {
                PetActivity::Sitting => 0.0,
                PetActivity::Walking => 0.32,
                PetActivity::Running => 0.75,
            };
            let pair_phase = if leg.0 % 2 == 0 {
                0.0
            } else {
                std::f32::consts::PI
            };
            let leg_speed = if stride > 0.5 { 13.0 } else { 7.0 };
            transform.rotation =
                Quat::from_rotation_x((*phase * leg_speed + pair_phase).sin() * stride);
        }
    }
    for (parent, mut transform) in &mut tails {
        if let Some((_, activity, phase, cat)) = states.iter().find(|state| state.0 == parent.get())
        {
            let wag = (*phase * if *cat { 3.0 } else { 8.0 }).sin();
            transform.rotation = Quat::from_rotation_x(1.0)
                * Quat::from_rotation_y(
                    wag * if *activity == PetActivity::Sitting {
                        0.3
                    } else {
                        0.14
                    },
                );
        }
    }
}

/// Ray filter for "what is in the world under the cursor": ignores your own
/// capsule and held weapons so they never block clicks.
fn world_ray_filter(player: Option<Entity>) -> QueryFilter<'static> {
    let mut f = QueryFilter::default().groups(CollisionGroups::new(
        Group::ALL,
        Group::ALL & !Group::GROUP_2,
    ));
    if let Some(pe) = player {
        f = f.exclude_collider(pe);
    }
    f
}

/// A sword built from boxes: grip, pommel, crossguard, blade and tip. It starts
/// lying flat at `pos` as a normal physics object.
fn spawn_sword(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &mut Assets<StandardMaterial>,
    pos: Vec3,
) {
    let steel = mats.add(StandardMaterial {
        base_color: Color::srgb(0.78, 0.8, 0.86),
        metallic: 0.9,
        perceptual_roughness: 0.25,
        ..default()
    });
    let gold = mats.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.68, 0.2),
        metallic: 0.8,
        perceptual_roughness: 0.35,
        ..default()
    });
    let leather = mats.add(Color::srgb(0.25, 0.14, 0.08));
    let groups = CollisionGroups::new(Group::GROUP_2, Group::ALL);

    // blade along +X, flat side up: local Y -> world X, local X -> world Y, local Z -> world -Z
    let lying = Quat::from_mat3(&Mat3::from_cols(Vec3::Y, Vec3::X, Vec3::NEG_Z));

    commands
        .spawn((
            SpatialBundle::from_transform(Transform::from_translation(pos).with_rotation(lying)),
            RigidBody::Dynamic,
            Velocity::default(),
            Restitution::coefficient(0.1),
            Friction::coefficient(0.8),
            Sleeping::disabled(),
            Sword {
                equipped: false,
                psi: SWORD_REST_PSI,
                phi: SWORD_REST_PHI,
                psi_vel: 0.0,
                roll: 0.0,
            },
        ))
        .with_children(|p| {
            // grip (origin of the sword is the middle of the grip)
            p.spawn((
                PbrBundle {
                    mesh: meshes.add(Cuboid::new(0.035, 0.18, 0.035)),
                    material: leather,
                    ..default()
                },
                Collider::cuboid(0.0175, 0.09, 0.0175),
                groups,
                SwordHandle,
            ));
            // pommel
            p.spawn(PbrBundle {
                mesh: meshes.add(Sphere::new(0.03)),
                material: gold.clone(),
                transform: Transform::from_xyz(0.0, -0.11, 0.0),
                ..default()
            });
            // crossguard (spans local Z)
            p.spawn((
                PbrBundle {
                    mesh: meshes.add(Cuboid::new(0.035, 0.03, 0.24)),
                    material: gold,
                    transform: Transform::from_xyz(0.0, 0.105, 0.0),
                    ..default()
                },
                Collider::cuboid(0.0175, 0.015, 0.12),
                groups,
            ));
            // blade: thin along X, wide along Z, 0.9 m long along Y
            p.spawn((
                PbrBundle {
                    mesh: meshes.add(Cuboid::new(0.012, 0.9, 0.055)),
                    material: steel.clone(),
                    transform: Transform::from_xyz(0.0, 0.57, 0.0),
                    ..default()
                },
                Collider::cuboid(0.006, 0.45, 0.0275),
                groups,
            ));
            // narrower tip
            p.spawn(PbrBundle {
                mesh: meshes.add(Cuboid::new(0.01, 0.1, 0.035)),
                material: steel,
                transform: Transform::from_xyz(0.0, 1.07, 0.0),
                ..default()
            });
        });
}

fn oak_leaf_mesh() -> Mesh {
    let rows = 17_u32;
    let mut positions = Vec::with_capacity((rows * 3) as usize);
    let mut normals = Vec::with_capacity((rows * 3) as usize);
    let mut colors = Vec::with_capacity((rows * 3) as usize);
    let mut uvs = Vec::with_capacity((rows * 3) as usize);
    let mut indices = Vec::with_capacity(((rows - 1) * 12) as usize);

    for row in 0..rows {
        let t = row as f32 / (rows - 1) as f32;
        let envelope = (std::f32::consts::PI * t).sin().max(0.0).powf(0.72);
        let lobes = 0.68 + 0.32 * (t * std::f32::consts::PI * 10.0).cos();
        let half_width = 0.42 * envelope * lobes;
        let y = t * 1.35;
        for (column, x) in [-half_width, 0.0, half_width].into_iter().enumerate() {
            positions.push([x, y, 0.0]);
            normals.push([0.0, 0.0, 1.0]);
            uvs.push([column as f32 * 0.5, t]);
            let green = if column == 1 { 0.58 } else { 1.0 };
            colors.push([0.48 * green, 0.78 * green, 0.34 * green, 1.0]);
        }
    }

    for row in 0..rows - 1 {
        let current = row * 3;
        let next = current + 3;
        indices.extend_from_slice(&[
            current,
            current + 1,
            next,
            current + 1,
            next + 1,
            next,
            current + 1,
            current + 2,
            next + 1,
            current + 2,
            next + 2,
            next + 1,
        ]);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(bevy::render::mesh::Indices::U32(indices));
    mesh
}

#[derive(Clone, Copy)]
struct OakBranchSegment {
    start: Vec3,
    end: Vec3,
    radius: f32,
    /// Index of the limb this one sprouts from (None for the limb that joins the trunk).
    parent: Option<usize>,
}

fn append_oak_branch(
    start: Vec3,
    direction: Vec3,
    length: f32,
    radius: f32,
    depth: u32,
    seed: u32,
    parent: Option<usize>,
    segments: &mut Vec<OakBranchSegment>,
    leaf_tips: &mut Vec<(Vec3, u32, usize)>,
) {
    let direction = direction.normalize_or_zero();
    let end = start + direction * length;
    let index = segments.len();
    segments.push(OakBranchSegment {
        start,
        end,
        radius,
        parent,
    });
    if depth == 0 {
        leaf_tips.push((end, seed, index));
        return;
    }

    let mut side = direction.cross(Vec3::Y).normalize_or_zero();
    if side.length_squared() < 0.01 {
        side = direction.cross(Vec3::X).normalize_or_zero();
    }
    let other_side = direction.cross(side).normalize_or_zero();
    let split = start.lerp(end, OAK_BRANCH_SPLIT);
    for child in 0..2 {
        let child_seed = seed.wrapping_mul(17).wrapping_add(child + 1);
        let sign = if child == 0 { -1.0 } else { 1.0 };
        let azimuth_jitter = (pseudo_rand(child_seed) - 0.5) * 0.6;
        let upward_jitter = (pseudo_rand(child_seed.wrapping_add(1)) - 0.5) * 0.3;
        let child_direction = (direction
            + side * (sign * (0.55 + azimuth_jitter))
            + other_side * (sign * 0.22)
            + Vec3::Y * (0.38 + upward_jitter))
            .normalize_or_zero();
        append_oak_branch(
            split,
            child_direction,
            length * (0.62 + pseudo_rand(child_seed.wrapping_add(2)) * 0.12),
            radius * 0.68,
            depth - 1,
            child_seed,
            Some(index),
            segments,
            leaf_tips,
        );
    }
}

fn trunk_radius_at(t: f32, base_radius: f32, top_radius: f32) -> f32 {
    let radius = base_radius + (top_radius - base_radius) * t;
    let flare = if t < 0.18 {
        ((0.18 - t) / 0.18).powf(1.4) * base_radius * 0.18
    } else {
        0.0
    };
    radius + flare
}

fn trunk_lean_at(t: f32, lean: Vec2) -> Vec2 {
    Vec2::new(lean.x * t * t, lean.y * t * t)
}

/// A V-shaped notch cut into a tree trunk (a real carve, not a decoration).
/// The trunk mesh is generated from y=0 (base) to y=height; angles around the
/// trunk's local Y run from 0 at +X to pi/2 at +Z. The notch opens toward
/// `phi`; its deepest point (the apex) is at `y_bottom`, and it tapers to
/// nothing at `y_bottom + height` and at the angular edges +/- `angular_half`.
#[derive(Clone, Copy)]
struct TrunkNotch {
    phi: f32,
    angular_half: f32,
    y_bottom: f32,
    height: f32,
    depth: f32,
}

impl TrunkNotch {
    /// How much to pull a vertex inward at this (angle, y). 0 = no cut.
    fn reduction_at(&self, angle: f32, y: f32) -> f32 {
        let mut da = (angle - self.phi).abs() % std::f32::consts::TAU;
        if da > std::f32::consts::PI {
            da = std::f32::consts::TAU - da;
        }
        if da >= self.angular_half {
            return 0.0;
        }
        if y < self.y_bottom || y > self.y_bottom + self.height {
            return 0.0;
        }
        // Linear falloff in angle (V) and height (V) = a wedge-shaped carve.
        let angular_factor = 1.0 - da / self.angular_half;
        let height_factor = (self.y_bottom + self.height - y) / self.height;
        self.depth * angular_factor * height_factor
    }
}

fn tapered_trunk_mesh(
    height: f32,
    base_radius: f32,
    top_radius: f32,
    lean: Vec2,
    notch: Option<TrunkNotch>,
    notch_vertex_color: [f32; 3],
) -> Mesh {
    tapered_trunk_section(
        height,
        0.0,
        height,
        base_radius,
        top_radius,
        lean,
        Vec2::ZERO,
        notch,
        notch_vertex_color,
    )
}

fn tapered_trunk_section(
    full_height: f32,
    y_start: f32,
    y_end: f32,
    base_radius: f32,
    top_radius: f32,
    lean: Vec2,
    origin_xz: Vec2,
    notch: Option<TrunkNotch>,
    notch_vertex_color: [f32; 3],
) -> Mesh {
    const SIDES: usize = 24;
    let section_h = (y_end - y_start).max(0.05);
    // With a notch we want plenty of rings so the V reads clearly.
    let rings = if let Some(n) = notch {
        let spacing = (n.height / 14.0).max(0.03);
        ((section_h / spacing).ceil() as usize).clamp(14, 80)
    } else {
        ((12.0 * (section_h / full_height.max(0.05))).ceil() as usize).clamp(3, 12)
    };
    let slope = (base_radius - top_radius) / full_height.max(0.05);

    let mut positions = Vec::with_capacity((rings + 1) * (SIDES + 1));
    let mut normals = Vec::with_capacity((rings + 1) * (SIDES + 1));
    let mut uvs = Vec::with_capacity((rings + 1) * (SIDES + 1));
    let mut colors = Vec::with_capacity((rings + 1) * (SIDES + 1));
    let mut indices = Vec::with_capacity(rings * SIDES * 6);

    for ring in 0..=rings {
        let t_section = ring as f32 / rings as f32;
        let y = y_start + section_h * t_section;
        let t = (y / full_height.max(0.05)).clamp(0.0, 1.0);
        let radius = trunk_radius_at(t, base_radius, top_radius);
        let center = trunk_lean_at(t, lean);
        for side in 0..=SIDES {
            let angle = side as f32 / SIDES as f32 * std::f32::consts::TAU;
            let (sin, cos) = angle.sin_cos();

            let (final_radius, normal, color) = match notch {
                Some(n) => {
                    let reduction = n.reduction_at(angle, y);
                    if reduction > 0.001 {
                        // Tilt the normal inward so the V reads under lighting.
                        let outward = Vec3::new(cos, slope, sin).normalize_or_zero();
                        let notch_dir = Vec3::new(n.phi.cos(), 0.0, n.phi.sin());
                        let normal = (outward - notch_dir * 0.5).normalize_or_zero();
                        (
                            (radius - reduction).max(0.02),
                            normal,
                            [
                                notch_vertex_color[0],
                                notch_vertex_color[1],
                                notch_vertex_color[2],
                            ],
                        )
                    } else {
                        (
                            radius,
                            Vec3::new(cos, slope, sin).normalize_or_zero(),
                            [1.0, 1.0, 1.0],
                        )
                    }
                }
                None => (
                    radius,
                    Vec3::new(cos, slope, sin).normalize_or_zero(),
                    [1.0, 1.0, 1.0],
                ),
            };

            positions.push([
                center.x - origin_xz.x + cos * final_radius,
                y - y_start,
                center.y - origin_xz.y + sin * final_radius,
            ]);
            normals.push(normal.to_array());
            uvs.push([side as f32 / SIDES as f32, t]);
            colors.push([color[0], color[1], color[2], 1.0]);
        }
    }

    for ring in 0..rings {
        for side in 0..SIDES {
            let a = (ring * (SIDES + 1) + side) as u32;
            let b = a + (SIDES + 1) as u32;
            indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(bevy::render::mesh::Indices::U32(indices));
    mesh
}

/// Colour of wood on a cut face at radius fraction `t`
/// (0 = pith at the centre, 1 = bark edge). Annual growth rings alternate
/// between lighter early-wood and darker late-wood; the pale outer band is
/// the sapwood, and just inside the bark there's a slightly darker
/// transition zone.
fn cut_wood_target_color(t: f32, ring_index: usize) -> Color {
    if t >= 0.96 {
        // Just inside the bark: darker transition zone.
        Color::srgb(0.45, 0.25, 0.10)
    } else if t >= 0.78 {
        // Sapwood: pale, living outer band.
        Color::srgb(0.86, 0.72, 0.46)
    } else if ring_index % 2 == 0 {
        // Heartwood early-wood (lighter, spring growth).
        Color::srgb(0.58, 0.34, 0.16)
    } else {
        // Heartwood late-wood (darker, summer growth ring).
        Color::srgb(0.40, 0.20, 0.08)
    }
}

/// Per-vertex multiplier so that `material_color` × this = `target`.
/// Used to bake the ring pattern into the cut disc while keeping the
/// existing `inner_wood` material in place.
fn cut_wood_vertex_ratio(target: Color, material_color: Color) -> [f32; 3] {
    let mat = material_color.to_linear().to_f32_array();
    let tgt = target.to_linear().to_f32_array();
    [
        tgt[0] / mat[0].max(1e-4),
        tgt[1] / mat[1].max(1e-4),
        tgt[2] / mat[2].max(1e-4),
    ]
}

/// A flat disc of cut trunk: pith at the centre, concentric annual rings,
/// a pale sapwood band near the edge, and a darker transition just inside
/// the bark. `material_color` is the base colour of the material the disc
/// will use; the vertex colours are calibrated so that they multiply back
/// to the intended wood colours.
#[allow(dead_code)]
fn trunk_cut_disc(
    radius: f32,
    offset: Vec2,
    upward: bool,
    material_color: Color,
) -> Mesh {
    const SIDES: usize = 32;
    const RINGS: usize = 16;

    let y_sign = if upward { 1.0 } else { -1.0 };
    let vertex_total = 1 + RINGS * (SIDES + 1);
    let mut positions = Vec::with_capacity(vertex_total);
    let mut normals = Vec::with_capacity(vertex_total);
    let mut uvs = Vec::with_capacity(vertex_total);
    let mut colors = Vec::with_capacity(vertex_total);
    let mut indices = Vec::with_capacity(SIDES * 3 + (RINGS - 1) * SIDES * 6);

    let color_at = |t: f32, ring: usize| -> [f32; 4] {
        let target = cut_wood_target_color(t, ring);
        let ratio = cut_wood_vertex_ratio(target, material_color);
        [ratio[0], ratio[1], ratio[2], 1.0]
    };

    // Pith (centre)
    positions.push([offset.x, 0.0, offset.y]);
    normals.push([0.0, y_sign, 0.0]);
    uvs.push([0.5, 0.5]);
    colors.push(color_at(0.0, 0));

    // Concentric rings outward from the pith.
    for ring in 1..=RINGS {
        let t = ring as f32 / RINGS as f32;
        let r = t * radius;
        let c = color_at(t, ring);
        for side in 0..=SIDES {
            let angle = side as f32 / SIDES as f32 * std::f32::consts::TAU;
            let (sin, cos) = angle.sin_cos();
            positions.push([offset.x + cos * r, 0.0, offset.y + sin * r]);
            normals.push([0.0, y_sign, 0.0]);
            uvs.push([cos * 0.5 + 0.5, sin * 0.5 + 0.5]);
            colors.push(c);
        }
    }

    // Fan from the centre to ring 1.
    for side in 0..SIDES {
        let a = 1 + side as u32;
        let b = 1 + (side + 1) as u32;
        if upward {
            indices.extend_from_slice(&[0, a, b]);
        } else {
            indices.extend_from_slice(&[0, b, a]);
        }
    }

    // Quad strips between consecutive rings.
    for ring in 0..RINGS - 1 {
        let inner = 1 + ring * (SIDES + 1);
        let outer = 1 + (ring + 1) * (SIDES + 1);
        for side in 0..SIDES {
            let a = (inner + side) as u32;
            let b = (inner + side + 1) as u32;
            let c = (outer + side) as u32;
            let d = (outer + side + 1) as u32;
            if upward {
                indices.extend_from_slice(&[a, b, c, b, d, c]);
            } else {
                indices.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(bevy::render::mesh::Indices::U32(indices));
    mesh
}

// ---------------------------------------------------------------------------
// Layered wood: a standing trunk is a grid of wood blocks. Rows stack upward
// (the grain runs up the tree), each row is split into sectors around the
// trunk and into concentric layers (outer sapwood -> heartwood -> core).
//
//  * An axe hit removes the outermost block of every column it touches. The
//    first hit is a small straight nick; hitting the same spot again goes one
//    layer deeper and the cut widens (a V), like a real notch.
//  * A block only stays if it is still connected (through other blocks) to the
//    ground. Anything cut loose falls off as its own physics chunk, and the
//    layer underneath becomes the new surface.
//  * Only faces between a block and empty space are ever built, so no two
//    surfaces share a plane (no z-fighting inside the trunk).
// ---------------------------------------------------------------------------

const WOOD_SECTORS: usize = 12;
const WOOD_LAYERS: usize = 4;
const WOOD_ROW_H: f32 = 0.15;
/// Outer radius of each layer as a fraction of the trunk radius (last entry = the axis).
const WOOD_LAYER_FRAC: [f32; WOOD_LAYERS + 1] = [1.0, 0.8, 0.55, 0.28, 0.0];
/// A loose piece needs at least this many blocks to count as the felled upper trunk.
const WOOD_FELL_MIN_CELLS: usize = 150;
const WOOD_MIN_CHOP_ROW: i32 = 4;
/// A piece of a chopped log needs at least this many blocks (about two rows) to stay a log.
const LOG_MIN_PIECE_CELLS: usize = 96;
/// A piece of a chopped limb needs at least this many blocks (half a ring).
const BRANCH_MIN_PIECE_CELLS: usize = 24;
/// When this little (0..1) of a row's cross-section is left, a log / limb snaps there.
const LOG_SNAP_AREA: f32 = 0.30;

#[derive(Clone, Copy)]
struct TrunkShape {
    height: f32,
    base_radius: f32,
    top_radius: f32,
    lean: Vec2,
}

impl TrunkShape {
    fn rows(&self) -> usize {
        ((self.height / WOOD_ROW_H).ceil() as usize).max(1)
    }

    fn center_at(&self, y: f32) -> Vec2 {
        trunk_lean_at((y / self.height.max(0.05)).clamp(0.0, 1.0), self.lean)
    }

    fn radius_at(&self, y: f32) -> f32 {
        trunk_radius_at(
            (y / self.height.max(0.05)).clamp(0.0, 1.0),
            self.base_radius,
            self.top_radius,
        )
    }

    /// Point at height `y`, angle `angle` around the trunk and `frac` of the radius.
    fn point(&self, y: f32, angle: f32, frac: f32) -> Vec3 {
        let center = self.center_at(y);
        let radius = self.radius_at(y) * frac;
        Vec3::new(
            center.x + angle.cos() * radius,
            y,
            center.y + angle.sin() * radius,
        )
    }

    fn slope(&self) -> f32 {
        (self.base_radius - self.top_radius) / self.height.max(0.05)
    }
}

/// Plain mesh buffers; converted to a Bevy `Mesh` by `wood_mesh_from_data`.
#[derive(Default)]
struct WoodMeshData {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    colors: Vec<[f32; 4]>,
    indices: Vec<u32>,
}

impl WoodMeshData {
    /// Adds one quad. `hint` is the direction the face must point; the winding is
    /// flipped if needed so it always faces that way.
    fn quad(&mut self, mut corners: [Vec3; 4], smooth: Option<[Vec3; 4]>, hint: Vec3, color: [f32; 4]) {
        let mut normal = (corners[2] - corners[0])
            .cross(corners[3] - corners[1])
            .normalize_or_zero();
        if normal == Vec3::ZERO {
            return;
        }
        let mut smooth = smooth;
        if normal.dot(hint) < 0.0 {
            corners.swap(1, 3);
            if let Some(s) = smooth.as_mut() {
                s.swap(1, 3);
            }
            normal = -normal;
        }
        let base = self.positions.len() as u32;
        for i in 0..4 {
            self.positions.push(corners[i].to_array());
            let n = smooth.map(|s| s[i]).unwrap_or(normal);
            self.normals.push(n.to_array());
            self.uvs
                .push([(corners[i].x + corners[i].z) * 0.5, corners[i].y * 0.5]);
            self.colors.push(color);
        }
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

fn wood_layer_weight(layer: usize) -> f32 {
    WOOD_LAYER_FRAC[layer] * WOOD_LAYER_FRAC[layer]
        - WOOD_LAYER_FRAC[layer + 1] * WOOD_LAYER_FRAC[layer + 1]
}

#[derive(Clone)]
struct TrunkWood {
    shape: TrunkShape,
    rows: usize,
    cells: Vec<bool>,
    /// (sector, row) where the current run of hits started, and how many hits so far.
    spot: Option<(i32, i32)>,
    spot_hits: u32,
}

impl TrunkWood {
    fn new(shape: TrunkShape) -> Self {
        let rows = shape.rows();
        Self {
            shape,
            rows,
            cells: vec![true; rows * WOOD_SECTORS * WOOD_LAYERS],
            spot: None,
            spot_hits: 0,
        }
    }

    fn index(row: usize, sector: usize, layer: usize) -> usize {
        (row * WOOD_SECTORS + sector) * WOOD_LAYERS + layer
    }

    /// (row, sector, layer) of a cell index.
    fn split(index: usize) -> (usize, usize, usize) {
        let layer = index % WOOD_LAYERS;
        let rest = index / WOOD_LAYERS;
        (rest / WOOD_SECTORS, rest % WOOD_SECTORS, layer)
    }

    /// Which (sector, row) of the trunk a point (in trunk-base coordinates) is on.
    fn locate(&self, local: Vec3) -> (i32, i32) {
        let center = self.shape.center_at(local.y);
        let angle = (local.z - center.y)
            .atan2(local.x - center.x)
            .rem_euclid(std::f32::consts::TAU);
        let sector = ((angle / std::f32::consts::TAU * WOOD_SECTORS as f32).floor() as i32)
            .rem_euclid(WOOD_SECTORS as i32);
        let max_row = (self.rows as i32 - 3).max(WOOD_MIN_CHOP_ROW);
        let row = ((local.y / WOOD_ROW_H).floor() as i32).clamp(WOOD_MIN_CHOP_ROW, max_row);
        (sector, row)
    }

    /// One axe hit at (`sector`, `row`). Returns the cells that were chopped away.
    ///
    /// Every hit removes the outermost remaining block of each column in its
    /// footprint. The footprint is a single column on the first hit (a small
    /// straight nick) and grows with every further hit on the same spot, so the
    /// centre goes deepest and the cut opens into a V.
    fn chop(&mut self, sector: i32, row: i32) -> Vec<usize> {
        self.chop_min(sector, row, WOOD_MIN_CHOP_ROW)
    }

    /// `chop`, but rows below `min_row` are never cut. The standing trunk keeps its base
    /// (`WOOD_MIN_CHOP_ROW`); a fallen log or a limb can be cut anywhere (0).
    fn chop_min(&mut self, sector: i32, row: i32, min_row: i32) -> Vec<usize> {
        let sectors = WOOD_SECTORS as i32;
        let same_spot = self.spot.is_some_and(|(s, r)| {
            let mut ds = (sector - s).rem_euclid(sectors);
            if ds > sectors / 2 {
                ds = sectors - ds;
            }
            ds <= 1 && (row - r).abs() <= 1
        });
        if same_spot {
            self.spot_hits += 1;
        } else {
            self.spot = Some((sector, row));
            self.spot_hits = 1;
        }
        let (center_sector, center_row) = self.spot.unwrap_or((sector, row));
        let hits = self.spot_hits as i32;
        let half = ((hits * 2) / 3).min(sectors / 2);
        let vertical = (hits - 1) / 5;

        let mut removed = Vec::new();
        for dr in -vertical..=vertical {
            let row_half = half - dr.abs() * 2;
            if row_half < 0 {
                continue;
            }
            let r = center_row + dr;
            if r < min_row || r >= self.rows as i32 {
                continue;
            }
            let mut columns: Vec<i32> = (-row_half..=row_half)
                .map(|d| (center_sector + d).rem_euclid(sectors))
                .collect();
            columns.sort_unstable();
            columns.dedup();
            for s in columns {
                for layer in 0..WOOD_LAYERS {
                    let i = Self::index(r as usize, s as usize, layer);
                    if self.cells[i] {
                        self.cells[i] = false;
                        removed.push(i);
                        break;
                    }
                }
            }
        }
        removed
    }

    /// Share (0..1) of the row's original cross-section that is still wood.
    fn row_area(&self, row: usize) -> f32 {
        let mut area = 0.0;
        for s in 0..WOOD_SECTORS {
            for l in 0..WOOD_LAYERS {
                if self.cells[Self::index(row, s, l)] {
                    area += wood_layer_weight(l);
                }
            }
        }
        area / WOOD_SECTORS as f32
    }

    /// Unit direction (horizontal) toward the side where wood has been cut away in `row`.
    fn removed_direction(&self, row: usize) -> Vec3 {
        let mut sum = Vec3::ZERO;
        for s in 0..WOOD_SECTORS {
            let angle = (s as f32 + 0.5) / WOOD_SECTORS as f32 * std::f32::consts::TAU;
            for l in 0..WOOD_LAYERS {
                if !self.cells[Self::index(row, s, l)] {
                    sum += Vec3::new(angle.cos(), 0.0, angle.sin()) * wood_layer_weight(l);
                }
            }
        }
        sum.normalize_or_zero()
    }

    /// Which (sector, row) a point (in the wood grid's own coordinates) is on. Unlike
    /// `locate`, every row can be hit: this is for fallen logs and limbs.
    fn locate_free(&self, local: Vec3) -> (i32, i32) {
        let center = self.shape.center_at(local.y);
        let angle = (local.z - center.y)
            .atan2(local.x - center.x)
            .rem_euclid(std::f32::consts::TAU);
        let sector = ((angle / std::f32::consts::TAU * WOOD_SECTORS as f32).floor() as i32)
            .rem_euclid(WOOD_SECTORS as i32);
        let row = ((local.y / WOOD_ROW_H).floor() as i32).clamp(0, self.rows as i32 - 1);
        (sector, row)
    }

    /// First and last row that still has any wood.
    fn row_extent(&self) -> Option<(usize, usize)> {
        let per_row = WOOD_SECTORS * WOOD_LAYERS;
        let has = |row: usize| (0..per_row).any(|k| self.cells[row * per_row + k]);
        let lo = (0..self.rows).find(|&row| has(row))?;
        let hi = (0..self.rows).rev().find(|&row| has(row))?;
        Some((lo, hi))
    }

    /// The row near `hit_row` where the cuts have eaten so much of the cross-section that the
    /// rest can't hold any more: the log / limb snaps there. Both halves must keep at
    /// least `min_cells` blocks. Rows `<= result` go to the lower half, the rest to the upper.
    fn snap_row(&self, hit_row: i32, min_cells: usize) -> Option<usize> {
        let (lo, hi) = self.row_extent()?;
        let from = (hit_row - 3).max(lo as i32);
        let to = (hit_row + 3).min(hi as i32);
        if to < from {
            return None;
        }
        let weak = ((from as usize)..=(to as usize)).min_by(|a, b| {
            self.row_area(*a)
                .partial_cmp(&self.row_area(*b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        if self.row_area(weak) > LOG_SNAP_AREA {
            return None;
        }
        let per_row = WOOD_SECTORS * WOOD_LAYERS;
        let split = (weak + 1) * per_row;
        let below = self.cells[..split].iter().filter(|&&c| c).count();
        let above = self.cells[split..].iter().filter(|&&c| c).count();
        (below >= min_cells && above >= min_cells).then_some(weak)
    }

    /// Splits the grid across the boundary below row `k` into (rows < k, rows >= k). The
    /// upper part starts again at its own row 0. Only for straight (non-tapered) wood.
    fn split_at_row(&self, k: usize) -> (TrunkWood, TrunkWood) {
        let per_row = WOOD_SECTORS * WOOD_LAYERS;
        let k = k.min(self.rows.saturating_sub(1)).max(1);
        let lower = TrunkWood {
            shape: TrunkShape {
                height: k as f32 * WOOD_ROW_H,
                ..self.shape
            },
            rows: k,
            cells: self.cells[..k * per_row].to_vec(),
            spot: None,
            spot_hits: 0,
        };
        let upper = TrunkWood {
            shape: TrunkShape {
                height: (self.shape.height - k as f32 * WOOD_ROW_H).max(0.05),
                ..self.shape
            },
            rows: self.rows - k,
            cells: self.cells[k * per_row..].to_vec(),
            spot: None,
            spot_hits: 0,
        };
        (lower, upper)
    }

    /// Mass of the blocks in `mask` (green oak).
    fn mass_of(&self, mask: &[bool]) -> f32 {
        let mut mass = 0.0;
        for (index, &present) in mask.iter().enumerate() {
            if !present {
                continue;
            }
            let (row, _, layer) = Self::split(index);
            let (y0, y1) = self.row_bounds(row);
            let radius = self.shape.radius_at((y0 + y1) * 0.5);
            mass += wood_layer_weight(layer) * std::f32::consts::PI * radius * radius
                / WOOD_SECTORS as f32
                * (y1 - y0)
                * 500.0;
        }
        mass
    }

    /// Centre of mass of the blocks in `mask`, in the wood grid's coordinates.
    fn centroid_of(&self, mask: &[bool]) -> Vec3 {
        let mut sum = Vec3::ZERO;
        let mut weight = 0.0;
        for (index, &present) in mask.iter().enumerate() {
            if !present {
                continue;
            }
            let (_, _, layer) = Self::split(index);
            let w = wood_layer_weight(layer);
            sum += self.cell_center(index) * w;
            weight += w;
        }
        if weight > 0.0 {
            sum / weight
        } else {
            Vec3::ZERO
        }
    }

    /// Collision shape of the blocks in `mask`: one convex hull per few rows around the
    /// outermost wood that is left, so a notch shows up in the collision too.
    /// `offset` is added to every point.
    fn collider_of(&self, mask: &[bool], offset: Vec3) -> Option<Collider> {
        const CHUNK_ROWS: usize = 3;
        let mut shapes = Vec::new();
        let mut row = 0;
        while row < self.rows {
            let end = (row + CHUNK_ROWS).min(self.rows);
            let mut points: Vec<Vec3> = Vec::new();
            for r in row..end {
                for s in 0..WOOD_SECTORS {
                    for l in 0..WOOD_LAYERS {
                        let index = Self::index(r, s, l);
                        if mask[index] {
                            let corners = self.cell_corners(index);
                            for k in [0usize, 1, 4, 5] {
                                points.push(corners[k] + offset);
                            }
                            break;
                        }
                    }
                }
            }
            if points.len() >= 4 {
                if let Some(hull) = Collider::convex_hull(&points) {
                    shapes.push((Vec3::ZERO, Quat::IDENTITY, hull));
                }
            }
            row = end;
        }
        (!shapes.is_empty()).then(|| Collider::compound(shapes))
    }

    fn neighbors(&self, index: usize) -> [Option<usize>; 6] {
        let (row, sector, layer) = Self::split(index);
        let mut out = [None; 6];
        if row > 0 {
            out[0] = Some(Self::index(row - 1, sector, layer));
        }
        if row + 1 < self.rows {
            out[1] = Some(Self::index(row + 1, sector, layer));
        }
        out[2] = Some(Self::index(
            row,
            (sector + WOOD_SECTORS - 1) % WOOD_SECTORS,
            layer,
        ));
        out[3] = Some(Self::index(row, (sector + 1) % WOOD_SECTORS, layer));
        if layer > 0 {
            out[4] = Some(Self::index(row, sector, layer - 1));
        }
        if layer + 1 < WOOD_LAYERS {
            out[5] = Some(Self::index(row, sector, layer + 1));
        }
        out
    }

    /// Cells of `mask` still connected to the ground (any block in row 0).
    fn grounded(&self, mask: &[bool]) -> Vec<bool> {
        let mut seen = vec![false; mask.len()];
        let mut stack = Vec::new();
        for s in 0..WOOD_SECTORS {
            for l in 0..WOOD_LAYERS {
                let i = Self::index(0, s, l);
                if mask[i] {
                    seen[i] = true;
                    stack.push(i);
                }
            }
        }
        while let Some(i) = stack.pop() {
            for n in self.neighbors(i).into_iter().flatten() {
                if mask[n] && !seen[n] {
                    seen[n] = true;
                    stack.push(n);
                }
            }
        }
        seen
    }

    /// Connected groups of cells in `mask`.
    fn components(&self, mask: &[bool]) -> Vec<Vec<usize>> {
        let mut seen = vec![false; mask.len()];
        let mut groups = Vec::new();
        for start in 0..mask.len() {
            if !mask[start] || seen[start] {
                continue;
            }
            seen[start] = true;
            let mut group = Vec::new();
            let mut stack = vec![start];
            while let Some(i) = stack.pop() {
                group.push(i);
                for n in self.neighbors(i).into_iter().flatten() {
                    if mask[n] && !seen[n] {
                        seen[n] = true;
                        stack.push(n);
                    }
                }
            }
            groups.push(group);
        }
        groups
    }

    fn top_row(&self, mask: &[bool]) -> Option<usize> {
        (0..self.rows).rev().find(|&row| {
            (0..WOOD_SECTORS * WOOD_LAYERS)
                .any(|k| mask[row * WOOD_SECTORS * WOOD_LAYERS + k])
        })
    }

    fn row_bounds(&self, row: usize) -> (f32, f32) {
        (
            row as f32 * WOOD_ROW_H,
            ((row + 1) as f32 * WOOD_ROW_H).min(self.shape.height),
        )
    }

    /// Centre of a cell, in trunk-base coordinates.
    fn cell_center(&self, index: usize) -> Vec3 {
        let (row, sector, layer) = Self::split(index);
        let (y0, y1) = self.row_bounds(row);
        let angle = (sector as f32 + 0.5) / WOOD_SECTORS as f32 * std::f32::consts::TAU;
        let frac = (WOOD_LAYER_FRAC[layer] + WOOD_LAYER_FRAC[layer + 1]) * 0.5;
        self.shape.point((y0 + y1) * 0.5, angle, frac)
    }

    /// The 8 corner points of a cell, in trunk-base coordinates.
    fn cell_corners(&self, index: usize) -> [Vec3; 8] {
        let (row, sector, layer) = Self::split(index);
        let (y0, y1) = self.row_bounds(row);
        let a0 = sector as f32 / WOOD_SECTORS as f32 * std::f32::consts::TAU;
        let a1 = (sector + 1) as f32 / WOOD_SECTORS as f32 * std::f32::consts::TAU;
        let (fo, fi) = (WOOD_LAYER_FRAC[layer], WOOD_LAYER_FRAC[layer + 1]);
        [
            self.shape.point(y0, a0, fo),
            self.shape.point(y0, a1, fo),
            self.shape.point(y0, a0, fi),
            self.shape.point(y0, a1, fi),
            self.shape.point(y1, a0, fo),
            self.shape.point(y1, a1, fo),
            self.shape.point(y1, a0, fi),
            self.shape.point(y1, a1, fi),
        ]
    }

    /// Builds the surface of the cells in `mask`. Only faces that border empty space
    /// are created: bark on the outside of the outer layer, wood-coloured faces
    /// everywhere else. `origin` is subtracted from every position. `ground_closed`
    /// hides the underside of row 0 (it sits in the ground). `cut_ratio[layer]` is
    /// the per-vertex colour multiplier that turns the bark material into the wood
    /// colour of that layer.
    fn build_mesh(
        &self,
        mask: &[bool],
        origin: Vec3,
        ground_closed: bool,
        cut_ratio: &[[f32; 3]; WOOD_LAYERS],
    ) -> WoodMeshData {
        let rows = self.rows as i32;
        let sectors = WOOD_SECTORS as i32;
        let has = |r: i32, s: i32, l: i32| -> bool {
            if r < 0 {
                return ground_closed;
            }
            if r >= rows || l < 0 {
                return false;
            }
            if l >= WOOD_LAYERS as i32 {
                return true; // the axis of the trunk
            }
            mask[Self::index(r as usize, s.rem_euclid(sectors) as usize, l as usize)]
        };

        let slope = self.shape.slope();
        let mut data = WoodMeshData::default();
        for index in 0..mask.len() {
            if !mask[index] {
                continue;
            }
            let (row, sector, layer) = Self::split(index);
            let (r, s, l) = (row as i32, sector as i32, layer as i32);
            let (y0, y1) = self.row_bounds(row);
            let a0 = sector as f32 / WOOD_SECTORS as f32 * std::f32::consts::TAU;
            let a1 = (sector + 1) as f32 / WOOD_SECTORS as f32 * std::f32::consts::TAU;
            let am = (a0 + a1) * 0.5;
            let (fo, fi) = (WOOD_LAYER_FRAC[layer], WOOD_LAYER_FRAC[layer + 1]);
            let p = |y: f32, a: f32, f: f32| self.shape.point(y, a, f) - origin;

            let shade = 1.0 - 0.07 * layer as f32;
            let ratio = cut_ratio[layer];
            let cut = [ratio[0] * shade, ratio[1] * shade, ratio[2] * shade, 1.0];
            let bark = [1.0, 1.0, 1.0, 1.0];

            // Outside (toward the bark, or toward the layer above that was cut away).
            if !has(r, s, l - 1) {
                let corners = [p(y0, a0, fo), p(y1, a0, fo), p(y1, a1, fo), p(y0, a1, fo)];
                if layer == 0 {
                    let n0 = Vec3::new(a0.cos(), slope, a0.sin()).normalize_or_zero();
                    let n1 = Vec3::new(a1.cos(), slope, a1.sin()).normalize_or_zero();
                    data.quad(
                        corners,
                        Some([n0, n0, n1, n1]),
                        Vec3::new(am.cos(), slope, am.sin()),
                        bark,
                    );
                } else {
                    data.quad(corners, None, Vec3::new(am.cos(), 0.0, am.sin()), cut);
                }
            }
            // Inside (toward a deeper layer that was cut away).
            if layer + 1 < WOOD_LAYERS && !has(r, s, l + 1) {
                data.quad(
                    [p(y0, a1, fi), p(y1, a1, fi), p(y1, a0, fi), p(y0, a0, fi)],
                    None,
                    -Vec3::new(am.cos(), 0.0, am.sin()),
                    cut,
                );
            }
            // Sides.
            if !has(r, s - 1, l) {
                data.quad(
                    [p(y0, a0, fo), p(y1, a0, fo), p(y1, a0, fi), p(y0, a0, fi)],
                    None,
                    Vec3::new(a0.sin(), 0.0, -a0.cos()),
                    cut,
                );
            }
            if !has(r, s + 1, l) {
                data.quad(
                    [p(y0, a1, fo), p(y1, a1, fo), p(y1, a1, fi), p(y0, a1, fi)],
                    None,
                    Vec3::new(-a1.sin(), 0.0, a1.cos()),
                    cut,
                );
            }
            // Underside / top.
            if !has(r - 1, s, l) {
                data.quad(
                    [p(y0, a0, fo), p(y0, a1, fo), p(y0, a1, fi), p(y0, a0, fi)],
                    None,
                    Vec3::NEG_Y,
                    cut,
                );
            }
            if !has(r + 1, s, l) {
                data.quad(
                    [p(y1, a0, fo), p(y1, a1, fo), p(y1, a1, fi), p(y1, a0, fi)],
                    None,
                    Vec3::Y,
                    cut,
                );
            }
        }
        data
    }
}

// ----- glue between the layered wood grid and Bevy -----

/// Per-layer colour multipliers that turn the bark material into the colour of cut wood.
fn wood_cut_ratios(bark: Color) -> [[f32; 3]; WOOD_LAYERS] {
    let mut out = [[1.0; 3]; WOOD_LAYERS];
    for (layer, slot) in out.iter_mut().enumerate() {
        let t = (WOOD_LAYER_FRAC[layer] + WOOD_LAYER_FRAC[layer + 1]) * 0.5;
        *slot = cut_wood_vertex_ratio(cut_wood_target_color(t, layer), bark);
    }
    out
}

fn wood_mesh_from_data(data: WoodMeshData) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, data.positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, data.normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, data.uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, data.colors);
    mesh.insert_indices(bevy::render::mesh::Indices::U32(data.indices));
    mesh
}

/// A cylinder wall WITHOUT end caps. The flat cut discs (`trunk_cut_disc`) close its
/// ends, so no two surfaces ever share a plane (a capped `Cylinder` plus a disc on top
/// of its cap is what made the cut faces flicker).
#[allow(dead_code)]
fn open_tube_mesh(radius: f32, length: f32) -> Mesh {
    const SIDES: usize = 24;
    let half = length * 0.5;
    let mut positions = Vec::with_capacity((SIDES + 1) * 2);
    let mut normals = Vec::with_capacity((SIDES + 1) * 2);
    let mut uvs = Vec::with_capacity((SIDES + 1) * 2);
    let mut indices = Vec::with_capacity(SIDES * 6);
    for ring in 0..2 {
        let y = if ring == 0 { -half } else { half };
        for side in 0..=SIDES {
            let angle = side as f32 / SIDES as f32 * std::f32::consts::TAU;
            let (sin, cos) = angle.sin_cos();
            positions.push([cos * radius, y, sin * radius]);
            normals.push([cos, 0.0, sin]);
            uvs.push([side as f32 / SIDES as f32, ring as f32]);
        }
    }
    for side in 0..SIDES {
        let bottom = side as u32;
        let top = bottom + (SIDES + 1) as u32;
        indices.extend_from_slice(&[bottom, top, bottom + 1, bottom + 1, top, top + 1]);
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(bevy::render::mesh::Indices::U32(indices));
    mesh
}

/// Turns `cells` of the trunk into a real physics block (one cell = a chip, several
/// connected cells = a chunk that was cut loose). It appears just outside the trunk
/// surface at the height of the wood it came from and falls away.
#[allow(clippy::too_many_arguments)]
fn spawn_wood_debris(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    bark: &Handle<StandardMaterial>,
    base: Vec3,
    wood: &TrunkWood,
    cells: &[usize],
    cut_ratio: &[[f32; 3]; WOOD_LAYERS],
    seed: u32,
    kick: f32,
    push_out: f32,
    born: f32,
) {
    if cells.is_empty() {
        return;
    }
    let mut mask = vec![false; wood.cells.len()];
    let mut center = Vec3::ZERO;
    let mut points = Vec::with_capacity(cells.len() * 8);
    for &cell in cells {
        mask[cell] = true;
        center += wood.cell_center(cell);
    }
    center /= cells.len() as f32;
    for &cell in cells {
        for corner in wood.cell_corners(cell) {
            points.push(corner - center);
        }
    }
    let mesh = meshes.add(wood_mesh_from_data(wood.build_mesh(
        &mask, center, false, cut_ratio,
    )));
    let axis = wood.shape.center_at(center.y);
    let mut outward = Vec3::new(center.x - axis.x, 0.0, center.z - axis.y).normalize_or_zero();
    if outward == Vec3::ZERO {
        outward = Vec3::Z;
    }
    let surface = wood.shape.radius_at(center.y) + push_out;
    let position = base + Vec3::new(axis.x, center.y, axis.y) + outward * surface;
    let collider = Collider::convex_hull(&points)
        .unwrap_or_else(|| Collider::cuboid(0.07, 0.07, 0.07));
    let rnd = |k: u32| pseudo_rand(seed.wrapping_add(k));
    let side = Vec3::new(-outward.z, 0.0, outward.x) * (rnd(6) - 0.5) * kick;
    commands.spawn((
        PbrBundle {
            mesh,
            material: bark.clone(),
            transform: Transform::from_translation(position),
            ..default()
        },
        RigidBody::Dynamic,
        collider,
        ColliderMassProperties::Density(450.0),
        Velocity {
            linvel: outward * kick * (0.5 + rnd(1)) + side + Vec3::Y * (0.6 + rnd(2) * 1.2),
            angvel: Vec3::new(rnd(3) - 0.5, rnd(4) - 0.5, rnd(5) - 0.5) * 9.0,
        },
        Friction::coefficient(0.9),
        Restitution::coefficient(0.1),
        WoodDebris { born },
    ));
}


/// Like `spawn_wood_debris`, but for wood that is NOT a standing trunk: a fallen log or a
/// limb. `frame` is the world transform of the wood grid's own coordinate system.
#[allow(clippy::too_many_arguments)]
fn spawn_wood_frame_debris(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    material: &Handle<StandardMaterial>,
    frame: &Transform,
    wood: &TrunkWood,
    cells: &[usize],
    cut_ratio: &[[f32; 3]; WOOD_LAYERS],
    body_linvel: Vec3,
    body_angvel: Vec3,
    body_center: Vec3,
    seed: u32,
    kick: f32,
    born: f32,
) -> Option<Entity> {
    if cells.is_empty() {
        return None;
    }
    let mut mask = vec![false; wood.cells.len()];
    let mut center = Vec3::ZERO;
    for &cell in cells {
        mask[cell] = true;
        center += wood.cell_center(cell);
    }
    center /= cells.len() as f32;
    let mut points = Vec::with_capacity(cells.len() * 8);
    for &cell in cells {
        for corner in wood.cell_corners(cell) {
            points.push(corner - center);
        }
    }
    let mesh = meshes.add(wood_mesh_from_data(wood.build_mesh(
        &mask, center, false, cut_ratio,
    )));
    let axis = wood.shape.center_at(center.y);
    let mut outward_local =
        Vec3::new(center.x - axis.x, 0.0, center.z - axis.y).normalize_or_zero();
    if outward_local == Vec3::ZERO {
        outward_local = Vec3::Z;
    }
    let outward = frame.rotation * outward_local;
    let position = frame.translation + frame.rotation * center + outward * 0.04;
    let collider = Collider::convex_hull(&points)
        .unwrap_or_else(|| Collider::cuboid(0.05, 0.05, 0.05));
    let rnd = |k: u32| pseudo_rand(seed.wrapping_add(k));
    let side = Vec3::new(-outward.z, 0.0, outward.x) * (rnd(6) - 0.5) * kick;
    Some(
        commands
            .spawn((
                PbrBundle {
                    mesh,
                    material: material.clone(),
                    transform: Transform {
                        translation: position,
                        rotation: frame.rotation,
                        scale: Vec3::ONE,
                    },
                    ..default()
                },
                RigidBody::Dynamic,
                collider,
                ColliderMassProperties::Density(450.0),
                Velocity {
                    linvel: body_linvel
                        + body_angvel.cross(position - body_center)
                        + outward * kick * (0.5 + rnd(1))
                        + side
                        + Vec3::Y * (0.5 + rnd(2)),
                    angvel: Vec3::new(rnd(3) - 0.5, rnd(4) - 0.5, rnd(5) - 0.5) * 9.0,
                },
                Friction::coefficient(0.9),
                Restitution::coefficient(0.1),
                WoodDebris { born },
            ))
            .id(),
    )
}

struct OakMeshBatch {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
}

impl OakMeshBatch {
    fn new() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
        }
    }

    fn append(&mut self, mesh: &Mesh, transform: Transform) {
        let Some(positions) = mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(|attribute| attribute.as_float3())
        else {
            return;
        };
        let normals = mesh
            .attribute(Mesh::ATTRIBUTE_NORMAL)
            .and_then(|attribute| attribute.as_float3());
        let uvs = match mesh.attribute(Mesh::ATTRIBUTE_UV_0) {
            Some(bevy::render::mesh::VertexAttributeValues::Float32x2(values)) => Some(values),
            _ => None,
        };
        let base = self.positions.len() as u32;
        for (index, position) in positions.iter().enumerate() {
            let local = Vec3::from(*position) * transform.scale;
            self.positions
                .push((transform.translation + transform.rotation * local).to_array());
            let normal = normals
                .and_then(|values| values.get(index))
                .map(|normal| {
                    let inverse_scale = Vec3::new(
                        normal[0] / transform.scale.x.max(0.0001),
                        normal[1] / transform.scale.y.max(0.0001),
                        normal[2] / transform.scale.z.max(0.0001),
                    );
                    (transform.rotation * inverse_scale).normalize_or_zero()
                })
                .unwrap_or(Vec3::Y);
            self.normals.push(normal.to_array());
            self.uvs.push(
                uvs.and_then(|values| values.get(index).copied())
                    .unwrap_or(Vec2::ZERO.to_array()),
            );
        }
        if let Some(indices) = mesh.indices() {
            match indices {
                bevy::render::mesh::Indices::U16(indices) => self
                    .indices
                    .extend(indices.iter().map(|index| base + *index as u32)),
                bevy::render::mesh::Indices::U32(indices) => self
                    .indices
                    .extend(indices.iter().map(|index| base + *index)),
            }
        } else {
            self.indices.extend(base..base + positions.len() as u32);
        }
    }

    fn finish(self) -> Mesh {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, self.positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, self.uvs);
        mesh.insert_indices(bevy::render::mesh::Indices::U32(self.indices));
        mesh
    }
}

fn unit_oak_cylinder_mesh() -> Mesh {
    const SIDES: usize = 10;
    let mut positions = Vec::with_capacity((SIDES + 1) * 2 + 2);
    let mut normals = Vec::with_capacity((SIDES + 1) * 2 + 2);
    let mut uvs = Vec::with_capacity((SIDES + 1) * 2 + 2);
    let mut indices = Vec::with_capacity(SIDES * 12);
    for ring in 0..2 {
        let y = if ring == 0 { -0.5 } else { 0.5 };
        for side in 0..=SIDES {
            let angle = side as f32 / SIDES as f32 * std::f32::consts::TAU;
            let (sin, cos) = angle.sin_cos();
            positions.push([cos, y, sin]);
            normals.push([cos, 0.0, sin]);
            uvs.push([side as f32 / SIDES as f32, ring as f32]);
        }
    }
    for side in 0..SIDES {
        let bottom = side as u32;
        let top = bottom + (SIDES + 1) as u32;
        indices.extend_from_slice(&[bottom, top, bottom + 1, bottom + 1, top, top + 1]);
    }
    let bottom_center = positions.len() as u32;
    positions.push([0.0, -0.5, 0.0]);
    normals.push([0.0, -1.0, 0.0]);
    uvs.push([0.5, 0.5]);
    let top_center = positions.len() as u32;
    positions.push([0.0, 0.5, 0.0]);
    normals.push([0.0, 1.0, 0.0]);
    uvs.push([0.5, 0.5]);
    for side in 0..SIDES {
        let next = (side + 1) as u32;
        let current = side as u32;
        let top_current = current + (SIDES + 1) as u32;
        let top_next = next + (SIDES + 1) as u32;
        indices.extend_from_slice(&[
            bottom_center,
            next,
            current,
            top_center,
            top_current,
            top_next,
        ]);
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(bevy::render::mesh::Indices::U32(indices));
    mesh
}

/// Spawn a tapered, varied oak with recursively split limbs and dense leaf clusters.
fn spawn_tree(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &mut Assets<StandardMaterial>,
    base: Vec3,
    index: u32,
) {
    let height = 5.5 + pseudo_rand(index.wrapping_add(71)) * 2.7;
    let lean = Vec2::new(
        (pseudo_rand(index.wrapping_add(72)) - 0.5) * 0.8,
        (pseudo_rand(index.wrapping_add(73)) - 0.5) * 0.8,
    );
    let bark_variation = pseudo_rand(index.wrapping_add(74));
    let bark = mats.add(Color::srgb(
        0.24 + bark_variation * 0.13,
        0.12 + bark_variation * 0.08,
        0.055 + bark_variation * 0.045,
    ));
    let branch_material = mats.add(Color::srgb(
        0.23 + bark_variation * 0.11,
        0.12 + bark_variation * 0.07,
        0.055 + bark_variation * 0.04,
    ));
    let foliage = [
        mats.add(StandardMaterial {
            base_color: Color::srgb(0.32, 0.54, 0.19),
            cull_mode: None,
            perceptual_roughness: 0.94,
            ..default()
        }),
        mats.add(StandardMaterial {
            base_color: Color::srgb(0.44, 0.65, 0.25),
            cull_mode: None,
            perceptual_roughness: 0.94,
            ..default()
        }),
        mats.add(StandardMaterial {
            base_color: Color::srgb(0.25, 0.45, 0.15),
            cull_mode: None,
            perceptual_roughness: 0.94,
            ..default()
        }),
    ];
    let branch_mesh = unit_oak_cylinder_mesh();
    let base_radius = 0.36 + pseudo_rand(index.wrapping_add(75)) * 0.12;
    let top_radius = 0.09 + pseudo_rand(index.wrapping_add(76)) * 0.045;
    let inner_wood = mats.add(INNER_WOOD_COLOR);
    // No notch on a freshly spawned tree, so no special vertex colouring yet.
    let trunk_mesh = meshes.add(tapered_trunk_mesh(
        height,
        base_radius,
        top_radius,
        lean,
        None,
        [1.0, 1.0, 1.0],
    ));
    let mut roots_batch = OakMeshBatch::new();
    let mut root_segments = Vec::new();
    for root_index in 0..6 {
        let angle = root_index as f32 * std::f32::consts::TAU / 6.0
            + (pseudo_rand(index.wrapping_add(root_index + 121)) - 0.5) * 0.45;
        let direction = Vec3::new(angle.cos(), -0.08, angle.sin()).normalize();
        let side = Vec3::new(-direction.z, 0.0, direction.x).normalize();
        let points = [
            direction * 0.22,
            direction * 0.72 + side * 0.08,
            direction * 1.3 - side * 0.06 + Vec3::NEG_Y * 0.08,
        ];
        for segment in 0..points.len() - 1 {
            let start = points[segment];
            let end = points[segment + 1];
            let delta = end - start;
            let radius = 0.12 - segment as f32 * 0.025;
            roots_batch.append(
                &branch_mesh,
                Transform::from_translation((start + end) * 0.5)
                    .with_rotation(Quat::from_rotation_arc(Vec3::Y, delta.normalize()))
                    .with_scale(Vec3::new(radius, delta.length(), radius)),
            );
            root_segments.push((start, end, radius));
        }
    }
    let roots_mesh = meshes.add(roots_batch.finish());

    let tree_entity = commands
        .spawn((
            SpatialBundle::from_transform(Transform::from_translation(
                base + Vec3::Y * (height / 2.0),
            )),
            RigidBody::Fixed,
            Collider::cylinder(height / 2.0, 0.42),
            Tree {
                notch_depth: 0.0,
                notch_y_bottom: 0.0,
                notch_height: 0.4,
                notch_phi: 0.0,
                height,
                far_lod: false,
                base_radius,
                top_radius,
                lean,
                bark: bark.clone(),
                inner_wood: inner_wood.clone(),
                trunk_mesh: trunk_mesh.clone(),
                cavity: None,
                notch_center: Vec3::ZERO,
                notch_normal: Vec3::Z,
                cut_height: 1.15,
                fall_dir: Vec3::Z,
                wood: None,
                stump_collider: None,
            },
        ))
        .with_children(|tree| {
            tree.spawn(SpatialBundle::from_transform(Transform::from_translation(
                Vec3::NEG_Y * (height / 2.0),
            )))
            .with_children(|trunk_parent| {
                trunk_parent.spawn(PbrBundle {
                    mesh: trunk_mesh,
                    material: bark.clone(),
                    ..default()
                });

                trunk_parent.spawn(PbrBundle {
                    mesh: roots_mesh,
                    material: branch_material.clone(),
                    ..default()
                });
            });
        })
        .id();
    for (start, end, radius) in root_segments {
        let delta = end - start;
        let center = (start + end) * 0.5 - Vec3::Y * (height / 2.0);
        commands.entity(tree_entity).with_children(|tree| {
            tree.spawn((
                SpatialBundle::from_transform(
                    Transform::from_translation(center)
                        .with_rotation(Quat::from_rotation_arc(Vec3::Y, delta.normalize())),
                ),
                Collider::capsule_y(delta.length() * 0.5, radius),
            ));
        });
    }
    commands.entity(tree_entity).with_children(|tree| {
        let crown_mesh = meshes.add(Sphere::new(1.0));
        let crown_material = foliage[1].clone();
        for (offset, scale) in [
            (
                Vec3::new(-0.7, height * 0.38, 0.0),
                Vec3::new(1.65, 1.2, 1.5),
            ),
            (
                Vec3::new(0.7, height * 0.4, 0.0),
                Vec3::new(1.65, 1.25, 1.5),
            ),
            (
                Vec3::new(0.0, height * 0.56, 0.0),
                Vec3::new(1.5, 1.35, 1.5),
            ),
        ] {
            tree.spawn((
                PbrBundle {
                    mesh: crown_mesh.clone(),
                    material: crown_material.clone(),
                    transform: Transform::from_translation(offset).with_scale(scale),
                    visibility: Visibility::Hidden,
                    ..default()
                },
                TreeLodPiece {
                    tree: tree_entity,
                    high_detail: false,
                },
            ));
        }
    });

    let branch_count = 4 + (pseudo_rand(index.wrapping_add(77)) * 4.0) as u32;
    for branch_index in 0..branch_count {
        let seed = index
            .wrapping_mul(97)
            .wrapping_add(branch_index.wrapping_mul(31))
            .wrapping_add(37);
        let angle = branch_index as f32 * std::f32::consts::TAU / branch_count as f32
            + (pseudo_rand(seed) - 0.5) * 0.7;
        let height_fraction = 0.52 + pseudo_rand(seed.wrapping_add(1)) * 0.3;
        let direction = Vec3::new(
            angle.cos() * (0.82 + pseudo_rand(seed.wrapping_add(2)) * 0.28),
            0.28 + pseudo_rand(seed.wrapping_add(3)) * 0.35,
            angle.sin() * (0.82 + pseudo_rand(seed.wrapping_add(4)) * 0.28),
        )
        .normalize();
        let anchor = base
            + Vec3::new(
                lean.x * height_fraction,
                height * height_fraction,
                lean.y * height_fraction,
            );
        let rotation = Quat::from_rotation_arc(Vec3::Y, direction);
        let mut segments = Vec::new();
        let mut leaf_tips = Vec::new();
        append_oak_branch(
            Vec3::ZERO,
            Vec3::Y,
            1.35 + pseudo_rand(seed.wrapping_add(5)) * 0.75,
            0.13 + pseudo_rand(seed.wrapping_add(6)) * 0.035,
            2,
            seed,
            None,
            &mut segments,
            &mut leaf_tips,
        );
        let carved = vec![None; segments.len()];
        let shape = BranchShape {
            segments,
            tips: leaf_tips,
            wood: branch_material.clone(),
            foliage: foliage.clone(),
            carved,
            cut_ratio: [[1.0; 3]; WOOD_LAYERS],
            plucked: Vec::new(),
        };

        let branch_entity = commands
            .spawn((
                SpatialBundle::from_transform(
                    Transform::from_translation(anchor).with_rotation(rotation),
                ),
                RigidBody::Dynamic,
                Sensor,
                GravityScale(0.0),
                Velocity::default(),
                TreeLodPiece {
                    tree: tree_entity,
                    high_detail: true,
                },
                TreeBranch {
                    tree: tree_entity,
                    anchor,
                    broken: false,
                    chops_remaining: 3,
                },
                shape.clone(),
            ))
            .id();
        attach_branch_parts(commands, branch_entity, &shape, meshes, true);
    }
}

fn spawn_bird(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    index: u32,
) {
    let feathers = materials.add(Color::srgb(0.16, 0.19, 0.22));
    let wing_material = materials.add(Color::srgb(0.11, 0.13, 0.16));
    let beak_material = materials.add(Color::srgb(0.85, 0.57, 0.17));
    let phase = index as f32 * std::f32::consts::TAU / 8.0;
    let radius = 7.5 + pseudo_rand(index.wrapping_add(120)) * 5.0;
    let altitude = 7.0 + pseudo_rand(index.wrapping_add(240)) * 4.0;
    let initial = phase;
    let center_x = 18.0;
    let center_z = 4.0;
    let position = Vec3::new(
        center_x + phase.cos() * radius,
        altitude,
        center_z + phase.sin() * radius,
    );
    commands
        .spawn((
            SpatialBundle::from_transform(Transform::from_translation(position)),
            Bird {
                phase: initial,
                radius,
                altitude,
                speed: 0.12 + pseudo_rand(index.wrapping_add(360)) * 0.08,
            },
        ))
        .with_children(|parent| {
            parent.spawn(PbrBundle {
                mesh: meshes.add(Sphere::new(0.16)),
                material: feathers.clone(),
                transform: Transform {
                    scale: Vec3::new(0.72, 0.68, 1.55),
                    ..default()
                },
                ..default()
            });
            parent.spawn(PbrBundle {
                mesh: meshes.add(Sphere::new(0.105)),
                material: feathers.clone(),
                transform: Transform::from_xyz(0.0, 0.035, -0.21),
                ..default()
            });
            parent.spawn(PbrBundle {
                mesh: meshes.add(Cone {
                    radius: 0.055,
                    height: 0.16,
                }),
                material: beak_material,
                transform: Transform::from_xyz(0.0, 0.0, -0.34)
                    .with_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
                ..default()
            });
            for side in [-1.0_f32, 1.0] {
                parent.spawn((
                    PbrBundle {
                        mesh: meshes.add(Cuboid::new(0.72, 0.035, 0.22)),
                        material: wing_material.clone(),
                        transform: Transform::from_xyz(side * 0.34, 0.0, 0.015),
                        ..default()
                    },
                    BirdWing {
                        side,
                        phase: index as f32 * 0.7,
                    },
                ));
            }
        });
}

fn bird_flight_system(
    time: Res<Time>,
    mut birds: Query<(&mut Bird, &mut Transform)>,
    mut wings: Query<(&BirdWing, &mut Transform), Without<Bird>>,
) {
    let elapsed = time.elapsed_seconds();
    for (mut bird, mut transform) in &mut birds {
        bird.phase =
            (bird.phase + bird.speed * time.delta_seconds()).rem_euclid(std::f32::consts::TAU);
        let angle = bird.phase;
        let center = Vec3::new(18.0, 0.0, 4.0);
        let next = Vec3::new(
            center.x + (angle + 0.01).cos() * bird.radius,
            0.0,
            center.z + (angle + 0.01).sin() * bird.radius,
        );
        transform.translation = Vec3::new(
            center.x + angle.cos() * bird.radius,
            bird.altitude + (elapsed * 1.8 + angle).sin() * 0.25,
            center.z + angle.sin() * bird.radius,
        );
        let forward =
            (next - Vec3::new(transform.translation.x, 0.0, transform.translation.z)).normalize();
        transform.rotation = Quat::from_rotation_y((-forward.x).atan2(-forward.z));
    }
    for (wing, mut transform) in &mut wings {
        let flap = (elapsed * 13.0 + wing.phase).sin() * 0.62;
        transform.rotation = Quat::from_rotation_z(wing.side * flap);
    }
}

/// A stick from `a` to `b` (a unit cylinder scaled to fit).
fn gh_limb(
    p: &mut ChildBuilder,
    stick: &Handle<Mesh>,
    material: &Handle<StandardMaterial>,
    a: Vec3,
    b: Vec3,
    r: f32,
) {
    let d = b - a;
    p.spawn(PbrBundle {
        mesh: stick.clone(),
        material: material.clone(),
        transform: Transform {
            translation: (a + b) * 0.5,
            rotation: Quat::from_rotation_arc(Vec3::Y, d.normalize()),
            scale: Vec3::new(r, d.length(), r),
        },
        ..default()
    });
}

/// An ellipsoid (a unit-diameter sphere scaled to fit).
fn gh_blob(
    p: &mut ChildBuilder,
    ball: &Handle<Mesh>,
    material: &Handle<StandardMaterial>,
    at: Vec3,
    scale: Vec3,
    rot: Quat,
) {
    p.spawn(PbrBundle {
        mesh: ball.clone(),
        material: material.clone(),
        transform: Transform {
            translation: at,
            rotation: rot,
            scale,
        },
        ..default()
    });
}

/// A little grasshopper built from ellipsoids and sticks (facing -Z, feet on y = 0).
fn spawn_grasshopper(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &mut Assets<StandardMaterial>,
    pos: Vec3,
    seed: u32,
) {
    let mat = |mats: &mut Assets<StandardMaterial>, c: Color, rough: f32| {
        mats.add(StandardMaterial {
            base_color: c,
            perceptual_roughness: rough,
            ..default()
        })
    };
    let green = mat(mats, Color::srgb(0.38, 0.64, 0.1), 0.65);
    let dark_green = mat(mats, Color::srgb(0.22, 0.42, 0.08), 0.7);
    let olive = mat(mats, Color::srgb(0.34, 0.5, 0.1), 0.75);
    let tibia_color = mat(mats, Color::srgb(0.6, 0.55, 0.16), 0.75);
    let wing_color = mat(mats, Color::srgb(0.5, 0.43, 0.17), 0.8);
    let eye_color = mat(mats, Color::srgb(0.06, 0.05, 0.03), 0.25);
    let ball = meshes.add(Sphere::new(0.5)); // unit ellipsoid: scale it
    let stick = meshes.add(Cylinder::new(1.0, 1.0)); // radius 1, height 1: scale (r, len, r)
    let flat = meshes.add(Cuboid::new(1.0, 1.0, 1.0));

    let root = commands
        .spawn(SpatialBundle::from_transform(Transform::from_translation(pos)))
        .id();
    let mut hips = [Entity::PLACEHOLDER; 2];
    let mut knees = [Entity::PLACEHOLDER; 2];
    let mut antennae = [Entity::PLACEHOLDER; 2];
    let mut wings = [Entity::PLACEHOLDER; 2];

    commands.entity(root).with_children(|p| {
        // body: thorax, saddle-shaped pronotum, long abdomen, head, eyes, mouth
        gh_blob(p, &ball, &green, Vec3::new(0.0, 0.058, -0.005), Vec3::new(0.04, 0.042, 0.075), Quat::IDENTITY);
        gh_blob(p, &ball, &dark_green, Vec3::new(0.0, 0.074, -0.03), Vec3::new(0.046, 0.03, 0.06), Quat::IDENTITY);
        gh_blob(p, &ball, &green, Vec3::new(0.0, 0.058, 0.088), Vec3::new(0.036, 0.036, 0.125), Quat::from_rotation_x(-0.12));
        gh_blob(p, &ball, &green, Vec3::new(0.0, 0.062, -0.075), Vec3::new(0.034, 0.047, 0.042), Quat::from_rotation_x(0.25));
        for side in [-1.0_f32, 1.0] {
            gh_blob(p, &ball, &eye_color, Vec3::new(side * 0.018, 0.073, -0.083), Vec3::splat(0.022), Quat::IDENTITY);
        }
        gh_blob(p, &ball, &dark_green, Vec3::new(0.0, 0.041, -0.099), Vec3::new(0.02, 0.012, 0.012), Quat::IDENTITY);

        for (i, side) in [-1.0_f32, 1.0].into_iter().enumerate() {
            // front and middle legs: two sticks each, standing on the ground
            gh_limb(p, &stick, &olive, Vec3::new(side * 0.02, 0.05, -0.04), Vec3::new(side * 0.036, 0.048, -0.072), 0.0045);
            gh_limb(p, &stick, &olive, Vec3::new(side * 0.036, 0.048, -0.072), Vec3::new(side * 0.042, 0.0, -0.092), 0.0035);
            gh_limb(p, &stick, &olive, Vec3::new(side * 0.022, 0.05, -0.005), Vec3::new(side * 0.046, 0.05, -0.016), 0.0045);
            gh_limb(p, &stick, &olive, Vec3::new(side * 0.046, 0.05, -0.016), Vec3::new(side * 0.056, 0.0, -0.018), 0.0035);

            // big jumping leg: femur from the hip to the knee, then the tibia down to the foot
            let knee_at = Vec3::new(side * 0.012, 0.028, 0.085);
            let foot_at = Vec3::new(side * 0.004, -0.086, -0.02);
            let hip = p
                .spawn(SpatialBundle::from_transform(Transform::from_xyz(
                    side * 0.024,
                    0.058,
                    0.03,
                )))
                .with_children(|h| {
                    gh_limb(h, &stick, &olive, Vec3::ZERO, knee_at, 0.011);
                    let knee = h
                        .spawn(SpatialBundle::from_transform(Transform::from_translation(knee_at)))
                        .with_children(|k| {
                            gh_limb(k, &stick, &tibia_color, Vec3::ZERO, foot_at, 0.0045);
                        })
                        .id();
                    knees[i] = knee;
                })
                .id();
            hips[i] = hip;

            // antenna on a pivot at the front of the head
            let antenna = p
                .spawn(SpatialBundle::from_transform(Transform::from_xyz(side * 0.012, 0.09, -0.088)))
                .with_children(|a| {
                    a.spawn(PbrBundle {
                        mesh: stick.clone(),
                        material: dark_green.clone(),
                        transform: Transform {
                            translation: Vec3::new(0.0, 0.0, -0.05),
                            rotation: Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
                            scale: Vec3::new(0.0025, 0.1, 0.0025),
                        },
                        ..default()
                    });
                })
                .id();
            antennae[i] = antenna;

            // wing lying along the back, on a pivot so it can open
            let wing = p
                .spawn(SpatialBundle::from_transform(Transform::from_xyz(side * 0.013, 0.088, -0.022)))
                .with_children(|w| {
                    w.spawn(PbrBundle {
                        mesh: flat.clone(),
                        material: wing_color.clone(),
                        transform: Transform {
                            translation: Vec3::new(0.0, 0.0, 0.065),
                            rotation: Quat::IDENTITY,
                            scale: Vec3::new(0.024, 0.003, 0.13),
                        },
                        ..default()
                    });
                })
                .id();
            wings[i] = wing;
        }
    });

    let yaw = pseudo_rand(seed) * std::f32::consts::TAU;
    commands.entity(root).insert((
        Transform::from_translation(pos).with_rotation(Quat::from_rotation_y(yaw)),
        Grasshopper {
            state: HopState::Resting,
            timer: 0.5 + pseudo_rand(seed.wrapping_add(1)) * 3.0,
            crouch: 0.2,
            t: 0.0,
            duration: 0.4,
            height: 0.15,
            start: pos,
            target: pos,
            pos,
            yaw,
            home: pos,
            seed,
            phase: pseudo_rand(seed.wrapping_add(2)) * 10.0,
            parts: GrasshopperParts {
                hips,
                knees,
                antennae,
                wings,
            },
        },
    ));
}

/// Somewhere a grasshopper may land: on dry land, not in the house.
fn grasshopper_spot_ok(x: f32, z: f32) -> bool {
    (x - river_center_x(z)).abs() > RIVER_WIDTH * 0.5 + 0.6
        && !(x.abs() < FOOT * 0.5 + 0.5 && z.abs() < FOOT * 0.5 + 0.5)
        && x.abs() < GROUND_HALF_SIZE - 1.0
        && z.abs() < GROUND_HALF_SIZE - 1.0
}

fn lerp_angle(a: f32, b: f32, t: f32) -> f32 {
    let d = (b - a + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
    a + d * t
}

/// Picks where the next hop lands: away from `flee_from` if given, otherwise wandering
/// (drifting back toward home when it has strayed).
fn grasshopper_pick_target(gh: &mut Grasshopper, flee_from: Option<Vec3>) -> Option<Vec3> {
    let from = gh.pos;
    for _ in 0..6 {
        gh.seed = gh.seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let r1 = pseudo_rand(gh.seed);
        let r2 = pseudo_rand(gh.seed.wrapping_add(1));
        let to_home = gh.home - from;
        let angle = if let Some(p) = flee_from {
            let away = from - p;
            away.z.atan2(away.x) + (r1 - 0.5) * 1.6
        } else if Vec2::new(to_home.x, to_home.z).length() > 4.5 {
            to_home.z.atan2(to_home.x) + (r1 - 0.5) * 1.4
        } else {
            r1 * std::f32::consts::TAU
        };
        let dist = if flee_from.is_some() {
            1.0 + r2 * 0.9
        } else {
            0.35 + r2 * 0.75
        };
        let x = from.x + angle.cos() * dist;
        let z = from.z + angle.sin() * dist;
        let from_home = Vec2::new(x - gh.home.x, z - gh.home.z).length();
        if grasshopper_spot_ok(x, z) && from_home < 7.0 {
            return Some(Vec3::new(x, land_surface_height(x, z), z));
        }
    }
    None
}

fn grasshopper_system(
    time: Res<Time>,
    player: Query<&Transform, With<Player>>,
    mut hoppers: Query<(Entity, &mut Grasshopper)>,
    mut tfs: Query<&mut Transform, Without<Player>>,
) {
    use std::f32::consts::PI;
    let dt = time.delta_seconds().min(0.05);
    let player_pos = player.get_single().ok().map(|t| t.translation);
    let smooth = |x: f32| {
        let x = x.clamp(0.0, 1.0);
        x * x * (3.0 - 2.0 * x)
    };

    for (entity, mut gh) in &mut hoppers {
        gh.phase += dt;
        // a person coming close makes it bolt
        let near = player_pos.map_or(false, |p| {
            Vec2::new(p.x - gh.pos.x, p.z - gh.pos.z).length() < 2.2
        });

        let state = gh.state;
        match state {
            HopState::Resting => {
                gh.timer -= dt;
                if near || gh.timer <= 0.0 {
                    let flee_from = if near { player_pos } else { None };
                    match grasshopper_pick_target(&mut gh, flee_from) {
                        Some(target) => {
                            gh.start = gh.pos;
                            gh.target = target;
                            gh.state = HopState::Crouching;
                            gh.crouch = if near { 0.07 } else { 0.22 };
                            gh.timer = gh.crouch;
                        }
                        None => gh.timer = 0.4,
                    }
                }
            }
            HopState::Crouching => {
                gh.timer -= dt;
                let d = gh.target - gh.pos;
                if d.x != 0.0 || d.z != 0.0 {
                    let want = (-d.x).atan2(-d.z);
                    gh.yaw = lerp_angle(gh.yaw, want, (dt * 18.0).min(1.0));
                }
                if gh.timer <= 0.0 {
                    let dist = Vec2::new(d.x, d.z).length();
                    gh.duration = 0.3 + dist * 0.2;
                    gh.height = 0.08 + dist * 0.14;
                    gh.t = 0.0;
                    gh.state = HopState::Hopping;
                }
            }
            HopState::Hopping => {
                gh.t += dt / gh.duration;
                if gh.t >= 1.0 {
                    gh.pos = gh.target;
                    gh.state = HopState::Resting;
                    gh.seed = gh.seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    gh.timer = 0.3 + pseudo_rand(gh.seed) * 2.6;
                } else {
                    let flat = gh.start.lerp(gh.target, gh.t);
                    gh.pos = Vec3::new(flat.x, land_surface_height(flat.x, flat.z), flat.z);
                }
            }
        }

        // pose for the current state
        let (hip, knee, flare, pitch, squash, antenna_pitch) = match gh.state {
            HopState::Resting => (0.0, 0.0, 0.0, 0.0, 1.0 + (gh.phase * 3.0).sin() * 0.015, 0.7),
            HopState::Crouching => {
                let c = smooth(1.0 - gh.timer / gh.crouch.max(0.001));
                (-0.3 * c, 0.5 * c, 0.0, 0.05 * c, 1.0 - 0.2 * c, 0.7 - 0.2 * c)
            }
            HopState::Hopping => {
                let t = gh.t.clamp(0.0, 1.0);
                // legs: from the crouch, snap out straight, then fold back for landing
                let (hip, knee) = if t < 0.65 {
                    let a = smooth(t / 0.2);
                    (-0.3 + 0.85 * a, 0.5 - 2.15 * a)
                } else {
                    let b = smooth((t - 0.65) / 0.35);
                    (0.55 * (1.0 - b), -1.65 * (1.0 - b))
                };
                let arc = (t * PI).sin();
                let flare = arc * 0.9 + 0.2 * (t * 70.0).sin() * arc;
                (hip, knee, flare, 0.45 * (1.0 - 2.0 * t), 1.0, 0.25)
            }
        };
        let lift = if gh.state == HopState::Hopping {
            4.0 * gh.height * gh.t * (1.0 - gh.t)
        } else {
            0.0
        };

        if let Ok(mut t) = tfs.get_mut(entity) {
            t.translation = gh.pos + Vec3::Y * lift;
            t.rotation = Quat::from_rotation_y(gh.yaw) * Quat::from_rotation_x(pitch);
            t.scale = Vec3::new(1.0, squash, 1.0);
        }
        for i in 0..2 {
            let side = if i == 0 { -1.0 } else { 1.0 };
            if let Ok(mut t) = tfs.get_mut(gh.parts.hips[i]) {
                t.rotation = Quat::from_rotation_x(hip);
            }
            if let Ok(mut t) = tfs.get_mut(gh.parts.knees[i]) {
                t.rotation = Quat::from_rotation_x(knee);
            }
            if let Ok(mut t) = tfs.get_mut(gh.parts.antennae[i]) {
                let sway = (gh.phase * 2.3 + side * 1.7).sin() * 0.16;
                t.rotation = Quat::from_rotation_y(-side * (0.35 + sway))
                    * Quat::from_rotation_x(antenna_pitch + (gh.phase * 3.1 + side).sin() * 0.05);
            }
            if let Ok(mut t) = tfs.get_mut(gh.parts.wings[i]) {
                t.rotation = Quat::from_rotation_y(side * flare) * Quat::from_rotation_z(side * flare * 0.4);
            }
        }
    }
}

/// A leaf mantis built from flat leaf-shaped plates and sticks (facing -Z, feet on y = 0).
fn spawn_leaf_mantis(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &mut Assets<StandardMaterial>,
    pos: Vec3,
    seed: u32,
) {
    let mat = |mats: &mut Assets<StandardMaterial>, c: Color, rough: f32| {
        mats.add(StandardMaterial {
            base_color: c,
            perceptual_roughness: rough,
            ..default()
        })
    };
    let dry = seed % 2 == 1; // some are fresh-leaf green, some dead-leaf brown
    let leaf = mat(
        mats,
        if dry { Color::srgb(0.5, 0.36, 0.15) } else { Color::srgb(0.32, 0.5, 0.13) },
        0.8,
    );
    let rib = mat(
        mats,
        if dry { Color::srgb(0.33, 0.22, 0.09) } else { Color::srgb(0.2, 0.32, 0.07) },
        0.8,
    );
    let body = mat(
        mats,
        if dry { Color::srgb(0.42, 0.32, 0.14) } else { Color::srgb(0.38, 0.45, 0.14) },
        0.7,
    );
    let lobe = mat(
        mats,
        if dry { Color::srgb(0.55, 0.42, 0.18) } else { Color::srgb(0.45, 0.6, 0.17) },
        0.8,
    );
    let eye = mat(mats, Color::srgb(0.16, 0.1, 0.02), 0.25);
    let ball = meshes.add(Sphere::new(0.5));
    let stick = meshes.add(Cylinder::new(1.0, 1.0));

    let root = commands
        .spawn(SpatialBundle::from_transform(Transform::from_translation(pos)))
        .id();
    let mut head = Entity::PLACEHOLDER;
    let mut shoulders = [Entity::PLACEHOLDER; 2];
    let mut elbows = [Entity::PLACEHOLDER; 2];
    let mut legs = [Entity::PLACEHOLDER; 4];
    let mut wings = [Entity::PLACEHOLDER; 2];

    commands.entity(root).with_children(|p| {
        // abdomen and thorax
        gh_blob(p, &ball, &body, Vec3::new(0.0, 0.075, 0.075), Vec3::new(0.05, 0.032, 0.13), Quat::IDENTITY);
        gh_blob(p, &ball, &body, Vec3::new(0.0, 0.085, -0.025), Vec3::new(0.034, 0.034, 0.075), Quat::IDENTITY);
        // the long neck (pronotum) rising forward, with a leafy flange along it
        gh_limb(p, &stick, &body, Vec3::new(0.0, 0.09, -0.05), Vec3::new(0.0, 0.142, -0.118), 0.016);
        gh_blob(p, &ball, &lobe, Vec3::new(0.0, 0.12, -0.086), Vec3::new(0.05, 0.006, 0.075), Quat::from_rotation_x(0.75));

        // head on a pivot so it can turn to watch you
        head = p
            .spawn(SpatialBundle::from_transform(Transform::from_xyz(0.0, 0.148, -0.125)))
            .with_children(|h| {
                gh_blob(h, &ball, &body, Vec3::ZERO, Vec3::new(0.052, 0.034, 0.032), Quat::from_rotation_x(0.2));
                for side in [-1.0_f32, 1.0] {
                    gh_blob(h, &ball, &eye, Vec3::new(side * 0.028, 0.004, -0.006), Vec3::splat(0.024), Quat::IDENTITY);
                    gh_limb(
                        h,
                        &stick,
                        &rib,
                        Vec3::new(side * 0.012, 0.012, -0.012),
                        Vec3::new(side * 0.03, 0.045, -0.05),
                        0.0016,
                    );
                }
            })
            .id();

        for (i, side) in [-1.0_f32, 1.0].into_iter().enumerate() {
            // raptorial front legs: shoulder -> femur, elbow -> folded tibia
            shoulders[i] = p
                .spawn(SpatialBundle::from_transform(Transform::from_xyz(side * 0.02, 0.125, -0.11)))
                .with_children(|sh| {
                    let elbow_at = Vec3::new(0.0, -0.02, -0.06);
                    gh_limb(sh, &stick, &body, Vec3::ZERO, elbow_at, 0.008);
                    gh_blob(sh, &ball, &lobe, elbow_at * 0.5, Vec3::new(0.02, 0.004, 0.04), Quat::IDENTITY);
                    elbows[i] = sh
                        .spawn(SpatialBundle::from_transform(Transform::from_translation(elbow_at)))
                        .with_children(|el| {
                            gh_limb(el, &stick, &rib, Vec3::ZERO, Vec3::new(0.0, 0.05, 0.035), 0.0055);
                        })
                        .id();
                })
                .id();

            // wing: a broad leaf plate with a midrib, on a pivot so it can open
            wings[i] = p
                .spawn(SpatialBundle::from_transform(Transform::from_xyz(side * 0.016, 0.094, -0.005)))
                .with_children(|w| {
                    gh_blob(
                        w,
                        &ball,
                        &leaf,
                        Vec3::new(side * 0.03, 0.0, 0.1),
                        Vec3::new(0.07, 0.005, 0.2),
                        Quat::from_rotation_z(side * 0.1),
                    );
                    gh_limb(
                        w,
                        &stick,
                        &rib,
                        Vec3::new(side * 0.03, 0.004, 0.0),
                        Vec3::new(side * 0.03, 0.004, 0.2),
                        0.0018,
                    );
                })
                .id();

            // middle (row 0) and hind (row 1) legs, each with a little leaf lobe at the knee
            for (row, (hip_z, knee_dz, foot_dz)) in [(-0.005_f32, -0.01_f32, -0.01_f32), (0.075, 0.04, 0.055)]
                .into_iter()
                .enumerate()
            {
                let knee_at = Vec3::new(side * 0.05, 0.012, knee_dz);
                let foot_at = Vec3::new(side * 0.066, -0.075, foot_dz);
                legs[row * 2 + i] = p
                    .spawn(SpatialBundle::from_transform(Transform::from_xyz(side * 0.02, 0.075, hip_z)))
                    .with_children(|l| {
                        gh_limb(l, &stick, &body, Vec3::ZERO, knee_at, 0.0045);
                        gh_limb(l, &stick, &body, knee_at, foot_at, 0.0032);
                        gh_blob(l, &ball, &lobe, knee_at, Vec3::new(0.03, 0.004, 0.024), Quat::IDENTITY);
                    })
                    .id();
            }
        }
    });

    let yaw = pseudo_rand(seed) * std::f32::consts::TAU;
    commands.entity(root).insert((
        Transform::from_translation(pos).with_rotation(Quat::from_rotation_y(yaw)),
        LeafMantis {
            state: MantisState::Hold,
            timer: 1.0 + pseudo_rand(seed.wrapping_add(1)) * 4.0,
            pos,
            yaw,
            target: pos,
            home: pos,
            seed,
            phase: pseudo_rand(seed.wrapping_add(2)) * 10.0,
            strike: -1.0,
            cooldown: 0.0,
            head_yaw: 0.0,
            rear: 0.0,
            parts: MantisParts {
                head,
                shoulders,
                elbows,
                legs,
                wings,
            },
        },
    ));
}

fn leaf_mantis_system(
    time: Res<Time>,
    player: Query<&Transform, With<Player>>,
    mut mantises: Query<(Entity, &mut LeafMantis)>,
    mut hoppers: Query<&mut Grasshopper>,
    mut tfs: Query<&mut Transform, Without<Player>>,
) {
    let dt = time.delta_seconds().min(0.05);
    let player_pos = player.get_single().ok().map(|t| t.translation);
    let smooth = |x: f32| {
        let x = x.clamp(0.0, 1.0);
        x * x * (3.0 - 2.0 * x)
    };

    for (entity, mut m) in &mut mantises {
        m.phase += dt;
        m.cooldown -= dt;
        let to_player = player_pos.map(|p| Vec2::new(p.x - m.pos.x, p.z - m.pos.z));
        let dist = to_player.map_or(99.0, |v| v.length());

        // ----- what it is doing -----
        if dist < 1.4 && m.state != MantisState::Rear {
            m.state = MantisState::Rear;
        }
        let state = m.state;
        match state {
            MantisState::Hold => {
                m.timer -= dt;
                if m.timer <= 0.0 {
                    m.seed = m.seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let angle = pseudo_rand(m.seed) * std::f32::consts::TAU;
                    let d = 0.4 + pseudo_rand(m.seed.wrapping_add(1)) * 0.5;
                    let x = m.pos.x + angle.cos() * d;
                    let z = m.pos.z + angle.sin() * d;
                    let from_home = Vec2::new(x - m.home.x, z - m.home.z).length();
                    if pseudo_rand(m.seed.wrapping_add(2)) < 0.55
                        && grasshopper_spot_ok(x, z)
                        && from_home < 3.5
                    {
                        m.target = Vec3::new(x, land_surface_height(x, z), z);
                        m.state = MantisState::Walk;
                    } else {
                        m.timer = 3.0 + pseudo_rand(m.seed.wrapping_add(3)) * 5.0;
                    }
                }
            }
            MantisState::Walk => {
                let d = m.target - m.pos;
                let flat = Vec2::new(d.x, d.z);
                if flat.length() < 0.03 {
                    m.state = MantisState::Hold;
                    m.seed = m.seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    m.timer = 3.0 + pseudo_rand(m.seed) * 5.0;
                } else {
                    let want = (-d.x).atan2(-d.z);
                    m.yaw = lerp_angle(m.yaw, want, (dt * 3.0).min(1.0));
                    // a stalking step-and-pause rhythm
                    let speed = 0.13 * (0.3 + 0.7 * (m.phase * 4.0).sin().abs());
                    let step = flat.normalize() * (speed * dt).min(flat.length());
                    let (x, z) = (m.pos.x + step.x, m.pos.z + step.y);
                    m.pos = Vec3::new(x, land_surface_height(x, z), z);
                }
            }
            MantisState::Rear => {
                if let Some(v) = to_player {
                    let want = (-v.x).atan2(-v.y);
                    m.yaw = lerp_angle(m.yaw, want, (dt * 4.0).min(1.0));
                }
                if dist > 2.2 {
                    m.state = MantisState::Hold;
                    m.timer = 2.0;
                }
            }
        }

        // ----- striking: at a person who is very close, or a grasshopper that hops too near -----
        if m.strike < 0.0 && m.cooldown <= 0.0 {
            let mut go = m.state == MantisState::Rear && dist < 0.75;
            if m.state == MantisState::Hold {
                for mut g in &mut hoppers {
                    let d = Vec2::new(g.pos.x - m.pos.x, g.pos.z - m.pos.z);
                    let in_front = lerp_angle(m.yaw, (-d.x).atan2(-d.y), 1.0) - m.yaw;
                    if g.state == HopState::Resting && d.length() < 0.45 && in_front.abs() < 1.2 {
                        g.timer = 0.0; // it was nearly caught: it bolts
                        go = true;
                    }
                }
            }
            if go {
                m.strike = 0.0;
                m.seed = m.seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                m.cooldown = 1.6 + pseudo_rand(m.seed) * 1.2;
            }
        }
        let mut strike_ext = 0.0;
        if m.strike >= 0.0 {
            m.strike += dt / 0.6;
            let t = m.strike;
            strike_ext = if t < 0.2 {
                smooth(t / 0.2)
            } else if t < 0.5 {
                1.0
            } else {
                1.0 - smooth((t - 0.5) / 0.5)
            };
            if t >= 1.0 {
                m.strike = -1.0;
            }
        }

        // ----- pose -----
        let rear_target = if m.state == MantisState::Rear { 1.0 } else { 0.0 };
        m.rear += (rear_target - m.rear) * (dt * 5.0).min(1.0);
        let rear = m.rear;
        let roll = if m.state == MantisState::Rear {
            (m.phase * 2.0).sin() * 0.03
        } else {
            (m.phase * 1.1).sin() * 0.05 + (m.phase * 2.7).sin() * 0.02 // a leaf in the breeze
        };
        // the head turns to watch you
        let head_want = match to_player {
            Some(v) if dist < 6.0 => {
                let rel = lerp_angle(m.yaw, (-v.x).atan2(-v.y), 1.0) - m.yaw;
                rel.clamp(-1.1, 1.1)
            }
            _ => (m.phase * 0.5).sin() * 0.25,
        };
        m.head_yaw = lerp_angle(m.head_yaw, head_want, (dt * 5.0).min(1.0));

        if let Ok(mut t) = tfs.get_mut(entity) {
            t.translation = m.pos;
            t.rotation = Quat::from_rotation_y(m.yaw)
                * Quat::from_rotation_z(roll)
                * Quat::from_rotation_x(0.3 * rear);
        }
        if let Ok(mut t) = tfs.get_mut(m.parts.head) {
            t.rotation = Quat::from_rotation_y(m.head_yaw) * Quat::from_rotation_x(-0.25 * rear);
        }
        let walking = m.state == MantisState::Walk;
        let swing = if walking { (m.phase * 4.0).sin() * 0.35 } else { 0.0 };
        let leg_swing = [swing, -swing, -swing, swing];
        for (leg, angle) in m.parts.legs.iter().zip(leg_swing) {
            if let Ok(mut t) = tfs.get_mut(*leg) {
                t.rotation = Quat::from_rotation_x(angle);
            }
        }
        let spread = rear * 0.8 + strike_ext * 0.3;
        let flutter = (m.phase * 30.0).sin() * 0.05 * rear;
        for i in 0..2 {
            let side = if i == 0 { -1.0 } else { 1.0 };
            if let Ok(mut t) = tfs.get_mut(m.parts.shoulders[i]) {
                t.rotation = Quat::from_rotation_y(-side * (0.1 + 0.4 * rear))
                    * Quat::from_rotation_x(0.7 * rear + 0.35 * strike_ext);
            }
            if let Ok(mut t) = tfs.get_mut(m.parts.elbows[i]) {
                t.rotation = Quat::from_rotation_x(-0.3 * rear - 2.2 * strike_ext);
            }
            if let Ok(mut t) = tfs.get_mut(m.parts.wings[i]) {
                t.rotation = Quat::from_rotation_y(side * (spread + flutter))
                    * Quat::from_rotation_z(side * spread * 0.4);
            }
        }
    }
}

/// A wooden-handled axe with a metal head, initially resting on the table.
fn spawn_axe(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &mut Assets<StandardMaterial>,
    pos: Vec3,
) {
    let steel = mats.add(StandardMaterial {
        base_color: Color::srgb(0.65, 0.68, 0.72),
        metallic: 0.85,
        perceptual_roughness: 0.28,
        ..default()
    });
    let wood = mats.add(Color::srgb(0.36, 0.2, 0.09));
    let groups = CollisionGroups::new(Group::GROUP_2, Group::ALL);
    let lying = Quat::from_mat3(&Mat3::from_cols(Vec3::Y, Vec3::X, Vec3::NEG_Z));

    commands
        .spawn((
            SpatialBundle::from_transform(Transform::from_translation(pos).with_rotation(lying)),
            RigidBody::Dynamic,
            Velocity::default(),
            Restitution::coefficient(0.1),
            Friction::coefficient(0.8),
            Sleeping::disabled(),
            Axe {
                equipped: false,
                psi: SWORD_REST_PSI,
                phi: SWORD_REST_PHI,
                psi_vel: 0.0,
                roll: 0.0,
                chop_cooldown: 0.0,
            },
        ))
        .with_children(|p| {
            p.spawn((
                PbrBundle {
                    mesh: meshes.add(Cuboid::new(0.065, 0.72, 0.065)),
                    material: wood,
                    ..default()
                },
                Collider::cuboid(0.0325, 0.36, 0.0325),
                groups,
                AxeHandle,
            ));
            p.spawn((
                PbrBundle {
                    mesh: meshes.add(Cuboid::new(0.42, 0.2, 0.14)),
                    material: steel,
                    transform: Transform::from_xyz(0.0, 0.42, 0.0),
                    ..default()
                },
                Collider::cuboid(0.21, 0.1, 0.07),
                groups,
            ));
        });
}

/// A two-handed shovel with a grabbable handle and a broad metal blade.
fn spawn_shovel(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &mut Assets<StandardMaterial>,
    pos: Vec3,
) {
    let wood = mats.add(Color::srgb(0.42, 0.25, 0.12));
    let steel = mats.add(StandardMaterial {
        base_color: Color::srgb(0.48, 0.53, 0.58),
        metallic: 0.72,
        perceptual_roughness: 0.38,
        ..default()
    });
    let groups = CollisionGroups::new(Group::GROUP_2, Group::ALL);
    let lying = Quat::from_mat3(&Mat3::from_cols(Vec3::Y, Vec3::X, Vec3::NEG_Z));

    commands
        .spawn((
            SpatialBundle::from_transform(Transform::from_translation(pos).with_rotation(lying)),
            RigidBody::Dynamic,
            Velocity::default(),
            Restitution::coefficient(0.1),
            Friction::coefficient(0.8),
            Sleeping::disabled(),
            Shovel {
                equipped: false,
                swing: 0.0,
                dig_cooldown: 0.0,
            },
        ))
        .with_children(|p| {
            p.spawn((
                PbrBundle {
                    mesh: meshes.add(Cylinder::new(0.035, 1.0)),
                    material: wood.clone(),
                    ..default()
                },
                Collider::cylinder(0.5, 0.035),
                groups,
                ShovelHandle,
            ));
            p.spawn(PbrBundle {
                mesh: meshes.add(Sphere::new(0.075)),
                material: wood,
                transform: Transform::from_xyz(0.0, -0.48, 0.0),
                ..default()
            });
            p.spawn((
                PbrBundle {
                    mesh: meshes.add(Cuboid::new(0.34, 0.25, 0.055)),
                    material: steel,
                    transform: Transform::from_xyz(0.0, 0.58, 0.0),
                    ..default()
                },
                Collider::cuboid(0.17, 0.125, 0.0275),
                groups,
            ));
        });
}

/// A small tapered wooden pail with a wire bail handle. The origin is the middle
/// of the underside of the base. Every part is a collider, so you can grab it by
/// the handle, the rim or the side and it will hang / tilt accordingly.
fn spawn_bucket(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    position: Vec3,
) {
    let wood = materials.add(Color::srgb(0.43, 0.27, 0.13));
    let metal = materials.add(StandardMaterial {
        base_color: Color::srgb(0.42, 0.45, 0.47),
        metallic: 0.72,
        perceptual_roughness: 0.34,
        ..default()
    });
    let water_material = materials.add(StandardMaterial {
        base_color: Color::srgba(0.12, 0.48, 0.72, 0.86),
        alpha_mode: AlphaMode::Blend,
        perceptual_roughness: 0.22,
        ..default()
    });

    const SIDES: usize = 12;
    let wall_tilt = ((BUCKET_TOP_R - BUCKET_BOTTOM_R) / BUCKET_HEIGHT).atan();
    let slat_len = (BUCKET_HEIGHT * BUCKET_HEIGHT
        + (BUCKET_TOP_R - BUCKET_BOTTOM_R) * (BUCKET_TOP_R - BUCKET_BOTTOM_R))
        .sqrt();
    let slat_radius = (BUCKET_BOTTOM_R + BUCKET_TOP_R) * 0.5 - BUCKET_WALL_T * 0.5;
    let slat_width = std::f32::consts::TAU * slat_radius / SIDES as f32 * 1.08;
    let slat_mesh = meshes.add(Cuboid::new(slat_width, slat_len, BUCKET_WALL_T));

    // Bail handle: a half circle of thin wire segments above the rim, pinned at two ears.
    const BAIL_SEGMENTS: usize = 8;
    let bail_radius = BUCKET_TOP_R + 0.004;
    let bail_point = |i: usize| {
        let a = i as f32 / BAIL_SEGMENTS as f32 * std::f32::consts::PI;
        Vec3::new(
            a.cos() * bail_radius,
            BUCKET_HEIGHT + a.sin() * bail_radius * 0.95,
            0.0,
        )
    };
    let segment_len = (bail_point(1) - bail_point(0)).length();
    let wire_mesh = meshes.add(Cylinder::new(0.008, segment_len));
    let ear_mesh = meshes.add(Sphere::new(0.012));

    commands
        .spawn((
            SpatialBundle::from_transform(Transform::from_translation(position)),
            RigidBody::Dynamic,
            Velocity::default(),
            Restitution::coefficient(0.08),
            Friction::coefficient(0.82),
            Sleeping::disabled(),
            // Collider mass is tiny (density 1), so give the pail a believable weight;
            // bucket_scoop_system adds the weight of the water on top of this.
            AdditionalMassProperties::Mass(BUCKET_EMPTY_MASS),
            Bucket::default(),
        ))
        .with_children(|bucket| {
            // base
            bucket.spawn((
                PbrBundle {
                    mesh: meshes.add(Cylinder::new(BUCKET_BOTTOM_R, BUCKET_BASE_T)),
                    material: wood.clone(),
                    transform: Transform::from_xyz(0.0, BUCKET_BASE_T * 0.5, 0.0),
                    ..default()
                },
                Collider::cylinder(BUCKET_BASE_T * 0.5, BUCKET_BOTTOM_R),
            ));

            // tapered wall made of slats, wider at the rim
            for side in 0..SIDES {
                let angle = side as f32 * std::f32::consts::TAU / SIDES as f32;
                let (sin, cos) = angle.sin_cos();
                bucket.spawn((
                    PbrBundle {
                        mesh: slat_mesh.clone(),
                        material: wood.clone(),
                        transform: Transform::from_xyz(
                            sin * slat_radius,
                            BUCKET_HEIGHT * 0.5,
                            cos * slat_radius,
                        )
                        .with_rotation(
                            Quat::from_rotation_y(angle) * Quat::from_rotation_x(wall_tilt),
                        ),
                        ..default()
                    },
                    Collider::cuboid(slat_width * 0.5, slat_len * 0.5, BUCKET_WALL_T * 0.5),
                ));
            }

            // rim ring and one metal hoop
            bucket.spawn(PbrBundle {
                mesh: meshes.add(Torus::new(BUCKET_TOP_R - 0.014, BUCKET_TOP_R + 0.006)),
                material: metal.clone(),
                transform: Transform::from_xyz(0.0, BUCKET_HEIGHT, 0.0),
                ..default()
            });
            let hoop_y = BUCKET_HEIGHT * 0.28;
            let hoop_r = BUCKET_BOTTOM_R + (BUCKET_TOP_R - BUCKET_BOTTOM_R) * 0.28;
            bucket.spawn(PbrBundle {
                mesh: meshes.add(Torus::new(hoop_r - 0.002, hoop_r + 0.006)),
                material: metal.clone(),
                transform: Transform::from_xyz(0.0, hoop_y, 0.0),
                ..default()
            });

            // bail handle (grabbable): all its wire pieces hang off one pivot at the ears,
            // which bail_swing_system rotates
            bucket
                .spawn((
                    SpatialBundle::from_transform(Transform::from_xyz(0.0, BUCKET_HEIGHT, 0.0)),
                    BailPivot::default(),
                ))
                .with_children(|pivot| {
                    for i in 0..BAIL_SEGMENTS {
                        let (p0, p1) = (bail_point(i), bail_point(i + 1));
                        let dir = (p1 - p0).normalize();
                        pivot.spawn((
                            PbrBundle {
                                mesh: wire_mesh.clone(),
                                material: metal.clone(),
                                transform: Transform::from_translation(
                                    (p0 + p1) * 0.5 - Vec3::Y * BUCKET_HEIGHT,
                                )
                                .with_rotation(Quat::from_rotation_arc(Vec3::Y, dir)),
                                ..default()
                            },
                            Collider::capsule_y(segment_len * 0.5, 0.03), // generous, so the thin wire is easy to click
                        ));
                    }
                });
            for sign in [-1.0_f32, 1.0] {
                bucket.spawn(PbrBundle {
                    mesh: ear_mesh.clone(),
                    material: metal.clone(),
                    transform: Transform::from_xyz(sign * bail_radius, BUCKET_HEIGHT, 0.0),
                    ..default()
                });
            }

            // the water inside: a flat disc that always stays level with the world
            bucket.spawn((
                PbrBundle {
                    mesh: meshes.add(Cylinder::new(1.0, 1.0)),
                    material: water_material,
                    transform: Transform::from_xyz(0.0, BUCKET_BASE_T, 0.0)
                        .with_scale(Vec3::new(0.01, 0.012, 0.01)),
                    visibility: Visibility::Hidden,
                    ..default()
                },
                BucketWater,
            ));
        });
}

/// Scooping: water flows into the pail whenever the lowest point of its rim is
/// under the river surface (so you can dunk it, or just tip it into the water).
/// Pouring: when it is lifted out and tilted, the water stands level in the pail
/// and everything above the lowest point of the rim spills out.
fn bucket_scoop_system(
    time: Res<Time>,
    river: Res<RiverState>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut splash_assets: Local<Option<(Handle<Mesh>, Handle<StandardMaterial>)>>,
    mut buckets: Query<
        (
            Entity,
            &Transform,
            &Velocity,
            &mut Bucket,
            Option<&mut AdditionalMassProperties>,
        ),
        Without<BucketWater>,
    >,
    mut water_visuals: Query<(&Parent, &mut Transform, &mut Visibility), With<BucketWater>>,
    mut status: ResMut<Status>,
) {
    let dt = time.delta_seconds().min(0.05);
    let level = river.water_level + BUCKET_WATER_TOLERANCE;
    let interior = BUCKET_HEIGHT - BUCKET_BASE_T;

    for (entity, tf, velocity, mut bucket, extra_mass) in &mut buckets {
        let up = tf.rotation * Vec3::Y;
        let floor = tf.translation + up * BUCKET_BASE_T;
        let rim_center = tf.translation + up * BUCKET_HEIGHT;
        let tilt_sin = (1.0 - up.y * up.y).max(0.0).sqrt();
        let low_rim = rim_center.y - BUCKET_TOP_R * tilt_sin;
        let upright = up.y > 0.05;
        let axis_y = up.y.max(0.05);
        // fill fraction at which the (level) water surface, measured on the pail's axis, is at height y
        let fraction_at = |y: f32| ((y - floor.y) / axis_y / interior).clamp(0.0, 1.0);

        let over_river = (rim_center.x - river_center_x(rim_center.z)).abs()
            < RIVER_WIDTH * 0.5 + river.overflow
            && rim_center.z.abs() < GROUND_HALF_SIZE;
        let mouth_in_water = over_river && upright && low_rim < level;
        let before = bucket.fill;

        if mouth_in_water {
            // water pours in over the submerged part of the rim until it is level with the river
            let depth = ((level - low_rim) / 0.05).clamp(0.2, 1.0);
            let target = fraction_at(level);
            if bucket.fill < target {
                bucket.fill = (bucket.fill + 1.8 * depth * dt).min(target);
            } else {
                bucket.fill = (bucket.fill - 1.0 * dt).max(target);
            }
            if !bucket.in_water && velocity.linvel.length() > 0.8 {
                spawn_water_splash(
                    Vec3::new(rim_center.x, 0.0, rim_center.z),
                    river.water_level,
                    &mut commands,
                    &mut meshes,
                    &mut materials,
                    &mut splash_assets,
                );
            }
        } else {
            // out of the river: nothing can stay above the lowest point of the rim
            let limit = if upright { fraction_at(low_rim) } else { 0.0 };
            if bucket.fill > limit {
                let pour = (0.7 + 1.5 * (1.0 - up.y).clamp(0.0, 2.0)) * dt;
                bucket.fill = (bucket.fill - pour).max(limit);
            }
        }
        bucket.in_water = mouth_in_water;

        // drops falling from the lowest point of the rim while pouring
        let poured = (before - bucket.fill).max(0.0);
        if poured > 0.0 && !mouth_in_water {
            bucket.drip_debt += poured * 120.0;
            let (drop_mesh, drop_material) =
                water_drop_assets(&mut splash_assets, &mut meshes, &mut materials);
            let out = up * up.y - Vec3::Y; // points down the slope of the rim plane
            while bucket.drip_debt >= 1.0 {
                bucket.drip_debt -= 1.0;
                let seed = ((time.elapsed_seconds() * 1000.0) as u32)
                    ^ entity.index().wrapping_mul(2_654_435_761)
                    ^ bucket.drip_count.wrapping_mul(97);
                bucket.drip_count = bucket.drip_count.wrapping_add(1);
                let dir = if out.length() > 0.05 {
                    out.normalize()
                } else {
                    // upside down: water leaves all around the rim
                    Quat::from_axis_angle(up, pseudo_rand(seed) * std::f32::consts::TAU)
                        * up.any_orthonormal_vector()
                };
                let flat = Vec3::new(dir.x, 0.0, dir.z).normalize_or_zero();
                let jitter = Vec3::new(
                    pseudo_rand(seed.wrapping_add(1)) - 0.5,
                    0.0,
                    pseudo_rand(seed.wrapping_add(2)) - 0.5,
                ) * 0.3;
                commands.spawn((
                    PbrBundle {
                        mesh: drop_mesh.clone(),
                        material: drop_material.clone(),
                        transform: Transform::from_translation(rim_center + dir * BUCKET_TOP_R)
                            .with_scale(Vec3::splat(0.28)),
                        ..default()
                    },
                    SplashParticle {
                        velocity: flat * (0.5 + pseudo_rand(seed.wrapping_add(3)) * 0.8)
                            + Vec3::Y * (0.15 + pseudo_rand(seed.wrapping_add(4)) * 0.3)
                            + velocity.linvel * 0.4
                            + jitter,
                        lifetime: 0.9,
                    },
                ));
            }
        }

        // water has weight: a full pail is much heavier to swing around
        let load = bucket.fill + bucket.mud * BUCKET_MUD_MASS_FACTOR;
        if (load - bucket.mass_fill).abs() > 0.03 || (load == 0.0 && bucket.mass_fill != 0.0) {
            if let Some(mut extra) = extra_mass {
                *extra = AdditionalMassProperties::Mass(
                    BUCKET_EMPTY_MASS + BUCKET_WATER_MASS * load,
                );
            }
            bucket.mass_fill = load;
        }

        let state = if bucket.fill >= 0.9 {
            2
        } else if bucket.fill > 0.05 {
            1
        } else {
            0
        };
        if state != bucket.state {
            status.0 = match (bucket.state, state) {
                (_, 2) => "The bucket is full of river water.".into(),
                (0, 1) => "You scoop some river water into the bucket.".into(),
                (_, 0) => "The bucket is empty.".into(),
                _ => "Water spills out of the bucket.".into(),
            };
            bucket.state = state;
        }
    }

    // The surface disc stays level with the world however the pail is tilted.
    for (parent, mut transform, mut visibility) in &mut water_visuals {
        let Ok((_, bucket_tf, _, bucket, _)) = buckets.get(parent.get()) else {
            continue;
        };
        if bucket.fill < 0.01 {
            if *visibility != Visibility::Hidden {
                *visibility = Visibility::Hidden;
            }
            continue;
        }
        if *visibility != Visibility::Visible {
            *visibility = Visibility::Visible;
        }
        let y = BUCKET_BASE_T + bucket.fill * interior;
        let wall_r = BUCKET_BOTTOM_R + (BUCKET_TOP_R - BUCKET_BOTTOM_R) * (y / BUCKET_HEIGHT);
        let radius = (wall_r - BUCKET_WALL_T - 0.004).max(0.01);
        transform.translation = Vec3::Y * y;
        transform.rotation = bucket_tf.rotation.inverse();
        transform.scale = Vec3::new(radius, 0.012, radius);
    }
}

fn spawn_spear(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &mut Assets<StandardMaterial>,
    pos: Vec3,
) {
    let wood = mats.add(Color::srgb(0.38, 0.23, 0.12));
    let steel = mats.add(StandardMaterial {
        base_color: Color::srgb(0.72, 0.76, 0.8),
        metallic: 0.88,
        perceptual_roughness: 0.24,
        ..default()
    });
    let groups = CollisionGroups::new(Group::GROUP_2, Group::ALL);
    let lying = Quat::from_mat3(&Mat3::from_cols(Vec3::Y, Vec3::X, Vec3::NEG_Z));
    commands
        .spawn((
            SpatialBundle::from_transform(Transform::from_translation(pos).with_rotation(lying)),
            RigidBody::Dynamic,
            Velocity::default(),
            Restitution::coefficient(0.08),
            Friction::coefficient(0.8),
            Sleeping::disabled(),
            Spear {
                equipped: false,
                impaled_fish: None,
                attack_left: 0.0,
                attack_duration: 0.34,
                push_distance: 0.0,
                attack_direction: Vec3::NEG_Z,
                cooldown: 0.0,
                cook_left: 0.0,
                cook_point: Vec3::ZERO,
            },
        ))
        .with_children(|parent| {
            parent.spawn((
                PbrBundle {
                    mesh: meshes.add(Cylinder::new(0.035, 1.55)),
                    material: wood.clone(),
                    transform: Transform::from_xyz(0.0, 0.78, 0.0),
                    ..default()
                },
                Collider::cylinder(0.775, 0.035),
                groups,
                SpearHandle,
            ));
            parent.spawn(PbrBundle {
                mesh: meshes.add(Cone {
                    radius: 0.09,
                    height: 0.38,
                }),
                material: steel,
                transform: Transform::from_xyz(0.0, 1.72, 0.0),
                ..default()
            });
            parent.spawn(PbrBundle {
                mesh: meshes.add(Cuboid::new(0.09, 0.08, 0.12)),
                material: wood,
                transform: Transform::from_xyz(0.0, 1.52, 0.0),
                ..default()
            });
        });
}

/// A machete: short blade with a curved tip, on a stubby handle. Lay it flat
/// on the table with the same orientation the sword uses.
fn spawn_machete(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &mut Assets<StandardMaterial>,
    pos: Vec3,
) {
    let steel = mats.add(StandardMaterial {
        base_color: Color::srgb(0.72, 0.75, 0.79),
        metallic: 0.86,
        perceptual_roughness: 0.3,
        ..default()
    });
    let grip = mats.add(Color::srgb(0.16, 0.09, 0.05));
    let ferrule = mats.add(StandardMaterial {
        base_color: Color::srgb(0.52, 0.52, 0.48),
        metallic: 0.7,
        perceptual_roughness: 0.45,
        ..default()
    });
    let groups = CollisionGroups::new(Group::GROUP_2, Group::ALL);
    let lying = Quat::from_mat3(&Mat3::from_cols(Vec3::Y, Vec3::X, Vec3::NEG_Z));

    commands
        .spawn((
            SpatialBundle::from_transform(Transform::from_translation(pos).with_rotation(lying)),
            RigidBody::Dynamic,
            Velocity::default(),
            Restitution::coefficient(0.1),
            Friction::coefficient(0.8),
            Sleeping::disabled(),
            Machete {
                equipped: false,
                psi: SWORD_REST_PSI,
                phi: SWORD_REST_PHI,
                psi_vel: 0.0,
                roll: 0.0,
                slash_cooldown: 0.0,
                strip_cooldown: 0.0,
            },
        ))
        .with_children(|p| {
            // grip (the origin of the machete is the middle of the grip)
            p.spawn((
                PbrBundle {
                    mesh: meshes.add(Cuboid::new(0.04, 0.20, 0.055)),
                    material: grip,
                    ..default()
                },
                Collider::cuboid(0.02, 0.10, 0.0275),
                groups,
                MacheteHandle,
            ));
            // small metal ferrule between grip and blade
            p.spawn((
                PbrBundle {
                    mesh: meshes.add(Cuboid::new(0.05, 0.03, 0.08)),
                    material: ferrule,
                    transform: Transform::from_xyz(0.0, 0.115, 0.0),
                    ..default()
                },
                Collider::cuboid(0.025, 0.015, 0.04),
                groups,
            ));
            // broad blade (wider than a sword blade, shorter than the sword)
            p.spawn((
                PbrBundle {
                    mesh: meshes.add(Cuboid::new(0.014, 0.46, 0.095)),
                    material: steel.clone(),
                    transform: Transform::from_xyz(0.0, 0.36, 0.012),
                    ..default()
                },
                Collider::cuboid(0.007, 0.23, 0.0475),
                groups,
            ));
            // curved-ish tip
            p.spawn(PbrBundle {
                mesh: meshes.add(Cuboid::new(0.012, 0.16, 0.06)),
                material: steel,
                transform: Transform::from_xyz(0.0, 0.66, 0.035),
                ..default()
            });
        });
}

/// A small knife: short blade with a pointed tip on a stubby handle, laid flat on
/// the table with the same orientation the sword and machete use.
fn spawn_knife(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &mut Assets<StandardMaterial>,
    pos: Vec3,
) {
    let steel = mats.add(StandardMaterial {
        base_color: Color::srgb(0.8, 0.82, 0.86),
        metallic: 0.9,
        perceptual_roughness: 0.22,
        ..default()
    });
    let grip = mats.add(Color::srgb(0.12, 0.07, 0.04));
    let bolster = mats.add(StandardMaterial {
        base_color: Color::srgb(0.55, 0.55, 0.5),
        metallic: 0.75,
        perceptual_roughness: 0.42,
        ..default()
    });
    let groups = CollisionGroups::new(Group::GROUP_2, Group::ALL);
    let lying = Quat::from_mat3(&Mat3::from_cols(Vec3::Y, Vec3::X, Vec3::NEG_Z));

    commands
        .spawn((
            SpatialBundle::from_transform(Transform::from_translation(pos).with_rotation(lying)),
            RigidBody::Dynamic,
            Velocity::default(),
            Restitution::coefficient(0.1),
            Friction::coefficient(0.8),
            Sleeping::disabled(),
            Knife {
                equipped: false,
                psi: SWORD_REST_PSI,
                phi: SWORD_REST_PHI,
                psi_vel: 0.0,
                roll: 0.0,
                slash_cooldown: 0.0,
            },
        ))
        .with_children(|p| {
            // grip (the origin of the knife is the middle of the grip)
            p.spawn((
                PbrBundle {
                    mesh: meshes.add(Cuboid::new(0.03, 0.13, 0.038)),
                    material: grip,
                    ..default()
                },
                Collider::cuboid(0.015, 0.065, 0.019),
                groups,
                KnifeHandle,
            ));
            // small metal bolster between grip and blade
            p.spawn((
                PbrBundle {
                    mesh: meshes.add(Cuboid::new(0.042, 0.018, 0.05)),
                    material: bolster,
                    transform: Transform::from_xyz(0.0, 0.074, 0.0),
                    ..default()
                },
                Collider::cuboid(0.021, 0.009, 0.025),
                groups,
            ));
            // thin blade
            p.spawn((
                PbrBundle {
                    mesh: meshes.add(Cuboid::new(0.009, 0.20, 0.032)),
                    material: steel.clone(),
                    transform: Transform::from_xyz(0.0, 0.184, 0.004),
                    ..default()
                },
                Collider::cuboid(0.0045, 0.10, 0.016),
                groups,
            ));
            // pointed tip
            p.spawn(PbrBundle {
                mesh: meshes.add(Cuboid::new(0.008, 0.06, 0.022)),
                material: steel,
                transform: Transform::from_xyz(0.0, 0.312, 0.008),
                ..default()
            });
        });
}

/// Pick the sword up by its grip (within 3 m), and drop it with Q.
fn sword_input_system(
    mut commands: Commands,
    player: Query<Entity, With<Player>>,
    mut consumed: ResMut<ClickConsumed>,
    mut ui: ResMut<SwordUi>,
    mut status: ResMut<Status>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    rapier: Res<RapierContext>,
    handles: Query<&Parent, With<SwordHandle>>,
    mut tools_and_sword: ParamSet<(
        ToolsHeld,
        Query<(Entity, &mut Sword)>,
    )>,
) {
    let Ok((camera, cam_gt)) = cameras.get_single() else {
        return;
    };
    let Some((sword_e, sword_equipped)) = tools_and_sword
        .p1()
        .get_single()
        .map(|(e, s)| (e, s.equipped))
        .ok()
    else {
        return;
    };

    if sword_equipped {
        ui.hover = false;
        ui.swinging = buttons.pressed(MouseButton::Left);
        if buttons.pressed(MouseButton::Left) {
            consumed.0 = true;
        }
        if keys.just_pressed(KeyCode::KeyQ) {
            let fwd = *cam_gt.forward();
            commands.entity(sword_e).insert((
                RigidBody::Dynamic,
                Velocity::linear(fwd * 2.0 + Vec3::Y * 0.5),
                Sleeping::disabled(),
            ));
            if let Ok((_, mut sword)) = tools_and_sword.p1().get_single_mut() {
                sword.equipped = false;
            }
            ui.swinging = false;
            status.0 = "Dropped the sword.".into();
        }
        return;
    }

    ui.swinging = false;
    ui.hover = false;
    if tools_and_sword.p0().any() {
        return;
    }
    let Ok(window) = windows.get_single() else {
        return;
    };
    let Some(ray) = pointer_position(&window)
        .and_then(|c| camera.viewport_to_world(cam_gt, c))
    else {
        return;
    };
    let mut filter = QueryFilter::default();
    if let Ok(pe) = player.get_single() {
        filter = filter.exclude_collider(pe);
    }
    ui.hover = rapier
        .cast_ray(ray.origin, *ray.direction, 3.0, true, filter)
        .map_or(false, |(e, _)| {
            handles.get(e).map_or(false, |p| p.get() == sword_e)
        });

    if ui.hover && buttons.just_pressed(MouseButton::Left) && !consumed.0 {
        commands
            .entity(sword_e)
            .insert(RigidBody::KinematicPositionBased)
            .remove::<Velocity>();
        if let Ok((_, mut sword)) = tools_and_sword.p1().get_single_mut() {
            sword.equipped = true;
            sword.psi = SWORD_REST_PSI;
            sword.phi = SWORD_REST_PHI;
            sword.psi_vel = 0.0;
            sword.roll = 0.0;
        }
        consumed.0 = true;
        status.0 =
            "Sword in hand: hold the left mouse button and move the mouse to swing. Q drops it."
                .into();
    }
}

/// Pick the axe up by its handle and drop it with Q.
fn axe_input_system(
    mut commands: Commands,
    player: Query<Entity, With<Player>>,
    mut consumed: ResMut<ClickConsumed>,
    mut ui: ResMut<AxeUi>,
    mut status: ResMut<Status>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    rapier: Res<RapierContext>,
    handles: Query<&Parent, With<AxeHandle>>,
    mut tools_and_axe: ParamSet<(
        ToolsHeld,
        Query<(Entity, &mut Axe)>,
    )>,
) {
    let Ok((camera, cam_gt)) = cameras.get_single() else { return; };
    let Some((axe_e, axe_equipped)) = tools_and_axe
        .p1().get_single().map(|(e, a)| (e, a.equipped)).ok()
    else { return; };

    if axe_equipped {
        ui.hover = false;
        ui.swinging = buttons.pressed(MouseButton::Left);
        if buttons.pressed(MouseButton::Left) { consumed.0 = true; }
        if keys.just_pressed(KeyCode::KeyQ) {
            commands.entity(axe_e).insert((
                RigidBody::Dynamic,
                Velocity::linear(*cam_gt.forward() * 2.0 + Vec3::Y * 0.5),
                Sleeping::disabled(),
            ));
            if let Ok((_, mut axe)) = tools_and_axe.p1().get_single_mut() {
                axe.equipped = false;
            }
            ui.swinging = false;
            status.0 = "Dropped the axe.".into();
        }
        return;
    }

    ui.hover = false;
    ui.swinging = false;
    if tools_and_axe.p0().any() { return; }

    let Ok(window) = windows.get_single() else { return; };
    let Some(ray) = pointer_position(&window)
        .and_then(|c| camera.viewport_to_world(cam_gt, c))
    else { return; };
    let mut filter = QueryFilter::default();
    if let Ok(pe) = player.get_single() {
        filter = filter.exclude_collider(pe);
    }
    ui.hover = rapier
        .cast_ray(ray.origin, *ray.direction, 3.0, true, filter)
        .map_or(false, |(e, _)| {
            handles.get(e).map_or(false, |p| p.get() == axe_e)
        });

    if ui.hover && buttons.just_pressed(MouseButton::Left) && !consumed.0 {
        commands
            .entity(axe_e)
            .insert(RigidBody::KinematicPositionBased)
            .remove::<Velocity>();
        if let Ok((_, mut axe)) = tools_and_axe.p1().get_single_mut() {
            axe.equipped = true;
            axe.psi = SWORD_REST_PSI;
            axe.phi = SWORD_REST_PHI;
            axe.psi_vel = 0.0;
            axe.roll = 0.0;
            axe.chop_cooldown = 0.0;
        }
        consumed.0 = true;
        status.0 =
            "Axe equipped: hold LMB and swing at a nearby tree to chop it. Q drops the axe.".into();
    }
}

/// Pick the shovel up by its handle and drop it with Q.
fn shovel_input_system(
    mut commands: Commands,
    player: Query<Entity, With<Player>>,
    mut consumed: ResMut<ClickConsumed>,
    mut ui: ResMut<ShovelUi>,
    mut status: ResMut<Status>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    rapier: Res<RapierContext>,
    handles: Query<&Parent, With<ShovelHandle>>,
    mut tools_and_shovel: ParamSet<(
        ToolsHeld,
        Query<(Entity, &mut Shovel)>,
    )>,
) {
    let Ok((camera, cam_gt)) = cameras.get_single() else { return; };
    let Some((shovel_e, shovel_equipped)) = tools_and_shovel
        .p1().get_single().map(|(e, s)| (e, s.equipped)).ok()
    else { return; };

    if shovel_equipped {
        ui.hover = false;
        ui.swinging = buttons.pressed(MouseButton::Left);
        if buttons.pressed(MouseButton::Left) { consumed.0 = true; }
        if keys.just_pressed(KeyCode::KeyQ) {
            commands.entity(shovel_e).insert((
                RigidBody::Dynamic,
                Velocity::linear(*cam_gt.forward() * 2.0 + Vec3::Y * 0.5),
                Sleeping::disabled(),
            ));
            if let Ok((_, mut shovel)) = tools_and_shovel.p1().get_single_mut() {
                shovel.equipped = false;
            }
            ui.swinging = false;
            status.0 = "Dropped the shovel.".into();
        }
        return;
    }

    ui.hover = false;
    ui.swinging = false;
    if tools_and_shovel.p0().any() { return; }

    let Ok(window) = windows.get_single() else { return; };
    let Some(ray) = pointer_position(&window)
        .and_then(|c| camera.viewport_to_world(cam_gt, c))
    else { return; };
    let mut filter = QueryFilter::default();
    if let Ok(pe) = player.get_single() {
        filter = filter.exclude_collider(pe);
    }
    ui.hover = rapier
        .cast_ray(ray.origin, *ray.direction, 3.0, true, filter)
        .map_or(false, |(e, _)| {
            handles.get(e).map_or(false, |p| p.get() == shovel_e)
        });

    if ui.hover && buttons.just_pressed(MouseButton::Left) && !consumed.0 {
        commands
            .entity(shovel_e)
            .insert(RigidBody::KinematicPositionBased)
            .remove::<Velocity>();
        if let Ok((_, mut shovel)) = tools_and_shovel.p1().get_single_mut() {
            shovel.equipped = true;
            shovel.swing = 0.0;
            shovel.dig_cooldown = 0.0;
        }
        consumed.0 = true;
        status.0 =
            "Shovel equipped: hold LMB, aim at the ground, and swing the mouse to dig. Q drops it."
                .into();
    }
}

/// Skewers a fish on the spear: the fish becomes a child of the spear, with the shaft
/// running through its body, so it travels with the spear wherever you carry it.
fn attach_fish_to_spear(
    commands: &mut Commands,
    spear_entity: Entity,
    fish_entity: Entity,
    tf: &mut Transform,
) {
    tf.translation = Vec3::new(0.0, SPEAR_FISH_Y, 0.0);
    tf.rotation = Quat::from_rotation_x(0.12);
    tf.scale = Vec3::ONE;
    commands.entity(spear_entity).add_child(fish_entity);
}

/// Browns a fish's materials according to how far it has cooked.
fn paint_fish(f: &Fish, materials: &mut Assets<StandardMaterial>) {
    let t = f.cooked.clamp(0.0, 1.0);
    let mix = |a: [f32; 3], b: [f32; 3]| {
        Color::srgb(
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        )
    };
    if let Some(m) = materials.get_mut(&f.body_mat) {
        m.base_color = mix(f.raw_body, COOKED_BODY);
    }
    if let Some(m) = materials.get_mut(&f.fin_mat) {
        m.base_color = mix(f.raw_fin, COOKED_FIN);
    }
}

/// With a fish on the spear, click a burning fire to hold the fish over the flames. It
/// browns while it cooks and is done after a few seconds.
fn spear_cook_system(
    time: Res<Time>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut consumed: ResMut<ClickConsumed>,
    mut status: ResMut<Status>,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    fires: Query<(&GlobalTransform, &Fire)>,
    mut spears: Query<&mut Spear>,
    mut fish: Query<&mut Fish>,
) {
    let Ok(mut spear) = spears.get_single_mut() else {
        return;
    };
    let fish_entity = match (spear.equipped, spear.impaled_fish) {
        (true, Some(e)) => e,
        _ => {
            spear.cook_left = 0.0;
            return;
        }
    };
    let Ok(mut f) = fish.get_mut(fish_entity) else {
        return;
    };
    let Ok((camera, cam_gt)) = cameras.get_single() else {
        return;
    };
    let cam_pos = cam_gt.translation();

    // --- Roasting in progress ---
    if spear.cook_left > 0.0 {
        if buttons.just_pressed(MouseButton::Left) {
            consumed.0 = true; // no thrusting while the fish is over the fire
        }
        let still_burning = fires.iter().any(|(gt, fire)| {
            fire.fuel > 0.0
                && fire.size > 0.1
                && gt.translation().distance(spear.cook_point) < 1.2
        });
        if !still_burning || cam_pos.distance(spear.cook_point) > FIRE_COOK_REACH + 0.8 {
            spear.cook_left = 0.0;
            status.0 = "You pull the fish away from the fire.".into();
            return;
        }
        let dt = time.delta_seconds();
        spear.cook_left = (spear.cook_left - dt).max(0.0);
        f.cooked = (f.cooked + dt / FISH_COOK_SECONDS).min(1.0);
        paint_fish(&f, &mut materials);
        if f.cooked >= 1.0 {
            spear.cook_left = 0.0;
            status.0 = "The fish is cooked!".into();
        } else {
            status.0 = "Roasting the fish over the fire...".into();
        }
        return;
    }

    // --- Clicking a fire starts roasting ---
    if !buttons.just_pressed(MouseButton::Left)
        || buttons.pressed(MouseButton::Right)
        || consumed.0
    {
        return;
    }
    let Ok(window) = windows.get_single() else {
        return;
    };
    let pos = pointer_position(&window)
        .unwrap_or(Vec2::new(window.width() * 0.5, window.height() * 0.5));
    let Some(ray) = camera.viewport_to_world(cam_gt, pos) else {
        return;
    };
    let dir: Vec3 = *ray.direction;
    let mut best: Option<(f32, Vec3, f32)> = None; // (distance along ray, fire pos, fuel)
    for (gt, fire) in &fires {
        let fire_pos = gt.translation();
        let center = fire_pos + Vec3::Y * (0.25 + 0.15 * fire.size.min(1.5));
        let to = center - ray.origin;
        let t = to.dot(dir);
        if t < 0.0 {
            continue;
        }
        let miss = (to - dir * t).length();
        if miss < 0.5 + 0.2 * fire.size && best.map_or(true, |(bt, _, _)| t < bt) {
            best = Some((t, fire_pos, if fire.size > 0.1 { fire.fuel } else { 0.0 }));
        }
    }
    let Some((_, fire_pos, fuel)) = best else {
        return; // not aimed at a fire: the click is a normal thrust
    };
    consumed.0 = true;
    if fuel <= 0.0 {
        status.0 = "That fire has burnt out.".into();
        return;
    }
    if cam_pos.distance(fire_pos) > FIRE_COOK_REACH {
        status.0 = "Get closer to the fire to hold the fish over it.".into();
        return;
    }
    if f.cooked >= 1.0 {
        status.0 = "The fish is already cooked.".into();
        return;
    }
    spear.cook_point = fire_pos + Vec3::Y * 0.42;
    spear.cook_left = (1.0 - f.cooked) * FISH_COOK_SECONDS;
    status.0 = "You hold the fish over the fire...".into();
}

fn spear_input_system(
    time: Res<Time>,
    mut commands: Commands,
    player: Query<Entity, With<Player>>,
    mut consumed: ResMut<ClickConsumed>,
    mut ui: ResMut<SpearUi>,
    mut status: ResMut<Status>,
    mut fish_click: ResMut<FishClickState>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    rapier: Res<RapierContext>,
    handles: Query<&Parent, With<SpearHandle>>,
    mut tools_and_spear: ParamSet<(
        ToolsHeld,
        Query<(Entity, &mut Spear, &mut Transform)>,
    )>,
    mut targets: Query<
        (Option<&mut Velocity>, Option<&Parent>),
        Or<(With<Velocity>, With<Parent>)>,
    >,
    mut fish: Query<(Entity, &mut Fish, &mut Transform, &GlobalTransform), Without<Spear>>,
) {
    let Ok((camera, camera_transform)) = cameras.get_single() else {
        return;
    };
    let Some((spear_entity, spear_equipped)) = tools_and_spear
        .p1()
        .get_single()
        .map(|(e, s, _)| (e, s.equipped))
        .ok()
    else {
        return;
    };

    // Update cooldown (needs a momentary mutable borrow).
    if let Ok((_, mut spear, _)) = tools_and_spear.p1().get_single_mut() {
        spear.cooldown = (spear.cooldown - time.delta_seconds()).max(0.0);
    }

    if spear_equipped {
        let mut spear_query = tools_and_spear.p1(); // lives until end of block
        let Ok((_, mut spear, _)) = spear_query.get_single_mut() else {
            return;
        };

        ui.hover = false;
        ui.attacking = spear.attack_left > 0.0;

        // After the double-click detaches the fish, keep it attached to the cursor
        // while the second click is held. This is a visual drag, not a physics grab,
        // because the fish itself has no rigid body/collider.
        if let Some((held_entity, distance)) = fish_click.holding {
            if buttons.pressed(MouseButton::Left) {
                if let Some(ray) = windows.get_single().ok().and_then(|window| {
                    pointer_position(window).and_then(|pos| camera.viewport_to_world(camera_transform, pos))
                }) {
                    if let Ok((_, mut held_fish, mut held_tf, _)) = fish.get_mut(held_entity) {
                        held_fish.impaled = false;
                        held_tf.translation = ray.origin + *ray.direction * distance;
                        consumed.0 = true;
                        return;
                    }
                }
                // Keep the click consumed even if the cursor ray is temporarily unavailable.
                consumed.0 = true;
                return;
            } else {
                // Release the dragged fish as a real physics object so gravity drops it
                // onto the ground and it can collide with the world.
                commands.entity(held_entity).insert((
                    RigidBody::Dynamic,
                    Velocity::zero(),
                    Collider::ball(0.16),
                    Sleeping::disabled(),
                ));
                fish_click.holding = None;
            }
        }

        // Double-click the skewered fish to pull it free from the spear.
        // The fish has no collider, so test the cursor ray against its visible position.
        if buttons.just_pressed(MouseButton::Left) && spear.impaled_fish.is_some() && !consumed.0 {
            if let (Some(fish_entity), Ok(window)) = (
                spear.impaled_fish,
                windows.get_single(),
            ) {
                if let Some(ray) = pointer_position(window)
                    .and_then(|pos| camera.viewport_to_world(camera_transform, pos))
                {
                    if let Ok((_, mut f, mut fish_tf, fish_gt)) = fish.get_mut(fish_entity) {
                        let world_pos = fish_gt.translation();
                        let to_fish = world_pos - ray.origin;
                        let along = to_fish.dot(*ray.direction);
                        let miss = (to_fish - *ray.direction * along).length();
                        if along >= 0.0 && along <= 5.0 && miss < 0.42 {
                            let now = time.elapsed_seconds();
                            match fish_click.last_click {
                                Some((last_entity, last_time))
                                    if last_entity == fish_entity && now - last_time < 0.45 =>
                                {
                                    // Preserve the fish's full world pose when removing it from
                                    // the spear hierarchy, then make it kinematic while dragged.
                                    let world_transform = fish_gt.compute_transform();
                                    commands.entity(fish_entity).remove_parent().insert((
                                        RigidBody::KinematicPositionBased,
                                        Collider::ball(0.16),
                                        Velocity::zero(),
                                    ));
                                    fish_tf.translation = world_pos;
                                    fish_tf.rotation = world_transform.rotation;
                                    fish_tf.scale = world_transform.scale;
                                    f.impaled = false;
                                    f.dropped = true;
                                    spear.impaled_fish = None;
                                    spear.cook_left = 0.0;
                                    fish_click.last_click = None;
                                    fish_click.holding = Some((fish_entity, along.clamp(0.8, 5.0)));
                                    consumed.0 = true;
                                    status.0 = "You pull the fish off the spear. Drag it with the mouse.".into();
                                    return;
                                }
                                _ => fish_click.last_click = Some((fish_entity, now)),
                            }
                            consumed.0 = true;
                            return;
                        }
                    }
                }
            }
        }

        let strike = buttons.just_pressed(MouseButton::Left)
            && !buttons.pressed(MouseButton::Right)
            && spear.cooldown <= 0.0
            && !consumed.0;
        if buttons.pressed(MouseButton::Left) {
            consumed.0 = true;
        }
        if keys.just_pressed(KeyCode::KeyQ) {
            commands.entity(spear_entity).insert((
                RigidBody::Dynamic,
                Velocity::linear(*camera_transform.forward() * 2.0 + Vec3::Y * 0.5),
                Sleeping::disabled(),
            ));
            spear.equipped = false;
            spear.attack_left = 0.0;
            ui.attacking = false;
            status.0 = "Dropped the spear.".into();
            return;
        }

        if strike {
            // Begin the thrust: spear_pose_system slides the spear forward over
            // `attack_duration` seconds, then back, using these fields.
            spear.attack_left = spear.attack_duration;
            spear.push_distance = 0.55;
            spear.cooldown = 0.55;

            // Where the player clicked: a fish under the cursor (within reach) is the target
            // and the spear thrusts straight at it. Otherwise the thrust goes where you look.
            let cursor_ray = windows.get_single().ok().and_then(|w| {
                let pos = pointer_position(&w)
                    .unwrap_or(Vec2::new(w.width() * 0.5, w.height() * 0.5));
                camera.viewport_to_world(camera_transform, pos)
            });
            let mut aimed_fish: Option<(Entity, Vec3)> = None;
            if spear.impaled_fish.is_none() {
                if let Some(ray) = cursor_ray {
                    let dir: Vec3 = *ray.direction;
                    let mut best = f32::MAX;
                    for (fish_entity, f, f_tf, _) in fish.iter() {
                        if f.impaled {
                            continue;
                        }
                        let to = f_tf.translation - ray.origin;
                        let t = to.dot(dir);
                        if t < 0.0 || to.length() > SPEAR_FISH_REACH {
                            continue;
                        }
                        let miss = (to - dir * t).length();
                        if miss < 0.4 && t < best {
                            best = t;
                            aimed_fish = Some((fish_entity, dir));
                        }
                    }
                }
            }
            spear.attack_direction = match aimed_fish {
                Some((_, dir)) => dir,
                None => *camera_transform.forward(),
            };

            let origin = camera_transform.translation();
            let direction = spear.attack_direction;

            // 0) A fish under the cursor is stabbed and sticks to the spear.
            if let Some((fish_entity, _)) = aimed_fish {
                if let Ok((_, mut f, mut f_tf, _)) = fish.get_mut(fish_entity) {
                    f.impaled = true;
                    attach_fish_to_spear(&mut commands, spear_entity, fish_entity, &mut f_tf);
                    spear.impaled_fish = Some(fish_entity);
                    status.0 = "You spear a fish! Carry it to a fire and click the fire to cook it.".into();
                    return;
                }
            }
            // Approximate tip of the blade when the thrust is fully extended.
            let tip = origin + direction * 1.4;

            // 1) Fish near the blade get impaled (only one at a time).
            if spear.impaled_fish.is_none() {
                let mut impaled: Option<Entity> = None;
                for (fish_entity, mut f, mut f_tf, _) in &mut fish {
                    if f.impaled {
                        continue;
                    }
                    let to_fish = f_tf.translation - tip;
                    // The thrust points along `direction`; a fish within ~0.55 m of
                    // the tip and in front of it counts as a hit.
                    if to_fish.length() < 0.55 && to_fish.dot(direction) > -0.25 {
                        f.impaled = true;
                        attach_fish_to_spear(&mut commands, spear_entity, fish_entity, &mut f_tf);
                        impaled = Some(fish_entity);
                        break;
                    }
                }
                if let Some(fish_entity) = impaled {
                    spear.impaled_fish = Some(fish_entity);
                    status.0 = "You spear a fish! Carry it to a fire and click the fire to cook it.".into();
                    return;
                }
            }

            // 2) Whatever else the thrust ray reaches gets shoved.
            let filter =
                world_ray_filter(player.get_single().ok()).exclude_rigid_body(spear_entity);
            if let Some((hit, _toi)) = rapier.cast_ray(origin, direction, 2.5, true, filter) {
                // Walk up the hierarchy to the body that owns a Velocity.
                let mut walker = hit;
                loop {
                    let Ok((vel_opt, parent_opt)) = targets.get_mut(walker) else {
                        break;
                    };
                    if let Some(mut vel) = vel_opt {
                        vel.linvel += direction * 5.5 + Vec3::Y * 0.4;
                        break;
                    }
                    if let Some(parent) = parent_opt {
                        walker = parent.get();
                    } else {
                        break;
                    }
                }
                status.0 = "Thrust!".into();
            } else {
                status.0 = "You thrust the spear through empty air.".into();
            }
        }
        return;
    }

    ui.hover = false;
    ui.attacking = false;
    if tools_and_spear.p0().any() {
        return;
    }

    // Hover detection: cursor is over the spear's grip when the nearest collider
    // under it belongs to the spear's handle.
    let Ok(window) = windows.get_single() else {
        return;
    };
    let Some(ray) = pointer_position(&window)
        .and_then(|c| camera.viewport_to_world(camera_transform, c))
    else {
        return;
    };
    let mut filter = QueryFilter::default();
    if let Ok(pe) = player.get_single() {
        filter = filter.exclude_collider(pe);
    }
    ui.hover = rapier
        .cast_ray(ray.origin, *ray.direction, 3.0, true, filter)
        .map_or(false, |(e, _)| {
            handles.get(e).map_or(false, |p| p.get() == spear_entity)
        });

    if ui.hover && buttons.just_pressed(MouseButton::Left) && !consumed.0 {
        let had_fish = tools_and_spear
            .p1()
            .get_single()
            .map(|(_, s, _)| s.impaled_fish.is_some())
            .unwrap_or(false);
        commands
            .entity(spear_entity)
            .insert(RigidBody::KinematicPositionBased)
            .remove::<Velocity>();
        if let Ok((_, mut spear, _)) = tools_and_spear.p1().get_single_mut() {
            spear.equipped = true;
            spear.attack_left = 0.0;
            spear.cooldown = 0.0;
        }
        consumed.0 = true;
        status.0 = if had_fish {
            "You pull the spear and fish free from the riverbed.".into()
        } else {
            "Spear equipped: aim at a target and click to thrust. Q drops the spear.".into()
        };
    }
}

/// Pick the machete up by its grip and drop it with Q.
fn machete_input_system(
    mut commands: Commands,
    player: Query<Entity, With<Player>>,
    mut consumed: ResMut<ClickConsumed>,
    mut ui: ResMut<MacheteUi>,
    mut status: ResMut<Status>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    rapier: Res<RapierContext>,
    handles: Query<&Parent, With<MacheteHandle>>,
    mut tools_and_machete: ParamSet<(
        ToolsHeld,
        Query<(Entity, &mut Machete)>,
    )>,
) {
    let Ok((camera, cam_gt)) = cameras.get_single() else { return; };
    let Some((machete_e, machete_equipped)) = tools_and_machete
        .p1().get_single().map(|(e, m)| (e, m.equipped)).ok()
    else { return; };

    if machete_equipped {
        ui.hover = false;
        ui.swinging = buttons.pressed(MouseButton::Left);
        if buttons.pressed(MouseButton::Left) { consumed.0 = true; }
        if keys.just_pressed(KeyCode::KeyQ) {
            commands.entity(machete_e).insert((
                RigidBody::Dynamic,
                Velocity::linear(*cam_gt.forward() * 2.0 + Vec3::Y * 0.5),
                Sleeping::disabled(),
            ));
            if let Ok((_, mut machete)) = tools_and_machete.p1().get_single_mut() {
                machete.equipped = false;
            }
            ui.swinging = false;
            status.0 = "Dropped the machete.".into();
        }
        return;
    }

    ui.hover = false;
    ui.swinging = false;
    if tools_and_machete.p0().any() { return; }

    let Ok(window) = windows.get_single() else { return; };
    let Some(ray) = pointer_position(&window)
        .and_then(|c| camera.viewport_to_world(cam_gt, c))
    else { return; };
    let mut filter = QueryFilter::default();
    if let Ok(pe) = player.get_single() {
        filter = filter.exclude_collider(pe);
    }
    ui.hover = rapier
        .cast_ray(ray.origin, *ray.direction, 3.0, true, filter)
        .map_or(false, |(e, _)| {
            handles.get(e).map_or(false, |p| p.get() == machete_e)
        });

    if ui.hover && buttons.just_pressed(MouseButton::Left) && !consumed.0 {
        commands
            .entity(machete_e)
            .insert(RigidBody::KinematicPositionBased)
            .remove::<Velocity>();
        if let Ok((_, mut machete)) = tools_and_machete.p1().get_single_mut() {
            machete.equipped = true;
            machete.psi = SWORD_REST_PSI;
            machete.phi = SWORD_REST_PHI;
            machete.psi_vel = 0.0;
            machete.roll = 0.0;
            machete.slash_cooldown = 0.0;
        }
        consumed.0 = true;
        status.0 =
            "Machete equipped: hold LMB and move the mouse to slash at grass, plants and sticks. Q drops it."
                .into();
    }
}

/// Pick the knife up by its grip and drop it with Q.
fn knife_input_system(
    mut commands: Commands,
    player: Query<Entity, With<Player>>,
    mut consumed: ResMut<ClickConsumed>,
    mut ui: ResMut<KnifeUi>,
    mut status: ResMut<Status>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    rapier: Res<RapierContext>,
    handles: Query<&Parent, With<KnifeHandle>>,
    mut tools_and_knife: ParamSet<(
        ToolsHeld,
        Query<(Entity, &mut Knife)>,
    )>,
) {
    let Ok((camera, cam_gt)) = cameras.get_single() else { return; };
    let Some((knife_e, knife_equipped)) = tools_and_knife
        .p1()
        .get_single()
        .map(|(e, k)| (e, k.equipped))
        .ok()
    else {
        return;
    };

    if knife_equipped {
        ui.hover = false;
        ui.swinging = buttons.pressed(MouseButton::Left);
        if buttons.pressed(MouseButton::Left) {
            consumed.0 = true;
        }
        if keys.just_pressed(KeyCode::KeyQ) {
            commands.entity(knife_e).insert((
                RigidBody::Dynamic,
                Velocity::linear(*cam_gt.forward() * 2.0 + Vec3::Y * 0.5),
                Sleeping::disabled(),
            ));
            if let Ok((_, mut knife)) = tools_and_knife.p1().get_single_mut() {
                knife.equipped = false;
            }
            ui.swinging = false;
            status.0 = "Dropped the knife.".into();
        }
        return;
    }

    ui.hover = false;
    ui.swinging = false;
    if tools_and_knife.p0().any() {
        return;
    }
    let Ok(window) = windows.get_single() else {
        return;
    };
    let Some(ray) = pointer_position(&window)
        .and_then(|c| camera.viewport_to_world(cam_gt, c))
    else {
        return;
    };
    let mut filter = QueryFilter::default();
    if let Ok(pe) = player.get_single() {
        filter = filter.exclude_collider(pe);
    }
    ui.hover = rapier
        .cast_ray(ray.origin, *ray.direction, 3.0, true, filter)
        .map_or(false, |(e, _)| {
            handles.get(e).map_or(false, |p| p.get() == knife_e)
        });

    if ui.hover && buttons.just_pressed(MouseButton::Left) && !consumed.0 {
        commands
            .entity(knife_e)
            .insert(RigidBody::KinematicPositionBased)
            .remove::<Velocity>();
        if let Ok((_, mut knife)) = tools_and_knife.p1().get_single_mut() {
            knife.equipped = true;
            knife.psi = SWORD_REST_PSI;
            knife.phi = SWORD_REST_PHI;
            knife.psi_vel = 0.0;
            knife.roll = 0.0;
            knife.slash_cooldown = 0.0;
        }
        consumed.0 = true;
        status.0 =
            "Knife equipped: flick the mouse to slash grass, plants and sticks; click a wood chip to cut a V notch in it. Q drops it."
                .into();
    }
}



// ---------------------------------------------------------------------------
// Paint brush and paint cans
// ---------------------------------------------------------------------------

const BRUSH_REACH: f32 = 5.0; // m, how far the brush can paint
const BRUSH_DIP_REACH: f32 = 4.0; // m, how far away a can can be dipped into
const MAX_PAINT_DABS: usize = 500; // the oldest paint is removed beyond this

/// What the brush is pointing at.
#[derive(Clone, Copy)]
struct BrushHit {
    entity: Entity,
    point: Vec3,
    normal: Vec3,
    distance: f32,
    is_can: bool,
}

/// A paint brush held in front of the camera. Click a paint can to load it with that colour,
/// then click or drag on a surface to paint it. Q drops it.
#[derive(Component)]
struct Brush {
    equipped: bool,
    /// world point the bristles point at
    aim: Vec3,
    lunge: f32,
    /// 0..1, how much paint is left on the bristles
    paint: f32,
    paint_mat: Option<Handle<StandardMaterial>>,
    tip_mat: Handle<StandardMaterial>,
    hit: Option<BrushHit>,
    last_dab: Option<Vec3>,
}

/// Grab this part of the brush (its handle) to pick it up.
#[derive(Component)]
struct BrushHandle;

/// A can of paint. Click it with the brush to load the brush with its colour.
#[derive(Component)]
struct PaintCan {
    name: &'static str,
    color: Color,
    dab_mat: Handle<StandardMaterial>,
}

/// One blob of paint on a surface.
#[derive(Component)]
struct PaintDab;

#[derive(Resource, Default)]
struct BrushUi {
    hover: bool,
    painting: bool,
}

#[derive(Resource, Default)]
struct PaintDabs {
    queue: std::collections::VecDeque<Entity>,
    count: u32,
}

/// A brush lying at `pos` (a normal physics object until you pick it up).
fn spawn_brush(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &mut Assets<StandardMaterial>,
    pos: Vec3,
) {
    let wood = mats.add(Color::srgb(0.45, 0.28, 0.12));
    let metal = mats.add(StandardMaterial {
        base_color: Color::srgb(0.7, 0.7, 0.72),
        metallic: 0.9,
        perceptual_roughness: 0.3,
        ..default()
    });
    let tip_mat = mats.add(StandardMaterial {
        base_color: Color::srgb(0.92, 0.86, 0.7),
        perceptual_roughness: 0.9,
        ..default()
    });
    let groups = CollisionGroups::new(Group::GROUP_2, Group::ALL);
    let lying = Quat::from_mat3(&Mat3::from_cols(Vec3::Y, Vec3::X, Vec3::NEG_Z));

    commands
        .spawn((
            SpatialBundle::from_transform(Transform::from_translation(pos).with_rotation(lying)),
            RigidBody::Dynamic,
            Velocity::default(),
            Restitution::coefficient(0.1),
            Friction::coefficient(0.8),
            Sleeping::disabled(),
            Brush {
                equipped: false,
                aim: pos,
                lunge: 0.0,
                paint: 0.0,
                paint_mat: None,
                tip_mat: tip_mat.clone(),
                hit: None,
                last_dab: None,
            },
        ))
        .with_children(|p| {
            // wooden handle (the origin of the brush is the middle of the handle)
            p.spawn((
                PbrBundle {
                    mesh: meshes.add(Cylinder::new(0.012, 0.17)),
                    material: wood,
                    ..default()
                },
                Collider::cuboid(0.024, 0.12, 0.024),
                groups,
                BrushHandle,
            ));
            // metal ferrule
            p.spawn(PbrBundle {
                mesh: meshes.add(Cylinder::new(0.017, 0.05)),
                material: metal,
                transform: Transform::from_xyz(0.0, 0.11, 0.0),
                ..default()
            });
            // bristles (they take the colour of the paint on the brush)
            p.spawn(PbrBundle {
                mesh: meshes.add(Cuboid::new(0.046, 0.08, 0.016)),
                material: tip_mat,
                transform: Transform::from_xyz(0.0, 0.175, 0.0),
                ..default()
            });
        });
}

/// A row of paint cans in different colours, starting at `first`.
fn spawn_paint_cans(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &mut Assets<StandardMaterial>,
    first: Vec3,
) {
    let colors: [(&'static str, [f32; 3]); 8] = [
        ("red", [0.82, 0.08, 0.08]),
        ("blue", [0.1, 0.25, 0.85]),
        ("yellow", [0.95, 0.8, 0.1]),
        ("green", [0.12, 0.62, 0.2]),
        ("orange", [0.95, 0.45, 0.08]),
        ("purple", [0.5, 0.15, 0.7]),
        ("white", [0.95, 0.95, 0.92]),
        ("black", [0.05, 0.05, 0.06]),
    ];
    let body_mesh = meshes.add(Cylinder::new(0.075, 0.18));
    let band_mesh = meshes.add(Cylinder::new(0.0765, 0.07));
    let top_mesh = meshes.add(Cylinder::new(0.066, 0.004));
    let metal = mats.add(StandardMaterial {
        base_color: Color::srgb(0.62, 0.62, 0.65),
        metallic: 0.8,
        perceptual_roughness: 0.4,
        ..default()
    });
    for (i, (name, c)) in colors.into_iter().enumerate() {
        let color = Color::srgb(c[0], c[1], c[2]);
        let wet = mats.add(StandardMaterial {
            base_color: color,
            perceptual_roughness: 0.3,
            ..default()
        });
        let dab_mat = mats.add(StandardMaterial {
            base_color: color,
            perceptual_roughness: 0.6,
            double_sided: true,
            cull_mode: None,
            ..default()
        });
        commands
            .spawn((
                SpatialBundle::from_transform(Transform::from_translation(
                    first + Vec3::X * (0.4 * i as f32),
                )),
                RigidBody::Dynamic,
                Collider::cylinder(0.09, 0.077),
                ColliderMassProperties::Mass(1.0),
                Velocity::default(),
                Restitution::coefficient(0.1),
                Friction::coefficient(0.9),
                Sleeping::disabled(),
                PaintCan {
                    name,
                    color,
                    dab_mat,
                },
            ))
            .with_children(|can| {
                can.spawn(PbrBundle {
                    mesh: body_mesh.clone(),
                    material: metal.clone(),
                    ..default()
                });
                // coloured label band
                can.spawn(PbrBundle {
                    mesh: band_mesh.clone(),
                    material: wet.clone(),
                    transform: Transform::from_xyz(0.0, -0.02, 0.0),
                    ..default()
                });
                // the open top, full of paint
                can.spawn(PbrBundle {
                    mesh: top_mesh.clone(),
                    material: wet,
                    transform: Transform::from_xyz(0.0, 0.091, 0.0),
                    ..default()
                });
            });
    }
}

/// Pick the brush up by its handle (within 3 m), drop it with Q. While it is in hand:
/// clicking a paint can loads the brush with that colour, and the brush follows the cursor.
fn brush_input_system(
    mut commands: Commands,
    player: Query<Entity, With<Player>>,
    mut consumed: ResMut<ClickConsumed>,
    mut ui: ResMut<BrushUi>,
    mut status: ResMut<Status>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    rapier: Res<RapierContext>,
    handles: Query<&Parent, With<BrushHandle>>,
    cans: Query<&PaintCan>,
    parents: Query<&Parent>,
    mut mats: ResMut<Assets<StandardMaterial>>,
    mut tools_and_brush: ParamSet<(ToolsHeld, Query<(Entity, &mut Brush)>)>,
) {
    let Ok((camera, cam_gt)) = cameras.get_single() else {
        return;
    };
    let Some((brush_e, equipped)) = tools_and_brush
        .p1()
        .get_single()
        .map(|(e, b)| (e, b.equipped))
        .ok()
    else {
        return;
    };
    let ray = windows
        .get_single()
        .ok()
        .and_then(|w| pointer_position(&w))
        .and_then(|c| camera.viewport_to_world(cam_gt, c));

    if equipped {
        ui.hover = false;
        if buttons.pressed(MouseButton::Left) {
            consumed.0 = true; // every click goes to the brush while it is in hand
        }
        if keys.just_pressed(KeyCode::KeyQ) {
            commands.entity(brush_e).insert((
                RigidBody::Dynamic,
                Velocity::linear(*cam_gt.forward() * 2.0 + Vec3::Y * 0.5),
                Sleeping::disabled(),
            ));
            if let Ok((_, mut brush)) = tools_and_brush.p1().get_single_mut() {
                brush.equipped = false;
                brush.hit = None;
                brush.last_dab = None;
            }
            ui.painting = false;
            status.0 = "Dropped the brush.".into();
            return;
        }

        // What is the brush pointing at? (Held tools are ignored by this ray.)
        let mut hit: Option<BrushHit> = None;
        let mut can_entity: Option<Entity> = None;
        if let Some(ray) = ray {
            let filter = world_ray_filter(player.get_single().ok());
            if let Some((e, info)) =
                rapier.cast_ray_and_get_normal(ray.origin, *ray.direction, BRUSH_REACH, true, filter)
            {
                let mut normal = info.normal.normalize_or_zero();
                if normal == Vec3::ZERO || normal.dot(*ray.direction) > 0.0 {
                    normal = -normal;
                }
                // is it (part of) a paint can?
                let mut walker = e;
                loop {
                    if cans.get(walker).is_ok() {
                        can_entity = Some(walker);
                        break;
                    }
                    match parents.get(walker) {
                        Ok(p) => walker = p.get(),
                        Err(_) => break,
                    }
                }
                hit = Some(BrushHit {
                    entity: e,
                    point: ray.origin + *ray.direction * info.time_of_impact,
                    normal,
                    distance: info.time_of_impact,
                    is_can: can_entity.is_some(),
                });
            }
        }
        let aim = hit
            .map(|h| h.point)
            .or_else(|| ray.map(|r| r.origin + *r.direction * 3.0))
            .unwrap_or(cam_gt.translation() + *cam_gt.forward() * 3.0);
        ui.hover = hit.map_or(false, |h| h.is_can);

        // Clicking a paint can dips the brush in it.
        if let (Some(can_e), Some(h)) = (can_entity, hit) {
            if buttons.just_pressed(MouseButton::Left) {
                if h.distance > BRUSH_DIP_REACH {
                    status.0 = "Move closer to dip the brush in that can.".into();
                } else if let Ok(can) = cans.get(can_e) {
                    if let Ok((_, mut brush)) = tools_and_brush.p1().get_single_mut() {
                        brush.paint = 1.0;
                        brush.paint_mat = Some(can.dab_mat.clone());
                        brush.last_dab = None;
                        if let Some(m) = mats.get_mut(&brush.tip_mat) {
                            m.base_color = can.color;
                        }
                    }
                    status.0 = format!(
                        "Dipped the brush in the {} paint. Click or drag on something to paint it.",
                        can.name
                    );
                }
            }
        }
        if let Ok((_, mut brush)) = tools_and_brush.p1().get_single_mut() {
            brush.hit = hit;
            brush.aim = aim;
        }
        return;
    }

    ui.hover = false;
    ui.painting = false;
    if tools_and_brush.p0().any() {
        return;
    }
    let Some(ray) = ray else {
        return;
    };
    let mut filter = QueryFilter::default();
    if let Ok(pe) = player.get_single() {
        filter = filter.exclude_collider(pe);
    }
    ui.hover = rapier
        .cast_ray(ray.origin, *ray.direction, 3.0, true, filter)
        .map_or(false, |(e, _)| {
            handles.get(e).map_or(false, |p| p.get() == brush_e)
        });

    if ui.hover && buttons.just_pressed(MouseButton::Left) && !consumed.0 {
        commands
            .entity(brush_e)
            .insert(RigidBody::KinematicPositionBased)
            .remove::<Velocity>();
        if let Ok((_, mut brush)) = tools_and_brush.p1().get_single_mut() {
            brush.equipped = true;
            brush.lunge = 0.0;
            brush.hit = None;
            brush.last_dab = None;
        }
        consumed.0 = true;
        status.0 =
            "Brush equipped: click a paint can to dip it, then click or drag on something to paint it. Q drops it."
                .into();
    }
}

/// Holds the brush in front of the camera, bristles pointing at what the cursor aims at;
/// it pushes forward a little while painting.
fn brush_pose_system(
    time: Res<Time>,
    buttons: Res<ButtonInput<MouseButton>>,
    cam: Query<&Transform, (With<Camera3d>, Without<Brush>)>,
    mut brushes: Query<(&mut Brush, &mut Transform)>,
) {
    let Ok(cam) = cam.get_single() else {
        return;
    };
    let Ok((mut brush, mut tf)) = brushes.get_single_mut() else {
        return;
    };
    if !brush.equipped {
        return;
    }
    let dt = time.delta_seconds().clamp(1e-3, 0.05);
    let r = cam.rotation;
    let (right, up, fwd) = (r * Vec3::X, r * Vec3::Y, r * Vec3::NEG_Z);
    let hand = cam.translation + r * Vec3::new(0.26, -0.22, -0.42);
    let mut dir = (brush.aim - hand).normalize_or_zero();
    if dir == Vec3::ZERO || dir.dot(fwd) < 0.3 {
        dir = (fwd + up * 0.2).normalize();
    }
    let painting = buttons.pressed(MouseButton::Left)
        && !buttons.pressed(MouseButton::Right)
        && brush.paint > 0.0
        && brush.hit.map_or(false, |h| !h.is_can);
    let target = if painting {
        0.10 + 0.02 * (time.elapsed_seconds() * 28.0).sin()
    } else {
        0.0
    };
    brush.lunge += (target - brush.lunge) * (1.0 - (-14.0 * dt).exp());

    let mut z = right - dir * right.dot(dir);
    if z.length_squared() < 1e-4 {
        z = up - dir * up.dot(dir);
    }
    let z = z.normalize();
    let x = dir.cross(z);
    tf.translation = hand + dir * brush.lunge;
    tf.rotation = Quat::from_mat3(&Mat3::from_cols(x, dir, z));
}

/// With paint on the brush, holding the left button over a surface lays down blobs of paint
/// as the cursor moves. A blob follows the thing it is painted on if that thing moves.
fn brush_paint_system(
    mut commands: Commands,
    mut ui: ResMut<BrushUi>,
    mut status: ResMut<Status>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut dab_mesh: Local<Option<Handle<Mesh>>>,
    mut dabs: ResMut<PaintDabs>,
    globals: Query<&GlobalTransform>,
    bodies: Query<&RigidBody>,
    parents: Query<&Parent>,
    mut brushes: Query<&mut Brush>,
) {
    ui.painting = false;
    let Ok(mut brush) = brushes.get_single_mut() else {
        return;
    };
    if !brush.equipped {
        return;
    }
    let held = buttons.pressed(MouseButton::Left) && !buttons.pressed(MouseButton::Right);
    if !held {
        brush.last_dab = None;
        return;
    }
    let Some(hit) = brush.hit else {
        brush.last_dab = None;
        return;
    };
    if hit.is_can {
        return; // clicking a can dips the brush; it does not paint the can
    }
    let Some(paint_mat) = brush.paint_mat.clone() else {
        if buttons.just_pressed(MouseButton::Left) {
            status.0 = "The brush is dry. Click a paint can to dip it.".into();
        }
        return;
    };
    if brush.paint <= 0.0 {
        if buttons.just_pressed(MouseButton::Left) {
            status.0 = "The brush is out of paint. Dip it in a can again.".into();
        }
        return;
    }
    ui.painting = true;

    let far_enough = brush
        .last_dab
        .map_or(true, |p| p.distance(hit.point) >= 0.035);
    if !far_enough {
        return;
    }
    brush.last_dab = Some(hit.point);
    brush.paint = (brush.paint - 0.018).max(0.0);
    if brush.paint <= 0.0 {
        status.0 = "The brush ran out of paint. Dip it in a can again.".into();
    }

    // The blob: a slightly irregular flat disc lying on the surface. Newer paint sits a hair
    // higher than older paint so overlapping strokes layer in the order they were painted.
    let n = dabs.count;
    dabs.count = n.wrapping_add(1);
    let offset = 0.0025 + (n % MAX_PAINT_DABS as u32) as f32 * 0.00004;
    let base = if buttons.just_pressed(MouseButton::Left) { 0.15 } else { 0.10 };
    let diameter = base * (0.85 + 0.35 * pseudo_rand(n.wrapping_mul(3).wrapping_add(1)));
    let stretch = 0.85 + 0.3 * pseudo_rand(n.wrapping_mul(5).wrapping_add(2));
    let spin = pseudo_rand(n.wrapping_mul(11).wrapping_add(3)) * std::f32::consts::TAU;
    let world = Transform {
        translation: hit.point + hit.normal * offset,
        rotation: Quat::from_rotation_arc(Vec3::Z, hit.normal) * Quat::from_rotation_z(spin),
        scale: Vec3::new(diameter * stretch, diameter, 1.0),
    };

    // If it was painted on something that can move, the paint moves with it.
    let mut body = None;
    let mut walker = hit.entity;
    loop {
        if matches!(bodies.get(walker), Ok(RigidBody::Dynamic)) {
            body = Some(walker);
            break;
        }
        match parents.get(walker) {
            Ok(p) => walker = p.get(),
            Err(_) => break,
        }
    }
    let parent_global = body.and_then(|b| globals.get(b).ok().map(|g| (b, *g)));
    let transform = match parent_global {
        Some((_, g)) => Transform::from_matrix(g.compute_matrix().inverse() * world.compute_matrix()),
        None => world,
    };

    let mesh = dab_mesh
        .get_or_insert_with(|| meshes.add(Circle::new(0.5)))
        .clone();
    let id = commands
        .spawn((
            PbrBundle {
                mesh,
                material: paint_mat,
                transform,
                ..default()
            },
            PaintDab,
        ))
        .id();
    if let Some((b, _)) = parent_global {
        commands.entity(id).set_parent(b);
    }

    dabs.queue.push_back(id);
    while dabs.queue.len() > MAX_PAINT_DABS {
        if let Some(old) = dabs.queue.pop_front() {
            if let Some(ec) = commands.get_entity(old) {
                ec.despawn_recursive();
            }
        }
    }
}

/// Mesh and collider data for a wood chip that has had a V notch cut into it.
struct NotchedChip {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    colors: Vec<[f32; 4]>,
    indices: Vec<u32>,
    /// The two solid halves either side of the notch, as point clouds whose convex hulls
    /// make the collider.
    halves: [Vec<Vec3>; 2],
}

/// Adds one flat, convex face. The winding is flipped if needed so it faces `hint`.
fn notch_face(out: &mut NotchedChip, pts: &[Vec3], hint: Vec3, color: [f32; 4]) {
    let mut ordered = pts.to_vec();
    if (ordered[1] - ordered[0])
        .cross(ordered[2] - ordered[0])
        .dot(hint)
        < 0.0
    {
        ordered.reverse();
    }
    let normal = hint.normalize_or_zero();
    let base = out.positions.len() as u32;
    for p in &ordered {
        out.positions.push(p.to_array());
        out.normals.push(normal.to_array());
        out.uvs.push([(p.x + p.z) * 0.5, p.y * 0.5]);
        out.colors.push(color);
    }
    for i in 1..ordered.len() as u32 - 1 {
        out.indices.extend_from_slice(&[base, base + i, base + i + 1]);
    }
}

/// Rebuilds a chip as a block with a V notch cut out of the middle of one face.
///
/// The block is the chip's own bounding box (mesh-local space, `positions`). The notch sits at
/// the centre of the chip's longest side, opens on the face that was clicked (`face_dir`, the
/// outward direction of that face in mesh-local space) and runs across the full width of the
/// chip, so material is really missing there. The outer faces keep the colours of the original
/// chip (bark stays bark); the faces of the cut get the colour of cut wood.
fn notch_chip(
    positions: &[[f32; 3]],
    normals: Option<&[[f32; 3]]>,
    colors: Option<&[[f32; 4]]>,
    face_dir: Vec3,
) -> Option<NotchedChip> {
    if positions.len() < 3 {
        return None;
    }
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    for p in positions {
        lo = lo.min(Vec3::from(*p));
        hi = hi.max(Vec3::from(*p));
    }
    let center = (lo + hi) * 0.5;
    let half = (hi - lo) * 0.5;
    if half.min_element() < 1e-5 {
        return None;
    }
    let h = [half.x, half.y, half.z];
    let hint = [face_dir.x, face_dir.y, face_dir.z];
    let unit = |i: usize| match i {
        0 => Vec3::X,
        1 => Vec3::Y,
        _ => Vec3::Z,
    };

    // l: the long axis (the notch goes in the middle of it)
    // n: the axis of the clicked face (the notch opens on that face and cuts into the chip)
    // t: the remaining axis (the notch runs straight across the chip along it)
    let l = (0..3).max_by(|&a, &b| h[a].total_cmp(&h[b])).unwrap_or(0);
    let rest: Vec<usize> = (0..3).filter(|&i| i != l).collect();
    let (n, t) = if hint[rest[0]].abs() >= hint[rest[1]].abs() {
        (rest[0], rest[1])
    } else {
        (rest[1], rest[0])
    };
    let sign = if hint[n] < 0.0 { -1.0 } else { 1.0 };
    let (eu, ev, ew) = (unit(l), unit(n) * sign, unit(t));
    let (hl, hn, hw) = (h[l], h[n], h[t]);

    // The V: `depth` deep, opening `a` either side of the middle.
    let depth = hn;
    let a = (depth * 0.6).min(hl * 0.45);

    // Cross-section in (u along the long side, v toward the notch face), extruded along w.
    let pt = |u: f32, v: f32, w: f32| center + eu * u + ev * v + ew * w;
    let uv = [
        (-hl, -hn), // 0 bottom left
        (hl, -hn),  // 1 bottom right
        (hl, hn),   // 2 top right
        (a, hn),    // 3 right lip of the V
        (0.0, hn - depth), // 4 bottom of the V
        (-a, hn),   // 5 left lip of the V
        (-hl, hn),  // 6 top left
    ];

    let color_toward = |dir: Vec3| -> [f32; 4] {
        let (Some(ns), Some(cs)) = (normals, colors) else {
            return [1.0; 4];
        };
        let mut best = (f32::MIN, [1.0; 4]);
        for (nv, c) in ns.iter().zip(cs) {
            let d = Vec3::from(*nv).dot(dir);
            if d > best.0 {
                best = (d, *c);
            }
        }
        best.1
    };
    // Cut wood: the first vertex colour that isn't plain bark.
    let cut_color = colors
        .and_then(|cs| {
            cs.iter()
                .copied()
                .find(|c| c[0] < 0.97 || c[1] < 0.97 || c[2] < 0.97)
        })
        .unwrap_or([1.0; 4]);

    let mut out = NotchedChip {
        positions: Vec::new(),
        normals: Vec::new(),
        uvs: Vec::new(),
        colors: Vec::new(),
        indices: Vec::new(),
        halves: [Vec::new(), Vec::new()],
    };

    // The faces along the outline of the cross-section.
    let wall_right = eu * -depth + ev * a; // faces into the V, toward the middle
    let wall_left = eu * depth + ev * a;
    let sides: [(usize, usize, Vec3, [f32; 4]); 7] = [
        (0, 1, -ev, color_toward(-ev)),
        (1, 2, eu, color_toward(eu)),
        (2, 3, ev, color_toward(ev)),
        (3, 4, wall_right, cut_color),
        (4, 5, wall_left, cut_color),
        (5, 6, ev, color_toward(ev)),
        (6, 0, -eu, color_toward(-eu)),
    ];
    for (i, j, dir, color) in sides {
        let (p, q) = (uv[i], uv[j]);
        notch_face(
            &mut out,
            &[
                pt(p.0, p.1, -hw),
                pt(q.0, q.1, -hw),
                pt(q.0, q.1, hw),
                pt(p.0, p.1, hw),
            ],
            dir,
            color,
        );
    }

    // The two flat ends (along w), cut into convex triangles.
    let triangles = [(0, 1, 4), (1, 2, 3), (1, 3, 4), (0, 4, 5), (0, 5, 6)];
    for (w, dir) in [(hw, ew), (-hw, -ew)] {
        let color = color_toward(dir);
        for (i, j, k) in triangles {
            notch_face(
                &mut out,
                &[
                    pt(uv[i].0, uv[i].1, w),
                    pt(uv[j].0, uv[j].1, w),
                    pt(uv[k].0, uv[k].1, w),
                ],
                dir,
                color,
            );
        }
    }

    // Collider: the solid on each side of the middle plane (each half is convex).
    let bottom_mid = (0.0, -hn);
    let left = [uv[0], bottom_mid, uv[4], uv[5], uv[6]];
    let right = [bottom_mid, uv[1], uv[2], uv[3], uv[4]];
    for (slot, half_outline) in [left, right].iter().enumerate() {
        for &(u, v) in half_outline {
            out.halves[slot].push(pt(u, v, -hw));
            out.halves[slot].push(pt(u, v, hw));
        }
    }
    Some(out)
}

/// Knife: with the knife equipped, click a wood chip (a piece knocked off a tree or log) to cut
/// a V notch into the middle of it, on the face you clicked. The chip's mesh is replaced by one
/// that really has the wood removed (nothing is stuck on top of it) and its collider is rebuilt
/// to match. Logs and standing trees are left alone.
fn knife_carve_system(
    mut commands: Commands,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    rapier: Res<RapierContext>,
    player: Query<Entity, With<Player>>,
    knives: Query<&Knife>,
    parents: Query<&Parent>,
    chips: Query<
        (&Handle<Mesh>, &GlobalTransform),
        (Or<(With<WoodDebris>, With<TreeChip>)>, Without<KnifeNotched>),
    >,
    already_cut: Query<(), With<KnifeNotched>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut status: ResMut<Status>,
) {
    let Ok(knife) = knives.get_single() else { return };
    if !knife.equipped
        || !buttons.just_pressed(MouseButton::Left)
        || buttons.pressed(MouseButton::Right)
    {
        return;
    }
    let (Ok(window), Ok((camera, cam_tf))) = (windows.get_single(), cameras.get_single()) else {
        return;
    };
    let Some(ray) = pointer_position(&window)
        .and_then(|c| camera.viewport_to_world(cam_tf, c))
    else {
        return;
    };

    // What is under the cursor? (The first thing hit, so a chip behind a wall can't be cut.)
    let filter = world_ray_filter(player.get_single().ok());
    let Some((hit, hit_info)) =
        rapier.cast_ray_and_get_normal(ray.origin, *ray.direction, 3.5, true, filter)
    else {
        return;
    };

    // Walk up from the collider to the chip itself.
    let mut walker = hit;
    let chip = loop {
        if chips.contains(walker) {
            break Some(walker);
        }
        if already_cut.contains(walker) {
            status.0 = "That piece already has a V notch.".into();
            return;
        }
        match parents.get(walker) {
            Ok(p) => walker = p.get(),
            Err(_) => break None,
        }
    };
    let Some(chip) = chip else { return };
    let Ok((mesh_handle, chip_tf)) = chips.get(chip) else { return };
    let Some(mesh) = meshes.get(mesh_handle) else { return };
    let Some(positions) = mesh
        .attribute(Mesh::ATTRIBUTE_POSITION)
        .and_then(|a| a.as_float3())
    else {
        return;
    };
    let normals = mesh
        .attribute(Mesh::ATTRIBUTE_NORMAL)
        .and_then(|a| a.as_float3());
    let colors = match mesh.attribute(Mesh::ATTRIBUTE_COLOR) {
        Some(bevy::render::mesh::VertexAttributeValues::Float32x4(values)) => {
            Some(values.as_slice())
        }
        _ => None,
    };

    // The face you clicked, in the chip's own space: the V opens on that face.
    let (_, rotation, _) = chip_tf.to_scale_rotation_translation();
    let face_dir = rotation.inverse() * hit_info.normal;

    let Some(cut) = notch_chip(positions, normals, colors, face_dir) else {
        return;
    };
    let shapes: Vec<(Vec3, Quat, Collider)> = cut
        .halves
        .iter()
        .filter_map(|points| Collider::convex_hull(points))
        .map(|hull| (Vec3::ZERO, Quat::IDENTITY, hull))
        .collect();
    if shapes.is_empty() {
        return;
    }

    let mut notched = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    notched.insert_attribute(Mesh::ATTRIBUTE_POSITION, cut.positions);
    notched.insert_attribute(Mesh::ATTRIBUTE_NORMAL, cut.normals);
    notched.insert_attribute(Mesh::ATTRIBUTE_UV_0, cut.uvs);
    notched.insert_attribute(Mesh::ATTRIBUTE_COLOR, cut.colors);
    notched.insert_indices(bevy::render::mesh::Indices::U32(cut.indices));

    // The chip becomes the notched piece. It gets its own mesh (the old one may be shared with
    // other chips), keeps its material and physics body, and stops being a short-lived chip.
    let new_mesh = meshes.add(notched);
    commands
        .entity(chip)
        .insert((new_mesh, Collider::compound(shapes), KnifeNotched))
        .remove::<TreeChip>()
        .remove::<bevy::render::primitives::Aabb>();
    status.0 = "You cut a V notch into the middle of the chip.".into();
}

/// Holds the sword in a ready stance in front of you. While the left mouse
/// button is held, mouse movement sweeps the blade through the air (the speed
/// is limited like a real arm), the edge turns into the cut, and when you let
/// go it eases back to the ready stance. The blade is a kinematic body, so a
/// fast swing really knocks boxes and balls around.
fn sword_pose_system(
    time: Res<Time>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion: EventReader<MouseMotion>,
    cam: Query<&Transform, (With<Camera3d>, Without<Sword>)>,
    mut swords: Query<(&mut Sword, &mut Transform)>,
) {
    let mut mouse = Vec2::ZERO;
    for ev in motion.read() {
        mouse += ev.delta;
    }
    let Ok(cam) = cam.get_single() else { return };
    let Ok((mut s, mut tf)) = swords.get_single_mut() else {
        return;
    };
    if !s.equipped {
        return;
    }
    let dt = time.delta_seconds().clamp(1e-3, 0.05);

    // --- swing dynamics ---
    let swinging = buttons.pressed(MouseButton::Left) && !buttons.pressed(MouseButton::Right);
    let old_psi = s.psi;
    if swinging {
        let max_step = 16.0 * dt; // ~16 rad/s: fast, but not teleporting
        s.psi = (s.psi + (mouse.x * 0.006).clamp(-max_step, max_step)).clamp(-1.8, 1.8);
        s.phi = (s.phi - (mouse.y * 0.006).clamp(-max_step, max_step)).clamp(-0.6, 1.5);
    } else {
        let k = 1.0 - (-6.0 * dt).exp();
        s.psi += (SWORD_REST_PSI - s.psi) * k;
        s.phi += (SWORD_REST_PHI - s.phi) * k;
    }
    let psi_vel = (s.psi - old_psi) / dt;
    s.psi_vel += (psi_vel - s.psi_vel) * (1.0 - (-15.0 * dt).exp());
    let target_roll = (s.psi_vel * 0.12).clamp(-1.3, 1.3);
    s.roll += (target_roll - s.roll) * (1.0 - (-12.0 * dt).exp());

    // --- build the pose in camera space ---
    let r = cam.rotation;
    let (right, up, fwd) = (r * Vec3::X, r * Vec3::Y, r * Vec3::NEG_Z);
    let (sp, cp) = s.psi.sin_cos();
    let (sf, cf) = s.phi.sin_cos();
    let blade = (right * (sp * cf) + up * sf + fwd * (cp * cf)).normalize();

    // the hand drifts a little with the swing so it feels like an arm, not a pivot in space
    let hand_local = Vec3::new(
        0.30 + (s.psi - SWORD_REST_PSI) * 0.12,
        -0.30 + (s.phi - SWORD_REST_PHI) * 0.10,
        -0.55,
    );
    let hand = cam.translation + r * hand_local;

    // the cutting edge faces forward in the stance and leads into the swing direction
    let edge_hint = fwd * s.roll.cos() + right * s.roll.sin();
    let mut z = edge_hint - blade * edge_hint.dot(blade);
    if z.length_squared() < 1e-4 {
        z = right - blade * right.dot(blade);
    }
    let z = z.normalize();
    let x = blade.cross(z);

    tf.translation = hand;
    tf.rotation = Quat::from_mat3(&Mat3::from_cols(x, blade, z));
}

/// Hold the axe in a first-person ready stance and animate mouse-driven swings.
fn axe_pose_system(
    time: Res<Time>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion: EventReader<MouseMotion>,
    cam: Query<&Transform, (With<Camera3d>, Without<Axe>)>,
    mut axes: Query<(&mut Axe, &mut Transform)>,
) {
    let mut mouse = Vec2::ZERO;
    for ev in motion.read() {
        mouse += ev.delta;
    }
    let Ok(cam) = cam.get_single() else { return };
    let Ok((mut axe, mut tf)) = axes.get_single_mut() else {
        return;
    };
    if !axe.equipped {
        return;
    }
    let dt = time.delta_seconds().clamp(1e-3, 0.05);
    let swinging = buttons.pressed(MouseButton::Left) && !buttons.pressed(MouseButton::Right);
    let old_psi = axe.psi;
    if swinging {
        let max_step = 16.0 * dt;
        axe.psi = (axe.psi + (mouse.x * 0.006).clamp(-max_step, max_step)).clamp(-1.8, 1.8);
        axe.phi = (axe.phi - (mouse.y * 0.006).clamp(-max_step, max_step)).clamp(-0.6, 1.5);
    } else {
        let k = 1.0 - (-6.0 * dt).exp();
        axe.psi += (SWORD_REST_PSI - axe.psi) * k;
        axe.phi += (SWORD_REST_PHI - axe.phi) * k;
    }
    let psi_vel = (axe.psi - old_psi) / dt;
    axe.psi_vel += (psi_vel - axe.psi_vel) * (1.0 - (-15.0 * dt).exp());
    let target_roll = (axe.psi_vel * 0.12).clamp(-1.3, 1.3);
    axe.roll += (target_roll - axe.roll) * (1.0 - (-12.0 * dt).exp());

    let r = cam.rotation;
    let (right, up, fwd) = (r * Vec3::X, r * Vec3::Y, r * Vec3::NEG_Z);
    let (sp, cp) = axe.psi.sin_cos();
    let (sf, cf) = axe.phi.sin_cos();
    let blade = (right * (sp * cf) + up * sf + fwd * (cp * cf)).normalize();
    let hand_local = Vec3::new(
        0.34 + (axe.psi - SWORD_REST_PSI) * 0.12,
        -0.34 + (axe.phi - SWORD_REST_PHI) * 0.1,
        -0.58,
    );
    let hand = cam.translation + r * hand_local;
    let edge_hint = fwd * axe.roll.cos() + right * axe.roll.sin();
    let mut z = edge_hint - blade * edge_hint.dot(blade);
    if z.length_squared() < 1e-4 {
        z = right - blade * right.dot(blade);
    }
    let z = z.normalize();
    let x = blade.cross(z);
    tf.translation = hand;
    tf.rotation = Quat::from_mat3(&Mat3::from_cols(x, blade, z));
}

/// Bring the shovel into a low two-handed digging stance while LMB is held.
fn shovel_pose_system(
    time: Res<Time>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion: EventReader<MouseMotion>,
    cam: Query<&Transform, (With<Camera3d>, Without<Shovel>)>,
    mut shovels: Query<(&mut Shovel, &mut Transform)>,
) {
    let mut mouse = Vec2::ZERO;
    for event in motion.read() {
        mouse += event.delta;
    }
    let Ok(cam) = cam.get_single() else { return };
    let Ok((mut shovel, mut transform)) = shovels.get_single_mut() else {
        return;
    };
    if !shovel.equipped {
        return;
    }

    let dt = time.delta_seconds().clamp(1e-3, 0.05);
    let digging_stance = buttons.pressed(MouseButton::Left) && !buttons.pressed(MouseButton::Right);
    if digging_stance {
        shovel.swing = (shovel.swing + mouse.y * 0.006).clamp(-0.65, 0.65);
    } else {
        shovel.swing += (0.0 - shovel.swing) * (1.0 - (-7.0 * dt).exp());
    }

    let rotation = if digging_stance {
        Quat::from_rotation_x(-1.9 + shovel.swing)
            * Quat::from_rotation_y(mouse.x.clamp(-80.0, 80.0) * 0.002)
    } else {
        Quat::from_rotation_x(-0.45)
    };
    let right = cam.rotation * Vec3::X;
    let forward = cam.rotation * Vec3::NEG_Z;
    transform.translation = cam.translation + right * 0.34 - Vec3::Y * 0.48 + forward * 0.42;
    transform.rotation = cam.rotation * rotation;
}

fn spear_pose_system(
    time: Res<Time>,
    cam: Query<&Transform, (With<Camera3d>, Without<Spear>)>,
    mut spears: Query<(&mut Spear, &mut Transform)>,
) {
    let Ok(cam) = cam.get_single() else { return };
    let Ok((mut spear, mut transform)) = spears.get_single_mut() else {
        return;
    };
    if !spear.equipped {
        return;
    }

    let dt = time.delta_seconds();
    spear.attack_left = (spear.attack_left - dt).max(0.0);
    let progress = 1.0 - spear.attack_left / spear.attack_duration;
    let thrust = if spear.attack_left > 0.0 {
        if progress < 0.45 {
            progress / 0.45
        } else {
            (1.0 - progress) / 0.55
        }
    } else {
        0.0
    }
    .clamp(0.0, 1.0);

    let right = cam.rotation * Vec3::X;
    let up = cam.rotation * Vec3::Y;
    let forward = cam.rotation * Vec3::NEG_Z;
    let hand = cam.translation + right * 0.34 - up * 0.22;
    // While roasting, the spear reaches out so the stabbed fish hangs over the flames.
    let cooking = spear.cook_left > 0.0 && spear.impaled_fish.is_some();
    let direction = if cooking {
        (spear.cook_point - hand).try_normalize().unwrap_or(forward)
    } else if spear.attack_left > 0.0 {
        spear.attack_direction
    } else {
        (forward + up * 0.12).normalize()
    };
    let mut z_axis = right - direction * right.dot(direction);
    if z_axis.length_squared() < 1e-4 {
        z_axis = up - direction * up.dot(direction);
    }
    let z_axis = z_axis.normalize();
    let x_axis = direction.cross(z_axis);
    transform.translation = if cooking {
        // fish sits SPEAR_FISH_Y along the shaft, so back the spear off by that much
        spear.cook_point - direction * SPEAR_FISH_Y
            + Vec3::new(0.0, (time.elapsed_seconds() * 9.0).sin() * 0.004, 0.0)
    } else {
        hand + direction * (0.42 + spear.push_distance * thrust)
    };
    transform.rotation = Quat::from_mat3(&Mat3::from_cols(x_axis, direction, z_axis));
}

/// Holds the machete in a ready stance in front of you. Mouse motion while LMB is
/// held sweeps the blade; releasing eases back to the ready stance.
fn machete_pose_system(
    time: Res<Time>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion: EventReader<MouseMotion>,
    peel: Res<MachetePeel>,
    cam: Query<&Transform, (With<Camera3d>, Without<Machete>)>,
    trunks: Query<(&ChoppableTrunk, &GlobalTransform)>,
    mut machetes: Query<(&mut Machete, &mut Transform)>,
) {
    let mut mouse = Vec2::ZERO;
    for ev in motion.read() {
        mouse += ev.delta;
    }
    let Ok(cam) = cam.get_single() else { return };
    let Ok((mut machete, mut tf)) = machetes.get_single_mut() else {
        return;
    };
    if !machete.equipped {
        return;
    }

    // --- Peeling: hold the blade flat against the trunk at the current row. ---
    if let Some(trunk_entity) = peel.trunk {
        if let Ok((trunk, trunk_gt)) = trunks.get(trunk_entity) {
            let body_tf = trunk_gt.compute_transform();
            let shape = &trunk.wood.shape;
            let row_y = ((peel.current_row as f32 + 0.5) * WOOD_ROW_H)
                .clamp(WOOD_ROW_H, shape.height.max(WOOD_ROW_H));
            let center_xz = shape.center_at(row_y);
            let r = shape.radius_at(row_y);
            let angle =
                (peel.sector as f32 + 0.5) / WOOD_SECTORS as f32 * std::f32::consts::TAU;
            let surface_local = Vec3::new(
                center_xz.x + angle.cos() * (r + 0.02),
                row_y,
                center_xz.y + angle.sin() * (r + 0.02),
            );
            let grid_origin = body_tf.translation + body_tf.rotation * trunk.offset;
            let world_surface = grid_origin + body_tf.rotation * surface_local;
            let outward = (body_tf.rotation
                * Vec3::new(angle.cos(), 0.0, angle.sin()))
            .normalize_or_zero();
            let up = (body_tf.rotation * Vec3::Y).normalize_or_zero();
            // Blade's flat (±X) faces outward; length (+Y) runs up the trunk.
            let x = outward;
            let y = up;
            let z = x.cross(y).normalize_or_zero();
            // The blade's midpoint is ~0.36 m up the local +Y axis.
            const BLADE_MID: f32 = 0.30;
            tf.translation = world_surface - up * BLADE_MID;
            tf.rotation = Quat::from_mat3(&Mat3::from_cols(x, y, z));
            return;
        }
    }

    let dt = time.delta_seconds().clamp(1e-3, 0.05);
    let swinging = buttons.pressed(MouseButton::Left) && !buttons.pressed(MouseButton::Right);
    let old_psi = machete.psi;
    if swinging {
        let max_step = 16.0 * dt;
        machete.psi =
            (machete.psi + (mouse.x * 0.006).clamp(-max_step, max_step)).clamp(-1.8, 1.8);
        machete.phi =
            (machete.phi - (mouse.y * 0.006).clamp(-max_step, max_step)).clamp(-0.6, 1.5);
    } else {
        let k = 1.0 - (-6.0 * dt).exp();
        machete.psi += (SWORD_REST_PSI - machete.psi) * k;
        machete.phi += (SWORD_REST_PHI - machete.phi) * k;
    }
    let psi_vel = (machete.psi - old_psi) / dt;
    machete.psi_vel += (psi_vel - machete.psi_vel) * (1.0 - (-15.0 * dt).exp());
    let target_roll = (machete.psi_vel * 0.12).clamp(-1.3, 1.3);
    machete.roll += (target_roll - machete.roll) * (1.0 - (-12.0 * dt).exp());

    let r = cam.rotation;
    let (right, up, fwd) = (r * Vec3::X, r * Vec3::Y, r * Vec3::NEG_Z);
    let (sp, cp) = machete.psi.sin_cos();
    let (sf, cf) = machete.phi.sin_cos();
    let blade = (right * (sp * cf) + up * sf + fwd * (cp * cf)).normalize();
    let hand_local = Vec3::new(
        0.32 + (machete.psi - SWORD_REST_PSI) * 0.12,
        -0.32 + (machete.phi - SWORD_REST_PHI) * 0.10,
        -0.56,
    );
    let hand = cam.translation + r * hand_local;
    let edge_hint = fwd * machete.roll.cos() + right * machete.roll.sin();
    let mut z = edge_hint - blade * edge_hint.dot(blade);
    if z.length_squared() < 1e-4 {
        z = right - blade * right.dot(blade);
    }
    let z = z.normalize();
    let x = blade.cross(z);
    tf.translation = hand;
    tf.rotation = Quat::from_mat3(&Mat3::from_cols(x, blade, z));
}

/// Holds the knife in a ready stance in front of you. It sits a little closer and
/// lower than the machete because the blade is short.
fn knife_pose_system(
    time: Res<Time>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion: EventReader<MouseMotion>,
    cam: Query<&Transform, (With<Camera3d>, Without<Knife>)>,
    mut knives: Query<(&mut Knife, &mut Transform)>,
) {
    let mut mouse = Vec2::ZERO;
    for ev in motion.read() {
        mouse += ev.delta;
    }
    let Ok(cam) = cam.get_single() else { return };
    let Ok((mut knife, mut tf)) = knives.get_single_mut() else {
        return;
    };
    if !knife.equipped {
        return;
    }
    let dt = time.delta_seconds().clamp(1e-3, 0.05);
    let swinging = buttons.pressed(MouseButton::Left) && !buttons.pressed(MouseButton::Right);
    let old_psi = knife.psi;
    if swinging {
        let max_step = 20.0 * dt; // a knife is lighter and flicks faster than the machete
        knife.psi = (knife.psi + (mouse.x * 0.007).clamp(-max_step, max_step)).clamp(-1.8, 1.8);
        knife.phi = (knife.phi - (mouse.y * 0.007).clamp(-max_step, max_step)).clamp(-0.6, 1.5);
    } else {
        let k = 1.0 - (-6.0 * dt).exp();
        knife.psi += (SWORD_REST_PSI - knife.psi) * k;
        knife.phi += (SWORD_REST_PHI - knife.phi) * k;
    }
    let psi_vel = (knife.psi - old_psi) / dt;
    knife.psi_vel += (psi_vel - knife.psi_vel) * (1.0 - (-15.0 * dt).exp());
    let target_roll = (knife.psi_vel * 0.12).clamp(-1.3, 1.3);
    knife.roll += (target_roll - knife.roll) * (1.0 - (-12.0 * dt).exp());

    let r = cam.rotation;
    let (right, up, fwd) = (r * Vec3::X, r * Vec3::Y, r * Vec3::NEG_Z);
    let (sp, cp) = knife.psi.sin_cos();
    let (sf, cf) = knife.phi.sin_cos();
    let blade = (right * (sp * cf) + up * sf + fwd * (cp * cf)).normalize();
    let hand_local = Vec3::new(
        0.22 + (knife.psi - SWORD_REST_PSI) * 0.10,
        -0.22 + (knife.phi - SWORD_REST_PHI) * 0.08,
        -0.42,
    );
    let hand = cam.translation + r * hand_local;
    let edge_hint = fwd * knife.roll.cos() + right * knife.roll.sin();
    let mut z = edge_hint - blade * edge_hint.dot(blade);
    if z.length_squared() < 1e-4 {
        z = right - blade * right.dot(blade);
    }
    let z = z.normalize();
    let x = blade.cross(z);
    tf.translation = hand;
    tf.rotation = Quat::from_mat3(&Mat3::from_cols(x, blade, z));
}

/// A deliberate shovel stroke aimed at the ground opens or enlarges a dirt patch.
fn shovel_dig_system(
    time: Res<Time>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion: EventReader<MouseMotion>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    player: Query<Entity, With<Player>>,
    rapier: Res<RapierContext>,
    mut shovels: Query<&mut Shovel>,
    mut sites: Query<(&Transform, &mut DigSite)>,
    mut terrain: Query<(Entity, &Handle<Mesh>), With<DiggableGround>>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
    mut dig_assets: Local<Option<(Handle<Mesh>, Handle<StandardMaterial>)>>,
    mut status: ResMut<Status>,
) {
    let mut mouse_delta = Vec2::ZERO;
    for event in motion.read() {
        mouse_delta += event.delta;
    }
    let Ok(mut shovel) = shovels.get_single_mut() else {
        return;
    };
    shovel.dig_cooldown = (shovel.dig_cooldown - time.delta_seconds()).max(0.0);
    if !shovel.equipped
        || !buttons.pressed(MouseButton::Left)
        || buttons.pressed(MouseButton::Right)
        || mouse_delta.length() < 10.0
        || shovel.dig_cooldown > 0.0
    {
        return;
    }
    let Ok(ground_entity) = terrain.get_single().map(|(entity, _)| entity) else {
        return;
    };

    let Ok(window) = windows.get_single() else {
        return;
    };
    let Ok((camera, camera_transform)) = cameras.get_single() else {
        return;
    };
    let Some(ray) = pointer_position(&window)
        .and_then(|cursor| camera.viewport_to_world(camera_transform, cursor))
    else {
        return;
    };
    if ray.direction.y >= -0.08 {
        return;
    }
    let filter = world_ray_filter(player.get_single().ok());
    let Some((hit_entity, distance)) =
        rapier.cast_ray(ray.origin, *ray.direction, 5.0, true, filter)
    else {
        return;
    };
    if hit_entity != ground_entity {
        return;
    }
    let point = ray.origin + *ray.direction * distance;
    if point.y > GROUND_SURFACE_Y + 0.15
        || point.y < GROUND_SURFACE_Y - DIG_MAX_DEPTH - 0.12
        || point.distance(camera_transform.translation()) > 4.5
    {
        return;
    }

    let point_xz = Vec2::new(point.x, point.z);
    let mut site_shapes = Vec::new();
    let mut enlarged = false;
    for (transform, mut site) in &mut sites {
        let center = Vec2::new(transform.translation.x, transform.translation.z);
        if !enlarged && center.distance(point_xz) < 0.9 {
            site.radius = (site.radius + 0.16).min(1.6);
            enlarged = true;
        }
        site_shapes.push((center, site.radius));
    }

    if !enlarged {
        let radius = 0.75;
        site_shapes.push((point_xz, radius));
        let (mound_mesh, mound_material) = dig_assets.get_or_insert_with(|| {
            (
                meshes.add(Sphere::new(1.0)),
                mats.add(Color::srgb(0.32, 0.2, 0.11)),
            )
        });
        let mound_mesh = mound_mesh.clone();
        let mound_material = mound_material.clone();
        commands
            .spawn((
                SpatialBundle::from_transform(Transform::from_xyz(
                    point.x,
                    GROUND_SURFACE_Y,
                    point.z,
                )),
                DigSite { radius },
            ))
            .with_children(move |parent| {
                parent.spawn(PbrBundle {
                    mesh: mound_mesh,
                    material: mound_material,
                    transform: Transform {
                        translation: Vec3::new(radius + 0.45, 0.08, 0.0),
                        scale: Vec3::new(0.55, 0.14, 0.4),
                        ..default()
                    },
                    ..default()
                });
            });
    }

    update_diggable_ground(&site_shapes, &mut terrain, &mut meshes, &mut commands);
    shovel.dig_cooldown = 0.55;
    status.0 = if enlarged {
        "You shovel another scoop; the hole is deeper and wider.".into()
    } else {
        "You dig a real depression in the ground; step into it to feel the hole.".into()
    };
}

/// Each deliberate cursor swing that hits a nearby trunk chops the outermost block of
/// wood out of every column it touches (a small straight cut first, deeper and wider
/// when the same spot is hit again). There is no "health" counter: blocks that lose
/// their connection to the ground fall off, and the tree goes over when the remaining
/// cross-section can no longer hold its weight and lean.
fn axe_chop_system(
    time: Res<Time>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion: EventReader<MouseMotion>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    mut player: Query<(Entity, &mut Player)>,
    rapier: Res<RapierContext>,
    mut axes: Query<&mut Axe>,
    mut tree_queries: TreeHitQueries,
    transforms: Query<&GlobalTransform>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
    mut status: ResMut<Status>,
    mut npc_swing: ResMut<NpcSwing>,
) {
    let mut mouse_delta = Vec2::ZERO;
    for ev in motion.read() {
        mouse_delta += ev.delta;
    }
    let Ok(mut axe) = axes.get_single_mut() else {
        return;
    };
    axe.chop_cooldown = (axe.chop_cooldown - time.delta_seconds()).max(0.0);
    // A swing made by Tom (the person) instead of the player brings its own origin and direction
    let npc_swing_ray = npc_swing.axe.take();
    if npc_swing_ray.is_some() {
        if axe.chop_cooldown > 0.0 {
            return;
        }
    } else if !axe.equipped
        || !buttons.pressed(MouseButton::Left)
        || buttons.pressed(MouseButton::Right)
        || mouse_delta.length() < 10.0
        || axe.chop_cooldown > 0.0
    {
        return;
    }
    let (ray_origin, ray_dir): (Vec3, Vec3) = if let Some((o, d)) = npc_swing_ray {
        (o, d.normalize_or_zero())
    } else {
        let Ok(window) = windows.get_single() else {
            return;
        };
        let Ok((camera, cam_tf)) = cameras.get_single() else {
            return;
        };
        let Some(ray) = pointer_position(&window)
            .and_then(|cursor| camera.viewport_to_world(cam_tf, cursor))
        else {
            return;
        };
        (ray.origin, *ray.direction)
    };
    if ray_dir == Vec3::ZERO {
        return;
    }
    let player_data = player.get_single_mut().ok();
    let filter = world_ray_filter(player_data.as_ref().map(|(entity, _)| *entity));
    let Some((hit_entity, hit_distance)) =
        rapier.cast_ray(ray_origin, ray_dir, 4.5, true, filter)
    else {
        return;
    };
    let hit_world = ray_origin + ray_dir * hit_distance;

    let mut branch_ancestor = hit_entity;
    loop {
        if tree_queries.felled_branches.get(branch_ancestor).is_ok() {
            chop_felled_branch(
                branch_ancestor,
                hit_world,
                time.elapsed_seconds(),
                &mut axe,
                &mut tree_queries,
                &transforms,
                &mut commands,
                &mut meshes,
                &mats,
                &mut status,
            );
            return;
        }
        let Ok(parent) = tree_queries.parents.get(branch_ancestor) else {
            break;
        };
        branch_ancestor = parent.get();
    }

    let mut log_ancestor = hit_entity;
    loop {
        if tree_queries.choppable_trunks.get(log_ancestor).is_ok() {
            chop_felled_log(
                log_ancestor,
                hit_world,
                time.elapsed_seconds(),
                &mut axe,
                &mut tree_queries,
                &transforms,
                &mut commands,
                &mut meshes,
                &mats,
                &mut status,
            );
            return;
        }
        let Ok(parent) = tree_queries.parents.get(log_ancestor) else {
            break;
        };
        log_ancestor = parent.get();
    }

    let mut ancestor = hit_entity;
    let mut hit_branch = None;
    let tree_entity = loop {
        if tree_queries.trees.get(ancestor).is_ok() {
            break ancestor;
        }
        if let Ok((branch_entity, branch)) = tree_queries.branches.get(ancestor) {
            hit_branch = Some(branch_entity);
            break branch.tree;
        }
        let Ok(parent) = tree_queries.parents.get(ancestor) else {
            return;
        };
        ancestor = parent.get();
    };
    let Ok((_, mut tree, tree_tf)) = tree_queries.trees.get_mut(tree_entity) else {
        return;
    };

    if hit_branch.is_some()
        && player_data
            .as_ref()
            .and_then(|(_, player)| player.climbing_tree)
            .is_some_and(|climbing_tree| climbing_tree == tree_entity)
    {
        let Some(branch_entity) = hit_branch else {
            return;
        };
        let remaining = {
            let Ok((_, mut branch)) = tree_queries.branches.get_mut(branch_entity) else {
                return;
            };
            if branch.broken {
                return;
            }
            branch.chops_remaining = branch.chops_remaining.saturating_sub(1);
            if branch.chops_remaining == 0 {
                branch.chops_remaining = 3;
            }
            branch.chops_remaining
        };
        axe.chop_cooldown = 0.4;
        if remaining == 3 {
            add_branch_cut_mark(
                branch_entity,
                hit_world,
                ray_dir,
                tree.inner_wood.clone(),
                &transforms,
                &mut commands,
                &mut meshes,
            );
            status.0 = "You cut a section from the branch.".into();
        } else {
            status.0 = format!("Chop! The branch needs {} more hit(s).", remaining);
        }
        return;
    }

    let hit_point = ray_origin + ray_dir * hit_distance;
    let tree_origin = tree_tf.translation();
    let base = tree_origin - Vec3::Y * (tree.height / 2.0);
    let shape = TrunkShape {
        height: tree.height,
        base_radius: tree.base_radius,
        top_radius: tree.top_radius,
        lean: tree.lean,
    };
    let tree_height = tree.height;
    let tree_lean = tree.lean;
    let bark_handle = tree.bark.clone();
    let trunk_mesh_handle = tree.trunk_mesh.clone();
    // The first hit turns the smooth trunk into the layered block trunk.
    let mut wood = tree.wood.take().unwrap_or_else(|| TrunkWood::new(shape));
    let bark_color = mats
        .get(&bark_handle)
        .map(|m| m.base_color)
        .unwrap_or(Color::WHITE);
    let cut_ratio = wood_cut_ratios(bark_color);
    let hit_seed = time.elapsed_seconds().to_bits() ^ tree_entity.index();

    // ----- The hit: a small straight cut that gets deeper (and wider) when you keep
    // chopping the same spot. The chopped blocks fly off as little physics chips. -----
    let (sector, row) = wood.locate(hit_point - base);
    let removed = wood.chop(sector, row);
    axe.chop_cooldown = 0.4;
    for (n, &cell) in removed.iter().take(14).enumerate() {
        spawn_wood_debris(
        &mut commands, 
        &mut meshes, 
        &bark_handle, 
        base, 
        &wood,
        &[cell], 
        &cut_ratio,
        hit_seed.wrapping_add(n as u32 * 29),
        1.6, 
        0.18,
        time.elapsed_seconds(),
);
    }

    // ----- Does the upper trunk come away? Two ways: wood that is no longer joined to
    // the ground at all, or a cut so deep that the remaining hinge can't hold the tree. -----
    let mut upper: Option<Vec<bool>> = None;
    let mut fall_dir = Vec3::ZERO;
    {
        let grounded = wood.grounded(&wood.cells);
        let floating: Vec<bool> = wood
            .cells
            .iter()
            .zip(&grounded)
            .map(|(&cell, &ground)| cell && !ground)
            .collect();
        let top = wood.top_row(&wood.cells);
        for group in wood.components(&floating) {
            let reaches_top =
                top.is_some_and(|t| group.iter().any(|&i| TrunkWood::split(i).0 == t));
            if reaches_top && group.len() >= WOOD_FELL_MIN_CELLS {
                let mut mask = vec![false; wood.cells.len()];
                for &i in &group {
                    mask[i] = true;
                }
                let lowest = group
                    .iter()
                    .map(|&i| TrunkWood::split(i).0)
                    .min()
                    .unwrap_or(1);
                fall_dir = wood.removed_direction(lowest.saturating_sub(1));
                upper = Some(mask);
            }
        }
    }
    if upper.is_none() {
        let lo = (row - 3).max(WOOD_MIN_CHOP_ROW) as usize;
        let hi = ((row + 3) as usize).min(wood.rows.saturating_sub(2));
        let weakest = (lo..=hi).min_by(|a, b| {
            wood.row_area(*a)
                .partial_cmp(&wood.row_area(*b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(weak) = weakest {
            let dir = wood.removed_direction(weak);
            let lean_toward = tree_lean.dot(Vec2::new(dir.x, dir.z));
            let height_factor = ((tree_height - 4.0) / 4.0).clamp(0.0, 1.0) * 0.08;
            let snap_area = (0.32 + lean_toward * 0.18 + height_factor).clamp(0.14, 0.55);
            let mask: Vec<bool> = wood
                .cells
                .iter()
                .enumerate()
                .map(|(i, &present)| present && TrunkWood::split(i).0 > weak)
                .collect();
            let above = mask.iter().filter(|&&present| present).count();
            if wood.row_area(weak) <= snap_area && above >= WOOD_FELL_MIN_CELLS {
                fall_dir = dir;
                upper = Some(mask);
            }
        }
    }
    let felled = upper.is_some();
    if let Some(mask) = &upper {
        if fall_dir.length_squared() < 0.01 {
            let (spot_sector, _) = wood.spot.unwrap_or((sector, row));
            let angle = (spot_sector as f32 + 0.5) / WOOD_SECTORS as f32 * std::f32::consts::TAU;
            fall_dir = Vec3::new(angle.cos(), 0.0, angle.sin());
        }
        for (cell, &gone) in wood.cells.iter_mut().zip(mask) {
            if gone {
                *cell = false;
            }
        }
    }

    // ----- Blocks that lost their connection fall off as chunks. The layer beneath
    // them is now the surface. -----
    let loose_groups = {
        let grounded = wood.grounded(&wood.cells);
        let floating: Vec<bool> = wood
            .cells
            .iter()
            .zip(&grounded)
            .map(|(&cell, &ground)| cell && !ground)
            .collect();
        wood.components(&floating)
    };
    for (n, group) in loose_groups.iter().enumerate() {
        spawn_wood_debris(
            &mut commands,
            &mut meshes,
            &bark_handle,
            base,
            &wood,
            group,
            &cut_ratio,
            hit_seed.wrapping_add(500 + n as u32 * 53),
            0.9,
            0.12,
            14.0,
        );
        for &i in group {
            wood.cells[i] = false;
        }
    }

    // ----- Rebuild what the trunk looks like now (one mesh, no overlapping faces). -----
    if let Some(m) = meshes.get_mut(&trunk_mesh_handle) {
        *m = wood_mesh_from_data(wood.build_mesh(&wood.cells, Vec3::ZERO, true, &cut_ratio));
    }

    let cut_percent = ((1.0 - wood.row_area(row as usize)) * 100.0).round();
    let spot_hits = wood.spot_hits;
    let had_loose = !loose_groups.is_empty();
    let nothing_removed = removed.is_empty();

    if !felled {
        tree.wood = Some(wood);
        status.0 = if had_loose {
            "A block of wood is cut loose and falls away; the layer underneath shows.".into()
        } else if nothing_removed {
            "The cut already goes right through here; widen it or cut somewhere else.".into()
        } else if spot_hits <= 1 {
            "A small straight cut opens in the trunk. Keep chopping the same spot to go deeper."
                .into()
        } else {
            format!("Chop! The cut goes deeper: about {cut_percent:.0}% of the trunk is cut through.")
        };
        return;
    }

    // ----- Timber! Everything above the break goes over. -----
    let upper_mask = upper.unwrap_or_default();
    let lowest_upper_row = (0..wood.rows)
        .find(|&r| {
            let start = r * WOOD_SECTORS * WOOD_LAYERS;
            (start..start + WOOD_SECTORS * WOOD_LAYERS).any(|i| upper_mask[i])
        })
        .unwrap_or(1);
    let y_cut = lowest_upper_row as f32 * WOOD_ROW_H;
    let cut_center = shape.center_at(y_cut);
    let upper_origin = Vec3::new(cut_center.x, y_cut, cut_center.y);
    // The felled trunk keeps the same layered wood blocks, so it is chopped the same way.
    let log_wood = TrunkWood {
        shape,
        rows: wood.rows,
        cells: upper_mask.clone(),
        spot: None,
        spot_hits: 0,
    };

    // The stump keeps a simple collider up to the height of its highest remaining block.
    let stump_top = wood.top_row(&wood.cells).map_or(0, |r| r + 1) as f32 * WOOD_ROW_H;
    let stump_h = stump_top.max(0.3);
    let stump_r = shape.radius_at(stump_h * 0.5).max(0.16);
    if let Some(old) = tree.stump_collider.take() {
        commands.entity(old).despawn_recursive();
    }
    let stump_collider = commands
        .spawn((
            SpatialBundle::from_transform(Transform::from_translation(Vec3::new(
                0.0,
                -tree_height * 0.5 + stump_h * 0.5,
                0.0,
            ))),
            Collider::cylinder(stump_h * 0.5, stump_r),
        ))
        .id();
    commands
        .entity(tree_entity)
        .remove::<Collider>()
        .add_child(stump_collider);
    tree.stump_collider = Some(stump_collider);
    // The far-away crown stand-ins belonged to the standing tree.
    if let Ok(children) = tree_queries.children.get(tree_entity) {
        for child in children.iter() {
            if tree_queries.lod_pieces.get(*child).is_ok() {
                commands.entity(*child).despawn_recursive();
            }
        }
    }
    tree.wood = Some(wood);
    if let Some((_, mut player)) = player_data {
        if player.climbing_tree == Some(tree_entity) {
            player.climbing_tree = None;
        }
    }

    let lean_toward = tree_lean.dot(Vec2::new(fall_dir.x, fall_dir.z));
    fell_wood_trunk(
        tree_entity,
        base,
        shape,
        y_cut,
        fall_dir,
        log_wood,
        upper_origin,
        &cut_ratio,
        bark_handle,
        &mut tree_queries,
        &transforms,
        &mut commands,
        &mut meshes,
    );
    let lean_note = if lean_toward > 0.2 {
        " It was already leaning that way."
    } else if lean_toward < -0.2 {
        " It took a lot of cutting against the lean."
    } else {
        ""
    };
    status.0 = format!("Timber! The remaining hinge gives way and the tree goes over.{lean_note}");
}

/// Keeps the number of wood-debris entities bounded: once there are more than
/// `MAX_WOOD_DEBRIS`, the oldest ones are quietly removed.
fn cap_wood_debris_system(
    mut commands: Commands,
    debris: Query<
        (Entity, &WoodDebris),
        (Without<KnifeNotched>, Without<BarkStrip>),
    >,
) {
    let count = debris.iter().count();
    if count <= MAX_WOOD_DEBRIS {
        return;
    }
    let mut all: Vec<(Entity, f32)> = debris.iter().map(|(e, d)| (e, d.born)).collect();
    all.sort_by(|a, b| a.1.total_cmp(&b.1));
    for (entity, _) in all.into_iter().take(count - MAX_WOOD_DEBRIS) {
        commands.entity(entity).despawn_recursive();
    }
}

/// A fast swing of the held machete cuts through soft vegetation and bats sticks away.
/// Anything the blade segment passes near gets an outward impulse; grass/plants are
/// also uprooted (so they fall and can be picked up like any other prop).
fn machete_slash_system(
    time: Res<Time>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion: EventReader<MouseMotion>,
    mut commands: Commands,
    mut machetes: Query<(&mut Machete, &GlobalTransform)>,
    mut plants: Query<
        (Entity, &mut PullablePlant, &mut Velocity, &Transform, &mut GravityScale),
        (Without<Machete>, Without<Stick>),
    >,
    mut sticks: Query<
        (&mut Velocity, &Transform),
        (With<Stick>, Without<Machete>, Without<PullablePlant>),
    >,
    peel: Res<MachetePeel>,
    mut status: ResMut<Status>,
) {
    if peel.trunk.is_some() {
        return;
    }
    let mut mouse = Vec2::ZERO;
    for ev in motion.read() {
        mouse += ev.delta;
    }
    let dt = time.delta_seconds().min(0.05);
    let Ok((mut machete, machete_gt)) = machetes.get_single_mut() else {
        return;
    };
    machete.slash_cooldown = (machete.slash_cooldown - dt).max(0.0);
    if !machete.equipped
        || !buttons.pressed(MouseButton::Left)
        || buttons.pressed(MouseButton::Right)
        || mouse.length() < 12.0
        || machete.slash_cooldown > 0.0
        || machete.psi_vel.abs() < 0.6
    {
        return;
    }

    // The blade runs from the hilt along +Y in the machete's own space.
    let hilt = machete_gt.translation();
    let tip = machete_gt.transform_point(Vec3::Y * 0.9);
    let segment = tip - hilt;
    let segment_len_sq = segment.length_squared().max(1e-4);
    let closest_distance = |p: Vec3| -> (f32, f32) {
        let t = ((p - hilt).dot(segment) / segment_len_sq).clamp(0.0, 1.0);
        let closest = hilt + segment * t;
        (p.distance(closest), t)
    };

    let mut cut_anything = false;

    for (entity, mut plant, mut vel, tf, mut grav) in &mut plants {
        let p = tf.translation;
        let (dist, _t) = closest_distance(p);
        if dist < 0.42 && !plant.uprooted {
            plant.uprooted = true;
            grav.0 = 1.0;
            commands.entity(entity).remove::<Sensor>();
            let outward = (p - hilt).normalize_or_zero();
            vel.linvel += outward * 4.0
                + Vec3::Y * 3.0
                + Vec3::new(0.0, 0.0, 0.0);
            vel.angvel += Vec3::new(0.0, 2.0, 0.0);
            cut_anything = true;
        }
    }

    for (mut vel, tf) in &mut sticks {
        let p = tf.translation;
        let (dist, _t) = closest_distance(p);
        if dist < 0.42 {
            let outward = (p - hilt).normalize_or_zero();
            vel.linvel += outward * 5.0 + Vec3::Y * 3.5;
            vel.angvel += Vec3::new(0.0, 0.0, 1.0) * 6.0;
            cut_anything = true;
        }
    }

    if cut_anything {
        machete.slash_cooldown = 0.28;
        status.0 = "The machete slashes through the vegetation.".into();
    }
}

/// Machete: with the blade equipped, hold LMB and rub the cursor up and down against an
/// already-felled trunk (a `ChoppableTrunk`). Each stroke scrapes the outer layer of bark
/// off the spot you're rubbing. The bark cells are removed from the wood grid, the log's
/// mesh and collider are rebuilt, and the pale sapwood underneath shows through.
fn machete_strip_bark_system(
    time: Res<Time>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion: EventReader<MouseMotion>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    player: Query<Entity, With<Player>>,
    rapier: Res<RapierContext>,
    mut machetes: Query<&mut Machete>,
    mut trunks: Query<(Entity, &mut ChoppableTrunk, &GlobalTransform)>,
    parents: Query<&Parent>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mats: Res<Assets<StandardMaterial>>,
    mut status: ResMut<Status>,
) {
    // The gesture is a deliberate up-and-down stroke: sum the *vertical* motion only.
    let mut vertical: f32 = 0.0;
    for ev in motion.read() {
        vertical += ev.delta.y.abs();
    }
    let dt = time.delta_seconds().min(0.05);
    let Ok(mut machete) = machetes.get_single_mut() else {
        return;
    };
    machete.strip_cooldown = (machete.strip_cooldown - dt).max(0.0);
    if !machete.equipped
        || !buttons.pressed(MouseButton::Left)
        || buttons.pressed(MouseButton::Right)
        || vertical < 12.0
        || machete.strip_cooldown > 0.0
    {
        return;
    }

    let (Ok(window), Ok((camera, cam_tf))) = (windows.get_single(), cameras.get_single())
    else {
        return;
    };
    let Some(ray) = pointer_position(&window)
        .and_then(|c| camera.viewport_to_world(cam_tf, c))
    else {
        return;
    };
    let filter = world_ray_filter(player.get_single().ok());
    let Some((hit, toi)) = rapier.cast_ray(ray.origin, *ray.direction, 3.5, true, filter)
    else {
        return;
    };
    let hit_world = ray.origin + *ray.direction * toi;

    // Walk up from the collider to the log body.
    let mut walker = hit;
    let log_entity = loop {
        if trunks.get(walker).is_ok() {
            break walker;
        }
        match parents.get(walker) {
            Ok(p) => walker = p.get(),
            Err(_) => return,
        }
    };

    let Ok((_, mut trunk, body_global)) = trunks.get_mut(log_entity) else {
        return;
    };

    // Where on the wood grid did we hit?
    let local = body_global.affine().inverse().transform_point3(hit_world);
    let (sector, row) = trunk.wood.locate_free(local - trunk.offset);

    // Strip a small band of bark around the hit: a couple of sectors either side and a
    // row up/down. Only layer 0 (bark). We never strip a bark cell whose layer 1 is
    // missing — that would leave a hole all the way to the axis.
    let sectors = WOOD_SECTORS as i32;
    let mut stripped = 0;
    for dr in -1i32..=1 {
        let r = row as i32 + dr;
        if r < 0 || r >= trunk.wood.rows as i32 {
            continue;
        }
        for ds in -2i32..=2 {
            let s = (sector as i32 + ds).rem_euclid(sectors) as usize;
            let r = r as usize;
            let i = TrunkWood::index(r, s, 0);
            if !trunk.wood.cells[i] {
                continue;
            }
            let behind = TrunkWood::index(r, s, 1);
            if !trunk.wood.cells[behind] {
                continue; // already cut into the sapwood here
            }
            trunk.wood.cells[i] = false;
            stripped += 1;
        }
    }

    machete.strip_cooldown = 0.12;
    if stripped == 0 {
        status.0 = "The bark here is already stripped away.".into();
        return;
    }

    // Rebuild the log's surface and its collider so the stripped band shows up in both.
    let bark_color = mats
        .get(&trunk.bark)
        .map(|m| m.base_color)
        .unwrap_or(Color::WHITE);
    let cut_ratio = wood_cut_ratios(bark_color);
    if let Some(m) = meshes.get_mut(&trunk.mesh) {
        *m = wood_mesh_from_data(trunk.wood.build_mesh(
            &trunk.wood.cells,
            Vec3::ZERO,
            false,
            &cut_ratio,
        ));
    }
    if let Some(collider) = trunk.wood.collider_of(&trunk.wood.cells, trunk.offset) {
        commands.entity(trunk.collider).insert(collider);
    }
    status.0 = "You scrape a strip of bark off the log with the machete.".into();
}

/// Removes the bark (layer 0) from a band around `sector` in the given row, rebuilds the
/// trunk's mesh and collider, and spawns the peeled bark as a grabbable physics object.
/// Returns true if anything was actually removed.
fn strip_bark_row(
    trunk: &mut ChoppableTrunk,
    trunk_gt: &GlobalTransform,
    sector: i32,
    row: i32,
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &Assets<StandardMaterial>,
) -> bool {
    let rows = trunk.wood.rows as i32;
    if row < 0 || row >= rows {
        return false;
    }
    let sectors = WOOD_SECTORS as i32;
    let r = row as usize;
    let half_band = 2; // 5 sectors, roughly a hand's width

    let mut removed: Vec<usize> = Vec::new();
    for ds in -half_band..=half_band {
        let s = (sector + ds).rem_euclid(sectors) as usize;
        let i0 = TrunkWood::index(r, s, 0);
        let i1 = TrunkWood::index(r, s, 1);
        // Only peel the bark where there's still wood under it, so the trunk doesn't
        // develop a hole straight through to the pith.
        if trunk.wood.cells[i0] && trunk.wood.cells[i1] {
            trunk.wood.cells[i0] = false;
            removed.push(i0);
        }
    }
    if removed.is_empty() {
        return false;
    }

    let bark_color = mats
        .get(&trunk.bark)
        .map(|m| m.base_color)
        .unwrap_or(Color::WHITE);
    let cut_ratio = wood_cut_ratios(bark_color);

    // Rebuild the trunk's surface so the pale wood underneath shows where we scraped.
    if let Some(m) = meshes.get_mut(&trunk.mesh) {
        *m = wood_mesh_from_data(trunk.wood.build_mesh(
            &trunk.wood.cells,
            Vec3::ZERO,
            false,
            &cut_ratio,
        ));
    }
    if let Some(collider) = trunk.wood.collider_of(&trunk.wood.cells, trunk.offset) {
        commands.entity(trunk.collider).insert(collider);
    }

    // Spawn the peeled bark as its own physics object.
    let body_tf = trunk_gt.compute_transform();
    let frame = Transform {
        translation: body_tf.translation + body_tf.rotation * trunk.offset,
        rotation: body_tf.rotation,
        scale: Vec3::ONE,
    };
    let seed = (row as u32)
        .wrapping_mul(31)
        .wrapping_add((sector as u32).wrapping_mul(17))
        .wrapping_add(0xb02c_1234);
    if let Some(bark) = spawn_wood_frame_debris(
        commands,
        meshes,
        &trunk.bark,
        &frame,
        &trunk.wood,
        &removed,
        &cut_ratio,
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::ZERO,
        seed,
        0.7,   // outward kick: it pops off the trunk
        0.0,
    ) {
        commands.entity(bark).insert(BarkStrip);
    }
    true
}

/// Hold the left mouse button on an already-felled trunk with the machete equipped: after a
/// short moment you settle into a peeling stance against the trunk. From there, moving the
/// mouse up and down scrapes the outer bark off one row-band per stroke. Each strip comes
/// away as its own grabbable physics object.
#[allow(clippy::too_many_arguments)]
fn machete_peel_system(
    time: Res<Time>,
    mut motion: EventReader<MouseMotion>,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    player: Query<Entity, With<Player>>,
    rapier: Res<RapierContext>,
    machetes: Query<&Machete>,
    mut trunks: Query<(&mut ChoppableTrunk, &GlobalTransform)>,
    parents: Query<&Parent>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mats: Res<Assets<StandardMaterial>>,
    mut peel: ResMut<MachetePeel>,
    mut status: ResMut<Status>,
) {
    // Read this frame's vertical motion (EventReader is per-system, so this doesn't starve
    // the pose / slash systems).
    let mut mouse_dy: f32 = 0.0;
    for ev in motion.read() {
        mouse_dy += ev.delta.y;
    }

    let Ok(machete) = machetes.get_single() else {
        peel.trunk = None;
        peel.hold_time = 0.0;
        return;
    };
    if !machete.equipped {
        peel.trunk = None;
        peel.hold_time = 0.0;
        return;
    }

    let dt = time.delta_seconds().min(0.05);
    let lmb = buttons.pressed(MouseButton::Left);

    // ---- Already peeling: keep going until LMB is released or we walk away. ----
    if let Some(entity) = peel.trunk {
        let Ok((mut trunk, trunk_gt)) = trunks.get_mut(entity) else {
            peel.trunk = None;
            return;
        };
        let too_far = cameras
            .get_single()
            .ok()
            .map(|(_, t)| t.translation().distance(trunk_gt.translation()) > 4.0)
            .unwrap_or(true);
        if !lmb || too_far {
            peel.trunk = None;
            peel.accumulated = 0.0;
            if too_far {
                status.0 = "You step back from the trunk.".into();
            }
            return;
        }

        peel.accumulated += mouse_dy;
        // Every ~24 px of vertical mouse motion peels one row off, in the direction of the
        // gesture: down => lower rows, up => higher rows.
        const STROKE_PX: f32 = 24.0;
        while peel.accumulated.abs() >= STROKE_PX {
            let sign = peel.accumulated.signum();
            peel.accumulated -= sign * STROKE_PX;
            let dir: i32 = if sign > 0.0 { -1 } else { 1 };
            strip_bark_row(
                &mut trunk,
                &trunk_gt,
                peel.sector,
                peel.current_row,
                &mut commands,
                &mut meshes,
                &mats,
            );
            let rows = trunk.wood.rows as i32;
            peel.current_row = (peel.current_row + dir).clamp(0, rows - 1);
        }
        return;
    }

    // ---- Not yet peeling: if LMB is held over a chopped trunk, start the long-press timer. ----
    if !lmb {
        peel.hold_time = 0.0;
        return;
    }
    let (Ok(window), Ok((camera, cam_tf))) = (windows.get_single(), cameras.get_single()) else {
        return;
    };
    let Some(ray) = pointer_position(&window)
        .and_then(|c| camera.viewport_to_world(cam_tf, c))
    else {
        peel.hold_time = 0.0;
        return;
    };
    let filter = world_ray_filter(player.get_single().ok());
    let Some((hit, toi)) = rapier.cast_ray(ray.origin, *ray.direction, 3.5, true, filter) else {
        peel.hold_time = 0.0;
        return;
    };
    let hit_world = ray.origin + *ray.direction * toi;

    // Walk up the hierarchy to a ChoppableTrunk.
    let mut walker = hit;
    let trunk_entity = loop {
        if trunks.get(walker).is_ok() {
            break Some(walker);
        }
        match parents.get(walker) {
            Ok(p) => walker = p.get(),
            Err(_) => break None,
        }
    };
    let Some(trunk_entity) = trunk_entity else {
        peel.hold_time = 0.0;
        return;
    };

    peel.hold_time += dt;
    const LONG_PRESS_SECONDS: f32 = 0.15;
    if peel.hold_time < LONG_PRESS_SECONDS {
        return;
    }

    let Ok((trunk, trunk_gt)) = trunks.get(trunk_entity) else {
        return;
    };
    let local = trunk_gt.affine().inverse().transform_point3(hit_world);
    let (sector, row) = trunk.wood.locate_free(local - trunk.offset);
    peel.trunk = Some(trunk_entity);
    peel.sector = sector;
    peel.current_row = row;
    peel.accumulated = 0.0;
    status.0 =
        "Machete flat against the trunk. Move the mouse up and down to peel the bark away."
            .into();
}

/// A quick flick of the held knife cuts soft vegetation and bats sticks away.
/// The knife has a shorter reach than the machete but a much shorter cooldown.
fn knife_slash_system(
    time: Res<Time>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion: EventReader<MouseMotion>,
    mut commands: Commands,
    mut knives: Query<(&mut Knife, &GlobalTransform)>,
    mut plants: Query<
        (Entity, &mut PullablePlant, &mut Velocity, &Transform, &mut GravityScale),
        (Without<Knife>, Without<Stick>),
    >,
    mut sticks: Query<
        (&mut Velocity, &Transform),
        (With<Stick>, Without<Knife>, Without<PullablePlant>),
    >,
    mut status: ResMut<Status>,
) {
    let mut mouse = Vec2::ZERO;
    for ev in motion.read() {
        mouse += ev.delta;
    }
    let dt = time.delta_seconds().min(0.05);
    let Ok((mut knife, knife_gt)) = knives.get_single_mut() else {
        return;
    };
    knife.slash_cooldown = (knife.slash_cooldown - dt).max(0.0);
    if !knife.equipped
        || !buttons.pressed(MouseButton::Left)
        || buttons.pressed(MouseButton::Right)
        || mouse.length() < 10.0
        || knife.slash_cooldown > 0.0
        || knife.psi_vel.abs() < 0.8
    {
        return;
    }

    // The knife blade runs from the hilt along +Y in the knife's own space (shorter than the machete).
    let hilt = knife_gt.translation();
    let tip = knife_gt.transform_point(Vec3::Y * 0.36);
    let segment = tip - hilt;
    let segment_len_sq = segment.length_squared().max(1e-4);
    let closest_distance = |p: Vec3| -> (f32, f32) {
        let t = ((p - hilt).dot(segment) / segment_len_sq).clamp(0.0, 1.0);
        let closest = hilt + segment * t;
        (p.distance(closest), t)
    };

    let mut cut_anything = false;

    for (entity, mut plant, mut vel, tf, mut grav) in &mut plants {
        let p = tf.translation;
        let (dist, _t) = closest_distance(p);
        if dist < 0.35 && !plant.uprooted {
            plant.uprooted = true;
            grav.0 = 1.0;
            commands.entity(entity).remove::<Sensor>();
            let outward = (p - hilt).normalize_or_zero();
            vel.linvel += outward * 3.5 + Vec3::Y * 2.6;
            vel.angvel += Vec3::new(0.0, 2.5, 0.0);
            cut_anything = true;
        }
    }

    for (mut vel, tf) in &mut sticks {
        let p = tf.translation;
        let (dist, _t) = closest_distance(p);
        if dist < 0.35 {
            let outward = (p - hilt).normalize_or_zero();
            vel.linvel += outward * 4.5 + Vec3::Y * 3.0;
            vel.angvel += Vec3::new(0.0, 0.0, 1.0) * 7.0;
            cut_anything = true;
        }
    }

    if cut_anything {
        knife.slash_cooldown = 0.18;
        status.0 = "The knife slashes through the vegetation.".into();
    }
}

/// One axe hit on a felled trunk. It works exactly like a hit on the standing tree: the
/// outermost block of wood is chipped out of every column the swing touches (a small straight
/// cut first, deeper and wider when the same spot is hit again) and the layer underneath
/// becomes the new surface. When too little wood is left in one place the log snaps there and
/// falls into two pieces (each one can be chopped further). Branches go with their piece.
#[allow(clippy::too_many_arguments)]
fn chop_felled_log(
    log_entity: Entity,
    hit_world: Vec3,
    elapsed: f32,
    axe: &mut Axe,
    tree_queries: &mut TreeHitQueries,
    transforms: &Query<&GlobalTransform>,
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &Assets<StandardMaterial>,
    status: &mut Status,
) {
    let Ok(body_global) = transforms.get(log_entity) else {
        return;
    };
    let body_global = *body_global;
    let body_tf = body_global.compute_transform();
    let (mut wood, bark, mesh_handle, collider_entity, offset, center_offset) = {
        let Ok((_, log)) = tree_queries.choppable_trunks.get(log_entity) else {
            return;
        };
        (
            log.wood.clone(),
            log.bark.clone(),
            log.mesh.clone(),
            log.collider,
            log.offset,
            log.center_offset,
        )
    };
    let bark_color = mats
        .get(&bark)
        .map(|m| m.base_color)
        .unwrap_or(Color::WHITE);
    let cut_ratio = wood_cut_ratios(bark_color);
    let (body_linvel, body_angvel) = tree_queries
        .velocities
        .get(log_entity)
        .map(|v| (v.linvel, v.angvel))
        .unwrap_or((Vec3::ZERO, Vec3::ZERO));
    let com_world = body_tf.translation + body_tf.rotation * center_offset;
    // World transform of the wood grid's own coordinate system.
    let frame = Transform {
        translation: body_tf.translation + body_tf.rotation * offset,
        rotation: body_tf.rotation,
        scale: Vec3::ONE,
    };

    // ----- The hit: chips the outermost block out of every column it touches. -----
    let local = body_global.affine().inverse().transform_point3(hit_world);
    let (sector, row) = wood.locate_free(local - offset);
    let removed = wood.chop_min(sector, row, 0);
    axe.chop_cooldown = 0.4;
    let seed = elapsed.to_bits() ^ log_entity.index();
    for (n, &cell) in removed.iter().take(14).enumerate() {
        spawn_wood_frame_debris(
            commands,
            meshes,
            &bark,
            &frame,
            &wood,
            &[cell],
            &cut_ratio,
            body_linvel,
            body_angvel,
            com_world,
            seed.wrapping_add(n as u32 * 29),
            1.6,
            8.0,
        );
    }

    // ----- Does it snap here? Only when the cut has eaten most of the cross-section. -----
    let Some(weak) = wood.snap_row(row, LOG_MIN_PIECE_CELLS) else {
        if let Some(m) = meshes.get_mut(&mesh_handle) {
            *m = wood_mesh_from_data(wood.build_mesh(&wood.cells, Vec3::ZERO, false, &cut_ratio));
        }
        if let Some(collider) = wood.collider_of(&wood.cells, offset) {
            commands.entity(collider_entity).insert(collider);
        }
        let cut_percent = ((1.0 - wood.row_area(row as usize)) * 100.0).round();
        let spot_hits = wood.spot_hits;
        let nothing_removed = removed.is_empty();
        if let Ok((_, mut log)) = tree_queries.choppable_trunks.get_mut(log_entity) {
            log.wood = wood;
        }
        status.0 = if nothing_removed {
            "The cut already goes right through here; widen it or cut somewhere else.".into()
        } else if spot_hits <= 1 {
            "A small straight cut opens in the log. Keep chopping the same spot to go deeper."
                .into()
        } else {
            format!("Chop! The cut goes deeper: about {cut_percent:.0}% of the log is cut through.")
        };
        return;
    };

    // ----- Snap: the log breaks into two pieces, each with its own wood, mesh and collider. -----
    let per_row = WOOD_SECTORS * WOOD_LAYERS;
    let split = (weak + 1) * per_row;
    let lower_mask: Vec<bool> = wood
        .cells
        .iter()
        .enumerate()
        .map(|(i, &present)| present && i < split)
        .collect();
    let upper_mask: Vec<bool> = wood
        .cells
        .iter()
        .enumerate()
        .map(|(i, &present)| present && i >= split)
        .collect();

    let axis_up = (body_tf.rotation * Vec3::Y).normalize_or_zero();
    let inverse_rotation = body_tf.rotation.inverse();
    let mut spawned: Vec<(Entity, Vec3)> = Vec::new();
    for (index, mask) in [lower_mask, upper_mask].into_iter().enumerate() {
        let centroid = wood.centroid_of(&mask);
        let origin_world = body_tf.translation + body_tf.rotation * (centroid + offset);
        let piece_offset = -centroid;
        let piece_wood = TrunkWood {
            shape: wood.shape,
            rows: wood.rows,
            cells: mask.clone(),
            spot: None,
            spot_hits: 0,
        };
        let piece_mesh = meshes.add(wood_mesh_from_data(piece_wood.build_mesh(
            &mask,
            Vec3::ZERO,
            false,
            &cut_ratio,
        )));
        let collider = piece_wood
            .collider_of(&mask, piece_offset)
            .unwrap_or_else(|| Collider::ball(0.15));
        let piece_collider = commands.spawn((SpatialBundle::default(), collider)).id();
        let push = axis_up * if index == 0 { -0.25 } else { 0.25 };
        let linvel = body_linvel + body_angvel.cross(origin_world - com_world) + push;
        let mass = piece_wood.mass_of(&mask).max(5.0);
        let radius = piece_wood.shape.radius_at(centroid.y).max(0.05);
        let piece_entity = commands
            .spawn((
                SpatialBundle::from_transform(Transform {
                    translation: origin_world,
                    rotation: body_tf.rotation,
                    scale: Vec3::ONE,
                }),
                RigidBody::Dynamic,
                Velocity {
                    linvel,
                    angvel: body_angvel,
                },
                // Mass, mass read-out and force slot are what the grabbing hand needs.
                AdditionalMassProperties::Mass(mass),
                ReadMassProperties::default(),
                ExternalForce::default(),
                Damping {
                    linear_damping: 0.8,
                    angular_damping: 3.0,
                },
                Friction::coefficient(0.95),
                Restitution::coefficient(0.04),
                Ccd::enabled(),
                ChoppableTrunk {
                    radius,
                    center_offset: Vec3::ZERO,
                    bark: bark.clone(),
                    wood: piece_wood,
                    offset: piece_offset,
                    mesh: piece_mesh.clone(),
                    collider: piece_collider,
                },
            ))
            .id();
        commands.entity(piece_entity).add_child(piece_collider);
        commands.entity(piece_entity).with_children(|piece| {
            piece.spawn(PbrBundle {
                mesh: piece_mesh,
                material: bark.clone(),
                transform: Transform::from_translation(piece_offset),
                ..default()
            });
        });
        spawned.push((piece_entity, origin_world));
    }

    // Branches stay on the piece they grow from.
    let branches: Vec<(Entity, GlobalTransform)> = tree_queries
        .children
        .get(log_entity)
        .into_iter()
        .flat_map(|children| children.iter())
        .filter_map(|child| {
            tree_queries
                .felled_branches
                .get(*child)
                .ok()
                .and_then(|_| transforms.get(*child).ok())
                .map(|branch_transform| (*child, *branch_transform))
        })
        .collect();
    let inverse_body = body_global.affine().inverse();
    for (branch_entity, branch_global) in branches {
        let grid_y = inverse_body.transform_point3(branch_global.translation()).y - offset.y;
        let branch_row = (grid_y / WOOD_ROW_H).floor() as i32;
        let (piece_entity, origin_world) = spawned[usize::from(branch_row > weak as i32)];
        let world = branch_global.compute_transform();
        commands.entity(branch_entity).insert(Transform {
            translation: inverse_rotation * (world.translation - origin_world),
            rotation: inverse_rotation * world.rotation,
            scale: world.scale,
        });
        commands.entity(piece_entity).add_child(branch_entity);
    }

    commands.entity(log_entity).despawn_recursive();
    status.0 = "Crack! Too little wood is left at the cut and the log snaps in two.".into();
}

/// The upper trunk (its layered wood blocks, cut away from the stump) breaks free of the stump
/// and topples about a hinge at the back of the cut. The branches above the break go with
/// it; the stump stays in the ground as the original tree entity.
#[allow(clippy::too_many_arguments)]
fn fell_wood_trunk(
    tree_entity: Entity,
    base: Vec3,
    shape: TrunkShape,
    y_cut: f32,
    fall_dir: Vec3,
    log_wood: TrunkWood,
    upper_origin: Vec3,
    cut_ratio: &[[f32; 3]; WOOD_LAYERS],
    bark: Handle<StandardMaterial>,
    tree_queries: &mut TreeHitQueries,
    transforms: &Query<&GlobalTransform>,
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
) {
    let remaining = (shape.height - y_cut).max(0.8);
    let cut_center = shape.center_at(y_cut);
    let cut_radius = shape.radius_at(y_cut);
    let top_radius = shape.radius_at(y_cut + remaining * 0.5).max(0.1);
    let cut_world = base + Vec3::new(cut_center.x, y_cut, cut_center.y);

    let attached: Vec<(Entity, bool, f32)> = tree_queries
        .branches
        .iter_mut()
        .filter_map(|(entity, branch)| {
            (branch.tree == tree_entity).then_some((entity, branch.broken, branch.anchor.y - base.y))
        })
        .collect();

    // Hinge axis: horizontal, perpendicular to the fall direction.
    let fall_dir = fall_dir.normalize_or_zero();
    let fall_dir = if fall_dir.length_squared() < 0.0001 {
        Vec3::Z
    } else {
        fall_dir
    };
    let hinge_axis = Vec3::Y.cross(fall_dir).normalize_or_zero();
    let hinge_axis = if hinge_axis.length_squared() < 0.0001 {
        Vec3::X
    } else {
        hinge_axis
    };

    // The falling piece pivots about the back of the trunk (opposite the cut).
    let hinge_world = cut_world - fall_dir * cut_radius;
    let mesh_offset = fall_dir * cut_radius;
    // Wood-grid coordinates -> body-local coordinates.
    let log_offset = mesh_offset - upper_origin;
    let top_mesh = meshes.add(wood_mesh_from_data(log_wood.build_mesh(
        &log_wood.cells,
        Vec3::ZERO,
        false,
        cut_ratio,
    )));
    let log_collider = log_wood
        .collider_of(&log_wood.cells, log_offset)
        .unwrap_or_else(|| Collider::cylinder(remaining * 0.5, top_radius));
    let log_collider_entity = commands.spawn((SpatialBundle::default(), log_collider)).id();

    let mass = (std::f32::consts::PI * top_radius * top_radius * remaining * 450.0)
        .clamp(60.0, 500.0);
    let rod_inertia = mass * remaining * remaining / 12.0;
    let com_local = mesh_offset + Vec3::Y * (remaining * 0.5);
    let initial_angvel = 0.5_f32;

    let falling = commands
        .spawn((
            SpatialBundle::from_transform(Transform::from_translation(hinge_world)),
            RigidBody::Dynamic,
            Velocity {
                linvel: Vec3::ZERO,
                angvel: hinge_axis * initial_angvel,
            },
            Damping {
                linear_damping: 0.15,
                angular_damping: 0.6,
            },
            AdditionalMassProperties::MassProperties(MassProperties {
                local_center_of_mass: com_local,
                mass,
                principal_inertia: Vec3::splat(rod_inertia),
                principal_inertia_local_frame: Quat::IDENTITY,
            }),
            Friction::coefficient(0.95),
            Restitution::coefficient(0.04),
            Ccd::enabled(),
            ChoppableTrunk {
                radius: top_radius,
                center_offset: com_local,
                bark: bark.clone(),
                wood: log_wood,
                offset: log_offset,
                mesh: top_mesh.clone(),
                collider: log_collider_entity,
            },
        ))
        .with_children(|top| {
            // The whole upper trunk is ONE mesh: bark outside, layered wood where it
            // broke away. No separate cap discs, so nothing overlaps.
            top.spawn(PbrBundle {
                mesh: top_mesh,
                material: bark.clone(),
                transform: Transform::from_translation(log_offset),
                ..default()
            });
        })
        .id();
    commands.entity(falling).add_child(log_collider_entity);

    for (branch_entity, broken, anchor_height) in attached {
        // Branches below the break stay on the stump.
        if broken || anchor_height < y_cut - 0.35 {
            continue;
        }
        let Ok(branch_tf) = transforms.get(branch_entity) else {
            continue;
        };
        let world = branch_tf.compute_transform();
        commands
            .entity(branch_entity)
            .remove::<RigidBody>()
            .remove::<Velocity>()
            .remove::<GravityScale>()
            .remove::<Sensor>()
            .remove::<TreeLodPiece>()
            .remove::<TreeBranch>()
            .insert(FelledBranch { detached: false })
            .insert(Transform {
                translation: world.translation - hinge_world,
                rotation: world.rotation,
                scale: world.scale,
            });
        commands.entity(falling).add_child(branch_entity);
        if let Some(branch_shape) = tree_queries.shapes.get(branch_entity).ok().cloned() {
            commands.entity(branch_entity).despawn_descendants();
            attach_branch_parts(commands, branch_entity, &branch_shape, meshes, false);
        }
    }
}

/// Builds the wood / leaf meshes and the colliders for a branch and attaches them as children.
/// `sensor` colliders can be walked through (branches still on a tree / trunk); solid ones
/// are used for branches that have been cut free and lie on the ground.
fn attach_branch_parts(
    commands: &mut Commands,
    entity: Entity,
    shape: &BranchShape,
    meshes: &mut Assets<Mesh>,
    sensor: bool,
) {
    let branch_mesh = unit_oak_cylinder_mesh();

    let mut wood_batch = OakMeshBatch::new();
    let mut carved_meshes: Vec<(Handle<Mesh>, Transform)> = Vec::new();
    for (segment_index, segment) in shape.segments.iter().enumerate() {
        let delta = segment.end - segment.start;
        // A limb the axe has hit is made of layered wood blocks instead of a plain cylinder.
        if let Some(Some(limb_wood)) = shape.carved.get(segment_index) {
            let mesh = meshes.add(wood_mesh_from_data(limb_wood.build_mesh(
                &limb_wood.cells,
                Vec3::ZERO,
                false,
                &shape.cut_ratio,
            )));
            carved_meshes.push((
                mesh,
                Transform::from_translation(segment.start)
                    .with_rotation(Quat::from_rotation_arc(Vec3::Y, delta.normalize())),
            ));
            continue;
        }
        wood_batch.append(
            &branch_mesh,
            Transform::from_translation((segment.start + segment.end) * 0.5)
                .with_rotation(Quat::from_rotation_arc(Vec3::Y, delta.normalize()))
                .with_scale(Vec3::new(segment.radius, delta.length(), segment.radius)),
        );
    }
    let wood_mesh = meshes.add(wood_batch.finish());

    commands.entity(entity).with_children(|branch| {
        for segment in &shape.segments {
            let delta = segment.end - segment.start;
            let midpoint = (segment.start + segment.end) * 0.5;
            let mut collider = branch.spawn((
                SpatialBundle::from_transform(
                    Transform::from_translation(midpoint)
                        .with_rotation(Quat::from_rotation_arc(Vec3::Y, delta.normalize())),
                ),
                Collider::capsule_y(delta.length() * 0.5, segment.radius.max(0.06)),
            ));
            if sensor {
                collider.insert(Sensor);
            }
        }
        if !sensor {
            for (tip, _, _) in &shape.tips {
                branch.spawn((
                    SpatialBundle::from_transform(Transform::from_translation(*tip)),
                    Collider::ball(0.3),
                ));
            }
        }
        branch.spawn(PbrBundle {
            mesh: wood_mesh,
            material: shape.wood.clone(),
            ..default()
        });
        for (mesh, transform) in carved_meshes {
            branch.spawn(PbrBundle {
                mesh,
                material: shape.wood.clone(),
                transform,
                ..default()
            });
        }
    });
    attach_branch_foliage(commands, entity, shape, meshes);
}

/// One oak leaf on a branch. Leaves are not entities: the ones of a branch are merged into a
/// few meshes, and this is where each one is (branch-local) so that one can be found and pulled.
struct OakLeaf {
    /// Never changes for a given leaf (from the tip's seed and the leaf's number in its cluster).
    id: u64,
    material_index: usize,
    position: Vec3,
    rotation: Quat,
    scale: f32,
    phase: f32,
}

/// Every leaf the branch has grown, including the ones that have been pulled off.
fn oak_leaf_records(shape: &BranchShape) -> Vec<OakLeaf> {
    let mut leaves = Vec::new();
    for (tip_index, (tip, tip_seed, _)) in shape.tips.iter().enumerate() {
        let cluster_size = 10 + (pseudo_rand(*tip_seed) * 7.0) as usize;
        for leaf_index in 0..cluster_size {
            let leaf_seed = tip_seed
                .wrapping_mul(13)
                .wrapping_add(leaf_index as u32 * 7);
            let random_dir = Vec3::new(
                pseudo_rand(leaf_seed) * 2.0 - 1.0,
                pseudo_rand(leaf_seed.wrapping_add(1)) * 2.0 - 1.0,
                pseudo_rand(leaf_seed.wrapping_add(2)) * 2.0 - 1.0,
            )
            .normalize_or_zero();
            let position =
                *tip + random_dir * (0.22 + pseudo_rand(leaf_seed.wrapping_add(3)) * 0.24);
            let angles = Vec3::new(
                pseudo_rand(leaf_seed.wrapping_add(4)) * std::f32::consts::TAU,
                pseudo_rand(leaf_seed.wrapping_add(5)) * std::f32::consts::TAU,
                pseudo_rand(leaf_seed.wrapping_add(6)) * std::f32::consts::TAU,
            );
            leaves.push(OakLeaf {
                id: ((*tip_seed as u64) << 32) | leaf_index as u64,
                material_index: ((pseudo_rand(leaf_seed.wrapping_add(7)) * 3.0) as usize).min(2),
                position,
                rotation: Quat::from_euler(EulerRot::XYZ, angles.x, angles.y, angles.z),
                scale: 0.24 + pseudo_rand(leaf_seed.wrapping_add(8)) * 0.14,
                phase: (tip_index * 37 + leaf_index) as f32 * 0.13
                    + pseudo_rand(leaf_seed.wrapping_add(9)) * std::f32::consts::TAU,
            });
        }
    }
    leaves
}

/// Builds the leaf meshes of a branch (without the leaves that were pulled off) and attaches
/// them as children. Call it again after pulling a leaf, once the old ones are despawned.
fn attach_branch_foliage(
    commands: &mut Commands,
    entity: Entity,
    shape: &BranchShape,
    meshes: &mut Assets<Mesh>,
) {
    let leaf_mesh = oak_leaf_mesh();
    let mut centers = [Vec3::ZERO; 3];
    let mut counts = [0_u32; 3];
    let mut phases = [0.0_f32; 3];
    let leaves: Vec<OakLeaf> = oak_leaf_records(shape)
        .into_iter()
        .filter(|leaf| !shape.plucked.contains(&leaf.id))
        .collect();
    for leaf in &leaves {
        centers[leaf.material_index] += leaf.position;
        counts[leaf.material_index] += 1;
        phases[leaf.material_index] += leaf.phase;
    }
    for material_index in 0..3 {
        if counts[material_index] > 0 {
            centers[material_index] /= counts[material_index] as f32;
            phases[material_index] /= counts[material_index] as f32;
        }
    }
    let mut foliage_batches: [OakMeshBatch; 3] = std::array::from_fn(|_| OakMeshBatch::new());
    for leaf in &leaves {
        foliage_batches[leaf.material_index].append(
            &leaf_mesh,
            Transform::from_translation(leaf.position - centers[leaf.material_index])
                .with_rotation(leaf.rotation)
                .with_scale(Vec3::splat(leaf.scale)),
        );
    }
    let foliage_meshes: [Option<Handle<Mesh>>; 3] = std::array::from_fn(|material_index| {
        (counts[material_index] > 0).then(|| {
            meshes.add(
                std::mem::replace(&mut foliage_batches[material_index], OakMeshBatch::new())
                    .finish(),
            )
        })
    });
    commands.entity(entity).with_children(|branch| {
        for material_index in 0..3 {
            if let Some(mesh) = foliage_meshes[material_index].clone() {
                branch.spawn((
                    PbrBundle {
                        mesh,
                        material: shape.foliage[material_index].clone(),
                        transform: Transform::from_translation(centers[material_index]),
                        ..default()
                    },
                    LeafSway {
                        phase: phases[material_index],
                        rest_rotation: Quat::IDENTITY,
                    },
                ));
            }
        }
    });
}

/// A leaf that has been pulled off a branch: a tiny, light physics object. It flutters down
/// when you let go, and you can pick it up again like anything else.
#[derive(Component)]
struct LooseLeaf {
    born: f32,
}

/// How many loose leaves may lie around at once (the oldest are removed beyond this).
const MAX_LOOSE_LEAVES: usize = 300;
/// How far you can reach to pull a leaf (metres from the camera).
const LEAF_REACH: f32 = 3.5;

/// Click a leaf on a tree to pull it off: it comes out of the branch and follows your hand
/// (hold and drag), and falls when you let go. Aim at the wood instead to grab the branch.
fn leaf_pluck_system(
    mut commands: Commands,
    tools: ToolsHeld,
    mut consumed: ResMut<ClickConsumed>,
    mut grab: ResMut<Grab>,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    rapier: Res<RapierContext>,
    player: Query<Entity, With<Player>>,
    mut branches: Query<(Entity, &GlobalTransform, &mut BranchShape)>,
    leaf_nodes: Query<(Entity, &Parent), With<LeafSway>>,
    mut meshes: ResMut<Assets<Mesh>>,
    loose: Query<(Entity, &LooseLeaf)>,
    time: Res<Time>,
    mut leaf_mesh: Local<Option<Handle<Mesh>>>,
) {
    if !buttons.just_pressed(MouseButton::Left)
        || buttons.pressed(MouseButton::Right)
        || consumed.0
        || grab.entity.is_some()
        || tools.any()
    {
        return;
    }
    let Ok(window) = windows.get_single() else {
        return;
    };
    let Ok((camera, cam_gt)) = cameras.get_single() else {
        return;
    };
    let Some(ray) = pointer_position(&window)
        .and_then(|cursor| camera.viewport_to_world(cam_gt, cursor))
    else {
        return;
    };
    let dir: Vec3 = *ray.direction;

    // anything solid in front of the leaf (a trunk, a wall) hides it
    let filter = world_ray_filter(player.get_single().ok()).exclude_sensors();
    let solid = rapier
        .cast_ray(ray.origin, dir, LEAF_REACH, true, filter)
        .map(|(_, toi)| toi)
        .unwrap_or(LEAF_REACH);

    // the leaf nearest to the cursor ray
    let mut best: Option<(f32, Entity, usize)> = None; // (distance along the ray, branch, leaf)
    let mut best_leaf: Option<OakLeaf> = None;
    for (branch_e, gt, shape) in &branches {
        let (_, branch_rot, branch_pos) = gt.to_scale_rotation_translation();
        if branch_pos.distance(ray.origin) > LEAF_REACH + 3.5 {
            continue;
        }
        for leaf in oak_leaf_records(&shape) {
            if shape.plucked.contains(&leaf.id) {
                continue;
            }
            // the middle of the leaf (its mesh runs 1.35 along +Y from the stem)
            let centre = branch_pos + branch_rot * (leaf.position + leaf.rotation * (Vec3::Y * 0.675 * leaf.scale));
            let to = centre - ray.origin;
            let along = to.dot(dir);
            if along < 0.0 || along > solid.min(LEAF_REACH) {
                continue;
            }
            let miss = (to - dir * along).length();
            if miss < 0.45 * leaf.scale + 0.02 && best.map_or(true, |b| along < b.0) {
                best = Some((along, branch_e, 0));
                best_leaf = Some(leaf);
            }
        }
    }
    let (Some((along, branch_e, _)), Some(leaf)) = (best, best_leaf) else {
        return; // no leaf there: the click goes on to the normal grab
    };
    let Ok((_, gt, mut shape)) = branches.get_mut(branch_e) else {
        return;
    };
    let (_, branch_rot, branch_pos) = gt.to_scale_rotation_translation();
    let world_pos = branch_pos + branch_rot * leaf.position;
    let world_rot = branch_rot * leaf.rotation;

    // take it out of the branch: rebuild the branch's leaf meshes without it
    shape.plucked.push(leaf.id);
    for (node, parent) in &leaf_nodes {
        if parent.get() == branch_e {
            commands.entity(node).despawn_recursive();
        }
    }
    attach_branch_foliage(&mut commands, branch_e, &shape, &mut meshes);

    // ... and make it a loose leaf that the hand is already holding
    let mesh = leaf_mesh.get_or_insert_with(|| meshes.add(oak_leaf_mesh())).clone();
    let leaf_tf = Transform::from_translation(world_pos)
        .with_rotation(world_rot)
        .with_scale(Vec3::splat(leaf.scale));
    let leaf_e = commands
        .spawn((
            PbrBundle {
                mesh,
                material: shape.foliage[leaf.material_index].clone(),
                transform: leaf_tf,
                // right from the start, so the hand grabs it where it really is
                global_transform: GlobalTransform::from(leaf_tf),
                ..default()
            },
            LooseLeaf {
                born: time.elapsed_seconds(),
            },
            RigidBody::Dynamic,
            // a thin plate over the leaf (the collider ignores the visual scale)
            Collider::compound(vec![(
                Vec3::new(0.0, 0.675, 0.0),
                Quat::IDENTITY,
                Collider::cuboid(0.3, 0.675, 0.012),
            )]),
            ColliderMassProperties::Mass(0.004),
            Velocity::default(),
            ExternalForce::default(),
            ReadMassProperties::default(),
            // air drag makes it flutter down instead of dropping like a stone
            Damping {
                linear_damping: 1.6,
                angular_damping: 1.5,
            },
            Friction::coefficient(0.9),
            Sleeping::disabled(),
        ))
        .id();
    grab.entity = Some(leaf_e);
    grab.local_anchor = Vec3::new(0.0, 0.05, 0.0) / leaf.scale;
    grab.distance = along.clamp(0.5, 20.0);
    grab.bail_reach = 0.0;
    consumed.0 = true; // this click was spent on the leaf

    // keep the number of loose leaves down
    let mut all: Vec<(Entity, f32)> = loose.iter().map(|(e, l)| (e, l.born)).collect();
    if all.len() >= MAX_LOOSE_LEAVES {
        all.sort_by(|a, b| a.1.total_cmp(&b.1));
        for (e, _) in all.iter().take(all.len() + 1 - MAX_LOOSE_LEAVES) {
            if Some(*e) != grab.entity {
                commands.entity(*e).despawn_recursive();
            }
        }
    }
}

/// Which limb of the branch is closest to `point` (branch-local), and how far along it (0..1).
fn nearest_branch_segment(shape: &BranchShape, point: Vec3) -> Option<(usize, f32)> {
    shape
        .segments
        .iter()
        .enumerate()
        .map(|(index, segment)| {
            let delta = segment.end - segment.start;
            let u = ((point - segment.start).dot(delta) / delta.length_squared().max(1e-6))
                .clamp(0.0, 1.0);
            let distance = (segment.start + delta * u).distance(point) - segment.radius;
            (index, u, distance)
        })
        .min_by(|a, b| a.2.total_cmp(&b.2))
        .map(|(index, u, _)| (index, u))
}

/// Cuts a branch across limb `cut_index` at fraction `u` of that limb; the limb's layered
/// wood is split at `cut_row` (so `u * limb length == cut_row * WOOD_ROW_H`).
/// Returns (the part that stays on the trunk, the part that falls off, cut point, limb direction).
/// The falling part is re-centred so that the cut point is its local origin.
/// Limbs are stored parent-before-child, so one forward pass is enough: a side limb that
/// sprouts beyond the cut leaves with the falling part, one that sprouts before it stays.
fn split_branch_shape(
    shape: &BranchShape,
    cut_index: usize,
    u: f32,
    cut_row: usize,
) -> (BranchShape, BranchShape, Vec3, Vec3) {
    let count = shape.segments.len();
    let cut = shape.segments[cut_index];
    let cut_point = cut.start.lerp(cut.end, u);
    let direction = (cut.end - cut.start).normalize_or_zero();
    let carved_of = |index: usize| shape.carved.get(index).cloned().flatten();

    let mut falls = vec![false; count];
    for index in 0..count {
        if let Some(parent) = shape.segments[index].parent {
            falls[index] = if parent == cut_index {
                u < OAK_BRANCH_SPLIT
            } else {
                falls[parent]
            };
        }
    }

    // The cut limb's wood breaks into a lower and an upper half.
    let (inner_cut_wood, outer_cut_wood) = match carved_of(cut_index) {
        Some(limb_wood) => {
            let (lower, upper) = limb_wood.split_at_row(cut_row);
            (Some(lower), Some(upper))
        }
        None => (None, None),
    };

    let mut inner_map = vec![usize::MAX; count];
    let mut outer_map = vec![usize::MAX; count];
    let mut inner_segments: Vec<OakBranchSegment> = Vec::new();
    let mut inner_carved: Vec<Option<TrunkWood>> = Vec::new();
    let mut outer_segments = vec![OakBranchSegment {
        start: Vec3::ZERO,
        end: cut.end - cut_point,
        radius: cut.radius,
        parent: None,
    }];
    let mut outer_carved: Vec<Option<TrunkWood>> = vec![outer_cut_wood];
    outer_map[cut_index] = 0;
    for index in 0..count {
        let segment = shape.segments[index];
        if index == cut_index {
            inner_map[index] = inner_segments.len();
            inner_segments.push(OakBranchSegment {
                end: cut_point,
                parent: segment.parent.map(|parent| inner_map[parent]),
                ..segment
            });
            inner_carved.push(inner_cut_wood.clone());
        } else if falls[index] {
            outer_map[index] = outer_segments.len();
            outer_segments.push(OakBranchSegment {
                start: segment.start - cut_point,
                end: segment.end - cut_point,
                radius: segment.radius,
                parent: segment.parent.map(|parent| outer_map[parent]),
            });
            outer_carved.push(carved_of(index));
        } else {
            inner_map[index] = inner_segments.len();
            inner_segments.push(OakBranchSegment {
                parent: segment.parent.map(|parent| inner_map[parent]),
                ..segment
            });
            inner_carved.push(carved_of(index));
        }
    }

    let mut inner_tips = Vec::new();
    let mut outer_tips = Vec::new();
    for &(position, seed, segment_index) in &shape.tips {
        if segment_index == cut_index || falls[segment_index] {
            outer_tips.push((position - cut_point, seed, outer_map[segment_index]));
        } else {
            inner_tips.push((position, seed, inner_map[segment_index]));
        }
    }

    let inner = BranchShape {
        segments: inner_segments,
        tips: inner_tips,
        wood: shape.wood.clone(),
        foliage: shape.foliage.clone(),
        carved: inner_carved,
        cut_ratio: shape.cut_ratio,
        plucked: shape.plucked.clone(),
    };
    let outer = BranchShape {
        segments: outer_segments,
        tips: outer_tips,
        wood: shape.wood.clone(),
        foliage: shape.foliage.clone(),
        carved: outer_carved,
        cut_ratio: shape.cut_ratio,
        plucked: shape.plucked.clone(),
    };
    (inner, outer, cut_point, direction)
}

/// Rough mass of a cut-off branch: green oak wood plus a little for the leaves.
fn branch_mass(shape: &BranchShape) -> f32 {
    let wood: f32 = shape
        .segments
        .iter()
        .map(|segment| {
            std::f32::consts::PI
                * segment.radius
                * segment.radius
                * (segment.end - segment.start).length()
                * 500.0
        })
        .sum();
    (wood + shape.tips.len() as f32 * 0.2).max(0.4)
}

#[allow(dead_code)]
fn spawn_wood_chips(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &mut Assets<StandardMaterial>,
    assets: &mut Option<(Handle<Mesh>, Handle<StandardMaterial>)>,
    point: Vec3,
    hit_seed: u32,
    count: u32,
) {
    let (chip_mesh, chip_material) = assets.get_or_insert_with(|| {
        (
            meshes.add(Cuboid::new(0.12, 0.12, 0.28)),
            mats.add(Color::srgb(0.62, 0.4, 0.21)),
        )
    });
    for chip_index in 0..count {
        let seed = hit_seed.wrapping_add(chip_index * 23);
        let direction = Vec3::new(
            pseudo_rand(seed) - 0.5,
            0.3 + pseudo_rand(seed.wrapping_add(1)) * 0.8,
            pseudo_rand(seed.wrapping_add(2)) - 0.5,
        )
        .normalize_or_zero();
        commands.spawn((
            PbrBundle {
                mesh: chip_mesh.clone(),
                material: chip_material.clone(),
                transform: Transform::from_translation(point)
                    .with_rotation(Quat::from_euler(
                        EulerRot::XYZ,
                        pseudo_rand(seed.wrapping_add(3)) * 3.0,
                        pseudo_rand(seed.wrapping_add(4)) * 3.0,
                        pseudo_rand(seed.wrapping_add(5)) * 3.0,
                    ))
                    .with_scale(Vec3::splat(0.2 + pseudo_rand(seed.wrapping_add(6)) * 0.3)),
                ..default()
            },
            RigidBody::Dynamic,
            Collider::cuboid(0.06, 0.06, 0.14),
            Velocity {
                linvel: direction * (1.5 + pseudo_rand(seed.wrapping_add(7)) * 1.8),
                angvel: Vec3::new(
                    pseudo_rand(seed.wrapping_add(8)) * 8.0,
                    pseudo_rand(seed.wrapping_add(9)) * 8.0,
                    pseudo_rand(seed.wrapping_add(10)) * 8.0,
                ),
            },
            TreeChip { remaining: 1.5 },
        ));
    }
}

/// One axe hit on a branch of a felled tree. It works like a hit on the standing tree: the
/// limb you hit turns into layered wood blocks, every swing chips the outer layer off the spot
/// (deeper and wider when you keep hitting the same place), and when too little wood is left
/// there the branch snaps: everything beyond the cut (the end of that limb and every side limb
/// that sprouts beyond it, with their leaves) becomes a free physics object; everything nearer
/// the trunk stays attached to the felled trunk.
#[allow(clippy::too_many_arguments)]
fn chop_felled_branch(
    branch_entity: Entity,
    hit_world: Vec3,
    elapsed: f32,
    axe: &mut Axe,
    tree_queries: &mut TreeHitQueries,
    transforms: &Query<&GlobalTransform>,
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &Assets<StandardMaterial>,
    status: &mut Status,
) {
    let Ok(branch_global) = transforms.get(branch_entity) else {
        return;
    };
    let branch_global = *branch_global;
    let Some(mut shape) = tree_queries.shapes.get(branch_entity).ok().cloned() else {
        return;
    };
    let local_hit = branch_global.affine().inverse().transform_point3(hit_world);
    let Some((segment_index, _)) = nearest_branch_segment(&shape, local_hit) else {
        return;
    };
    let segment = shape.segments[segment_index];
    let segment_vector = segment.end - segment.start;
    let segment_len = segment_vector.length();
    if segment_len < 0.05 {
        return;
    }
    let segment_rotation = Quat::from_rotation_arc(Vec3::Y, segment_vector / segment_len);

    // The body this branch belongs to (the felled trunk, or the branch itself once it is free).
    let mut body = None;
    let mut walker = branch_entity;
    loop {
        if tree_queries.velocities.get(walker).is_ok() {
            body = Some(walker);
            break;
        }
        match tree_queries.parents.get(walker) {
            Ok(parent) => walker = parent.get(),
            Err(_) => break,
        }
    }
    let trunk_axis = body.and_then(|body| {
        let (_, trunk) = tree_queries.choppable_trunks.get(body).ok()?;
        let (_, rotation, position) = transforms.get(body).ok()?.to_scale_rotation_translation();
        Some((
            position + rotation * trunk.center_offset,
            (rotation * Vec3::Y).normalize_or_zero(),
            trunk.radius,
        ))
    });
    // Distance of a branch-local point from the trunk's axis (None when the branch is free).
    let axis_clearance = |point: Vec3| -> Option<(f32, f32)> {
        let (axis_point, axis_dir, trunk_radius) = trunk_axis?;
        let offset = branch_global.transform_point(point) - axis_point;
        Some((
            (offset - axis_dir * offset.dot(axis_dir)).length(),
            trunk_radius * 1.1 + 0.05,
        ))
    };
    if let Some((distance, needed)) = axis_clearance(local_hit) {
        if distance < needed {
            status.0 = "Too close to the trunk to chop here; chop further out on the branch.".into();
            return;
        }
    }
    axe.chop_cooldown = 0.4;

    let (body_linvel, body_angvel, body_position) = body
        .and_then(|body| {
            let velocity = tree_queries.velocities.get(body).ok()?;
            let position = transforms.get(body).ok()?.translation();
            Some((velocity.linvel, velocity.angvel, position))
        })
        .unwrap_or((Vec3::ZERO, Vec3::ZERO, hit_world));

    // ----- The limb turns into layered wood (if it isn't already) and the hit chips it. -----
    let wood_color = mats
        .get(&shape.wood)
        .map(|m| m.base_color)
        .unwrap_or(Color::WHITE);
    shape.cut_ratio = wood_cut_ratios(wood_color);
    if shape.carved.len() != shape.segments.len() {
        shape.carved.resize(shape.segments.len(), None);
    }
    let mut wood = shape.carved[segment_index].take().unwrap_or_else(|| {
        TrunkWood::new(TrunkShape {
            height: segment_len,
            base_radius: segment.radius,
            top_radius: segment.radius,
            lean: Vec2::ZERO,
        })
    });
    let on_limb = segment_rotation.inverse() * (local_hit - segment.start);
    let (sector, row) = wood.locate_free(on_limb);
    let removed = wood.chop_min(sector, row, 0);

    let branch_tf = branch_global.compute_transform();
    let frame = Transform {
        translation: branch_tf.translation
            + branch_tf.rotation * (branch_tf.scale * segment.start),
        rotation: branch_tf.rotation * segment_rotation,
        scale: Vec3::ONE,
    };
    let seed = elapsed.to_bits() ^ branch_entity.index();
    for (n, &cell) in removed.iter().take(8).enumerate() {
        spawn_wood_frame_debris(
            commands,
            meshes,
            &shape.wood,
            &frame,
            &wood,
            &[cell],
            &shape.cut_ratio,
            body_linvel,
            body_angvel,
            body_position,
            seed.wrapping_add(n as u32 * 29),
            1.2,
            6.0,
        );
    }

    // ----- Does it snap here? Only when the cut has eaten most of the limb's cross-section. -----
    let mut snap: Option<(usize, f32)> = None;
    if let Some(weak) = wood.snap_row(row, BRANCH_MIN_PIECE_CELLS) {
        let cut_row = weak + 1;
        let u = (cut_row as f32 * WOOD_ROW_H / segment_len).clamp(0.01, 0.99);
        let far_enough = axis_clearance(segment.start.lerp(segment.end, u))
            .map_or(true, |(distance, needed)| distance >= needed);
        if far_enough {
            snap = Some((cut_row, u));
        }
    }

    let cut_percent = ((1.0 - wood.row_area(row as usize)) * 100.0).round();
    let nothing_removed = removed.is_empty();
    shape.carved[segment_index] = Some(wood);

    let Some((cut_row, u)) = snap else {
        // Show the deeper cut: rebuild the branch with the carved limb.
        commands.entity(branch_entity).despawn_descendants();
        commands.entity(branch_entity).insert(shape.clone());
        attach_branch_parts(commands, branch_entity, &shape, meshes, false);
        status.0 = if nothing_removed {
            "The cut already goes right through here; widen it or cut somewhere else.".into()
        } else {
            format!("Chop! The cut goes deeper: about {cut_percent:.0}% of the branch is cut through here.")
        };
        return;
    };

    // ----- Snap: the part beyond the cut becomes a free physics object. -----
    let (inner, outer, cut_point, direction) = split_branch_shape(&shape, segment_index, u, cut_row);
    let (_, world_rotation, _) = branch_global.to_scale_rotation_translation();
    let cut_world = branch_global.transform_point(cut_point);
    let outward = (world_rotation * direction).normalize_or_zero();
    let detached = tree_queries
        .felled_branches
        .get(branch_entity)
        .map(|branch| branch.detached)
        .unwrap_or(false);
    let body_position = if body.is_some() { body_position } else { cut_world };

    let outer_entity = commands
        .spawn((
            SpatialBundle::from_transform(Transform {
                translation: cut_world,
                rotation: world_rotation,
                scale: Vec3::ONE,
            }),
            RigidBody::Dynamic,
            Velocity {
                linvel: body_linvel + body_angvel.cross(cut_world - body_position) + outward * 0.35,
                angvel: body_angvel,
            },
            Damping {
                linear_damping: 0.4,
                angular_damping: 1.5,
            },
            AdditionalMassProperties::Mass(branch_mass(&outer)),
            ReadMassProperties::default(),
            ExternalForce::default(),
            Friction::coefficient(0.9),
            Restitution::coefficient(0.05),
            Ccd::enabled(),
            (FelledBranch { detached: true }, outer.clone()),
        ))
        .id();
    attach_branch_parts(commands, outer_entity, &outer, meshes, false);

    commands.entity(branch_entity).despawn_descendants();
    commands.entity(branch_entity).insert(inner.clone());
    if detached {
        commands
            .entity(branch_entity)
            .insert(AdditionalMassProperties::Mass(branch_mass(&inner)));
    }
    attach_branch_parts(commands, branch_entity, &inner, meshes, false);

    status.0 =
        "Crack! Too little wood is left at the cut: the end falls off, the rest stays on the trunk."
            .into();
}

fn reset_click_consumed(mut consumed: ResMut<ClickConsumed>) {
    consumed.0 = false;
}

/// Click a person to start or continue a conversation; Esc closes it.
fn npc_interact_system(
    player: Query<Entity, With<Player>>,
    mut consumed: ResMut<ClickConsumed>,
    mut dialogue: ResMut<DialogueState>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    rapier: Res<RapierContext>,
    mut npcs: Query<&mut Npc>,
) {
    if keys.just_pressed(KeyCode::Escape) {
        dialogue.entity = None;
        dialogue.focus_active = false;
    }
    if !buttons.just_pressed(MouseButton::Left) || consumed.0 {
        return;
    }
    let Ok(window) = windows.get_single() else {
        return;
    };
    let Ok((camera, cam_tf)) = cameras.get_single() else {
        return;
    };
    let Some(ray) = pointer_position(&window)
        .and_then(|cursor| camera.viewport_to_world(cam_tf, cursor))
    else {
        return;
    };
    let filter = world_ray_filter(player.get_single().ok());
    if let Some((entity, _toi)) = rapier.cast_ray(ray.origin, *ray.direction, 100.0, true, filter) {
        if let Ok(mut npc) = npcs.get_mut(entity) {
            if dialogue.entity != Some(entity) {
                npc.index = 0;
            } else if !npc.lines.is_empty() {
                npc.index = (npc.index + 1) % npc.lines.len();
            }
            dialogue.entity = Some(entity);
            dialogue.focus_active = true;
            consumed.0 = true;
            return;
        }
    }
    dialogue.entity = None;
    dialogue.focus_active = false;
}

fn horse_interact_system(
    player_entity: Query<Entity, With<Player>>,
    mut player: Query<(&mut Player, &mut Transform), With<Player>>,
    mut consumed: ResMut<ClickConsumed>,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    rapier: Res<RapierContext>,
    horses: Query<&Transform, (With<Horse>, Without<Player>)>,
    horse_backs: Query<Entity, With<HorseBack>>,
    parents: Query<&Parent>,
) {
    if !buttons.just_pressed(MouseButton::Left) || consumed.0 {
        return;
    }
    let Ok(window) = windows.get_single() else {
        return;
    };
    let Ok((camera, camera_transform)) = cameras.get_single() else {
        return;
    };
    let Some(ray) = pointer_position(&window)
        .and_then(|cursor| camera.viewport_to_world(camera_transform, cursor))
    else {
        return;
    };
    let player_entity = player_entity.get_single().ok();
    let filter = world_ray_filter(player_entity);
    let Some((hit, _distance)) = rapier.cast_ray(ray.origin, *ray.direction, 100.0, true, filter)
    else {
        return;
    };
    if !horse_backs.contains(hit) {
        return;
    }
    let Ok(parent) = parents.get(hit) else {
        return;
    };
    let horse = parent.get();
    let Ok(horse_transform) = horses.get(horse) else {
        return;
    };
    let Ok((mut rider, mut rider_transform)) = player.get_single_mut() else {
        return;
    };
    rider.mounted = Some(horse);
    rider.yaw = horse_transform.rotation.to_euler(EulerRot::YXZ).0;
    rider.vy = 0.0;
    rider_transform.translation = horse_transform.translation + Vec3::Y * 2.8;
    consumed.0 = true;
}

fn update_dialogue_ui(
    dialogue: Res<DialogueState>,
    npcs: Query<&Npc>,
    mut q: Query<&mut Text, With<DialogueBox>>,
) {
    let text = match dialogue.entity.and_then(|e| npcs.get(e).ok()) {
        Some(npc) if !npc.lines.is_empty() => {
            format!(
                "{}: \"{}\"  (click again to continue - Esc to close)",
                npc.name, npc.lines[npc.index]
            )
        }
        _ => String::new(),
    };
    if let Ok(mut t) = q.get_single_mut() {
        if t.sections[0].value != text {
            t.sections[0].value = text;
        }
    }
}

fn update_dialogue_focus(
    time: Res<Time>,
    mut dialogue: ResMut<DialogueState>,
    windows: Query<&Window, With<PrimaryWindow>>,
    npcs: Query<&GlobalTransform, With<Npc>>,
    mut cameras: Query<(&Camera, &GlobalTransform, &mut DialogueFocusSettings)>,
) {
    let Ok(window) = windows.get_single() else {
        return;
    };
    let Ok((camera, camera_transform, mut focus)) = cameras.get_single_mut() else {
        return;
    };

    let mut target_amount: f32 = 0.0;
    let target_center = Vec2::splat(0.5);
    if dialogue.focus_active {
        if let Some(entity) = dialogue.entity {
            if let Ok(npc_transform) = npcs.get(entity) {
                let point = npc_transform.translation() + Vec3::Y * 0.35;
                let in_front = camera_transform
                    .forward()
                    .dot(point - camera_transform.translation())
                    > 0.0;
                if in_front {
                    if let Some(screen_point) = camera.world_to_viewport(camera_transform, point) {
                        let speaker_uv = Vec2::new(
                            screen_point.x / window.width().max(1.0),
                            screen_point.y / window.height().max(1.0),
                        );
                        let circle_distance = ((speaker_uv - Vec2::splat(0.5))
                            * Vec2::new(focus.aspect, 1.0))
                        .length();
                        if screen_point.x >= 0.0
                            && screen_point.x <= window.width()
                            && screen_point.y >= 0.0
                            && screen_point.y <= window.height()
                            && circle_distance <= focus.radius
                        {
                            target_amount = 1.0;
                        } else {
                            dialogue.focus_active = false;
                        }
                    } else {
                        dialogue.focus_active = false;
                    }
                } else {
                    dialogue.focus_active = false;
                }
            } else {
                dialogue.focus_active = false;
            }
        }
    }

    let blend = 1.0 - (-time.delta_seconds().min(0.1) * 3.5).exp();
    let zoom_target = if target_amount == 0.0 && focus.blur_amount > 0.001 {
        1.0
    } else {
        target_amount
    };
    focus.amount += (zoom_target - focus.amount) * blend;
    let blur_target = if target_amount > 0.0 && focus.amount >= 0.99 {
        1.0
    } else {
        0.0
    };
    focus.blur_amount += (blur_target - focus.blur_amount) * blend;
    focus.center = focus.center.lerp(target_center, blend);
    let target_magnification = if target_amount > 0.0 { 1.2 } else { 1.0 };
    focus.magnification += (target_magnification - focus.magnification) * blend;
    focus.aspect = window.width() / window.height().max(1.0);
}

/// Turn an image into a 3D physics object and drop it into the world.
fn spawn_relief(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    mats: &mut Assets<StandardMaterial>,
    img: &RgbaImage,
    transform: Transform,
) {
    let cell = OBJECT_SIZE / img.width().max(img.height()) as f32;
    let (mesh, points) = relief_mesh(img, cell);
    let collider = Collider::convex_hull(&points)
        .unwrap_or_else(|| Collider::cuboid(OBJECT_SIZE / 2.0, OBJECT_SIZE / 2.0, 0.15));

    commands.spawn((
        PbrBundle {
            mesh: meshes.add(mesh),
            material: mats.add(StandardMaterial {
                base_color: Color::WHITE,
                perceptual_roughness: 0.7,
                ..default()
            }),
            transform,
            ..default()
        },
        ImageObject,
        RigidBody::Dynamic,
        collider,
        Velocity::default(),
        Restitution::coefficient(0.2),
        Friction::coefficient(0.8),
        Sleeping::disabled(),
    ));
}

/// Quick sanity check on a .glb/.gltf: does its JSON mention any meshes?
fn gltf_has_meshes(path: &std::path::Path) -> bool {
    let Ok(bytes) = std::fs::read(path) else {
        return true;
    };
    let json: &[u8] = if bytes.starts_with(b"glTF") && bytes.len() > 20 {
        let len = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
        &bytes[20..(20 + len).min(bytes.len())]
    } else {
        &bytes
    };
    String::from_utf8_lossy(json).contains("\"meshes\"")
}

/// Where Bevy's AssetServer reads from (same rule Bevy itself uses).
fn asset_root() -> PathBuf {
    let base = std::env::var("BEVY_ASSET_ROOT")
        .or_else(|_| std::env::var("CARGO_MANIFEST_DIR"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.to_path_buf()))
                .unwrap_or_default()
        });
    base.join("assets")
}

/// Drag & drop: .glb/.gltf -> 3D model, images -> relief.
fn handle_dropped_files(
    mut events: EventReader<FileDragAndDrop>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
    chan: Res<ConvertChannel>,
    mut status: ResMut<Status>,
    rapier: Res<RapierContext>,
    player: Query<Entity, With<Player>>,
    cam: Query<&Transform, With<Camera3d>>,
) {
    let Ok(cam) = cam.get_single() else { return };
    for ev in events.read() {
        let FileDragAndDrop::DroppedFile { path_buf, .. } = ev else {
            continue;
        };

        let yaw = cam.rotation.to_euler(EulerRot::YXZ).0;
        let filter = world_ray_filter(player.get_single().ok());
        let dist = rapier
            .cast_ray(cam.translation, *cam.forward(), 3.0, true, filter)
            .map(|(_, toi)| (toi - 0.9).clamp(1.2, 2.2))
            .unwrap_or(2.2);
        let pos = cam.translation + *cam.forward() * dist + Vec3::Y * 0.3;
        let tf = Transform::from_translation(pos).with_rotation(Quat::from_rotation_y(yaw));

        let ext = path_buf
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .unwrap_or_default();

        match ext.as_str() {
            "glb" | "gltf" => {
                if !gltf_has_meshes(path_buf) {
                    status.0 = "That file has no objects in it. In Blender's export panel, untick \
                                'Selected Objects' (or select your model first) and make sure it's visible."
                        .into();
                    continue;
                }
                let dir = asset_root().join("dropped");
                let stamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0);
                let file = path_buf
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("model.glb");
                let safe: String = file
                    .chars()
                    .map(|c| {
                        if c.is_ascii_alphanumeric() || c == '.' || c == '-' {
                            c
                        } else {
                            '_'
                        }
                    })
                    .collect();
                let name = format!("{stamp}_{safe}");
                if let Err(e) = std::fs::create_dir_all(&dir)
                    .and_then(|_| std::fs::copy(path_buf, dir.join(&name)).map(|_| ()))
                {
                    status.0 = format!("Could not copy '{file}': {e}");
                    continue;
                }
                spawn_model_entity(&mut commands, &asset_server, format!("dropped/{name}"), tf);
                status.0 = format!("Loading {file} ...");
            }
            "fbx" => {
                let dir = asset_root().join("dropped");
                if let Err(e) = std::fs::create_dir_all(&dir) {
                    status.0 = format!("Could not create {}: {e}", dir.display());
                    continue;
                }
                let stamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0);
                let stem = path_buf
                    .file_stem()
                    .and_then(|n| n.to_str())
                    .unwrap_or("model")
                    .to_string();
                let safe: String = stem
                    .chars()
                    .map(|c| {
                        if c.is_ascii_alphanumeric() || c == '-' {
                            c
                        } else {
                            '_'
                        }
                    })
                    .collect();
                let name = format!("{stamp}_{safe}");
                let input = path_buf.clone();
                let out_glb = dir.join(format!("{name}.glb"));
                let rel = format!("dropped/{name}.glb");
                let tx = chan.tx.clone();
                let label = stem.clone();
                std::thread::spawn(move || {
                    let result = convert_fbx(&input, &out_glb).map(|_| rel);
                    let _ = tx.send(Converted {
                        result,
                        transform: tf,
                        name: label,
                    });
                });
                status.0 = format!("Converting {stem}.fbx ... (Blender can take ~10 s to start)");
            }
            "blend" => {
                status.0 =
                    "Can't read .blend directly. In Blender: File > Export > glTF 2.0 (.glb)"
                        .into();
            }
            "obj" | "stl" | "dae" | "3ds" => {
                status.0 =
                    format!(".{ext} isn't supported. In Blender: File > Export > glTF 2.0 (.glb)");
            }
            _ => match open_image(&path_buf.to_string_lossy()) {
                Some(img) => {
                    spawn_relief(&mut commands, &mut meshes, &mut mats, &img, tf);
                    status.0 = "Image turned into 3D.".into();
                }
                None => {
                    status.0 = format!("'{}' is not a readable image or model.", path_buf.display())
                }
            },
        }
    }
}

/// Spawn a glTF scene from assets/<rel>; `finalize_models` finishes it once loaded.
fn spawn_model_entity(
    commands: &mut Commands,
    asset_server: &AssetServer,
    rel: String,
    tf: Transform,
) {
    let scene: Handle<Scene> = asset_server.load(GltfAssetLabel::Scene(0).from_asset(rel));
    commands.spawn((
        SceneBundle {
            scene,
            transform: tf,
            ..default()
        },
        ImageObject,
        PendingModel(0.0),
    ));
}

fn find_blender() -> Option<PathBuf> {
    use std::process::{Command, Stdio};
    if let Ok(p) = std::env::var("BLENDER_PATH") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    if Command::new("blender")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
    {
        return Some(PathBuf::from("blender"));
    }
    let mut found: Vec<PathBuf> = Vec::new();
    for root in [
        r"C:\Program Files\Blender Foundation",
        r"D:\Program Files\Blender Foundation",
        r"C:\Program Files (x86)\Blender Foundation",
    ] {
        if let Ok(entries) = std::fs::read_dir(root) {
            for e in entries.flatten() {
                let exe = e.path().join("blender.exe");
                if exe.exists() {
                    found.push(exe);
                }
            }
        }
    }
    found.sort();
    if let Some(latest) = found.pop() {
        return Some(latest);
    }
    let mac = PathBuf::from("/Applications/Blender.app/Contents/MacOS/Blender");
    mac.exists().then_some(mac)
}

/// Convert an .fbx to a self-contained .glb. Runs on a background thread.
fn convert_fbx(input: &Path, out_glb: &Path) -> Result<(), String> {
    use std::process::{Command, Stdio};

    let f2g = std::env::var("FBX2GLTF_PATH").unwrap_or_else(|_| "FBX2glTF".into());
    let out_stem = out_glb.with_extension("");
    if let Ok(st) = Command::new(&f2g)
        .arg("--binary")
        .arg("--input")
        .arg(input)
        .arg("--output")
        .arg(&out_stem)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
    {
        if st.success() && out_glb.exists() {
            return Ok(());
        }
    }

    let Some(blender) = find_blender() else {
        return Err(
            "Can't read .fbx directly. Install Blender (or set BLENDER_PATH), or install \
                    FBX2glTF, or export a .glb from Blender instead."
                .into(),
        );
    };
    let script = std::env::temp_dir().join("fbx_to_glb.py");
    let py = r#"import bpy, os
bpy.ops.wm.read_factory_settings(use_empty=True)
src = os.environ['FBX_IN']
try:
    bpy.ops.import_scene.fbx(filepath=src)
except Exception:
    bpy.ops.wm.fbx_import(filepath=src)
bpy.ops.export_scene.gltf(filepath=os.environ['FBX_OUT'], export_format='GLB')
"#;
    std::fs::write(&script, py).map_err(|e| format!("Could not write temp script: {e}"))?;

    let out = Command::new(&blender)
        .args([
            "-b",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
        ])
        .arg(&script)
        .env("FBX_IN", input)
        .env("FBX_OUT", out_glb)
        .output()
        .map_err(|e| format!("Could not start Blender: {e}"))?;

    if out.status.success() && out_glb.exists() {
        Ok(())
    } else {
        let log = String::from_utf8_lossy(&out.stdout);
        let err = String::from_utf8_lossy(&out.stderr);
        eprintln!("--- Blender FBX conversion failed ---\n{log}\n{err}");
        Err("Blender could not convert that .fbx (details in the terminal).".into())
    }
}

/// Receives finished .fbx conversions and spawns them like any other model.
fn poll_conversions(
    chan: Res<ConvertChannel>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut status: ResMut<Status>,
) {
    let Ok(rx) = chan.rx.lock() else { return };
    while let Ok(done) = rx.try_recv() {
        match done.result {
            Ok(rel) => {
                if !gltf_has_meshes(&asset_root().join(&rel)) {
                    status.0 = format!("'{}' converted but contains no meshes.", done.name);
                    continue;
                }
                spawn_model_entity(&mut commands, &asset_server, rel, done.transform);
                status.0 = format!("Converted {} - loading ...", done.name);
            }
            Err(e) => status.0 = e,
        }
    }
}

/// Once a dropped model has finished loading: measure it, scale it to a sensible
/// size, center it, and give it a dynamic body with convex-hull colliders.
fn finalize_models(
    time: Res<Time>,
    mut commands: Commands,
    spawner: Res<SceneSpawner>,
    meshes: Res<Assets<Mesh>>,
    mut status: ResMut<Status>,
    mut pending: Query<(
        Entity,
        &SceneInstance,
        &GlobalTransform,
        &mut Transform,
        &mut PendingModel,
    )>,
    children: Query<&Children>,
    mesh_q: Query<(&Handle<Mesh>, &GlobalTransform)>,
) {
    for (entity, instance, root_gt, mut tf, mut wait) in &mut pending {
        wait.0 += time.delta_seconds();
        if !spawner.instance_is_ready(**instance) {
            if wait.0 > 30.0 {
                commands.entity(entity).despawn_recursive();
                status.0 = "Model failed to load (see terminal). Use a self-contained .glb.".into();
            }
            continue;
        }

        let inv = root_gt.affine().inverse();
        let (mut min, mut max) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for e in children.iter_descendants(entity) {
            let Ok((handle, gt)) = mesh_q.get(e) else {
                continue;
            };
            let Some(mesh) = meshes.get(handle) else {
                continue;
            };
            let Some(pos) = mesh
                .attribute(Mesh::ATTRIBUTE_POSITION)
                .and_then(|a| a.as_float3())
            else {
                continue;
            };
            let m = inv * gt.affine();
            for v in pos {
                let w = m.transform_point3(Vec3::from(*v));
                min = min.min(w);
                max = max.max(w);
            }
        }
        let size = max - min;
        if !(size.max_element() > 0.0) || !size.is_finite() {
            commands.entity(entity).despawn_recursive();
            status.0 = "That model has no meshes.".into();
            continue;
        }

        let s = MODEL_SIZE / size.max_element();
        let center = (min + max) * 0.5;
        let offset = tf.rotation * (center * s);
        tf.scale = Vec3::splat(s);
        tf.translation -= offset;

        commands.entity(entity).remove::<PendingModel>().insert((
            RigidBody::Dynamic,
            AsyncSceneCollider {
                shape: Some(ComputedColliderShape::ConvexHull),
                ..default()
            },
            Velocity::default(),
            Restitution::coefficient(0.2),
            Friction::coefficient(0.8),
            Sleeping::disabled(),
        ));
        status.0 = "Model ready - grab it!".into();
    }
}

fn update_status(status: Res<Status>, mut q: Query<&mut Text, With<StatusText>>) {
    if status.is_changed() {
        for mut t in &mut q {
            t.sections[1].value = status.0.clone();
        }
    }
}

/// Press X to remove every dropped object.
fn clear_objects(
    keys: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
    objects: Query<Entity, With<ImageObject>>,
    dig_sites: Query<Entity, With<DigSite>>,
    mut terrain: Query<(Entity, &Handle<Mesh>), With<DiggableGround>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut grab: ResMut<Grab>,
) {
    if keys.just_pressed(KeyCode::KeyX) {
        for e in &objects {
            commands.entity(e).despawn_recursive();
        }
        for e in &dig_sites {
            commands.entity(e).despawn_recursive();
        }
        update_diggable_ground(&[], &mut terrain, &mut meshes, &mut commands);
        grab.entity = None;
    }
}

// ---------------------------------------------------------------------------
// Player: walk, run, jump (RMB look)
// ---------------------------------------------------------------------------

fn player_system(
    kneel: Res<Kneel>,
    time: Res<Time>,
    river: Res<RiverState>,
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion: EventReader<MouseMotion>,
    mut q: Query<
        (
            &mut Player,
            &mut Transform,
            &mut KinematicCharacterController,
            Option<&KinematicCharacterControllerOutput>,
        ),
        With<Player>,
    >,
    mut horses: Query<&mut Transform, (With<Horse>, Without<Player>, Without<Tree>)>,
    trees: Query<(Entity, &Transform, &Tree), (With<Tree>, Without<Player>)>,
    poss: Res<Possession>,
) {
    let Ok((mut p, mut tf, mut ctrl, out)) = q.get_single_mut() else {
        return;
    };
    let dt = time.delta_seconds().min(0.05);

    // look: hold the right mouse button
    if buttons.pressed(MouseButton::Right) {
        for ev in motion.read() {
            p.yaw -= ev.delta.x * 0.003;
            p.pitch = (p.pitch - ev.delta.y * 0.003).clamp(-1.5, 1.5);
        }
    } else {
        motion.clear();
    }

    // possessing a creature: your own body just stays where it is
    if poss.target.is_some() {
        ctrl.translation = Some(Vec3::new(0.0, -0.05, 0.0));
        p.vy = 0.0;
        p.last_dy = 0.0;
        return;
    }

    // walk (relative to where you're facing, never up/down)
    let (sin, cos) = p.yaw.sin_cos();
    let fwd = Vec3::new(-sin, 0.0, -cos);
    let right = Vec3::new(cos, 0.0, -sin);
    let mut dir = Vec3::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        dir += fwd;
    }
    if keys.pressed(KeyCode::KeyS) {
        dir -= fwd;
    }
    if keys.pressed(KeyCode::KeyD) {
        dir += right;
    }
    if keys.pressed(KeyCode::KeyA) {
        dir -= right;
    }
    let in_river_channel = (tf.translation.x - river_center_x(tf.translation.z)).abs()
        < RIVER_WIDTH * 0.5 + river.overflow
        && tf.translation.y - 0.9 < river.water_level;
    let speed = WALK_SPEED
        * if in_river_channel { 0.48 } else { 1.0 }
        * (1.0 - 0.65 * kneel.amount);
    let horizontal = if dir != Vec3::ZERO {
        dir.normalize() * speed * dt
    } else {
        Vec3::ZERO
    };

    if let Some(horse_entity) = p.mounted {
        p.climbing_tree = None;
        if keys.just_pressed(KeyCode::KeyE) {
            if let Ok(horse_transform) = horses.get(horse_entity) {
                tf.translation = horse_transform.translation + right * 1.5 + Vec3::Y * 0.92;
            }
            p.mounted = None;
            p.vy = 0.0;
        } else if let Ok(mut horse_transform) = horses.get_mut(horse_entity) {
            ctrl.filter_groups = Some(CollisionGroups::new(
                Group::ALL,
                Group::ALL & !Group::GROUP_2 & !Group::GROUP_3,
            ));
            horse_transform.translation += horizontal * 1.7;
            horse_transform.translation.y =
                land_surface_height(horse_transform.translation.x, horse_transform.translation.z);
            horse_transform.rotation = Quat::from_rotation_y(p.yaw);
            tf.translation = horse_transform.translation + Vec3::Y * 2.8;
            ctrl.translation = Some(Vec3::ZERO);
            p.vy = 0.0;
            p.last_dy = 0.0;
            return;
        } else {
            p.mounted = None;
        }
    }
    ctrl.filter_groups = Some(CollisionGroups::new(
        Group::ALL,
        Group::ALL & !Group::GROUP_2,
    ));

    // gravity + jump
    let grounded = out.map(|o| o.grounded).unwrap_or(false);
    let bumped_head = p.vy > 0.0
        && out
            .map(|o| o.effective_translation.y < p.last_dy * 0.5)
            .unwrap_or(false);
    if bumped_head {
        p.vy = 0.0;
    }
    if grounded && p.vy <= 0.0 {
        p.vy = -1.0; // keep pressed against the floor
    }
    if grounded && keys.just_pressed(KeyCode::Space) && p.climbing_tree.is_none() && kneel.amount < 0.3 {
        p.vy = JUMP_V * if in_river_channel { 0.62 } else { 1.0 };
    }

    let climbing = p
        .climbing_tree
        .and_then(|entity| {
            trees
                .iter()
                .find(|(tree_entity, _, _)| *tree_entity == entity)
        })
        .filter(|(_, tree_tf, tree)| {
            let toward = Vec3::new(
                tree_tf.translation.x - tf.translation.x,
                0.0,
                tree_tf.translation.z - tf.translation.z,
            );
            let distance = toward.length();
            let tree_base_y = tree_tf.translation.y - tree.height * 0.5;
            let crown_limit = tree_base_y + tree.height - 0.75;
            distance < 1.45
                && (dir == Vec3::ZERO
                    || toward.normalize_or_zero().dot(dir.normalize_or_zero()) > 0.15)
                && tf.translation.y < crown_limit
        })
        .map(|(entity, _, _)| entity)
        .or_else(|| {
            if p.vy <= 0.0 || dir == Vec3::ZERO {
                return None;
            }
            trees
                .iter()
                .filter_map(|(entity, tree_tf, tree)| {
                    let toward = Vec3::new(
                        tree_tf.translation.x - tf.translation.x,
                        0.0,
                        tree_tf.translation.z - tf.translation.z,
                    );
                    let distance = toward.length();
                    let tree_base_y = tree_tf.translation.y - tree.height * 0.5;
                    let crown_limit = tree_base_y + tree.height - 0.75;
                    (distance < 1.45
                        && toward.normalize_or_zero().dot(dir.normalize_or_zero()) > 0.15
                        && tf.translation.y < crown_limit
                        && tf.translation.y > tree_base_y - 0.2)
                        .then_some((entity, distance))
                })
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(entity, _)| entity)
        });
    p.climbing_tree = climbing;

    if climbing.is_some() {
        let climb_step = if keys.just_pressed(KeyCode::Space) {
            0.55_f32
        } else {
            0.0
        };
        let dy = if let Some(tree_entity) = climbing {
            trees
                .iter()
                .find(|(entity, _, _)| *entity == tree_entity)
                .map(|(_, tree_tf, tree)| {
                    let crown_limit = tree_tf.translation.y + tree.height * 0.5 - 0.75;
                    climb_step
                        .max(0.0)
                        .min((crown_limit - tf.translation.y).max(0.0))
                })
                .unwrap_or(0.0)
        } else {
            0.0
        };
        p.vy = 0.0;
        p.last_dy = dy;
        ctrl.translation = Some(horizontal + Vec3::Y * dy);
        if tf.translation.y < -20.0 {
            tf.translation = PLAYER_SPAWN;
            p.vy = 0.0;
            p.climbing_tree = None;
        }
        return;
    }

    p.vy -= GRAVITY * (if in_river_channel { 0.68 } else { 1.0 }) * dt;
    if in_river_channel && p.vy < 0.0 {
        p.vy *= (-1.2 * dt).exp();
    }
    let dy = p.vy * dt;
    p.last_dy = dy;

    ctrl.translation = Some(horizontal + Vec3::Y * dy);

    // fell off the world? back to the start
    if tf.translation.y < -20.0 {
        tf.translation = PLAYER_SPAWN;
        p.vy = 0.0;
    }
}

fn horse_wander_system(
    time: Res<Time>,
    player: Query<(Entity, &Player)>,
    mut horses: Query<(Entity, &mut Horse, &mut Transform)>,
) {
    let mounted_horse = player
        .get_single()
        .ok()
        .and_then(|(_, rider)| rider.mounted);
    let dt = time.delta_seconds().min(0.05);

    for (entity, mut horse, mut transform) in &mut horses {
        if mounted_horse == Some(entity) {
            continue;
        }
        horse.pause -= dt;
        if horse.pause > 0.0 {
            continue;
        }

        let offset = horse.target - transform.translation;
        let distance = Vec2::new(offset.x, offset.z).length();
        if distance < 0.25 {
            horse.seed = horse.seed.wrapping_add(1);
            let angle = pseudo_rand(horse.seed) * std::f32::consts::TAU;
            let radius = 2.0 + pseudo_rand(horse.seed.wrapping_add(1)) * 3.0;
            horse.target = horse.home + Vec3::new(angle.cos() * radius, 0.0, angle.sin() * radius);
            horse.pause = 1.0 + pseudo_rand(horse.seed.wrapping_add(2)) * 2.5;
            continue;
        }

        let direction = Vec3::new(offset.x, 0.0, offset.z).normalize();
        let step = 1.2 * dt;
        if distance <= step {
            transform.translation.x = horse.target.x;
            transform.translation.z = horse.target.z;
            horse.pause = 0.8 + pseudo_rand(horse.seed.wrapping_add(3)) * 2.0;
        } else {
            transform.translation += direction * step;
            transform.rotation = Quat::from_rotation_y((-direction.x).atan2(-direction.z));
        }
        transform.translation.y =
            land_surface_height(transform.translation.x, transform.translation.z);
    }
}

/// The camera sits at the player's eyes and looks where the player looks.
fn camera_follow(
    kneel: Res<Kneel>,
    poss: Res<Possession>,
    player: Query<(&Transform, &Player), Without<Camera3d>>,
    mut cam: Query<&mut Transform, With<Camera3d>>,
) {
    let (Ok((pt, p)), Ok(mut ct)) = (player.get_single(), cam.get_single_mut()) else {
        return;
    };
    // possessing a creature: the camera is at its eyes
    if let (Some(_), Some(kind)) = (poss.target, poss.kind) {
        ct.translation = poss.pos + Vec3::Y * kind.eye;
        ct.rotation = Quat::from_euler(EulerRot::YXZ, p.yaw, p.pitch, 0.0);
        return;
    }
    ct.translation = pt.translation + Vec3::Y * (EYE_OFFSET - 0.5 * kneel.amount);
    ct.rotation = Quat::from_euler(EulerRot::YXZ, p.yaw, p.pitch, 0.0);
}

// ---------------------------------------------------------------------------
// Your body: hands in view, and legs when you look down
// ---------------------------------------------------------------------------

fn setup_fp_body(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
    cameras: Query<Entity, With<Camera3d>>,
) {
    let Ok(cam) = cameras.get_single() else {
        return;
    };
    let mut mat = |c: Color, rough: f32| {
        mats.add(StandardMaterial {
            base_color: c,
            perceptual_roughness: rough,
            ..default()
        })
    };
    let skin = mat(Color::srgb(0.87, 0.67, 0.52), 0.7);
    let sleeve = mat(Color::srgb(0.22, 0.32, 0.5), 0.9);
    let pants = mat(Color::srgb(0.2, 0.23, 0.32), 0.9);
    let boot = mat(Color::srgb(0.26, 0.17, 0.1), 0.8);
    let ball = meshes.add(Sphere::new(0.5));
    let stick = meshes.add(Cylinder::new(1.0, 1.0));

    // ----- legs and torso: placed under the camera every frame -----
    commands
        .spawn((SpatialBundle::default(), FpLegsRoot))
        .with_children(|p| {
            // hips and chest (below and behind the eyes, so only seen when looking down)
            gh_blob(p, &ball, &pants, Vec3::new(0.0, 0.95, 0.02), Vec3::new(0.32, 0.22, 0.22), Quat::IDENTITY);
            gh_blob(p, &ball, &sleeve, Vec3::new(0.0, 1.17, 0.09), Vec3::new(0.4, 0.52, 0.26), Quat::IDENTITY);
            for side_i in 0..2usize {
                let side = if side_i == 0 { -1.0_f32 } else { 1.0 };
                p.spawn((
                    SpatialBundle::from_transform(Transform::from_xyz(side * 0.1, 0.92, 0.0)),
                    FpLegPart { side: side_i, kind: 0 },
                ))
                .with_children(|hip| {
                    gh_limb(hip, &stick, &pants, Vec3::ZERO, Vec3::new(0.0, -0.44, 0.0), 0.078);
                    hip.spawn((
                        SpatialBundle::from_transform(Transform::from_xyz(0.0, -0.44, 0.0)),
                        FpLegPart { side: side_i, kind: 1 },
                    ))
                    .with_children(|knee| {
                        gh_limb(knee, &stick, &pants, Vec3::ZERO, Vec3::new(0.0, -0.42, 0.0), 0.058);
                        knee.spawn((
                            SpatialBundle::from_transform(Transform::from_xyz(0.0, -0.42, 0.0)),
                            FpLegPart { side: side_i, kind: 2 },
                        ))
                        .with_children(|ankle| {
                            gh_blob(ankle, &ball, &boot, Vec3::new(0.0, 0.0, -0.07), Vec3::new(0.11, 0.13, 0.28), Quat::IDENTITY);
                        });
                    });
                });
            }
        });
}

/// Animates your hands (reaching for and gripping what you hold, holding tools, swaying as
/// you walk) and your legs (walking).
fn fp_body_system(
    time: Res<Time>,
    grab: Res<Grab>,
    sticks_state: Res<StickState>,
    kneel: Res<Kneel>,
    poss: Res<Possession>,
    mut st: Local<FpState>,
    player: Query<(&Transform, &Player)>,
    cam: Query<&GlobalTransform, With<Camera3d>>,
    tools: ToolsHeld,
    gts: Query<&GlobalTransform>,
    mut root: Query<(&mut Transform, &mut Visibility), (With<FpLegsRoot>, Without<Camera3d>, Without<Player>)>,
    mut leg_parts: Query<
        (&FpLegPart, &mut Transform),
        (Without<FpLegsRoot>, Without<FpArm>, Without<FpFinger>, Without<Camera3d>, Without<Player>),
    >,
    mut arms: Query<
        (&FpArm, &mut Transform),
        (Without<FpFinger>, Without<FpLegsRoot>, Without<FpLegPart>, Without<Camera3d>, Without<Player>),
    >,
    mut fingers: Query<
        (&FpFinger, &mut Transform),
        (Without<FpArm>, Without<FpLegsRoot>, Without<FpLegPart>, Without<Camera3d>, Without<Player>),
    >,
) {
    use std::f32::consts::PI;
    let dt = time.delta_seconds().clamp(1e-3, 0.05);
    let (Ok((pt, p)), Ok(cam_gt)) = (player.get_single(), cam.get_single()) else {
        return;
    };

    // how fast you are walking
    let flat = Vec3::new(pt.translation.x, 0.0, pt.translation.z);
    let mut moved = st.last.map_or(0.0, |l| (flat - l).length());
    if moved > 1.5 {
        moved = 0.0; // teleported (respawn)
    }
    st.last = Some(flat);
    let speed = (moved / dt).min(8.0);
    st.speed += (speed - st.speed) * (1.0 - (-8.0 * dt).exp());
    st.phase += moved * PI / 0.75; // one step every 0.75 m
    let amp = (st.speed / 3.0).clamp(0.0, 1.0);
    let phase = st.phase;

    // ----- legs -----
    if let Ok((mut t, mut vis)) = root.get_single_mut() {
        t.translation = pt.translation - Vec3::Y * (0.9 + 0.42 * kneel.amount);
        t.rotation = Quat::from_rotation_y(p.yaw);
        let want = if p.mounted.is_some() || p.climbing_tree.is_some() || poss.target.is_some() {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        };
        if *vis != want {
            *vis = want;
        }
    }
    for (part, mut t) in &mut leg_parts {
        let sp = phase + if part.side == 0 { 0.0 } else { PI };
        let walk_hip = sp.sin() * 0.55 * amp;
        let walk_knee = -(sp.cos().max(0.0)) * 0.9 * amp;
        // kneeling: thighs straight down, shins flat on the ground behind, toes tucked
        let k = kneel.amount;
        let hip = walk_hip * (1.0 - k);
        let knee = walk_knee * (1.0 - k) - std::f32::consts::FRAC_PI_2 * k;
        t.rotation = Quat::from_rotation_x(match part.kind {
            0 => hip,
            1 => knee,
            _ => -(walk_hip + walk_knee) * (1.0 - k), // keep the boot flat when walking
        });
    }

    // ----- hands -----
    let tool = tools.any();
    let inverse_cam = cam_gt.affine().inverse();
    let held_local = grab.entity.and_then(|e| {
        gts.get(e)
            .ok()
            .map(|gt| inverse_cam.transform_point3(gt.transform_point(grab.local_anchor)))
    });
    let breathe = (time.elapsed_seconds() * 1.6).sin() * 0.002;
    let k = 1.0 - (-16.0 * dt).exp();
    for (arm, mut t) in &mut arms {
        let side = if arm.side == 0 { -1.0_f32 } else { 1.0 };
        let shoulder = Vec3::new(side * 0.3, -0.45, 0.1);
        let mut target = Vec3::new(side * 0.22, -0.2, -0.52);
        let mut grip = 0.0;
        if let Some(g) = sticks_state.grips[arm.side] {
            // holding a stick (or rubbing two together): the hand goes to its grip
            let to = inverse_cam.transform_point3(g) - shoulder;
            let d = to.length().max(1e-3);
            target = shoulder + to / d * d.clamp(0.4, 0.8);
            grip = 1.0;
        } else if tool && arm.side == 1 {
            target = Vec3::new(0.31, -0.31, -0.56); // where the tool's grip is held
            grip = 1.0;
        } else if let Some(held) = held_local.filter(|h| h.z < -0.1) {
            // reach toward what you are holding (the left hand helps from the side)
            let aim = held + if arm.side == 0 { Vec3::new(-0.13, -0.02, 0.05) } else { Vec3::ZERO };
            let to = aim - shoulder;
            let d = to.length().max(1e-3);
            target = shoulder + to / d * d.clamp(0.4, 0.8);
            grip = if arm.side == 1 { 1.0 } else { 0.7 };
        }
        target += Vec3::new(side * 0.004 * phase.sin() * amp, (phase * 2.0).sin() * 0.01 * amp + breathe, 0.0);
        let dir = (target - shoulder).normalize();
        let want_rot = Quat::from_rotation_arc(Vec3::NEG_Z, dir) * Quat::from_rotation_z(side * 0.2);
        t.translation = t.translation.lerp(target, k);
        t.rotation = t.rotation.slerp(want_rot, k);
        let g = &mut st.grips[arm.side];
        *g += (grip - *g) * (1.0 - (-14.0 * dt).exp());
    }
    for (f, mut t) in &mut fingers {
        let grip = st.grips[f.side];
        let side = if f.side == 0 { -1.0_f32 } else { 1.0 };
        if f.thumb {
            t.rotation = if f.level == 0 {
                Quat::from_rotation_y(side * (0.45 + 0.7 * grip)) * Quat::from_rotation_x(-(0.1 + 0.3 * grip))
            } else {
                Quat::from_rotation_x(-(0.2 + 0.6 * grip))
            };
        } else {
            let relaxed = [0.25_f32, 0.4, 0.3][f.level as usize];
            let fist = [1.35_f32, 1.5, 1.0][f.level as usize];
            t.rotation = Quat::from_rotation_x(-(relaxed + (fist - relaxed) * grip + f.bias));
        }
    }
}

// ---------------------------------------------------------------------------
// Sticks and fire
// ---------------------------------------------------------------------------

fn setup_fire_assets(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
) {
    let mut glow = |r: f32, g: f32, b: f32, a: f32| {
        mats.add(StandardMaterial {
            base_color: Color::srgba(r, g, b, a),
            unlit: true,
            alpha_mode: AlphaMode::Blend,
            ..default()
        })
    };
    let outer = glow(1.0, 0.42, 0.06, 0.75);
    let mid = glow(1.0, 0.72, 0.15, 0.85);
    let core = glow(1.0, 0.95, 0.6, 0.9);
    let ember = glow(1.0, 0.3, 0.05, 0.9);
    let spark = mats.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.7, 0.2),
        unlit: true,
        ..default()
    });
    let smoke = mats.add(StandardMaterial {
        base_color: Color::srgba(0.35, 0.35, 0.35, 0.3),
        perceptual_roughness: 1.0,
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    let charred = mats.add(StandardMaterial {
        base_color: Color::srgb(0.08, 0.05, 0.04),
        perceptual_roughness: 0.95,
        ..default()
    });
    let ash = mats.add(StandardMaterial {
        base_color: Color::srgb(0.16, 0.15, 0.14),
        perceptual_roughness: 1.0,
        ..default()
    });
        let dust = mats.add(StandardMaterial {
        base_color: Color::srgb(0.72, 0.55, 0.32),
        perceptual_roughness: 0.9,
        ..default()
    });
    let burned_dust = mats.add(StandardMaterial {              // <-- new
        base_color: Color::srgb(0.035, 0.028, 0.024),         // near-black char
        perceptual_roughness: 1.0,
        reflectance: 0.04,
        ..default()
    });
    commands.insert_resource(FireAssets {
        ball: meshes.add(Sphere::new(0.5)),
        cylinder: meshes.add(Cylinder::new(1.0, 1.0)),
        outer,
        mid,
        core,
        ember,
        smoke,
        spark,
        charred,
        ash,
        dust,
        burned_dust,
    });
}

/// Hold C to kneel: the camera drops, you move slowly and can't jump.
fn kneel_system(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut kneel: ResMut<Kneel>,
    player: Query<&Player>,
) {
    let dt = time.delta_seconds().min(0.05);
    let blocked = player
        .get_single()
        .map_or(false, |p| p.mounted.is_some() || p.climbing_tree.is_some());
    let holding_shift =
        keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    let target = if holding_shift && !blocked { 1.0 } else { 0.0 };
    kneel.amount += (target - kneel.amount) * (1.0 - (-9.0 * dt).exp());
    if (kneel.amount - target).abs() < 0.002 {
        kneel.amount = target;
    }
}

/// Gives a stick back to the physics world (after it was held).
fn restore_stick_physics(commands: &mut Commands, entity: Entity, velocity: Vec3) {
    commands.entity(entity).insert((
        RigidBody::Dynamic,
        Collider::capsule_y(STICK_HALF, STICK_RADIUS),
        Velocity::linear(velocity),
        ExternalForce::default(),
        Sleeping::disabled(),
    ));
}

/// Takes a stick out of the physics world so it can be placed by hand.
fn hold_stick_physics(commands: &mut Commands, entity: Entity) {
    commands
        .entity(entity)
        .remove::<RigidBody>()
        .remove::<Collider>()
        .remove::<Velocity>();
}

fn spawn_fire_particle(
    commands: &mut Commands,
    assets: &FireAssets,
    material: Handle<StandardMaterial>,
    position: Vec3,
    velocity: Vec3,
    life: f32,
    size0: f32,
    grow: f32,
    buoyancy: f32,
) {
    commands.spawn((
        PbrBundle {
            mesh: assets.ball.clone(),
            material,
            transform: Transform::from_translation(position).with_scale(Vec3::splat(size0)),
            ..default()
        },
        FireParticle {
            velocity,
            life,
            max_life: life,
            size0, grow,
            buoyancy },
    ));
}

/// Lays a small pile of charred wood dust at `pos`.
fn spawn_burned_pile(
    commands: &mut Commands,
    assets: &FireAssets,
    pos: Vec3,
    seed: u32,
    now: f32,
) -> Entity {
    commands
        .spawn((
            PbrBundle {
                mesh: assets.ball.clone(),
                material: assets.burned_dust.clone(),
                transform: Transform::from_translation(pos + Vec3::Y * 0.004)
                    .with_scale(Vec3::new(0.032, 0.010, 0.032)),
                ..default()
            },
            BurnedDustPile {
                growth: 0.1,
                smoke_timer: 0.0,
                seed,
                smoke_left: PILE_SMOKE_SECONDS,
                born: now,
            },
            // A real physics body: the unit ball collider follows the squashed mesh scale,
            // so the pile rests on the ground, rolls, and can be knocked or grabbed.
            RigidBody::Dynamic,
            Collider::ball(0.5),
            ColliderMassProperties::Mass(0.03),
            Velocity::default(),
            Friction::coefficient(1.2),
            Restitution::coefficient(0.0),
            Damping {
                linear_damping: 0.6,
                angular_damping: 4.0,
            },
            ExternalForce::default(),
            ReadMassProperties::default(),
            Sleeping::disabled(),
            Ccd::enabled(),
        ))
        .id()
}

/// Every burned-dust pile smokes: a thin wisp when it is fresh, a fuller plume as it grows.
/// With no bark lying near it the smoke dies away after a while (it thins out at the end);
/// bark nearby, or rubbing it again, keeps it going.
fn burned_dust_smoke_system(
    time: Res<Time>,
    mut commands: Commands,
    assets: Res<FireAssets>,
    bark: Query<&GlobalTransform, With<BarkStrip>>,
    mut piles: Query<(&mut BurnedDustPile, &GlobalTransform)>,
) {
    let dt = time.delta_seconds().clamp(1e-3, 0.05);
    for (mut pile, gt) in &mut piles {
        let pos = gt.translation();
        let bark_near = bark
            .iter()
            .any(|b| b.translation().distance(pos) < PILE_BARK_RADIUS);
        if bark_near {
            pile.smoke_left = PILE_SMOKE_SECONDS;
        } else {
            pile.smoke_left = (pile.smoke_left - dt).max(0.0);
        }
        if pile.smoke_left <= 0.0 {
            continue;
        }
        // thins out over the last few seconds
        let fade = (pile.smoke_left / 6.0).clamp(0.0, 1.0);
        let g = pile.growth.clamp(0.0, 1.0);
        pile.smoke_timer -= dt * (0.35 + 2.5 * g) * (0.2 + 0.8 * fade);
        while pile.smoke_timer <= 0.0 {
            pile.smoke_timer += 0.22 - 0.1 * g;
            pile.seed = pile.seed.wrapping_add(1);
            let j = pseudo_rand(pile.seed) * 2.0 - 1.0;
            let k = pseudo_rand(pile.seed.wrapping_add(3)) * 2.0 - 1.0;
            spawn_fire_particle(
                &mut commands,
                &assets,
                assets.smoke.clone(),
                pos + Vec3::Y * 0.02 + Vec3::new(j, 0.0, k) * 0.015,
                Vec3::new(j * 0.05, 0.14 + 0.22 * g, k * 0.05),
                1.6 + 0.6 * g,
                (0.014 + 0.02 * g) * (0.6 + 0.4 * fade),
                3.0,
                0.10,
            );
        }
    }
}

/// While kneeling next to a pile of charred dust, press C to feed it any pieces of bark
/// lying nearby. The bark burns up: it tops up a fire already burning there, or the embers
/// catch and a fresh fire springs up right where the dust is.
fn bark_feed_fire_system(
    mut commands: Commands,
    kneel: Res<Kneel>,
    grab: Res<Grab>,
    keys: Res<ButtonInput<KeyCode>>,
    mut status: ResMut<Status>,
    fire_assets: Res<FireAssets>,
    player: Query<&Transform, With<Player>>,
    dust_piles: Query<&GlobalTransform, With<BurnedDustPile>>,
    bark: Query<(Entity, &GlobalTransform), With<BarkStrip>>,
    mut fires: Query<(&Transform, &mut Fire), Without<Player>>,
) {
    // Only reacts to a fresh tap of C, and only while you're actually kneeling.
    if !keys.just_pressed(KeyCode::KeyC) || kneel.amount < 0.5 {
        return;
    }
    let Ok(player_tf) = player.get_single() else {
        return;
    };
    let player_pos = player_tf.translation;

    // Nearest charred dust pile within a comfortable kneeling reach.
    let mut nearest: Option<(Vec3, f32)> = None;
    for gt in &dust_piles {
        let p = gt.translation();
        let d = p.distance(player_pos);
        if d < 2.5 && nearest.map_or(true, |(_, nd)| d < nd) {
            nearest = Some((p, d));
        }
    }
    let Some((pile_pos, _)) = nearest else {
        return;
    };

    // Bark strips lying around the pile (the one you're holding doesn't count).
    let mut to_burn: Vec<Entity> = Vec::new();
    for (e, gt) in &bark {
        if grab.entity == Some(e) {
            continue;
        }
        if gt.translation().distance(pile_pos) < 2.5 {
            to_burn.push(e);
        }
    }
    if to_burn.is_empty() {
        status.0 = "No bark lying near the charred dust to feed the fire.".into();
        return;
    }

    let count = to_burn.len();
    for e in &to_burn {
        commands.entity(*e).despawn_recursive();
    }
    let fuel_added = 0.65 * count as f32;

    // Boost a fire already burning at the pile, else light one.
    let mut boosted = false;
    for (fire_tf, mut fire) in &mut fires {
        if fire_tf.translation.distance(pile_pos) < 3.5 {
            fire.fuel = (fire.fuel + fuel_added).min(FIRE_MAX_FUEL);
            fire.flare = 1.0;
            boosted = true;
            break;
        }
    }
    if !boosted {
        let seed = pile_pos.x.to_bits() ^ pile_pos.z.to_bits().rotate_left(13);
        spawn_fire(
            &mut commands,
            &fire_assets,
            pile_pos + Vec3::Y * 0.02,
            1.2 + fuel_added,
            seed,
        );
    }

    status.0 = if count == 1 {
        "You lay a strip of bark on the embers; the fire crackles and flares up.".into()
    } else {
        format!("You feed {count} strips of bark to the fire; the flames leap higher.")
    };
}

/// Double-click a stick to hold it. Kneel (hold C) beside another stick that is lying on the
/// ground, click it and keep the button down to rub your stick against it. Q puts it down.
fn stick_input_system(
    time: Res<Time>,
    mut commands: Commands,
    grab: Res<Grab>,
    kneel: Res<Kneel>,
    mut consumed: ResMut<ClickConsumed>,
    mut status: ResMut<Status>,
    mut st: ResMut<StickState>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    rapier: Res<RapierContext>,
    tools: ToolsHeld,
    mut sticks: Query<(Entity, &mut Stick, &GlobalTransform)>,
    notched_blocks: Query<&GlobalTransform, With<KnifeNotched>>, // <-- new
) {
    let (Ok(window), Ok((camera, cam_gt))) = (windows.get_single(), cameras.get_single()) else {
        return;
    };
    let mut equipped = sticks.iter().find(|(_, s, _)| s.equipped).map(|(e, _, _)| e);

    if let Some(eq) = equipped {
        let q = keys.just_pressed(KeyCode::KeyQ);
        if q || tools.any() {
            if let Ok((_, mut s, _)) = sticks.get_mut(eq) {
                s.equipped = false;
            }
            let velocity = if q { *cam_gt.forward() * 2.0 + Vec3::Y * 0.5 } else { Vec3::ZERO };
            restore_stick_physics(&mut commands, eq, velocity);
            if let Some(other) = st.rubbing.take() {
                restore_stick_physics(&mut commands, other, Vec3::ZERO);
            }
             st.rubbing_block = None;
            // the burned dust stays in the world; just stop tracking it
            st.burned_pile = None;
            st.heat = 0.0;
        }
    }

    if !buttons.just_pressed(MouseButton::Left) || consumed.0 {
        return;
    }
    let Some(ray) = pointer_position(&window)
        .and_then(|cursor| camera.viewport_to_world(cam_gt, cursor))
    else {
        return;
    };
    let Some((hit, hit_toi)) = rapier.cast_ray(
        ray.origin,
        *ray.direction,
        4.0,
        true,
        QueryFilter::only_dynamic(),
    ) else {
        return;
    };
    let hit_pos = ray.origin + *ray.direction * hit_toi;

    if let Some(eq) = equipped {
        if hit == eq {
            return;
        }
        let (_, rot, cam_pos) = cam_gt.to_scale_rotation_translation();
        let shoulder = cam_pos + rot * Vec3::new(0.3, -0.45, 0.1);

        // 1) another stick lying on the ground
        if let Ok((_, hit_stick, hit_gt)) = sticks.get(hit) {
            if hit_stick.equipped {
                return;
            }
            let center = hit_gt.translation();
            if kneel.amount < 0.6 {
                status.0 = "Kneel down first (hold C) beside the stick, then click it and hold the left mouse button to rub."
                    .into();
            } else if shoulder.distance(center) > 1.3 {
                status.0 = "Too far away: kneel closer to the stick.".into();
            } else {
                hold_stick_physics(&mut commands, hit);
                st.rubbing = Some(hit);
                st.rubbing_block = None;
                st.phase = 0.0;
                consumed.0 = true;
                status.0 = "Rubbing the sticks together... keep holding the left mouse button (move the mouse to rub faster)."
                    .into();
            }
            return;
        }

        // 2) a wood chip with a V notch cut into it: rub the stick in the groove
        if let Ok(block_gt) = notched_blocks.get(hit) {
            if kneel.amount < 0.6 {
                status.0 = "Kneel down first (hold C) beside the notched block, then click it and move the mouse up and down to rub the stick in the groove."
                    .into();
            } else if shoulder.distance(hit_pos) > 1.3 {
                status.0 = "Too far away: kneel closer to the notched block.".into();
            } else {
                st.rubbing_block = Some(hit);
                st.burned_pile = None; // this rub makes (or continues) its own pile
                st.rubbing_local = block_gt.affine().inverse().transform_point3(hit_pos);
                st.rubbing = None;
                st.phase = 0.0;
                consumed.0 = true;
                status.0 = "Rubbing the stick up and down in the V notch. Move the mouse up and down; wood dust and smoke will come out."
                    .into();
            }
            return;
        }
        return;
    }

    // no stick in hand: a quick second click on the same stick picks it up
    if tools.any() || grab.entity.is_some() {
        return;
    }
    let Ok((_, hit_stick, _)) = sticks.get(hit) else {
        return;
    };
    if hit_stick.equipped {
        return;
    }
    let now = time.elapsed_seconds();
    match st.last_click {
        Some((e, t)) if e == hit && now - t < 0.45 => {
            hold_stick_physics(&mut commands, hit);
            if let Ok((_, mut s, _)) = sticks.get_mut(hit) {
                s.equipped = true;
            }
            st.last_click = None;
            consumed.0 = true;
            status.0 = "Stick in your hand. Kneel (hold C) beside another stick or a notched block, click it and hold to rub for fire. Q puts it down."
                .into();
        }
        _ => st.last_click = Some((hit, now)),
    }
}

/// Holds the stick in your hand, and plays out the rubbing: kneeling beside the other stick
/// you rub yours across it, it smokes and sparks, and at full heat a fire starts right there.
fn stick_pose_system(
    time: Res<Time>,
    mut commands: Commands,
    kneel: Res<Kneel>,
    mut st: ResMut<StickState>,
    mut status: ResMut<Status>,
    assets: Res<FireAssets>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion: EventReader<MouseMotion>,
    cameras: Query<&GlobalTransform, With<Camera3d>>,
    mut sticks: Query<
        (Entity, &mut Stick, &mut Transform),
        Without<BurnedDustPile>,          // <-- new
    >,
    blocks: Query<&GlobalTransform, With<KnifeNotched>>, // <-- new
    mut piles: Query<(Entity, &mut BurnedDustPile, &mut Transform), Without<Stick>>,
) {
    let dt = time.delta_seconds().clamp(1e-3, 0.05);
    let Ok(cam_gt) = cameras.get_single() else { return };
    let (_, rot, cam_pos) = cam_gt.to_scale_rotation_translation();
    let shoulder = cam_pos + rot * Vec3::new(0.3, -0.45, 0.1);

    let mut mouse = 0.0;
    let mut mouse_y = 0.0;
    for ev in motion.read() {
        mouse += ev.delta.length();
        mouse_y += ev.delta.y.abs();          // <-- vertical gesture, for the notch
    }
    let mouse_f = (mouse / 40.0).clamp(0.0, 1.0);
    let mouse_yf = (mouse_y / 25.0).clamp(0.0, 1.0);
    st.grips = [None, None];
    if st.rubbing_block.is_none() {
        st.burned_pile = None; // not rubbing: the pile just sits there on its own
    }

    let equipped = sticks.iter().find(|(_, s, _)| s.equipped).map(|(e, _, _)| e);
    let Some(eq) = equipped else {
        st.heat = (st.heat - 0.5 * dt).max(0.0);
        if let Some(other) = st.rubbing.take() {
            restore_stick_physics(&mut commands, other, Vec3::ZERO);
        }
        st.rubbing_block = None;
        return;
    };

    // stop rubbing a stick if you let go, stand up or move away
    if let Some(target) = st.rubbing {
        let too_far = sticks
            .get(target)
            .map_or(true, |(_, _, t)| shoulder.distance(t.translation) > 1.7);
        if !buttons.pressed(MouseButton::Left) || kneel.amount < 0.5 || too_far {
            restore_stick_physics(&mut commands, target, Vec3::ZERO);
            st.rubbing = None;
        }
    }
    // stop rubbing a block the same way
    if let Some(target) = st.rubbing_block {
        let too_far = blocks
            .get(target)
            .map_or(true, |gt| shoulder.distance(gt.translation()) > 1.7);
        if !buttons.pressed(MouseButton::Left) || kneel.amount < 0.5 || too_far {
            st.rubbing_block = None;
        }
    }

    // ---- existing stick-on-stick rubbing (unchanged) ----
    if let Some(target) = st.rubbing {
        if let Ok((_, _, tt)) = sticks.get(target) {
            // ... exactly the existing code, ending with `return;` ...
            // (leave this whole branch as it was)
        }
        st.rubbing = None;
    }

        // ---- rubbing the stick in the V notch of a wood chip ----
    if let Some(target) = st.rubbing_block {
        if let Ok(block_gt) = blocks.get(target) {
            let notch_point = block_gt.transform_point(st.rubbing_local);

            // Make sure a burned-dust pile exists on this notch.
            let pile = match st.burned_pile {
                Some((b, p)) if b == target => p,
                _ => {
                    // Keep building a pile already sitting at this spot, else start a new one.
                    let existing = piles
                        .iter()
                        .filter(|(_, _, tf)| tf.translation.distance(notch_point) < PILE_REUSE_RADIUS)
                        .min_by(|a, b| {
                            a.2.translation
                                .distance(notch_point)
                                .partial_cmp(&b.2.translation.distance(notch_point))
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(e, _, _)| e);
                    let chosen = match existing {
                        Some(e) => e,
                        None => {
                            if piles.iter().count() >= MAX_BURNED_PILES {
                                let oldest = piles
                                    .iter()
                                    .min_by(|a, b| {
                                        a.1.born
                                            .partial_cmp(&b.1.born)
                                            .unwrap_or(std::cmp::Ordering::Equal)
                                    })
                                    .map(|(e, _, _)| e);
                                if let Some(e) = oldest {
                                    commands.entity(e).despawn_recursive();
                                }
                            }
                            let seed = st
                                .seed
                                .wrapping_add(12345)
                                .wrapping_add(time.elapsed_seconds().to_bits());
                            spawn_burned_pile(
                                &mut commands,
                                &assets,
                                notch_point,
                                seed,
                                time.elapsed_seconds(),
                            )
                        }
                    };
                    st.burned_pile = Some((target, chosen));
                    chosen
                }
            };

            // The hand holds the far end; the tip bobs up and down.
            let bob = st.phase.sin() * 0.045;
            let tip = notch_point + Vec3::Y * bob;
            let dir = (tip - shoulder).normalize_or_zero();
            let dir = if dir == Vec3::ZERO { rot * Vec3::NEG_Z } else { dir };
            let origin = tip - dir * (STICK_LENGTH * 0.5);
            if let Ok((_, _, mut et)) = sticks.get_mut(eq) {
                et.translation = origin;
                et.rotation = Quat::from_rotation_arc(Vec3::Y, dir);
            }
            st.grips[1] = Some(origin - dir * (STICK_LENGTH * 0.4));
            st.grips[0] = Some(notch_point - rot * Vec3::X * 0.18 + Vec3::Y * 0.05);

            // Gesturing up and down drives the rub: faster mouse = faster rub.
            st.phase += dt * (2.0 + 32.0 * mouse_yf);
            st.heat += dt * (0.12 + 0.35 * mouse_yf);
            let heat = st.heat.min(1.0);

            // The pile grows (and darkens) with the heat.
            if let Ok((_, mut pile_comp, mut pile_tf)) = piles.get_mut(pile) {
                pile_comp.smoke_left = PILE_SMOKE_SECONDS; // rubbing keeps it smoking
                pile_comp.growth = pile_comp.growth.max(heat);
                let base = 0.032 + 0.05 * pile_comp.growth;
                pile_tf.scale =
                    Vec3::new(base, 0.010 + 0.016 * pile_comp.growth, base);
            }

            if mouse_yf > 0.05 {
                st.smoke_timer -= dt * (0.6 + 1.4 * mouse_yf);
                while st.smoke_timer <= 0.0 {
                    st.smoke_timer += 0.14 - 0.08 * heat;
                    st.seed = st.seed.wrapping_add(1);
                    let j = pseudo_rand(st.seed) * 2.0 - 1.0;
                    let k2 = pseudo_rand(st.seed.wrapping_add(3)) * 2.0 - 1.0;

                    // A pale flake of wood dust falls out of the notch.
                    spawn_fire_particle(
                        &mut commands,
                        &assets,
                        assets.dust.clone(),
                        notch_point + Vec3::Y * 0.01,
                        Vec3::new(j * 0.12, -0.1 - 0.1 * (k2 * 0.5 + 0.5), k2 * 0.12),
                        1.1,
                        0.012 + 0.012 * heat,
                        0.2,
                        -1.2,
                    );
                    // Extra smoke right off the pile.
                    if heat > 0.15 {
                        spawn_fire_particle(
                            &mut commands,
                            &assets,
                            assets.smoke.clone(),
                            notch_point + Vec3::Y * 0.03,
                            Vec3::new(j * 0.05, 0.22 + 0.2 * heat, k2 * 0.05),
                            1.6,
                            0.018 + 0.02 * heat,
                            3.0,
                            0.15,
                        );
                    }
                }
                st.msg_timer -= dt;
                if st.msg_timer <= 0.0 {
                    st.msg_timer = 0.3;
                    status.0 = format!(
                        "Rubbing the stick in the notch... {}% (move the mouse up and down)",
                        (heat * 100.0) as i32
                    );
                }
            }
            return;
        }
        st.rubbing_block = None;
    }
    // ---- just holding it, ready in your right hand ----
    let grip = cam_pos
        + rot * Vec3::new(0.28, -0.27, -0.5)
        + Vec3::Y * ((time.elapsed_seconds() * 1.7).sin() * 0.004);
    let dir = (rot * Vec3::new(-0.15, 0.55, -0.82)).normalize();
    if let Ok((_, _, mut et)) = sticks.get_mut(eq) {
        let k = (dt * 20.0).min(1.0);
        et.translation = et.translation.lerp(grip + dir * (STICK_LENGTH * 0.3), k);
        et.rotation = et.rotation.slerp(Quat::from_rotation_arc(Vec3::Y, dir), k);
    }
    st.grips[1] = Some(grip);
    st.heat = (st.heat - 0.4 * dt).max(0.0);
}

fn spawn_fire(commands: &mut Commands, a: &FireAssets, pos: Vec3, fuel: f32, seed: u32) {
    let root = commands
        .spawn(SpatialBundle::from_transform(Transform::from_translation(pos)))
        .id();
    let mut tongues = Vec::new();
    let mut glow = Entity::PLACEHOLDER;
    let mut light = Entity::PLACEHOLDER;
    commands.entity(root).with_children(|p| {
        // a few charred logs under the flames
        for i in 0..3u32 {
            let ang = i as f32 / 3.0 * std::f32::consts::TAU + pseudo_rand(seed.wrapping_add(i)) * 0.8;
            p.spawn(PbrBundle {
                mesh: a.cylinder.clone(),
                material: a.charred.clone(),
                transform: Transform {
                    translation: Vec3::new(ang.cos() * 0.08, 0.03, ang.sin() * 0.08),
                    rotation: Quat::from_rotation_y(-ang) * Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
                    scale: Vec3::new(0.035, 0.4, 0.035),
                },
                ..default()
            });
        }
        // a glowing bed of embers
        glow = p
            .spawn(PbrBundle {
                mesh: a.ball.clone(),
                material: a.ember.clone(),
                transform: Transform::from_xyz(0.0, 0.025, 0.0).with_scale(Vec3::new(0.4, 0.03, 0.4)),
                ..default()
            })
            .id();
        // flame tongues: 8 orange outside, 4 yellow, 2 pale in the middle
        for (layer, count) in [(0usize, 8u32), (1, 4), (2, 2)] {
            let material = [&a.outer, &a.mid, &a.core][layer];
            for i in 0..count {
                let n = seed.wrapping_add(layer as u32 * 31 + i * 7);
                let entity = p
                    .spawn(PbrBundle {
                        mesh: a.ball.clone(),
                        material: material.clone(),
                        transform: Transform::from_scale(Vec3::splat(0.001)),
                        ..default()
                    })
                    .id();
                tongues.push(Tongue {
                    entity,
                    angle: i as f32 / count as f32 * std::f32::consts::TAU + pseudo_rand(n) * 0.6,
                    phase: pseudo_rand(n.wrapping_add(1)) * std::f32::consts::TAU,
                    speed: 5.0 + pseudo_rand(n.wrapping_add(2)) * 5.0,
                    layer,
                });
            }
        }
        light = p
            .spawn(PointLightBundle {
                point_light: PointLight {
                    intensity: 300_000.0,
                    color: Color::srgb(1.0, 0.6, 0.28),
                    range: 8.0,
                    shadows_enabled: false,
                    ..default()
                },
                transform: Transform::from_xyz(0.0, 0.5, 0.0),
                ..default()
            })
            .id();
    });
    commands.entity(root).insert(Fire {
        fuel,
        size: 0.0,
        flare: 1.0,
        age: 0.0,
        smoke_timer: 0.0,
        spark_timer: 0.0,
        seed,
        light,
        glow,
        tongues,
    });
}

/// Burns wood that lands in it, grows with its fuel, flickers, smokes, throws sparks, and
/// is weakened by rain and water. When the fuel is gone it leaves a patch of ash.
fn fire_system(
    time: Res<Time>,
    mut commands: Commands,
    grab: Res<Grab>,
    sticks_state: Res<StickState>,
    assets: Res<FireAssets>,
    climate: Res<Climate>,
    mut fires: Query<(Entity, &mut Fire, &Transform)>,
    mut tfs: Query<&mut Transform, (Without<Fire>, Without<SplashParticle>)>,
    mut lights: Query<&mut PointLight>,
    splashes: Query<&Transform, (With<SplashParticle>, Without<Fire>)>,
    sticks: Query<(Entity, &GlobalTransform, &Stick)>,
    felled: Query<(Entity, &GlobalTransform, &FelledBranch)>,
    broken: Query<(Entity, &GlobalTransform, &TreeBranch)>,
) {
    let dt = time.delta_seconds().clamp(1e-3, 0.05);
    let house_half = FOOT * 0.5 + 0.4;
    let drops: Vec<Vec3> = splashes.iter().take(160).map(|t| t.translation).collect();
    let mut eaten: Vec<Entity> = Vec::new();

    for (entity, mut fire, ftf) in &mut fires {
        let pos = ftf.translation;
        fire.age += dt;
        let radius = 0.12 + 0.18 * fire.size;

        // wood that lands in the fire is burnt: more fuel, a flare of sparks
        let mut added = 0.0;
        let mut wood: Vec<(Entity, Vec3, f32)> = Vec::new();
        for (e, gt, s) in &sticks {
            if !s.equipped && sticks_state.rubbing != Some(e) && grab.entity != Some(e) {
                wood.push((e, gt.translation(), 0.4));
            }
        }
        for (e, gt, f) in &felled {
            if f.detached && grab.entity != Some(e) {
                wood.push((e, gt.translation(), 0.9));
            }
        }
        for (e, gt, b) in &broken {
            if b.broken && grab.entity != Some(e) {
                wood.push((e, gt.translation(), 0.9));
            }
        }
        for (e, p, fuel_value) in wood {
            let reach = if fuel_value > 0.5 { 0.9 } else { radius + 0.25 };
            if !eaten.contains(&e)
                && Vec2::new(p.x - pos.x, p.z - pos.z).length() < reach
                && (p.y - pos.y).abs() < 1.2
            {
                eaten.push(e);
                added += fuel_value;
                commands.entity(e).despawn_recursive();
            }
        }
        if added > 0.0 {
            fire.fuel = (fire.fuel + added).min(FIRE_MAX_FUEL);
            fire.flare = 1.0;
            for _ in 0..8 {
                fire.seed = fire.seed.wrapping_add(1);
                let a = pseudo_rand(fire.seed) * std::f32::consts::TAU;
                let v = 0.6 + pseudo_rand(fire.seed.wrapping_add(2)) * 0.8;
                spawn_fire_particle(
                    &mut commands,
                    &assets,
                    assets.spark.clone(), 
                    pos + Vec3::Y * 0.1,
                    Vec3::new(a.cos() * v * 0.6, 1.4 + v, a.sin() * v * 0.6),
                    0.9,
                    0.012,
                    0.0,
                    -3.0,
                );
            }
        }

        // water and rain
        for d in &drops {
            if Vec2::new(d.x - pos.x, d.z - pos.z).length() < radius + 0.25 && d.y - pos.y < 1.5 {
                fire.fuel = (fire.fuel - 0.004).max(0.0);
            }
        }
        let outdoors = !(pos.x.abs() < house_half && pos.z.abs() < house_half);
        let rain = if climate.storm && outdoors { 4.0 } else { 1.0 };
        let burn = (0.012 + 0.014 * fire.size) * rain * dt;
        fire.fuel = (fire.fuel - burn).max(0.0);

        // size follows fuel
        let target = if fire.fuel > 0.0 { (fire.fuel.sqrt() * 0.8).clamp(0.15, 2.2) } else { 0.0 };
        fire.size += (target - fire.size) * (1.0 - (-2.0 * dt).exp());
        fire.flare = (fire.flare - dt * 1.5).max(0.0);
        let s = fire.size * (1.0 + 0.35 * fire.flare);
        let vis = (s / 0.15).min(1.0);
        let t = fire.age;

        // flames
        let r = 0.1 + 0.16 * s;
        for tongue in fire.tongues.iter() {
            let layer_scale = [1.0_f32, 0.72, 0.45][tongue.layer];
            let layer_radius = [0.75_f32, 0.45, 0.15][tongue.layer];
            let flicker = 0.75
                + 0.25 * (t * tongue.speed + tongue.phase).sin()
                + 0.1 * (t * tongue.speed * 2.7 + tongue.phase * 1.7).sin();
            let h = ((0.25 + 0.55 * s) * layer_scale * flicker).max(0.001) * vis;
            let w = (0.14 + 0.1 * s) * layer_scale * vis;
            if let Ok(mut tf) = tfs.get_mut(tongue.entity) {
                tf.translation = Vec3::new(
                    tongue.angle.cos() * layer_radius * r,
                    0.02 + h * 0.5,
                    tongue.angle.sin() * layer_radius * r,
                );
                tf.scale = Vec3::new(w.max(0.001), h.max(0.001), w.max(0.001));
                tf.rotation = Quat::from_rotation_x((t * 1.7 + tongue.phase).sin() * 0.14)
                    * Quat::from_rotation_z((t * 1.3 + tongue.phase).cos() * 0.14);
            }
        }
        if let Ok(mut tf) = tfs.get_mut(fire.glow) {
            tf.scale = Vec3::new((r * 2.6).max(0.001) * vis, 0.03, (r * 2.6).max(0.001) * vis);
        }
        let flicker = 0.85 + 0.15 * (t * 11.0).sin() + 0.08 * (t * 23.0).sin();
        if let Ok(mut l) = lights.get_mut(fire.light) {
            l.intensity = (150_000.0 + 450_000.0 * s) * flicker * vis;
            l.range = 6.0 + 6.0 * s;
        }

        // smoke and sparks
        fire.smoke_timer -= dt;
        if fire.smoke_timer <= 0.0 && fire.size > 0.05 {
            fire.smoke_timer = 0.28 / (0.3 + s);
            fire.seed = fire.seed.wrapping_add(1);
            let j = pseudo_rand(fire.seed) * 2.0 - 1.0;
            let k = pseudo_rand(fire.seed.wrapping_add(3)) * 2.0 - 1.0;
            spawn_fire_particle(
                &mut commands,
                &assets,
                assets.smoke.clone(), 
                pos + Vec3::new(j * 0.1, 0.35 + 0.5 * s, k * 0.1),
                Vec3::new(j * 0.08, 0.4 + 0.3 * s, k * 0.08),
                2.5 + s,
                0.1 + 0.1 * s,
                3.0,
                0.1,
            );
        }
        fire.spark_timer -= dt;
        if fire.spark_timer <= 0.0 && fire.size > 0.2 {
            fire.spark_timer = 0.5 / (0.4 + s * 1.5);
            fire.seed = fire.seed.wrapping_add(1);
            let a = pseudo_rand(fire.seed) * std::f32::consts::TAU;
            let v = pseudo_rand(fire.seed.wrapping_add(2));
            spawn_fire_particle(
                &mut commands,
                &assets,
                assets.smoke.clone(), 
                pos + Vec3::new(a.cos() * r * 0.4, 0.3 + 0.3 * s, a.sin() * r * 0.4),
                Vec3::new(a.cos() * 0.3, 1.0 + v * 1.0, a.sin() * 0.3),
                0.8 + v * 0.6,
                0.01,
                0.0,
                -0.5,
            );
        }

        // burnt out: leave ash behind
        if fire.fuel <= 0.0 && fire.size < 0.04 {
            commands.spawn(PbrBundle {
                mesh: assets.cylinder.clone(),
                material: assets.ash.clone(),
                transform: Transform::from_translation(pos + Vec3::Y * 0.005)
                    .with_scale(Vec3::new(0.3, 0.008, 0.3)),
                ..default()
            });
            commands.entity(entity).despawn_recursive();
        }
    }
}

fn fire_particle_system(
    time: Res<Time>,
    mut commands: Commands,
    mut particles: Query<(Entity, &mut FireParticle, &mut Transform)>,
) {
    let dt = time.delta_seconds().min(0.05);
    for (entity, mut p, mut tf) in &mut particles {
        p.life -= dt;
        if p.life <= 0.0 {
            commands.entity(entity).despawn();
            continue;
        }
        let drift = p.velocity;
        tf.translation += drift * dt;
        let buoyancy = p.buoyancy;
        p.velocity.y += buoyancy * dt;
        let f = 1.0 - p.life / p.max_life;
        // smoke swells then fades away; sparks just shrink
        tf.scale = Vec3::splat((p.size0 * (1.0 + p.grow * f) * (1.0 - f).powf(0.7)).max(0.0005));
    }
}

// ---------------------------------------------------------------------------
// Mud: pull it out of the river banks, carry it (it drips), splat it anywhere
// ---------------------------------------------------------------------------

/// A low, slightly lumpy dome of radius 1 (height is set through the Y scale).
fn mud_splat_mesh() -> Mesh {
    const N: usize = 18;
    let mut positions: Vec<[f32; 3]> = vec![[0.0, 0.16, 0.0]];
    let mut normals: Vec<[f32; 3]> = vec![[0.0, 1.0, 0.0]];
    let mut ring = |scale: f32, height: f32, tilt: f32, positions: &mut Vec<[f32; 3]>| {
        for i in 0..N {
            let a = i as f32 / N as f32 * std::f32::consts::TAU;
            let lump = 0.8 + 0.2 * pseudo_rand(i as u32 * 7 + 3) + 0.08 * (a * 3.0).sin();
            let (x, z) = (a.cos() * lump * scale, a.sin() * lump * scale);
            positions.push([x, height, z]);
            normals.push(Vec3::new(x * tilt, 1.0, z * tilt).normalize().to_array());
        }
    };
    ring(0.55, 0.11, 0.5, &mut positions); // inner ring: ring 1 = indices 1..=N
    ring(1.0, 0.0, 1.1, &mut positions); // outer ring: indices N+1..=2N
    let mut indices: Vec<u32> = Vec::new();
    for i in 0..N {
        let j = (i + 1) % N;
        let (a, b) = (1 + i as u32, 1 + j as u32);
        let (c, d) = (1 + N as u32 + i as u32, 1 + N as u32 + j as u32);
        indices.extend_from_slice(&[0, b, a]); // cap
        indices.extend_from_slice(&[a, b, d, a, d, c]); // slope
    }
    let uvs = vec![[0.0, 0.0]; positions.len()];
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(bevy::render::mesh::Indices::U32(indices));
    mesh
}

fn setup_mud_assets(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
) {
    commands.insert_resource(MudAssets {
        blob: meshes.add(Sphere::new(MUD_BLOB_RADIUS)),
        drip: meshes.add(Sphere::new(1.0)),
        splat: meshes.add(mud_splat_mesh()),
        cylinder: meshes.add(Cylinder::new(1.0, 1.0)),
        material: mats.add(StandardMaterial {
            base_color: Color::srgb(0.24, 0.15, 0.08),
            perceptual_roughness: 0.3,
            reflectance: 0.5,
            ..default()
        }),
    });
}

fn setup_water_assets(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
) {
    commands.insert_resource(WaterAssets {
        blob: meshes.add(Sphere::new(MUD_BLOB_RADIUS)),
        drop: meshes.add(Sphere::new(1.0)),
        material: mats.add(StandardMaterial {
            base_color: Color::srgba(0.25, 0.6, 0.95, 0.72),
            perceptual_roughness: 0.05,
            reflectance: 0.6,
            alpha_mode: AlphaMode::Blend,
            ..default()
        }),
    });
}

fn spawn_mud_blob(
    commands: &mut Commands,
    assets: &MudAssets,
    position: Vec3,
    amount: f32,
    seed: u32,
    no_absorb: f32,
) -> Entity {
    let transform = Transform::from_translation(position).with_scale(Vec3::splat(amount.cbrt()));
    commands
        .spawn((
            PbrBundle {
                mesh: assets.blob.clone(),
                material: assets.material.clone(),
                transform,
                // so the hand sees the right position on the very first frame
                global_transform: GlobalTransform::from(transform),
                ..default()
            },
            RigidBody::Dynamic,
            Collider::ball(MUD_BLOB_RADIUS),
            ColliderMassProperties::Density(400.0),
            Velocity::default(),
            Restitution::coefficient(0.0),
            Friction::coefficient(0.4), // it slides over the ground instead of gripping
            LockedAxes::ROTATION_LOCKED, // mud doesn't roll like a ball
            Sleeping::disabled(),
            // the same extras ensure_grab_components would add, already in the "held" state
            ReadMassProperties::default(),
            ExternalForce::default(),
            Damping {
                linear_damping: HELD_LINEAR_DAMPING,
                angular_damping: HELD_ANGULAR_DAMPING,
            },
            MudBlob {
                amount,
                drip_timer: 0.0,
                free_time: 0.0,
                seed,
                no_absorb,
            },
        ))
        .id()
}

/// Sticks a patch of mud onto a surface. If the surface belongs to a physics object the
/// patch becomes its child, so it travels with it.
fn place_mud(
    commands: &mut Commands,
    assets: &MudAssets,
    state: &mut MudState,
    point: Vec3,
    normal: Vec3,
    radius: f32,
    amount: f32,
    parent: Option<(Entity, &GlobalTransform)>,
) {
    state.seed = state.seed.wrapping_add(1);
    let n = normal.normalize_or_zero();
    let n = if n == Vec3::ZERO { Vec3::Y } else { n };
    let rotation = Quat::from_rotation_arc(Vec3::Y, n)
        * Quat::from_rotation_y(pseudo_rand(state.seed) * std::f32::consts::TAU);
    let scale = Vec3::new(
        radius * (0.9 + pseudo_rand(state.seed.wrapping_add(11)) * 0.25),
        radius,
        radius * (0.9 + pseudo_rand(state.seed.wrapping_add(23)) * 0.25),
    );
    let world = Mat4::from_scale_rotation_translation(scale, rotation, point + n * 0.006);
    let transform = match parent {
        Some((_, parent_gt)) => Transform::from_matrix(parent_gt.compute_matrix().inverse() * world),
        None => Transform::from_matrix(world),
    };
    let base_scale = transform.scale;
    let id = commands
        .spawn((
            PbrBundle {
                mesh: assets.splat.clone(),
                material: assets.material.clone(),
                transform,
                ..default()
            },
            MudPatch {
                amount,
                radius,
                base_scale,
                wash: 0.0,
            },
        ))
        .id();
    if let Some((parent_entity, _)) = parent {
        commands.entity(parent_entity).add_child(id);
    }
    state.patches.push_back(id);
    while state.patches.len() > MUD_MAX_PATCHES {
        if let Some(old) = state.patches.pop_front() {
            commands.entity(old).despawn_recursive();
        }
    }
}

/// Walks up from a hit collider to the physics body (something with a Velocity) it belongs to.
fn dynamic_body_of(
    mut entity: Entity,
    dynamic: &Query<(), With<Velocity>>,
    parents: &Query<&Parent>,
) -> Option<Entity> {
    loop {
        if dynamic.get(entity).is_ok() {
            return Some(entity);
        }
        entity = parents.get(entity).ok()?.get();
    }
}

/// Click on river mud (or on a mud splat you made) to pull a glob of it out and hold it.
/// Runs just before grab_system; anything physical under the cursor still wins.
fn mud_pick_system(
    mut commands: Commands,
    mut grab: ResMut<Grab>,
    mut consumed: ResMut<ClickConsumed>,
    mut state: ResMut<MudState>,
    mut status: ResMut<Status>,
    assets: Res<MudAssets>,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    rapier: Res<RapierContext>,
    player: Query<Entity, With<Player>>,
    ground: Query<(), With<DiggableGround>>,
    patches: Query<(Entity, &GlobalTransform, &MudPatch)>,
) {
    if !buttons.just_pressed(MouseButton::Left) || consumed.0 || grab.entity.is_some() {
        return;
    }
    let (Ok(window), Ok((camera, cam_tf))) = (windows.get_single(), cameras.get_single()) else {
        return;
    };
    let Some(ray) = pointer_position(&window)
        .and_then(|cursor| camera.viewport_to_world(cam_tf, cursor))
    else {
        return;
    };
    let dir = *ray.direction;
    let Some((hit, toi)) = rapier.cast_ray(
        ray.origin,
        dir,
        MUD_GRAB_REACH,
        true,
        world_ray_filter(player.get_single().ok()),
    ) else {
        return;
    };
    // If a physics object is the nearest thing under the cursor, the normal hand takes it.
    if let Some((_, dyn_toi)) =
        rapier.cast_ray(ray.origin, dir, MUD_GRAB_REACH, true, QueryFilter::only_dynamic())
    {
        if dyn_toi <= toi + 0.02 {
            return;
        }
    }

    let point = ray.origin + dir * toi;
    let on_ground = ground.get(hit).is_ok();
    let normal = if on_ground {
        terrain_normal(point.x, point.z)
    } else {
        -dir
    };

    // 1) a splat you made earlier?
    let mut best: Option<(Entity, f32, f32)> = None; // entity, amount, distance
    for (entity, gt, patch) in &patches {
        let distance = gt.translation().distance(point);
        let live = 1.0 - patch.wash; // a half-washed patch is smaller and holds less
        if patch.radius * live >= 0.09
            && distance < patch.radius * live * 1.15 + 0.05
            && best.map_or(true, |(_, _, d)| distance < d)
        {
            best = Some((entity, patch.amount * live, distance));
        }
    }
    let amount = if let Some((entity, amount, _)) = best {
        state.patches.retain(|e| *e != entity);
        commands.entity(entity).despawn_recursive();
        amount
    } else if on_ground && mud_strength(point.x, point.z) > 0.4 {
        1.0
    } else {
        return;
    };

    state.seed = state.seed.wrapping_add(1);
    let spawn_at = point + normal * (MUD_BLOB_RADIUS * amount.cbrt() + 0.03);
    let blob = spawn_mud_blob(&mut commands, &assets, spawn_at, amount, state.seed, 0.0);
    grab.entity = Some(blob);
    grab.local_anchor = Vec3::ZERO;
    grab.distance = toi.max(0.8);
    consumed.0 = true;
    status.0 = "Mud! It drips while you carry it. Let go to splat it, click a splat to pick it up again."
        .into();
}

/// Carried / flying mud: drips, shrinks, and splats when it hits something after release.
fn mud_blob_system(
    time: Res<Time>,
    mut commands: Commands,
    mut grab: ResMut<Grab>,
    mut state: ResMut<MudState>,
    assets: Res<MudAssets>,
    rapier: Res<RapierContext>,
    player: Query<Entity, With<Player>>,
    ground: Query<(), With<DiggableGround>>,
    dynamic: Query<(), With<Velocity>>,
    parents: Query<&Parent>,
    gts: Query<&GlobalTransform>,
    other_blobs: Query<(), Or<(With<MudBlob>, With<WaterBlob>)>>,
    bucket_ids: Query<(), With<Bucket>>,
    river: Res<RiverState>,
    mut buckets: Query<(&Transform, &mut Bucket), Without<MudBlob>>,
    mut blobs: Query<(Entity, &mut MudBlob, &mut Transform, &Velocity)>,
) {
    let dt = time.delta_seconds().min(0.05);
    let player_entity = player.get_single().ok();
    for (entity, mut blob, mut tf, vel) in &mut blobs {
        let held = grab.entity == Some(entity);
        let speed = vel.linvel.length();
        let radius = MUD_BLOB_RADIUS * tf.scale.x;
        blob.no_absorb = (blob.no_absorb - dt).max(0.0);

        // water washes it away: a glob dunked in the river dissolves in a couple of seconds
        let in_river = (tf.translation.x - river_center_x(tf.translation.z)).abs()
            < RIVER_WIDTH * 0.5 + river.overflow
            && tf.translation.y - radius * 0.4 < river.water_level;
        if in_river {
            blob.amount -= 0.5 * dt;
            blob.seed = blob.seed.wrapping_add(1);
            if pseudo_rand(blob.seed) < dt * 12.0 {
                let at = Vec3::new(tf.translation.x, river.water_level + 0.01, tf.translation.z);
                spawn_mud_puff(&mut commands, &assets, at, blob.seed);
            }
        }

        // dripping: faster when it is being swung around
        blob.drip_timer += dt * (3.5 + speed * 1.2);
        while blob.drip_timer >= 1.0 {
            blob.drip_timer -= 1.0;
            blob.seed = blob.seed.wrapping_add(1);
            let jx = pseudo_rand(blob.seed) * 2.0 - 1.0;
            let jz = pseudo_rand(blob.seed.wrapping_add(5)) * 2.0 - 1.0;
            let size = 0.022 + pseudo_rand(blob.seed.wrapping_add(9)) * 0.016;
            commands.spawn((
                PbrBundle {
                    mesh: assets.drip.clone(),
                    material: assets.material.clone(),
                    transform: Transform::from_translation(
                        tf.translation + Vec3::new(jx * 0.6 * radius, -0.8 * radius, jz * 0.6 * radius),
                    )
                    .with_scale(Vec3::splat(size)),
                    ..default()
                },
                MudDrip {
                    velocity: vel.linvel * 0.35 + Vec3::new(jx * 0.2, -0.3, jz * 0.2),
                    size,
                    water: false,
                },
            ));
            blob.amount -= 0.012;
        }
        if blob.amount < 0.06 {
            if held {
                grab.entity = None;
            }
            commands.entity(entity).despawn_recursive();
            continue;
        }
        tf.scale = Vec3::splat(blob.amount.cbrt());

        if held {
            blob.free_time = 0.0;
            continue;
        }
        blob.free_time += dt;
        if tf.translation.y < -6.0 {
            commands.entity(entity).despawn_recursive();
            continue;
        }

        // let go above an upright bucket: the mud drops in and stays there
        if blob.no_absorb <= 0.0 {
            for (bucket_tf, mut bucket) in &mut buckets {
                let up = bucket_tf.rotation * Vec3::Y;
                if up.y < 0.35 || bucket.mud >= 0.999 {
                    continue;
                }
                let local = bucket_tf.rotation.inverse() * (tf.translation - bucket_tf.translation);
                let lateral = Vec2::new(local.x, local.z).length();
                if local.y > BUCKET_BASE_T
                    && local.y < BUCKET_HEIGHT + 0.3
                    && lateral < BUCKET_TOP_R * 0.8
                {
                    let room = (1.0 - bucket.mud) / BUCKET_MUD_PER_BLOB;
                    let taken = blob.amount.min(room);
                    bucket.mud += taken * BUCKET_MUD_PER_BLOB;
                    blob.amount -= taken;
                    break;
                }
            }
            if blob.amount < 0.06 {
                commands.entity(entity).despawn_recursive();
                continue;
            }
            tf.scale = Vec3::splat(blob.amount.cbrt());
        }

        // let go: splat on whatever it touches (look a little way ahead along its motion)
        // Two probes: straight down (so a glob that is lying or rolling on the ground is
        // noticed, however fast it moves sideways) and along its motion (walls, flying into things).
        let filter = world_ray_filter(player_entity).exclude_rigid_body(entity);
        let mut probes = vec![(Vec3::NEG_Y, radius + 0.06)];
        if speed > 0.5 {
            probes.push((vel.linvel / speed, radius + speed * dt * 2.0 + 0.05));
        }
        let mut splat: Option<(Vec3, Vec3, Option<Entity>)> = None;
        for (dir, reach) in probes {
            let Some((hit, toi)) = rapier.cast_ray(tf.translation, dir, reach, true, filter) else {
                continue;
            };
            let body = dynamic_body_of(hit, &dynamic, &parents);
            let own_bucket = blob.no_absorb > 0.0 && body.is_some_and(|b| bucket_ids.get(b).is_ok());
            if other_blobs.get(hit).is_ok() || own_bucket {
                continue;
            }
            let point = tf.translation + dir * toi;
            let normal = if ground.get(hit).is_ok() {
                terrain_normal(point.x, point.z)
            } else {
                -dir
            };
            splat = Some((point, normal, body));
            break;
        }
        if splat.is_none() && blob.free_time > 8.0 {
            splat = Some((tf.translation, Vec3::Y, None)); // never landed: give up and drop it here
        }
        if let Some((point, normal, body)) = splat {
            let spread = 1.0 + speed.min(12.0) * 0.04;
            let parent = body
                .filter(|b| *b != entity)
                .and_then(|b| gts.get(b).ok().map(|gt| (b, gt)));
            place_mud(
                &mut commands,
                &assets,
                &mut state,
                point,
                normal,
                (0.08 + 0.18 * blob.amount) * spread,
                blob.amount,
                parent,
            );
            commands.entity(entity).despawn_recursive();
        }
    }
}

/// Falling drops. Where they land they leave a small spot (not over open water).
fn mud_drip_system(
    time: Res<Time>,
    mut commands: Commands,
    mut state: ResMut<MudState>,
    assets: Res<MudAssets>,
    rapier: Res<RapierContext>,
    player: Query<Entity, With<Player>>,
    ground: Query<(), With<DiggableGround>>,
    dynamic: Query<(), With<Velocity>>,
    parents: Query<&Parent>,
    gts: Query<&GlobalTransform>,
    blobs: Query<(), Or<(With<MudBlob>, With<WaterBlob>)>>,
    water: Res<WaterAssets>,
    mut drips: Query<(Entity, &mut MudDrip, &mut Transform)>,
) {
    let dt = time.delta_seconds().min(0.05);
    let player_entity = player.get_single().ok();
    for (entity, mut drip, mut tf) in &mut drips {
        drip.velocity.y -= WORLD_GRAVITY * dt;
        let step = drip.velocity * dt;
        let distance = step.length();
        if tf.translation.y < -6.0 {
            commands.entity(entity).despawn();
            continue;
        }
        if distance > 1e-5 {
            let dir = step / distance;
            let filter = world_ray_filter(player_entity);
            if let Some((hit, toi)) =
                rapier.cast_ray(tf.translation, dir, distance + drip.size, true, filter)
            {
                if blobs.get(hit).is_err() {
                    let point = tf.translation + dir * toi;
                    let over_water = (point.x - river_center_x(point.z)).abs() < RIVER_WIDTH * 0.5;
                    if drip.water {
                        // a drop of water: a tiny splash (and the ripple is lost in the river)
                        if !over_water {
                            state.seed = state.seed.wrapping_add(1);
                            spawn_water_burst(&mut commands, &water, point, Vec3::Y, 3, 0.5, state.seed);
                        }
                    } else if !over_water {
                        let normal = if ground.get(hit).is_ok() {
                            terrain_normal(point.x, point.z)
                        } else {
                            -dir
                        };
                        let parent = dynamic_body_of(hit, &dynamic, &parents)
                            .and_then(|b| gts.get(b).ok().map(|gt| (b, gt)));
                        let radius = drip.size * 1.7;
                        place_mud(
                            &mut commands,
                            &assets,
                            &mut state,
                            point,
                            normal,
                            radius,
                            0.01,
                            parent,
                        );
                    }
                    commands.entity(entity).despawn();
                    continue;
                }
            }
        }
        tf.translation += step;
        // stretch into a teardrop while falling fast
        let stretch = 1.0 + (drip.velocity.length() * 0.18).min(1.4);
        tf.scale = Vec3::new(drip.size, drip.size * stretch, drip.size);
    }
}

/// Transforms (bucket-local) of the mud pile inside a bucket holding `mud` (0..=1).
fn bucket_mud_transforms(mud: f32) -> (Transform, Transform) {
    let interior = BUCKET_HEIGHT - BUCKET_BASE_T;
    let height = (mud * interior).max(0.004);
    let y_top = BUCKET_BASE_T + height;
    let radius_at = |y: f32| {
        (BUCKET_BOTTOM_R + (BUCKET_TOP_R - BUCKET_BOTTOM_R) * (y / BUCKET_HEIGHT)
            - BUCKET_WALL_T
            - 0.004)
            .max(0.01)
    };
    let body_r = radius_at(BUCKET_BASE_T + height * 0.3);
    let top_r = radius_at(y_top);
    (
        Transform::from_xyz(0.0, BUCKET_BASE_T + height * 0.5, 0.0)
            .with_scale(Vec3::new(body_r, height, body_r)),
        Transform::from_xyz(0.0, y_top - 0.002, 0.0)
            .with_scale(Vec3::new(top_r, top_r * 1.1, top_r)),
    )
}

/// Mud in a bucket: shows the pile, adds its weight (see bucket_scoop_system), and when the
/// bucket is tipped far enough it slops out in globs that fall, splat and stay where they land.
fn bucket_mud_system(
    time: Res<Time>,
    mut commands: Commands,
    mut state: ResMut<MudState>,
    mut status: ResMut<Status>,
    assets: Res<MudAssets>,
    mut mats: ResMut<Assets<StandardMaterial>>,
    water_visuals: Query<(&Parent, &Handle<StandardMaterial>), With<BucketWater>>,
    mut buckets: Query<(Entity, &Transform, &Velocity, &mut Bucket), Without<BucketMudPart>>,
    mut parts: Query<&mut Transform, With<BucketMudPart>>,
) {
    let dt = time.delta_seconds().min(0.05);
    let interior = BUCKET_HEIGHT - BUCKET_BASE_T;
    for (entity, tf, velocity, mut bucket) in &mut buckets {
        // water in the bucket washes the mud out of the pile and turns brown
        if bucket.fill > 0.02 && bucket.mud > 0.0 {
            let washed = ((0.05 + 0.3 * bucket.fill) * dt).min(bucket.mud);
            bucket.mud -= washed;
            bucket.muddy = (bucket.muddy + washed * 1.8).min(1.0);
        } else if bucket.fill < 0.02 && bucket.muddy != 0.0 {
            bucket.muddy = 0.0; // poured out: the next water is clean again
        }
        if bucket.mud <= 0.0 && bucket.mud_debt <= 0.0 && bucket.mud_visual.is_none() {
            continue;
        }
        let up = tf.rotation * Vec3::Y;
        let floor = tf.translation + up * BUCKET_BASE_T;
        let rim_center = tf.translation + up * BUCKET_HEIGHT;
        let tilt_sin = (1.0 - up.y * up.y).max(0.0).sqrt();
        let low_rim = rim_center.y - BUCKET_TOP_R * tilt_sin;
        let upright = up.y > 0.05;
        let axis_y = up.y.max(0.05);

        // mud is thick: it only leaves once its level is above the lowest point of the rim,
        // and slowly (faster the further the bucket is turned over)
        let limit = if upright {
            ((low_rim - floor.y) / axis_y / interior).clamp(0.0, 1.0)
        } else {
            0.0
        };
        if bucket.mud > limit {
            let rate = (0.18 + 0.9 * (1.0 - up.y).clamp(0.0, 2.0)) * dt;
            let take = rate.min(bucket.mud - limit);
            bucket.mud -= take;
            bucket.mud_debt += take;
        }
        if bucket.mud <= 0.0005 {
            bucket.mud = 0.0;
        }

        // hand the mud over as globs
        while bucket.mud_debt >= BUCKET_MUD_CHUNK || (bucket.mud <= 0.0 && bucket.mud_debt > 0.02) {
            let chunk = bucket.mud_debt.min(BUCKET_MUD_CHUNK);
            bucket.mud_debt -= chunk;
            state.seed = state.seed.wrapping_add(1);
            let out = up * up.y - Vec3::Y; // points down the slope of the rim plane
            let dir = if out.length() > 0.05 {
                out.normalize()
            } else {
                // upside down: the mud comes out of the whole mouth
                Quat::from_axis_angle(up, pseudo_rand(state.seed) * std::f32::consts::TAU)
                    * up.any_orthonormal_vector()
            };
            let amount = (chunk / BUCKET_MUD_PER_BLOB).max(0.07);
            let blob = spawn_mud_blob(
                &mut commands,
                &assets,
                rim_center + dir * 0.01,
                amount,
                state.seed,
                1.5,
            );
            commands.entity(blob).insert(Velocity {
                linvel: velocity.linvel * 0.8 + dir * 0.7 + Vec3::NEG_Y * 0.3,
                angvel: Vec3::ZERO,
            });
        }
        if bucket.mud <= 0.0 && bucket.mud_debt <= 0.02 {
            bucket.mud_debt = 0.0;
        }

        // the pile inside the pail
        if bucket.mud > 0.004 {
            let (body_tf, cap_tf) = bucket_mud_transforms(bucket.mud);
            let visual = bucket.mud_visual;
            match visual {
                Some((body, cap)) => {
                    if let Ok(mut t) = parts.get_mut(body) {
                        *t = body_tf;
                    }
                    if let Ok(mut t) = parts.get_mut(cap) {
                        *t = cap_tf;
                    }
                }
                None => {
                    let body = commands
                        .spawn((
                            PbrBundle {
                                mesh: assets.cylinder.clone(),
                                material: assets.material.clone(),
                                transform: body_tf,
                                ..default()
                            },
                            BucketMudPart,
                        ))
                        .id();
                    let cap = commands
                        .spawn((
                            PbrBundle {
                                mesh: assets.splat.clone(),
                                material: assets.material.clone(),
                                transform: cap_tf,
                                ..default()
                            },
                            BucketMudPart,
                        ))
                        .id();
                    commands.entity(entity).add_child(body).add_child(cap);
                    bucket.mud_visual = Some((body, cap));
                }
            }
        } else if let Some((body, cap)) = bucket.mud_visual.take() {
            commands.entity(body).despawn_recursive();
            commands.entity(cap).despawn_recursive();
        }

        let mud_state = if bucket.mud >= 0.9 {
            2
        } else if bucket.mud > 0.02 {
            1
        } else {
            0
        };
        if mud_state != bucket.mud_state {
            status.0 = match (bucket.mud_state, mud_state) {
                (_, 2) => "The bucket is full of mud.".into(),
                (0, _) => "You put mud in the bucket.".into(),
                (_, 0) if bucket.fill > 0.02 => "The water washes the mud out of the bucket.".into(),
                (_, 0) => "The bucket is empty of mud.".into(),
                _ => "Mud slops out of the bucket.".into(),
            };
            bucket.mud_state = mud_state;
        }
    }

    // colour the water inside: clear blue -> muddy brown
    for (parent, handle) in &water_visuals {
        let Ok((_, _, _, mut bucket)) = buckets.get_mut(parent.get()) else {
            continue;
        };
        if (bucket.muddy - bucket.muddy_shown).abs() < 0.01 && bucket.muddy != 0.0 {
            continue;
        }
        if bucket.muddy == bucket.muddy_shown {
            continue;
        }
        bucket.muddy_shown = bucket.muddy;
        let t = bucket.muddy;
        if let Some(material) = mats.get_mut(handle) {
            material.base_color = Color::srgba(
                0.12 + (0.32 - 0.12) * t,
                0.48 + (0.21 - 0.48) * t,
                0.72 + (0.11 - 0.72) * t,
                0.86 + (0.93 - 0.86) * t,
            );
        }
    }
}

fn spawn_mud_puff(commands: &mut Commands, assets: &MudAssets, position: Vec3, seed: u32) {
    let r = |k: u32| pseudo_rand(seed.wrapping_add(k)) * 2.0 - 1.0;
    commands.spawn((
        PbrBundle {
            mesh: assets.drip.clone(),
            material: assets.material.clone(),
            transform: Transform::from_translation(position + Vec3::new(r(1), 0.0, r(2)) * 0.06)
                .with_scale(Vec3::splat(0.035 + 0.02 * (r(3) * 0.5 + 0.5))),
            ..default()
        },
        MudPuff {
            velocity: Vec3::new(r(4) * 0.15, 0.05 + 0.1 * (r(5) * 0.5 + 0.5), r(6) * 0.15),
            life: 0.9,
        },
    ));
}

/// Water washes mud off: splats in the river (or under a flood) dissolve in a few seconds,
/// water poured or splashed onto a splat wears it away, and rain slowly washes the ones
/// that aren't under the roof.
fn mud_wash_system(
    time: Res<Time>,
    mut commands: Commands,
    mut state: ResMut<MudState>,
    assets: Res<MudAssets>,
    river: Res<RiverState>,
    climate: Res<Climate>,
    splashes: Query<&Transform, (With<SplashParticle>, Without<MudPatch>, Without<MudPuff>)>,
    mut patches: Query<(Entity, &GlobalTransform, &mut Transform, &mut MudPatch)>,
    mut puffs: Query<(Entity, &mut MudPuff, &mut Transform), Without<MudPatch>>,
) {
    let dt = time.delta_seconds().min(0.05);
    let drops: Vec<Vec3> = splashes.iter().take(160).map(|t| t.translation).collect();
    let house_half = FOOT * 0.5 + 0.4;

    for (entity, gt, mut tf, mut patch) in &mut patches {
        let p = gt.translation();
        let mut washed = 0.0;
        let in_river = (p.x - river_center_x(p.z)).abs() < RIVER_WIDTH * 0.5 + river.overflow
            && p.y < river.water_level + 0.03;
        if in_river {
            washed += 0.35 * dt;
        }
        if climate.storm && !(p.x.abs() < house_half && p.z.abs() < house_half) {
            washed += 0.02 * dt;
        }
        for drop in &drops {
            if drop.distance_squared(p) < 0.15 * 0.15 {
                washed += 0.006;
            }
        }
        if washed <= 0.0 {
            continue;
        }
        patch.wash += washed;
        if in_river {
            state.seed = state.seed.wrapping_add(1);
            if pseudo_rand(state.seed) < dt * 5.0 {
                let at = Vec3::new(p.x, river.water_level + 0.01, p.z);
                spawn_mud_puff(&mut commands, &assets, at, state.seed);
            }
        }
        if patch.wash >= 1.0 {
            state.patches.retain(|e| *e != entity);
            commands.entity(entity).despawn_recursive();
            continue;
        }
        tf.scale = patch.base_scale * (1.0 - patch.wash).max(0.05);
    }

    for (entity, mut puff, mut tf) in &mut puffs {
        puff.life -= dt;
        let drift = puff.velocity;
        tf.translation += drift * dt;
        puff.velocity *= (1.0 - 1.5 * dt).max(0.0);
        tf.scale *= (1.0 - 1.2 * dt).max(0.0);
        if puff.life <= 0.0 {
            commands.entity(entity).despawn();
        }
    }
}

fn spawn_water_blob(
    commands: &mut Commands,
    water: &WaterAssets,
    position: Vec3,
    amount: f32,
    seed: u32,
) -> Entity {
    let transform = Transform::from_translation(position).with_scale(Vec3::splat(amount.cbrt()));
    commands
        .spawn((
            PbrBundle {
                mesh: water.blob.clone(),
                material: water.material.clone(),
                transform,
                global_transform: GlobalTransform::from(transform),
                ..default()
            },
            RigidBody::Dynamic,
            Collider::ball(MUD_BLOB_RADIUS),
            ColliderMassProperties::Density(300.0),
            Velocity::default(),
            Restitution::coefficient(0.0),
            Friction::coefficient(0.1),
            LockedAxes::ROTATION_LOCKED,
            Sleeping::disabled(),
            ReadMassProperties::default(),
            ExternalForce::default(),
            Damping {
                linear_damping: HELD_LINEAR_DAMPING,
                angular_damping: HELD_ANGULAR_DAMPING,
            },
            WaterBlob {
                amount,
                drip_timer: 0.0,
                free_time: 0.0,
                seed,
            },
        ))
        .id()
}

/// A burst of water droplets thrown up and out from a surface.
fn spawn_water_burst(
    commands: &mut Commands,
    water: &WaterAssets,
    position: Vec3,
    normal: Vec3,
    count: u32,
    strength: f32,
    seed: u32,
) {
    let n = normal.normalize_or_zero();
    let n = if n == Vec3::ZERO { Vec3::Y } else { n };
    for i in 0..count {
        let r = |k: u32| pseudo_rand(seed.wrapping_mul(31).wrapping_add(i * 7 + k)) * 2.0 - 1.0;
        let side = Vec3::new(r(1), r(2), r(3)) * 0.9;
        let velocity = (n * (1.0 + 0.6 * (r(4) * 0.5 + 0.5)) + side) * strength * 1.4;
        commands.spawn((
            PbrBundle {
                mesh: water.drop.clone(),
                material: water.material.clone(),
                transform: Transform::from_translation(position + n * 0.02)
                    .with_scale(Vec3::splat(0.045)),
                ..default()
            },
            SplashParticle {
                velocity,
                lifetime: 0.65,
            },
        ));
    }
}

/// Click and hold on the river to pull a glob of water out of it (the nearest solid thing
/// under the cursor still wins). Runs before mud_pick_system.
fn water_pick_system(
    mut commands: Commands,
    mut grab: ResMut<Grab>,
    mut consumed: ResMut<ClickConsumed>,
    mut status: ResMut<Status>,
    water: Res<WaterAssets>,
    river: Res<RiverState>,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    rapier: Res<RapierContext>,
    player: Query<Entity, With<Player>>,
) {
    if !buttons.just_pressed(MouseButton::Left) || consumed.0 || grab.entity.is_some() {
        return;
    }
    let (Ok(window), Ok((camera, cam_tf))) = (windows.get_single(), cameras.get_single()) else {
        return;
    };
    let Some(ray) = pointer_position(&window)
        .and_then(|cursor| camera.viewport_to_world(cam_tf, cursor))
    else {
        return;
    };
    let dir = *ray.direction;
    if dir.y > -0.001 {
        return;
    }
    // where the cursor ray meets the water surface
    let t = (river.water_level - ray.origin.y) / dir.y;
    if !(0.3..=MUD_GRAB_REACH + 2.0).contains(&t) {
        return;
    }
    let point = ray.origin + dir * t;
    let in_river = (point.x - river_center_x(point.z)).abs() < RIVER_WIDTH * 0.5 + river.overflow
        && point.z.abs() < GROUND_HALF_SIZE;
    if !in_river {
        return;
    }
    // something solid between you and the water (a bucket, a bank...) takes the click instead
    if rapier
        .cast_ray(
            ray.origin,
            dir,
            t - 0.02,
            true,
            world_ray_filter(player.get_single().ok()),
        )
        .is_some()
    {
        return;
    }

    let seed = point.x.to_bits() ^ point.z.to_bits().rotate_left(13);
    let blob = spawn_water_blob(
        &mut commands,
        &water,
        point + Vec3::Y * (MUD_BLOB_RADIUS + 0.02),
        1.0,
        seed,
    );
    grab.entity = Some(blob);
    grab.local_anchor = Vec3::ZERO;
    grab.distance = t;
    grab.bail_reach = 0.0;
    consumed.0 = true;
    status.0 = "Water! It drips while you carry it. Let go over the bucket to fill it, or anywhere to splash it."
        .into();
}

/// Carried / flying water: drips, shrinks, and when let go it fills a bucket it drops into,
/// rejoins the river, or splashes (washing mud off) where it hits something.
fn water_blob_system(
    time: Res<Time>,
    mut commands: Commands,
    mut grab: ResMut<Grab>,
    mut state: ResMut<MudState>,
    water: Res<WaterAssets>,
    river: Res<RiverState>,
    rapier: Res<RapierContext>,
    player: Query<Entity, With<Player>>,
    other_blobs: Query<(), Or<(With<WaterBlob>, With<MudBlob>)>>,
    mut buckets: Query<
        (&Transform, &mut Bucket),
        (Without<WaterBlob>, Without<MudBlob>, Without<MudPatch>),
    >,
    mut patches: Query<(Entity, &GlobalTransform, &mut Transform, &mut MudPatch), Without<WaterBlob>>,
    mut fires: Query<(&Transform, &mut Fire), (Without<MudPatch>, Without<WaterBlob>)>,
    mut blobs: Query<(Entity, &mut WaterBlob, &mut Transform, &Velocity), Without<MudPatch>>,
) {
    let dt = time.delta_seconds().min(0.05);
    let player_entity = player.get_single().ok();
    for (entity, mut blob, mut tf, vel) in &mut blobs {
        let held = grab.entity == Some(entity);
        let speed = vel.linvel.length();
        let radius = MUD_BLOB_RADIUS * tf.scale.x;

        // dripping: faster when swung around
        blob.drip_timer += dt * (3.5 + speed * 1.2);
        while blob.drip_timer >= 1.0 {
            blob.drip_timer -= 1.0;
            blob.seed = blob.seed.wrapping_add(1);
            let jx = pseudo_rand(blob.seed) * 2.0 - 1.0;
            let jz = pseudo_rand(blob.seed.wrapping_add(5)) * 2.0 - 1.0;
            let size = 0.02 + pseudo_rand(blob.seed.wrapping_add(9)) * 0.014;
            commands.spawn((
                PbrBundle {
                    mesh: water.drop.clone(),
                    material: water.material.clone(),
                    transform: Transform::from_translation(
                        tf.translation + Vec3::new(jx * 0.6 * radius, -0.8 * radius, jz * 0.6 * radius),
                    )
                    .with_scale(Vec3::splat(size)),
                    ..default()
                },
                MudDrip {
                    velocity: vel.linvel * 0.35 + Vec3::new(jx * 0.2, -0.3, jz * 0.2),
                    size,
                    water: true,
                },
            ));
            blob.amount -= 0.012;
        }
        if blob.amount < 0.06 {
            if held {
                grab.entity = None;
            }
            commands.entity(entity).despawn_recursive();
            continue;
        }
        tf.scale = Vec3::splat(blob.amount.cbrt());

        if held {
            blob.free_time = 0.0;
            continue;
        }
        blob.free_time += dt;
        if tf.translation.y < -6.0 {
            commands.entity(entity).despawn_recursive();
            continue;
        }

        // let go above an upright bucket: it pours in
        for (bucket_tf, mut bucket) in &mut buckets {
            let up = bucket_tf.rotation * Vec3::Y;
            if up.y < 0.35 || bucket.fill >= 0.999 {
                continue;
            }
            let local = bucket_tf.rotation.inverse() * (tf.translation - bucket_tf.translation);
            let lateral = Vec2::new(local.x, local.z).length();
            if local.y > BUCKET_BASE_T && local.y < BUCKET_HEIGHT + 0.3 && lateral < BUCKET_TOP_R * 0.8 {
                let room = (1.0 - bucket.fill) / BUCKET_WATER_PER_BLOB;
                let taken = blob.amount.min(room);
                bucket.fill += taken * BUCKET_WATER_PER_BLOB;
                blob.amount -= taken;
                break;
            }
        }
        if blob.amount < 0.06 {
            commands.entity(entity).despawn_recursive();
            continue;
        }
        tf.scale = Vec3::splat(blob.amount.cbrt());

        // back into the river
        let over_river = (tf.translation.x - river_center_x(tf.translation.z)).abs()
            < RIVER_WIDTH * 0.5 + river.overflow;
        if over_river && tf.translation.y - radius * 0.4 < river.water_level {
            blob.seed = blob.seed.wrapping_add(1);
            let at = Vec3::new(tf.translation.x, river.water_level, tf.translation.z);
            spawn_water_burst(&mut commands, &water, at, Vec3::Y, 6, 0.7, blob.seed);
            commands.entity(entity).despawn_recursive();
            continue;
        }

        // hits something: down (lying on the ground) and along its motion
        let filter = world_ray_filter(player_entity).exclude_rigid_body(entity);
        let mut probes = vec![(Vec3::NEG_Y, radius + 0.06)];
        if speed > 0.5 {
            probes.push((vel.linvel / speed, radius + speed * dt * 2.0 + 0.05));
        }
        let mut impact: Option<(Vec3, Vec3)> = None;
        for (dir, reach) in probes {
            let Some((hit, toi)) = rapier.cast_ray(tf.translation, dir, reach, true, filter) else {
                continue;
            };
            if other_blobs.get(hit).is_ok() {
                continue;
            }
            let point = tf.translation + dir * toi;
            let normal = if (hit == entity) || dir.y < -0.9 { Vec3::Y } else { -dir };
            impact = Some((point, normal));
            break;
        }
        if impact.is_none() && blob.free_time > 8.0 {
            impact = Some((tf.translation, Vec3::Y));
        }
        if let Some((point, normal)) = impact {
            blob.seed = blob.seed.wrapping_add(1);
            let count = 6 + (8.0 * blob.amount) as u32;
            spawn_water_burst(
                &mut commands,
                &water,
                point,
                normal,
                count,
                0.6 + speed.min(10.0) * 0.08,
                blob.seed,
            );
            // water on a fire puts it out a bit
            for (fire_tf, mut fire) in &mut fires {
                if fire_tf.translation.distance(point) < 0.6 + 0.25 * fire.size {
                    fire.fuel = (fire.fuel - 1.6 * blob.amount).max(0.0);
                }
            }
            // the splash washes nearby mud off
            let reach = 0.2 + 0.25 * blob.amount;
            for (patch_entity, patch_gt, mut patch_tf, mut patch) in &mut patches {
                let d = patch_gt.translation().distance(point);
                if d < reach + patch.radius * 0.5 {
                    patch.wash += 0.8 * blob.amount;
                    if patch.wash >= 1.0 {
                        state.patches.retain(|e| *e != patch_entity);
                        commands.entity(patch_entity).despawn_recursive();
                    } else {
                        patch_tf.scale = patch.base_scale * (1.0 - patch.wash).max(0.05);
                    }
                }
            }
            commands.entity(entity).despawn_recursive();
        }
    }
}

/// Click the bucket's bail to pick the bucket up by it: the bucket then hangs from the
/// ears and the handle stays taut toward your hand. Runs just before grab_system.
fn bail_grab_system(
    mut grab: ResMut<Grab>,
    mut consumed: ResMut<ClickConsumed>,
    mut status: ResMut<Status>,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    rapier: Res<RapierContext>,
    parents: Query<&Parent>,
    pivots: Query<&Parent, With<BailPivot>>,
    mut buckets: Query<(&GlobalTransform, Option<&mut Damping>), With<Bucket>>,
) {
    if !buttons.just_pressed(MouseButton::Left) || consumed.0 || grab.entity.is_some() {
        return;
    }
    let (Ok(window), Ok((camera, cam_tf))) = (windows.get_single(), cameras.get_single()) else {
        return;
    };
    let Some(ray) = pointer_position(&window)
        .and_then(|cursor| camera.viewport_to_world(cam_tf, cursor))
    else {
        return;
    };
    let dir = *ray.direction;
    // same rule as the normal hand: the nearest physics collider under the cursor
    let Some((hit, toi)) = rapier.cast_ray(ray.origin, dir, 100.0, true, QueryFilter::only_dynamic())
    else {
        return;
    };
    let Ok(pivot) = parents.get(hit) else {
        return;
    };
    let Ok(bucket) = pivots.get(pivot.get()).map(|p| p.get()) else {
        return;
    };
    let Ok((gt, damping)) = buckets.get_mut(bucket) else {
        return;
    };
    let ears_local = Vec3::new(0.0, BUCKET_HEIGHT, 0.0);
    let ears_world = gt.transform_point(ears_local);
    let hit_point = ray.origin + dir * toi;

    grab.entity = Some(bucket);
    grab.local_anchor = ears_local;
    grab.distance = toi;
    grab.bail_reach = (hit_point - ears_world).length().clamp(0.05, 0.25);
    consumed.0 = true;
    if let Some(mut damping) = damping {
        damping.linear_damping = HELD_LINEAR_DAMPING;
        damping.angular_damping = HELD_ANGULAR_DAMPING;
    }
    status.0 = "Holding the bucket by its handle.".into();
}

/// The bail swings like a pendulum about the ear-to-ear axis: gravity (and the bucket's own
/// acceleration) pull it toward the lower side, so it sways when you carry the bucket, flops
/// over when you tip it and hangs down when the bucket is upside down.
fn bail_swing_system(
    time: Res<Time>,
    grab: Res<Grab>,
    buckets: Query<(&Transform, &Velocity, Option<&ExternalForce>), (With<Bucket>, Without<BailPivot>)>,
    mut pivots: Query<(&Parent, &mut Transform, &mut BailPivot)>,
) {
    const SWING_LENGTH: f32 = 0.116; // effective pendulum length of a half-hoop (pi/4 * radius)
    const DAMPING: f32 = 2.5; // 1/s
    const MAX_ANGLE: f32 = 1.2; // rad: it comes to rest leaning on the rim
    let dt = time.delta_seconds().clamp(1e-4, 0.033);
    for (parent, mut transform, mut bail) in &mut pivots {
        let Ok((bucket_tf, velocity, force)) = buckets.get(parent.get()) else {
            continue;
        };
        if grab.entity == Some(parent.get()) && grab.bail_reach > 0.0 {
            // held by the bail: the handle is pulled taut along the pull of the hand
            bail.last_linvel = velocity.linvel;
            if let Some(force) = force.filter(|f| f.force.length() > 0.1) {
                let pull = bucket_tf.rotation.inverse() * force.force.normalize();
                let target = pull.z.atan2(pull.y).clamp(-1.45, 1.45);
                bail.vel = (target - bail.angle) * 18.0;
                bail.angle += bail.vel * dt;
                transform.rotation = Quat::from_rotation_x(bail.angle);
            }
            continue;
        }
        let accel = ((velocity.linvel - bail.last_linvel) / dt).clamp_length_max(60.0);
        bail.last_linvel = velocity.linvel;
        // effective gravity as felt by the bucket, in the bucket's own axes
        let g = bucket_tf.rotation.inverse() * (Vec3::new(0.0, -WORLD_GRAVITY, 0.0) - accel);

        let angle = bail.angle;
        bail.vel += (angle.cos() * g.z - angle.sin() * g.y) / SWING_LENGTH * dt;
        bail.vel *= (1.0 - DAMPING * dt).max(0.0);
        bail.angle += bail.vel * dt;
        if bail.angle.abs() > MAX_ANGLE {
            bail.angle = bail.angle.clamp(-MAX_ANGLE, MAX_ANGLE);
            bail.vel *= -0.25; // soft stop
        }
        transform.rotation = Quat::from_rotation_x(bail.angle);
    }
}

// ---------------------------------------------------------------------------
// The "hand": hover / grab / drag / throw with physics
// ---------------------------------------------------------------------------

/// Gives every physics body the "hand" can grab what it needs: a mass read-out, an
/// external force slot (the hand pulls with it) and damping that is only raised
/// while the body is held.
fn ensure_grab_components(
    mut commands: Commands,
    new_bodies: Query<Entity, (Added<Velocity>, With<RigidBody>, Without<ExternalForce>)>,
) {
    for entity in &new_bodies {
        commands.entity(entity).insert((
            ReadMassProperties::default(),
            ExternalForce::default(),
            Damping {
                linear_damping: 0.0,
                angular_damping: 0.0,
            },
        ));
    }
}

fn grab_system(
    mut grab: ResMut<Grab>,
    door_grab: Res<DoorGrab>,
    uis: ToolUis,
    consumed: Res<ClickConsumed>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut wheel: EventReader<MouseWheel>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    rapier: Res<RapierContext>,
    mut bodies: Query<(
        &GlobalTransform,
        &mut Velocity,
        Option<&ReadMassProperties>,
        Option<&mut ExternalForce>,
        Option<&mut Damping>,
        Option<&GravityScale>,
    )>,
    mut pebbles: Query<&mut Pebble>,
    apples: Query<(), With<Apple>>,
    parents: Query<&Parent>,
) {

    let Ok(mut window) = windows.get_single_mut() else {
        return;
    };
    let Ok((camera, cam_tf)) = cameras.get_single() else {
        return;
    };

    let scroll: f32 = wheel.read().map(|e| e.y).sum();
    let ray = pointer_position(&window)
        .and_then(|cursor| camera.viewport_to_world(cam_tf, cursor));

    // What's under the cursor right now?
    // (A model's collider lives on a child entity, so walk up to the body that has a Velocity.)
    let hit = ray.and_then(|r| {
        let (mut e, toi) = rapier.cast_ray(
            r.origin,
            *r.direction,
            100.0,
            true,
            QueryFilter::only_dynamic(),
        )?;
        loop {
            if bodies.get(e).is_ok() {
                return Some((e, toi, r));
            }
            e = parents.get(e).ok()?.get();
        }
    });

    // Middle mouse button: throw an apple. It is the apple in your hand if you are holding
    // one, otherwise the apple you are pointing at (within reach). It flies where you aim.
    if buttons.just_pressed(MouseButton::Middle) {
        let held_apple = grab.entity.filter(|e| apples.get(*e).is_ok());
        let pointed_apple = hit.and_then(|(e, toi, _)| {
            (apples.get(e).is_ok() && toi < APPLE_REACH).then_some(e)
        });
        if let (Some(entity), Some(r)) = (held_apple.or(pointed_apple), ray) {
            if let Ok((_, mut vel, _, force, damping, _)) = bodies.get_mut(entity) {
                let dir = (*r.direction + Vec3::Y * 0.06).normalize();
                vel.linvel = dir * APPLE_THROW_SPEED;
                // a little tumbling spin around the axis across the throw
                vel.angvel = dir.cross(Vec3::Y).normalize_or_zero() * -10.0;
                if let Some(mut force) = force {
                    *force = ExternalForce::default();
                }
                if let Some(mut damping) = damping {
                    damping.linear_damping = 0.0;
                    damping.angular_damping = 0.0;
                }
            }
            if grab.entity == Some(entity) {
                grab.entity = None; // the hand lets go of it
                grab.bail_reach = 0.0;
            }
        }
    }

    // Start grabbing (unless that click was just spent talking to someone).
    // We remember the exact point of the body that was clicked, in the body's own space.
    if buttons.just_pressed(MouseButton::Left) && !consumed.0 {
        if let Some((entity, toi, r)) = hit {
            if let Ok((gt, ..)) = bodies.get(entity) {
                let hit_point = r.origin + *r.direction * toi;
                grab.entity = Some(entity);
                grab.local_anchor = gt.affine().inverse().transform_point3(hit_point);
                grab.distance = toi;
                grab.bail_reach = 0.0;
            }
            if let Ok((_, _, _, _, Some(mut damping), _)) = bodies.get_mut(entity) {
                damping.linear_damping = HELD_LINEAR_DAMPING;
                damping.angular_damping = HELD_ANGULAR_DAMPING;
            }
        }
    }

    // Let go (the body keeps its velocity -> you can throw things)
    if !buttons.pressed(MouseButton::Left) {
        if let Some(entity) = grab.entity {
            if let (Ok((_, velocity, ..)), Ok(mut pebble)) =
                (bodies.get(entity), pebbles.get_mut(entity))
            {
                pebble.thrown_velocity = if velocity.linvel.length() >= 13.0 {
                    velocity.linvel
                } else {
                    Vec3::ZERO
                };
            }
            if let Ok((_, _, _, force, damping, _)) = bodies.get_mut(entity) {
                if let Some(mut force) = force {
                    *force = ExternalForce::default();
                }
                if let Some(mut damping) = damping {
                    damping.linear_damping = 0.0;
                    damping.angular_damping = 0.0;
                }
            }
        }
        grab.entity = None;
        grab.bail_reach = 0.0;
    }

    // Drag. The hand is a spring-damper attached to the grabbed POINT only. It pulls
    // that point toward the cursor and also holds up the body's weight; the force is
    // applied at that point, so it also produces torque: the rest of the body swings,
    // tilts, hangs from the grip and trails behind exactly like a real held object.
    if let (Some(entity), Some(r)) = (grab.entity, ray) {
        grab.distance = (grab.distance + scroll * 0.3).clamp(0.5, 20.0);
        if let Ok((gt, mut vel, mass, force, _, gravity)) = bodies.get_mut(entity) {
            let anchor_world = gt.transform_point(grab.local_anchor);
            let mut target = r.origin + *r.direction * grab.distance;
            if grab.bail_reach > 0.0 {
                // held by the bail: the pull goes through the ears, the handle itself reaches
                // from the ears toward the cursor, so the ears stop short of it by that much
                let to_cursor = target - anchor_world;
                let d = to_cursor.length();
                target = if d > grab.bail_reach {
                    anchor_world + to_cursor * (1.0 - grab.bail_reach / d)
                } else {
                    anchor_world
                };
            }
            let props = mass.map(|m| *m.get()).filter(|p| p.mass > 1e-5);
            match (props, force) {
                (Some(props), Some(mut force)) => {
                    let (_, rotation, position) = gt.to_scale_rotation_translation();
                    let center_of_mass = position + rotation * props.local_center_of_mass;
                    let lever = anchor_world - center_of_mass;
                    let point_velocity = vel.linvel + vel.angvel.cross(lever);

                    let error = (target - anchor_world).clamp_length_max(HAND_MAX_REACH_ERROR);
                    let accel = error * HAND_STIFFNESS - point_velocity * HAND_DAMPING;
                    let weight = props.mass * WORLD_GRAVITY * gravity.map_or(1.0, |g| g.0);
                    let pull = (accel * props.mass).clamp_length_max(props.mass * HAND_MAX_ACCEL)
                        + Vec3::Y * weight;

                    force.force = pull;
                    force.torque = lever.cross(pull);
                }
                _ => {
                    // No mass information yet (still loading): old rigid steering.
                    let error = target - anchor_world;
                    vel.linvel = (error * 14.0).clamp_length_max(30.0);
                    vel.angvel *= 0.9;
                }
            }
        }
    }

    // Hand cursor: open hand over something grabbable, closed fist while holding
        window.cursor.icon = if grab.entity.is_some()
        || door_grab.active
        || uis.sword.swinging
        || uis.axe.swinging
        || uis.shovel.swinging
        || uis.spear.attacking
        || uis.machete.swinging
        || uis.knife.swinging
        || uis.brush.painting
    {
        CursorIcon::Grabbing
    } else if hit.is_some()
        || door_grab.hover
        || uis.sword.hover
        || uis.axe.hover
        || uis.shovel.hover
        || uis.spear.hover
        || uis.machete.hover
        || uis.knife.hover
        || uis.brush.hover
    {
        CursorIcon::Grab
    } else {
        CursorIcon::Default
    };
}