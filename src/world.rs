//! The shape of the land: terrain height, the river, and a cheap hash-based random number.
//! Pure functions, so they can be tested and benchmarked without starting the game.

use bevy::math::Vec2;

pub const GROUND_SURFACE_Y: f32 = -0.02;
pub const SHORELINE_X: f32 = -16.0;
pub const RIVER_WIDTH: f32 = 7.0;

pub fn river_center_x(z: f32) -> f32 {
    SHORELINE_X - RIVER_WIDTH * 0.5 + 2.0 * (z * 0.105).sin() + 0.75 * (z * 0.24).sin()
}

pub fn terrain_elevation(x: f32, z: f32) -> f32 {
    let point = Vec2::new(x, z);
    let hills = [
        (Vec2::new(12.0, -23.0), 2.6, 11.0),
        (Vec2::new(31.0, 5.0), 3.4, 13.0),
        (Vec2::new(-31.0, 17.0), 2.8, 12.0),
        (Vec2::new(4.0, 31.0), 2.1, 10.0),
    ];
    let rolling = hills
        .iter()
        .map(|(center, height, radius)| {
            let distance = point.distance(*center) / radius;
            height * (-0.5 * distance * distance).exp()
        })
        .sum::<f32>();

    let house_t = ((point.length() - 7.0) / 6.0).clamp(0.0, 1.0);
    let house_mask = house_t * house_t * (3.0 - 2.0 * house_t);
    let river_distance = (x - river_center_x(z)).abs();
    let river_t = ((river_distance - (RIVER_WIDTH * 0.5 + 2.0)) / 6.0).clamp(0.0, 1.0);
    let river_mask = river_t * river_t * (3.0 - 2.0 * river_t);
    rolling * house_mask * river_mask
}

pub fn land_surface_height(x: f32, z: f32) -> f32 {
    GROUND_SURFACE_Y + terrain_elevation(x, z)
}

/// Cheap, dependency-free pseudo-random number in [0, 1) from an integer seed.
pub fn pseudo_rand(seed: u32) -> f32 {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    x ^= x >> 15;
    x = x.wrapping_mul(2_246_822_519);
    x ^= x >> 13;
    (x as f32) / (u32::MAX as f32)
}
