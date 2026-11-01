//! Criterion benchmarks for the engine-free parts of the game (see `src/lib.rs`).
//!
//! Run them with `cargo bench`; open `target/criterion/report/index.html` for the graphs.
//! Run one group with e.g. `cargo bench -- terrain`, and compare against an earlier run with
//! `cargo bench -- --save-baseline before` ... change the code ... `cargo bench -- --baseline before`.

use bevy::math::{Vec2, Vec3};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use image3d_house::nav::{hunter_seg_hits_box, hunter_step_target};
use image3d_house::world::{land_surface_height, pseudo_rand, terrain_elevation};
use std::hint::black_box;

fn bench_pseudo_rand(c: &mut Criterion) {
    c.bench_function("pseudo_rand x1000", |b| {
        b.iter(|| {
            let mut sum = 0.0_f32;
            for seed in 0..1000_u32 {
                sum += pseudo_rand(black_box(seed));
            }
            sum
        })
    });
}

fn bench_terrain(c: &mut Criterion) {
    let mut group = c.benchmark_group("terrain");
    group.bench_function("terrain_elevation (one point)", |b| {
        b.iter(|| terrain_elevation(black_box(12.3), black_box(-7.9)))
    });
    // roughly what building the ground mesh costs: one height per grid point
    group.bench_function("land_surface_height 160x160 grid", |b| {
        b.iter(|| {
            let mut sum = 0.0_f32;
            for ix in 0..160 {
                for iz in 0..160 {
                    sum += land_surface_height(ix as f32 * 0.5 - 40.0, iz as f32 * 0.5 - 40.0);
                }
            }
            black_box(sum)
        })
    });
    group.finish();
}

/// Walks from `start` to `dest` the way Tom does (one step per frame at 60 fps) and returns how
/// many steps it took.
fn walk(start: Vec3, dest: Vec3) -> usize {
    let mut pos = start;
    let step_len = 1.7_f32 / 60.0;
    for i in 0..20_000 {
        let (target, is_final) = hunter_step_target(pos, dest);
        let mut to = target - pos;
        to.y = 0.0;
        let dist = to.length();
        if dist > 0.02 {
            pos += to / dist * step_len.min(dist);
        }
        let mut left = dest - pos;
        left.y = 0.0;
        if is_final && left.length() < 0.3 {
            return i;
        }
    }
    20_000
}

fn bench_navigation(c: &mut Criterion) {
    let table = Vec3::new(-2.6, 0.0, -1.35);
    let sofa = Vec3::new(0.0, 0.0, 1.0);
    let river = Vec3::new(-19.5, 0.0, -5.0);
    let cases = [
        ("inside_to_inside", table, sofa),
        ("inside_to_river", table, river),
        ("river_to_inside", river, sofa),
        ("outside_clear", Vec3::new(8.0, 0.0, 6.0), Vec3::new(12.0, 0.0, -3.0)),
        ("outside_round_the_house", Vec3::new(8.0, 0.0, 0.0), Vec3::new(-19.0, 0.0, 0.0)),
    ];

    // one decision: "where do I step next?"
    let mut step = c.benchmark_group("hunter_step_target");
    for (name, from, to) in cases {
        step.bench_with_input(BenchmarkId::from_parameter(name), &(from, to), |b, &(from, to)| {
            b.iter(|| hunter_step_target(black_box(from), black_box(to)))
        });
    }
    step.finish();

    c.bench_function("hunter_seg_hits_box", |b| {
        b.iter(|| {
            hunter_seg_hits_box(
                black_box(Vec2::new(8.0, 0.0)),
                black_box(Vec2::new(-19.0, 0.0)),
                black_box(4.4),
            )
        })
    });

    // a whole trip, all the way there
    let mut trip = c.benchmark_group("whole_walk");
    trip.sample_size(30);
    trip.bench_function("table_to_river", |b| {
        b.iter(|| walk(black_box(table), black_box(river)))
    });
    trip.bench_function("river_to_sofa", |b| {
        b.iter(|| walk(black_box(river), black_box(sofa)))
    });
    trip.finish();
}

criterion_group!(benches, bench_pseudo_rand, bench_terrain, bench_navigation);
criterion_main!(benches);
