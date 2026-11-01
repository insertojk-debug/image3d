//! Where Tom walks: the house is a block with one door (in the +X wall), so he goes through
//! the door to get in or out and round the corners of the house otherwise.
//!
//! These are pure functions (no game state), which makes them easy to test and to benchmark.

use bevy::math::{Vec2, Vec3};

pub const HOUSE_IN: f32 = 4.0; // inside the walls (the walls are at +-4.2)
pub const HOUSE_BOX: f32 = 4.4; // keep out of this square when walking round the outside
pub const HOUSE_CORNER: f32 = 5.0; // the corners he walks round

pub fn hunter_inside(p: Vec3) -> bool {
    p.x.abs() < HOUSE_IN && p.z.abs() < HOUSE_IN
}

/// The doorway and the strip of floor and ground on either side of it (the door is in the +X
/// wall, 1.2 m wide, centred on z = 0).
pub fn hunter_in_lane(p: Vec3) -> bool {
    p.z.abs() < 0.6 && p.x > 3.2 && p.x < 5.6
}

/// Does the straight line a-b cross the square |x| < half, |z| < half?
pub fn hunter_seg_hits_box(a: Vec2, b: Vec2, half: f32) -> bool {
    let d = b - a;
    let mut t0 = 0.0_f32;
    let mut t1 = 1.0_f32;
    let checks = [
        (-d.x, a.x + half),
        (d.x, half - a.x),
        (-d.y, a.y + half),
        (d.y, half - a.y),
    ];
    for (p, q) in checks {
        if p.abs() < 1e-6 {
            if q < 0.0 {
                return false;
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                if r > t1 {
                    return false;
                }
                if r > t0 {
                    t0 = r;
                }
            } else {
                if r < t0 {
                    return false;
                }
                if r < t1 {
                    t1 = r;
                }
            }
        }
    }
    true
}

/// Walking round the OUTSIDE of the house: straight to `dest` if that does not cut through
/// the house, otherwise round its corners along the shortest way. The bool is true when the
/// returned point is `dest` itself.
pub fn hunter_outside_step(pos: Vec3, dest: Vec3) -> (Vec3, bool) {
    let a = Vec2::new(pos.x, pos.z);
    let b = Vec2::new(dest.x, dest.z);
    if a.x.abs() < HOUSE_BOX && a.y.abs() < HOUSE_BOX {
        // right up against the walls: step straight out first
        let cands = [
            (HOUSE_BOX - a.x, Vec2::new(HOUSE_BOX + 0.3, a.y)),
            (a.x + HOUSE_BOX, Vec2::new(-HOUSE_BOX - 0.3, a.y)),
            (HOUSE_BOX - a.y, Vec2::new(a.x, HOUSE_BOX + 0.3)),
            (a.y + HOUSE_BOX, Vec2::new(a.x, -HOUSE_BOX - 0.3)),
        ];
        let mut best = cands[0];
        for c in cands {
            if c.0 < best.0 {
                best = c;
            }
        }
        return (Vec3::new(best.1.x, 0.0, best.1.y), false);
    }
    if !hunter_seg_hits_box(a, b, HOUSE_BOX) {
        return (dest, true);
    }
    // shortest way over: me -> (some corners) -> dest
    let h = HOUSE_CORNER;
    let nodes = [
        a,
        Vec2::new(h, h),
        Vec2::new(h, -h),
        Vec2::new(-h, -h),
        Vec2::new(-h, h),
        b,
    ];
    let inf = 1.0e9_f32;
    let mut dist = [inf; 6];
    let mut prev = [usize::MAX; 6];
    dist[0] = 0.0;
    let mut edges: Vec<(usize, usize, f32)> = Vec::new();
    for i in 0..6 {
        for j in (i + 1)..6 {
            if !hunter_seg_hits_box(nodes[i], nodes[j], HOUSE_BOX) {
                let w = nodes[i].distance(nodes[j]);
                edges.push((i, j, w));
                edges.push((j, i, w));
            }
        }
    }
    for _ in 0..6 {
        for &(i, j, w) in &edges {
            if dist[i] + w < dist[j] - 1e-6 {
                dist[j] = dist[i] + w;
                prev[j] = i;
            }
        }
    }
    if dist[5] >= inf {
        return (dest, true);
    }
    let mut path: Vec<usize> = Vec::new();
    let mut k = 5_usize;
    loop {
        path.push(k);
        if prev[k] == usize::MAX {
            break;
        }
        k = prev[k];
    }
    path.reverse();
    for &k in path.iter().skip(1) {
        if k == 5 {
            return (dest, true);
        }
        if nodes[k].distance(a) > 0.05 {
            return (Vec3::new(nodes[k].x, 0.0, nodes[k].y), false);
        }
    }
    (dest, true)
}

/// The next point to walk to on the way to `dest`: through the front door (in the +X wall)
/// when going in or out of the house, and round the outside of the house otherwise.
/// The bool is true when that point is `dest` itself.
pub fn hunter_step_target(pos: Vec3, dest: Vec3) -> (Vec3, bool) {
    let door_in = Vec3::new(3.3, 0.0, 0.0);
    let door_out = Vec3::new(5.4, 0.0, 0.0);
    let here = hunter_inside(pos);
    let there = hunter_inside(dest);
    if here && there {
        return (dest, true);
    }
    if here {
        // going out: line up with the door, go through it, then on to dest
        if !hunter_in_lane(pos) {
            return (door_in, false);
        }
        if pos.x < 5.3 {
            return (door_out, false);
        }
        return hunter_outside_step(pos, dest);
    }
    if there {
        // going in: walk round to the front of the door, then in
        if hunter_in_lane(pos) {
            return (door_in, false);
        }
        let (w, _) = hunter_outside_step(pos, door_out);
        return (w, false);
    }
    if hunter_in_lane(pos) && pos.x < 5.3 && !hunter_in_lane(dest) {
        return (door_out, false);
    }
    hunter_outside_step(pos, dest)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Walks from `start` to `dest` the way the game does and returns the path.
    fn walk(start: Vec3, dest: Vec3) -> Option<Vec<Vec3>> {
        let mut pos = start;
        let mut path = vec![pos];
        for _ in 0..20_000 {
            let (target, is_final) = hunter_step_target(pos, dest);
            let mut to = target - pos;
            to.y = 0.0;
            let dist = to.length();
            if dist > 0.02 {
                pos += to / dist * (1.7_f32 / 60.0).min(dist);
            }
            path.push(pos);
            let mut left = dest - pos;
            left.y = 0.0;
            if is_final && left.length() < 0.3 {
                return Some(path);
            }
        }
        None
    }

    /// True if the point is inside a wall (between inside and outside) other than in the doorway.
    fn in_wall(p: Vec3) -> bool {
        let in_ring = p.x.abs() < 4.3 && p.z.abs() < 4.3 && !hunter_inside(p);
        in_ring && !(p.x > 3.9 && p.z.abs() < 0.65)
    }

    #[test]
    fn walks_from_the_table_to_the_river_without_crossing_a_wall() {
        let path = walk(Vec3::new(-2.6, 0.0, -1.35), Vec3::new(-19.5, 0.0, -5.0)).expect("arrives");
        assert!(path.iter().all(|p| !in_wall(*p)));
    }

    #[test]
    fn walks_from_the_river_back_to_the_house() {
        let path = walk(Vec3::new(-19.5, 0.0, 10.0), Vec3::new(0.0, 0.0, 1.0)).expect("arrives");
        assert!(path.iter().all(|p| !in_wall(*p)));
        assert!(hunter_inside(*path.last().unwrap()));
    }

    #[test]
    fn goes_straight_when_nothing_is_in_the_way() {
        let (target, is_final) = hunter_step_target(Vec3::new(8.0, 0.0, 6.0), Vec3::new(12.0, 0.0, -3.0));
        assert!(is_final);
        assert_eq!(target, Vec3::new(12.0, 0.0, -3.0));
    }
}
