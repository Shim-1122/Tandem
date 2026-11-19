//! The arm, leg, core and finger solvers of vtrack (kinematics.py, legs.py, fingers.py), operation for operation.
//! Inputs are MediaPipe world landmarks as flat (N, 3) row-major slices.

use crate::math3::*;

// MediaPipe pose indices
const L_SHOULDER: usize = 11;
const R_SHOULDER: usize = 12;
const L_HIP: usize = 23;
const R_HIP: usize = 24;

fn side_mirror(left: bool) -> V3 {
    [1.0, 1.0, if left { -1.0 } else { 1.0 }]
}

/// kinematics.thorax_frame: (R_t, shoulder midpoint).
pub fn thorax_frame(pose: &[f64], up_hint: Option<V3>) -> (M3, V3) {
    let (ls, rs) = (row(pose, L_SHOULDER), row(pose, R_SHOULDER));
    let z = unit(sub(rs, ls));
    let mid_sh = scale(add(rs, ls), 0.5);
    let y0 = match up_hint {
        Some(u) => u,
        None => sub(mid_sh, scale(add(row(pose, R_HIP), row(pose, L_HIP)), 0.5)),
    };
    let y = unit(reject(y0, z));
    let x = cross(y, z);
    (cols(x, y, z), mid_sh)
}

/// legs.pelvis_frame: (R_p, hip midpoint, obliquity).
pub fn pelvis_frame(pose: &[f64], up: V3) -> (M3, V3, f64) {
    let (lh, rh) = (row(pose, L_HIP), row(pose, R_HIP));
    let z = unit(sub(rh, lh));
    let u = unit(up);
    let y = unit(reject(u, z));
    let obliquity = clip1(dot(z, u)).asin();
    (cols(cross(y, z), y, z), scale(add(rh, lh), 0.5), obliquity)
}

/// legs.core_angles in degrees: (flexion, lateral bend, rotation).
pub fn core_angles(pose: &[f64], up: V3) -> (f64, f64, f64) {
    let rp = pelvis_frame(pose, up).0;
    let rt = thorax_frame(pose, None).0;
    let (z, x, y) = euler_zxy(&mm(&t(&rp), &rt));
    (deg(-z), deg(x), deg(y))
}

// ----------------------------------------------------------------------------------------------------- arm
pub struct Arm {
    pub left: bool,
    pub min_elbow: f64,
    pub pole: f64,
    pub tau: f64,
    pub phi: f64,
}

pub struct ArmOut {
    pub angles: [f64; 7], // shoulder flexion, abduction, rotation, elbow, forearm rotation, wrist flexion, deviation
    pub rotation_valid: bool,
    pub wrist_valid: bool,
}

impl Arm {
    fn to_thorax(&self, v: V3, rt: &M3) -> V3 {
        mul(mtv(rt, v), side_mirror(self.left))
    }

    pub fn solve(&mut self, pose: &[f64], hand: Option<&[f64]>, up_hint: Option<V3>) -> ArmOut {
        let (rt, origin) = thorax_frame(pose, up_hint);
        let (s_i, e_i, w_i) = if self.left { (11, 13, 15) } else { (12, 14, 16) };
        let s = self.to_thorax(sub(row(pose, s_i), origin), &rt);
        let e = self.to_thorax(sub(row(pose, e_i), origin), &rt);
        let w = self.to_thorax(sub(row(pose, w_i), origin), &rt);
        let yh = unit(sub(s, e));
        let fd = unit(sub(w, e));
        let elbow = 180.0 - deg(clip1(dot(yh, fd)).acos());
        let mut out = ArmOut { angles: [f64::NAN; 7], rotation_valid: false, wrist_valid: false };
        out.angles[3] = elbow;
        let beta = clip1(yh[2]).asin();
        let near_pole = yh[2].abs() > self.pole.to_radians().cos();
        let mut zh = cross(fd, yh);
        let plane_ok = elbow >= self.min_elbow && norm(zh) > self.min_elbow.to_radians().sin();
        let mut rh: Option<M3> = None;
        if plane_ok {
            zh = unit(zh);
            rh = Some(cols(cross(yh, zh), yh, zh));
        }
        let phi;
        if !near_pole {
            phi = (-yh[0]).atan2(yh[1]);
            let swing = mm(&rot_z(phi), &rot_x(beta));
            match rh {
                Some(r) => {
                    let tm = mm(&t(&swing), &r);
                    self.tau = tm[0][2].atan2(tm[0][0]);
                    out.rotation_valid = true;
                }
                None => rh = Some(mm(&swing, &rot_y(self.tau))),
            }
            self.phi = phi;
        } else {
            if let Some(r) = rh {
                let total = r[0][2].atan2(r[0][0]);
                self.phi = if beta > 0.0 { total - self.tau } else { self.tau - total };
            }
            phi = self.phi;
            if rh.is_none() {
                rh = Some(mm(&mm(&rot_z(phi), &rot_x(beta)), &rot_y(self.tau)));
            }
        }
        out.angles[0] = deg(phi);
        out.angles[1] = deg(-beta);
        out.angles[2] = deg(self.tau);
        let rf = mm(&rh.unwrap(), &rot_z(elbow.to_radians()));
        let (hy, hx) = match hand {
            Some(h) => (sub(row(h, 0), row(h, 9)), sub(row(h, 5), row(h, 17))),
            None => {
                let (pi, ii) = if self.left { (17, 19) } else { (18, 20) };
                let (p, ix, pk) = (row(pose, w_i), row(pose, ii), row(pose, pi));
                (sub(p, scale(add(ix, pk), 0.5)), sub(ix, pk))
            }
        };
        let (hy, hx) = (self.to_thorax(hy, &rt), self.to_thorax(hx, &rt));
        let y_hand = unit(hy);
        let x_hand = reject(hx, y_hand);
        if norm(y_hand) > 0.0 && norm(x_hand) > 1e-9 {
            let x_hand = unit(x_hand);
            let r_hand = cols(x_hand, y_hand, cross(x_hand, y_hand));
            let (ps, fe, dev) = euler_yxz(&mm(&t(&rf), &r_hand));
            out.angles[4] = deg(ps);
            out.angles[5] = deg(fe);
            out.angles[6] = deg(dev);
            out.wrist_valid = true;
        }
        out
    }
}

// ----------------------------------------------------------------------------------------------------- leg
pub struct Leg {
    pub left: bool,
    pub min_knee: f64,
    pub pole: f64,
    pub tau: f64,
    pub phi: f64,
}

pub struct LegOut {
    // hip flexion, abduction, rotation, pelvic hike, knee, ankle dorsiflexion, inversion, abduction, foot pitch, roll
    pub angles: [f64; 10],
    pub source: u8, // 0 held, 1 knee, 2 foot
    pub ankle_valid: bool,
}

fn foot_frame(a: V3, hl: V3, toe: V3) -> Option<M3> {
    let x = unit(sub(toe, hl));
    let y = reject(sub(a, hl), x);
    if norm(x) < 0.5 || norm(y) < 1e-3 {
        return None;
    }
    let y = unit(y);
    Some(cols(x, y, cross(x, y)))
}

impl Leg {
    pub fn solve(&mut self, pose: &[f64], up: V3, foot: bool, correction: &M3) -> LegOut {
        let (rp, origin, _) = pelvis_frame(pose, up);
        let mirror = side_mirror(self.left);
        let idx: [usize; 5] = if self.left { [23, 25, 27, 29, 31] } else { [24, 26, 28, 30, 32] };
        let local = |i: usize| mul(mtv(&rp, sub(row(pose, i), origin)), mirror);
        let (h, k, a, hl, toe) = (local(idx[0]), local(idx[1]), local(idx[2]), local(idx[3]), local(idx[4]));
        let u = mul(mtv(&rp, unit(up)), mirror);
        let mut out = LegOut { angles: [f64::NAN; 10], source: 0, ankle_valid: false };
        out.angles[3] = deg(clip1(u[2]).asin());
        let yt = unit(sub(h, k));
        let sd = unit(sub(a, k));
        let knee = 180.0 - deg(clip1(dot(yt, sd)).acos());
        out.angles[4] = knee;
        let r_foot = if foot { foot_frame(a, hl, toe).map(|r| mm(&r, correction)) } else { None };
        let beta = clip1(yt[2]).asin();
        let near_pole = yt[2].abs() > self.pole.to_radians().cos();
        let zt = cross(yt, sd);
        let plane_ok = knee >= self.min_knee && norm(zt) > self.min_knee.to_radians().sin();
        let r_meas = if plane_ok {
            let zt = unit(zt);
            Some(cols(cross(yt, zt), yt, zt))
        } else {
            None
        };
        if !near_pole {
            let phi = (-yt[0]).atan2(yt[1]);
            let swing = mm(&rot_z(phi), &rot_x(beta));
            if let Some(r) = r_meas {
                let m = mm(&t(&swing), &r);
                self.tau = m[0][2].atan2(m[0][0]);
                out.source = 1;
            } else if let Some(rf) = r_foot {
                let v = mtv(&swing, col(&rf, 0));
                if v[0].hypot(v[2]) > 0.2 {
                    self.tau = (-v[2]).atan2(v[0]);
                    out.source = 2;
                }
            }
            self.phi = phi;
        } else if let Some(r) = r_meas {
            let total = r[0][2].atan2(r[0][0]);
            self.phi = if beta > 0.0 { total - self.tau } else { self.tau - total };
        }
        let (phi, tau) = (self.phi, self.tau);
        out.angles[0] = deg(phi);
        out.angles[1] = deg(-beta);
        out.angles[2] = deg(tau);
        if let Some(rf) = r_foot {
            let r_thigh = mm(&mm(&rot_z(phi), &rot_x(beta)), &rot_y(tau));
            let r_shank = mm(&r_thigh, &rot_z((-knee).to_radians()));
            let (d, i, ab) = euler_zxy(&mm(&t(&r_shank), &rf));
            out.angles[5] = deg(d);
            out.angles[6] = deg(i);
            out.angles[7] = deg(-ab);
            let (xf, yf, zf) = (col(&rf, 0), col(&rf, 1), col(&rf, 2));
            out.angles[8] = deg(clip1(dot(xf, u)).asin());
            out.angles[9] = deg((-dot(zf, u)).atan2(dot(yf, u)));
            out.ankle_valid = true;
        }
        out
    }
}

// ----------------------------------------------------------------------------------------------------- hand
pub struct Hand {
    pub left: bool,
    pub tau: f64,
}

const FINGERS4: [[usize; 4]; 4] = [[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];
const THUMB_BASE: M3 = [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]; // columns (0,-1,0) (0,0,-1) (1,0,0)

fn hinge(r: &M3, d: V3) -> f64 {
    let q = mtv(r, d);
    q[1].atan2(q[0])
}

fn finger_base(m: V3) -> M3 {
    let u = unit(m);
    let v = unit(reject([0.0, 0.0, -1.0], u));
    cols(u, v, cross(u, v))
}

impl Hand {
    /// 21 angles in degrees: index, middle, ring, pinky (mcp_flex, mcp_abd, pip, dip each), then the thumb
    /// (cmc_palmar, cmc_radial, cmc_rotation, mcp, ip).
    pub fn solve(&mut self, world: &[f64]) -> [f64; 21] {
        let flip = if self.left { [-1.0, 1.0, 1.0] } else { [1.0, 1.0, 1.0] };
        let w = |i: usize| mul(row(world, i), flip);
        let w0 = w(0);
        let y = unit(sub(w0, w(9)));
        let x = unit(reject(sub(w(5), w(17)), y));
        let hm = cols(x, y, cross(x, y));
        let p = |i: usize| mtv(&hm, sub(w(i), w0));
        let mut out = [0.0; 21];
        for (f, &[m, pp, d, tt]) in FINGERS4.iter().enumerate() {
            let (pm, pp_, pd, pt) = (p(m), p(pp), p(d), p(tt));
            let b = finger_base(pm);
            let q = mtv(&b, unit(sub(pp_, pm)));
            let flex = q[1].atan2(q[0]);
            let abd = clip1(q[2]).asin();
            let r1 = mm(&mm(&b, &rot_z(flex)), &rot_y(-abd));
            let pip = hinge(&r1, unit(sub(pd, pp_)));
            let r2 = mm(&r1, &rot_z(pip));
            let dip = hinge(&r2, unit(sub(pt, pd)));
            out[4 * f] = deg(flex);
            out[4 * f + 1] = deg(abd);
            out[4 * f + 2] = deg(pip);
            out[4 * f + 3] = deg(dip);
        }
        let (pc, pm, pi, pt) = (p(1), p(2), p(3), p(4));
        let q = mtv(&THUMB_BASE, unit(sub(pm, pc)));
        let palmar = q[1].atan2(q[0]);
        let radial = clip1(q[2]).asin();
        let r1p = mm(&mm(&THUMB_BASE, &rot_z(palmar)), &rot_y(-radial));
        let bend = mtv(&r1p, unit(sub(pt, pm)));
        if bend[1].hypot(bend[2]) > 8f64.to_radians().sin() {
            self.tau = bend[2].atan2(bend[1]);
        }
        let r1 = mm(&r1p, &rot_x(self.tau));
        let mcp = hinge(&r1, unit(sub(pi, pm)));
        let r2 = mm(&r1, &rot_z(mcp));
        let ip = hinge(&r2, unit(sub(pt, pi)));
        out[16] = deg(palmar);
        out[17] = deg(radial);
        out[18] = pymod(deg(self.tau) + 90.0 + 180.0, 360.0) - 180.0;
        out[19] = deg(mcp);
        out[20] = deg(ip);
        out
    }
}
