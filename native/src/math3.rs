//! 3-vectors and 3x3 matrices (row-major, `m[row][col]`), with the same conventions as vtrack's NumPy code:
//! a frame is built from column vectors, `unit` returns zeros below 1e-9, angles in radians.

pub type V3 = [f64; 3];
pub type M3 = [[f64; 3]; 3];

#[inline]
pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub fn mul(a: V3, b: V3) -> V3 {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

#[inline]
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

/// a / |a|, or zeros when |a| <= 1e-9 (geometry.unit).
#[inline]
pub fn unit(a: V3) -> V3 {
    let n = norm(a);
    if n > 1e-9 {
        [a[0] / n, a[1] / n, a[2] / n]
    } else {
        [0.0, 0.0, 0.0]
    }
}

#[inline]
pub fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// a minus its component along the unit vector u.
#[inline]
pub fn reject(a: V3, u: V3) -> V3 {
    sub(a, scale(u, dot(a, u)))
}

/// The matrix whose columns are x, y, z (np.column_stack).
#[inline]
pub fn cols(x: V3, y: V3, z: V3) -> M3 {
    [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]
}

#[inline]
pub fn col(m: &M3, c: usize) -> V3 {
    [m[0][c], m[1][c], m[2][c]]
}

#[inline]
pub fn t(m: &M3) -> M3 {
    [[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]]
}

#[inline]
pub fn mm(a: &M3, b: &M3) -> M3 {
    let mut o = [[0.0; 3]; 3];
    for (r, row) in o.iter_mut().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            *v = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    o
}

/// m^T v
#[inline]
pub fn mtv(a: &M3, v: V3) -> V3 {
    [
        a[0][0] * v[0] + a[1][0] * v[1] + a[2][0] * v[2],
        a[0][1] * v[0] + a[1][1] * v[1] + a[2][1] * v[2],
        a[0][2] * v[0] + a[1][2] * v[1] + a[2][2] * v[2],
    ]
}

pub fn rot_x(a: f64) -> M3 {
    let (s, c) = a.sin_cos();
    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}

pub fn rot_y(a: f64) -> M3 {
    let (s, c) = a.sin_cos();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

pub fn rot_z(a: f64) -> M3 {
    let (s, c) = a.sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

#[inline]
pub fn clip1(x: f64) -> f64 {
    x.clamp(-1.0, 1.0)
}

#[inline]
pub fn deg(x: f64) -> f64 {
    x.to_degrees()
}

/// (z, x, y) with R = Rz(z) Rx(x) Ry(y) (legs.euler_zxy).
pub fn euler_zxy(r: &M3) -> (f64, f64, f64) {
    let x = clip1(r[2][1]).asin();
    ((-r[0][1]).atan2(r[1][1]), x, (-r[2][0]).atan2(r[2][2]))
}

/// (a, b, c) with R = Ry(a) Rx(b) Rz(c) (kinematics.euler_yxz).
pub fn euler_yxz(r: &M3) -> (f64, f64, f64) {
    let b = clip1(-r[1][2]).asin();
    (r[0][2].atan2(r[2][2]), b, r[1][0].atan2(r[1][1]))
}

/// Python's float modulo (the result takes the divisor's sign).
#[inline]
pub fn pymod(a: f64, b: f64) -> f64 {
    let m = a % b;
    if m != 0.0 && ((b < 0.0) != (m < 0.0)) {
        m + b
    } else {
        m
    }
}

/// Row i of an (N, 3) slice.
#[inline]
pub fn row(p: &[f64], i: usize) -> V3 {
    [p[3 * i], p[3 * i + 1], p[3 * i + 2]]
}
