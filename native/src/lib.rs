//! tandem_core: Tandem's per-frame math in Rust, called from vtrack when it is installed.
//!
//! The Python code in vtrack stays the reference: every class here mirrors one there operation for operation,
//! and tests/test_native.py checks they agree. Python keeps working without this module.

use numpy::{PyArrayDyn, PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

mod math3;
mod solvers;

use math3::{pymod, M3, V3};

fn as_v3(a: Option<PyReadonlyArrayDyn<f64>>) -> PyResult<Option<V3>> {
    match a {
        None => Ok(None),
        Some(a) => {
            let s = a.as_slice()?;
            if s.len() != 3 {
                return Err(PyValueError::new_err("expected 3 values"));
            }
            Ok(Some([s[0], s[1], s[2]]))
        }
    }
}

fn points<'a>(a: &'a PyReadonlyArray2<f64>, n: usize, what: &str) -> PyResult<&'a [f64]> {
    let shape = a.shape();
    if shape[0] < n || shape[1] != 3 {
        return Err(PyValueError::new_err(format!("{what}: expected ({n}, 3) points")));
    }
    Ok(a.as_slice()?)
}

// ------------------------------------------------------------------------------------------- One Euro filter
/// One Euro filter (Casiez et al. 2012) on an array of any shape; `min_cutoff` is one value or one per
/// element of the last axis (as filters.OneEuroFilter broadcasts a tuple).
#[pyclass]
struct OneEuro {
    min_cutoff: Vec<f64>,
    beta: f64,
    d_cutoff: f64,
    x: Vec<f64>,
    dx: Vec<f64>,
    t: Option<f64>,
}

#[inline]
fn alpha(cutoff: f64, dt: f64) -> f64 {
    let tau = 1.0 / (2.0 * std::f64::consts::PI * cutoff);
    1.0 / (1.0 + tau / dt)
}

impl OneEuro {
    fn step(&mut self, x: &[f64], t: f64) -> bool {
        let reset = match self.t {
            None => true,
            Some(t0) => t <= t0 || self.x.len() != x.len(),
        };
        if reset {
            self.x = x.to_vec();
            self.dx = vec![0.0; x.len()];
            self.t = Some(t);
            return true;
        }
        let dt = t - self.t.unwrap();
        let a_d = alpha(self.d_cutoff, dt);
        let n = self.min_cutoff.len();
        for i in 0..x.len() {
            let dx = (x[i] - self.x[i]) / dt;
            self.dx[i] = a_d * dx + (1.0 - a_d) * self.dx[i];
            let cutoff = self.min_cutoff[i % n] + self.beta * self.dx[i].abs();
            let a = alpha(cutoff, dt);
            self.x[i] = a * x[i] + (1.0 - a) * self.x[i];
        }
        self.t = Some(t);
        false
    }
}

#[pymethods]
impl OneEuro {
    #[new]
    #[pyo3(signature = (min_cutoff, beta=0.05, d_cutoff=1.0))]
    fn new(min_cutoff: Vec<f64>, beta: f64, d_cutoff: f64) -> PyResult<Self> {
        if min_cutoff.is_empty() {
            return Err(PyValueError::new_err("min_cutoff is empty"));
        }
        Ok(OneEuro { min_cutoff, beta, d_cutoff, x: vec![], dx: vec![], t: None })
    }

    /// The filtered array (a new array of the input's shape).
    fn __call__<'py>(&mut self, py: Python<'py>, x: PyReadonlyArrayDyn<'py, f64>, t: f64)
                     -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let shape = x.shape().to_vec();
        let data = x.as_array();
        let flat: Vec<f64> = data.iter().copied().collect();
        if self.min_cutoff.len() > 1 && shape.last().copied().unwrap_or(1) != self.min_cutoff.len() {
            return Err(PyValueError::new_err("min_cutoff has one value per element of the last axis"));
        }
        self.step(&flat, t);
        let out = numpy::ndarray::ArrayD::from_shape_vec(shape, self.x.clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(out.to_pyarray(py))
    }

    /// A scalar sample (no array overhead).
    fn scalar(&mut self, x: f64, t: f64) -> f64 {
        self.step(&[x], t);
        self.x[0]
    }

    fn reset(&mut self) {
        self.x.clear();
        self.dx.clear();
        self.t = None;
    }
}

// ------------------------------------------------------------------------------------------- AngleSmoother
/// filters.AngleSmoother: One Euro filtering of named angles in degrees, unwrapped across +-180, with NaN
/// and None passed through (and resetting that angle's filter).
#[pyclass]
struct AngleSmoother {
    min_cutoff: f64,
    beta: f64,
    d_cutoff: f64,
    f: HashMap<String, OneEuro>,
    last: HashMap<String, f64>,
}

#[pymethods]
impl AngleSmoother {
    #[new]
    #[pyo3(signature = (min_cutoff=1.0, beta=0.02, d_cutoff=1.0))]
    fn new(min_cutoff: f64, beta: f64, d_cutoff: f64) -> Self {
        AngleSmoother { min_cutoff, beta, d_cutoff, f: HashMap::new(), last: HashMap::new() }
    }

    fn __call__<'py>(&mut self, py: Python<'py>, values: &Bound<'py, PyDict>, t: f64) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in values.iter() {
            let key: String = k.extract()?;
            let x: Option<f64> = if v.is_none() { None } else { v.extract::<f64>().ok() };
            let x = match x {
                Some(x) if x.is_finite() => x,
                _ => {
                    self.f.remove(&key);
                    self.last.remove(&key);
                    out.set_item(k, v)?;
                    continue;
                }
            };
            let x = match self.last.get(&key) {
                Some(&l) => l + (pymod(x - l + 180.0, 360.0) - 180.0),
                None => x,
            };
            let (mc, b, dc) = (self.min_cutoff, self.beta, self.d_cutoff);
            let f = self.f.entry(key.clone()).or_insert_with(|| OneEuro {
                min_cutoff: vec![mc], beta: b, d_cutoff: dc, x: vec![], dx: vec![], t: None });
            f.step(&[x], t);
            let y = f.x[0];
            self.last.insert(key, y);
            out.set_item(k, if y.abs() > 180.0 { pymod(y + 180.0, 360.0) - 180.0 } else { y })?;
        }
        Ok(out)
    }

    fn reset(&mut self) {
        self.f.clear();
        self.last.clear();
    }
}

// ------------------------------------------------------------------------------------------- solvers
#[pyclass]
struct ArmSolver(solvers::Arm);

#[pymethods]
impl ArmSolver {
    #[new]
    fn new(side: &str, min_elbow_deg: f64, pole_deg: f64) -> Self {
        ArmSolver(solvers::Arm { left: side == "left", min_elbow: min_elbow_deg, pole: pole_deg, tau: 0.0, phi: 0.0 })
    }

    /// (7 angles in degrees, shoulder rotation valid, wrist valid); see kinematics.ArmKinematics.solve.
    #[pyo3(signature = (pose, hand=None, up_hint=None))]
    fn solve(&mut self, pose: PyReadonlyArray2<f64>, hand: Option<PyReadonlyArray2<f64>>,
             up_hint: Option<PyReadonlyArrayDyn<f64>>) -> PyResult<([f64; 7], bool, bool)> {
        let p = points(&pose, 25, "pose")?;
        let h = match &hand {
            Some(h) => Some(points(h, 21, "hand")?),
            None => None,
        };
        let o = self.0.solve(p, h, as_v3(up_hint)?);
        Ok((o.angles, o.rotation_valid, o.wrist_valid))
    }

    fn reset(&mut self) {
        self.0.tau = 0.0;
        self.0.phi = 0.0;
    }
}

#[pyclass]
struct LegSolver(solvers::Leg);

#[pymethods]
impl LegSolver {
    #[new]
    fn new(side: &str, min_knee_deg: f64, pole_deg: f64) -> Self {
        LegSolver(solvers::Leg { left: side == "left", min_knee: min_knee_deg, pole: pole_deg, tau: 0.0, phi: 0.0 })
    }

    /// (10 angles in degrees, hip rotation source 0 held / 1 knee / 2 foot, ankle valid); see
    /// legs.LegKinematics.solve. `correction` is its foot_correction matrix.
    fn solve(&mut self, pose: PyReadonlyArray2<f64>, up: PyReadonlyArrayDyn<f64>, foot: bool,
             correction: PyReadonlyArray2<f64>) -> PyResult<([f64; 10], u8, bool)> {
        let p = points(&pose, 33, "pose")?;
        let c = correction.as_slice()?;
        if c.len() != 9 {
            return Err(PyValueError::new_err("correction: expected a 3x3 matrix"));
        }
        let cm: M3 = [[c[0], c[1], c[2]], [c[3], c[4], c[5]], [c[6], c[7], c[8]]];
        let o = self.0.solve(p, as_v3(Some(up))?.unwrap(), foot, &cm);
        Ok((o.angles, o.source, o.ankle_valid))
    }

    fn reset(&mut self) {
        self.0.tau = 0.0;
        self.0.phi = 0.0;
    }
}

#[pyclass]
struct HandSolver(solvers::Hand);

#[pymethods]
impl HandSolver {
    #[new]
    fn new(side: &str) -> Self {
        HandSolver(solvers::Hand { left: side == "left", tau: -std::f64::consts::FRAC_PI_2 })
    }

    /// 21 finger angles in degrees; see fingers.HandKinematics.solve.
    fn solve(&mut self, world: PyReadonlyArray2<f64>) -> PyResult<[f64; 21]> {
        Ok(self.0.solve(points(&world, 21, "hand")?))
    }
}

/// legs.core_angles in degrees: (flexion, lateral bend, rotation).
#[pyfunction]
fn core_angles(pose: PyReadonlyArray2<f64>, up: PyReadonlyArrayDyn<f64>) -> PyResult<(f64, f64, f64)> {
    Ok(solvers::core_angles(points(&pose, 25, "pose")?, as_v3(Some(up))?.unwrap()))
}

#[pymodule]
fn tandem_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OneEuro>()?;
    m.add_class::<AngleSmoother>()?;
    m.add_class::<ArmSolver>()?;
    m.add_class::<LegSolver>()?;
    m.add_class::<HandSolver>()?;
    m.add_function(wrap_pyfunction!(core_angles, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
