use pyo3::prelude::*;
use numpy::{IntoPyArray, PyReadonlyArray3, PyArray3};
use rand_pcg::rand_core::SeedableRng;
use rand_pcg::Pcg64Dxsm;

mod shuffle;


#[pyfunction]
#[pyo3(name = "_shuffle")]
#[pyo3(text_signature = "(seqs, seed)")]
fn batched_shuffle_py(
    py: Python,
    py_seqs: PyReadonlyArray3<u8>,
    seed: u64,
) -> Py<PyArray3<u8>> {
    let seqs = py_seqs.as_array().to_owned(); 

    let shuffled = py.allow_threads(|| {
        let mut rng = Pcg64Dxsm::seed_from_u64(seed);
        shuffle::batched_shuffle(&seqs.view(), &mut rng)
    });

    shuffled.into_pyarray(py).into()
}


#[pymodule]
#[pyo3(name="dinuc_shuf")]
fn dinuc_shuf_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(batched_shuffle_py, m)?)?;

    Ok(())
}


