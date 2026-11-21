pub mod basic_tokenizer;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
#[pyclass]
struct NativeTokenizer {
    inner: basic_tokenizer::BasicTokenizer,
}
#[cfg(feature = "python")]
#[pymethods]
impl NativeTokenizer {
    #[new]
    fn new() -> Self {
        Self {
            inner: basic_tokenizer::BasicTokenizer::new(),
        }
    }
    fn train(&mut self, text: &str, vocab_size: usize) -> PyResult<()> {
        self.inner
            .train(text, vocab_size)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
    fn encode(&self, text: &str) -> Vec<usize> {
        self.inner.encoder(text)
    }
    fn encode_batch(&self, texts: Vec<String>) -> Vec<Vec<usize>> {
        texts.iter().map(|s| self.inner.encoder(s)).collect()
    }
    fn decode(&self, ids: Vec<Bound<'_, PyAny>>) -> PyResult<String> {
        let ids: Vec<usize> = ids
            .iter()
            .map(|id| {
                if !id.is_exact_instance_of::<pyo3::types::PyInt>() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Token IDs must be nonnegative integers",
                    ));
                }
                id.extract::<usize>().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(
                        "Token IDs must be nonnegative integers",
                    )
                })
            })
            .collect::<PyResult<_>>()?;
        self.inner
            .decoder(ids)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
    #[staticmethod]
    fn from_json(data: &str) -> PyResult<Self> {
        Ok(Self {
            inner: basic_tokenizer::BasicTokenizer::from_json(data)
                .map_err(pyo3::exceptions::PyValueError::new_err)?,
        })
    }
}
#[cfg(feature = "python")]
#[pymodule]
fn thoth_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeTokenizer>()?;
    Ok(())
}
