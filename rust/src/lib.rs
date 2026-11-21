pub mod basic_tokenizer;
pub mod llama2_tokenizer;
mod sentencepiece_bpe;
mod sentencepiece_training;
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
#[pyclass]
struct NativeLlama2Tokenizer {
    inner: llama2_tokenizer::Llama2Tokenizer,
}
#[cfg(feature = "python")]
#[pymethods]
impl NativeLlama2Tokenizer {
    #[staticmethod]
    fn train(py: Python<'_>, text: &str, vocab_size: usize) -> PyResult<Self> {
        let inner = py
            .detach(|| llama2_tokenizer::Llama2Tokenizer::train(text, vocab_size))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner })
    }

    #[new]
    fn new(model_file: &str) -> PyResult<Self> {
        Ok(Self {
            inner: llama2_tokenizer::Llama2Tokenizer::load(model_file)
                .map_err(pyo3::exceptions::PyValueError::new_err)?,
        })
    }
    #[staticmethod]
    fn from_bytes(model: &[u8]) -> PyResult<Self> {
        Ok(Self {
            inner: llama2_tokenizer::Llama2Tokenizer::from_bytes(model)
                .map_err(pyo3::exceptions::PyValueError::new_err)?,
        })
    }
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        pyo3::types::PyBytes::new(py, self.inner.to_bytes())
    }
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner
            .save(path)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
    #[pyo3(signature = (text, *, lossless=true, add_bos=false, add_eos=false))]
    fn encode(
        &self,
        py: Python<'_>,
        text: &str,
        lossless: bool,
        add_bos: bool,
        add_eos: bool,
    ) -> PyResult<Vec<usize>> {
        py.detach(|| {
            self.inner
                .encode_with_options(text, lossless, add_bos, add_eos)
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)
    }
    #[pyo3(signature = (texts, *, lossless=true))]
    fn encode_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        lossless: bool,
    ) -> PyResult<Vec<Vec<usize>>> {
        py.detach(|| self.inner.encode_batch(&texts, lossless))
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
    fn decode(&self, py: Python<'_>, ids: Vec<Bound<'_, PyAny>>) -> PyResult<String> {
        let ids = ids
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
            .collect::<PyResult<Vec<_>>>()?;
        py.detach(|| self.inner.decode(&ids))
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn bos_id(&self) -> Option<usize> {
        self.inner.bos_id()
    }
    fn eos_id(&self) -> Option<usize> {
        self.inner.eos_id()
    }
    fn piece(&self, id: usize) -> PyResult<String> {
        self.inner
            .piece(id)
            .map(str::to_owned)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn thoth_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeTokenizer>()?;
    m.add_class::<NativeLlama2Tokenizer>()?;
    Ok(())
}
