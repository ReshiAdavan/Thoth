//! SentencePiece model execution in Rust.
//!
//! Uses the pure-Rust sentencepiece-rs runtime. Loading the original model is
//! required for Llama vocabulary compatibility; no chat template is implied.
use sentencepiece_rs::SentencePieceProcessor;
use std::io::Read;
use std::path::Path;

const MAX_MODEL_BYTES: usize = 16 * 1024 * 1024;

pub struct Llama2Tokenizer {
    processor: SentencePieceProcessor,
    serialized: Vec<u8>,
    fast_bpe: Option<crate::sentencepiece_bpe::FastBpe>,
}

impl Llama2Tokenizer {
    /// Train a custom Thoth Unicode BPE vocabulary, not original Llama IDs.
    pub fn train(text: &str, vocab_size: usize) -> Result<Self, String> {
        Self::from_bytes(&crate::sentencepiece_training::train(text, vocab_size)?)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.is_empty() || bytes.len() > MAX_MODEL_BYTES {
            return Err("SentencePiece model must be nonempty and at most 16 MiB".into());
        }
        let processor =
            SentencePieceProcessor::from_serialized_model(bytes).map_err(|e| e.to_string())?;
        let fast_bpe = crate::sentencepiece_bpe::FastBpe::new(&processor);
        Ok(Self {
            fast_bpe,
            processor,
            serialized: bytes.to_vec(),
        })
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let mut bytes = Vec::new();
        file.take((MAX_MODEL_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        Self::from_bytes(&bytes)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        std::fs::write(path, &self.serialized).map_err(|e| e.to_string())
    }

    pub fn to_bytes(&self) -> &[u8] {
        &self.serialized
    }

    pub fn vocab_size(&self) -> usize {
        self.processor.model().vocab_size()
    }

    pub fn bos_id(&self) -> Option<usize> {
        self.processor.bos_id()
    }
    pub fn eos_id(&self) -> Option<usize> {
        self.processor.eos_id()
    }

    /// Canonical model normalization and segmentation, with no implicit framing.
    pub fn encode(&self, text: &str) -> Result<Vec<usize>, String> {
        if let Some(fast) = &self.fast_bpe {
            let normalized = self.processor.normalize(text).map_err(|e| e.to_string())?;
            return Ok(fast.encode_normalized(&normalized));
        }

        self.processor
            .encode_to_ids(text)
            .map_err(|e| e.to_string())
    }

    /// Exact recovery or an error. Exceptional text may use explicit byte IDs,
    /// which are not canonical Llama segmentation and can increase token counts.
    pub fn encode_lossless(&self, text: &str) -> Result<Vec<usize>, String> {
        let ids = self.encode(text)?;
        if self.decode(&ids)? == text {
            return Ok(ids);
        }
        let mut fallback = Vec::with_capacity(text.len());
        for byte in text.bytes() {
            let piece = format!("<0x{byte:02X}>");
            let id = self
                .processor
                .model()
                .try_piece_to_id(&piece)
                .filter(|&id| self.processor.model().is_byte(id))
                .ok_or("This SentencePiece model cannot preserve the input exactly")?;
            fallback.push(id);
        }
        if self.decode(&fallback)? != text {
            return Err("This SentencePiece model cannot preserve the input exactly".into());
        }
        Ok(fallback)
    }

    pub fn encode_with_options(
        &self,
        text: &str,
        lossless: bool,
        add_bos: bool,
        add_eos: bool,
    ) -> Result<Vec<usize>, String> {
        let mut ids = if lossless {
            self.encode_lossless(text)?
        } else {
            self.encode(text)?
        };
        if add_bos {
            ids.insert(0, self.bos_id().ok_or("Model has no BOS token")?);
        }
        if add_eos {
            ids.push(self.eos_id().ok_or("Model has no EOS token")?);
        }
        Ok(ids)
    }

    pub fn encode_batch(
        &self,
        texts: &[String],
        lossless: bool,
    ) -> Result<Vec<Vec<usize>>, String> {
        texts
            .iter()
            .map(|text| self.encode_with_options(text, lossless, false, false))
            .collect()
    }

    pub fn decode(&self, ids: &[usize]) -> Result<String, String> {
        if ids.iter().any(|&id| id >= self.vocab_size()) {
            return Err("Token ID is outside the model vocabulary".into());
        }
        self.processor.decode_ids(ids).map_err(|e| e.to_string())
    }

    pub fn piece(&self, id: usize) -> Result<&str, String> {
        self.processor.id_to_piece(id).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const MODEL: &[u8] = include_bytes!("../../models/thoth-sentencepiece-en-code-v1.model");

    #[test]
    fn saved_model_edges_and_framing() {
        let t = Llama2Tokenizer::from_bytes(MODEL).unwrap();
        assert_eq!(t.vocab_size(), 16384);
        for text in ["", "Hello 🦉", "a▁b", "  a\r\n\t\0", "e\u{301} 中文"] {
            assert_eq!(t.decode(&t.encode_lossless(text).unwrap()).unwrap(), text);
        }
        let ids = t.encode_with_options("hello", true, true, true).unwrap();
        assert_eq!(ids.first(), t.bos_id().as_ref());
        assert_eq!(ids.last(), t.eos_id().as_ref());
        assert_eq!(t.decode(&ids).unwrap(), "hello");
        assert_eq!(t.to_bytes(), MODEL);
        assert!(t.decode(&[usize::MAX]).is_err());
        assert!(t.piece(usize::MAX).is_err());
    }

    #[test]
    fn invalid_models_return_errors() {
        for bytes in [&[][..], &[0xff, 0xff][..], b"not a model"] {
            assert!(Llama2Tokenizer::from_bytes(bytes).is_err());
        }
        assert!(Llama2Tokenizer::from_bytes(&vec![0; MAX_MODEL_BYTES + 1]).is_err());
    }

    #[test]
    fn native_training_and_reload() {
        let t = Llama2Tokenizer::train("abababab hello hello world", 300).unwrap();
        let copy = Llama2Tokenizer::from_bytes(t.to_bytes()).unwrap();
        assert!(t.vocab_size() <= 300);
        assert!(t.encode("abababab").unwrap().len() < 8);
        for text in ["hello world", "unseen 中文 🦉", "a▁b", "\r\n\0"] {
            let ids = t.encode_lossless(text).unwrap();
            assert_eq!(ids, copy.encode_lossless(text).unwrap());
            assert_eq!(copy.decode(&ids).unwrap(), text);
        }
        assert_eq!(
            t.to_bytes(),
            Llama2Tokenizer::train("abababab hello hello world", 300)
                .unwrap()
                .to_bytes()
        );
        assert!(Llama2Tokenizer::train("", 300).is_err());
        assert!(Llama2Tokenizer::train("hello", 259).is_err());
    }
}
