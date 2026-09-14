//! N-gram language model stored as a flat table of quantised conditional
//! log-probabilities, indexed by the base-26 rolling hash of the n-gram.

use crate::rng::Rng;
use std::fmt;
use std::io::{Read, Write};

pub const ALPHABET: usize = 26;
pub const MIN_ORDER: usize = 2;
pub const MAX_ORDER: usize = 5;
/// Table entries are log-probabilities in millinats. Integer scores make the
/// incremental update exact, so it can be checked for equality, not closeness.
pub const SCALE: f64 = 1000.0;

const MAGIC: &[u8; 4] = b"HPHM";
const VERSION: u8 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    BadOrder(usize),
    CorpusTooShort { letters: usize, order: usize },
    Io(String),
    Format(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::BadOrder(n) => {
                write!(
                    f,
                    "model order must be between {MIN_ORDER} and {MAX_ORDER}, got {n}"
                )
            }
            ModelError::CorpusTooShort { letters, order } => write!(
                f,
                "corpus has {letters} letters, not enough for a single {order}-gram"
            ),
            ModelError::Io(e) => write!(f, "i/o error: {e}"),
            ModelError::Format(e) => write!(f, "invalid model file: {e}"),
        }
    }
}

impl std::error::Error for ModelError {}

/// Maps `A-Z`/`a-z` to `0..26` and drops everything else, as is conventional
/// for classical ciphers (no spaces or punctuation).
pub fn letters_from_text(text: &str) -> Vec<u8> {
    text.bytes()
        .filter(|b| b.is_ascii_alphabetic())
        .map(|b| b.to_ascii_uppercase() - b'A')
        .collect()
}

pub fn letters_to_string(letters: &[u8]) -> String {
    letters.iter().map(|&l| (b'A' + l) as char).collect()
}

#[derive(Clone, Debug)]
pub struct NgramModel {
    order: usize,
    table: Vec<i16>,
    floor: i16,
}

impl NgramModel {
    /// Estimates `log P(last letter | previous order-1 letters)` from a corpus.
    ///
    /// Seen n-grams get their maximum-likelihood conditional probability.
    /// Every unseen n-gram gets one shared floor: the value of an n-gram seen
    /// half a time in the busiest context. That is strictly below every
    /// observed value, so garbage text never outscores observed text, and it
    /// stays finite so a single unseen window cannot sink a candidate key.
    pub fn train(letters: &[u8], order: usize) -> Result<Self, ModelError> {
        check_order(order)?;
        if letters.len() < order {
            return Err(ModelError::CorpusTooShort {
                letters: letters.len(),
                order,
            });
        }
        let size = ALPHABET.pow(order as u32);
        let mut counts = vec![0u32; size];
        let high = ALPHABET.pow(order as u32 - 1);
        let mut h = 0usize;
        for &l in &letters[..order] {
            h = h * ALPHABET + l as usize;
        }
        counts[h] += 1;
        for i in order..letters.len() {
            h = (h - letters[i - order] as usize * high) * ALPHABET + letters[i] as usize;
            counts[h] = counts[h].saturating_add(1);
        }

        let contexts = size / ALPHABET;
        let context_totals: Vec<u64> = (0..contexts)
            .map(|c| {
                counts[c * ALPHABET..(c + 1) * ALPHABET]
                    .iter()
                    .map(|&x| x as u64)
                    .sum()
            })
            .collect();
        let busiest = context_totals.iter().copied().max().unwrap_or(1).max(1);
        let floor = quantise((0.5 / busiest as f64).ln());

        let mut table = vec![floor; size];
        for c in 0..contexts {
            let total = context_totals[c];
            if total == 0 {
                continue;
            }
            for l in 0..ALPHABET {
                let count = counts[c * ALPHABET + l];
                if count > 0 {
                    table[c * ALPHABET + l] = quantise((count as f64 / total as f64).ln());
                }
            }
        }
        Ok(NgramModel {
            order,
            table,
            floor,
        })
    }

    pub fn order(&self) -> usize {
        self.order
    }

    /// Score given to every n-gram absent from the training corpus.
    pub fn floor(&self) -> i16 {
        self.floor
    }

    pub fn table(&self) -> &[i16] {
        &self.table
    }

    /// Log-probability (millinats) of one n-gram given as letters `0..26`.
    pub fn ngram_score(&self, ngram: &[u8]) -> i16 {
        assert_eq!(
            ngram.len(),
            self.order,
            "n-gram length must equal model order"
        );
        let h = ngram.iter().fold(0usize, |h, &l| h * ALPHABET + l as usize);
        self.table[h]
    }

    /// Sum of every window's score, computed from scratch with a rolling hash.
    /// Text shorter than the order has no windows and scores 0.
    pub fn score_text(&self, letters: &[u8]) -> i64 {
        let n = self.order;
        if letters.len() < n {
            return 0;
        }
        let high = ALPHABET.pow(n as u32 - 1);
        let mut h = letters[..n]
            .iter()
            .fold(0usize, |h, &l| h * ALPHABET + l as usize);
        let mut total = self.table[h] as i64;
        for i in n..letters.len() {
            h = (h - letters[i - n] as usize * high) * ALPHABET + letters[i] as usize;
            total += self.table[h] as i64;
        }
        total
    }

    /// Samples text from the model's own conditional distributions, after a
    /// burn-in so the random starting context does not bias the output.
    ///
    /// Floor entries are excluded: the floor is a scoring penalty, not an
    /// estimate of probability mass, and sampling from it would inject noise.
    /// A context never seen in training continues uniformly.
    pub fn sample(&self, len: usize, rng: &mut Rng) -> Vec<u8> {
        let n = self.order;
        let burn_in = 256;
        let mut out: Vec<u8> = (0..n - 1).map(|_| rng.below(ALPHABET) as u8).collect();
        let mut weights = [0f64; ALPHABET];
        while out.len() < len + burn_in + n - 1 {
            let ctx = out[out.len() - (n - 1)..]
                .iter()
                .fold(0usize, |h, &l| h * ALPHABET + l as usize);
            for (l, w) in weights.iter_mut().enumerate() {
                let v = self.table[ctx * ALPHABET + l];
                *w = if v > self.floor {
                    (v as f64 / SCALE).exp()
                } else {
                    0.0
                };
            }
            out.push(rng.weighted(&weights) as u8);
        }
        out.split_off(burn_in + n - 1)
    }

    pub fn write_to<W: Write>(&self, mut w: W) -> Result<(), ModelError> {
        let io = |e: std::io::Error| ModelError::Io(e.to_string());
        w.write_all(MAGIC).map_err(io)?;
        w.write_all(&[VERSION, self.order as u8]).map_err(io)?;
        w.write_all(&self.floor.to_le_bytes()).map_err(io)?;
        let mut bytes = Vec::with_capacity(self.table.len() * 2);
        for v in &self.table {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        w.write_all(&bytes).map_err(io)?;
        w.flush().map_err(io)
    }

    pub fn read_from<R: Read>(mut r: R) -> Result<Self, ModelError> {
        let mut bytes = Vec::new();
        r.read_to_end(&mut bytes)
            .map_err(|e| ModelError::Io(e.to_string()))?;
        if bytes.len() < 8 || &bytes[..4] != MAGIC {
            return Err(ModelError::Format("missing HPHM header".into()));
        }
        if bytes[4] != VERSION {
            return Err(ModelError::Format(format!(
                "unsupported version {}",
                bytes[4]
            )));
        }
        let order = bytes[5] as usize;
        check_order(order)?;
        let floor = i16::from_le_bytes([bytes[6], bytes[7]]);
        let size = ALPHABET.pow(order as u32);
        let body = &bytes[8..];
        if body.len() != size * 2 {
            return Err(ModelError::Format(format!(
                "expected {} table bytes for order {order}, found {}",
                size * 2,
                body.len()
            )));
        }
        // chunks_exact rather than as_chunks: the latter needs Rust 1.88, above
        // the MSRV this crate declares and CI tests.
        let table = body
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        Ok(NgramModel {
            order,
            table,
            floor,
        })
    }
}

fn check_order(order: usize) -> Result<(), ModelError> {
    if (MIN_ORDER..=MAX_ORDER).contains(&order) {
        Ok(())
    } else {
        Err(ModelError::BadOrder(order))
    }
}

fn quantise(ln_p: f64) -> i16 {
    (ln_p * SCALE).round().clamp(i16::MIN as f64 + 1.0, 0.0) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conditional_probabilities_match_counts() {
        // Context "AB" is followed by C twice and D once.
        let text = letters_from_text("ABCABCABD");
        let m = NgramModel::train(&text, 3).unwrap();
        let abc = m.ngram_score(&letters_from_text("ABC")) as f64 / SCALE;
        let abd = m.ngram_score(&letters_from_text("ABD")) as f64 / SCALE;
        assert!((abc - (2.0f64 / 3.0).ln()).abs() < 1e-3);
        assert!((abd - (1.0f64 / 3.0).ln()).abs() < 1e-3);
        // "BCA" is the only continuation of "BC".
        assert_eq!(m.ngram_score(&letters_from_text("BCA")), 0);
    }

    #[test]
    fn rolling_score_equals_sum_of_windows() {
        let mut rng = Rng::new(5);
        let corpus: Vec<u8> = (0..5000).map(|_| rng.below(26) as u8).collect();
        for order in MIN_ORDER..=MAX_ORDER {
            let m = NgramModel::train(&corpus, order).unwrap();
            let text = &corpus[100..400];
            let naive: i64 = text.windows(order).map(|w| m.ngram_score(w) as i64).sum();
            assert_eq!(m.score_text(text), naive, "order {order}");
        }
    }

    #[test]
    fn round_trips_through_bytes_and_rejects_corruption() {
        let text = letters_from_text("the quick brown fox jumps over the lazy dog");
        let m = NgramModel::train(&text, 4).unwrap();
        let mut buf = Vec::new();
        m.write_to(&mut buf).unwrap();
        let back = NgramModel::read_from(&buf[..]).unwrap();
        assert_eq!(back.table(), m.table());
        assert_eq!(back.floor(), m.floor());

        let truncated = &buf[..buf.len() - 2];
        assert!(matches!(
            NgramModel::read_from(truncated),
            Err(ModelError::Format(_))
        ));
        let mut bad_magic = buf.clone();
        bad_magic[0] = b'X';
        assert!(matches!(
            NgramModel::read_from(&bad_magic[..]),
            Err(ModelError::Format(_))
        ));
        let mut bad_order = buf.clone();
        bad_order[5] = 9;
        assert_eq!(
            NgramModel::read_from(&bad_order[..]).unwrap_err(),
            ModelError::BadOrder(9)
        );
    }

    #[test]
    fn rejects_bad_orders_and_short_corpora() {
        let text = letters_from_text("abcdef");
        assert_eq!(
            NgramModel::train(&text, 1).unwrap_err(),
            ModelError::BadOrder(1)
        );
        assert_eq!(
            NgramModel::train(&text, 6).unwrap_err(),
            ModelError::BadOrder(6)
        );
        assert_eq!(
            NgramModel::train(&text[..2], 3).unwrap_err(),
            ModelError::CorpusTooShort {
                letters: 2,
                order: 3
            }
        );
    }

    #[test]
    fn sampling_follows_a_deterministic_model() {
        // Every context in this corpus has exactly one continuation.
        let text = letters_from_text(&"ABCDE".repeat(200));
        let m = NgramModel::train(&text, 3).unwrap();
        // A random start context is usually unseen and wanders uniformly until
        // it lands in the cycle, after which it can never leave.
        let s = m.sample(3000, &mut Rng::new(11));
        assert_eq!(s.len(), 3000);
        assert!(s[1000..].windows(2).all(|w| (w[0] + 1) % 5 == w[1]));
    }
}
