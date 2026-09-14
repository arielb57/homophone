//! Shared fixtures: a seeded synthetic language, a model trained on a corpus
//! sampled from it, and random homophonic encryptions of fresh samples.

#![allow(dead_code)]

use homophone::cipher::letter_frequencies;
use homophone::{HomophonicKey, MarkovLanguage, NgramModel, Rng};
use std::sync::OnceLock;

pub const LANGUAGE_SEED: u64 = 2026;
pub const ZIPF_EXPONENT: f64 = 1.5;
pub const CORPUS_LETTERS: usize = 1_000_000;

pub struct Fixture {
    pub language: MarkovLanguage,
    pub model: NgramModel,
}

/// Trained once per test binary; the corpus is disjoint from every test plaintext
/// because it uses its own sampling stream.
pub fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let language = MarkovLanguage::new(3, ZIPF_EXPONENT, LANGUAGE_SEED);
        let corpus = language.sample(CORPUS_LETTERS, &mut Rng::new(u64::MAX));
        let model = NgramModel::train(&corpus, 3).expect("corpus is long enough");
        Fixture { language, model }
    })
}

pub struct Case {
    pub plaintext: Vec<u8>,
    pub key: HomophonicKey,
    pub ciphertext: Vec<u16>,
}

/// Plaintext from the language, key with `symbols` symbols and 1–4 homophones
/// per letter, extra homophones given to frequent letters.
pub fn case(seed: u64, len: usize, symbols: usize) -> Case {
    let mut rng = Rng::new(seed);
    let plaintext = fixture().language.sample(len, &mut rng);
    let key = HomophonicKey::random(symbols, 4, &letter_frequencies(&plaintext), &mut rng)
        .expect("valid symbol count");
    let ciphertext = key.encrypt(&plaintext, &mut rng);
    Case {
        plaintext,
        key,
        ciphertext,
    }
}

/// Symbol count drawn uniformly from 40..=80, as in the specification.
pub fn random_symbol_count(seed: u64) -> usize {
    40 + Rng::derive(seed, 0xC0FFEE).below(41)
}
