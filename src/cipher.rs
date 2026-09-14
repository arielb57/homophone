//! Homophonic keys, encryption, ciphertext parsing and recovery measurement.

use crate::model::ALPHABET;
use crate::rng::Rng;
use std::collections::HashMap;

/// A homophonic substitution key: each plaintext letter owns one or more
/// ciphertext symbols, and each symbol decrypts to exactly one letter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HomophonicKey {
    /// `symbol_letter[s]` is the plaintext letter of symbol `s`.
    pub symbol_letter: Vec<u8>,
    /// `homophones[l]` lists the symbols that encrypt letter `l`.
    pub homophones: Vec<Vec<u16>>,
}

impl HomophonicKey {
    /// Builds a random key with `num_symbols` symbols and at most
    /// `max_per_letter` homophones per letter. Every letter gets one symbol;
    /// the extra symbols go to letters in proportion to `letter_weights`,
    /// which is how a cipher designer flattens the frequency profile.
    pub fn random(
        num_symbols: usize,
        max_per_letter: usize,
        letter_weights: &[f64; ALPHABET],
        rng: &mut Rng,
    ) -> Result<Self, String> {
        if max_per_letter == 0 || num_symbols < ALPHABET || num_symbols > ALPHABET * max_per_letter
        {
            return Err(format!(
                "cannot build a key with {num_symbols} symbols and at most {max_per_letter} \
                 homophones per letter (need {ALPHABET}..={})",
                ALPHABET * max_per_letter
            ));
        }
        let mut per_letter = [1usize; ALPHABET];
        for _ in ALPHABET..num_symbols {
            let weights: Vec<f64> = (0..ALPHABET)
                .map(|l| {
                    if per_letter[l] < max_per_letter {
                        // Small epsilon keeps letters absent from the sample eligible.
                        letter_weights[l].max(0.0) + 1e-9
                    } else {
                        0.0
                    }
                })
                .collect();
            per_letter[rng.weighted(&weights)] += 1;
        }
        let mut symbols: Vec<u16> = (0..num_symbols as u16).collect();
        rng.shuffle(&mut symbols);
        let mut symbol_letter = vec![0u8; num_symbols];
        let mut homophones = vec![Vec::new(); ALPHABET];
        let mut next = symbols.into_iter();
        for (letter, &count) in per_letter.iter().enumerate() {
            for _ in 0..count {
                let s = next.next().expect("symbol count matches allocation");
                symbol_letter[s as usize] = letter as u8;
                homophones[letter].push(s);
            }
        }
        Ok(HomophonicKey {
            symbol_letter,
            homophones,
        })
    }

    pub fn num_symbols(&self) -> usize {
        self.symbol_letter.len()
    }

    /// Each occurrence of a letter uses one of its homophones chosen uniformly.
    pub fn encrypt(&self, plaintext: &[u8], rng: &mut Rng) -> Vec<u16> {
        plaintext
            .iter()
            .map(|&l| {
                let options = &self.homophones[l as usize];
                options[rng.below(options.len())]
            })
            .collect()
    }

    pub fn decrypt(&self, ciphertext: &[u16]) -> Vec<u8> {
        decrypt_with(&self.symbol_letter, ciphertext)
    }
}

pub fn decrypt_with(symbol_letter: &[u8], ciphertext: &[u16]) -> Vec<u8> {
    ciphertext
        .iter()
        .map(|&s| symbol_letter[s as usize])
        .collect()
}

/// Relative frequency of each letter in `letters`.
pub fn letter_frequencies(letters: &[u8]) -> [f64; ALPHABET] {
    let mut f = [0f64; ALPHABET];
    for &l in letters {
        f[l as usize] += 1.0;
    }
    let n = letters.len().max(1) as f64;
    f.map(|c| c / n)
}

/// Fraction of positions where `guess` matches `truth`.
pub fn recovery_rate(truth: &[u8], guess: &[u8]) -> f64 {
    assert_eq!(truth.len(), guess.len(), "recovery needs equal lengths");
    if truth.is_empty() {
        return 1.0;
    }
    let hits = truth.iter().zip(guess).filter(|(a, b)| a == b).count();
    hits as f64 / truth.len() as f64
}

/// A ciphertext with its symbols renumbered densely from 0, in order of first
/// appearance, plus the original spelling of each symbol.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ciphertext {
    pub symbols: Vec<u16>,
    pub names: Vec<String>,
}

impl Ciphertext {
    pub fn num_symbols(&self) -> usize {
        self.names.len()
    }

    /// Whitespace-separated tokens, e.g. `12 7 33 7`. Any token is a symbol.
    pub fn parse_tokens(text: &str) -> Result<Self, String> {
        Self::from_names(text.split_whitespace().map(str::to_string))
    }

    /// One symbol per non-whitespace character, e.g. `K+p9K`.
    pub fn parse_chars(text: &str) -> Result<Self, String> {
        Self::from_names(
            text.chars()
                .filter(|c| !c.is_whitespace())
                .map(String::from),
        )
    }

    fn from_names(names: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut index: HashMap<String, u16> = HashMap::new();
        let mut ordered = Vec::new();
        let mut symbols = Vec::new();
        for name in names {
            let next_id = ordered.len();
            let id = match index.get(&name) {
                Some(&id) => id,
                None => {
                    if next_id > u16::MAX as usize {
                        return Err("ciphertext has more than 65536 distinct symbols".into());
                    }
                    index.insert(name.clone(), next_id as u16);
                    ordered.push(name);
                    next_id as u16
                }
            };
            symbols.push(id);
        }
        Ok(Ciphertext {
            symbols,
            names: ordered,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_key_respects_bounds_and_round_trips() {
        let mut rng = Rng::new(2);
        let weights = [1.0 / 26.0; 26];
        for num_symbols in [26, 40, 63, 80, 104] {
            let key = HomophonicKey::random(num_symbols, 4, &weights, &mut rng).unwrap();
            assert_eq!(key.num_symbols(), num_symbols);
            assert!(key.homophones.iter().all(|h| (1..=4).contains(&h.len())));
            let mut all: Vec<u16> = key.homophones.concat();
            all.sort_unstable();
            assert_eq!(all, (0..num_symbols as u16).collect::<Vec<_>>());
            let plain: Vec<u8> = (0..500).map(|_| rng.below(26) as u8).collect();
            let cipher = key.encrypt(&plain, &mut rng);
            assert_eq!(key.decrypt(&cipher), plain);
        }
    }

    #[test]
    fn extra_homophones_follow_letter_weights() {
        let mut weights = [0.0; 26];
        weights[4] = 1.0;
        weights[19] = 1.0;
        let key = HomophonicKey::random(32, 4, &weights, &mut Rng::new(1)).unwrap();
        assert_eq!(key.homophones[4].len(), 4);
        assert_eq!(key.homophones[19].len(), 4);
    }

    #[test]
    fn impossible_keys_are_rejected() {
        let w = [1.0; 26];
        let mut rng = Rng::new(0);
        assert!(HomophonicKey::random(25, 4, &w, &mut rng).is_err());
        assert!(HomophonicKey::random(105, 4, &w, &mut rng).is_err());
        assert!(HomophonicKey::random(30, 0, &w, &mut rng).is_err());
    }

    #[test]
    fn parsing_renumbers_by_first_appearance() {
        let c = Ciphertext::parse_tokens("17 03 17\n99  03").unwrap();
        assert_eq!(c.symbols, vec![0, 1, 0, 2, 1]);
        assert_eq!(c.names, vec!["17", "03", "99"]);
        let c = Ciphertext::parse_chars("K+ pK").unwrap();
        assert_eq!(c.symbols, vec![0, 1, 2, 0]);
        assert_eq!(c.num_symbols(), 3);
    }

    #[test]
    fn recovery_counts_matching_positions() {
        assert_eq!(recovery_rate(&[1, 2, 3, 4], &[1, 0, 3, 0]), 0.5);
        assert_eq!(recovery_rate(&[], &[]), 1.0);
    }
}
