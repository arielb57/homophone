//! Incremental scorer: keeps the decryption and its score in sync with a
//! symbol-to-letter key, and rescores only the windows a move touches.
//!
//! The score is the log-likelihood of the *ciphertext* under a key, not just
//! of the decrypted text:
//!
//! ```text
//! score = Σ windows  log P(letter | context)                  n-gram term
//!       + Σ letters  ln Γ(k_l) − ln Γ(n_l + k_l)              homophone term
//!       + Σ symbols  ln n_s!
//! ```
//!
//! `n_l` is how often letter `l` occurs in the decryption, `k_l` how many
//! symbols map to it and `n_s` how often symbol `s` occurs. The homophone term
//! is `log P(symbol sequence | letter sequence)` when each letter picks among
//! its symbols with unknown probabilities under a uniform Dirichlet prior,
//! integrated out. Without it, mapping every symbol to one letter that loops
//! on itself in the model outscores the true key; with it, piling symbols onto
//! one letter costs what it should. It is O(1) to update per move.

use crate::model::{NgramModel, ALPHABET, SCALE};

const NO_PENDING: usize = usize::MAX;

pub struct Scorer<'m> {
    model: &'m NgramModel,
    order: usize,
    high: usize,
    /// Positions of symbol `s` are `positions[offsets[s]..offsets[s + 1]]`,
    /// ascending. One flat buffer keeps a move's lookups cache-local.
    offsets: Vec<u32>,
    positions: Vec<u32>,
    ln_factorial: Vec<i64>,
    key: Vec<u8>,
    plain: Vec<u8>,
    /// Occurrences of each letter in the decryption.
    letter_counts: [u32; ALPHABET],
    /// Number of occurring symbols currently mapped to each letter.
    letter_symbols: [u32; ALPHABET],
    score: i64,
    pending_symbol: usize,
    pending_old: u8,
    pending_delta: i64,
}

impl<'m> Scorer<'m> {
    /// `key[s]` is the initial letter of symbol `s`; every symbol in
    /// `ciphertext` must be below `key.len()`.
    pub fn new(model: &'m NgramModel, ciphertext: &[u16], key: Vec<u8>) -> Self {
        assert!(
            key.iter().all(|&l| (l as usize) < ALPHABET),
            "key letters must be < 26"
        );
        let num_symbols = key.len();
        let mut counts = vec![0u32; num_symbols + 1];
        for &s in ciphertext {
            assert!((s as usize) < num_symbols, "symbol {s} has no key entry");
            counts[s as usize + 1] += 1;
        }
        let mut offsets = counts;
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }
        let mut fill = offsets.clone();
        let mut positions = vec![0u32; ciphertext.len()];
        for (p, &s) in ciphertext.iter().enumerate() {
            positions[fill[s as usize] as usize] = p as u32;
            fill[s as usize] += 1;
        }
        let ln_factorial = ln_factorial_table(ciphertext.len() + num_symbols);
        let plain: Vec<u8> = ciphertext.iter().map(|&s| key[s as usize]).collect();
        let order = model.order();
        let mut scorer = Scorer {
            model,
            order,
            high: ALPHABET.pow(order as u32 - 1),
            offsets,
            positions,
            ln_factorial,
            key,
            plain,
            letter_counts: [0; ALPHABET],
            letter_symbols: [0; ALPHABET],
            score: 0,
            pending_symbol: NO_PENDING,
            pending_old: 0,
            pending_delta: 0,
        };
        scorer.recount();
        scorer
    }

    pub fn score(&self) -> i64 {
        self.score
    }

    pub fn key(&self) -> &[u8] {
        &self.key
    }

    pub fn plaintext(&self) -> &[u8] {
        &self.plain
    }

    pub fn num_symbols(&self) -> usize {
        self.key.len()
    }

    pub fn occurrences(&self, symbol: usize) -> usize {
        (self.offsets[symbol + 1] - self.offsets[symbol]) as usize
    }

    /// Score of the current decryption recomputed from nothing but the
    /// ciphertext, the key and the model.
    pub fn full_rescore(&self) -> i64 {
        let mut letters = [0usize; ALPHABET];
        for &l in &self.plain {
            letters[l as usize] += 1;
        }
        let mut symbols_per_letter = [0usize; ALPHABET];
        let mut symbol_term = 0i64;
        for s in 0..self.num_symbols() {
            let m = self.occurrences(s);
            if m > 0 {
                symbols_per_letter[self.key[s] as usize] += 1;
                symbol_term += self.ln_factorial[m];
            }
        }
        let letter_term: i64 = (0..ALPHABET)
            .map(|l| self.letter_term(letters[l], symbols_per_letter[l]))
            .sum();
        self.model.score_text(&self.plain) + symbol_term + letter_term
    }

    /// `ln Γ(k) − ln Γ(n + k)` for a letter with `n` occurrences spread over
    /// `k` symbols; a letter no symbol maps to contributes nothing.
    fn letter_term(&self, n: usize, k: usize) -> i64 {
        if k == 0 {
            0
        } else {
            self.ln_factorial[k - 1] - self.ln_factorial[n + k - 1]
        }
    }

    /// Tentatively assigns `letter` to `symbol` and returns the score change.
    /// Must be followed by [`commit`](Self::commit) or
    /// [`rollback`](Self::rollback) before the next proposal.
    pub fn propose(&mut self, symbol: usize, letter: u8) -> i64 {
        assert_eq!(
            self.pending_symbol, NO_PENDING,
            "previous proposal not resolved"
        );
        let old = self.key[symbol];
        self.pending_symbol = symbol;
        self.pending_old = old;
        if old == letter {
            self.pending_delta = 0;
            return 0;
        }
        let before = self.touched_sum(symbol);
        self.write_symbol(symbol, letter);
        let after = self.touched_sum(symbol);

        let m = self.occurrences(symbol);
        let homophone_delta = if m == 0 {
            0
        } else {
            let (a, b) = (old as usize, letter as usize);
            let (na, ka) = (
                self.letter_counts[a] as usize,
                self.letter_symbols[a] as usize,
            );
            let (nb, kb) = (
                self.letter_counts[b] as usize,
                self.letter_symbols[b] as usize,
            );
            self.letter_term(na - m, ka - 1) + self.letter_term(nb + m, kb + 1)
                - self.letter_term(na, ka)
                - self.letter_term(nb, kb)
        };

        self.pending_delta = after - before + homophone_delta;
        self.pending_delta
    }

    pub fn commit(&mut self) {
        let symbol = self.pending_symbol;
        assert_ne!(symbol, NO_PENDING, "nothing to commit");
        let m = self.occurrences(symbol) as u32;
        if m > 0 {
            let (a, b) = (self.pending_old as usize, self.key[symbol] as usize);
            self.letter_counts[a] -= m;
            self.letter_counts[b] += m;
            self.letter_symbols[a] -= 1;
            self.letter_symbols[b] += 1;
        }
        self.score += self.pending_delta;
        self.pending_symbol = NO_PENDING;
    }

    pub fn rollback(&mut self) {
        assert_ne!(self.pending_symbol, NO_PENDING, "nothing to roll back");
        self.write_symbol(self.pending_symbol, self.pending_old);
        self.pending_symbol = NO_PENDING;
    }

    /// Unconditionally applies a move and returns the score change.
    pub fn apply(&mut self, symbol: usize, letter: u8) -> i64 {
        let delta = self.propose(symbol, letter);
        self.commit();
        delta
    }

    /// Replaces the whole key, e.g. to restore the best key seen so far.
    pub fn set_key(&mut self, key: &[u8]) {
        assert_eq!(
            self.pending_symbol, NO_PENDING,
            "previous proposal not resolved"
        );
        assert_eq!(key.len(), self.key.len());
        for (s, &l) in key.iter().enumerate() {
            assert!((l as usize) < ALPHABET, "key letters must be < 26");
            if self.key[s] != l {
                self.write_symbol(s, l);
            }
        }
        self.recount();
    }

    fn recount(&mut self) {
        self.letter_counts = [0; ALPHABET];
        self.letter_symbols = [0; ALPHABET];
        for &l in &self.plain {
            self.letter_counts[l as usize] += 1;
        }
        for s in 0..self.num_symbols() {
            if self.occurrences(s) > 0 {
                self.letter_symbols[self.key[s] as usize] += 1;
            }
        }
        self.score = self.full_rescore();
    }

    fn write_symbol(&mut self, symbol: usize, letter: u8) {
        self.key[symbol] = letter;
        let range = self.offsets[symbol] as usize..self.offsets[symbol + 1] as usize;
        for &p in &self.positions[range] {
            self.plain[p as usize] = letter;
        }
    }

    /// Sum of the scores of every window that covers at least one position of
    /// `symbol`, each window counted once even when occurrences are closer
    /// together than the model order.
    fn touched_sum(&self, symbol: usize) -> i64 {
        let n = self.order;
        let len = self.plain.len();
        if len < n {
            return 0;
        }
        let last_start = len - n;
        let table = self.model.table();
        let plain = &self.plain;
        let range = self.offsets[symbol] as usize..self.offsets[symbol + 1] as usize;

        let mut sum = 0i64;
        // `next_start` is the first window start not yet counted. While
        // `rolling`, `h` is the hash of the window starting at `next_start - 1`.
        let mut next_start = 0usize;
        let mut rolling = false;
        let mut h = 0usize;
        for &p in &self.positions[range] {
            let p = p as usize;
            let lo = p.saturating_sub(n - 1).max(next_start);
            let hi = p.min(last_start);
            if lo > hi {
                continue;
            }
            let mut start = lo;
            h = if rolling && lo == next_start {
                (h - plain[start - 1] as usize * self.high) * ALPHABET
                    + plain[start + n - 1] as usize
            } else {
                plain[start..start + n]
                    .iter()
                    .fold(0usize, |acc, &l| acc * ALPHABET + l as usize)
            };
            sum += table[h] as i64;
            while start < hi {
                start += 1;
                h = (h - plain[start - 1] as usize * self.high) * ALPHABET
                    + plain[start + n - 1] as usize;
                sum += table[h] as i64;
            }
            next_start = hi + 1;
            rolling = true;
        }
        sum
    }
}

/// `table[x] = round(SCALE * ln x!)`. Entries are rounded independently, so
/// every score built from them is an exact integer regardless of move order.
fn ln_factorial_table(max: usize) -> Vec<i64> {
    let mut acc = 0f64;
    (0..=max)
        .map(|x| {
            if x > 1 {
                acc += (x as f64).ln();
            }
            (SCALE * acc).round() as i64
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::letters_from_text;

    #[test]
    fn delta_matches_rescoring_on_overlapping_occurrences() {
        let corpus = letters_from_text(&"thequickbrownfoxjumpsoverthelazydog".repeat(20));
        let model = NgramModel::train(&corpus, 4).unwrap();
        // Symbol 0 appears at adjacent positions, so its windows overlap.
        let cipher = [0u16, 0, 1, 0, 2, 2, 0, 1, 0, 0, 0, 3];
        let mut sc = Scorer::new(&model, &cipher, vec![19, 7, 4, 16]);
        for (s, l) in [(0usize, 4u8), (2, 0), (0, 14), (3, 3), (1, 1), (1, 1)] {
            let before = sc.score();
            let d = sc.apply(s, l);
            assert_eq!(sc.score(), sc.full_rescore());
            assert_eq!(before + d, sc.score());
        }
    }

    #[test]
    fn homophone_term_is_the_dirichlet_marginal_likelihood() {
        let corpus = letters_from_text(&"abcd".repeat(50));
        let model = NgramModel::train(&corpus, 2).unwrap();
        // Symbols 0 and 1 both decrypt to A; 0 occurs 3 times, 1 once.
        // ln Γ(2) − ln Γ(4 + 2) + ln 3! + ln 1!  =  ln(6 / 120).
        let cipher = [0u16, 0, 1, 0];
        let sc = Scorer::new(&model, &cipher, vec![0, 0]);
        let expected_homophone = (6.0f64 / 120.0).ln();
        let ngrams = model.score_text(&[0, 0, 0, 0]) as f64;
        let got = (sc.score() as f64 - ngrams) / SCALE;
        assert!(
            (got - expected_homophone).abs() < 0.01,
            "{got} vs {expected_homophone}"
        );
    }

    #[test]
    fn merging_symbols_onto_one_letter_is_penalised() {
        // A model where A->A is certain: the n-gram term alone prefers "AAAA".
        let corpus = letters_from_text(&"a".repeat(100));
        let model = NgramModel::train(&corpus, 2).unwrap();
        let cipher: Vec<u16> = (0..40).map(|i| (i % 4) as u16).collect();
        let merged = Scorer::new(&model, &cipher, vec![0, 0, 0, 0]);
        let n_gram_only = model.score_text(merged.plaintext());
        assert_eq!(n_gram_only, 0);
        assert!(merged.score() < -(40.0 * 4f64.ln() * SCALE * 0.99) as i64);
    }

    #[test]
    fn rollback_restores_text_and_score() {
        let corpus = letters_from_text(&"abcdefghijklmnopqrstuvwxyz".repeat(10));
        let model = NgramModel::train(&corpus, 3).unwrap();
        let cipher = [0u16, 1, 2, 1, 0, 2, 2, 1];
        let mut sc = Scorer::new(&model, &cipher, vec![0, 1, 2]);
        let text = sc.plaintext().to_vec();
        let score = sc.score();
        let d = sc.propose(1, 9);
        assert_ne!(d, 0);
        sc.rollback();
        assert_eq!(sc.plaintext(), &text[..]);
        assert_eq!(sc.score(), score);
        assert_eq!(sc.key(), &[0, 1, 2]);
        // A later committed move must see the restored letter counts.
        sc.apply(0, 1);
        assert_eq!(sc.score(), sc.full_rescore());
    }

    #[test]
    #[should_panic(expected = "previous proposal not resolved")]
    fn unresolved_proposal_is_a_programming_error() {
        let corpus = letters_from_text("abcabcabc");
        let model = NgramModel::train(&corpus, 3).unwrap();
        let mut sc = Scorer::new(&model, &[0, 1, 0], vec![0, 1]);
        sc.propose(0, 2);
        sc.propose(1, 2);
    }
}
