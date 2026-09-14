//! Synthetic language: a random Markov chain over 26 letters whose transition
//! distributions are Zipf-shaped. It gives the solver a language with known
//! structure, so recovery rates can be measured without any external corpus.

use crate::model::{ALPHABET, MAX_ORDER, MIN_ORDER};
use crate::rng::Rng;

#[derive(Clone, Debug)]
pub struct MarkovLanguage {
    order: usize,
    /// Cumulative transition probabilities, `ALPHABET` entries per context.
    cumulative: Vec<f64>,
}

impl MarkovLanguage {
    /// `order` counts letters in the n-gram sense: order 3 means each letter
    /// depends on the previous two, which an order-3 model captures exactly.
    ///
    /// Each context ranks the 26 letters and gives rank `r` (1-based)
    /// probability proportional to `r^-exponent`. The ranking of a context is
    /// a noisy copy of the ranking of its one-letter-shorter suffix, down to a
    /// random global ranking. A first version drew every context's ranking
    /// independently; that language has flat unigram and bigram statistics, so
    /// a partly correct key scores no better than a random one and annealing
    /// has nothing to climb. Natural languages are not like that, and neither
    /// is this one.
    pub fn new(order: usize, exponent: f64, seed: u64) -> Self {
        assert!(
            (MIN_ORDER..=MAX_ORDER).contains(&order),
            "order must be between {MIN_ORDER} and {MAX_ORDER}"
        );
        assert!(exponent >= 0.0, "Zipf exponent must be non-negative");
        let mut rng = Rng::new(seed);
        let mut global: Vec<u8> = (0..ALPHABET as u8).collect();
        rng.shuffle(&mut global);
        // rankings[j] holds one ranking per context of length j.
        let mut level: Vec<[u8; ALPHABET]> = vec![global.try_into().expect("26 letters")];
        for j in 1..order {
            let parents = ALPHABET.pow(j as u32 - 1);
            let next: Vec<[u8; ALPHABET]> = (0..ALPHABET.pow(j as u32))
                .map(|ctx| perturb(&level[ctx % parents], &mut rng))
                .collect();
            level = next;
        }

        let zipf: Vec<f64> = (1..=ALPHABET).map(|r| (r as f64).powf(-exponent)).collect();
        let norm: f64 = zipf.iter().sum();
        let mut cumulative = Vec::with_capacity(level.len() * ALPHABET);
        let mut probs = [0f64; ALPHABET];
        for ranking in &level {
            for (rank, &letter) in ranking.iter().enumerate() {
                probs[letter as usize] = zipf[rank] / norm;
            }
            let mut acc = 0.0;
            for p in probs {
                acc += p;
                cumulative.push(acc);
            }
        }
        MarkovLanguage { order, cumulative }
    }

    pub fn order(&self) -> usize {
        self.order
    }

    /// Probability of `letter` following `context` (`order - 1` letters).
    pub fn transition(&self, context: &[u8], letter: u8) -> f64 {
        assert_eq!(context.len(), self.order - 1);
        let c = context
            .iter()
            .fold(0usize, |h, &l| h * ALPHABET + l as usize);
        let row = &self.cumulative[c * ALPHABET..(c + 1) * ALPHABET];
        let l = letter as usize;
        row[l] - if l == 0 { 0.0 } else { row[l - 1] }
    }

    /// Samples `len` letters, discarding a burn-in so the output starts near
    /// the chain's stationary distribution.
    pub fn sample(&self, len: usize, rng: &mut Rng) -> Vec<u8> {
        let k = self.order - 1;
        let burn_in = 256;
        let mut out: Vec<u8> = (0..k).map(|_| rng.below(ALPHABET) as u8).collect();
        out.reserve(len + burn_in);
        let high = ALPHABET.pow(k as u32 - 1);
        let mut ctx = out.iter().fold(0usize, |h, &l| h * ALPHABET + l as usize);
        while out.len() < len + burn_in + k {
            let row = &self.cumulative[ctx * ALPHABET..(ctx + 1) * ALPHABET];
            let u = rng.unit() * row[ALPHABET - 1];
            let letter = row.iter().position(|&c| u < c).unwrap_or(ALPHABET - 1);
            out.push(letter as u8);
            ctx = (ctx - out[out.len() - 1 - k] as usize * high) * ALPHABET + letter;
        }
        out.split_off(burn_in + k)
    }
}

/// How far, in ranks, a letter can drift between a context and its extension.
const RANK_NOISE: f64 = 14.0;

/// Re-sorts letters by their parent rank plus uniform noise in `[0, RANK_NOISE)`.
fn perturb(parent: &[u8; ALPHABET], rng: &mut Rng) -> [u8; ALPHABET] {
    let mut keyed: Vec<(f64, u8)> = parent
        .iter()
        .enumerate()
        .map(|(rank, &letter)| (rank as f64 + rng.unit() * RANK_NOISE, letter))
        .collect();
    keyed.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out = [0u8; ALPHABET];
    for (i, (_, letter)) in keyed.into_iter().enumerate() {
        out[i] = letter;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_distributions_with_zipf_shape() {
        let lang = MarkovLanguage::new(3, 1.2, 9);
        let ctx = [4u8, 17];
        let mut probs: Vec<f64> = (0..26).map(|l| lang.transition(&ctx, l)).collect();
        assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        probs.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let ratio = probs[0] / probs[1];
        assert!(
            (ratio - 2f64.powf(1.2)).abs() < 1e-9,
            "rank1/rank2 = {ratio}"
        );
    }

    #[test]
    fn sampled_bigram_frequencies_match_transitions() {
        let lang = MarkovLanguage::new(2, 1.0, 4);
        let text = lang.sample(400_000, &mut Rng::new(1));
        let mut from_a = [0usize; 26];
        for w in text.windows(2).filter(|w| w[0] == 0) {
            from_a[w[1] as usize] += 1;
        }
        let total: usize = from_a.iter().sum();
        for l in 0..26u8 {
            let expected = lang.transition(&[0], l);
            let observed = from_a[l as usize] as f64 / total as f64;
            assert!(
                (observed - expected).abs() < 0.02,
                "letter {l}: {observed} vs {expected}"
            );
        }
    }

    #[test]
    fn same_seed_gives_same_language() {
        let a = MarkovLanguage::new(3, 1.0, 42).sample(100, &mut Rng::new(1));
        let b = MarkovLanguage::new(3, 1.0, 42).sample(100, &mut Rng::new(1));
        let c = MarkovLanguage::new(3, 1.0, 43).sample(100, &mut Rng::new(1));
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
