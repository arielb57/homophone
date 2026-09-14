//! Small deterministic PRNG so every experiment is reproducible from a seed
//! without pulling in a runtime dependency.

/// xoshiro256** seeded through SplitMix64.
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut x = seed;
        let mut next = || {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Rng {
            s: [next(), next(), next(), next()],
        }
    }

    /// Derives an independent stream, e.g. one per restart or per trial.
    pub fn derive(seed: u64, stream: u64) -> Self {
        Rng::new(seed ^ stream.wrapping_mul(0xD6E8_FEB8_6659_FD93).rotate_left(17))
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform integer in `0..n`. `n` must be non-zero.
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "Rng::below called with n = 0");
        // Multiply-shift; the bias is below 2^-32 for any n used here.
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    /// Uniform float in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }

    /// Index sampled proportionally to non-negative `weights`.
    /// Falls back to uniform when every weight is zero.
    pub fn weighted(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return self.below(weights.len());
        }
        let mut target = self.unit() * total;
        for (i, &w) in weights.iter().enumerate() {
            if target < w {
                return i;
            }
            target -= w;
        }
        weights.iter().rposition(|&w| w > 0.0).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream_and_different_seeds_diverge() {
        let a: Vec<u64> = (0..8)
            .map({
                let mut r = Rng::new(7);
                move |_| r.next_u64()
            })
            .collect();
        let b: Vec<u64> = (0..8)
            .map({
                let mut r = Rng::new(7);
                move |_| r.next_u64()
            })
            .collect();
        let c: Vec<u64> = (0..8)
            .map({
                let mut r = Rng::new(8);
                move |_| r.next_u64()
            })
            .collect();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn below_is_in_range_and_covers_all_values() {
        let mut r = Rng::new(1);
        let mut seen = [0usize; 7];
        for _ in 0..7000 {
            seen[r.below(7)] += 1;
        }
        assert!(seen.iter().all(|&c| c > 800 && c < 1200), "{seen:?}");
    }

    #[test]
    fn weighted_respects_zero_weights() {
        let mut r = Rng::new(3);
        for _ in 0..1000 {
            let i = r.weighted(&[0.0, 2.0, 0.0, 1.0]);
            assert!(i == 1 || i == 3);
        }
    }
}
