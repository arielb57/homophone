//! Simulated annealing over symbol-to-letter assignments, with independent
//! restarts run in parallel on std threads.

use crate::model::{NgramModel, ALPHABET};
use crate::rng::Rng;
use crate::scorer::Scorer;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Starting temperature as a multiple of the mean score loss of a random move
/// from a random key. Tuned on the synthetic language; see the README.
pub const START_TEMPERATURE_FACTOR: f64 = 0.35;
const CALIBRATION_MOVES: usize = 400;

/// Best key of one restart and its score.
type RestartResult = (Vec<u8>, i64);

#[derive(Clone, Debug)]
pub struct SolverConfig {
    /// Independent annealing runs; the best-scoring one wins.
    pub restarts: usize,
    /// Moves per restart. `0` picks a budget from the ciphertext size.
    pub iterations: u64,
    /// Worker threads. `0` uses all available cores.
    pub threads: usize,
    pub seed: u64,
    /// Starting temperature as a multiple of the calibrated mean move loss.
    pub temperature: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig {
            restarts: 8,
            iterations: 0,
            threads: 0,
            seed: 1,
            temperature: START_TEMPERATURE_FACTOR,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Solution {
    /// Letter for each symbol. Symbols that never occur keep an arbitrary letter.
    pub key: Vec<u8>,
    pub plaintext: Vec<u8>,
    /// Sum of window log-probabilities, in millinats.
    pub score: i64,
    /// Index of the restart that produced this solution.
    pub best_restart: usize,
    pub restart_scores: Vec<i64>,
    pub iterations_per_restart: u64,
    pub elapsed: Duration,
}

/// Default move budget: enough sweeps over (symbol, letter) pairs that every
/// assignment is tried many times while the temperature falls.
pub fn auto_iterations(num_symbols: usize) -> u64 {
    (num_symbols.max(1) as u64 * ALPHABET as u64 * 800).clamp(200_000, 20_000_000)
}

/// Mean score loss of a random worsening move from the current key, used to
/// set the starting temperature. A move's score change scales with how often
/// the symbol occurs (length / symbols) times the model order, so measuring
/// real moves adapts the schedule to the length and homophone count without a
/// hand-tuned constant per cipher.
pub fn mean_move_loss(scorer: &mut Scorer, active: &[usize], rng: &mut Rng) -> f64 {
    let mut loss = 0f64;
    let mut losses = 0usize;
    for _ in 0..CALIBRATION_MOVES {
        let symbol = active[rng.below(active.len())];
        let letter = random_other_letter(scorer.key()[symbol], rng);
        let d = scorer.propose(symbol, letter);
        scorer.rollback();
        if d < 0 {
            loss += -d as f64;
            losses += 1;
        }
    }
    if losses == 0 {
        return 1.0;
    }
    (loss / losses as f64).max(1.0)
}

fn random_other_letter(current: u8, rng: &mut Rng) -> u8 {
    let l = rng.below(ALPHABET - 1) as u8;
    if l >= current {
        l + 1
    } else {
        l
    }
}

/// One annealing run from a random key. Returns the best key found and its score.
pub fn anneal(
    model: &NgramModel,
    ciphertext: &[u16],
    num_symbols: usize,
    iterations: u64,
    temperature: f64,
    rng: &mut Rng,
) -> (Vec<u8>, i64) {
    let key: Vec<u8> = (0..num_symbols)
        .map(|_| rng.below(ALPHABET) as u8)
        .collect();
    let mut scorer = Scorer::new(model, ciphertext, key);
    let active: Vec<usize> = (0..num_symbols)
        .filter(|&s| scorer.occurrences(s) > 0)
        .collect();
    if active.is_empty() || ciphertext.len() < model.order() {
        let score = scorer.score();
        return (scorer.key().to_vec(), score);
    }

    let t0 = mean_move_loss(&mut scorer, &active, rng) * temperature;
    let mut best_key = scorer.key().to_vec();
    let mut best_score = scorer.score();
    for i in 0..iterations {
        // Linear cooling to zero: the tail of the run is effectively greedy.
        let t = t0 * (1.0 - i as f64 / iterations as f64);
        let symbol = active[rng.below(active.len())];
        let letter = random_other_letter(scorer.key()[symbol], rng);
        let d = scorer.propose(symbol, letter);
        if d >= 0 || rng.unit() < (d as f64 / t).exp() {
            scorer.commit();
            if scorer.score() > best_score {
                best_score = scorer.score();
                best_key.copy_from_slice(scorer.key());
            }
        } else {
            scorer.rollback();
        }
    }

    scorer.set_key(&best_key);
    hill_climb(&mut scorer, &active);
    (scorer.key().to_vec(), scorer.score())
}

/// Steepest-ascent polish: for each symbol take its best letter, until no
/// single reassignment improves the score.
pub fn hill_climb(scorer: &mut Scorer, active: &[usize]) {
    loop {
        let mut improved = false;
        for &symbol in active {
            let current = scorer.key()[symbol];
            let mut best = (0i64, current);
            for letter in 0..ALPHABET as u8 {
                if letter == current {
                    continue;
                }
                let d = scorer.propose(symbol, letter);
                scorer.rollback();
                if d > best.0 {
                    best = (d, letter);
                }
            }
            if best.1 != current {
                scorer.apply(symbol, best.1);
                improved = true;
            }
        }
        if !improved {
            return;
        }
    }
}

/// Runs `config.restarts` annealing runs in parallel and keeps the best.
/// Each restart has its own seed stream, so the result does not depend on the
/// thread count.
pub fn solve(
    model: &NgramModel,
    ciphertext: &[u16],
    num_symbols: usize,
    config: &SolverConfig,
) -> Solution {
    let start = Instant::now();
    let restarts = config.restarts.max(1);
    let iterations = if config.iterations == 0 {
        auto_iterations(num_symbols)
    } else {
        config.iterations
    };
    let threads = if config.threads == 0 {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    } else {
        config.threads
    }
    .clamp(1, restarts);

    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<RestartResult>>> = Mutex::new(vec![None; restarts]);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let r = next.fetch_add(1, Ordering::Relaxed);
                if r >= restarts {
                    break;
                }
                let mut rng = Rng::derive(config.seed, r as u64);
                let out = anneal(
                    model,
                    ciphertext,
                    num_symbols,
                    iterations,
                    config.temperature,
                    &mut rng,
                );
                results
                    .lock()
                    .expect("no worker panics while holding the lock")[r] = Some(out);
            });
        }
    });

    let results: Vec<(Vec<u8>, i64)> = results
        .into_inner()
        .expect("workers finished")
        .into_iter()
        .map(|r| r.expect("every restart ran"))
        .collect();
    let restart_scores: Vec<i64> = results.iter().map(|r| r.1).collect();
    let best_restart = (0..restarts)
        .max_by_key(|&r| (restart_scores[r], std::cmp::Reverse(r)))
        .expect("at least one restart");
    let key = results[best_restart].0.clone();
    let plaintext = ciphertext.iter().map(|&s| key[s as usize]).collect();
    Solution {
        key,
        plaintext,
        score: restart_scores[best_restart],
        best_restart,
        restart_scores,
        iterations_per_restart: iterations,
        elapsed: start.elapsed(),
    }
}
