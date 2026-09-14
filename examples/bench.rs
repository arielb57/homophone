//! Benchmark: incremental versus full rescoring throughput, and the move budget
//! and wall time needed for 95% recovery on the 1500-symbol synthetic set.
//!
//! Run with: cargo run --release --example bench

use homophone::cipher::letter_frequencies;
use homophone::{
    recovery_rate, solve, HomophonicKey, MarkovLanguage, NgramModel, Rng, Scorer, SolverConfig,
};
use std::time::Instant;

const LANGUAGE_SEED: u64 = 2026;
const SEEDS: u64 = 20;

fn random_case(
    lang: &MarkovLanguage,
    seed: u64,
    len: usize,
    symbols: usize,
) -> (Vec<u8>, Vec<u16>) {
    let mut rng = Rng::new(seed);
    let plain = lang.sample(len, &mut rng);
    let key = HomophonicKey::random(symbols, 4, &letter_frequencies(&plain), &mut rng).unwrap();
    let cipher = key.encrypt(&plain, &mut rng);
    (plain, cipher)
}

/// Moves per second when each move is scored by the incremental scorer.
fn incremental_rate(model: &NgramModel, cipher: &[u16], symbols: usize, moves: u64) -> f64 {
    let mut rng = Rng::new(9);
    let key = (0..symbols).map(|_| rng.below(26) as u8).collect();
    let mut sc = Scorer::new(model, cipher, key);
    let start = Instant::now();
    let mut sink = 0i64;
    for i in 0..moves {
        let s = rng.below(symbols);
        let d = sc.propose(s, rng.below(26) as u8);
        sink = sink.wrapping_add(d);
        if i % 2 == 0 {
            sc.commit()
        } else {
            sc.rollback()
        }
    }
    std::hint::black_box(sink);
    moves as f64 / start.elapsed().as_secs_f64()
}

/// Moves per second when each move rewrites the key and rescores the whole text.
fn full_rate(model: &NgramModel, cipher: &[u16], symbols: usize, moves: u64) -> f64 {
    let mut rng = Rng::new(9);
    let mut key: Vec<u8> = (0..symbols).map(|_| rng.below(26) as u8).collect();
    let mut plain: Vec<u8> = cipher.iter().map(|&s| key[s as usize]).collect();
    let mut score = model.score_text(&plain);
    let start = Instant::now();
    for i in 0..moves {
        let s = rng.below(symbols);
        let old = key[s];
        key[s] = rng.below(26) as u8;
        for (p, &c) in cipher.iter().enumerate() {
            if c as usize == s {
                plain[p] = key[s];
            }
        }
        let new_score = model.score_text(&plain);
        if i % 2 == 0 {
            score = new_score;
        } else {
            key[s] = old;
            for (p, &c) in cipher.iter().enumerate() {
                if c as usize == s {
                    plain[p] = old;
                }
            }
        }
    }
    std::hint::black_box(score);
    moves as f64 / start.elapsed().as_secs_f64()
}

fn main() {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("threads available: {threads}");
    let lang = MarkovLanguage::new(3, 1.5, LANGUAGE_SEED);
    let corpus = lang.sample(1_000_000, &mut Rng::new(u64::MAX));

    println!("\n## Rescoring throughput (60 symbols, single thread)\n");
    println!("| order | length | incremental moves/s | full rescoring moves/s | speedup |");
    println!("|---|---|---|---|---|");
    for order in [3usize, 4, 5] {
        let model = NgramModel::train(&corpus, order).unwrap();
        for len in [500usize, 2000, 10000] {
            let (_, cipher) = random_case(&lang, 7, len, 60);
            let inc = incremental_rate(&model, &cipher, 60, 2_000_000);
            let full_moves = (40_000_000 / len as u64).max(2_000);
            let full = full_rate(&model, &cipher, 60, full_moves);
            println!(
                "| {order} | {len} | {:.2}M | {:.3}M | {:.0}x |",
                inc / 1e6,
                full / 1e6,
                inc / full
            );
        }
    }

    println!(
        "\n## Move budget for 95% recovery (1500-letter ciphertexts, 40-80 distinct symbols, \
         {SEEDS} seeds, 8 restarts on {threads} threads)\n"
    );
    let model = NgramModel::train(&corpus, 3).unwrap();
    let cases: Vec<_> = (1..=SEEDS)
        .map(|seed| {
            let symbols = 40 + Rng::derive(seed, 0xC0FFEE).below(41);
            let (plain, cipher) = random_case(&lang, seed, 1500, symbols);
            (seed, symbols, plain, cipher)
        })
        .collect();
    println!("| moves/restart | seeds >= 95% | mean recovery | mean solve time | max solve time |");
    println!("|---|---|---|---|---|");
    // Double the budget until every seed reaches 95%, then report the CLI
    // default budget (`iterations: 0`) for comparison.
    let mut budget = 1_000u64;
    let mut reached = false;
    loop {
        if reached {
            budget = 0;
        }
        let mut ok = 0;
        let mut rec = 0.0;
        let mut total_time = 0.0f64;
        let mut max_time = 0.0f64;
        for (seed, symbols, plain, cipher) in &cases {
            let config = SolverConfig {
                iterations: budget,
                seed: *seed,
                ..SolverConfig::default()
            };
            let solution = solve(&model, cipher, *symbols, &config);
            let r = recovery_rate(plain, &solution.plaintext);
            let t = solution.elapsed.as_secs_f64();
            rec += r;
            total_time += t;
            max_time = max_time.max(t);
            if r >= 0.95 {
                ok += 1;
            }
        }
        let label = if budget == 0 {
            "default".to_string()
        } else {
            budget.to_string()
        };
        println!(
            "| {label} | {ok}/{SEEDS} | {:.4} | {:.3}s | {:.3}s |",
            rec / SEEDS as f64,
            total_time / SEEDS as f64,
            max_time
        );
        if budget == 0 || budget >= 3_200_000 {
            break;
        }
        reached = ok == SEEDS;
        if !reached {
            budget *= 2;
        }
    }
}
