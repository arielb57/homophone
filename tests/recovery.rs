mod common;

use common::{case, fixture, random_symbol_count};
use homophone::{recovery_rate, solve, SolverConfig};

#[test]
fn homophonic_1500_symbols_recovers_95_percent_on_20_seeds() {
    let f = fixture();
    let mut failures = Vec::new();
    let mut total = 0.0;
    for seed in 1..=20u64 {
        let symbols = random_symbol_count(seed);
        let c = case(seed, 1500, symbols);
        let config = SolverConfig {
            seed,
            ..SolverConfig::default()
        };
        let solution = solve(&f.model, &c.ciphertext, symbols, &config);
        let r = recovery_rate(&c.plaintext, &solution.plaintext);
        println!("seed {seed:2} symbols {symbols} recovery {:.4}", r);
        total += r;
        if r < 0.95 {
            failures.push((seed, symbols, r));
        }
    }
    println!("mean recovery {:.4}", total / 20.0);
    assert!(
        failures.is_empty(),
        "seeds below 95% recovery: {failures:?}"
    );
}
