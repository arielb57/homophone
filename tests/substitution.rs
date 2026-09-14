//! A one-to-one substitution cipher is the special case of 26 symbols.

mod common;

use common::{case, fixture};
use homophone::{recovery_rate, solve, SolverConfig};

#[test]
fn plain_substitution_is_solved_and_key_is_a_permutation() {
    let f = fixture();
    for seed in 101..=105u64 {
        let c = case(seed, 1500, 26);
        assert!(c.key.homophones.iter().all(|h| h.len() == 1));
        let config = SolverConfig {
            seed,
            ..SolverConfig::default()
        };
        let solution = solve(&f.model, &c.ciphertext, 26, &config);
        let r = recovery_rate(&c.plaintext, &solution.plaintext);
        println!("seed {seed}: recovery {r:.4}");
        assert!(r >= 0.98, "seed {seed}: recovery {r}");

        // Every symbol that occurs often enough to be identifiable must be
        // mapped to its true letter, and no two such symbols share a letter.
        let mut used = [false; 26];
        for s in 0..26u16 {
            let count = c.ciphertext.iter().filter(|&&x| x == s).count();
            if count >= 10 {
                let letter = solution.key[s as usize];
                assert_eq!(
                    letter, c.key.symbol_letter[s as usize],
                    "seed {seed} symbol {s}"
                );
                assert!(
                    !used[letter as usize],
                    "seed {seed}: letter {letter} reused"
                );
                used[letter as usize] = true;
            }
        }
    }
}

#[test]
fn restarts_are_deterministic_regardless_of_thread_count() {
    let f = fixture();
    let c = case(7, 600, 50);
    let base = SolverConfig {
        restarts: 4,
        iterations: 150_000,
        seed: 3,
        ..SolverConfig::default()
    };
    let one = solve(
        &f.model,
        &c.ciphertext,
        50,
        &SolverConfig {
            threads: 1,
            ..base.clone()
        },
    );
    let many = solve(
        &f.model,
        &c.ciphertext,
        50,
        &SolverConfig { threads: 4, ..base },
    );
    assert_eq!(one.key, many.key);
    assert_eq!(one.restart_scores, many.restart_scores);
    assert_eq!(one.score, *one.restart_scores.iter().max().unwrap());
}

#[test]
fn solver_prefers_the_true_key_over_random_keys() {
    // The objective itself must rank the truth above what search returns on
    // garbage: a solved cipher's score is at least the true key's score.
    let f = fixture();
    let c = case(21, 1500, 60);
    let truth =
        homophone::Scorer::new(&f.model, &c.ciphertext, c.key.symbol_letter.clone()).score();
    let solution = solve(
        &f.model,
        &c.ciphertext,
        60,
        &SolverConfig {
            seed: 21,
            ..SolverConfig::default()
        },
    );
    assert!(
        solution.score >= truth,
        "search stopped below the true key: {} < {truth}",
        solution.score
    );
    let few = solve(
        &f.model,
        &c.ciphertext,
        60,
        &SolverConfig {
            restarts: 1,
            iterations: 1,
            seed: 21,
            ..SolverConfig::default()
        },
    );
    assert!(
        few.score < truth,
        "a one-move search should not reach the true key's score"
    );
}
