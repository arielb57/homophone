//! The unicity curve: recovery must rise with ciphertext length, and the
//! reported 90% threshold must be consistent with the measured rows.

mod common;

use common::fixture;
use homophone::unicity::{self, UnicityConfig};
use homophone::SolverConfig;

#[test]
fn recovery_is_monotone_in_length_and_crosses_ninety_percent() {
    let f = fixture();
    let config = UnicityConfig {
        lengths: vec![20, 60, 150, 300, 600, 1200],
        symbol_counts: vec![26, 60],
        max_homophones: 4,
        trials: 6,
        seed: 17,
        solver: SolverConfig {
            restarts: 4,
            iterations: 200_000,
            ..SolverConfig::default()
        },
    };
    let rows = unicity::run(&f.model, &config, |len, rng| f.language.sample(len, rng)).unwrap();
    print!("{}", unicity::to_csv(&rows));
    assert_eq!(rows.len(), 12);

    for &symbols in &config.symbol_counts {
        let curve: Vec<_> = rows.iter().filter(|r| r.symbols == symbols).collect();
        for pair in curve.windows(2) {
            assert!(
                pair[1].mean_recovery >= pair[0].mean_recovery - 0.03,
                "{symbols} symbols: recovery fell from {:.3} at {} to {:.3} at {}",
                pair[0].mean_recovery,
                pair[0].length,
                pair[1].mean_recovery,
                pair[1].length
            );
        }
        // Not a vacuous curve: short ciphertexts fail, long ones succeed.
        assert!(
            curve[0].mean_recovery < 0.6,
            "{symbols}: 20 letters should not be solvable"
        );
        assert!(curve.last().unwrap().mean_recovery > 0.95);
        let threshold = unicity::threshold_length(&rows, symbols, 0.9).expect("crosses 90%");
        assert!(curve
            .iter()
            .filter(|r| r.length >= threshold)
            .all(|r| r.mean_recovery >= 0.9));
    }

    // More homophones need at least as much ciphertext.
    let t26 = unicity::threshold_length(&rows, 26, 0.9).unwrap();
    let t60 = unicity::threshold_length(&rows, 60, 0.9).unwrap();
    assert!(
        t60 >= t26,
        "60 symbols solved from {t60} but 26 symbols only from {t26}"
    );
}

#[test]
fn invalid_configurations_are_errors() {
    let f = fixture();
    let base = UnicityConfig {
        lengths: vec![100],
        symbol_counts: vec![40],
        max_homophones: 4,
        trials: 1,
        seed: 1,
        solver: SolverConfig {
            restarts: 1,
            iterations: 1000,
            ..SolverConfig::default()
        },
    };
    let sample = |len: usize, rng: &mut homophone::Rng| f.language.sample(len, rng);
    let no_trials = UnicityConfig {
        trials: 0,
        ..base.clone()
    };
    assert!(unicity::run(&f.model, &no_trials, sample).is_err());
    let too_many = UnicityConfig {
        symbol_counts: vec![200],
        ..base.clone()
    };
    assert!(unicity::run(&f.model, &too_many, sample).is_err());
    let short_sampler = |_: usize, _: &mut homophone::Rng| vec![0u8; 10];
    assert!(unicity::run(&f.model, &base, short_sampler).is_err());
}
