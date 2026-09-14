//! Unicity measurement: how much ciphertext the solver needs, per homophone
//! count, before it reliably recovers the plaintext.

use crate::anneal::{solve, SolverConfig};
use crate::cipher::{letter_frequencies, recovery_rate, HomophonicKey};
use crate::model::NgramModel;
use crate::rng::Rng;
use std::fmt::Write;

#[derive(Clone, Debug)]
pub struct UnicityConfig {
    /// Ciphertext lengths to test, in letters.
    pub lengths: Vec<usize>,
    /// Total symbol counts to test (26 = plain substitution).
    pub symbol_counts: Vec<usize>,
    pub max_homophones: usize,
    pub trials: usize,
    pub seed: u64,
    pub solver: SolverConfig,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UnicityRow {
    pub symbols: usize,
    pub length: usize,
    pub trials: usize,
    pub mean_recovery: f64,
    pub min_recovery: f64,
}

/// Measures recovery for every (symbol count, length) pair.
///
/// Trial `t` for a symbol count uses one plaintext, one key and one encryption
/// at the longest length, and every shorter length takes a prefix of it. The
/// lengths are therefore nested, which removes sampling noise between rows:
/// a longer row sees the same cipher plus more text, not a different cipher.
pub fn run(
    model: &NgramModel,
    config: &UnicityConfig,
    mut sample_plaintext: impl FnMut(usize, &mut Rng) -> Vec<u8>,
) -> Result<Vec<UnicityRow>, String> {
    if config.lengths.is_empty() || config.symbol_counts.is_empty() || config.trials == 0 {
        return Err("unicity needs at least one length, one symbol count and one trial".into());
    }
    let mut lengths = config.lengths.clone();
    lengths.sort_unstable();
    lengths.dedup();
    let longest = *lengths.last().expect("non-empty");

    let mut rows = Vec::new();
    for &symbols in &config.symbol_counts {
        let mut per_length = vec![Vec::with_capacity(config.trials); lengths.len()];
        for trial in 0..config.trials {
            let stream = (symbols as u64) << 32 | trial as u64;
            let mut rng = Rng::derive(config.seed, stream);
            let plain = sample_plaintext(longest, &mut rng);
            if plain.len() < longest {
                return Err(format!(
                    "sampler returned {} letters, wanted {longest}",
                    plain.len()
                ));
            }
            let freqs = letter_frequencies(&plain);
            let key = HomophonicKey::random(symbols, config.max_homophones, &freqs, &mut rng)?;
            let cipher = key.encrypt(&plain, &mut rng);
            for (i, &len) in lengths.iter().enumerate() {
                let solver = SolverConfig {
                    seed: config.seed ^ stream.rotate_left(8) ^ len as u64,
                    ..config.solver.clone()
                };
                let solution = solve(model, &cipher[..len], symbols, &solver);
                per_length[i].push(recovery_rate(&plain[..len], &solution.plaintext));
            }
        }
        for (i, &length) in lengths.iter().enumerate() {
            let r = &per_length[i];
            rows.push(UnicityRow {
                symbols,
                length,
                trials: r.len(),
                mean_recovery: r.iter().sum::<f64>() / r.len() as f64,
                min_recovery: r.iter().copied().fold(f64::INFINITY, f64::min),
            });
        }
    }
    Ok(rows)
}

/// Shortest measured length from which mean recovery stays at or above
/// `target` for every longer measured length. `None` if it never gets there.
pub fn threshold_length(rows: &[UnicityRow], symbols: usize, target: f64) -> Option<usize> {
    let mut curve: Vec<&UnicityRow> = rows.iter().filter(|r| r.symbols == symbols).collect();
    curve.sort_by_key(|r| r.length);
    let mut answer = None;
    for row in curve.iter().rev() {
        if row.mean_recovery >= target {
            answer = Some(row.length);
        } else {
            break;
        }
    }
    answer
}

pub fn to_csv(rows: &[UnicityRow]) -> String {
    let mut out = String::from("symbols,length,trials,mean_recovery,min_recovery\n");
    for r in rows {
        writeln!(
            out,
            "{},{},{},{:.4},{:.4}",
            r.symbols, r.length, r.trials, r.mean_recovery, r.min_recovery
        )
        .expect("writing to a String cannot fail");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(symbols: usize, length: usize, mean: f64) -> UnicityRow {
        UnicityRow {
            symbols,
            length,
            trials: 1,
            mean_recovery: mean,
            min_recovery: mean,
        }
    }

    #[test]
    fn threshold_requires_staying_above_target() {
        let rows = vec![
            row(60, 100, 0.3),
            row(60, 200, 0.92),
            row(60, 300, 0.85),
            row(60, 400, 0.95),
            row(60, 500, 0.99),
            row(26, 100, 0.95),
        ];
        assert_eq!(threshold_length(&rows, 60, 0.9), Some(400));
        assert_eq!(threshold_length(&rows, 26, 0.9), Some(100));
        assert_eq!(threshold_length(&rows, 60, 0.999), None);
        assert_eq!(threshold_length(&rows, 80, 0.9), None);
    }

    #[test]
    fn csv_has_header_and_one_line_per_row() {
        let csv = to_csv(&[row(40, 250, 0.5), row(40, 500, 0.875)]);
        assert_eq!(
            csv,
            "symbols,length,trials,mean_recovery,min_recovery\n\
             40,250,1,0.5000,0.5000\n40,500,1,0.8750,0.8750\n"
        );
    }
}
