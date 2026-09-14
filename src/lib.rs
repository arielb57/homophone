//! Homophonic-substitution cipher solver.
//!
//! * [`model::NgramModel`] — n-gram log-probability tables (order 2–5).
//! * [`scorer::Scorer`] — incremental rescoring of only the windows a move touches.
//! * [`anneal::solve`] — simulated annealing with parallel restarts.
//! * [`unicity`] — recovery rate against ciphertext length and homophone count.
//! * [`synth::MarkovLanguage`] — a synthetic language for self-contained evaluation.

pub mod anneal;
pub mod cipher;
pub mod model;
pub mod rng;
pub mod scorer;
pub mod synth;
pub mod unicity;

pub use anneal::{solve, Solution, SolverConfig};
pub use cipher::{recovery_rate, Ciphertext, HomophonicKey};
pub use model::{letters_from_text, letters_to_string, NgramModel};
pub use rng::Rng;
pub use scorer::Scorer;
pub use synth::MarkovLanguage;
