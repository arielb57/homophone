//! Property test: after any sequence of random moves, the incrementally
//! maintained score equals a from-scratch rescoring, exactly.

mod common;

use homophone::{MarkovLanguage, NgramModel, Rng, Scorer};

fn check_random_moves(
    model: &NgramModel,
    cipher: &[u16],
    num_symbols: usize,
    moves: usize,
    seed: u64,
) {
    let mut rng = Rng::new(seed);
    let key = (0..num_symbols).map(|_| rng.below(26) as u8).collect();
    let mut sc = Scorer::new(model, cipher, key);
    assert_eq!(sc.score(), sc.full_rescore());
    for i in 0..moves {
        let symbol = rng.below(num_symbols);
        // Letter may equal the current one: a no-op move must also be exact.
        let letter = rng.below(26) as u8;
        let before = sc.score();
        let before_text = sc.plaintext().to_vec();
        let delta = sc.propose(symbol, letter);
        if rng.below(3) == 0 {
            sc.rollback();
            assert_eq!(sc.score(), before, "rollback changed score at move {i}");
            assert_eq!(
                sc.plaintext(),
                &before_text[..],
                "rollback changed text at move {i}"
            );
        } else {
            sc.commit();
            assert_eq!(sc.score(), before + delta);
            assert_eq!(sc.key()[symbol], letter);
        }
        assert_eq!(
            sc.score(),
            sc.full_rescore(),
            "incremental score drifted at move {i} (order {}, len {}, symbols {num_symbols})",
            model.order(),
            cipher.len()
        );
    }
    let expected_text: Vec<u8> = cipher.iter().map(|&s| sc.key()[s as usize]).collect();
    assert_eq!(sc.plaintext(), &expected_text[..]);
}

#[test]
fn ten_thousand_moves_stay_exact_for_every_order() {
    let lang = MarkovLanguage::new(3, 1.5, 77);
    let corpus = lang.sample(200_000, &mut Rng::new(1));
    for order in 2..=5 {
        let model = NgramModel::train(&corpus, order).unwrap();
        let mut rng = Rng::new(order as u64);
        // Realistic homophonic shape: 1500 letters over 60 symbols.
        let cipher: Vec<u16> = (0..1500).map(|_| rng.below(60) as u16).collect();
        check_random_moves(&model, &cipher, 60, 10_000, 100 + order as u64);
    }
}

#[test]
fn dense_repeats_where_windows_overlap_many_times() {
    // Five symbols over 400 positions: a symbol often occurs several times
    // inside one window, which is where double counting would show up.
    let lang = MarkovLanguage::new(3, 1.5, 5);
    let corpus = lang.sample(100_000, &mut Rng::new(2));
    let model = NgramModel::train(&corpus, 5).unwrap();
    let mut rng = Rng::new(3);
    let mut cipher: Vec<u16> = (0..400).map(|_| rng.below(5) as u16).collect();
    // A long run of a single symbol.
    cipher[100..130].fill(2);
    check_random_moves(&model, &cipher, 5, 10_000, 4);
}

#[test]
fn symbols_absent_from_the_ciphertext_do_not_disturb_the_score() {
    let f = common::fixture();
    let mut rng = Rng::new(8);
    // Only even symbols occur; odd ones exist in the key but never in the text.
    let cipher: Vec<u16> = (0..700).map(|_| (rng.below(40) * 2) as u16).collect();
    check_random_moves(&f.model, &cipher, 80, 10_000, 9);
}
