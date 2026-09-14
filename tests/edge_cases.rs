mod common;

use common::{case, fixture};
use homophone::{letters_from_text, recovery_rate, solve, NgramModel, Scorer, SolverConfig};

#[test]
fn symbols_that_occur_once_are_scored_exactly_and_do_not_hurt_recovery() {
    let f = fixture();
    let mut c = case(31, 1500, 50);
    // Append ten fresh symbols, each used exactly once, in place of the
    // original symbol at spread-out positions.
    for i in 0..10u16 {
        let p = 100 + i as usize * 130;
        c.ciphertext[p] = 50 + i;
    }
    let mut key = c.key.symbol_letter.clone();
    for i in 0..10usize {
        key.push(c.plaintext[100 + i * 130]);
    }
    let mut sc = Scorer::new(&f.model, &c.ciphertext, key);
    for s in 50..60 {
        assert_eq!(sc.occurrences(s), 1);
        for l in 0..26 {
            sc.apply(s, l);
            assert_eq!(sc.score(), sc.full_rescore());
        }
    }

    let solution = solve(
        &f.model,
        &c.ciphertext,
        60,
        &SolverConfig {
            seed: 31,
            ..SolverConfig::default()
        },
    );
    let r = recovery_rate(&c.plaintext, &solution.plaintext);
    assert!(r >= 0.95, "singletons broke recovery: {r}");
}

#[test]
fn ciphertext_shorter_than_the_model_order() {
    let f = fixture();
    let model_order = f.model.order();
    for len in 0..model_order {
        let cipher: Vec<u16> = (0..len as u16).collect();
        let mut sc = Scorer::new(&f.model, &cipher, vec![0; 4]);
        // No n-gram windows: only the homophone term can contribute.
        assert_eq!(f.model.score_text(sc.plaintext()), 0);
        for l in 0..26 {
            if len > 0 {
                sc.apply(0, l);
            }
            assert_eq!(sc.score(), sc.full_rescore());
        }
        let solution = solve(
            &f.model,
            &cipher,
            4,
            &SolverConfig {
                restarts: 2,
                ..SolverConfig::default()
            },
        );
        assert_eq!(solution.plaintext.len(), len);
        assert_eq!(solution.key.len(), 4);
    }
}

#[test]
fn unseen_ngrams_get_a_finite_floor_below_every_seen_ngram() {
    let corpus = letters_from_text("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG");
    let model = NgramModel::train(&corpus, 3).unwrap();
    let floor = model.floor();
    assert!(floor < 0, "floor must be a penalty");
    assert!(floor > i16::MIN, "floor must be finite, not saturated");

    let seen: Vec<i16> = corpus.windows(3).map(|w| model.ngram_score(w)).collect();
    assert!(seen.iter().all(|&v| v > floor));
    // "THE" occurs twice after "TH", so it is certain in this corpus.
    assert_eq!(model.ngram_score(&letters_from_text("THE")), 0);

    // Text made only of unseen trigrams scores exactly windows * floor.
    let garbage = letters_from_text("QQQQQQQQQQ");
    assert_eq!(model.score_text(&garbage), 8 * floor as i64);

    // A single unseen window costs a bounded amount instead of -infinity, so
    // an otherwise good decryption still outranks garbage.
    let mostly_good = letters_from_text("THEQUICKBROWNFOXJUMPSQ");
    let good_len = mostly_good.len();
    let all_garbage = vec![16u8; good_len];
    assert!(model.score_text(&mostly_good) > model.score_text(&all_garbage));
}

#[test]
fn solver_handles_a_sparse_model_without_panicking() {
    // Order-5 model from a tiny corpus: nearly every 5-gram is unseen.
    let corpus = letters_from_text(&"attackatdawnholdthebridge".repeat(4));
    let model = NgramModel::train(&corpus, 5).unwrap();
    let plain = letters_from_text("attackatdawnholdthebridgeattackatdawn");
    let key: Vec<u8> = (0..26).collect();
    let cipher: Vec<u16> = plain.iter().map(|&l| l as u16).collect();
    let truth = Scorer::new(&model, &cipher, key).score();
    let solution = solve(
        &model,
        &cipher,
        26,
        &SolverConfig {
            restarts: 4,
            iterations: 100_000,
            ..SolverConfig::default()
        },
    );
    assert!(solution.score >= truth);
    assert_eq!(solution.plaintext.len(), plain.len());
}
