# homophone

A homophonic-substitution cipher solver in Rust that also measures how much ciphertext it needs before its answers can be trusted.

## The problem

A homophonic cipher gives each plaintext letter several ciphertext symbols, so frequent letters stop standing out and frequency analysis stops working. The solvers hobbyists actually use, such as AZdecrypt, are closed-source Windows GUIs. Academic code tends to be slow Python that reports no success rates. Nobody publishes recovery rate as a function of ciphertext length and homophone count. So when someone claims to have broken a short cipher, you can't check whether a solver could even recover that much text from that little ciphertext. `homophone` is a solver and a measuring instrument in one: it solves ciphers, and it produces that curve for any language model you give it.

## How it works

**Search.** A key maps each ciphertext symbol to a letter. Simulated annealing starts from a random key. Each move gives one symbol a different letter, accepts it if the score goes up, and otherwise accepts it with probability `exp(Δ/T)`. Temperature falls linearly to zero, then a steepest-ascent pass polishes the best key found. Independent restarts run in parallel on `std::thread::scope`, and the highest score wins. Each restart has its own seed stream, so the result is the same for any thread count.

**Scoring.** The language model is a flat `Vec<i16>` of conditional log-probabilities `log P(c_n | c_1..c_{n-1})` for order 2–5, indexed by the base-26 hash `c_1·26^(n-1) + … + c_n`. Values are stored in millinats. A key's score is the log-likelihood of the *ciphertext*:

```text
score = Σ windows  log P(letter | previous n-1 letters)       n-gram term
      + Σ letters  ln Γ(k_l) − ln Γ(n_l + k_l)                homophone term
      + Σ symbols  ln n_s!                                    constant
```

Here `n_l` is how often letter `l` occurs in the decryption, `k_l` is how many symbols map to it, and `n_s` is how often symbol `s` occurs. The homophone term is the probability of which symbol got picked for each letter, with the per-letter symbol probabilities integrated out under a uniform Dirichlet prior.

**Incremental update.** The scorer precomputes each symbol's sorted position list. To move symbol `s`, it walks those positions and merges the windows that overlap them into runs, so no window is counted twice. It rolls the hash across each run and sums the table entries before and after rewriting the letters. Cost is O(occurrences × n), not O(length). The homophone term changes in O(1), because only two letters' `(n_l, k_l)` change.

Worked example, order 3, where symbol `s` occurs at positions 10 and 11 of a 1500-letter text:

```text
position:   8  9 10 11 12 13
                  s  s
windows touching 10: starts 8, 9, 10
windows touching 11: starts 9, 10, 11   -> merged run: starts 8..=11, 4 windows
```

Four table lookups before and four after, against 1498 for a full rescore.

**Temperature calibration.** Before a run, the annealer tries 400 random moves from the random key and measures the mean score loss of the worsening ones. The starting temperature is 0.35 × that loss. A move's score change grows with how often the symbol occurs (length ÷ symbols) and with the model order, so this sets the schedule from the ciphertext's own length and homophone count.

**Unicity mode.** For each symbol count and trial, it samples one plaintext from the model at the longest requested length, draws a random key, and encrypts. Each shorter length uses a prefix of that same ciphertext. It solves every prefix and reports mean and minimum recovery as CSV, plus the shortest length from which mean recovery stays at or above 90%.

**Synthetic language.** `MarkovLanguage` is a random order-3 Markov chain over 26 letters (each letter depends on the previous two). For each context, the letters are ranked and rank `r` gets probability ∝ `r^-1.5` (Zipf). Each context's ranking is a noisy copy of its shorter suffix's ranking, so the language has unigram and bigram structure as well as trigram structure. Tests and the benchmark use it to generate corpora and plaintexts without shipping or downloading any text.

## Install and usage

Requires a stable Rust toolchain (developed and tested with 1.94). The crate has no dependencies at all.

```sh
git clone <this repository> homophone && cd homophone
cargo build --release
cargo test            # about 20 s; the test profile is optimised
```

End-to-end with the synthetic language. Every command below was run as shown, and the output is real.

```console
$ cargo run --release -q -- synth --out corpus.txt --letters 1000000
wrote 1000000 letters to corpus.txt

$ cargo run --release -q -- train-model --corpus corpus.txt --out model.bin --order 3
trained order-3 model on 1000000 letters: 13520 of 17576 n-grams seen, floor -11.261 nats

$ cargo run --release -q -- synth --out plain.txt --letters 1500 --seed 42
wrote 1500 letters to plain.txt

$ cargo run --release -q -- encrypt --input plain.txt --symbols 64 --seed 42 --key-out key.txt > cipher.txt
encrypted 1500 letters with 64 symbols

$ head -2 cipher.txt
47 39 21 51 19 50 25 11 21 53 47 25 28 3 14 28 16 25 32 51
24 13 3 48 3 47 17 19 55 28 47 17 38 14 32 14 35 47 17 17

$ cargo run --release -q -- solve --model model.bin --input cipher.txt --truth plain.txt
length 1500  symbols 64  order 3  restarts 8  moves/restart 1331200  time 0.45s
score -3.3371 nats/window (best restart #0, 8 of 8 restarts reached it)
recovery 99.87%

plaintext:
KQMFLTGGMLKGCCMCUGCFFCCXCKALTCKAQMCMLKAAOUUMACMGGZAAGMRCCKUG
LUGMGMGMFQLUBKKGGZTZGQTGGGGGCTSGCMLLCETGLZZUUMMMMUGLUQVQMFQM
...

key (letter: symbols):
A: 17 62
B: 0
C: 28 3 32 13
...
```

To use real text, point `train-model` at any plain-text corpus. Only the letters A–Z are kept, case is folded, and everything else is dropped. Ciphertext can be whitespace-separated tokens (the default) or one symbol per character (`--chars`). `--truth` is optional.

The unicity table:

```sh
cargo run --release -q -- unicity --model model.bin \
  --lengths 50,100,150,200,300,400,600,800,1000 --symbols 26,40,60,80 \
  --trials 10 --restarts 8 > unicity.csv
```

Run `cargo run --release -- help` to see every option.

## Results

Measured on an 8-core arm64 Mac (macOS 26), Rust 1.94, release profile. The throughput and recovery numbers come from one command:

```sh
cargo run --release --example bench
```

### Incremental versus full rescoring

These are single-threaded moves per second on a 60-symbol ciphertext, half committed and half rolled back. "Full" updates the decryption and rescores the whole text with the same rolling hash.

| order | length | incremental moves/s | full rescoring moves/s | speedup |
|---|---|---|---|---|
| 3 | 500 | 11.15M | 0.958M | 12x |
| 3 | 2000 | 3.83M | 0.248M | 15x |
| 3 | 10000 | 0.79M | 0.052M | 15x |
| 4 | 500 | 10.37M | 0.985M | 11x |
| 4 | 2000 | 3.01M | 0.247M | 12x |
| 4 | 10000 | 0.62M | 0.051M | 12x |
| 5 | 500 | 3.67M | 0.806M | 5x |
| 5 | 2000 | 1.07M | 0.207M | 5x |
| 5 | 10000 | 0.24M | 0.043M | 6x |

The speedup stays roughly flat as length grows. With a fixed symbol count, each symbol's occurrences grow in step with the length, so both costs scale with length. The ratio is about symbols ÷ order. Order 5 gains less because its table is 26⁵ × 2 bytes = 24 MB, and random lookups into it miss the cache.

### Time to 95% recovery

The test set is 20 seeds. Each is a 1500-letter plaintext from the synthetic language, encrypted with a random key of 40–80 symbols and 1–4 homophones per letter, with extra homophones going to frequent letters. The solver uses an order-3 model trained on a separate 1M-letter sample, 8 restarts, and 8 threads. The move budget doubles until every seed reaches 95%:

| moves/restart | seeds >= 95% | mean recovery | mean solve time | max solve time |
|---|---|---|---|---|
| 1000 | 13/20 | 0.7017 | 0.005s | 0.007s |
| 2000 | 15/20 | 0.7911 | 0.005s | 0.006s |
| 4000 | 19/20 | 0.9482 | 0.005s | 0.006s |
| 8000 | 19/20 | 0.9541 | 0.007s | 0.009s |
| 16000 | 20/20 | 0.9933 | 0.009s | 0.012s |
| default (0.8–1.7M) | 20/20 | 0.9954 | 0.453s | 0.488s |

At 16,000 moves per restart, all 20 seeds reach 95% in 12 ms or less of wall time. The default budget is about 80 times larger. That buys margin on harder ciphers, such as shorter ones or real-language models, where one restart often lands in a local optimum. The test suite enforces the same claim: `tests/recovery.rs` requires at least 95% on each of the 20 seeds with default settings. Its lowest seed is 98.9%.

### Unicity curve

These are mean recovery rates from the `unicity` command shown above, run against the order-3 model from the walkthrough. Each cell averages 10 trials with default settings and 8 restarts. Plaintexts are sampled from the model. A column's rows are nested prefixes of the same ciphertexts.

| length | 26 symbols | 40 symbols | 60 symbols | 80 symbols |
|---|---|---|---|---|
| 50 | 0.730 | 0.258 | 0.076 | 0.088 |
| 100 | **0.923** | 0.798 | 0.223 | 0.161 |
| 150 | 0.952 | 0.892 | 0.379 | 0.376 |
| 200 | 0.977 | **0.948** | 0.771 | 0.532 |
| 300 | 0.987 | 0.981 | **0.947** | **0.919** |
| 400 | 0.993 | 0.989 | 0.971 | 0.944 |
| 600 | 0.996 | 0.997 | 0.987 | 0.968 |
| 800 | 0.999 | 0.999 | 0.991 | 0.983 |
| 1000 | 1.000 | 0.999 | 0.995 | 0.990 |

Bold marks the reported threshold, the shortest length from which mean recovery stays at or above 90%: 100 letters for plain substitution, 200 for 40 symbols, and 300 for both 60 and 80 symbols. The transition is sharp. At 60 symbols, 200 letters gives 77% on average but the worst trial recovers nothing, while 300 letters gives at least 92% on every trial. The full CSV also reports the minimum recovery per cell.

These thresholds belong to this synthetic language. English has different redundancy, so run `unicity` with a model trained on English to get English numbers.

## Design notes

**Score the ciphertext, not the plaintext.** The first version scored only the n-gram log-probability of the decryption, and it failed completely: 4% recovery, which is chance. The annealer found keys that send most symbols to two or three letters forming a high-probability loop in the model. Those keys score better than the true key, because the true plaintext is less probable than a degenerate one. The fix is to score what was actually observed, which is the symbols. Adding `log P(symbols | letters)` with the homophone probabilities integrated out under a Dirichlet prior makes piling symbols onto one letter cost what it should. I also tried a maximum-likelihood version of that term, `Σ n_s ln n_s − Σ n_l ln n_l`. It also works, but it gave slightly lower recovery in my comparison, mostly from over-fitting rare symbols. Either way the term updates in O(1) per move, so correctness costs no speed. On the 1500-letter cases I inspected, and in `tests/substitution.rs`, the solver's best score was at or above the true key's score. So the remaining errors come from the objective, not from search.

**Integer scores so exactness is checkable.** Log-probabilities are quantised to millinats in an `i16` table, and `ln x!` is tabulated as rounded `i64`s. Every score is therefore a sum of integers, so the incremental score must *equal* a full rescore after any sequence of moves, not just approximately match it. The property test checks exactly that over 10,000 moves, orders 2–5, dense repeats and absent symbols. A floating-point scorer would need a tolerance, and a tolerance hides double-counted windows. The cost is 0.001-nat resolution, far below the differences that decide a move. The `i16` table also halves the memory of an order-5 model to 24 MB.

**Why the synthetic language has structure at every order.** My first generator gave every context an independent random ranking of the letters. Under the new scoring, the annealer still sat at garbage on most restarts, and occasionally a restart jumped to the exact answer. The cause: when contexts are independent, unigram and bigram statistics are flat, so a key that is 80% right scores no better than a random one, and annealing has no gradient to climb. Natural languages aren't like that. Each context's ranking is now a noisy copy of its shorter context's ranking. With that change every restart converges, and the benchmark results reflect the solver rather than a pathological test language.

## Limitations

- All published numbers come from a synthetic language. I haven't measured recovery on English, because the project ships no corpus and must run offline. Train a model on your own corpus and run `unicity` to get numbers for your language.
- The model is a plain conditional n-gram table with one shared floor for unseen n-grams. It has no interpolation or back-off. Order-5 models from small corpora are mostly floor, which blunts the score. Order 3 or 4 is the practical choice unless the corpus runs to tens of millions of letters.
- The only alphabet is A–Z, and word boundaries are ignored. Transposition, nulls, polyalphabetic ciphers and the Zodiac-340-style transposition + homophonic combination are not handled.
- Each move reassigns one symbol. There are no swap or block moves, which can help escape local optima on very short ciphertexts.
- The solver doesn't constrain the key shape. Its output can map a symbol to a letter that appears nowhere else, or give a letter more homophones than the encipherer would plausibly use. If you know such constraints, you have to apply them outside the solver.
- The default move budget is generous rather than adaptive. On easy ciphers, most of the 0.45 s is spent confirming an answer found in the first few percent of the run.
- Unicity measurements cost one full solve per (length, symbol count, trial) cell, so large grids take minutes.

## License

MIT. See [LICENSE](LICENSE).
