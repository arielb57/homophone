use homophone::cipher::letter_frequencies;
use homophone::unicity::{self, UnicityConfig};
use homophone::{
    letters_from_text, letters_to_string, recovery_rate, solve, Ciphertext, HomophonicKey,
    MarkovLanguage, NgramModel, Rng, SolverConfig,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::process::ExitCode;

const USAGE: &str = "\
homophone — homophonic-substitution cipher solver

USAGE:
  homophone synth       --out FILE [--letters N] [--language SEED] [--seed SEED]
                        [--order N] [--zipf S]
  homophone train-model --corpus FILE --out MODEL [--order N]
  homophone encrypt     --input FILE [--symbols N] [--max-homophones N] [--seed SEED]
                        [--key-out FILE]
  homophone solve       --model MODEL --input FILE [--chars] [--restarts N]
                        [--iterations N] [--threads N] [--seed SEED]
                        [--temperature F] [--truth FILE]
  homophone unicity     --model MODEL [--lengths L1,L2,..] [--symbols S1,S2,..]
                        [--trials N] [--restarts N] [--iterations N] [--threads N]
                        [--max-homophones N] [--seed SEED] [--temperature F]

Plaintext and corpus files may contain anything; only the letters A-Z are used.
Ciphertext is whitespace-separated symbol tokens, or one symbol per character
with --chars.";

struct Args {
    values: HashMap<String, String>,
    flags: Vec<String>,
}

impl Args {
    fn parse(raw: &[String], known_flags: &[&str], known_values: &[&str]) -> Result<Self, String> {
        let mut values = HashMap::new();
        let mut flags = Vec::new();
        let mut it = raw.iter();
        while let Some(arg) = it.next() {
            let name = arg
                .strip_prefix("--")
                .ok_or_else(|| format!("unexpected argument '{arg}'"))?;
            if known_flags.contains(&name) {
                flags.push(name.to_string());
            } else if known_values.contains(&name) {
                let v = it.next().ok_or_else(|| format!("--{name} needs a value"))?;
                values.insert(name.to_string(), v.clone());
            } else {
                return Err(format!("unknown option --{name}"));
            }
        }
        Ok(Args { values, flags })
    }

    fn required(&self, name: &str) -> Result<&str, String> {
        self.values
            .get(name)
            .map(String::as_str)
            .ok_or_else(|| format!("missing required option --{name}"))
    }

    fn num<T: std::str::FromStr>(&self, name: &str, default: T) -> Result<T, String> {
        match self.values.get(name) {
            None => Ok(default),
            Some(v) => v
                .parse()
                .map_err(|_| format!("--{name}: cannot parse '{v}'")),
        }
    }

    fn list(&self, name: &str, default: &[usize]) -> Result<Vec<usize>, String> {
        match self.values.get(name) {
            None => Ok(default.to_vec()),
            Some(v) => v
                .split(',')
                .map(|x| {
                    x.trim()
                        .parse()
                        .map_err(|_| format!("--{name}: cannot parse '{x}'"))
                })
                .collect(),
        }
    }

    fn flag(&self, name: &str) -> bool {
        self.flags.iter().any(|f| f == name)
    }
}

fn read(path: &str) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))
}

fn load_model(path: &str) -> Result<NgramModel, String> {
    let file = File::open(path).map_err(|e| format!("cannot open {path}: {e}"))?;
    NgramModel::read_from(std::io::BufReader::new(file)).map_err(|e| format!("{path}: {e}"))
}

fn solver_config(args: &Args) -> Result<SolverConfig, String> {
    let d = SolverConfig::default();
    Ok(SolverConfig {
        restarts: args.num("restarts", d.restarts)?,
        iterations: args.num("iterations", d.iterations)?,
        threads: args.num("threads", d.threads)?,
        seed: args.num("seed", d.seed)?,
        temperature: args.num("temperature", d.temperature)?,
    })
}

fn cmd_synth(raw: &[String]) -> Result<(), String> {
    let args = Args::parse(
        raw,
        &[],
        &["out", "letters", "language", "seed", "order", "zipf"],
    )?;
    let out = args.required("out")?;
    let order: usize = args.num("order", 3)?;
    if !(2..=5).contains(&order) {
        return Err("--order must be between 2 and 5".into());
    }
    let zipf: f64 = args.num("zipf", 1.5)?;
    if zipf.is_nan() || zipf < 0.0 {
        return Err("--zipf must be a non-negative number".into());
    }
    let lang = MarkovLanguage::new(order, zipf, args.num("language", 1)?);
    let letters = lang.sample(
        args.num("letters", 1_000_000)?,
        &mut Rng::new(args.num("seed", 1)?),
    );
    std::fs::write(out, letters_to_string(&letters))
        .map_err(|e| format!("cannot write {out}: {e}"))?;
    eprintln!("wrote {} letters to {out}", letters.len());
    Ok(())
}

fn cmd_train(raw: &[String]) -> Result<(), String> {
    let args = Args::parse(raw, &[], &["corpus", "out", "order"])?;
    let corpus = letters_from_text(&read(args.required("corpus")?)?);
    let order = args.num("order", 3)?;
    let model = NgramModel::train(&corpus, order).map_err(|e| e.to_string())?;
    let out = args.required("out")?;
    let file = File::create(out).map_err(|e| format!("cannot create {out}: {e}"))?;
    model
        .write_to(BufWriter::new(file))
        .map_err(|e| e.to_string())?;
    let seen = model.table().iter().filter(|&&v| v > model.floor()).count();
    eprintln!(
        "trained order-{order} model on {} letters: {seen} of {} n-grams seen, floor {:.3} nats",
        corpus.len(),
        model.table().len(),
        model.floor() as f64 / 1000.0
    );
    Ok(())
}

fn cmd_encrypt(raw: &[String]) -> Result<(), String> {
    let args = Args::parse(
        raw,
        &[],
        &["input", "symbols", "max-homophones", "seed", "key-out"],
    )?;
    let plain = letters_from_text(&read(args.required("input")?)?);
    let mut rng = Rng::new(args.num("seed", 1)?);
    let key = HomophonicKey::random(
        args.num("symbols", 60)?,
        args.num("max-homophones", 4)?,
        &letter_frequencies(&plain),
        &mut rng,
    )?;
    let cipher = key.encrypt(&plain, &mut rng);
    let mut stdout = BufWriter::new(std::io::stdout().lock());
    for line in cipher.chunks(20) {
        let tokens: Vec<String> = line.iter().map(|s| s.to_string()).collect();
        writeln!(stdout, "{}", tokens.join(" ")).map_err(|e| e.to_string())?;
    }
    stdout.flush().map_err(|e| e.to_string())?;
    if let Some(path) = args.values.get("key-out") {
        let mut text = String::new();
        for (letter, symbols) in key.homophones.iter().enumerate() {
            let names: Vec<String> = symbols.iter().map(|s| s.to_string()).collect();
            text.push_str(&format!(
                "{} {}\n",
                (b'A' + letter as u8) as char,
                names.join(" ")
            ));
        }
        std::fs::write(path, text).map_err(|e| format!("cannot write {path}: {e}"))?;
    }
    eprintln!(
        "encrypted {} letters with {} symbols",
        plain.len(),
        key.num_symbols()
    );
    Ok(())
}

fn cmd_solve(raw: &[String]) -> Result<(), String> {
    let args = Args::parse(
        raw,
        &["chars"],
        &[
            "model",
            "input",
            "restarts",
            "iterations",
            "threads",
            "seed",
            "temperature",
            "truth",
        ],
    )?;
    let model = load_model(args.required("model")?)?;
    let text = read(args.required("input")?)?;
    let cipher = if args.flag("chars") {
        Ciphertext::parse_chars(&text)?
    } else {
        Ciphertext::parse_tokens(&text)?
    };
    if cipher.symbols.is_empty() {
        return Err("ciphertext is empty".into());
    }
    let config = solver_config(&args)?;
    let solution = solve(&model, &cipher.symbols, cipher.num_symbols(), &config);

    let windows = cipher
        .symbols
        .len()
        .saturating_sub(model.order() - 1)
        .max(1);
    println!(
        "length {}  symbols {}  order {}  restarts {}  moves/restart {}  time {:.2}s",
        cipher.symbols.len(),
        cipher.num_symbols(),
        model.order(),
        solution.restart_scores.len(),
        solution.iterations_per_restart,
        solution.elapsed.as_secs_f64()
    );
    println!(
        "score {:.4} nats/window (best restart #{}, {} of {} restarts reached it)",
        solution.score as f64 / 1000.0 / windows as f64,
        solution.best_restart,
        solution
            .restart_scores
            .iter()
            .filter(|&&s| s == solution.score)
            .count(),
        solution.restart_scores.len()
    );
    if let Some(path) = args.values.get("truth") {
        let truth = letters_from_text(&read(path)?);
        if truth.len() != solution.plaintext.len() {
            return Err(format!(
                "--truth has {} letters but the ciphertext has {} symbols",
                truth.len(),
                solution.plaintext.len()
            ));
        }
        println!(
            "recovery {:.2}%",
            100.0 * recovery_rate(&truth, &solution.plaintext)
        );
    }
    println!("\nplaintext:");
    for chunk in solution.plaintext.chunks(60) {
        println!("{}", letters_to_string(chunk));
    }
    println!("\nkey (letter: symbols):");
    for letter in 0..26u8 {
        let names: Vec<&str> = (0..cipher.num_symbols())
            .filter(|&s| solution.key[s] == letter)
            .map(|s| cipher.names[s].as_str())
            .collect();
        if !names.is_empty() {
            println!("{}: {}", (b'A' + letter) as char, names.join(" "));
        }
    }
    Ok(())
}

fn cmd_unicity(raw: &[String]) -> Result<(), String> {
    let args = Args::parse(
        raw,
        &[],
        &[
            "model",
            "lengths",
            "symbols",
            "trials",
            "restarts",
            "iterations",
            "threads",
            "max-homophones",
            "seed",
            "temperature",
        ],
    )?;
    let model = load_model(args.required("model")?)?;
    let mut solver = solver_config(&args)?;
    solver.restarts = args.num("restarts", 4)?;
    let config = UnicityConfig {
        lengths: args.list("lengths", &[100, 200, 300, 400, 600, 800, 1000, 1500])?,
        symbol_counts: args.list("symbols", &[26, 52, 78])?,
        max_homophones: args.num("max-homophones", 4)?,
        trials: args.num("trials", 5)?,
        seed: args.num("seed", 1)?,
        solver,
    };
    let rows = unicity::run(&model, &config, |len, rng| model.sample(len, rng))?;
    print!("{}", unicity::to_csv(&rows));
    for &symbols in &config.symbol_counts {
        match unicity::threshold_length(&rows, symbols, 0.9) {
            Some(len) => eprintln!("# {symbols} symbols: mean recovery >= 90% from length {len}"),
            None => {
                eprintln!("# {symbols} symbols: mean recovery never stays >= 90% in this range")
            }
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = argv.first() else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let rest = &argv[1..];
    let result = match command.as_str() {
        "synth" => cmd_synth(rest),
        "train-model" => cmd_train(rest),
        "encrypt" => cmd_encrypt(rest),
        "solve" => cmd_solve(rest),
        "unicity" => cmd_unicity(rest),
        "-h" | "--help" | "help" => {
            println!("{USAGE}");
            Ok(())
        }
        other => Err(format!("unknown command '{other}'\n\n{USAGE}")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
