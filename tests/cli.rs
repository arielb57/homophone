//! End-to-end runs of the binary: synth -> train-model -> encrypt -> solve,
//! plus unicity output and user-facing errors.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_homophone"))
}

fn workdir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn run(cmd: &mut Command) -> Output {
    let out = cmd.output().expect("binary runs");
    assert!(
        out.status.success(),
        "command failed: {}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

fn path(dir: &Path, file: &str) -> String {
    dir.join(file).to_string_lossy().into_owned()
}

#[test]
fn full_pipeline_recovers_an_encrypted_sample() {
    let dir = workdir("pipeline");
    let (corpus, model, plain, cipher) = (
        path(&dir, "corpus.txt"),
        path(&dir, "model.bin"),
        path(&dir, "plain.txt"),
        path(&dir, "cipher.txt"),
    );
    run(bin().args([
        "synth",
        "--out",
        &corpus,
        "--letters",
        "600000",
        "--zipf",
        "1.5",
        "--language",
        "9",
    ]));
    run(bin().args([
        "train-model",
        "--corpus",
        &corpus,
        "--out",
        &model,
        "--order",
        "3",
    ]));
    run(bin().args([
        "synth",
        "--out",
        &plain,
        "--letters",
        "1500",
        "--zipf",
        "1.5",
        "--language",
        "9",
        "--seed",
        "5",
    ]));
    let enc = run(bin().args([
        "encrypt",
        "--input",
        &plain,
        "--symbols",
        "55",
        "--seed",
        "3",
    ]));
    std::fs::write(&cipher, &enc.stdout).unwrap();
    let tokens = String::from_utf8(enc.stdout).unwrap();
    assert_eq!(tokens.split_whitespace().count(), 1500);

    let solved = run(bin().args([
        "solve", "--model", &model, "--input", &cipher, "--truth", &plain,
    ]));
    let stdout = String::from_utf8(solved.stdout).unwrap();
    let recovery: f64 = stdout
        .lines()
        .find_map(|l| l.strip_prefix("recovery "))
        .and_then(|v| v.trim_end_matches('%').parse().ok())
        .expect("solve prints a recovery line");
    assert!(recovery >= 95.0, "CLI recovery {recovery}%\n{stdout}");
    assert!(stdout.contains("plaintext:") && stdout.contains("key (letter: symbols):"));
}

#[test]
fn chars_mode_and_unicity_csv() {
    let dir = workdir("unicity");
    let (corpus, model, cipher) = (
        path(&dir, "corpus.txt"),
        path(&dir, "model.bin"),
        path(&dir, "c.txt"),
    );
    run(bin().args([
        "synth",
        "--out",
        &corpus,
        "--letters",
        "300000",
        "--zipf",
        "1.5",
    ]));
    run(bin().args(["train-model", "--corpus", &corpus, "--out", &model]));

    std::fs::write(&cipher, "ab+cab+c\nab").unwrap();
    let out = run(bin().args([
        "solve",
        "--model",
        &model,
        "--input",
        &cipher,
        "--chars",
        "--restarts",
        "2",
    ]));
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("length 10  symbols 4"), "{stdout}");

    let out = run(bin().args([
        "unicity",
        "--model",
        &model,
        "--lengths",
        "40,400",
        "--symbols",
        "26,40",
        "--trials",
        "2",
        "--restarts",
        "2",
        "--iterations",
        "50000",
    ]));
    let csv = String::from_utf8(out.stdout).unwrap();
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(lines[0], "symbols,length,trials,mean_recovery,min_recovery");
    assert_eq!(lines.len(), 5);
    assert!(lines[1].starts_with("26,40,2,"));
    assert!(lines[4].starts_with("40,400,2,"));
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(stderr.contains("# 26 symbols:") && stderr.contains("# 40 symbols:"));
}

#[test]
fn user_errors_exit_non_zero_with_a_message() {
    let dir = workdir("errors");
    let cases: Vec<(Vec<String>, &str)> = vec![
        (vec!["frobnicate".into()], "unknown command"),
        (
            vec!["solve".into(), "--input".into(), "x".into()],
            "missing required option --model",
        ),
        (
            vec![
                "solve".into(),
                "--model".into(),
                path(&dir, "nope.bin"),
                "--input".into(),
                "x".into(),
            ],
            "cannot open",
        ),
        (
            vec![
                "train-model".into(),
                "--corpus".into(),
                "a".into(),
                "--bogus".into(),
            ],
            "unknown option --bogus",
        ),
        (
            vec![
                "synth".into(),
                "--out".into(),
                path(&dir, "s.txt"),
                "--order".into(),
                "9".into(),
            ],
            "--order",
        ),
    ];
    for (args, message) in cases {
        let out = bin().args(&args).output().unwrap();
        assert!(!out.status.success(), "{args:?} should fail");
        let stderr = String::from_utf8(out.stderr).unwrap();
        assert!(
            stderr.contains(message),
            "{args:?}: expected '{message}' in {stderr}"
        );
    }

    let not_a_model = path(&dir, "fake.bin");
    std::fs::write(&not_a_model, b"hello world").unwrap();
    let cipher = path(&dir, "c.txt");
    std::fs::write(&cipher, "1 2 3").unwrap();
    let out = bin()
        .args(["solve", "--model", &not_a_model, "--input", &cipher])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8(out.stderr)
        .unwrap()
        .contains("invalid model file"));
}
