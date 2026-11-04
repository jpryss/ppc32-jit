//! The differential test against Unicorn, short enough for `cargo test`: 3,000 random
//! sequences and the self-modifying-code check (examples/fuzz.rs has the whole thing).
#![allow(dead_code)]
#[path = "../examples/fuzz.rs"]
mod fuzz;

#[test]
fn agrees_with_unicorn() {
    let rc = fuzz::run(vec!["fuzz".into(), "3000".into(), "42".into()]);
    assert_eq!(rc, 0, "the JIT and Unicorn disagree on some sequence (see the output)");
}
