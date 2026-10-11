//! Causes found by reading an unattributed core eval, where the engine
//! has no `Unsupported` to name one.
//!
//! A table rather than a match in [`super::report`]: a fix or a re-pin
//! that leaves an entry behind is then caught by [`stale`], which the
//! promqltest harness runs as a trial, instead of the report going on
//! explaining a failure that no longer happens.

use super::report::{reason, Reason};
use super::run::Outcome;

/// `(file, id as spelled in SUPPORTED.toml, cause)`.
pub const KNOWN: &[(&str, &str, &str)] = &[
    (
        "type_and_unit",
        r#"(http_requests_total{group="canary"} + 1) and http_requests_total{instance="0"} @ 3000000"#,
        DELAYED_NAME_REMOVAL,
    ),
    (
        "type_and_unit",
        r#"(http_requests_total{group="canary"} + 1) or http_requests_total{instance="1"} @ 3000000"#,
        DELAYED_NAME_REMOVAL,
    ),
    (
        "type_and_unit",
        r#"(http_requests_total{group="canary"} + 1) and http_requests_total{instance="0"} @ 3000000 #2"#,
        DELAYED_NAME_REMOVAL,
    ),
    (
        "type_and_unit",
        r#"(http_requests_total{group="canary"} + 1) or http_requests_total{instance="1"} @ 3000000 #2"#,
        DELAYED_NAME_REMOVAL,
    ),
];

const DELAYED_NAME_REMOVAL: &str = "Needs delayed name removal. The corpus engine runs with \
     `EnableDelayedNameRemoval` (`NewTestEngine`, `promql/promqltest/test.go` at 83962c35), \
     under which `(x + 1)` keeps `__type__` and `__unit__` until the end, so the set \
     operator's signature, which sets aside `__name__` alone, matches the right side; an \
     immediate drop cannot reproduce that.";

/// The cause `causes` names for one eval.
pub fn cause<'a>(causes: &[(&str, &str, &'a str)], file: &str, id: &str) -> Option<&'a str> {
    causes
        .iter()
        .find(|(f, i, _)| *f == file && *i == id)
        .map(|(_, _, c)| *c)
}

/// The entries of `causes` that explain nothing: their eval is gone from
/// the corpus, passes, or fails for a reason the engine now names. Each
/// as `file/id`.
pub fn stale(causes: &[(&str, &str, &str)], outcomes: &[Outcome]) -> Vec<String> {
    causes
        .iter()
        .filter(|(file, id, _)| {
            !outcomes
                .iter()
                .any(|o| o.file == *file && o.id == *id && reason(o) == Some(Reason::Unattributed))
        })
        .map(|(file, id, _)| format!("{file}/{id}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prometheus::run::Verdict;

    fn outcome(id: &str, verdict: Verdict) -> Outcome {
        Outcome {
            file: "f".to_string(),
            id: id.to_string(),
            line: 1,
            query: id.to_string(),
            verdict,
            expected_rows: 1,
            unchecked_annotations: false,
            histograms: false,
            subquery: false,
        }
    }

    #[test]
    fn an_entry_is_stale_once_its_eval_is_not_an_unattributed_failure() {
        let outcomes = [
            outcome("fails", Verdict::Fail("x".into())),
            outcome("passes", Verdict::Pass),
            outcome("named", Verdict::Unsupported("the frob function".into())),
        ];
        let known = [
            ("f", "fails", "c"),
            ("f", "passes", "c"),
            ("f", "named", "c"),
            ("f", "gone", "c"),
            ("g", "fails", "c"),
        ];
        assert_eq!(
            stale(&known, &outcomes),
            ["f/passes", "f/named", "f/gone", "g/fails"]
        );
    }
}
