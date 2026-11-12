"""
The reliability system: measure whether the recommender actually works.

Run it with:

    python -m src.evaluation            # offline, deterministic, no API key
    python -m src.evaluation --online   # exercise the real Claude calls

It scores seven things against the golden cases in `eval/cases.json`:

  retrieval_recall_at_k   did retrieval surface the documents that matter?
  retrieval_mrr           how high up was the first relevant document?
  profile_field_accuracy  did the request get parsed into the right profile?
  schema_valid_rate       did every model output match its schema?
  groundedness_rate       did every citation point at a document we retrieved?
  top1_consistency        does the same request give the same #1 every time?
  topk_jaccard            how stable is the whole top-k across repeats?

plus two diagnostics that are reported but not gated:

  retrieval_impact_rate   how often RAG changed the answer vs rules alone
                          (if this is 0, the retrieval layer is decorative)
  top1_acceptable_rate    how often #1 was a defensible pick

Any metric below its threshold in `src/config.py` fails the run (exit code 1),
so this doubles as a regression gate.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.config import (
    CONSISTENCY_REPEATS,
    EVAL_CASES,
    EVAL_REPORT,
    RETRIEVAL_TOP_K,
    THRESHOLDS,
    RunSettings,
)
from src.logging_setup import get_logger, setup_logging
from src.pipeline import PipelineResult, RecommendationPipeline
from src.recommender import recommend_songs

log = get_logger("evaluation")


# --- metric helpers --------------------------------------------------------


def recall_at_k(gold: Sequence[str], retrieved: Sequence[str]) -> Optional[float]:
    """Fraction of gold documents that appear in the retrieved set."""
    if not gold:
        return None  # nothing to measure for this case
    hit = len(set(gold) & set(retrieved))
    return hit / len(gold)


def reciprocal_rank(gold: Sequence[str], retrieved: Sequence[str]) -> Optional[float]:
    """1/rank of the first gold document, 0.0 if none appear."""
    if not gold:
        return None
    gold_set = set(gold)
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in gold_set:
            return 1.0 / i
    return 0.0


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def profile_accuracy(gold: Optional[Dict[str, Any]], actual: Dict[str, Any]) -> Optional[float]:
    """
    Fraction of the specified gold fields the parsed profile got right.

    `target_energy` is graded against an inclusive range because "calm" does
    not have one true numeric answer; a null gold field is skipped.
    """
    if not gold:
        return None

    checks: List[bool] = []
    for field_name in ("favorite_genre", "favorite_mood", "likes_acoustic"):
        expected = gold.get(field_name)
        if expected is None:
            continue
        checks.append(actual.get(field_name) == expected)

    energy_range = gold.get("target_energy")
    if energy_range is not None:
        low, high = energy_range
        checks.append(low <= float(actual.get("target_energy", -1)) <= high)

    if not checks:
        return None
    return sum(checks) / len(checks)


def mean(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return statistics.fmean(present) if present else None


# --- the harness -----------------------------------------------------------


@dataclass
class CaseOutcome:
    """Per-case metrics, kept so the report can show which case regressed."""

    case_id: str
    query: str
    answerable: bool = True   # False for the deliberately out-of-catalog case
    error: str = ""
    recall: Optional[float] = None
    mrr: Optional[float] = None
    profile_acc: Optional[float] = None
    schema_valid: bool = True
    groundedness: float = 1.0
    fell_back: bool = False
    top1_stable: Optional[bool] = None
    jaccard: Optional[float] = None
    rag_changed_topk: Optional[bool] = None
    rag_changed_top1: Optional[bool] = None
    mean_confidence: Optional[float] = None
    low_confidence_count: Optional[int] = None
    top1_acceptable: Optional[bool] = None
    top_song_ids: List[int] = field(default_factory=list)
    rule_only_top_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        def r(x):
            return round(x, 4) if isinstance(x, float) else x

        return {
            "case_id": self.case_id,
            "query": self.query,
            "error": self.error,
            "recall_at_k": r(self.recall),
            "mrr": r(self.mrr),
            "profile_field_accuracy": r(self.profile_acc),
            "schema_valid": self.schema_valid,
            "groundedness": r(self.groundedness),
            "fell_back": self.fell_back,
            "top1_stable": self.top1_stable,
            "topk_jaccard": r(self.jaccard),
            "rag_changed_topk": self.rag_changed_topk,
            "rag_changed_top1": self.rag_changed_top1,
            "mean_confidence": r(self.mean_confidence),
            "low_confidence_count": self.low_confidence_count,
            "top1_acceptable": self.top1_acceptable,
            "top_song_ids": self.top_song_ids,
            "rule_only_top_ids": self.rule_only_top_ids,
        }


def load_cases(path=EVAL_CASES) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    cases = data.get("cases", [])
    if not cases:
        raise RuntimeError(f"no cases found in {path}")
    return cases


def evaluate_case(
    pipeline: RecommendationPipeline,
    case: Dict[str, Any],
    k: int,
    repeats: int,
) -> CaseOutcome:
    """Run one golden case (repeated for the stability metrics) and score it."""
    outcome = CaseOutcome(
        case_id=case["id"],
        query=case["query"],
        answerable=bool(case.get("gold_docs")),
    )

    runs: List[PipelineResult] = []
    try:
        for _ in range(max(1, repeats)):
            runs.append(pipeline.run(case["query"], k=k))
    except Exception as exc:  # a crash is itself a reliability failure
        outcome.error = f"{type(exc).__name__}: {exc}"
        log.error("case %s crashed: %s", case["id"], traceback.format_exc())
        return outcome

    primary = runs[0]
    outcome.top_song_ids = primary.top_song_ids

    gold_docs = case.get("gold_docs") or []
    outcome.recall = recall_at_k(gold_docs, primary.retrieved_ids)
    outcome.mrr = reciprocal_rank(gold_docs, primary.retrieved_ids)
    outcome.profile_acc = profile_accuracy(case.get("gold_profile"), primary.profile)

    outcome.mean_confidence = statistics.fmean(r.mean_confidence for r in runs)
    outcome.low_confidence_count = primary.low_confidence_count
    outcome.schema_valid = all(r.guardrails.schema_valid for r in runs)
    outcome.groundedness = statistics.fmean(r.guardrails.groundedness for r in runs)
    outcome.fell_back = any(r.guardrails.fell_back for r in runs)

    if len(runs) > 1:
        first_picks = {r.top_song_ids[0] if r.top_song_ids else None for r in runs}
        outcome.top1_stable = len(first_picks) == 1
        outcome.jaccard = statistics.fmean(
            jaccard(a.top_song_ids, b.top_song_ids)
            for a, b in itertools.combinations(runs, 2)
        )

    # Ablation: what would the rule scorer alone have returned?
    rule_only = recommend_songs(primary.profile, pipeline.songs, k=k)
    outcome.rule_only_top_ids = [row[0]["id"] for row in rule_only]
    if outcome.top_song_ids and outcome.rule_only_top_ids:
        outcome.rag_changed_top1 = (
            outcome.top_song_ids[0] != outcome.rule_only_top_ids[0]
        )
        outcome.rag_changed_topk = outcome.top_song_ids != outcome.rule_only_top_ids

    acceptable = case.get("acceptable_top1")
    if acceptable and outcome.top_song_ids:
        outcome.top1_acceptable = outcome.top_song_ids[0] in acceptable

    return outcome


def aggregate(outcomes: Sequence[CaseOutcome]) -> Dict[str, Optional[float]]:
    """Roll per-case outcomes up into the headline metrics."""
    ok = [o for o in outcomes if not o.error]

    def rate(values: Sequence[Optional[bool]]) -> Optional[float]:
        present = [v for v in values if v is not None]
        return sum(present) / len(present) if present else None

    # Does the system know when it doesn't know? Confidence on answerable cases
    # minus confidence on the deliberately unanswerable one. Positive is correct.
    answerable = mean([o.mean_confidence for o in ok if o.answerable])
    unanswerable = mean([o.mean_confidence for o in ok if not o.answerable])
    discrimination = (
        answerable - unanswerable
        if answerable is not None and unanswerable is not None
        else None
    )

    return {
        "retrieval_recall_at_k": mean([o.recall for o in ok]),
        "retrieval_mrr": mean([o.mrr for o in ok]),
        "profile_field_accuracy": mean([o.profile_acc for o in ok]),
        "schema_valid_rate": rate([o.schema_valid for o in ok]),
        "groundedness_rate": mean([o.groundedness for o in ok]),
        "top1_consistency": rate([o.top1_stable for o in ok]),
        "topk_jaccard": mean([o.jaccard for o in ok]),
        "guardrail_error_rate": (
            len(outcomes) - len(ok)) / len(outcomes) if outcomes else 0.0,
        "fallback_rate": rate([o.fell_back for o in ok]),
        "retrieval_impact_rate": rate([o.rag_changed_topk for o in ok]),
        "retrieval_impact_top1_rate": rate([o.rag_changed_top1 for o in ok]),
        "mean_confidence": mean([o.mean_confidence for o in ok]),
        "confidence_discrimination": discrimination,
        "top1_acceptable_rate": rate([o.top1_acceptable for o in ok]),
    }


# Metrics that gate the run, and whether higher or lower is better.
GATED = {
    "retrieval_recall_at_k": ("min", THRESHOLDS.retrieval_recall_at_k),
    "profile_field_accuracy": ("min", THRESHOLDS.profile_field_accuracy),
    "schema_valid_rate": ("min", THRESHOLDS.schema_valid_rate),
    "groundedness_rate": ("min", THRESHOLDS.groundedness_rate),
    "top1_consistency": ("min", THRESHOLDS.top1_consistency),
    "topk_jaccard": ("min", THRESHOLDS.topk_jaccard),
    "guardrail_error_rate": ("max", THRESHOLDS.guardrail_error_rate),
    "confidence_discrimination": ("min", THRESHOLDS.confidence_discrimination),
}

DIAGNOSTIC = ("retrieval_mrr", "fallback_rate", "retrieval_impact_rate",
              "retrieval_impact_top1_rate", "top1_acceptable_rate",
              "mean_confidence")


def check_thresholds(metrics: Dict[str, Optional[float]]) -> List[Dict[str, Any]]:
    """Compare gated metrics to their thresholds."""
    results = []
    for name, (direction, threshold) in GATED.items():
        value = metrics.get(name)
        if value is None:
            passed = True  # not measurable in this configuration
        elif direction == "min":
            passed = value >= threshold - 1e-9
        else:
            passed = value <= threshold + 1e-9
        results.append(
            {
                "metric": name,
                "value": None if value is None else round(value, 4),
                "direction": direction,
                "threshold": threshold,
                "passed": passed,
            }
        )
    return results


def render_report(
    metrics: Dict[str, Optional[float]],
    checks: Sequence[Dict[str, Any]],
    outcomes: Sequence[CaseOutcome],
    mode: str,
    elapsed: float,
) -> str:
    def fmt(value: Optional[float]) -> str:
        return "  n/a" if value is None else f"{value:6.3f}"

    lines: List[str] = []
    lines.append("=" * 72)
    lines.append(f"Crate reliability report  (mode={mode}, {elapsed:.1f}s)")
    lines.append("=" * 72)

    lines.append("")
    lines.append("Gated metrics")
    lines.append(f"  {'metric':<26}{'value':>8}  {'bound':>12}   result")
    for check in checks:
        bound = f"{'>=' if check['direction'] == 'min' else '<='} {check['threshold']:.2f}"
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(
            f"  {check['metric']:<26}{fmt(check['value'])}  {bound:>12}   {status}"
        )

    lines.append("")
    lines.append("Diagnostics (reported, not gated)")
    for name in DIAGNOSTIC:
        lines.append(f"  {name:<26}{fmt(metrics.get(name))}")

    lines.append("")
    lines.append("Per case")
    lines.append(
        f"  {'case':<24}{'recall':>7}{'prof':>6}{'grnd':>6}{'conf':>6}"
        f"{'stable':>8}{'ragchg':>8}  top-1"
    )
    for out in outcomes:
        if out.error:
            lines.append(f"  {out.case_id:<24}  ERROR: {out.error}")
            continue
        top1 = out.top_song_ids[0] if out.top_song_ids else "-"
        lines.append(
            f"  {out.case_id:<24}"
            f"{'   n/a' if out.recall is None else f'{out.recall:7.2f}'}"
            f"{'   n/a' if out.profile_acc is None else f'{out.profile_acc:6.2f}'}"
            f"{out.groundedness:6.2f}"
            f"{'   n/a' if out.mean_confidence is None else f'{out.mean_confidence:6.2f}'}"
            f"{str(out.top1_stable):>8}"
            f"{str(out.rag_changed_topk):>8}"
            f"  song {top1}"
        )

    failed = [c["metric"] for c in checks if not c["passed"]]
    lines.append("")
    lines.append("-" * 72)
    if failed:
        lines.append(f"RESULT: FAIL - below threshold: {', '.join(failed)}")
    else:
        lines.append("RESULT: PASS - every gated metric met its threshold")
    lines.append(f"Full report written to {EVAL_REPORT}")
    return "\n".join(lines)


def run_evaluation(
    online: bool = False,
    k: int = 5,
    repeats: int = CONSISTENCY_REPEATS,
    cases_path=EVAL_CASES,
) -> Dict[str, Any]:
    """Run the whole harness and write eval/report.json. Returns the report."""
    started = time.monotonic()
    cases = load_cases(cases_path)
    pipeline = RecommendationPipeline(force_offline=not online)
    mode = pipeline.llm.mode

    log.info("evaluating %d cases (mode=%s, repeats=%d)", len(cases), mode, repeats)

    outcomes = [evaluate_case(pipeline, case, k, repeats) for case in cases]
    metrics = aggregate(outcomes)
    checks = check_thresholds(metrics)
    elapsed = time.monotonic() - started

    report = {
        "mode": mode,
        "elapsed_seconds": round(elapsed, 2),
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "recommendations_k": k,
        "repeats": repeats,
        "settings": RunSettings().to_dict(),
        "metrics": {
            name: (None if value is None else round(value, 4))
            for name, value in metrics.items()
        },
        "threshold_checks": checks,
        "passed": all(c["passed"] for c in checks),
        "cases": [o.to_dict() for o in outcomes],
    }

    EVAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    EVAL_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(render_report(metrics, checks, outcomes, mode, elapsed))
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Crate reliability / evaluation harness."
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="use the real Claude API (needs ANTHROPIC_API_KEY); default is offline",
    )
    parser.add_argument("-k", type=int, default=5, help="recommendations per case")
    parser.add_argument(
        "--repeats",
        type=int,
        default=CONSISTENCY_REPEATS,
        help="pipeline runs per case for the consistency metrics",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    args = parser.parse_args(argv)

    setup_logging(verbose=args.verbose)
    report = run_evaluation(online=args.online, k=args.k, repeats=args.repeats)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
