"""Model card (Markdown) and the JSON report."""

from __future__ import annotations

import json
from pathlib import Path

INTENDED_USE = (
    "Research on how coronary artery disease risk models transfer between public cohorts. "
    "Not a diagnostic device and not for decisions about individual patients."
)


def _fmt(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and v != v) else f"{v:.3f}" if isinstance(v, float) else str(v)


def model_card(result: dict, build: dict, synthetic: bool) -> str:
    t, op = result["test"], result["test"]["operating_point"]
    ci = t.get("ci95", {})
    lines = [
        "# Model card: heartstack",
        "",
        "## Intended use",
        INTENDED_USE,
        "",
        "## Data",
        f"- Synthetic data: {'yes (results say nothing about real patients)' if synthetic else 'no'}",
        f"- Rows after de-duplication: {build['rows']} ({build['duplicates_removed']} duplicates removed: "
        f"{build['removed_by_source']})",
    ]
    for s in build["sources"]:
        lines.append(f"- {s['source']}: {s['rows_read']} rows read, zero-as-missing {s['zero_as_missing']}, "
                     f"not-recorded codes {s['not_recorded_codes']}, out of range {s['out_of_range']}")
    lines += [
        "",
        "## Model selection (nested CV on the training set, AUROC)",
        "| Model | Mean | SD |",
        "|---|---|---|",
    ]
    lines += [f"| {m} | {v['mean']:.3f} | {v['sd']:.3f} |" for m, v in result["selection"].items()]
    lines += [
        "",
        f"Selected model: `{result['selected']}`, parameters {result['best_params']}, "
        f"threshold {result['threshold']} (target sensitivity on out-of-fold training predictions).",
        "",
        "## Held-out test set (used one time)",
        f"- n = {t['n']}, prevalence {t['prevalence']:.3f}",
        f"- AUROC {_fmt(t['auroc'])} (95% CI {ci.get('auroc', 'n/a')}), AUPRC {_fmt(t['auprc'])} "
        f"(95% CI {ci.get('auprc', 'n/a')}), Brier {_fmt(t['brier'])}",
        f"- Sensitivity {_fmt(op['sensitivity'])} (95% CI {ci.get('sensitivity', 'n/a')}), specificity "
        f"{_fmt(op['specificity'])} (95% CI {ci.get('specificity', 'n/a')}), PPV {_fmt(op['ppv'])}, NPV {_fmt(op['npv'])}",
        f"- Calibration: ECE {t['calibration']['ece']}, slope {t['calibration']['slope']}, "
        f"intercept {t['calibration']['intercept']}",
        "",
        "## Leave-one-source-out (external validation)",
        "| Held-out source | n | AUROC | ECE | Sensitivity | Specificity | Baseline AUROC |",
        "|---|---|---|---|---|---|---|",
    ]
    for src, v in result["loso"].items():
        base = result.get("loso_baseline", {}).get(src, {}).get("auroc")
        lines.append(f"| {src} | {v['n']} | {_fmt(v['auroc'])} | {_fmt(v['ece'])} | {_fmt(v['sensitivity'])} | "
                     f"{_fmt(v['specificity'])} | {_fmt(base)} |")
    lines += ["", "## Subgroups (test set)", "| Group | n | AUROC | Sensitivity | Specificity |", "|---|---|---|---|---|"]
    for g, v in result["subgroups"].items():
        lines.append(f"| {g} | {v['n']} | {_fmt(v.get('auroc'))} | {_fmt(v.get('sensitivity'))} | "
                     f"{_fmt(v.get('specificity'))} |")
    lines += ["", "## Permutation importance (test set, AUROC drop)"]
    lines += [f"- {k}: {v['mean']:.4f} (sd {v['sd']:.4f})" for k, v in list(result["importance"].items())[:8]]
    lines += [
        "",
        "## Limits",
        "- Retrospective, pooled public data. No prospective or real-time validation.",
        "- The sources differ in population, era and data collection. Read the leave-one-source-out table.",
        "- Missing-value indicators can carry source information. Compare with HEARTSTACK_MISSING_INDICATORS=false.",
        "- A clinician must review every use. The model is not a medical decision tool.",
    ]
    return "\n".join(lines) + "\n"


def write_report(out_dir: Path, result: dict, build: dict, synthetic: bool) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    j = out_dir / "report.json"
    j.write_text(json.dumps({"build": build, "result": result, "synthetic": synthetic}, indent=2, default=str),
                 encoding="utf-8")
    m = out_dir / "model_card.md"
    m.write_text(model_card(result, build, synthetic), encoding="utf-8")
    return j, m
