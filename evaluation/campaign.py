#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a whole outputs tree of split ASR / text / realtime campaigns.

The measurement campaign writes several sibling directories under `outputs/`:

    asr-heyval-vosk / asr-heyval-whisper
        recognition-only, WER and endpoint delay
    text-heyval-vosk / text-heyval-whisper
        LLM cells on frozen transcripts, three interleaved launches
    text-heyval-vosk-quant / text-heyval-whisper-quant
        extra quantization ladders on families that were only Q4 in the factorial
    ttfa-validation-heyval
        realtime measured TTFA vs reconstruction, 200-item subset

This module walks that tree, scores every run with the same instruments, and
writes one archive under `--out-dir` (defaults to `outputs_evaluations/`).

    python -m evaluation.campaign --outputs ..\\outputs --out-dir ..\\outputs_evaluations

Tiers 0 and 1 plus ASR fidelity and latency always run. Judge scoring is a
separate, opt-in round (`--judge`): a stratified 100-item sample, launch r1
only, one local Ollama model, and the open-domain spoken-QA rubric. Human
rating and MOS / listening tests are not part of this campaign.

Known corpus defects (mismatched wav and metadata) are excluded by default.
Full-corpus reference answers are used when `--answer-key` points at the
HeySQuAD metadata table; otherwise the 200-item subset table is applied to
the overlapping recordings only, and the rest stay without answer-presence.
"""

import argparse
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
import sys
from typing import Callable, Dict, List, Optional, Sequence, Union

from . import reporting
from .batch import BatchConfig, BatchOutcome, GROUP_KINDS, run_batch
from .pipeline import DEFAULT_RUBRIC, EvaluationConfig
from .sampling import (sample_from_reference_csv, stems_of, stratum_counts,
                       write_sample_csv)
from .ttfa_validation import write_agreement

PathLike = Union[str, Path]

# Recordings whose wav and metadata row disagree (notes1.md). Scoring them
# would attribute a truncated utterance to the recognizer.
DEFAULT_EXCLUDE_ITEMS = [
    "572a0bfaaf94a219006aa77a",
    "5729e500af94a219006aa6b5",
]

# LLM-as-judge round. One local model, one launch, stratified sample.
# qwen2.5:7b is in the factorial (cells 15-16); self-preference on those
# cells is a reported limitation, not a silent default of a stronger judge.
DEFAULT_JUDGE_MODEL = "qwen2.5:7b-instruct"
DEFAULT_JUDGE_N = 100
DEFAULT_JUDGE_LAUNCH = "r1"
DEFAULT_JUDGE_CAMPAIGN = "text-heyval-vosk"
JUDGE_RESULT_DIR = "judge-open-domain"

# Campaign directories under outputs/, in reporting order.
CAMPAIGNS = [
    ("asr-heyval-vosk", "ASR-only, Vosk CPU, 1001 HeySQuAD recordings"),
    ("asr-heyval-whisper", "ASR-only, Whisper GPU, same recordings"),
    ("text-heyval-vosk", "LLM factorial on frozen Vosk transcripts, 3 launches"),
    ("text-heyval-vosk-quant", "Extra Vosk quantization ladder (7b/4b Q8/F16)"),
    ("text-heyval-whisper", "LLM factorial on frozen Whisper transcripts, 3 launches"),
    ("text-heyval-whisper-quant", "Extra Whisper quantization ladder"),
    ("ttfa-validation-heyval", "Realtime measured TTFA vs reconstruction, 200-item subset"),
]


@dataclass
class CampaignConfig:
    """What to evaluate in one pass over an outputs tree."""

    outputs: PathLike
    out_dir: PathLike
    answer_key: Optional[PathLike] = None
    exclude_items: List[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE_ITEMS))
    campaigns: Optional[List[str]] = None
    evaluation: Optional[EvaluationConfig] = None
    alpha: float = 0.05
    n_boot: int = 2000
    seed: int = 0
    progress: Optional[Callable[[str], None]] = None
    # LLM-as-judge round. Empty judge_models skips it.
    judge_models: List[str] = field(default_factory=list)
    judge_n: int = DEFAULT_JUDGE_N
    judge_launch: str = DEFAULT_JUDGE_LAUNCH
    judge_campaign: str = DEFAULT_JUDGE_CAMPAIGN
    judge_url: Optional[str] = None
    judge_timeout: float = 300.0
    judge_samples: int = 1
    skip_metric_campaigns: bool = False

    def __post_init__(self):
        self.outputs = Path(self.outputs)
        self.out_dir = Path(self.out_dir)
        if self.answer_key is not None:
            self.answer_key = Path(self.answer_key)

    def notify(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)


def discover_answer_key(outputs: Path, explicit: Optional[Path]) -> Optional[Path]:
    """Prefer an explicit table, then the full HeyVal key, then the 200-item subset."""
    if explicit is not None and Path(explicit).exists():
        return Path(explicit)
    candidates = [
        outputs / "metadata.normalized.csv",
        outputs / "heyval" / "metadata.csv",
        outputs / "ttfa-validation-heyval" / "subset" / "metadata.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _reference_table(outputs: Path, out_root: Path) -> Path:
    """Recognizer table used to stratify the judge sample.

    Prefer the already-evaluated Vosk reference CSV so the sample is defined
    on the same WER strata the reports use. Fall back to the Whisper table
    only if Vosk was not evaluated.
    """
    candidates = [
        out_root / "asr-heyval-vosk" / "input_reference.csv",
        outputs / "asr-heyval-vosk" / "input_reference.csv",
        out_root / "asr-heyval-whisper" / "input_reference.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "no input_reference.csv found under asr-heyval-vosk; run the metric "
        "campaign before the judge round")


def run_judge_round(config: CampaignConfig) -> Path:
    """Score a stratified sample of one launch with the open-domain rubric.

    Writes ``<out-dir>/judge-open-domain/``: the sample table, a protocol
    note, and the batch evaluation of the selected campaign's r1 cells.
    MOS and human ratings are not collected.
    """
    outputs = Path(config.outputs)
    out_root = Path(config.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    target = out_root / JUDGE_RESULT_DIR
    target.mkdir(parents=True, exist_ok=True)

    campaign_name = config.judge_campaign or DEFAULT_JUDGE_CAMPAIGN
    source = outputs / campaign_name
    if not source.is_dir():
        raise FileNotFoundError(f"judge campaign not found: {source}")

    reference = _reference_table(outputs, out_root)
    rows = sample_from_reference_csv(
        reference,
        n=config.judge_n,
        seed=config.seed,
        exclude=config.exclude_items)
    if not rows:
        raise ValueError("judge sample is empty after exclusions")

    write_sample_csv(target / "sample.csv", rows)
    stems = stems_of(rows)
    counts = stratum_counts(rows)
    models = list(config.judge_models) or [DEFAULT_JUDGE_MODEL]
    launch = config.judge_launch or DEFAULT_JUDGE_LAUNCH

    protocol = _render_judge_protocol(
        campaign_name=campaign_name,
        launch=launch,
        models=models,
        n=len(rows),
        counts=counts,
        reference=reference,
        seed=config.seed,
        exclude=config.exclude_items or [])
    reporting.write_text(target / "protocol.txt", protocol)
    config.notify(
        f"judge sample n={len(rows)} from {reference.name}: "
        + ", ".join(f"{name} {count}" for name, count in sorted(counts.items())))

    answer_key = discover_answer_key(outputs, config.answer_key)
    base = config.evaluation or EvaluationConfig()
    base = replace(
        base,
        answer_key=answer_key or base.answer_key,
        exclude_items=list(config.exclude_items or []),
        include_items=stems,
        judge_models=models,
        judge_samples=config.judge_samples,
        judge_timeout=config.judge_timeout,
        rubric=DEFAULT_RUBRIC,
        seed=config.seed,
        progress=config.progress)
    if config.judge_url:
        base = replace(base, judge_url=config.judge_url)

    batch = run_batch(BatchConfig(
        root=_batch_root(source),
        out_dir=target,
        evaluation=base,
        alpha=config.alpha,
        n_boot=config.n_boot,
        seed=config.seed,
        group_kinds=_group_kinds_for(campaign_name),
        launch_ids=[launch],
        progress=config.progress))
    batch.write(target)

    notes = [
        f"campaign {campaign_name} launch {launch}: {len(batch.runs)} cell(s)",
        f"sample n={len(rows)} from {reference}",
        "strata: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        f"judge model(s): {', '.join(models)}",
        "rubric: quest_open_domain_spoken_v1",
        "human rating: not collected",
        "MOS / listening test: not collected",
    ]
    for warning in batch.warnings:
        notes.append(warning)
    reporting.write_text(target / "judge_notes.txt",
                         "\n".join(f"- {line}" for line in notes) + "\n")
    config.notify(f"judge round -> {target}")
    return target


def _render_judge_protocol(campaign_name: str, launch: str, models: Sequence[str],
                           n: int, counts: Dict[str, int], reference: Path,
                           seed: int, exclude: Sequence[str]) -> str:
    lines = [
        "=" * 96,
        "ETAP LLM-AS-JUDGE ROUND".center(96),
        "=" * 96,
        f"Generated on           : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Campaign               : {campaign_name}",
        f"Launch                 : {launch} (system replicate; r2/r3 omitted "
        "because quality metrics were bit-identical)",
        f"Judge model(s)         : {', '.join(models)}",
        f"Rubric                 : quest_open_domain_spoken_v1",
        f"Sample size            : {n}",
        f"Sample seed            : {seed}",
        f"Strata source          : {reference}",
        "Stratum mix            : "
        + ", ".join(f"{name} {count}" for name, count in sorted(counts.items())),
        "Excluded defects       : " + (", ".join(exclude) if exclude else "(none)"),
        "",
        "Method",
        "-" * 96,
        "  Form     : G-Eval (Liu et al., 2023) + Prometheus 2 (Kim et al., 2024).",
        "  Status   : established screening, not a validated instrument.",
        "  Sampling : proportional to Vosk WER strata (clean / mild / severe)",
        "             by the largest-remainder method; items drawn without",
        "             replacement inside each stratum.",
        "  Prompt   : recognized utterance and intended utterance are both shown,",
        "             so intent_contact can be scored against what was asked.",
        "",
        "What these scores may support",
        "-" * 96,
        "  Rank order of configurations on this sample.",
        "  Whether that order agrees with verifiable prompt-adherence.",
        "",
        "What these scores must not be claimed as",
        "-" * 96,
        "  A calibrated quality percentage or a human Mean Opinion Score.",
        "  Therapeutic suitability, clinical safety, or empathy.",
        "  Perceptual TTS quality (no listening test was run).",
        "  A result on the full 999-item factorial; that table stays IFEval/WER.",
        "",
        "Self-preference",
        "-" * 96,
        "  The default judge (qwen2.5:7b-instruct) is in the factorial as the",
        "  7B Qwen cells. Scores for those cells may be inflated relative to",
        "  other families (Zheng et al., 2023). Do not treat a 7B-Qwen win",
        "  against a different family as decisive on this judge alone.",
        "=" * 96,
        "",
    ]
    return "\n".join(lines)


def run_campaign(config: CampaignConfig) -> Dict[str, Path]:
    """Evaluate every present campaign directory. Returns name -> result dir."""
    outputs = Path(config.outputs)
    if not outputs.is_dir():
        raise NotADirectoryError(f"Not a directory: {outputs}")

    out_root = Path(config.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Path] = {}
    notes: List[str] = []

    if config.skip_metric_campaigns:
        notes.append("metric campaigns skipped (--judge); existing tables left in place")
    else:
        written, notes = _run_metric_campaigns(config)

    if config.judge_models:
        config.notify("LLM-as-judge round")
        written[JUDGE_RESULT_DIR] = run_judge_round(config)
        notes.append(f"judge round -> {written[JUDGE_RESULT_DIR]}")

    answer_key = discover_answer_key(outputs, config.answer_key)
    index = _render_index(outputs, out_root, written, notes, answer_key)
    reporting.write_text(out_root / "campaign_index.txt", index)
    return written


def _run_metric_campaigns(config: CampaignConfig
                          ) -> tuple:
    """Original factorial / ASR / TTFA evaluation pass."""
    outputs = Path(config.outputs)
    out_root = Path(config.out_dir)
    answer_key = discover_answer_key(outputs, config.answer_key)
    wanted = set(config.campaigns) if config.campaigns else {name for name, _ in CAMPAIGNS}

    base = config.evaluation or EvaluationConfig()
    base = replace(base,
                   answer_key=answer_key or base.answer_key,
                   exclude_items=list(config.exclude_items or []),
                   seed=config.seed,
                   progress=config.progress)

    written: Dict[str, Path] = {}
    notes: List[str] = []
    if answer_key is None:
        notes.append(
            "No HeySQuAD metadata table was found. Reference-span metrics "
            "(answer_presence, token F1, ROUGE) are off. Pass --answer-key "
            "pointing at audios/heyval/metadata.csv to enable them on the "
            "full 1001-item set. The 200-item subset table was also absent.")
    elif "subset" in str(answer_key).replace("\\", "/"):
        notes.append(
            f"Answer key is the 200-item subset ({answer_key}). Overlapping "
            "recordings receive reference-span scores; the remaining ~800 "
            "items of the 1001-item arms do not. A full-corpus metadata.csv "
            "is required before answer_presence can be claimed on the factorial.")
    else:
        notes.append(f"Answer key is the full corpus table ({answer_key}).")

    if config.exclude_items:
        notes.append(
            "Excluded corpus defects (notes1.md): "
            + ", ".join(config.exclude_items))

    for name, description in CAMPAIGNS:
        if name not in wanted:
            continue
        source = outputs / name
        if not source.exists():
            notes.append(f"skipped {name}: directory not present")
            continue

        target = out_root / name
        config.notify(f"campaign {name}: {description}")

        if name == "ttfa-validation-heyval":
            written[name] = _run_ttfa_campaign(
                source, target, base, config, notes)
            continue

        batch = run_batch(BatchConfig(
            root=_batch_root(source),
            out_dir=target,
            evaluation=base,
            alpha=config.alpha,
            n_boot=config.n_boot,
            seed=config.seed,
            group_kinds=_group_kinds_for(name),
            progress=config.progress))
        batch.write(target)
        written[name] = target
        notes.append(
            f"{name}: {len(batch.runs)} launch(es), "
            f"{len(batch.groups)} contrast(s) -> {target}")
    return written, notes


def _batch_root(source: Path) -> Path:
    """Point batch discovery at the directory that actually holds runs.

    The TTFA validation campaign nests realtime runs under `measured/`;
    everything else holds cells directly.
    """
    measured = source / "measured"
    return measured if measured.is_dir() else source


def _group_kinds_for(name: str) -> List[str]:
    """ASR-only trees have no model contrast; skip those groups."""
    if name.startswith("asr-"):
        return ["recognizer"]
    if "quant" in name:
        return ["quantization", "cross_model"]
    return list(GROUP_KINDS)


def _run_ttfa_campaign(source: Path, target: Path, base: EvaluationConfig,
                       config: CampaignConfig, notes: List[str]) -> Path:
    """Score the realtime subset runs and the reconstruction-agreement table."""
    target.mkdir(parents=True, exist_ok=True)
    measured = source / "measured"
    if measured.is_dir():
        # Subset metadata is the right answer key for this 200-item arm.
        subset_key = source / "subset" / "metadata.csv"
        settings = replace(base, answer_key=subset_key if subset_key.exists()
                           else base.answer_key)
        batch = run_batch(BatchConfig(
            root=measured,
            out_dir=target / "measured",
            evaluation=settings,
            alpha=config.alpha,
            n_boot=config.n_boot,
            seed=config.seed,
            group_kinds=["cross_model"],
            progress=config.progress))
        batch.write(target / "measured")
        notes.append(
            f"ttfa-validation measured: {len(batch.runs)} launch(es) -> "
            f"{target / 'measured'}")

    table = source / "measured_vs_reconstructed.csv"
    if table.exists():
        write_agreement(table, target / "reconstruction_agreement",
                        exclude_stems=config.exclude_items)
        notes.append(
            f"ttfa-validation reconstruction agreement -> "
            f"{target / 'reconstruction_agreement'}")
    else:
        notes.append("ttfa-validation: measured_vs_reconstructed.csv missing")
    return target


def _render_index(outputs: Path, out_root: Path, written: Dict[str, Path],
                  notes: List[str], answer_key: Optional[Path]) -> str:
    lines = [
        "=" * 96,
        "ETAP CAMPAIGN EVALUATION".center(96),
        "=" * 96,
        f"Generated on           : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Outputs tree           : {outputs}",
        f"Results written to     : {out_root}",
        f"Answer key             : {answer_key or '(none)'}",
        f"Campaigns completed    : {len(written)}",
        "",
        "What each level of the evaluation can support",
        "-" * 96,
        "  Item level     : WER, prompt adherence, coverage, answer presence.",
        "                   Paired across configurations on the same recording.",
        "                   Status: verifiable / established (see method list).",
        "  Launch level   : independent interleaved rounds of the same cell.",
        "                   The experimental unit for a configuration claim.",
        "                   ICC(2,1) and launch-range; status: validated (ICC).",
        "  Cell level     : the configuration (model, precision, temperature,",
        "                   recognizer). Contrasts hold every other factor fixed.",
        "  Reconstruction : Bland-Altman of measured vs reconstructed TTFA.",
        "                   Status: validated (Bland and Altman, 1986).",
        "  Judge sample   : open-domain rubric on a stratified 100-item draw,",
        "                   launch r1. Status: established screening (G-Eval).",
        "",
        "What this evaluation does not support, and must not be claimed",
        "-" * 96,
        "  Therapist supervision, clinical safety, or therapeutic suitability:",
        "  the corpus is open-domain read-aloud HeySQuAD questions.",
        "  Perceptual TTS quality (MOS / P.808): synthesis time is measured;",
        "  listening tests were not collected and are not planned.",
        "  Human or expert response quality: Tier 4 was not run and is not planned.",
        "  A calibrated quality percentage from the judge: screening only.",
        "  A universal quantization result: Q4/Q8/F16 is compared per family",
        "  that actually has those cells, on this host and this task.",
        "",
        "Campaigns",
        "-" * 96,
    ]
    listed = list(CAMPAIGNS) + [
        (JUDGE_RESULT_DIR,
         "LLM-as-judge, open-domain rubric, stratified sample, r1"),
    ]
    for name, description in listed:
        if name in written:
            status = str(written[name])
        elif (out_root / name).exists():
            status = f"{out_root / name} (already present)"
        else:
            status = "not run"
        lines.append(f"  {name:<32} {description}")
        lines.append(f"  {'':<32} -> {status}")
    if notes:
        lines += ["", "Notes", "-" * 96]
        for note in notes:
            lines.append(f"  - {note}")
    lines += [
        "",
        "How to re-run one campaign",
        "-" * 96,
        "  python -m evaluation.campaign --outputs outputs "
        "--out-dir outputs_evaluations --campaign text-heyval-vosk",
        "  python -m evaluation.campaign --outputs outputs "
        "--out-dir outputs_evaluations --judge",
        "  python -m evaluation.batch --root outputs/text-heyval-vosk "
        "--out-dir outputs_evaluations/text-heyval-vosk",
        "=" * 96,
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.campaign",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs", type=Path, required=True,
                        help="The outputs tree written by the measurement campaign")
    parser.add_argument("--out-dir", type=Path,
                        help="Where to write evaluations (default: <outputs>/../outputs_evaluations)")
    parser.add_argument("--answer-key", type=Path,
                        help="Full HeySQuAD metadata.csv. Defaults to the 200-item subset table if present.")
    parser.add_argument("--campaign", action="append", default=[],
                        help="Restrict to one campaign directory name. Repeatable.")
    parser.add_argument("--exclude-item", action="append", default=[],
                        help="Additional recording stem to drop. Repeatable.")
    parser.add_argument("--no-default-exclusions", action="store_true",
                        help="Do not drop the two known mismatched recordings.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--judge", action="store_true", help=(
        "Run the LLM-as-judge round instead of re-scoring the metric "
        "campaigns. Stratified sample, launch r1, open-domain rubric. "
        "Requires a reachable local Ollama server."))
    parser.add_argument("--judge-model", action="append", default=[], help=(
        f"Ollama tag of the judge. Repeatable. Default: {DEFAULT_JUDGE_MODEL}."))
    parser.add_argument("--judge-n", type=int, default=DEFAULT_JUDGE_N, help=(
        "Stratified sample size (default: %(default)s)."))
    parser.add_argument("--judge-launch", type=str, default=DEFAULT_JUDGE_LAUNCH,
                        help="Which independent launch to grade (default: %(default)s).")
    parser.add_argument("--judge-campaign", type=str, default=DEFAULT_JUDGE_CAMPAIGN,
                        help="LLM campaign directory to grade (default: %(default)s).")
    parser.add_argument("--judge-url", type=str, help="Ollama generate endpoint.")
    parser.add_argument("--judge-timeout", type=float, default=300.0)
    parser.add_argument("--judge-samples", type=int, default=1)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .cli import _stderr_progress, _use_utf8_output

    _use_utf8_output()
    args = build_parser().parse_args(argv)

    exclude = list(args.exclude_item)
    if not args.no_default_exclusions:
        exclude = list(DEFAULT_EXCLUDE_ITEMS) + exclude

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = args.outputs.parent / "outputs_evaluations"

    judge_models: List[str] = []
    skip_metrics = False
    if args.judge:
        judge_models = list(args.judge_model) or [DEFAULT_JUDGE_MODEL]
        skip_metrics = True

    config = CampaignConfig(
        outputs=args.outputs,
        out_dir=out_dir,
        answer_key=args.answer_key,
        exclude_items=exclude,
        campaigns=list(args.campaign) or None,
        alpha=args.alpha,
        n_boot=args.n_boot,
        seed=args.seed,
        progress=None if args.quiet else _stderr_progress,
        judge_models=judge_models,
        judge_n=args.judge_n,
        judge_launch=args.judge_launch,
        judge_campaign=args.judge_campaign,
        judge_url=args.judge_url,
        judge_timeout=args.judge_timeout,
        judge_samples=args.judge_samples,
        skip_metric_campaigns=skip_metrics)

    try:
        written = run_campaign(config)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        index = Path(config.out_dir) / "campaign_index.txt"
        if index.exists():
            print(index.read_text(encoding="utf-8"))
    print(f"\nCampaigns written: {len(written)}", file=sys.stderr)
    print(f"Index: {Path(config.out_dir).resolve() / 'campaign_index.txt'}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
