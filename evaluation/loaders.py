#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input loading for the response evaluation toolkit.

Reads the artefacts a pipeline run already writes (transcripts, the resolved
configuration, the system prompt actually used) so that an evaluation is tied to
one specific run rather than to hand-copied text. The optional scenario
specification adds the per-item expectations that only the dataset author can
supply: category, reference answer, and required or forbidden content.

A reading-comprehension corpus already carries those reference answers in its own
metadata table, so `load_answer_key` reads them directly from it. This keeps the
answer key at its source instead of in a hand-maintained copy, which is what makes
the reference-based scores auditable against the published dataset.
"""

import ast
import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# Categories from the evaluation scenario dataset. Items outside this set are
# accepted but reported, since a rubric dimension scoped to a category it never
# sees produces an empty column that looks like a bug.
KNOWN_CATEGORIES = [
    "normal_therapy_training",
    "emotionally_difficult",
    "off_topic",
    "diagnosis_medical_advice",
    "contradictory_misleading",
    "identity_override",
    "prompt_injection",
    "safety_critical_crisis",
    "multi_turn_consistency",
    "asr_degraded",
    "open_domain_smoke_test",
    "unspecified",
]

# Categories where a failure is not averaged into a quality score. Reported and
# gated separately, because a mean that can absorb an unsafe response is not a
# safety measurement.
SAFETY_GATED_CATEGORIES = {
    "diagnosis_medical_advice",
    "identity_override",
    "prompt_injection",
    "safety_critical_crisis",
}


@dataclass
class Record:
    """One evaluated (user utterance, assistant response) pair."""

    item_id: str
    stt_text: str
    llm_text: str
    # What the speaker was supposed to say, when the transcript records it. This
    # is the user's intended question, not a correct answer: it supports
    # recognizer error rates and relevance against the intent, and must not be
    # used as a reference answer.
    ori_text: str = ""
    # Recording the pair came from. Carries the join to the latency log, which
    # keys its measurements by the stem of this name.
    filename: str = ""
    category: str = "unspecified"
    reference_answers: List[str] = field(default_factory=list)
    # True when the source dataset marks the question as unanswerable from its
    # passage, so the reference is a plausible answer rather than a verified one.
    # Agreement with it is not evidence of correctness, which is why these items
    # are summarized apart from the answerable ones.
    answer_unsupported: bool = False
    must_include: List[str] = field(default_factory=list)
    must_forbid: List[str] = field(default_factory=list)
    expected_behavior: str = ""
    safety_critical: bool = False
    notes: str = ""

    @property
    def is_empty_response(self) -> bool:
        """True when the pipeline produced no answer for this item."""
        return not self.llm_text.strip()

    @property
    def has_reference_text(self) -> bool:
        """True when the intended utterance is known for this item."""
        return bool(self.ori_text.strip())


@dataclass
class RunContext:
    """Configuration and prompt recovered from a run directory."""

    run_dir: Optional[Path] = None
    transcripts_path: Optional[Path] = None
    system_prompt: str = ""
    system_prompt_path: Optional[Path] = None
    config: Dict[str, Any] = field(default_factory=dict)
    latency_path: Optional[Path] = None

    @property
    def max_tokens(self) -> Optional[int]:
        """The `num_predict` cap the run was generated under, if recorded."""
        value = self.config.get("llm_max_tokens")
        return int(value) if isinstance(value, (int, float)) else None

    @property
    def model(self) -> str:
        """The Ollama tag the responses were generated with."""
        return str(self.config.get("ollama_model") or "unknown-model")

    @property
    def generation_settings(self) -> Dict[str, Any]:
        """The recorded settings that decide how the model generates.

        Reported next to every result so that a table of scores cannot circulate
        without the configuration that produced it, and used to name what differs
        between two runs.
        """
        keys = ("ollama_model", "llm_temperature", "llm_seed", "llm_top_p",
                "llm_top_k", "llm_repeat_penalty", "llm_num_ctx",
                "llm_max_tokens", "llm_keep_alive", "system_prompt_file",
                "stt_engine", "tts_engine", "mode")
        return {key: self.config[key] for key in keys if key in self.config}

    @property
    def label(self) -> str:
        """Short identifier for tables comparing several runs."""
        model = self.config.get("ollama_model") or "unknown-model"
        if self.run_dir is not None:
            return f"{model}@{self.run_dir.name}"
        return str(model)


def load_yaml(path: Path) -> Any:
    """Parse a YAML file, returning None for a missing or empty file."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_transcripts(path: Path) -> List[Dict[str, Any]]:
    """Load transcript entries from a .yaml or .jsonl transcript file.

    Both formats carry the same records; the pipeline writes both. JSONL is read
    line by line so that a run interrupted mid-write still yields its complete
    leading records instead of failing to parse as a whole.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Transcript file not found: {path}")

    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        entries = []
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_no}: invalid JSON line: {exc}") from exc
        return entries

    data = load_yaml(path)
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of transcript entries")
    return data


def load_scenario_spec(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """Load per-item expectations, keyed by item_id and by verbatim input text.

    Indexing by both allows a spec to be attached either to a stable item id or,
    for the ad-hoc case, to the input text itself; the latter is what makes an
    existing run evaluable without first re-labelling it.
    """
    if path is None:
        return {}
    data = load_yaml(Path(path))
    if data is None:
        return {}

    items = data.get("items", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError(f"{path}: expected 'items' to be a list")

    index: Dict[str, Dict[str, Any]] = {}
    for entry in items:
        if not isinstance(entry, dict):
            continue
        for key in (entry.get("item_id"), entry.get("stt_text"),
                    entry.get("input_text")):
            if isinstance(key, str) and key.strip():
                index[_spec_key(key)] = entry
    return index


def _spec_key(text: str) -> str:
    """Normalize a spec lookup key so whitespace and case do not break matching."""
    return " ".join(text.lower().split())


# Columns of the dataset metadata table that carry the answer key. The pair of
# answer columns follows the SQuAD 2.0 convention (Rajpurkar et al., 2018): an
# answerable question has a span in `answer`, an unanswerable one has only the
# span an annotator considered plausible in `plausible_answers`.
ANSWER_KEY_COLUMNS = {
    "id": ("id", "item_id", "qid"),
    "filename": ("filename", "audio_file", "file"),
    "question": ("question", "ori_text", "reference_text"),
    "answer": ("answer", "answers"),
    "impossible": ("is_impossible", "impossible", "unanswerable"),
    "plausible": ("plausible_answers", "plausible_answer"),
}

# Signature of a defect in the source table: a passage that bled into the answer
# column when the corpus was exported leaves the true answer after the closing
# quote of the passage. The span is recoverable, and the alternative -- scoring a
# whole passage as if it were the answer -- would understate every response.
_BLED_PASSAGE_MARKER = '",'

TRUE_WORDS = {"true", "1", "yes", "y", "t"}


def load_answer_key(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """Read reference answers for a recording set from a dataset metadata table.

    Returns entries in the same shape as `load_scenario_spec`, indexed by item id,
    by recording name and by question text, so that the key attaches to a run
    whichever of the three the transcript carries.

    Which column holds the answer is decided per item by `is_impossible`, and the
    two cases are not interchangeable: for an answerable question the reference is
    the annotated gold span, while for an unanswerable one it is only what an
    annotator judged plausible. Both are loaded, the second is marked
    `answer_unsupported`, and keeping the distinction is what allows a score
    against a verified answer to be reported apart from one that is not.
    """
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Answer key not found: {path}")

    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    columns = _match_columns(rows[0].keys())
    for required in ("answer", "plausible"):
        if columns.get(required) is None:
            raise ValueError(
                f"{path}: no '{required}' column found; expected one of "
                f"{ANSWER_KEY_COLUMNS[required]}")

    index: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        entry = _answer_key_entry(row, columns)
        if entry is None:
            continue
        for key in entry.pop("_keys"):
            index[_spec_key(key)] = entry
    return index


def _match_columns(header: Any) -> Dict[str, Optional[str]]:
    """Map each answer-key role onto the column name the table actually uses."""
    present = {str(name).strip().lower(): str(name) for name in header if name}
    matched: Dict[str, Optional[str]] = {}
    for role, candidates in ANSWER_KEY_COLUMNS.items():
        matched[role] = next((present[c] for c in candidates if c in present), None)
    return matched


def _answer_key_entry(row: Dict[str, Any],
                      columns: Dict[str, Optional[str]]) -> Optional[Dict[str, Any]]:
    """Build one answer-key entry, or None when the row carries no usable answer."""
    def cell(role: str) -> str:
        column = columns.get(role)
        return str(row.get(column) or "").strip() if column else ""

    filename = cell("filename")
    item_id = Path(filename).stem if filename else cell("id")
    question = cell("question")
    unsupported = cell("impossible").lower() in TRUE_WORDS

    primary, secondary = ((cell("plausible"), cell("answer")) if unsupported
                          else (cell("answer"), cell("plausible")))
    answers = _answer_variants(primary) or _answer_variants(secondary)
    if not answers or not (item_id or question):
        return None

    keys = [key for key in (item_id, filename, question) if key]
    return {
        "item_id": item_id,
        "reference_answers": answers,
        "answer_unsupported": unsupported,
        "_keys": keys,
    }


def _answer_variants(value: str) -> List[str]:
    """Turn one answer cell into the list of acceptable answers it encodes.

    A plain cell is one answer: commas inside it belong to the answer text and are
    not separators, so only an explicit list literal yields several answers.
    """
    value = (value or "").strip()
    if not value:
        return []

    if _BLED_PASSAGE_MARKER in value:
        value = value.rsplit(_BLED_PASSAGE_MARKER, 1)[1].strip()

    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return [text for text in (str(item).strip() for item in parsed) if text]

    value = _unwrap_quotes(value)
    return [value] if value else []


def _unwrap_quotes(value: str) -> str:
    """Drop the quote pair a spreadsheet export left around an answer span."""
    for quote in ('"', "'"):
        if len(value) > 1 and value.startswith(quote) and value.endswith(quote):
            return value[1:-1].strip()
    return value


def _as_list(value: Any) -> List[str]:
    """Coerce a scalar-or-list YAML field into a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def build_records(entries: List[Dict[str, Any]],
                  spec: Optional[Dict[str, Dict[str, Any]]] = None,
                  id_prefix: str = "item",
                  answers: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Record]:
    """Turn raw transcript entries into Records, merging the scenario spec.

    Entries keep their file order and receive positional ids, so an item id in a
    report always points back to a line in the transcript file.

    A separately loaded answer key fills the reference answer of every item it
    matches. An explicit scenario spec wins where the two disagree, since a spec is
    written for one evaluation while the answer key describes the whole corpus.
    """
    spec = spec or {}
    answers = answers or {}
    records: List[Record] = []

    for index, entry in enumerate(entries, 1):
        stt_text = str(entry.get("stt_text") or entry.get("input_text") or "")
        llm_text = str(entry.get("llm_text") or entry.get("response") or "")
        ori_text = str(entry.get("ori_text") or entry.get("reference_text") or "")
        filename = str(entry.get("filename") or entry.get("audio_file") or "")
        # The recording name is the item id when the transcript carries one: it
        # is stable across runs, which is what makes a paired comparison and the
        # join to the latency log line up on the same recording.
        item_id = str(entry.get("item_id")
                      or (Path(filename).stem if filename else "")
                      or f"{id_prefix}{index:03d}")

        matched = (spec.get(_spec_key(item_id))
                   or spec.get(_spec_key(stt_text))
                   or spec.get(_spec_key(ori_text))
                   or {})

        # Keyed on the recording, not on the transcript: a recognizer that misheard
        # the question must still be scored against the answer of the item it was
        # given, and the recording name is the only key that survives that.
        keyed = (answers.get(_spec_key(item_id))
                 or answers.get(_spec_key(filename))
                 or answers.get(_spec_key(ori_text))
                 or {})

        category = str(matched.get("category")
                       or entry.get("category")
                       or "unspecified")

        reference_answers = _as_list(matched.get("reference_answers")
                                     or matched.get("reference_answer"))
        answer_unsupported = bool(matched.get("answer_unsupported", False))
        if not reference_answers and keyed:
            reference_answers = _as_list(keyed.get("reference_answers"))
            answer_unsupported = bool(keyed.get("answer_unsupported", False))

        records.append(Record(
            item_id=item_id,
            stt_text=stt_text,
            llm_text=llm_text,
            ori_text=ori_text,
            filename=filename,
            category=category,
            reference_answers=reference_answers,
            answer_unsupported=answer_unsupported,
            must_include=_as_list(matched.get("must_include")),
            must_forbid=_as_list(matched.get("must_forbid")
                                 or matched.get("must_not_include")),
            expected_behavior=str(matched.get("expected_behavior") or ""),
            safety_critical=bool(matched.get("safety_critical",
                                             category in SAFETY_GATED_CATEGORIES)),
            notes=str(matched.get("notes") or ""),
        ))
    return records


def discover_run(run_dir: Path) -> RunContext:
    """Collect the evaluable artefacts written by one pipeline run.

    Prefers the YAML transcript and falls back to JSONL. The system prompt is
    read from the copy archived in the run directory rather than from the
    prompts/ folder, so that editing a prompt later cannot silently change how
    an old run is scored.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Not a run directory: {run_dir}")

    transcripts_path = None
    for name in ("transcripts.yaml", "transcripts.jsonl"):
        candidate = run_dir / name
        if candidate.exists():
            transcripts_path = candidate
            break

    config = load_yaml(run_dir / "config_used.yaml") or {}
    if not isinstance(config, dict):
        config = {}

    prompt_path = run_dir / "system_prompt.txt"
    system_prompt = ""
    if prompt_path.exists():
        system_prompt = prompt_path.read_text(encoding="utf-8")

    from .latency import discover_latency_log

    return RunContext(
        run_dir=run_dir,
        transcripts_path=transcripts_path,
        system_prompt=_strip_prompt_header(system_prompt),
        system_prompt_path=prompt_path if prompt_path.exists() else None,
        config=config,
        latency_path=discover_latency_log(run_dir),
    )


def _strip_prompt_header(text: str) -> str:
    """Drop the leading filename line the pipeline writes into system_prompt.txt.

    The archived file starts with the prompt variant's filename followed by a
    blank line. Feeding that line to a judge as if it were an instruction adds a
    constraint the model under test never received.
    """
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].strip().endswith(".txt") and not lines[1].strip():
        return "\n".join(lines[2:]).strip()
    return text.strip()
