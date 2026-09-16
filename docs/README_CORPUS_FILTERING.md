# HeySQuAD validation corpus filtering

This document summarizes the filtering and quality-control procedure used to construct the final evaluation corpus from the HeySQuAD validation data.

## Overview

The corpus preparation and subsequent quality-control history was:

```text
4158 input records
   ↓ answerability filtering
1002 answerable records
   ↓ conservative audio/question mismatch screening
1001 recordings used in the evaluation runs
   ↓ two defective recordings identified during test execution
 999 valid records
```

---

## 1. Answerability filtering
  
**Input rows:** 4158  
**Selection rule:** `is_impossible` is exactly `False`  
**Other selection conditions:** none  
**Random sampling/manual selection:** none  
**Retained rows and WAV files:** 1002  
**Removed rows with `is_impossible=True`:** 3156  

The output CSV preserves the original column order, row values, and relative CSV row order. The filtered audio ZIP follows the CSV row order.

Verification was performed for exact labels, unique IDs, CSV-to-ZIP order, ZIP CRC, and byte-for-byte WAV identity with the source archive.

---

## 2. Question-type categorization of the 1002 answerable records

Question-type categorization was performed as an audit of the answerable subset. It was **not used as a sampling quota or filtering criterion**.

**Input records:** 1002  
**All records satisfy `is_impossible=False`:** yes  

### Method

Questions were tokenized case-insensitively. The first exact interrogative form determined the broad type. Regular contractions were mapped to their base interrogative form, `whence` was mapped to `where`, and `who` / `whom` / `whose` were merged into the `who` category.

`how` questions were additionally split according to the immediately following exact token into `many`, `much`, `long`, `old`, `often`, `far`, or `other`. If no interrogative form was present and the first token was an English auxiliary or modal, the record was categorized as `yes_no`; all remaining forms were assigned to `other`.

### Broad question-type counts

| Type | Count |
|---|---:|
| what | 638 |
| how | 114 |
| who | 84 |
| when | 71 |
| which | 37 |
| where | 29 |
| why | 19 |
| yes/no | 4 |
| other | 6 |
| **Total** | **1002** |

---

## 3. Audio/question mismatch screening

**Input records:** 1002  
**Screening model:** faster-whisper `small.en`  
**Relative phonetic-dominance candidates:** 44  
**Confirmed gross content mismatch excluded:** 1  
**Retained for evaluation runs:** 1001  

This stage was used as a **content-integrity check**, not as a WER or ASR-quality filter.

Every WAV file was independently transcribed using faster-whisper `small.en`. Content tokens were mapped to standard American Soundex codes, with consecutive one-letter tokens first collapsed when they represented a spelled acronym.

A record was marked as a mismatch candidate when the independent transcript shared strictly more distinct Soundex codes with the supplied source `transcription` than with the intended `question`. Exclusion additionally required the two overlap-evidence sets to be exactly disjoint. Ties, intersecting evidence sets, and unresolved cases were retained.

The screen excluded one gross content mismatch:

```text
5729fc3d1d046914007796a0.wav
```

The remaining flagged cases were retained because the observed differences were limited to minor transcription variations, such as acronym spelling, named-entity recognition, identifier formatting, or isolated ASR substitutions, while the intended question content remained unchanged.

---

## 4. Defective recordings identified during test execution

The evaluation runs were configured with **1001 recordings** after the initial mismatch screen. During test execution, two additional defective/truncated recordings were identified:

```text
572a0bfaaf94a219006aa77a.wav
5729e500af94a219006aa6b5.wav
```

These recordings were present in the test configuration and were therefore discovered only after the evaluation campaign had begun.

**Evaluation-run corpus:** 1001 recordings  
**Additional defective recordings discovered during testing:** 2  
**Valid set:** 999 recordings  

---

## Final corpus

**Original validation records:** 4158  
**After answerability filtering:** 1002  
**After pre-run mismatch screening:** 1001  
**Used in the recorded evaluation runs:** 1001  
**Additional defective recordings identified during testing:** 2  
**Valid set:** **999 WAV files / 999 metadata records**

The final corpus thus reflects the original answerability filter, one pre-run gross-mismatch exclusion, and two additional recording defects discovered during the evaluation runs.
