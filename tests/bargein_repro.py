#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Why both response WAVs survive a barge-in.

The audit finding this re-checks: when the endpointer fires a second time,
llm_worker puts `end_of_response` at the end of one iteration and the `cancel`
only at the top of the next. The TTS queue is FIFO, so by the time the cancel is
read, close_response() has already advanced response_index; the cancel then
computes a partial path for an `_r2.wav` that was never opened, os.path.exists
is False, and the finished WAV of the answer nobody heard is not deleted.

Run this after touching llm_worker, tts_worker or the queue between them. It
does not assume the message sequence -- it observes it, by running the real
llm_worker against a recording queue -- and then replays exactly that sequence
into the real tts_worker.

Everything happens in a temporary directory of its own, printed below and
removed on the way out, because response WAVs are named after the item and an
earlier version of this harness cleaned up by deleting *.wav in the working
directory.

Exit status is 0 while the finding holds and 1 once it does not.

Run:  python tests/bargein_repro.py
"""

import os
import queue
import shutil
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import assistant
from pipeline_channels import UtteranceMailbox
# The same scripted engines the accounting test uses, so the two cannot drift
# into disagreeing about what the pipeline was fed.
from test_bargein_accounting import (
    ScriptedLLM, ScriptedTTS, TTS_AUDIO_S, TTS_SYNTH_S,
)

BARGE_IN_AFTER_S = 0.5   # long enough for the first answer to reach the TTS


class RecordingQueue(queue.Queue):
    """A tts_queue that keeps what was put on it, in order."""

    def __init__(self):
        super().__init__()
        self.log = []

    def put(self, item, *args, **kwargs):
        self.log.append(item)
        super().put(item, *args, **kwargs)


def observe_llm_worker_sequence():
    """Run the real llm_worker through a barge-in and return what it enqueued."""
    recorded = RecordingQueue()
    mailbox = UtteranceMailbox()
    llm_metrics = {
        "llm_t0": None, "llm_ttfc_ms": None, "llm_ttft_ms": None,
        "first_chunk_chars": None, "full_assistant_text": "",
        "llm_chunk_count": 0, "ollama_stats": None,
        "ttft_stats": None, "ttfc_stats": None, "end_stats": None,
    }
    worker = threading.Thread(
        target=assistant.llm_worker,
        args=(ScriptedLLM(), mailbox, recorded, llm_metrics),
        daemon=True,
    )
    worker.start()
    mailbox.put("what did he")
    time.sleep(BARGE_IN_AFTER_S)
    mailbox.put("what did he inherit")
    # Closed while the second utterance is still unread, which is what the STT
    # worker does at the end of a file. take() hands over a pending utterance
    # before it reports the close, so the barge-in still happens; waiting for
    # the worker first would only wait for a take() that cannot return.
    mailbox.close()
    worker.join(timeout=30)
    if worker.is_alive():
        raise SystemExit("llm_worker did not finish; nothing below can be read")
    return recorded.log


def describe(message):
    if message is None:
        return "None (EOF)"
    kind = message.get("type", "text")
    return kind if kind != "text" else f"text {message.get('text')!r}"


def replay_into_tts_worker(sequence, work_dir):
    """Feed that sequence to the real tts_worker and report what it left."""
    tts_queue = queue.Queue()
    for message in sequence:
        tts_queue.put(message)
    if sequence and sequence[-1] is not None:
        tts_queue.put(None)

    out_wav = os.path.join(work_dir, "assistant_1_repro.wav")
    tts_metrics = {
        "tts_first_chunk_ms": None, "tts_first_chunk_t": None,
        "total_tts_time": 0.0, "first_chunk_stats": None, "end_stats": None,
    }
    assistant.tts_worker(ScriptedTTS(), tts_queue, out_wav, tts_metrics)
    return tts_metrics


def main():
    work_dir = tempfile.mkdtemp(prefix="bargein-repro-")
    print(f"[INFO] Working in {work_dir}")
    try:
        sequence = observe_llm_worker_sequence()
        print("\nWhat llm_worker enqueued:")
        for position, message in enumerate(sequence):
            print(f"  {position:2d}  {describe(message)}")

        kinds = [m.get("type", "text") for m in sequence if m is not None]
        adjacent = [(a, b) for a, b in zip(kinds, kinds[1:])]
        race_present = ("end_of_response", "cancel") in adjacent
        # How many chunks belong to each answer, taken from the sequence rather
        # than from the script, so the expected figures below are derived from
        # what the worker was actually given.
        after_last_cancel = len(kinds) - 1 - kinds[::-1].index("cancel")
        survived = kinds[after_last_cancel:].count("text")
        superseded = kinds[:after_last_cancel].count("text")

        tts_metrics = replay_into_tts_worker(sequence, work_dir)
        on_disk = sorted(f for f in os.listdir(work_dir) if f.endswith(".wav"))
        surviving = [os.path.basename(p) for p in tts_metrics["out_wavs"]]
        discarded = [os.path.basename(p) for p in tts_metrics["discarded_wavs"]]
        synthesis_ms = tts_metrics["total_tts_time"] * 1000
        discarded_ms = sum(assistant.wav_duration_ms(p)
                           for p in tts_metrics["discarded_wavs"])

        print("\nWhat tts_worker left behind:")
        print(f"  on disk         {on_disk}")
        print(f"  out_wavs        {surviving}")
        print(f"  discarded_wavs  {discarded}")
        print(f"  total_tts_time  {synthesis_ms:.0f} ms "
              f"({survived} surviving chunks, {superseded} superseded)")
        print(f"  discarded audio {discarded_ms} ms")

        checks = [
            ("cancel arrives directly after end_of_response, so the delete "
             "looks for a WAV that was never opened", race_present),
            ("both response WAVs are still on disk", len(on_disk) == 2),
            ("only the surviving answer is in out_wavs", len(surviving) == 1),
            ("the superseded answer is accounted for separately",
             len(discarded) == 1),
            (f"total_tts_time covers the {survived} surviving chunks and not "
             f"the {superseded} superseded ones",
             abs(synthesis_ms - survived * TTS_SYNTH_S * 1000)
             <= 0.2 * survived * TTS_SYNTH_S * 1000),
            (f"the discarded WAV holds the {superseded} superseded chunks",
             discarded_ms == round(superseded * TTS_AUDIO_S * 1000)),
        ]
        print()
        for description, held in checks:
            print(f"  [{'ok  ' if held else 'FAIL'}] {description}")

        if all(held for _, held in checks):
            print("\nThe finding holds: a barge-in leaves both WAVs on disk, and "
                  "the metrics now cover only the answer that survived it.")
            return 0
        print("\nThe finding no longer holds. The cancel path in tts_worker and "
              "the message order in llm_worker have to be re-read together "
              "before any figure on a double-fire item is trusted.")
        return 1
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
