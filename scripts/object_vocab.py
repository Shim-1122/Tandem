"""Rebuild the object detector's vocabulary after editing vtrack/data/object_prompts.json.

    python scripts/object_vocab.py

Downloads the CLIP text encoder once (254 MB, into models/) and writes vtrack/data/object_vocab.npz. A new tracked
object also needs an entry in vtrack/grip.py (OBJECTS) to get a grip force; without one it is tracked and shown
but has no grip plan.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # let `python scripts/object_vocab.py` find vtrack

import numpy as np                                               # noqa: E402

from vtrack.clip_text import ClipTextEncoder                     # noqa: E402
from vtrack.objects import DATA_DIR, Vocabulary                  # noqa: E402


def build(spec: dict, embed) -> Vocabulary:
    names, report, prompts, pclass, pattr = [], [], [], [], []
    for group, tracked in (("track", True), ("helpers", False)):
        for name, phr in spec[group].items():
            if not phr:
                raise ValueError(f"{name} has no phrasings")
            for p in phr:
                prompts.append(p)
                pclass.append(len(names))
                pattr.append(-1)
            names.append(name)
            report.append(tracked)
    attributes = list(spec.get("attributes", {}))
    for a, phr in spec.get("attributes", {}).items():
        for p in phr:
            prompts.append(p)
            pclass.append(-1)
            pattr.append(attributes.index(a))
    return Vocabulary(names, embed(prompts), np.array(pclass), np.array(report), prompts, attributes, np.array(pattr))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompts", type=Path, default=DATA_DIR / "object_prompts.json")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "object_vocab.npz")
    ap.add_argument("--model-dir", default="models")
    a = ap.parse_args(argv)
    spec = json.loads(a.prompts.read_text(encoding="utf-8"))
    vocab = build(spec, ClipTextEncoder(a.model_dir))
    vocab.save(a.out)
    print(f"wrote {a.out}: {sum(vocab.report)} tracked objects, {len(vocab.names) - sum(vocab.report)} helpers, "
          f"{len(vocab.attributes)} attributes, {len(vocab.prompts)} phrasings")


if __name__ == "__main__":
    main()
