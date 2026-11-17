"""
Data pipeline for the DeLAMA Language-Model (LM) experiments (Exp 2-LM).

  * Generic pre-training corpus : WikiText-103 (raw), first `max_pretrain_chars` characters.
  * Downstream domain corpus    : 20 Newsgroups (headers/footers/quotes removed).
      latent groups (= "domains", analogous to the 2 class-groups in the image protocol)
        comp : graphics, os.ms-windows.misc, sys.ibm.pc.hardware, sys.mac.hardware, windows.x
        rec  : autos, motorcycles, sport.baseball, sport.hockey
        sci  : crypt, electronics, med, space
      each newsgroup = one "sub-topic" (analogous to one image class).
      Held-out (cross-domain) groups for the transfer protocol:
        talk : politics.guns, politics.mideast, politics.misc, religion.misc
        misc : alt.atheism, soc.religion.christian, misc.forsale
  * Tokenizer : byte-level BPE, vocab V = 1024, trained on the generic corpus only.

Everything is cached under `data/lm/`.

Usage:
    python3 src/lm_delama/data.py
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "lm"

VOCAB_SIZE = 1024

GROUPS: Dict[str, List[str]] = {
    "comp": [
        "comp.graphics",
        "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware",
        "comp.sys.mac.hardware",
        "comp.windows.x",
    ],
    "rec": ["rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey"],
    "sci": ["sci.crypt", "sci.electronics", "sci.med", "sci.space"],
    "talk": ["talk.politics.guns", "talk.politics.mideast", "talk.politics.misc", "talk.religion.misc"],
    "misc": ["alt.atheism", "soc.religion.christian", "misc.forsale"],
    # Heterogeneous 3-group benchmark (Natural English Forum vs Python Code vs C/C++ System Headers)
    "english": ["rec.autos", "rec.sport.baseball", "sci.med", "talk.politics.misc"],
    "python": ["py.torch", "py.optim", "py.numpy", "py.stdlib"],
    "c_code": ["c.linux", "c.aten", "c.sys", "c.c10"],
}


def _detok_wikitext(text: str) -> str:
    """Remove Moses word-level tokenization artifacts from wikitext-103-raw so punctuation/newlines match natural text."""
    text = text.replace(" @-@ ", "-").replace(" @.@ ", ".").replace(" @,@ ", ",")
    text = re.sub(r" +([,.:;!?)\]}\%])", r"\1", text)
    text = re.sub(r"([(\[\{]) +", r"\1", text)
    text = re.sub(r" +(n't|'s|'m|'re|'ve|'ll|'d|'t)\b", r"\1", text, flags=re.I)
    text = re.sub(r' " ([^"\n]{1,80}?) "', r' "\1"', text)
    text = re.sub(r" ' ([^'\n]{1,80}?) '", r" '\1'", text)
    # Break overly long encyclopedia paragraphs at sentence boundaries (~every 2 sentences) so '\n' frequency is natural
    text = re.sub(r"(\. [A-Z][^\n.]{80,220}\.) +(?=[A-Z\"'])", r"\1\n", text)
    return text.strip()


def _clean(text: str) -> str:
    text = re.sub(r"\S+@\S+", " ", text)  # e-mail addresses
    text = re.sub(r"[-=_*~#|]{3,}", " ", text)  # ascii art / separators
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    # Unwrap 72-column mid-sentence Usenet hard line-wraps while keeping paragraph/sentence breaks
    text = re.sub(r"(?<![.!?:])\n(?=[a-zA-Z0-9(\"'])", " ", text)
    return text.strip()


def download_pretrain_corpus(max_pretrain_chars: int = 40_000_000) -> Path:
    out = DATA_DIR / "wikitext103_subset.txt"
    if out.exists():
        return out
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    chunks, total = [], 0
    for line in ds["text"]:
        line = _detok_wikitext(line.strip())
        if not line or line.startswith("="):
            continue
        chunks.append(line)
        total += len(line) + 1
        if total >= max_pretrain_chars:
            break
    out.write_text("\n".join(chunks), encoding="utf-8")
    val = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    (DATA_DIR / "wikitext103_valid.txt").write_text(
        "\n".join(_detok_wikitext(l.strip()) for l in val["text"] if l.strip() and not l.strip().startswith("=")),
        encoding="utf-8",
    )
    return out


def download_20ng() -> Path:
    out = DATA_DIR / "20ng.json"
    if out.exists():
        return out
    from sklearn.datasets import fetch_20newsgroups

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    res = {}
    for split in ["train", "test"]:
        d = fetch_20newsgroups(subset=split, remove=("headers", "footers", "quotes"), data_home=os.environ.get("SK20NG_HOME", "/tmp/sk20ng"))
        names = d.target_names
        per_cat: Dict[str, List[str]] = {n: [] for n in names}
        for txt, y in zip(d.data, d.target):
            txt = _clean(txt)
            if len(txt) >= 300:
                per_cat[names[y]].append(txt)
        res[split] = per_cat
    out.write_text(json.dumps(res), encoding="utf-8")
    return out


def train_tokenizer(corpus: Path, vocab_size: int = VOCAB_SIZE) -> Path:
    out = DATA_DIR / f"bpe{vocab_size}.json"
    if out.exists():
        return out
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|eos|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train([str(corpus)], trainer)
    tok.save(str(out))
    return out


def load_tokenizer(vocab_size: int = VOCAB_SIZE):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(DATA_DIR / f"bpe{vocab_size}.json"))


def encode_all(vocab_size: int = VOCAB_SIZE) -> None:
    """Encode pretrain corpus -> uint16 .npy; 20NG -> per-(split, category) list of uint16 doc arrays (.npz)."""
    tok = load_tokenizer(vocab_size)
    eos = tok.token_to_id("<|eos|>")
    for name in ["wikitext103_subset", "wikitext103_valid"]:
        out = DATA_DIR / f"{name}_bpe{vocab_size}.npy"
        if out.exists():
            continue
        lines = (DATA_DIR / f"{name}.txt").read_text(encoding="utf-8").split("\n")
        # Group lines into multi-paragraph documents joined by '\n' so the newline token ('Ċ') is naturally trained
        docs = ["\n".join(lines[j : j + 12]) for j in range(0, len(lines), 12)]
        ids: List[int] = []
        bs = 500
        for i in range(0, len(docs), bs):
            for enc in tok.encode_batch(docs[i : i + bs]):
                ids.extend(enc.ids)
                ids.append(eos)
        np.save(out, np.asarray(ids, dtype=np.uint16))
        print(f"[data] {name}: {len(ids):,} tokens")

    out = DATA_DIR / f"20ng_bpe{vocab_size}.npz"
    if out.exists():
        return
    raw = json.loads((DATA_DIR / "20ng.json").read_text(encoding="utf-8"))
    arrays = {}
    for split, per_cat in raw.items():
        for cat, docs in per_cat.items():
            encs = tok.encode_batch(docs)
            for k, e in enumerate(encs):
                arrays[f"{split}|{cat}|{k}"] = np.asarray(e.ids, dtype=np.uint16)
    np.savez_compressed(out, **arrays)
    stats = {}
    for key, arr in arrays.items():
        split, cat, _ = key.split("|")
        stats.setdefault(split, {}).setdefault(cat, [0, 0])
        stats[split][cat][0] += 1
        stats[split][cat][1] += len(arr)
    (DATA_DIR / "20ng_stats.json").write_text(json.dumps(stats, indent=1))
    print("[data] 20NG encoded:", {s: sum(v[1] for v in d.values()) for s, d in stats.items()}, "tokens")


def load_20ng_tokens(vocab_size: int = VOCAB_SIZE) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """Returns {split: {category: [doc_token_array, ...]}}."""
    z = np.load(DATA_DIR / f"20ng_bpe{vocab_size}.npz")
    res: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for key in z.files:
        split, cat, _ = key.split("|")
        res.setdefault(split, {}).setdefault(cat, []).append(z[key].astype(np.int64))
    return res


if __name__ == "__main__":
    c = download_pretrain_corpus()
    download_20ng()
    train_tokenizer(c)
    encode_all()
    print("[data] done")
