#!/usr/bin/env python3
"""Generate the DeepSeek-V4.1 engram hash sidecar: token_map + per-layer hash constants.

Reproduces inference/engram.py from the DeepSeek-V4.1-Flash repo exactly:
- build_compressed_token_map: token -> compressed id (NFKC, NFD, StripAccents,
  Lowercase, whitespace collapse, lone-space sentinel, Strip; byte tokens keyed
  by raw form)
- compute_hash_multipliers: per-layer RNG default_rng(10007 * layer_id), odd,
  bounded so token_id * multiplier cannot overflow int64
- EngramLayout primes: per layer, per n-gram size, n_heads primes above
  engram_vocab_size - 1, never reused
- offsets: cumulative bucket starts per layer

Output (single .bin):
  u32 magic 'DSE1', u32 version=1
  u32 token_map_len (= vocab 129280)
  s32 array token_map[vocab]
  u32 n_layers, u32 max_ngram, u32 n_heads
  per layer: u64 num_embeddings
  per layer: s64 multipliers[max_ngram]      (only ngram-1 used per col)
  per layer: s64 offsets[n_hash_cols]        (n_hash_cols = (max_ngram-1)*n_heads)
  per layer: s64 primes_fl
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sympy import isprime
from tokenizers import Regex, Tokenizer, normalizers

VOCAB = 129280
LAYER_IDS = (1, 14)
NUM_EMBEDDINGS = (384006168, 384016682)
MAX_NGRAM = 4
ENGRAM_VOCAB = 16000000
N_HEADS = 8
HEAD_DIM = 256
PAD_TOKEN_ID = 2


def build_compressed_token_map(tok):
    sentinel = "\ue000"
    normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])

    key_to_new = {}
    lookup = [0] * VOCAB
    backend = tok
    for token_id in range(VOCAB):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def find_next_prime(candidate, seen):
    candidate += 1
    while not isprime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def compute_hash_multipliers(layer_ids, max_ngram, vocab_size):
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        rng = np.random.default_rng(10007 * layer_id)
        values = rng.integers(low=0, high=bound, size=(max_ngram,), dtype=np.int64)
        rows.append(values * 2 + 1)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    assert tok.get_vocab_size(True) == VOCAB, f"vocab {tok.get_vocab_size(True)} != {VOCAB}"

    token_map, comp_vocab = build_compressed_token_map(tok)
    assert comp_vocab == 99092, f"compressed vocab {comp_vocab} != 99092 (config engram_compressed_vocab_size)"
    print(f"compressed vocab: {comp_vocab} (matches config)", file=sys.stderr)

    n_hash_cols = (MAX_NGRAM - 1) * N_HEADS
    layers = []
    seen = set()
    for li, layer_id in enumerate(LAYER_IDS):
        primes_fl = []
        offsets_acc = [0]
        for _ in range(MAX_NGRAM - 1):
            sizes = []
            current = ENGRAM_VOCAB - 1
            for _ in range(N_HEADS):
                current = find_next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            primes_fl.extend(sizes)
            offsets_acc.append(offsets_acc[-1] + sizes[0])
        # offsets accumulate per n-gram block: start of each (ngram,head) bucket range.
        # reference: offsets = cumsum([0, *sizes[:-1]]) over the flattened list
        flat = primes_fl
        # recompute exactly as reference: offsets[i] = sum of all previous primes
        offs = [0]
        for p in flat[:-1]:
            offs.append(offs[-1] + p)
        layers.append({
            "layer_id": layer_id,
            "num_embeddings": NUM_EMBEDDINGS[li],
            "multipliers": compute_hash_multipliers([layer_id], MAX_NGRAM, comp_vocab)[0],
            "primes_fl": flat,
            "offsets": offs,
        })

    with open(args.out, "wb") as f:
        f.write(b"DSE1")
        f.write(np.uint32(1).tobytes())
        f.write(np.uint32(VOCAB).tobytes())
        f.write(np.array(token_map, dtype=np.int32).tobytes())
        f.write(np.uint32(len(LAYER_IDS)).tobytes())
        f.write(np.uint32(MAX_NGRAM).tobytes())
        f.write(np.uint32(N_HEADS).tobytes())
        for L in layers:
            f.write(np.int64(L["num_embeddings"]).tobytes())
            f.write(np.array(L["multipliers"], dtype=np.int64).tobytes())
            f.write(np.array(L["offsets"], dtype=np.int64).tobytes())
            f.write(np.array(L["primes_fl"], dtype=np.int64).tobytes())
    print(f"wrote {args.out} ({Path(args.out).stat().st_size} bytes)", file=sys.stderr)

    # verification block: hashes for a sample prompt, exactly as NgramHashState.forward does
    ids = [1, 100, 770, 100, 770, 999, 42]
    print("sample hash check:", file=sys.stderr)
    cache = list(ids)
    for L in layers:
        prod = [c * m for c, m in zip(cache, L["multipliers"])]
        rolling = prod[0]
        out = []
        for i in range(1, MAX_NGRAM):
            rolling ^= prod[i]
            out.append(rolling % L["primes_fl"][(i - 1) * N_HEADS])
        print(f"  layer {L['layer_id']}: first-col hashes {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
