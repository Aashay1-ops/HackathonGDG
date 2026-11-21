"""Embed train, half_a, half_b and test_inputs once; save .npy + ticket-order json per part."""
import json
import time

import numpy as np
from _common import parser, setup_logging

from sentineldesk.triage.config import artifacts_dir, embedding_prefix, load_config, model_slug
from sentineldesk.triage.data import read_part
from sentineldesk.triage.embed import Embedder
from sentineldesk.triage.text import build_texts

PARTS = ("train", "half_a", "half_b", "test_inputs")


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--model", default=None, help="sentence-transformers model (default from config)")
    p.add_argument("--batch-size", type=int, default=None)
    args = p.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    root = artifacts_dir(cfg, args.artifacts, args.sample)
    model = args.model or cfg["embedding"]["model"]
    prefix = embedding_prefix(cfg, model)
    emb = Embedder(model, prefix, args.batch_size or cfg["embedding"]["batch_size"], cfg["embedding"]["max_seq_length"])
    out = root / "embeddings" / model_slug(model)
    out.mkdir(parents=True, exist_ok=True)
    for part in PARTS:
        df = read_part(root / "splits", part)
        t0 = time.perf_counter()
        X = emb.encode(build_texts(df["subject"], df["body"]), show_progress=True)
        np.save(out / f"{part}.npy", X)
        (out / f"{part}_ids.json").write_text(json.dumps(df["ticket_id"].tolist()))
        print(f"{part}: {X.shape} in {time.perf_counter() - t0:.1f}s")
    (out / "meta.json").write_text(json.dumps({"embedding_model": model, "prefix": prefix, "dim": emb.dim,
                                               "batch_size": emb.batch_size,
                                               "max_seq_length": cfg["embedding"]["max_seq_length"]}, indent=2))


if __name__ == "__main__":
    main()
