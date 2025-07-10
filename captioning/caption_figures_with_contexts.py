import argparse
import datetime as _dt
import json
import math
import multiprocessing as mp
import os
import sys
import time
from itertools import chain
from pathlib import Path
from typing import Any, Sequence
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pandas as pd
import torch
from huggingface_hub import login as hf_login
from PIL import Image
from tqdm.auto import tqdm

try:
    import s3fs
except ImportError:
    s3fs = None

_PROMPT_TEMPLATE = (
    "Analyze the scientific figure and any provided context to create a concise, retrieval-optimized caption. "
    "Follow these guidelines:\n\n"
    "Overview: Summarize the figure’s main subject, purpose and experimental focus.\n\n"
    "Visual Details: Identify and describe important visual elements (e.g. charts, diagrams, symbols, numeric labels, legends).\n\n"
    "Relationship & Patterns: Explain interactions, trends, or hierarchies evident in the figure.\n\n"
    "Terminology & Keywords: Use relevant domain-specific terms (with plain-language equivalents if needed), then list essential keywords for retrieval.\n\n"
    "Relevance & Accuracy: Include only information visible in the figure or clearly stated in the context. Avoid speculation, repetition or irrelevant details.\n\n"
    "Formatting: Present the description coherently, in a paragraph of 300- to 400 words. Omit statements about missing elements unless explicitly relevant."
)

KEYWORDS_COL = "keywords"
TITLE_COL = "title"
SENTENCES_COL = "sentences"
ABSTRACT_COL = "abstract"
CAPTION_COL = "original_caption"
IMAGE_COL = "image_path"


def load_minicpm_on_device(model_id: str, device: str, use_flash_attn: bool = True):
    attn_impl = "sdpa"
    if use_flash_attn:
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
        except Exception:
            use_flash_attn = False
    model = (
        AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    try:
        model = torch.compile(model, mode="max-autotune")
    except Exception:
        pass
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return model, tokenizer


def load_pil(path: str | Path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def call_batch(images: Sequence[Any], prompts: Sequence[str], model: Any, tokenizer: Any, max_new: int = 512):
    if len(images) != len(prompts):
        return [None] * len(prompts)
    msgs, idx_map = [], []
    for i, (img, prm) in enumerate(zip(images, prompts)):
        if img is None:
            continue
        msgs.append([{"role": "user", "content": [img, prm]}])
        idx_map.append(i)
    if not msgs:
        return [None] * len(prompts)
    try:
        with torch.inference_mode():
            outs = model.chat(image=None, msgs=msgs, tokenizer=tokenizer, max_new_tokens=max_new, use_fast=True)
    except Exception:
        return [None] * len(prompts)
    res = [None] * len(prompts)
    for lidx, gen in enumerate(outs):
        gidx = idx_map[lidx]
        if isinstance(gen, (list, tuple)):
            gen = gen[0]
        res[gidx] = gen.strip() if isinstance(gen, str) else None
    return res


def worker(gpu_id: int, chunk: list[dict[str, str]], model_id: str, bsz: int, flog: Path, max_new: int):
    device = f"cuda:{gpu_id}"
    model, tok = load_minicpm_on_device(model_id, device)
    thr = mp.pool.ThreadPool(min(4, os.cpu_count() // max(torch.cuda.device_count(), 1)))
    out: list[dict[str, Any]] = []
    for i in range(0, len(chunk), bsz):
        part = chunk[i : i + bsz]
        paths = [p["image_path"] for p in part]
        prms = [p["prompt"] for p in part]
        imgs = list(thr.map(load_pil, paths))
        caps = call_batch(imgs, prms, model, tok, max_new)
        for itm, cap in zip(part, caps):
            out.append({"image_path": itm["image_path"], "generated_caption": cap, "status": "success" if cap else "failed"})
            if cap is None:
                flog.write_text("failed\n", append=True)
    thr.close()
    del model, tok
    torch.cuda.empty_cache()
    return out


def prompt(row: pd.Series):
    parts: list[str] = []
    if KEYWORDS_COL in row and pd.notna(row[KEYWORDS_COL]):
        parts.append(f"[KEYWORDS-START]{row[KEYWORDS_COL]}[KEYWORDS-END]\n")
    elif TITLE_COL in row and pd.notna(row[TITLE_COL]):
        parts.append(f"[TITLE-START]{row[TITLE_COL]}[TITLE-END]\n")
    if SENTENCES_COL in row and pd.notna(row[SENTENCES_COL]):
        sent = row[SENTENCES_COL]
        if isinstance(sent, (list, tuple, np.ndarray)):
            sent = " ".join(map(str, sent))
        parts.append(f"[MENTION-START]{sent}[MENTION-END]\n")
    elif ABSTRACT_COL in row and pd.notna(row[ABSTRACT_COL]):
        parts.append(f"[ABSTRACT-START]{row[ABSTRACT_COL]}[ABSTRACT-END]\n")
    if CAPTION_COL in row and pd.notna(row[CAPTION_COL]):
        parts.append(f"[CAPTION-START]{row[CAPTION_COL]}[CAPTION-END]\n")
    ctx = "".join(parts)
    return f"{ctx}\n{_PROMPT_TEMPLATE}" if ctx else _PROMPT_TEMPLATE


def load_df(path: str | Path):
    path = str(path)
    if path.startswith("s3://"):
        if s3fs is None:
            sys.exit("s3fs required for S3 paths")
        fs = s3fs.S3FileSystem(anon=False)
        files = fs.glob(f"{path.rstrip('/') }/*.parquet")
        if not files:
            sys.exit("no parquet files found")
        frames = [pd.read_parquet(f"s3://{f}", filesystem=fs) for f in files]
    else:
        files = list(Path(path).glob("*.parquet"))
        if not files:
            sys.exit("no parquet files found")
        frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    if IMAGE_COL not in df.columns:
        sys.exit("image_path column missing")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True)
    p.add_argument("--hf-token", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--image-base-path", default="")
    p.add_argument("--model-id", default="openbmb/MiniCPM-V-2_6")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=512)
    a = p.parse_args()

    out_dir = Path(a.output_path).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_csv = out_dir / "image_captions.csv"
    fail_log = out_dir / "caption_fails.log"
    prompts_file = out_dir / "prompts.jsonl"

    hf_login(token=a.hf_token)
    df = load_df(a.data_path)

    processed: set[str] = set()
    if cache_csv.exists():
        processed.update(pd.read_csv(cache_csv)[IMAGE_COL].dropna().astype(str))

    prompts: list[dict[str, str]] = []
    with fail_log.open("w", encoding="utf-8") as fl:
        fl.write(f"log { _dt.datetime.now() }\n")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            rel = str(row[IMAGE_COL]) if pd.notna(row[IMAGE_COL]) else ""
            if not rel:
                fl.write(f"row {idx} missing path\n")
                continue
            full = os.path.normpath(os.path.join(a.image_base_path, rel))
            if full in processed:
                continue
            if not os.path.exists(full):
                fl.write(f"{full} not found\n")
                continue
            prm = prompt(row)
            obj = {"image_path": full, "prompt": prm}
            prompts.append(obj)
    prompts_file.write_text("\n".join(json.dumps(p) for p in prompts), encoding="utf-8")

    if not prompts:
        print("no new images")
        return
    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    gpus = torch.cuda.device_count()
    chunks = [list(c) for c in np.array_split(prompts, gpus)]
    mp.set_start_method("spawn", force=True, allow_none=True)
    args_for_pool = [
        (gid, c, a.model_id, a.batch_size, fail_log, a.max_new_tokens)
        for gid, c in enumerate(chunks) if c
    ]
    with mp.Pool(len(args_for_pool)) as pool:
        results_nested = pool.starmap(worker, args_for_pool)
    results = list(chain.from_iterable(results_nested))
    succ = [r for r in results if r["status"] == "success" and r["generated_caption"]]
    if succ:
        df_new = pd.DataFrame(succ)[[IMAGE_COL, "generated_caption"]]
        mode = "w" if not cache_csv.exists() else "a"
        df_new.to_csv(cache_csv, mode=mode, header=mode=="w", index=False)


if __name__ == "__main__":
    main()