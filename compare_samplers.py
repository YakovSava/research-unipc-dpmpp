"""
Compare a directory of ComfyUI-generated images against one reference image,
pulling sampler name / steps / cfg / scheduler / seed straight out of each
PNG's embedded ComfyUI metadata -- no manual bookkeeping of which file used
which sampler.

Metrics computed, each against the reference image:
    - SSIM   (skimage; pixel-structural, no GPU needed)
    - LPIPS  (learned perceptual distance; needs `pip install lpips torch`)
    - CLIP-score vs. the positive prompt pulled from each image's own
      metadata (needs `pip install open_clip_torch torch`)

Usage:
    python compare_samplers.py --dir /path/to/images --ref exp__00002_.png
    python compare_samplers.py --dir .                     # ref auto-detected
                                                             # if exactly one
                                                             # LoadImage name
                                                             # appears in the
                                                             # other files'
                                                             # metadata

Output: comparison.csv in --dir, plus a printed summary table sorted by
sampler/order, and (if matplotlib is available) a PNG bar chart per metric.

HONESTY NOTE: SSIM/LPIPS here compare each generated image against a SINGLE
reference image -- they tell you "how close is this sampler's output to
that one picture", not "how good is this sampler in general" or "how well
does it converge to its own high-step-count result" (that second, arguably
more meaningful question needs a same-seed high-step baseline PER sampler,
not one shared reference across all of them -- see the note printed at the
end of this script's output).
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------
# Metadata extraction
# --------------------------------------------------------------------------

def load_comfy_metadata(path: Path):
    """
    Returns (prompt_dict, workflow_dict) from a ComfyUI-saved PNG's embedded
    text chunks, or (None, None) if absent (e.g. the reference image, or a
    PNG re-saved/re-compressed by something that stripped metadata).
    """
    try:
        img = Image.open(path)
    except Exception as e:
        print(f"  ! could not open {path.name}: {e}", file=sys.stderr)
        return None, None
    text = getattr(img, "text", {}) or {}
    prompt = json.loads(text["prompt"]) if "prompt" in text else None
    workflow = json.loads(text["workflow"]) if "workflow" in text else None
    return prompt, workflow


def extract_sampler_params(prompt: dict):
    """
    Scans a ComfyUI prompt-graph dict for the first node whose class_type
    contains 'KSampler' (covers KSampler, KSamplerAdvanced, and most custom
    sampler nodes that follow the convention) and pulls out the fields we
    care about. Also grabs the positive-prompt text (CLIPTextEncode node
    feeding the sampler's 'positive' input), used later for CLIP-score.

    Returns a dict with keys: sampler_name, steps, cfg, scheduler, seed,
    positive_text (any of which may be None if not found/applicable).
    """
    out = {"sampler_name": None, "steps": None, "cfg": None,
           "scheduler": None, "seed": None, "positive_text": None}
    if not prompt:
        return out

    sampler_node = None
    for node_id, node in prompt.items():
        ct = node.get("class_type", "")
        if "KSampler" in ct:
            sampler_node = node
            break
    if sampler_node is None:
        return out

    inputs = sampler_node.get("inputs", {})
    out["sampler_name"] = inputs.get("sampler_name")
    out["steps"] = inputs.get("steps")
    out["cfg"] = inputs.get("cfg")
    out["scheduler"] = inputs.get("scheduler")
    out["seed"] = inputs.get("seed", inputs.get("noise_seed"))

    pos_ref = inputs.get("positive")
    if isinstance(pos_ref, list) and pos_ref:
        pos_node = prompt.get(str(pos_ref[0]))
        if pos_node:
            out["positive_text"] = pos_node.get("inputs", {}).get("text")

    return out


def guess_reference_filename(prompt: dict):
    """
    Looks for a LoadImage node (what the user's own KU_ImageSSIM/MAE setup
    feeds the reference through) and returns its filename, if any -- lets
    --ref be auto-detected from any one image's own metadata instead of
    being typed out.
    """
    if not prompt:
        return None
    for node in prompt.values():
        if node.get("class_type") == "LoadImage":
            fn = node.get("inputs", {}).get("image")
            if fn:
                return fn
    return None


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def to_float01_array(path: Path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.LANCZOS)
    return np.asarray(img).astype(np.float32) / 255.0


def compute_ssim(ref_arr, img_arr):
    from skimage.metrics import structural_similarity as ssim
    return float(ssim(ref_arr, img_arr, channel_axis=2, data_range=1.0))


_lpips_model = None


def compute_lpips(ref_arr, img_arr):
    global _lpips_model
    try:
        import torch
        import lpips
    except ImportError:
        return None
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net='alex')
        if torch.cuda.is_available():
            _lpips_model = _lpips_model.cuda()
    def to_tensor(arr):
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1  # [-1,1]
        return t.cuda() if torch.cuda.is_available() else t
    with torch.no_grad():
        d = _lpips_model(to_tensor(ref_arr), to_tensor(img_arr))
    return float(d.item())


_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None


def compute_clip_score(img_path: Path, text: str):
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if not text:
        return None
    try:
        import torch
        import open_clip
    except ImportError:
        return None
    if _clip_model is None:
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai')
        _clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        _clip_model.eval()
        if torch.cuda.is_available():
            _clip_model = _clip_model.cuda()
    img = Image.open(img_path).convert("RGB")
    image_input = _clip_preprocess(img).unsqueeze(0)
    text_input = _clip_tokenizer([text])
    if torch.cuda.is_available():
        image_input = image_input.cuda()
        text_input = text_input.cuda()
    with torch.no_grad():
        image_features = _clip_model.encode_image(image_input)
        text_features = _clip_model.encode_text(text_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        sim = (image_features @ text_features.T).item()
    return sim * 100  # conventional CLIP-score scaling


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default=".", help="Directory of ComfyUI PNGs")
    ap.add_argument("--ref", type=str, default=None,
                     help="Reference image filename (in --dir). If omitted, "
                          "auto-detected from any image's LoadImage metadata.")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--no-clip", action="store_true")
    args = ap.parse_args()

    root = Path(args.dir)
    pngs = sorted(p for p in root.glob("*.png"))
    if not pngs:
        print(f"No PNGs found in {root}", file=sys.stderr)
        sys.exit(1)

    ref_name = args.ref
    if ref_name is None:
        for p in pngs:
            prompt, _ = load_comfy_metadata(p)
            guess = guess_reference_filename(prompt)
            if guess:
                ref_name = guess
                break
        if ref_name is None:
            print("Could not auto-detect reference image; pass --ref explicitly.",
                  file=sys.stderr)
            sys.exit(1)
        print(f"Auto-detected reference image: {ref_name}")

    ref_path = root / ref_name
    if not ref_path.exists():
        print(f"Reference image {ref_path} not found.", file=sys.stderr)
        sys.exit(1)

    ref_img = Image.open(ref_path).convert("RGB")
    ref_arr_native = to_float01_array(ref_path)

    rows = []
    for p in pngs:
        if p.resolve() == ref_path.resolve():
            continue
        prompt, workflow = load_comfy_metadata(p)
        params = extract_sampler_params(prompt)
        if params["sampler_name"] is None:
            print(f"  ! no KSampler metadata found in {p.name}, skipping", file=sys.stderr)
            continue

        img_arr = to_float01_array(p, size=ref_img.size)
        row = {
            "file": p.name,
            "sampler_name": params["sampler_name"],
            "steps": params["steps"],
            "cfg": params["cfg"],
            "scheduler": params["scheduler"],
            "seed": params["seed"],
            "ssim": compute_ssim(ref_arr_native, img_arr),
        }
        if not args.no_lpips:
            row["lpips"] = compute_lpips(ref_arr_native, img_arr)
        if not args.no_clip:
            row["clip_score"] = compute_clip_score(p, params["positive_text"])
        rows.append(row)
        print(f"  {p.name:40s} sampler={row['sampler_name']:28s} "
              f"steps={str(row['steps']):4s} ssim={row['ssim']:.4f}"
              + (f" lpips={row['lpips']:.4f}" if row.get("lpips") is not None else "")
              + (f" clip={row['clip_score']:.2f}" if row.get("clip_score") is not None else ""))

    if not rows:
        print("No comparable images found (all missing sampler metadata?).", file=sys.stderr)
        sys.exit(1)

    # sort for readability: by sampler family/order if the name parses that
    # way, else alphabetically
    def sort_key(r):
        name = r["sampler_name"] or ""
        import re
        m = re.match(r"dpmpp_(\d+)([ms])_", name)
        if m:
            return (name.split("_", 2)[-1], m.group(2), int(m.group(1)))
        return ("~", "~", name)
    rows.sort(key=sort_key)

    out_csv = root / "comparison.csv"
    fieldnames = ["file", "sampler_name", "steps", "cfg", "scheduler", "seed",
                  "ssim", "lpips", "clip_score"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {out_csv}")

    # quick bar chart per available metric, if matplotlib is present
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metrics_present = [m for m in ("ssim", "lpips", "clip_score")
                            if any(r.get(m) is not None for r in rows)]
        for metric in metrics_present:
            labels = [r["sampler_name"] for r in rows if r.get(metric) is not None]
            values = [r[metric] for r in rows if r.get(metric) is not None]
            fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(labels)), 4))
            ax.bar(range(len(labels)), values)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
            ax.set_ylabel(metric.upper())
            ax.set_title(f"{metric.upper()} vs reference ({ref_name})")
            fig.tight_layout()
            out_png = root / f"comparison_{metric}.png"
            fig.savefig(out_png, dpi=150)
            plt.close(fig)
            print(f"Wrote {out_png}")
    except ImportError:
        print("(matplotlib not installed -- skipping chart output; CSV is still complete)")

    print(
        "\nNOTE on interpretation: every metric above compares each image to\n"
        "ONE shared reference picture. That is a legitimate thing to measure\n"
        "(it answers 'how close does each sampler land to this specific\n"
        "target image, same seed/prompt'), but it conflates two different\n"
        "questions: (1) does the sampler converge to ITS OWN high-step-count\n"
        "limit (numerical accuracy of the integrator), and (2) does it land\n"
        "near this particular reference, which may itself just be one\n"
        "sampler's output at some step count. If your reference image was\n"
        "generated by a baseline sampler (e.g. dpmpp_3m_sde_gpu), a low\n"
        "score for another sampler could mean 'this sampler is bad' OR\n"
        "'this sampler is good but legitimately converges to a visibly\n"
        "different image than the baseline does' -- diffusion sampling is\n"
        "not guaranteed to be sampler-invariant even in the well-behaved\n"
        "regime. For question (1) specifically, re-run this script with\n"
        "--ref set to a per-sampler high-step (e.g. 100+) same-seed image\n"
        "instead of one shared reference, if you want convergence numbers\n"
        "you can defend as such in the write-up.")


if __name__ == "__main__":
    main()
