"""Hai thước đo đúng mục tiêu đề tài (bổ sung cho CLIP score trong evaluate.py).

A. Độ giống tranh thật: độ tương đồng CLIP giữa ảnh sinh ra và "tâm" (trung bình embedding) của các tranh
   WikiArt thật cùng phong cách trong tập train. Kèm "độ đặc trưng phong cách" = giống phong cách đúng
   trừ trung bình giống các phong cách khác.
B. Khả năng điều khiển cảm xúc, đo bằng một bộ phân loại cảm xúc tuyến tính (trên embedding CLIP) huấn luyện
   trên tranh THẬT của tập train (CLIP zero-shot đoán cảm xúc quá kém, ~17% trên ảnh thật, nên không dùng):
   - Độ chính xác của bộ phân loại trên ảnh thật của tập val = mức trần của thước đo.
   - Với ảnh sinh ra: xác suất của cảm xúc được yêu cầu, và "lift" (đổi từ khóa cảm xúc, giữ nguyên nội dung và
     seed thì xác suất cảm xúc đó tăng bao nhiêu so với trung bình cùng nhóm).
   - Độ nhạy cảm xúc: cùng nội dung + cùng seed, đổi cảm xúc thì ảnh thay đổi bao nhiêu (1 - cosine trung bình).

Ảnh sinh ra được lưu để lần chạy sau dùng lại (không phải sinh lại).

Ví dụ:
    python evaluate_extra.py --data_dir data --lora_dir outputs/lora/final
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from clip_scorer import ClipScorer
from dataset import load_metadata
from inference import ArtGenerator
from prompts import EMOTIONS, build_prompt

CONTENTS = [
    "a quiet village by the river",
    "a person standing in a room",
    "a city street at night",
    "a path through a forest",
    "a portrait of a woman",
    "a table with fruit and flowers",
]
CTRL_STYLES = ["Impressionism", "Realism", "Expressionism"]


def ci95(x):
    x = np.asarray(x, dtype=float)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    return float(m), float(m - 1.96 * se), float(m + 1.96 * se)


def fmt_ci(x):
    m, lo, hi = ci95(x)
    return f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def gen_cached(gen, tag, names, prompts, seeds, cache_dir, steps, guidance):
    """Sinh những ảnh chưa có trong cache rồi trả về đủ ảnh theo thứ tự `names`."""
    d = Path(cache_dir) / tag
    d.mkdir(parents=True, exist_ok=True)
    todo = [i for i, n in enumerate(names) if not (d / f"{n}.png").exists()]
    print(f"  [{tag}] cần sinh {len(todo)}/{len(names)} ảnh")
    if todo:
        imgs = gen.run([prompts[i] for i in todo], [seeds[i] for i in todo], steps, guidance)
        for i, im in zip(todo, imgs):
            im.save(d / f"{names[i]}.png")
    return [Image.open(d / f"{n}.png").convert("RGB") for n in names]


def train_probe(X, y, k, seed, feat_scale=10.0):
    """Hồi quy logistic (softmax) trên embedding CLIP, có trọng số lớp để bù dữ liệu lệch.

    Chọn hệ số regularization trên 15% dữ liệu train giữ lại (không đụng tới tập val).
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=g)
    n_hold = int(0.15 * len(X))
    hold, fit = perm[:n_hold], perm[n_hold:]
    counts = torch.bincount(y, minlength=k).float().clamp(min=1)
    cw = (counts.sum() / (k * counts))

    def fit_linear(idx, wd, iters=200):
        lin = torch.nn.Linear(X.shape[1], k)
        torch.nn.init.zeros_(lin.weight)
        torch.nn.init.zeros_(lin.bias)
        opt = torch.optim.LBFGS(lin.parameters(), lr=1, max_iter=iters, line_search_fn="strong_wolfe")
        xs, ys = X[idx] * feat_scale, y[idx]

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(lin(xs), ys, weight=cw) + wd * (lin.weight ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        return lin

    def macro_recall(lin, idx):
        with torch.no_grad():
            pred = lin(X[idx] * feat_scale).argmax(1)
        rec = [(pred[y[idx] == c] == c).float().mean().item() for c in range(k) if (y[idx] == c).any()]
        return sum(rec) / len(rec)

    best, best_wd = -1, None
    for wd in (1e-4, 1e-3, 1e-2, 1e-1):
        s = macro_recall(fit_linear(fit, wd), hold)
        print(f"  probe cảm xúc: wd={wd:g} -> macro-recall trên phần giữ lại {s:.3f}")
        if s > best:
            best, best_wd = s, wd
    lin = fit_linear(torch.arange(len(X)), best_wd)
    return lambda emb: lin(emb * feat_scale)


def embed_paths(clip, paths, chunk=64):
    outs = []
    for i in range(0, len(paths), chunk):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + chunk]]
        outs.append(clip.embed_images(ims))
    return torch.cat(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--lora_dir", default="outputs/lora/final")
    ap.add_argument("--num_prompts", type=int, default=100, help="số prompt val cho thước đo A")
    ap.add_argument("--ref_per_style", type=int, default=60, help="số tranh thật/phong cách để tính tâm")
    ap.add_argument("--num_contents", type=int, default=len(CONTENTS))
    ap.add_argument("--ctrl_seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/eval_extra")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_dir)
    meta = load_metadata(args.data_dir)
    train = [it for it in meta["items"] if it["split"] == "train"]
    val = [it for it in meta["items"] if it["split"] == "val"]

    clip = ClipScorer()

    # ---------- embedding toàn bộ tranh thật tập train (dùng cho tâm phong cách + bộ phân loại cảm xúc) ----------
    print(f"Embedding {len(train)} tranh thật của tập train...")
    train_emb = embed_paths(clip, [root / it["image"] for it in train])

    by_style = {}
    for i, it in enumerate(train):
        by_style.setdefault(it["style"], []).append(i)
    styles, centroids = [], []
    for s, idxs in sorted(by_style.items()):
        random.Random(args.seed).shuffle(idxs)
        c = train_emb[idxs[: args.ref_per_style]].mean(0)
        styles.append(s)
        centroids.append(c / c.norm())
    C = torch.stack(centroids)  # (num_styles, D)
    sidx = {s: i for i, s in enumerate(styles)}

    # ---------- bộ phân loại cảm xúc (probe tuyến tính) huấn luyện trên tranh thật ----------
    y_train = torch.tensor([EMOTIONS.index(it["captions"][0]["emotion"]) for it in train])
    probe = train_probe(train_emb, y_train, len(EMOTIONS), args.seed)

    def emo_probs(emb):
        with torch.no_grad():
            return torch.softmax(probe(emb), dim=-1).numpy()

    def style_scores(emb, style_names):
        """Trả về (giống phong cách đúng, độ đặc trưng = đúng - trung bình các phong cách khác)."""
        sims = (emb @ C.T).numpy()
        own, spec = [], []
        for row, s in zip(sims, style_names):
            j = sidx[s]
            others = np.delete(row, j)
            own.append(row[j])
            spec.append(row[j] - others.mean())
        return np.array(own), np.array(spec)

    # ---------- A. Độ giống tranh thật (cùng prompt/seed như evaluate.py) ----------
    rng = random.Random(args.seed)
    vshuf = list(val)
    rng.shuffle(vshuf)
    vshuf = vshuf[: args.num_prompts]
    samples = []
    for it in vshuf:
        cap = rng.choice(it["captions"])
        samples.append((it, cap["emotion"], build_prompt(it["style"], cap["emotion"], cap["text"])))
    a_names = [f"val_{it['painting']}" for it, _, _ in samples]
    a_prompts = [p for _, _, p in samples]
    a_seeds = [args.seed + i for i in range(len(samples))]
    a_styles = [it["style"] for it, _, _ in samples]

    # ---------- B. Thí nghiệm điều khiển cảm xúc ----------
    contents = CONTENTS[: args.num_contents]
    b_names, b_prompts, b_seeds, b_groups, b_emo = [], [], [], [], []
    for ci, text in enumerate(contents):
        style = CTRL_STYLES[ci % len(CTRL_STYLES)]
        for si in range(args.ctrl_seeds):
            for ei, e in enumerate(EMOTIONS):
                b_names.append(f"ctrl_{ci}_{si}_{e}")
                b_prompts.append(build_prompt(style, e, text))
                b_seeds.append(1000 + ci * 10 + si)  # cùng seed cho cả 8 cảm xúc
                b_groups.append(ci * args.ctrl_seeds + si)
                b_emo.append(ei)
    b_groups, b_emo = np.array(b_groups), np.array(b_emo)

    print("Sinh ảnh (có cache)...")
    gen = ArtGenerator(lora_dir=args.lora_dir)
    imgs = {}
    for tag, use in (("base", False), ("lora", True)):
        gen.set_lora(use)
        imgs[tag] = {
            "A": gen_cached(gen, tag + "_A", a_names, a_prompts, a_seeds, out / "images", args.steps, args.guidance),
            "B": gen_cached(gen, tag + "_B", b_names, b_prompts, b_seeds, out / "images", args.steps, args.guidance),
        }
    del gen
    torch.cuda.empty_cache()

    summary = {}

    # ---------- tính A ----------
    real_imgs = [Image.open(root / it["image"]).convert("RGB") for it, _, _ in samples]
    emb_a = {"real": clip.embed_images(real_imgs)}
    for tag in ("base", "lora"):
        emb_a[tag] = clip.embed_images(imgs[tag]["A"])
    own, spec = {}, {}
    for k in emb_a:
        own[k], spec[k] = style_scores(emb_a[k], a_styles)
    print("\n=== A. Độ giống tranh thật cùng phong cách (CLIP cosine với tâm tranh thật) ===")
    print(f"{'':12s}{'giống phong cách đúng':>24s}{'độ đặc trưng':>16s}")
    for k, name in (("real", "ảnh thật"), ("base", "SD gốc"), ("lora", "SD + LoRA")):
        print(f"{name:12s}{own[k].mean():24.4f}{spec[k].mean():16.4f}")
    d_own, d_spec = own["lora"] - own["base"], spec["lora"] - spec["base"]
    print(f"LoRA - gốc, giống phong cách đúng: {fmt_ci(d_own)} | LoRA thắng {np.mean(d_own > 0):.0%}")
    print(f"LoRA - gốc, độ đặc trưng        : {fmt_ci(d_spec)} | LoRA thắng {np.mean(d_spec > 0):.0%}")
    summary["A_style_similarity"] = {
        k: {"own_mean": float(own[k].mean()), "specificity_mean": float(spec[k].mean())} for k in own}
    summary["A_style_similarity"]["lora_minus_base_own"] = ci95(d_own)
    summary["A_style_similarity"]["lora_minus_base_specificity"] = ci95(d_spec)
    pd.DataFrame({"painting": a_names, "style": a_styles, "own_real": own["real"], "own_base": own["base"],
                  "own_lora": own["lora"], "spec_base": spec["base"], "spec_lora": spec["lora"]}
                 ).to_csv(out / "A_results.csv", index=False)

    # ---------- tính B ----------
    # B1: độ tin cậy của thước đo, đo trên ảnh thật của tập val
    val_emb = embed_paths(clip, [root / it["image"] for it in val])
    vp = emo_probs(val_emb)
    v_lab = np.array([EMOTIONS.index(it["captions"][0]["emotion"]) for it in val])
    real_acc = float((vp.argmax(1) == v_lab).mean())
    real_recall = {e: float((vp[v_lab == i].argmax(1) == i).mean()) if (v_lab == i).any() else float("nan")
                   for i, e in enumerate(EMOTIONS)}
    macro = float(np.nanmean(list(real_recall.values())))
    print(f"\n=== B1. Bộ phân loại cảm xúc (probe CLIP) trên {len(val)} ảnh THẬT của tập val ===")
    print(f"độ chính xác {real_acc:.1%} | macro-recall {macro:.1%} (đoán ngẫu nhiên = {1 / len(EMOTIONS):.1%})")
    print("recall từng cảm xúc:", {e: f"{r:.0%}" for e, r in real_recall.items()})
    print("-> đây là mức trần của thước đo: nhãn cảm xúc của ArtEmis vốn chủ quan và nhiễu")
    summary["B1_real_val"] = {"accuracy": real_acc, "macro_recall": macro, "recall": real_recall, "n": len(val)}

    # B2: ảnh sinh ra với cảm xúc yêu cầu
    prob_req, acc, sens, lift = {}, {}, {}, {}
    for tag in ("base", "lora"):
        emb = clip.embed_images(imgs[tag]["B"])
        p = emo_probs(emb)
        prob_req[tag] = p[np.arange(len(b_emo)), b_emo]
        acc[tag] = (p.argmax(1) == b_emo).astype(float)
        # lift: đổi từ khóa cảm xúc (giữ nguyên nội dung + seed) thì xác suất cảm xúc đó tăng bao nhiêu
        # so với trung bình của 8 ảnh cùng nhóm
        lf = np.zeros(len(b_emo))
        for g in np.unique(b_groups):
            idx = np.where(b_groups == g)[0]
            for j in idx:
                e_j = b_emo[j]
                lf[j] = p[j, e_j] - p[idx, e_j].mean()
        lift[tag] = lf
        sims = []
        for g in np.unique(b_groups):
            e = emb[torch.from_numpy(b_groups == g)]
            m = e @ e.T
            n = len(e)
            sims.append(float((m.sum() - n) / (n * (n - 1))))
        sens[tag] = 1 - np.array(sims)
    print(f"\n=== B2. Điều khiển cảm xúc trên ảnh sinh ra ({len(b_emo)} ảnh/mô hình) ===")
    print(f"{'':12s}{'đúng cảm xúc':>14s}{'xác suất TB':>14s}{'lift':>10s}{'độ nhạy':>10s}")
    for tag, name in (("base", "SD gốc"), ("lora", "SD + LoRA")):
        print(f"{name:12s}{acc[tag].mean():14.1%}{prob_req[tag].mean():14.3f}"
              f"{lift[tag].mean():+10.3f}{sens[tag].mean():10.3f}")
    print(f"(đoán ngẫu nhiên: đúng {1 / len(EMOTIONS):.1%}, xác suất {1 / len(EMOTIONS):.3f}; lift = 0 nghĩa là "
          f"từ khóa cảm xúc không làm ảnh nghiêng về cảm xúc đó)")
    print(f"lift SD gốc   : {fmt_ci(lift['base'])}")
    print(f"lift SD + LoRA: {fmt_ci(lift['lora'])}")
    d_p = prob_req["lora"] - prob_req["base"]
    d_l = lift["lora"] - lift["base"]
    d_s = sens["lora"] - sens["base"]
    print(f"LoRA - gốc, xác suất cảm xúc đúng: {fmt_ci(d_p)}")
    print(f"LoRA - gốc, lift                 : {fmt_ci(d_l)}")
    print(f"LoRA - gốc, độ nhạy cảm xúc      : {fmt_ci(d_s)}")
    print("Xác suất cảm xúc đúng theo từng cảm xúc (gốc -> LoRA):")
    per_emo = {}
    for i, e in enumerate(EMOTIONS):
        b, l = prob_req["base"][b_emo == i].mean(), prob_req["lora"][b_emo == i].mean()
        per_emo[e] = {"base": float(b), "lora": float(l)}
        print(f"  {e:12s} {b:.3f} -> {l:.3f}")
    summary["B2_controlled"] = {
        "n_images": int(len(b_emo)),
        "accuracy": {t: float(acc[t].mean()) for t in acc},
        "prob_requested": {t: float(prob_req[t].mean()) for t in prob_req},
        "sensitivity": {t: float(sens[t].mean()) for t in sens},
        "lift": {t: ci95(lift[t]) for t in lift},
        "lora_minus_base_lift": ci95(d_l),
        "lora_minus_base_prob": ci95(d_p),
        "lora_minus_base_sensitivity": ci95(d_s),
        "per_emotion_prob": per_emo,
    }

    with open(out / "extra_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu kết quả vào {out}")


if __name__ == "__main__":
    main()
