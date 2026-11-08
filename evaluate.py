"""So sánh SD gốc với SD + LoRA trên tập val bằng CLIP score, xuất bảng điểm và lưới ảnh cho báo cáo.

Ví dụ:
    python evaluate.py --data_dir data --lora_dir outputs/lora/final --num_prompts 100
"""
import argparse
import csv
import json
import random
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw

from clip_scorer import ClipScorer
from dataset import load_metadata
from inference import ArtGenerator
from prompts import EMOTIONS, build_prompt


def make_grid(rows, col_titles, row_titles, thumb=256, pad=6, title_h=44, row_label_w=70):
    """rows: list[list[PIL.Image]] -> một ảnh lưới có tiêu đề cột và nhãn hàng."""
    ncols = len(rows[0])
    W = row_label_w + ncols * (thumb + pad)
    H = title_h + len(rows) * (thumb + pad)
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    for c, title in enumerate(col_titles):
        d.text((row_label_w + c * (thumb + pad) + 2, 4), "\n".join(textwrap.wrap(title, 38)[:3]), fill="black")
    for r, (row, label) in enumerate(zip(rows, row_titles)):
        y = title_h + r * (thumb + pad)
        d.text((4, y + thumb // 2), label, fill="black")
        for c, im in enumerate(row):
            canvas.paste(im.convert("RGB").resize((thumb, thumb)), (row_label_w + c * (thumb + pad), y))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--lora_dir", default="outputs/lora/final")
    ap.add_argument("--num_prompts", type=int, default=100)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/eval")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_dir)
    meta = load_metadata(args.data_dir)
    val = [it for it in meta["items"] if it["split"] == "val"]
    rng = random.Random(args.seed)
    rng.shuffle(val)
    val = val[: args.num_prompts]

    samples = []
    for it in val:
        cap = rng.choice(it["captions"])
        samples.append({"item": it, "emotion": cap["emotion"],
                        "prompt": build_prompt(it["style"], cap["emotion"], cap["text"])})
    prompts = [s["prompt"] for s in samples]
    seeds = [args.seed + i for i in range(len(samples))]
    print(f"Đánh giá trên {len(samples)} prompt của tập val")

    gen = ArtGenerator(lora_dir=args.lora_dir)
    scorer = ClipScorer()

    gen.set_lora(False)
    base_imgs = gen.run(prompts, seeds, args.steps, args.guidance)
    gen.set_lora(True)
    lora_imgs = gen.run(prompts, seeds, args.steps, args.guidance)
    real_imgs = [Image.open(root / s["item"]["image"]).convert("RGB") for s in samples]

    s_real = scorer.score(real_imgs, prompts)
    s_base = scorer.score(base_imgs, prompts)
    s_lora = scorer.score(lora_imgs, prompts)

    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["painting", "style", "emotion", "prompt", "clip_real", "clip_base", "clip_lora"])
        for s, a, b, c in zip(samples, s_real, s_base, s_lora):
            w.writerow([s["item"]["painting"], s["item"]["style"], s["emotion"], s["prompt"],
                        f"{a:.2f}", f"{b:.2f}", f"{c:.2f}"])

    mean = lambda x: sum(x) / len(x)
    summary = {
        "num_prompts": len(samples),
        "clip_real_mean": mean(s_real),
        "clip_base_mean": mean(s_base),
        "clip_lora_mean": mean(s_lora),
        "lora_beats_base_rate": mean([1.0 if c > b else 0.0 for b, c in zip(s_base, s_lora)]),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # lưới 1: ảnh thật | SD gốc | SD + LoRA
    k = min(8, len(samples))
    rows = [[real_imgs[i], base_imgs[i], lora_imgs[i]] for i in range(k)]
    # font mặc định của PIL không có dấu tiếng Việt nên dùng nhãn tiếng Anh
    titles = ["Real (WikiArt)", "SD 1.5 base", "SD 1.5 + LoRA"]
    make_grid(rows, titles, [f"#{i + 1}" for i in range(k)]).save(out / "compare_grid.png")
    with open(out / "compare_prompts.txt", "w", encoding="utf-8") as f:
        for i in range(k):
            f.write(f"#{i + 1}: {prompts[i]}\n")

    # lưới 2: cùng nội dung, đổi cảm xúc (kiểm tra khả năng điều khiển cảm xúc)
    text = "a quiet village by the river"
    e_prompts = [build_prompt("Impressionism", e, text) for e in EMOTIONS]
    e_seeds = [args.seed] * len(EMOTIONS)
    gen.set_lora(False)
    e_base = gen.run(e_prompts, e_seeds, args.steps, args.guidance)
    gen.set_lora(True)
    e_lora = gen.run(e_prompts, e_seeds, args.steps, args.guidance)
    make_grid([e_base, e_lora], EMOTIONS, ["Base SD", "+ LoRA"], thumb=192).save(out / "emotion_grid.png")
    print(f"Đã lưu kết quả vào {out}")


if __name__ == "__main__":
    main()
