"""Chuẩn bị dữ liệu ArtEmis + WikiArt cho huấn luyện.

- Đọc CSV ArtEmis, bỏ nhãn "something else" và câu quá ngắn.
- Ghép với ảnh WikiArt theo tên tranh (cột `painting`).
- Chọn tập con cân bằng theo phong cách, resize + center-crop về size x size.
- Ghi data/images/*.jpg và data/metadata.json (có tách train/val theo tranh).

Ví dụ:
    python prepare_data.py --csv data/artemis_dataset_release_v0.csv \
        --images-root D:/wikiart --out data --num-paintings 8000
"""
import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from prompts import EMOTIONS

IMG_EXT = {".jpg", ".jpeg", ".png"}


def index_images(root: str) -> dict[str, str]:
    """Quét thư mục WikiArt, trả về {tên file không đuôi: đường dẫn}."""
    idx: dict[str, str] = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.lower() in IMG_EXT:
                idx.setdefault(stem, os.path.join(dirpath, f))
    return idx


def pick_balanced(by_style: dict[str, list[str]], n: int, rng: random.Random) -> list[str]:
    """Lấy xoay vòng giữa các phong cách cho tới khi đủ n tranh."""
    for lst in by_style.values():
        rng.shuffle(lst)
    chosen: list[str] = []
    styles = sorted(by_style)
    pos = {s: 0 for s in styles}
    while len(chosen) < n:
        progressed = False
        for s in styles:
            if pos[s] < len(by_style[s]):
                chosen.append(by_style[s][pos[s]])
                pos[s] += 1
                progressed = True
                if len(chosen) >= n:
                    break
        if not progressed:
            break
    return chosen


def process_image(src: str, dst: str, size: int) -> bool:
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = size / min(w, h)
            im = im.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.LANCZOS)
            w, h = im.size
            left, top = (w - size) // 2, (h - size) // 2
            im.crop((left, top, left + size, top + size)).save(dst, quality=95)
        return True
    except Exception as e:  # ảnh hỏng thì bỏ qua
        print(f"[bỏ qua] {src}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="artemis_dataset_release_v0.csv")
    ap.add_argument("--images-root", required=True, help="thư mục gốc chứa ảnh WikiArt")
    ap.add_argument("--out", default="data")
    ap.add_argument("--num-paintings", type=int, default=8000)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--max-captions", type=int, default=12, help="số chú thích tối đa giữ lại cho mỗi tranh")
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--balance", choices=["emotion", "style"], default="emotion",
                    help="cân bằng tập con theo cảm xúc (mặc định, để mô hình học được cả cảm xúc hiếm) "
                         "hoặc theo phong cách")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    need = {"art_style", "painting", "emotion", "utterance"}
    if not need.issubset(df.columns):
        raise SystemExit(f"CSV thiếu cột {need - set(df.columns)}; các cột hiện có: {list(df.columns)}")

    df = df[df["emotion"].isin(EMOTIONS)].copy()
    df["utterance"] = df["utterance"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df = df[df["utterance"].str.split().str.len() >= args.min_words]
    print(f"Chú thích hợp lệ: {len(df):,} | số tranh trong CSV: {df['painting'].nunique():,}")

    print("Quét thư mục ảnh...")
    index = index_images(args.images_root)
    print(f"Tìm thấy {len(index):,} file ảnh")
    df = df[df["painting"].isin(index)]
    n_found = df["painting"].nunique()
    print(f"Tranh có cả chú thích lẫn ảnh: {n_found:,}")
    if n_found == 0:
        raise SystemExit(
            "Không ghép được tranh nào. Kiểm tra --images-root và tên file ảnh "
            "(phải trùng cột `painting` của ArtEmis, ví dụ 'claude-monet_water-lilies-1916.jpg')."
        )

    # gom chú thích theo tranh
    paintings: dict[str, dict] = {}
    by_style: dict[str, list[str]] = defaultdict(list)
    for (style, painting), g in df.groupby(["art_style", "painting"]):
        caps = list({(r.emotion, r.utterance) for r in g.itertuples()})
        rng.shuffle(caps)
        caps = caps[: args.max_captions]
        paintings[painting] = {
            "style": style,
            "captions": [{"emotion": e, "text": t} for e, t in caps],
        }
        # khóa cân bằng: phong cách, hoặc cảm xúc của chú thích đầu tiên (đã xáo trộn ngẫu nhiên)
        by_style[style if args.balance == "style" else caps[0][0]].append(painting)

    print(f"Số tranh theo '{args.balance}' trước khi chọn: "
          + ", ".join(f"{k}={len(v)}" for k, v in sorted(by_style.items())))
    chosen = pick_balanced(by_style, args.num_paintings, rng)
    print(f"Đã chọn {len(chosen):,} tranh (cân bằng theo {args.balance})")

    items = []
    for name in tqdm(chosen, desc="Xử lý ảnh"):
        dst = img_dir / f"{name}.jpg"
        if not dst.exists() and not process_image(index[name], str(dst), args.size):
            continue
        items.append({"image": f"images/{name}.jpg", "painting": name, **paintings[name]})

    rng.shuffle(items)
    n_val = max(1, int(len(items) * args.val_frac))
    for i, it in enumerate(items):
        it["split"] = "val" if i < n_val else "train"

    meta = {"emotions": EMOTIONS, "size": args.size, "items": items}
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    n_caps = sum(len(it["captions"]) for it in items)
    print(f"Xong: {len(items):,} tranh ({len(items) - n_val:,} train / {n_val:,} val), {n_caps:,} chú thích")
    print(f"Metadata: {out / 'metadata.json'}")


if __name__ == "__main__":
    main()
