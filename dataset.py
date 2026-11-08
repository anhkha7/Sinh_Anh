"""Dataset huấn luyện: mỗi lần lấy một tranh + một chú thích ngẫu nhiên của tranh đó."""
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from prompts import build_prompt


def load_metadata(data_dir: str) -> dict:
    with open(Path(data_dir) / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


class ArtEmisDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer, split: str = "train", size: int = 512,
                 flip: bool = True, limit: int | None = None):
        self.root = Path(data_dir)
        self.tokenizer = tokenizer
        meta = load_metadata(data_dir)
        self.items = [it for it in meta["items"] if it["split"] == split]
        if limit:
            self.items = self.items[:limit]
        if not self.items:
            raise ValueError(f"Không có mẫu nào cho split='{split}' trong {data_dir}/metadata.json")
        tfs = [transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
               transforms.CenterCrop(size)]
        if flip:
            tfs.append(transforms.RandomHorizontalFlip())
        tfs += [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        self.transform = transforms.Compose(tfs)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        cap = random.choice(it["captions"])
        prompt = build_prompt(it["style"], cap["emotion"], cap["text"])
        image = Image.open(self.root / it["image"]).convert("RGB")
        ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        return {"pixel_values": self.transform(image), "input_ids": ids}


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]).contiguous().float(),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
    }
