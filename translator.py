"""Dịch mô tả tiếng Việt sang tiếng Anh (chạy cục bộ) để đưa vào Stable Diffusion.

Stable Diffusion 1.5 và LoRA của đồ án chỉ hiểu tiếng Anh nên prompt tiếng Việt cần được dịch trước.
"""
import re

import torch
from transformers import MarianMTModel, MarianTokenizer

VI_EN_MODEL = "Helsinki-NLP/opus-mt-vi-en"

# ký tự chỉ có trong tiếng Việt (nguyên âm có dấu, ă â ê ô ơ ư đ)
_VI_CHARS = re.compile(
    r"[ăâđêôơưàáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ]", re.IGNORECASE)


def looks_vietnamese(text: str) -> bool:
    """Heuristic: có ký tự tiếng Việt có dấu thì coi là tiếng Việt (câu tiếng Anh sẽ không bị dịch nhầm)."""
    return bool(_VI_CHARS.search(text or ""))


class ViEnTranslator:
    def __init__(self, device: str = "cpu"):
        # mặc định chạy CPU: prompt ngắn nên đủ nhanh và không chiếm VRAM của Stable Diffusion
        self.device = device
        self.tokenizer = MarianTokenizer.from_pretrained(VI_EN_MODEL)
        self.model = MarianMTModel.from_pretrained(VI_EN_MODEL).to(device).eval()

    @torch.no_grad()
    def translate(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        batch = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True,
                               max_length=256).to(self.device)
        out = self.model.generate(**batch, num_beams=4, max_new_tokens=128)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
