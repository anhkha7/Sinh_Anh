"""Sinh ảnh bằng Stable Diffusion 1.5, có thể gắn/tắt LoRA đã huấn luyện."""
import random
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

from prompts import DEFAULT_NEGATIVE, build_prompt

# repo runwayml/stable-diffusion-v1-5 đã bị gỡ khỏi Hugging Face, dùng bản mirror này
DEFAULT_BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
LORA_FILE = "pytorch_lora_weights.safetensors"


class ArtGenerator:
    def __init__(self, base: str = DEFAULT_BASE, lora_dir: str | None = None,
                 device: str | None = None, safety_checker: bool = False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        kw = {} if safety_checker else {"safety_checker": None, "requires_safety_checker": False}
        self.pipe = StableDiffusionPipeline.from_pretrained(base, torch_dtype=dtype, **kw)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.has_lora = False
        if lora_dir:
            self.load_lora(lora_dir)

    def load_lora(self, lora_dir: str):
        if not (Path(lora_dir) / LORA_FILE).exists():
            raise FileNotFoundError(f"Không thấy {LORA_FILE} trong {lora_dir}")
        self.pipe.load_lora_weights(lora_dir, weight_name=LORA_FILE, adapter_name="art")
        self.has_lora = True

    def set_lora(self, enabled: bool, scale: float = 1.0):
        if not self.has_lora:
            return
        if enabled:
            self.pipe.enable_lora()
            self.pipe.set_adapters(["art"], adapter_weights=[scale])
        else:
            self.pipe.disable_lora()

    def run(self, prompts: list[str], seeds: list[int], steps: int = 30, guidance: float = 7.5,
            negative_prompt: str = DEFAULT_NEGATIVE, chunk: int = 4):
        """Mỗi prompt một ảnh, mỗi ảnh một seed riêng (để tái lập và so sánh công bằng)."""
        images = []
        for i in range(0, len(prompts), chunk):
            p, s = prompts[i:i + chunk], seeds[i:i + chunk]
            gens = [torch.Generator(device=self.device).manual_seed(int(x)) for x in s]
            with torch.inference_mode():
                images += self.pipe(prompt=p, negative_prompt=[negative_prompt] * len(p),
                                    num_inference_steps=steps, guidance_scale=guidance,
                                    generator=gens).images
        return images

    def generate(self, text: str = "", style: str | None = None, emotion: str | None = None,
                 n: int = 4, steps: int = 30, guidance: float = 7.5, seed: int | None = None,
                 use_lora: bool = True, lora_scale: float = 1.0,
                 negative_prompt: str = DEFAULT_NEGATIVE):
        """Trả về (danh sách ảnh, prompt đầy đủ đã dùng, seed gốc)."""
        prompt = build_prompt(style, emotion, text)
        if seed is None or seed < 0:
            seed = random.randint(0, 2**31 - 1)
        self.set_lora(use_lora, lora_scale)
        images = self.run([prompt] * n, [seed + i for i in range(n)], steps, guidance, negative_prompt)
        return images, prompt, seed


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="a cat sitting on a chair in a living room")
    ap.add_argument("--style", default=None)
    ap.add_argument("--emotion", default=None)
    ap.add_argument("--lora_dir", default="outputs/lora/final")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="outputs/gen")
    a = ap.parse_args()

    lora = a.lora_dir if (Path(a.lora_dir) / LORA_FILE).exists() else None
    gen = ArtGenerator(lora_dir=lora)
    imgs, prompt, seed = gen.generate(a.text, a.style, a.emotion, a.n, seed=a.seed, use_lora=lora is not None)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(imgs):
        im.save(Path(a.out) / f"seed{seed + i}.png")
    print(f"Prompt: {prompt}\nĐã lưu {len(imgs)} ảnh vào {a.out}")
