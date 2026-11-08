"""Fine-tune Stable Diffusion 1.5 bằng LoRA trên ArtEmis/WikiArt.

Chỉ huấn luyện các ma trận LoRA gắn vào lớp attention của UNet; VAE và text encoder đóng băng.
Hỗ trợ lưu checkpoint định kỳ + --resume để chạy được trên Colab (hay bị ngắt phiên).

Ví dụ:
    python train_lora.py --data_dir data --output_dir outputs/lora --max_steps 6000 --resume
"""
import argparse
import csv
import re
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from dataset import ArtEmisDataset, collate_fn
from inference import DEFAULT_BASE
from prompts import build_prompt

SAMPLE_PROMPTS = [
    build_prompt("Impressionism", "contentment", "a quiet village by the river at sunset"),
    build_prompt("Expressionism", "fear", "a lonely figure standing in a dark street"),
    build_prompt("Romanticism", "awe", "mountains covered in mist above a stormy sea"),
    build_prompt("Ukiyo_e", "amusement", "a cat sitting on a chair in a living room"),
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", default=DEFAULT_BASE)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--output_dir", default="outputs/lora")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--max_steps", type=int, default=6000, help="số bước cập nhật tham số (đã tính grad_accum)")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--mixed_precision", choices=["fp16", "bf16", "no"], default="fp16",
                    help="Colab T4 không hỗ trợ bf16 nên dùng fp16")
    ap.add_argument("--use_8bit_adam", action="store_true")
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--sample_every", type=int, default=500)
    ap.add_argument("--keep_checkpoints", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="chỉ dùng N tranh đầu (để chạy thử nhanh)")
    ap.add_argument("--resume", action="store_true", help="tiếp tục từ checkpoint mới nhất trong output_dir")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def latest_checkpoint(out: Path):
    cands = [(int(m.group(1)), p) for p in out.glob("checkpoint-*")
             if (m := re.fullmatch(r"checkpoint-(\d+)", p.name)) and (p / "train_state.pt").exists()]
    return max(cands, key=lambda x: x[0]) if cands else None


def save_lora_diffusers(directory: Path, unet):
    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    state = {k: v.detach().float().cpu() for k, v in state.items()}
    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(directory), unet_lora_layers=state, safe_serialization=True)


def save_checkpoint(accelerator, unet, optimizer, lr_scheduler, step, out: Path, keep: int):
    # ghi vào thư mục tạm rồi đổi tên, tránh checkpoint hỏng nếu phiên bị ngắt giữa chừng
    tmp = out / f".tmp-checkpoint-{step}"
    final = out / f"checkpoint-{step}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    unwrapped = accelerator.unwrap_model(unet)
    save_lora_diffusers(tmp, unwrapped)
    torch.save({"lora": get_peft_model_state_dict(unwrapped), "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(), "step": step}, tmp / "train_state.pt")
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    olds = sorted((int(re.fullmatch(r"checkpoint-(\d+)", p.name).group(1)), p)
                  for p in out.glob("checkpoint-*") if re.fullmatch(r"checkpoint-(\d+)", p.name))
    for _, p in olds[:-keep]:
        shutil.rmtree(p, ignore_errors=True)


@torch.no_grad()
def generate_samples(args, accelerator, unet, text_encoder, vae, weight_dtype, step, out: Path):
    sample_dir = out / "samples"
    sample_dir.mkdir(exist_ok=True)
    unwrapped = accelerator.unwrap_model(unet)
    was_training = unwrapped.training
    unwrapped.eval()
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained, unet=unwrapped, text_encoder=text_encoder, vae=vae,
        safety_checker=None, requires_safety_checker=False, torch_dtype=weight_dtype)
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(accelerator.device)
    gen = torch.Generator(device=accelerator.device).manual_seed(1234)
    use_amp = weight_dtype != torch.float32
    for i, prompt in enumerate(SAMPLE_PROMPTS):
        with torch.autocast(accelerator.device.type, dtype=weight_dtype, enabled=use_amp):
            img = pipe(prompt, num_inference_steps=25, generator=gen).images[0]
        img.save(sample_dir / f"step{step:06d}_{i}.png")
    del pipe
    torch.cuda.empty_cache()
    if was_training:
        unwrapped.train()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum,
                              mixed_precision=args.mixed_precision)
    set_seed(args.seed)
    device = accelerator.device
    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "no": torch.float32}[args.mixed_precision]

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained, subfolder="unet")

    for m in (text_encoder, vae, unet):
        m.requires_grad_(False)
    text_encoder.to(device, dtype=weight_dtype).eval()
    vae.to(device, dtype=weight_dtype).eval()
    unet.to(device, dtype=weight_dtype)

    unet.add_adapter(LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    # tham số LoRA phải ở fp32 để GradScaler hoạt động với fp16
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    for p in lora_params:
        p.data = p.data.float()
    unet.enable_gradient_checkpointing()
    n_params = sum(p.numel() for p in lora_params)
    accelerator.print(f"Tham số LoRA huấn luyện: {n_params / 1e6:.2f}M")

    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(lora_params, lr=args.lr, weight_decay=1e-2)
    else:
        optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=1e-2)
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=args.warmup,
                                 num_training_steps=args.max_steps)

    global_step = 0
    if args.resume:
        ckpt = latest_checkpoint(out)
        if ckpt:
            global_step, path = ckpt
            state = torch.load(path / "train_state.pt", map_location="cpu")
            set_peft_model_state_dict(unet, state["lora"])
            optimizer.load_state_dict(state["optimizer"])
            lr_scheduler.load_state_dict(state["scheduler"])
            accelerator.print(f"Tiếp tục từ {path.name} (bước {global_step})")
        else:
            accelerator.print("Không có checkpoint để resume, bắt đầu từ đầu")

    dataset = ArtEmisDataset(args.data_dir, tokenizer, "train", args.resolution, limit=args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                        num_workers=args.num_workers, drop_last=True,
                        persistent_workers=args.num_workers > 0)
    accelerator.print(f"Dữ liệu train: {len(dataset)} tranh")

    unet, optimizer, loader = accelerator.prepare(unet, optimizer, loader)
    unet.train()

    log_path = out / "loss_log.csv"
    new_log = not log_path.exists()
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["step", "loss", "lr"])

    progress = tqdm(total=args.max_steps, initial=global_step, desc="train",
                    disable=not accelerator.is_local_main_process)
    running, count = 0.0, 0
    while global_step < args.max_steps:
        for batch in loader:
            with accelerator.accumulate(unet):
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"].to(device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    cond = text_encoder(batch["input_ids"].to(device))[0]
                noise = torch.randn_like(latents)
                t = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                  (latents.shape[0],), device=device).long()
                noisy = noise_scheduler.add_noise(latents, noise, t)
                if noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, t)
                else:
                    target = noise
                pred = unet(noisy, t, cond).sample
                loss = F.mse_loss(pred.float(), target.float())
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            running += loss.item()
            count += 1
            if not accelerator.sync_gradients:
                continue

            lr_scheduler.step()
            global_step += 1
            progress.update(1)
            if global_step % 25 == 0:
                avg = running / count
                lr_now = lr_scheduler.get_last_lr()[0]
                progress.set_postfix(loss=f"{avg:.4f}", lr=f"{lr_now:.2e}")
                log.writerow([global_step, f"{avg:.5f}", f"{lr_now:.3e}"])
                log_f.flush()
                running, count = 0.0, 0

            if accelerator.is_main_process:
                if global_step % args.save_every == 0 and global_step < args.max_steps:
                    save_checkpoint(accelerator, unet, optimizer, lr_scheduler, global_step, out,
                                    args.keep_checkpoints)
                if global_step % args.sample_every == 0:
                    generate_samples(args, accelerator, unet, text_encoder, vae, weight_dtype,
                                     global_step, out)
            if global_step >= args.max_steps:
                break

    progress.close()
    log_f.close()
    if accelerator.is_main_process:
        final = out / "final"
        final.mkdir(exist_ok=True)
        save_lora_diffusers(final, accelerator.unwrap_model(unet))
        save_checkpoint(accelerator, unet, optimizer, lr_scheduler, global_step, out, args.keep_checkpoints)
        generate_samples(args, accelerator, unet, text_encoder, vae, weight_dtype, global_step, out)
        accelerator.print(f"Hoàn tất. Trọng số LoRA: {final / 'pytorch_lora_weights.safetensors'}")


if __name__ == "__main__":
    main()
