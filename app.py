"""Web UI Gradio: nhập mô tả, chọn phong cách + cảm xúc, sinh ảnh và xếp hạng bằng CLIP."""
import argparse
from pathlib import Path

import gradio as gr

from clip_scorer import ClipScorer
from inference import LORA_FILE, ArtGenerator
from prompts import EMOTIONS, STYLES, style_text
from translator import ViEnTranslator, looks_vietnamese

NONE = "(không chọn)"
_state: dict = {}


def get_models(lora_dir: str, device: str | None = None):
    if "gen" not in _state:
        has = (Path(lora_dir) / LORA_FILE).exists()
        _state["gen"] = ArtGenerator(lora_dir=lora_dir if has else None, device=device)
        _state["clip"] = ClipScorer(device=device)
    return _state["gen"], _state["clip"]


def get_translator():
    if "tr" not in _state:
        _state["tr"] = ViEnTranslator()
    return _state["tr"]


def rank(images, prompt, clip):
    scores = clip.score(images, prompt)
    order = sorted(range(len(images)), key=lambda i: -scores[i])
    return [(images[i], f"CLIP {scores[i]:.1f}") for i in order]


def build_ui(lora_dir: str, device: str | None = None):
    def run(text, style, emotion, n, steps, guidance, seed, use_lora, lora_scale, compare, translate=True):
        gen, clip = get_models(lora_dir, device)
        style = None if style == NONE else style
        emotion = None if emotion == NONE else emotion
        seed = int(seed)
        use_lora = bool(use_lora) and gen.has_lora

        # SD 1.5 chỉ hiểu tiếng Anh: dịch mô tả tiếng Việt trước khi sinh ảnh
        text_en, tr_note = text, ""
        if translate and looks_vietnamese(text):
            try:
                text_en = get_translator().translate(text)
                tr_note = f"Mô tả gốc: {text}\nBản dịch: {text_en}\n"
            except Exception as e:  # ví dụ chưa tải được mô hình dịch
                tr_note = f"(Không dịch được: {e}. Dùng nguyên văn tiếng Việt, kết quả có thể kém)\n"

        imgs, prompt, used_seed = gen.generate(text_en, style, emotion, int(n), int(steps), guidance,
                                               seed, use_lora, lora_scale)
        main = rank(imgs, prompt, clip)
        base = []
        if compare and gen.has_lora:
            base_imgs, _, _ = gen.generate(text_en, style, emotion, int(n), int(steps), guidance,
                                           used_seed, False)
            base = rank(base_imgs, prompt, clip)
        note = f"{tr_note}Prompt: {prompt}\nSeed: {used_seed}"
        if not gen.has_lora:
            note += "\n(Chưa có trọng số LoRA, đang dùng SD gốc)"
        return main, base, note

    with gr.Blocks(title="Tạo ảnh nghệ thuật từ văn bản") as demo:
        gr.Markdown("## Tạo ảnh nghệ thuật từ văn bản\nStable Diffusion 1.5 + LoRA huấn luyện trên ArtEmis/WikiArt, "
                    "xếp hạng bằng CLIP. Mô tả nhập được bằng tiếng Việt (tự dịch sang tiếng Anh) hoặc tiếng Anh.")
        with gr.Row():
            with gr.Column(scale=1):
                text = gr.Textbox(label="Mô tả (tiếng Việt hoặc tiếng Anh)", lines=3,
                                  value="một con mèo đang ngồi trên ghế trong phòng khách")
                translate = gr.Checkbox(value=True, label="Tự động dịch sang tiếng Anh khi nhập tiếng Việt")
                style = gr.Dropdown([NONE] + STYLES, value="Impressionism", label="Phong cách")
                emotion = gr.Dropdown([NONE] + EMOTIONS, value="contentment", label="Cảm xúc")
                n = gr.Slider(1, 8, value=4, step=1, label="Số ảnh")
                steps = gr.Slider(10, 50, value=30, step=1, label="Số bước khử nhiễu")
                guidance = gr.Slider(1, 15, value=7.5, step=0.5, label="Guidance scale")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = ngẫu nhiên)")
                use_lora = gr.Checkbox(value=True, label="Dùng LoRA ArtEmis")
                lora_scale = gr.Slider(0, 1.5, value=1.0, step=0.1, label="Cường độ LoRA")
                compare = gr.Checkbox(value=False, label="So sánh với SD gốc (cùng seed)")
                btn = gr.Button("Tạo ảnh", variant="primary")
            with gr.Column(scale=2):
                note = gr.Textbox(label="Thông tin", lines=3, interactive=False)
                g1 = gr.Gallery(label="Kết quả (xếp theo điểm CLIP giảm dần)", columns=4, height="auto")
                g2 = gr.Gallery(label="SD gốc (để so sánh)", columns=4, height="auto")
        btn.click(run, [text, style, emotion, n, steps, guidance, seed, use_lora, lora_scale, compare, translate],
                  [g1, g2, note])
        gr.Markdown(f"Phong cách có sẵn: {', '.join(style_text(s) for s in STYLES[:6])}, ...")
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora_dir", default="outputs/lora/final")
    ap.add_argument("--share", action="store_true", help="tạo link công khai (hữu ích trên Colab)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                    help="auto = dùng card NVIDIA nếu có; cpu = ép chạy CPU (rất chậm nhưng tránh lỗi ảnh đen trên một số card)")
    a = ap.parse_args()
    build_ui(a.lora_dir, None if a.device == "auto" else a.device).launch(share=a.share)
