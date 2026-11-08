"""Chấm độ khớp ảnh - văn bản bằng CLIP (cosine similarity x 100)."""
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

DEFAULT_CLIP = "openai/clip-vit-base-patch32"


class ClipScorer:
    def __init__(self, name: str = DEFAULT_CLIP, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(name)

    @torch.no_grad()
    def score(self, images, texts) -> list[float]:
        """images: list[PIL.Image]; texts: một chuỗi (dùng chung) hoặc list cùng độ dài với images."""
        if isinstance(texts, str):
            texts = [texts] * len(images)
        assert len(texts) == len(images), "số ảnh và số văn bản phải bằng nhau"
        inputs = self.processor(
            text=list(texts),
            images=list(images),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)
        out = self.model(**inputs)
        # image_embeds / text_embeds đã được chuẩn hóa L2 trong CLIPModel.forward
        sim = (out.image_embeds * out.text_embeds).sum(dim=-1) * 100
        return sim.float().cpu().tolist()

    @torch.no_grad()
    def embed_images(self, images, batch: int = 32) -> torch.Tensor:
        """Embedding ảnh đã chuẩn hóa L2, shape (N, D), nằm trên CPU."""
        outs = []
        for i in range(0, len(images), batch):
            px = self.processor(images=list(images[i:i + batch]), return_tensors="pt")["pixel_values"]
            pooled = self.model.vision_model(pixel_values=px.to(self.device)).pooler_output
            outs.append(F.normalize(self.model.visual_projection(pooled).float(), dim=-1).cpu())
        return torch.cat(outs)

    @torch.no_grad()
    def embed_texts(self, texts) -> torch.Tensor:
        """Embedding văn bản đã chuẩn hóa L2, shape (N, D), nằm trên CPU."""
        t = self.processor(text=list(texts), return_tensors="pt", padding=True,
                           truncation=True, max_length=77).to(self.device)
        pooled = self.model.text_model(input_ids=t["input_ids"],
                                       attention_mask=t["attention_mask"]).pooler_output
        return F.normalize(self.model.text_projection(pooled).float(), dim=-1).cpu()

    def logit_scale(self) -> float:
        return float(self.model.logit_scale.exp().item())
