"""Hằng số và hàm dựng prompt dùng chung cho train / inference / đánh giá."""

# 8 nhãn cảm xúc của ArtEmis (bỏ "something else")
EMOTIONS = [
    "amusement",
    "awe",
    "contentment",
    "excitement",
    "anger",
    "disgust",
    "fear",
    "sadness",
]

# 27 phong cách của WikiArt (đúng tên cột art_style trong ArtEmis)
STYLES = [
    "Abstract_Expressionism",
    "Action_painting",
    "Analytical_Cubism",
    "Art_Nouveau_Modern",
    "Baroque",
    "Color_Field_Painting",
    "Contemporary_Realism",
    "Cubism",
    "Early_Renaissance",
    "Expressionism",
    "Fauvism",
    "High_Renaissance",
    "Impressionism",
    "Mannerism_Late_Renaissance",
    "Minimalism",
    "Naive_Art_Primitivism",
    "New_Realism",
    "Northern_Renaissance",
    "Pointillism",
    "Pop_Art",
    "Post_Impressionism",
    "Realism",
    "Rococo",
    "Romanticism",
    "Symbolism",
    "Synthetic_Cubism",
    "Ukiyo_e",
]

DEFAULT_NEGATIVE = "blurry, low quality, deformed, text, watermark, signature"


def style_text(style: str) -> str:
    return style.replace("_", " ")


def build_prompt(style=None, emotion=None, text: str = "") -> str:
    """Dựng prompt theo đúng định dạng đã dùng khi huấn luyện.

    Ví dụ: "a painting in the Impressionism style that evokes awe, a quiet village by the river"
    """
    head = "a painting"
    if style:
        head += f" in the {style_text(style)} style"
    if emotion:
        head += f" that evokes {emotion}"
    text = (text or "").strip()
    return f"{head}, {text}" if text else head
