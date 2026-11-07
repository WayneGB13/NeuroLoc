import io
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import streamlit as st

st.set_page_config(
    page_title="NeuroLoc",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASS_COLORS = {
    "Glioma":     "#FF4C6A",
    "Meningioma": "#FFA94D",
    "No Tumor":   "#40C057",
    "Pituitary":  "#4DABF7",
}
CLASS_ICONS = {
    "Glioma":     "🔴",
    "Meningioma": "🟠",
    "No Tumor":   "✅",
    "Pituitary":  "🔵",
}
CLASS_DESC = {
    "Glioma":
        "A tumor that arises from glial cells in the brain or spine. "
        "Gliomas are among the most common primary brain tumors.",
    "Meningioma":
        "A tumor that forms on the membranes surrounding the brain and "
        "spinal cord. Usually benign and slow-growing.",
    "No Tumor":
        "No tumor region was detected in this MRI scan. "
        "The brain tissue appears normal.",
    "Pituitary":
        "A tumor located on the pituitary gland at the base of the brain. "
        "Most are benign (adenomas) and treatable.",
}

IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
BASE_DIR      = Path(__file__).resolve().parent
MODEL_PATH    = BASE_DIR / "results" / "best_cnn.pth"

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #E8EDF5; }

.stApp {
    background: #080C14;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(30,120,255,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%,  rgba(100,60,255,0.06) 0%, transparent 50%);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1400px; }

.hero-header { border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 2rem; margin-bottom: 2.5rem; }
.hero-title {
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2.8rem;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #FFFFFF 0%, #7EB8FF 60%, #A78BFA 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin: 0; line-height: 1.1;
}
.hero-sub { font-family:'DM Mono',monospace; font-size:0.78rem; color:#5E7498; letter-spacing:3px; text-transform:uppercase; margin-top:0.5rem; }
.hero-badges { margin-top:1rem; display:flex; gap:0.5rem; flex-wrap:wrap; }
.badge { font-family:'DM Mono',monospace; font-size:0.65rem; letter-spacing:1.5px; text-transform:uppercase; padding:3px 10px; border-radius:20px; border:1px solid rgba(126,184,255,0.25); color:#7EB8FF; background:rgba(126,184,255,0.06); }

.upload-zone { border:1.5px dashed rgba(126,184,255,0.25); border-radius:16px; padding:2.5rem; text-align:center; background:rgba(10,20,40,0.6); }
.upload-label { font-family:'Syne',sans-serif; font-size:1.05rem; font-weight:600; color:#C3D5F0; margin-bottom:0.3rem; }
.upload-hint { font-family:'DM Mono',monospace; font-size:0.72rem; color:#3E5270; letter-spacing:1px; }

.panel { background:rgba(255,255,255,0.028); border:1px solid rgba(255,255,255,0.07); border-radius:16px; padding:1.6rem 1.8rem; margin-bottom:1.2rem; }
.panel-title { font-family:'DM Mono',monospace; font-size:0.68rem; letter-spacing:3px; text-transform:uppercase; color:#3E5270; margin-bottom:1rem; }

.pred-class { font-family:'Syne',sans-serif; font-weight:700; font-size:2.2rem; line-height:1; }
.pred-conf  { font-family:'DM Mono',monospace; font-size:1rem; color:#5E7498; margin-top:0.2rem; }
.pred-desc  { font-size:0.82rem; color:#7A93B8; margin-top:0.8rem; line-height:1.6; border-left:2px solid rgba(126,184,255,0.2); padding-left:0.8rem; }

.conf-row   { display:flex; align-items:center; gap:0.7rem; margin-bottom:0.5rem; }
.conf-label { font-family:'DM Mono',monospace; font-size:0.72rem; width:110px; color:#5E7498; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.conf-bar-bg   { flex:1; height:6px; background:rgba(255,255,255,0.06); border-radius:3px; overflow:hidden; }
.conf-bar-fill { height:100%; border-radius:3px; }
.conf-pct   { font-family:'DM Mono',monospace; font-size:0.72rem; color:#3E5270; width:42px; text-align:right; }

.step-row  { display:flex; align-items:center; gap:0.8rem; padding:0.5rem 0; border-bottom:1px solid rgba(255,255,255,0.04); }
.step-row:last-child { border-bottom:none; }
.step-num  { font-family:'DM Mono',monospace; font-size:0.65rem; width:22px; height:22px; border-radius:50%; border:1px solid rgba(126,184,255,0.2); display:flex; align-items:center; justify-content:center; color:#4D7AB8; flex-shrink:0; }
.step-text { font-size:0.82rem; color:#7A93B8; }

.section-title { font-family:'Syne',sans-serif; font-weight:700; font-size:1.1rem; color:#C3D5F0; margin:0.5rem 0 1rem; display:flex; align-items:center; gap:0.5rem; }
.section-line  { flex:1; height:1px; background:rgba(255,255,255,0.06); }

[data-testid="stFileUploader"] > div { background:transparent !important; border:none !important; padding:0 !important; }
[data-testid="stFileUploader"] label { display:none; }
div[data-testid="stImage"] img { border-radius:12px; border:1px solid rgba(255,255,255,0.07); }
.stSpinner > div { border-top-color:#7EB8FF !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        backbone = models.resnet50(weights=None)
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(backbone.fc.in_features, num_classes),
        )
        self.backbone        = backbone
        self.last_conv_layer = backbone.layer4

    def forward(self, x):
        return self.backbone(x)

    def extract_features(self, x):
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        x = b.layer4(b.layer3(b.layer2(b.layer1(x))))
        return torch.flatten(b.avgpool(x), 1)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_cnn_model(path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = BrainTumorCNN(num_classes=len(CLASS_NAMES))
    state  = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, device


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(pil_img.convert("RGB")).unsqueeze(0)


def denormalize_tensor(t: torch.Tensor) -> np.ndarray:
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std  = np.array(IMAGENET_STD).reshape(3, 1, 1)
    img  = t.detach().cpu().numpy()
    img  = (img * std + mean).clip(0, 1)
    return (img.transpose(1, 2, 0) * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, tensor: torch.Tensor, device: str):
    logits = model(tensor.to(device))
    probs  = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
    idx    = int(np.argmax(probs))
    return CLASS_NAMES[idx], probs[idx], probs


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model: BrainTumorCNN, device: str):
        self.model  = model
        self.device = device
        self._act   = None
        self._grad  = None
        model.last_conv_layer.register_forward_hook(
            lambda m, i, o: self._save_act(o))
        model.last_conv_layer.register_full_backward_hook(
            lambda m, gi, go: self._save_grad(go[0]))

    def _save_act(self, out): self._act  = out.detach()
    def _save_grad(self, g):  self._grad = g.detach()

    def __call__(self, tensor: torch.Tensor, target_class: int) -> np.ndarray:
        inp = tensor.clone().to(self.device).requires_grad_(True)
        self.model.zero_grad()
        logits = self.model(inp)
        logits[0, target_class].backward()

        weights = self._grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._act).sum(dim=1, keepdim=True))
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE),
                            mode="bilinear", align_corners=False)
        return cam.squeeze().cpu().detach().numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM BOUNDING BOX
# ─────────────────────────────────────────────────────────────────────────────
def gradcam_bbox(heatmap: np.ndarray, threshold: float = 0.45):
    """Derive a tight bounding box from the hottest Grad-CAM region."""
    for thr in [threshold, 0.25, np.percentile(heatmap, 90)]:
        mask = (heatmap >= thr).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            break
    else:
        return (40, 40, 184, 184)

    largest  = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    pad = int(0.05 * IMG_SIZE)
    return (
        max(0,        x - pad),
        max(0,        y - pad),
        min(IMG_SIZE, x + w + pad),
        min(IMG_SIZE, y + h + pad),
    )


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_figure(
    original_np: np.ndarray,
    heatmap:     np.ndarray,
    pred_class:  str,
    confidence:  float,
    bbox:        tuple,
    alpha:       float = 0.48,
) -> plt.Figure:
    h_color = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    h_rgb   = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB) / 255.0
    blended = np.clip(
        (1 - alpha) * original_np.astype(np.float32) / 255.0 + alpha * h_rgb, 0, 1)

    accent = CLASS_COLORS.get(pred_class, "#7EB8FF")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor="#080C14")
    fig.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.84, bottom=0.04)

    for ax, (img_data, title) in zip(axes, [
        (original_np, "ORIGINAL MRI"),
        (blended,     "GRAD-CAM HEATMAP"),
        (blended,     "LOCALIZATION"),
    ]):
        ax.imshow(img_data)
        ax.set_title(title, fontsize=7, color="#3E5270",
                     fontfamily="monospace", pad=6)
        ax.axis("off")

    if bbox:
        x1, y1, x2, y2 = bbox
        axes[2].add_patch(mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle="round,pad=1",
            linewidth=2, edgecolor=accent, facecolor="none",
        ))
        axes[2].text(x1 + 4, max(y1 - 8, 4), "TUMOR REGION",
                     fontsize=6, color=accent, fontfamily="monospace")

    fig.suptitle(
        f"{pred_class.upper()}  —  {confidence:.1%} confidence",
        fontsize=13, fontweight="bold", color="#E8EDF5", y=0.97)
    return fig


def fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def html(content: str):
    st.markdown(content, unsafe_allow_html=True)


def confidence_bars(probs):
    bars = '<div class="panel"><div class="panel-title">Class Probabilities</div>'
    for name, p in zip(CLASS_NAMES, probs):
        color = CLASS_COLORS[name]
        bars += f"""
        <div class="conf-row">
            <span class="conf-label">{name}</span>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{p*100:.1f}%;background:{color};"></div>
            </div>
            <span class="conf-pct">{p*100:.1f}%</span>
        </div>"""
    bars += "</div>"
    html(bars)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():

    # Hero
    html("""
    <div class="hero-header">
        <p class="hero-sub">Diagnostic AI </p>
        <h1 class="hero-title">NeuroLoc </h1>
        <div class="hero-badges">
            <span class="badge">ResNet50 Backbone</span>
            <span class="badge">Grad-CAM XAI</span>
            <span class="badge">4-Class Classification</span>
            <span class="badge">Auto Localization</span>
        </div>
    </div>
    """)

    # Load model
    cnn_model, device = None, "cpu"
    if not MODEL_PATH.exists():
        st.warning(
            f"CNN model not found at `{MODEL_PATH}`. "
            "Run `python main.py --mode train_cnn` first.",
            icon=None)
    else:
        with st.spinner("Loading model…"):
            cnn_model, device = load_cnn_model(str(MODEL_PATH))

    # Layout
    col_left, col_right = st.columns([1, 1.5], gap="large")

    # ── LEFT: Upload ──────────────────────────────────────────────────────────
    with col_left:
        html("""
        <div class="section-title">Upload MRI Scan<div class="section-line"></div></div>
        <div class="upload-zone">
            <div class="upload-label">🧠 Drop MRI image here</div>
            <div class="upload-hint">JPG · PNG · JPEG</div>
        </div>
        """)

        uploaded = st.file_uploader(
            label="MRI upload",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, caption="Uploaded MRI", use_container_width=True)

        html("""
        <div style="margin-top:1.4rem;">
        <div class="section-title">How It Works<div class="section-line"></div></div>
        <div class="panel">
            <div class="step-row"><div class="step-num">1</div><span class="step-text">Image resized to 224×224 and normalised</span></div>
            <div class="step-row"><div class="step-num">2</div><span class="step-text">ResNet50 extracts deep spatial features</span></div>
            <div class="step-row"><div class="step-num">3</div><span class="step-text">FC head classifies into 4 tumor types</span></div>
            <div class="step-row"><div class="step-num">4</div><span class="step-text">Grad-CAM highlights influential image regions</span></div>
            <div class="step-row"><div class="step-num">5</div><span class="step-text">Bounding box derived from hottest activation</span></div>
        </div>
        </div>
        """)

    # ── RIGHT: Results ────────────────────────────────────────────────────────
    with col_right:
        html("""
        <div class="section-title">Analysis Results<div class="section-line"></div></div>
        """)

        if uploaded is None:
            html("""
            <div class="panel" style="min-height:220px;display:flex;
                 flex-direction:column;align-items:center;
                 justify-content:center;opacity:0.4;">
                <div style="font-size:3rem;">🔬</div>
                <div class="panel-title" style="margin:0.6rem 0 0;">
                    Awaiting Image Upload
                </div>
            </div>
            """)

        elif cnn_model is None:
            st.error("CNN model not loaded. Check path and retrain.", icon="🚫")

        else:
            # Preprocess
            tensor = preprocess_image(pil_img)

            # Predict
            with st.spinner("Running inference…"):
                pred_class, confidence, all_probs = predict(
                    cnn_model, tensor, device)

            accent = CLASS_COLORS[pred_class]

            # Prediction card
            html(f"""
            <div class="panel" style="border-color:{accent}33;
                 background:linear-gradient(135deg,rgba(0,0,0,0.3) 0%,{accent}0D 100%);">
                <div class="panel-title">Diagnosis</div>
                <div class="pred-class" style="color:{accent};">
                    {CLASS_ICONS[pred_class]}  {pred_class}
                </div>
                <div class="pred-conf">{confidence:.2%} confidence</div>
                <div class="pred-desc">{CLASS_DESC[pred_class]}</div>
            </div>
            """)

            # Probability bars
            confidence_bars(all_probs)

            # Grad-CAM
            html("""
            <div class="section-title" style="margin-top:1.4rem;">
                Explainability<div class="section-line"></div>
            </div>
            """)

            with st.spinner("Generating Grad-CAM…"):
                gradcam     = GradCAM(cnn_model, device)
                pred_idx    = CLASS_NAMES.index(pred_class)
                heatmap     = gradcam(tensor, target_class=pred_idx)
                original_np = denormalize_tensor(tensor.squeeze(0))

            # Bounding box from Grad-CAM
            bbox_px = gradcam_bbox(heatmap)

            # Build figure
            fig       = build_figure(
                original_np, heatmap, pred_class, confidence, bbox_px)
            img_bytes = fig_to_bytes(fig)
            plt.close(fig)

            st.image(img_bytes, use_container_width=True)

            # Bounding box info panel
            x1, y1, x2, y2 = bbox_px
            html(f"""
            <div class="panel" style="margin-top:0.8rem;">
                <div class="panel-title">Grad-CAM Localization — Bounding Box</div>
                <div style="display:flex;gap:2rem;">
                    <div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#3E5270;">ORIGIN</div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.9rem;color:{accent};">({x1}, {y1})</div>
                    </div>
                    <div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#3E5270;">SIZE</div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.9rem;color:{accent};">{x2-x1} × {y2-y1} px</div>
                    </div>
                    <div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#3E5270;">EXTENT</div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.9rem;color:{accent};">({x2}, {y2})</div>
                    </div>
                </div>
            </div>
            """)

            # Downloads
            html("""
            <div class="section-title" style="margin-top:1.4rem;">
                Export<div class="section-line"></div>
            </div>
            """)

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    label="⬇ Download Grad-CAM",
                    data=img_bytes,
                    file_name=f"gradcam_{pred_class.lower()}.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with dl2:
                report = (
                    f"NeuroScan AI Report\n{'='*30}\n"
                    f"Predicted Class : {pred_class}\n"
                    f"Confidence      : {confidence:.4f}\n"
                    f"Probabilities   :\n"
                    + "\n".join(f"  {n}: {p:.4f}"
                                for n, p in zip(CLASS_NAMES, all_probs))
                    + f"\nBounding Box    : {bbox_px}"
                    + f"\nDevice          : {device.upper()}\n"
                )
                st.download_button(
                    label="⬇ Download Report",
                    data=report,
                    file_name="neuroscan_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

    # Footer
    html("""
    <div style="text-align:center;margin-top:4rem;padding-top:2rem;
         border-top:1px solid rgba(255,255,255,0.05);">
        <span style="font-family:'DM Mono',monospace;font-size:0.65rem;
              color:#2A3A52;letter-spacing:2px;">
            NEUROLOC · ResNet50 + Grad-CAM · FOR RESEARCH USE ONLY
        </span>
    </div>
    """)


if __name__ == "__main__":
    main()