import sys
import os

UI_DIR = os.path.dirname(os.path.abspath(__file__))
if UI_DIR not in sys.path:
    sys.path.insert(0, UI_DIR)

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import json
import h5py
import io
import datetime
from Agents.langgraph_agent import generate_advisory

# ── PDF imports ──────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.pdfgen import canvas as rl_canvas

CONFIDENCE_THRESHOLD = 0.50

TEMPERATURE = 2.5
SMOOTH_EPS  = 1e-3
CONF_CAP    = 0.95

st.set_page_config(
    layout="wide",
    page_title="GuavaXAI · Explainability Dashboard",
    page_icon="🍈"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:      #0d1117;
    --card:    #161b22;
    --raised:  #1c2230;
    --green:   #39d98a;
    --amber:   #f5a623;
    --teal:    #4ecdc4;
    --red:     #ff6b6b;
    --txt:     #e6edf3;
    --muted:   #7d8590;
    --border:  #30363d;
    --r:       10px;
    --hdr:     46px;
}

html, body { height: 100vh !important; overflow: hidden !important; margin: 0; padding: 0; }
.stApp    { height: 100vh !important; overflow: hidden !important;
            background: var(--bg) !important; color: var(--txt) !important;
            font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; height: 0 !important; }
.block-container {
    padding: 0 !important; max-width: 100vw !important;
    height: 100vh !important; overflow: hidden !important;
}
.hdr {
    height: var(--hdr); min-height: var(--hdr);
    display: flex; align-items: center; gap: 12px;
    padding: 0 20px; border-bottom: 1px solid var(--border);
    background: var(--card);
}
.hdr-title {
    font-family: 'DM Serif Display', Georgia, serif; font-style: italic;
    font-size: 1.45rem;
    background: linear-gradient(135deg, var(--green), var(--teal));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1; margin: 0;
}
.hdr-badge {
    font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
    letter-spacing: 0.12em; text-transform: uppercase; color: var(--green);
    background: rgba(57,217,138,.12); border: 1px solid rgba(57,217,138,.3);
    padding: 2px 8px; border-radius: 20px;
}
.hdr-sub { font-size: 0.72rem; color: var(--muted); margin-left: auto; white-space: nowrap; }
.slbl {
    font-family: 'JetBrains Mono', monospace; font-size: 0.62rem;
    letter-spacing: .1em; text-transform: uppercase; color: var(--muted);
    margin: 0 0 4px 0; padding: 0;
}
.img-wrap { background: var(--raised); border: 1px solid var(--border); border-radius: var(--r); overflow: hidden; }
.img-wrap-hdr {
    padding: 4px 10px; border-bottom: 1px solid var(--border);
    font-family: 'JetBrains Mono', monospace; font-size: 0.6rem;
    letter-spacing: .08em; text-transform: uppercase; color: var(--muted);
    display: flex; align-items: center; gap: 6px;
}
.dot { width: 7px; height: 7px; border-radius: 50%; display:inline-block; }
.pred-card {
    background: var(--raised); border: 1px solid var(--border);
    border-radius: var(--r); padding: 8px 12px; position: relative; overflow: hidden;
}
.pred-card::before { content:''; position:absolute; left:0; top:0; bottom:0; width:3px; border-radius:3px 0 0 3px; }
.pred-card.mat::before { background: var(--green); }
.pred-card.dis::before { background: var(--amber); }
.pred-lbl { font-family:'JetBrains Mono',monospace; font-size:.6rem;
            letter-spacing:.1em; text-transform:uppercase; color:var(--muted); display:block; }
.pred-val { font-family:'DM Serif Display',serif; font-style:italic;
            font-size:1.25rem; line-height:1.15; display:block; }
.pred-card.mat .pred-val { color: var(--green); }
.pred-card.dis .pred-val { color: var(--amber); }
.pred-conf { font-size:.65rem; color:var(--muted); font-family:'JetBrains Mono',monospace; display:block; }
.bar-wrap  { background:var(--bg); border-radius:3px; height:3px; margin-top:5px; }
.bar-fill  { height:100%; border-radius:3px; }
.prob-row { display:flex; align-items:center; gap:6px; margin-bottom:5px; }
.prob-nm  { width:88px; font-size:.72rem; white-space:nowrap; overflow:hidden;
            text-overflow:ellipsis; color:var(--txt); flex-shrink:0; }
.prob-bg  { flex:1; background:var(--bg); border-radius:3px; height:5px; overflow:hidden; }
.prob-fg  { height:100%; border-radius:3px; }
.prob-pct { width:36px; text-align:right; font-family:'JetBrains Mono',monospace;
            color:var(--muted); font-size:.65rem; flex-shrink:0; }
.xai-desc {
    font-size:.72rem; color:var(--muted); background:var(--raised);
    border-radius:6px; padding:5px 8px; border-left:2px solid var(--teal);
    line-height:1.45; margin-bottom:8px;
}
.adv-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:5px; }
.adv-mode-pill {
    font-family:'JetBrains Mono',monospace; font-size:.58rem;
    letter-spacing:.08em; text-transform:uppercase;
    padding:2px 7px; border-radius:20px; display:inline-block;
}
.adv-mode-harvest   { color:#39d98a; background:rgba(57,217,138,.12);  border:1px solid rgba(57,217,138,.3); }
.adv-mode-treatment { color:#f5a623; background:rgba(245,166,35,.12);  border:1px solid rgba(245,166,35,.3); }
.adv-mode-uncertain { color:#4ecdc4; background:rgba(78,205,196,.12);  border:1px solid rgba(78,205,196,.3); }
.adv-mode-fallback  { color:#7d8590; background:rgba(125,133,144,.12); border:1px solid rgba(125,133,144,.3); }
.advisory-box {
    font-size:.73rem; color:var(--txt); background:var(--raised);
    border:1px solid var(--border); border-radius:6px;
    padding:8px 10px; line-height:1.65; max-height:24vh; overflow-y:auto;
}
.advisory-box .adv-section {
    font-family:'JetBrains Mono',monospace; font-size:.6rem;
    letter-spacing:.1em; color:var(--teal); text-transform:uppercase;
    display:block; margin-top:8px; margin-bottom:2px;
}
.advisory-warn {
    font-size:.68rem; color:var(--amber); background:rgba(245,166,35,.08);
    border:1px solid rgba(245,166,35,.25); border-radius:6px;
    padding:4px 8px; margin-bottom:5px;
}
.advisory-uncertain {
    font-size:.68rem; color:var(--teal); background:rgba(78,205,196,.08);
    border:1px solid rgba(78,205,196,.25); border-radius:6px;
    padding:4px 8px; margin-bottom:5px;
}
button[data-baseweb="tab"] {
    font-family:'DM Sans',sans-serif !important; font-weight:500 !important;
    font-size:.78rem !important; color:var(--muted) !important;
    background:transparent !important; padding:4px 14px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color:var(--green) !important; border-bottom:2px solid var(--green) !important;
}
[data-testid="stTabPanel"] {
    background:var(--card) !important; padding:8px !important;
    border:1px solid var(--border) !important;
    border-radius:0 var(--r) var(--r) var(--r) !important;
}
section[data-testid="stFileUploader"] {
    background:var(--raised) !important; border:1.5px dashed var(--border) !important;
    border-radius:var(--r) !important;
}
section[data-testid="stFileUploader"]:hover { border-color:var(--green) !important; }
[data-testid="stFileUploaderDropzone"] { padding:8px 12px !important; }
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] small { font-size:.72rem !important; }
[data-testid="stRadio"] label { font-size:.78rem !important; padding:2px 0 !important; }
[data-testid="stVerticalBlock"] > div { padding-top:0 !important; padding-bottom:0 !important; }
[data-testid="column"] { overflow:hidden; min-width:0; }
[data-testid="stImage"] img { max-height:32vh; object-fit:contain; width:100%; }
[data-testid="stButton"] > button {
    width:100%; background:rgba(78,205,196,.1) !important;
    border:1px solid rgba(78,205,196,.35) !important; color:var(--teal) !important;
    font-family:'DM Sans',sans-serif !important; font-size:.75rem !important;
    font-weight:500 !important; border-radius:6px !important;
    padding:4px 10px !important; transition:background .2s;
}
[data-testid="stButton"] > button:hover { background:rgba(78,205,196,.2) !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════
# PATHS
# ═══════════════════════════════════════
MULTITASK_MODEL_PATH = "/Users/dhruviramaiya/Downloads/Mtech Major Project/Multitask/best_optimized_multitask_model.pth"
FUSION_MODEL_PATH    = "/Users/dhruviramaiya/Downloads/Mtech Major Project/Multimodal_Fusion/models/final_fusion_model.h5"

IMG_SIZE         = 224
MATURITY_CLASSES = ['Immature', 'Semi-Mature', 'Mature']
DISEASE_CLASSES  = ['Anthracnose', 'Styler Rot', 'Scab', 'Healthy']
MATURITY_COLORS  = ['#39d98a', '#4ecdc4', '#a8edea']
DISEASE_COLORS   = ['#ff6b6b', '#f5a623', '#ffe08a', '#39d98a']

device = torch.device("cpu")


# ═══════════════════════════════════════
# CALIBRATION
# ═══════════════════════════════════════
def calibrate_probs(logits: torch.Tensor, n_classes: int) -> np.ndarray:
    scaled = logits / TEMPERATURE
    probs  = torch.softmax(scaled, dim=1).cpu().numpy()[0]
    probs  = (1.0 - SMOOTH_EPS) * probs + SMOOTH_EPS / n_classes
    probs  = probs / probs.sum()
    probs  = np.clip(probs, None, CONF_CAP)
    probs  = probs / probs.sum()
    return probs.astype(np.float32)


# ═══════════════════════════════════════
# H5 INSPECTION UTILITIES
# ═══════════════════════════════════════
def _recursive_patch(obj):
    if isinstance(obj, dict):
        if obj.get("class_name") == "InputLayer":
            cfg = obj.get("config", {})
            if "batch_shape" in cfg and "batch_input_shape" not in cfg:
                cfg["batch_input_shape"] = cfg.pop("batch_shape")
        return {k: _recursive_patch(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_recursive_patch(i) for i in obj]
    return obj


def _read_model_config(h5_path: str):
    try:
        with h5py.File(h5_path, "r") as f:
            raw = f.attrs.get("model_config", None)
            if raw is None:
                return None
            return json.loads(raw)
    except Exception:
        return None


def _extract_input_shapes(cfg: dict) -> list:
    shapes = []
    def _walk(obj):
        if isinstance(obj, dict):
            if obj.get("class_name") == "InputLayer":
                ic = obj.get("config", {})
                bs = ic.get("batch_shape") or ic.get("batch_input_shape")
                if bs:
                    shapes.append(int(bs[-1]))
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for i in obj:
                _walk(i)
    _walk(cfg)
    return shapes


def _collect_dense_kernels(h5_path: str) -> list:
    kernels = {}

    def _scan(group, top_layer_name):
        for key in group.keys():
            item = group[key]
            if hasattr(item, "shape") and "kernel" in key and len(item.shape) == 2:
                kernels[top_layer_name] = tuple(item.shape)
            elif hasattr(item, "keys"):
                _scan(item, top_layer_name)

    try:
        with h5py.File(h5_path, "r") as f:
            root = f.get("model_weights", f)
            for lname in root.keys():
                _scan(root[lname], lname)
    except Exception:
        pass

    cfg = _read_model_config(h5_path)
    order = []
    if cfg:
        def _get_dense_names(obj):
            if isinstance(obj, dict):
                if obj.get("class_name") in ("Dense", "dense"):
                    n = obj.get("config", {}).get("name", "")
                    if n:
                        order.append(n)
                for v in obj.values():
                    _get_dense_names(v)
            elif isinstance(obj, list):
                for i in obj:
                    _get_dense_names(i)
        _get_dense_names(cfg)

    ordered   = [(n, kernels[n]) for n in order if n in kernels]
    remaining = [(n, s) for n, s in kernels.items() if n not in order]
    return ordered + remaining


# ═══════════════════════════════════════
# FUSION MODEL
# ═══════════════════════════════════════
def _attempt1_direct(h5_path: str):
    return tf.keras.models.load_model(h5_path)


def _attempt2_json_patch(h5_path: str):
    cfg = _read_model_config(h5_path)
    if cfg is None:
        raise ValueError("model_config not found in H5 file.")
    patched = _recursive_patch(cfg)
    model   = tf.keras.models.model_from_json(json.dumps(patched))
    model.load_weights(h5_path, by_name=True, skip_mismatch=False)
    return model


def _attempt3_autodetect(h5_path: str):
    cfg          = _read_model_config(h5_path)
    input_shapes = _extract_input_shapes(cfg) if cfg else []
    dense_layers = _collect_dense_kernels(h5_path)

    if not dense_layers:
        raise RuntimeError("No Dense layer weights found in H5 file.")

    in0 = 3072
    in1 = 3072

    first_in_dims = {dl[1][0] for dl in dense_layers[:2]}
    two_branch = (
        len(input_shapes) >= 2
        and len(dense_layers) >= 2
        and dense_layers[0][1][0] in (in0, in1)
        and dense_layers[1][1][0] in (in0, in1)
        and dense_layers[0][1][0] != dense_layers[0][1][0] + dense_layers[1][1][0]
    )

    if in0 == in1 and len(input_shapes) < 2:
        two_branch = False

    if two_branch:
        inp_d = tf.keras.Input(shape=(in0,), name="digital_features")
        inp_t = tf.keras.Input(shape=(in1,), name="thermal_features")
        d_out = inp_d
        t_out = inp_t
        branch_idx = 0

        lname0, (_, out0) = dense_layers[0]
        lname1, (_, out1) = dense_layers[1]
        d_out = tf.keras.layers.Dense(out0, activation="relu", name=lname0)(d_out)
        t_out = tf.keras.layers.Dense(out1, activation="relu", name=lname1)(t_out)
        branch_idx = 2

        merged = tf.keras.layers.Concatenate()([d_out, t_out])
        x         = merged
        remaining = dense_layers[branch_idx:]
        for i, (lname, (_, out_d)) in enumerate(remaining):
            act = "softmax" if i == len(remaining) - 1 else "relu"
            x = tf.keras.layers.Dense(out_d, activation=act, name=lname)(x)

        keras_model = tf.keras.Model(inputs=[inp_d, inp_t], outputs=x)

    else:
        inp = tf.keras.Input(shape=(3072,), name="input_features")
        x = inp
        for i, (lname, (_, out_d)) in enumerate(dense_layers):
            act = "softmax" if i == len(dense_layers) - 1 else "relu"
            x = tf.keras.layers.Dense(out_d, activation=act, name=lname)(x)
        keras_model = tf.keras.Model(inputs=inp, outputs=x)

    keras_model.load_weights(h5_path, by_name=True, skip_mismatch=False)
    return keras_model


def load_fusion_model(h5_path: str) -> tf.keras.Model:
    errors = []
    for attempt, fn in enumerate(
        [_attempt1_direct, _attempt2_json_patch, _attempt3_autodetect], start=1
    ):
        try:
            return fn(h5_path)
        except Exception as exc:
            errors.append(f"Attempt {attempt} ({fn.__name__}): {exc}")

    combined = "\n".join(errors)
    raise RuntimeError(
        f"All fusion-model loading strategies failed:\n{combined}\n\n"
        "── Diagnostic snippet (run in a Python shell) ──\n"
        "import h5py, json\n"
        "with h5py.File('<fusion_model.h5>', 'r') as f:\n"
        "    print(json.dumps(json.loads(f.attrs['model_config']), indent=2))\n"
        "    f['model_weights'].visititems(lambda n,o: print(n, getattr(o,'shape','')))\n"
    )


# ═══════════════════════════════════════
# PYTORCH MODEL
# ═══════════════════════════════════════
class CrossStitchUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.eye(2))

    def forward(self, fm, fd):
        return (self.alpha[0, 0] * fm + self.alpha[0, 1] * fd,
                self.alpha[1, 0] * fm + self.alpha[1, 1] * fd)


class CrossStitchMultitaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2))
        self.d_conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2))
        self.cs1 = CrossStitchUnit(32)
        self.m_conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2))
        self.d_conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2))
        self.cs2 = CrossStitchUnit(64)
        self.m_conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2))
        self.d_conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2))
        self.cs3 = CrossStitchUnit(128)
        self.m_conv4 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256), nn.AdaptiveAvgPool2d((4, 4)))
        self.d_conv4 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256), nn.AdaptiveAvgPool2d((4, 4)))
        self.cs4 = CrossStitchUnit(256)
        self.m_fc = nn.Sequential(nn.Linear(4096, 512), nn.ReLU(), nn.Dropout(0.5),
                                   nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3))
        self.d_fc = nn.Sequential(nn.Linear(4096, 512), nn.ReLU(), nn.Dropout(0.5),
                                   nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3))
        self.maturity_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 3))
        self.disease_head  = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 4))

    def forward(self, x):
        fm, fd = self.cs1(self.m_conv1(x), self.d_conv1(x))
        fm, fd = self.cs2(self.m_conv2(fm), self.d_conv2(fd))
        fm, fd = self.cs3(self.m_conv3(fm), self.d_conv3(fd))
        fm, fd = self.cs4(self.m_conv4(fm), self.d_conv4(fd))
        fm = self.m_fc(fm.view(fm.size(0), -1))
        fd = self.d_fc(fd.view(fd.size(0), -1))
        return self.maturity_head(fm), self.disease_head(fd)


tf_resnet = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    pooling='avg'
)

tf_densenet = tf.keras.applications.DenseNet121(
    weights='imagenet',
    include_top=False,
    pooling='avg'
)

def load_models():
    pt_model = CrossStitchMultitaskModel()
    pt_model.load_state_dict(torch.load(MULTITASK_MODEL_PATH, map_location=device))
    pt_model.eval()
    fusion = load_fusion_model(FUSION_MODEL_PATH)
    return pt_model, fusion


model, fusion_model = load_models()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ═══════════════════════════════════════
# XAI METHODS
# ═══════════════════════════════════════
def gradcam(model, x, target, branch):
    layer = model.m_conv4 if branch == 'm' else model.d_conv4
    activations, gradients = [], []
    h1 = layer.register_forward_hook(lambda m, i, o: activations.append(o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0]))
    m_out, d_out = model(x)
    out = m_out if branch == 'm' else d_out
    model.zero_grad()
    out[0, target].backward()
    grads   = gradients[0][0].cpu().detach().numpy()
    acts    = activations[0][0].cpu().detach().numpy()
    weights = grads.mean(axis=(1, 2))
    cam = np.einsum('c,chw->hw', weights, acts)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = np.power(cam, 0.6)
    h1.remove(); h2.remove()
    return cam


def integrated_gradients(model, x, target, branch, steps=50):
    baseline = torch.zeros_like(x)
    grad_sum = np.zeros_like(x.cpu().numpy())
    for i in range(1, steps + 1):
        interp = (baseline + (i / steps) * (x - baseline)).clone().requires_grad_(True)
        m_out, d_out = model(interp)
        out = m_out if branch == 'm' else d_out
        model.zero_grad()
        out[0, target].backward()
        grad_sum += interp.grad.cpu().detach().numpy()
    avg_grad = grad_sum / steps
    ig = (x.cpu().detach().numpy() - baseline.cpu().numpy()) * avg_grad
    ig = np.mean(np.abs(ig[0]), axis=0)
    ig = (ig - ig.min()) / (ig.max() - ig.min() + 1e-8)
    ig = np.power(ig, 0.55)
    return ig


def occlusion(model, x, target, branch, patch=16, stride=8):
    with torch.no_grad():
        m_out, d_out = model(x)
        out       = m_out if branch == 'm' else d_out
        base_prob = torch.softmax(out, 1)[0, target].item()
    heatmap = np.zeros((IMG_SIZE, IMG_SIZE))
    count   = np.zeros((IMG_SIZE, IMG_SIZE))
    for i in range(0, IMG_SIZE - patch + 1, stride):
        for j in range(0, IMG_SIZE - patch + 1, stride):
            occ_x = x.clone()
            occ_x[:, :, i:i + patch, j:j + patch] = 0
            with torch.no_grad():
                m_out, d_out = model(occ_x)
                out  = m_out if branch == 'm' else d_out
                prob = torch.softmax(out, 1)[0, target].item()
            drop = base_prob - prob
            heatmap[i:i + patch, j:j + patch] += drop
            count  [i:i + patch, j:j + patch] += 1
    count    = np.maximum(count, 1)
    heatmap /= count
    heatmap  = np.maximum(heatmap, 0)
    heatmap  = cv2.GaussianBlur(heatmap, (7, 7), 0)
    heatmap  = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap  = np.power(heatmap, 0.65)
    return heatmap


def overlay_cam(img_np, cam, colormap=cv2.COLORMAP_INFERNO, alpha=0.55):
    img_bgr = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    heat    = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    fused   = cv2.addWeighted(img_bgr, alpha, heat, 1 - alpha, 0)
    return cv2.cvtColor(fused, cv2.COLOR_BGR2RGB)


def make_pure_heatmap(cam, colormap=cv2.COLORMAP_INFERNO):
    heat_bgr = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def prob_bars_html(probs, labels, colors):
    rows = ""
    for label, prob, color in zip(labels, probs, colors):
        pct = float(prob) * 100
        rows += f"""<div class="prob-row">
            <span class="prob-nm">{label}</span>
            <div class="prob-bg"><div class="prob-fg" style="width:{pct:.1f}%;background:{color};"></div></div>
            <span class="prob-pct">{pct:.1f}%</span>
        </div>"""
    return rows


# ═══════════════════════════════════════
# ADVISORY HELPERS
# ═══════════════════════════════════════
def _infer_graph_mode(disease: str, m_conf: float, d_conf: float) -> str:
    if m_conf < CONFIDENCE_THRESHOLD or d_conf < CONFIDENCE_THRESHOLD:
        return "uncertain"
    if disease.strip().title() == "Healthy":
        return "harvest"
    return "treatment"


def _mode_pill_html(mode: str) -> str:
    labels = {
        "harvest":   ("🌿 Harvest Branch",   "adv-mode-harvest"),
        "treatment": ("💊 Treatment Branch", "adv-mode-treatment"),
        "uncertain": ("⚠ Uncertain Branch",  "adv-mode-uncertain"),
        "fallback":  ("⚙ Rule-Based",        "adv-mode-fallback"),
    }
    label, css = labels.get(mode, ("— Unknown", "adv-mode-fallback"))
    return f'<span class="adv-mode-pill {css}">{label}</span>'


def _format_advice_html(advice: str) -> str:
    import re
    section_re = re.compile(r'^(\d+\.\s+[A-Z][A-Z &]+)$')
    out = []
    for line in advice.splitlines():
        stripped = line.strip()
        if section_re.match(stripped):
            out.append(f'<span class="adv-section">{stripped}</span>')
        elif stripped == "":
            out.append("<br>")
        else:
            out.append(stripped + "<br>")
    return "\n".join(out)


# ═══════════════════════════════════════════════════════════════
# PDF REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════

def _np_to_rl_image(arr: np.ndarray, width_mm: float, height_mm: float) -> RLImage:
    """Convert a numpy RGB array to a ReportLab Image flowable."""
    pil_img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return RLImage(buf, width=width_mm * mm, height=height_mm * mm)


def generate_pdf_report(
    img_np: np.ndarray,
    maturity_class: str,
    disease_class: str,
    m_conf: float,
    d_conf: float,
    m_probs: np.ndarray,
    d_probs: np.ndarray,
    cam_mat: np.ndarray,
    cam_dis: np.ndarray,
    ig_mat: np.ndarray,
    ig_dis: np.ndarray,
    occ_mat: np.ndarray,
    occ_dis: np.ndarray,
    advisory_text: str,
    graph_mode: str,
) -> bytes:
    """Build and return the PDF report as bytes."""

    buf = io.BytesIO()

    # ── Colour palette ─────────────────────────────────────────
    C_GREEN  = colors.HexColor("#1a7a4a")
    C_TEAL   = colors.HexColor("#0e7c7b")
    C_AMBER  = colors.HexColor("#b37400")
    C_DARK   = colors.HexColor("#0d1117")
    C_PANEL  = colors.HexColor("#f4f6f8")
    C_BORDER = colors.HexColor("#d0d7de")
    C_MUTED  = colors.HexColor("#57606a")

    # ── Doc setup ──────────────────────────────────────────────
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="GuavaXAI Analysis Report",
        author="GuavaXAI Explainability Dashboard",
    )

    W = A4[0] - 36 * mm   # usable width

    # ── Styles ─────────────────────────────────────────────────
    base = getSampleStyleSheet()

    S = {
        "cover_title": ParagraphStyle(
            "cover_title", parent=base["Normal"],
            fontName="Helvetica-Bold", fontSize=22,
            textColor=C_DARK, leading=28, spaceAfter=4
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub", parent=base["Normal"],
            fontName="Helvetica", fontSize=10,
            textColor=C_MUTED, leading=14, spaceAfter=2
        ),
        "section_hdr": ParagraphStyle(
            "section_hdr", parent=base["Normal"],
            fontName="Helvetica-Bold", fontSize=12,
            textColor=C_TEAL, leading=16,
            spaceBefore=14, spaceAfter=4,
        ),
        "sub_hdr": ParagraphStyle(
            "sub_hdr", parent=base["Normal"],
            fontName="Helvetica-Bold", fontSize=9,
            textColor=C_MUTED, leading=12,
            spaceBefore=8, spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"],
            fontName="Helvetica", fontSize=9,
            textColor=C_DARK, leading=14, spaceAfter=3,
        ),
        "mono": ParagraphStyle(
            "mono", parent=base["Normal"],
            fontName="Courier", fontSize=8,
            textColor=C_DARK, leading=12,
        ),
        "caption": ParagraphStyle(
            "caption", parent=base["Normal"],
            fontName="Helvetica-Oblique", fontSize=8,
            textColor=C_MUTED, leading=11, alignment=TA_CENTER,
        ),
        "adv_body": ParagraphStyle(
            "adv_body", parent=base["Normal"],
            fontName="Helvetica", fontSize=9,
            textColor=C_DARK, leading=15, spaceAfter=4,
        ),
        "adv_section": ParagraphStyle(
            "adv_section", parent=base["Normal"],
            fontName="Helvetica-Bold", fontSize=8,
            textColor=C_TEAL, leading=12,
            spaceBefore=8, spaceAfter=2,
        ),
    }

    # ── Page-number footer callback ────────────────────────────
    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(C_MUTED)
        ts = datetime.datetime.now().strftime("%d %b %Y  %H:%M")
        canvas.drawString(18 * mm, 10 * mm, f"GuavaXAI Explainability Dashboard  ·  {ts}")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    story = []

    # ══════════════════════════════════════
    # SECTION 1 — COVER / HEADER
    # ══════════════════════════════════════
    ts_str = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
    header_data = [[
        Paragraph("GuavaXAI", S["cover_title"]),
        Paragraph(f"Analysis Report<br/><font size='8' color='#57606a'>{ts_str}</font>",
                  S["cover_sub"]),
    ]]
    header_tbl = Table(header_data, colWidths=[W * 0.55, W * 0.45])
    header_tbl.setStyle(TableStyle([
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",       (1, 0), (1, 0),   "RIGHT"),
        ("LINEBELOW",   (0, 0), (-1, 0),  1.5, C_TEAL),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 6 * mm))

    # ══════════════════════════════════════
    # SECTION 2 — UPLOADED IMAGE
    # ══════════════════════════════════════
    story.append(Paragraph("1. Input Image", S["section_hdr"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceAfter=6))

    img_rl = _np_to_rl_image(img_np, 70, 70)
    img_note = Paragraph(
        "Original image uploaded by the user and resized to 224×224 px for model inference.",
        S["body"]
    )
    img_row = Table([[img_rl, img_note]], colWidths=[75 * mm, W - 75 * mm])
    img_row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (1, 0), (1, 0), 10),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
    ]))
    story.append(img_row)
    story.append(Spacer(1, 6 * mm))

    # ══════════════════════════════════════
    # SECTION 3 — PREDICTIONS
    # ══════════════════════════════════════
    story.append(Paragraph("2. Model Predictions", S["section_hdr"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceAfter=6))

    # Summary cards (2-column table)
    def _pred_cell(label, value, conf, accent):
        return [
            Paragraph(label.upper(), ParagraphStyle(
                "pl", fontName="Helvetica-Bold", fontSize=7, textColor=C_MUTED, leading=10
            )),
            Paragraph(value, ParagraphStyle(
                "pv", fontName="Helvetica-Bold", fontSize=16,
                textColor=accent, leading=20
            )),
            Paragraph(f"Confidence: {conf*100:.1f}%", ParagraphStyle(
                "pc", fontName="Courier", fontSize=8, textColor=C_MUTED, leading=11
            )),
        ]

    pred_tbl = Table(
        [[
            _pred_cell("Maturity Stage", maturity_class, m_conf, C_GREEN),
            _pred_cell("Disease Type",   disease_class,  d_conf, C_AMBER),
        ]],
        colWidths=[W / 2 - 3 * mm, W / 2 - 3 * mm],
        hAlign="LEFT"
    )
    pred_tbl.setStyle(TableStyle([
        ("BOX",         (0, 0), (0, 0), 0.75, C_GREEN),
        ("BOX",         (1, 0), (1, 0), 0.75, C_AMBER),
        ("BACKGROUND",  (0, 0), (0, 0), colors.HexColor("#f0fdf4")),
        ("BACKGROUND",  (1, 0), (1, 0), colors.HexColor("#fffbeb")),
        ("ROUNDEDCORNERS", [6]),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(pred_tbl)
    story.append(Spacer(1, 5 * mm))

    # Probability distribution tables
    story.append(Paragraph("Probability Distributions", S["sub_hdr"]))

    def _prob_table(labels, probs, title):
        rows = [[Paragraph(title, ParagraphStyle(
            "pt", fontName="Helvetica-Bold", fontSize=8, textColor=C_MUTED
        )), "", ""]]
        rows.append([
            Paragraph("Class", ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=8, textColor=C_DARK)),
            Paragraph("Probability", ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=8, textColor=C_DARK)),
            Paragraph("Confidence", ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=8, textColor=C_DARK)),
        ])
        for lbl, p in zip(labels, probs):
            rows.append([
                Paragraph(lbl, S["mono"]),
                Paragraph(f"{p*100:.2f}%", S["mono"]),
                Paragraph("█" * int(p * 20) + "░" * (20 - int(p * 20)), ParagraphStyle(
                    "bar", fontName="Courier", fontSize=7,
                    textColor=C_TEAL if p == max(probs) else C_MUTED
                )),
            ])
        t = Table(rows, colWidths=[40 * mm, 25 * mm, W / 2 - 70 * mm])
        t.setStyle(TableStyle([
            ("SPAN",        (0, 0), (2, 0)),
            ("LINEBELOW",   (0, 1), (2, 1), 0.5, C_BORDER),
            ("BACKGROUND",  (0, 1), (2, 1), C_PANEL),
            ("TOPPADDING",  (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return t

    prob_row_tbl = Table([
        [
            _prob_table(MATURITY_CLASSES, m_probs, "Maturity Distribution"),
            _prob_table(DISEASE_CLASSES,  d_probs, "Disease Distribution"),
        ]
    ], colWidths=[W / 2 - 3 * mm, W / 2 - 3 * mm])
    prob_row_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (1, 0), (1, 0), 6),
    ]))
    story.append(prob_row_tbl)
    story.append(Spacer(1, 6 * mm))

    # ══════════════════════════════════════
    # SECTION 4 — XAI MAPS
    # ══════════════════════════════════════
    story.append(Paragraph("3. Explainability (XAI) Maps", S["section_hdr"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceAfter=6))

    xai_methods = [
        ("Grad-CAM",
         "Weighted combination of final conv feature maps. Bright = attended regions.",
         cam_mat, cam_dis,
         cv2.COLORMAP_INFERNO),
        ("Integrated Gradients",
         "Accumulated gradients from baseline to input (50 steps). Satisfies completeness axiom.",
         ig_mat, ig_dis,
         cv2.COLORMAP_VIRIDIS),
        ("Occlusion Sensitivity",
         "16x16 patch occlusion with stride 8. High intensity = critical region.",
         occ_mat, occ_dis,
         cv2.COLORMAP_HOT),
    ]

    colmap_lookup = {
        cv2.COLORMAP_INFERNO: cv2.COLORMAP_INFERNO,
        cv2.COLORMAP_VIRIDIS: cv2.COLORMAP_VIRIDIS,
        cv2.COLORMAP_HOT:     cv2.COLORMAP_HOT,
    }

    IMG_W = 38
    IMG_H = 38

    for method_name, description, cam_m, cam_d, cmap in xai_methods:
        story.append(Paragraph(method_name, S["sub_hdr"]))
        story.append(Paragraph(description, S["body"]))

        ov_mat = _np_to_rl_image(overlay_cam(img_np, cam_m, cmap), IMG_W, IMG_H)
        hm_mat = _np_to_rl_image(make_pure_heatmap(cam_m, cmap),   IMG_W, IMG_H)
        ov_dis = _np_to_rl_image(overlay_cam(img_np, cam_d, cmap), IMG_W, IMG_H)
        hm_dis = _np_to_rl_image(make_pure_heatmap(cam_d, cmap),   IMG_W, IMG_H)

        xai_tbl = Table(
            [
                [
                    Paragraph("Maturity — Overlay",  S["caption"]),
                    Paragraph("Maturity — Heatmap",  S["caption"]),
                    Paragraph("Disease — Overlay",   S["caption"]),
                    Paragraph("Disease — Heatmap",   S["caption"]),
                ],
                [ov_mat, hm_mat, ov_dis, hm_dis],
            ],
            colWidths=[(IMG_W + 2) * mm] * 4,
        )
        xai_tbl.setStyle(TableStyle([
            ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",   (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
            ("LINEABOVE",    (0, 1), (-1, 1),  0.3, C_BORDER),
        ]))
        story.append(xai_tbl)
        story.append(Spacer(1, 4 * mm))

    story.append(Spacer(1, 2 * mm))

    # ══════════════════════════════════════
    # SECTION 5 — AI ADVISORY
    # ══════════════════════════════════════
    story.append(Paragraph("4. AI Advisory", S["section_hdr"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceAfter=6))

    mode_colors = {
        "harvest":   (colors.HexColor("#f0fdf4"), C_GREEN),
        "treatment": (colors.HexColor("#fffbeb"), C_AMBER),
        "uncertain": (colors.HexColor("#f0fdfa"), C_TEAL),
        "fallback":  (colors.HexColor("#f6f8fa"), C_MUTED),
    }
    mode_labels = {
        "harvest":   "Harvest Branch",
        "treatment": "Treatment Branch",
        "uncertain": "Uncertain — Low Confidence",
        "fallback":  "Rule-Based Fallback",
    }
    bg_col, accent_col = mode_colors.get(graph_mode, (C_PANEL, C_MUTED))
    mode_label = mode_labels.get(graph_mode, "Unknown")

    meta_tbl = Table([[
        Paragraph("Advisory Mode", ParagraphStyle(
            "am_lbl", fontName="Helvetica-Bold", fontSize=7, textColor=C_MUTED
        )),
        Paragraph(mode_label, ParagraphStyle(
            "am_val", fontName="Helvetica-Bold", fontSize=9, textColor=accent_col
        )),
        Paragraph(
            f"Maturity conf: {m_conf*100:.1f}%   Disease conf: {d_conf*100:.1f}%",
            ParagraphStyle("am_conf", fontName="Courier", fontSize=8, textColor=C_MUTED)
        ),
    ]], colWidths=[35 * mm, 55 * mm, W - 90 * mm])
    meta_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), bg_col),
        ("BOX",          (0, 0), (-1, 0), 0.75, accent_col),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 4 * mm))

    # Format advisory text: detect section headers (ALL CAPS lines) vs body
    import re
    section_re = re.compile(r'^(\d+\.\s+[A-Z][A-Z &]+)$')
    for line in advisory_text.splitlines():
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 2 * mm))
        elif section_re.match(stripped):
            story.append(Paragraph(stripped, S["adv_section"]))
        else:
            story.append(Paragraph(stripped, S["adv_body"]))

    story.append(Spacer(1, 6 * mm))

    # ══════════════════════════════════════
    # SECTION 6 — METHODOLOGY NOTE
    # ══════════════════════════════════════
    story.append(Paragraph("5. Methodology", S["section_hdr"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceAfter=6))

    methodology = [
        ("Model Architecture", "Cross-Stitch Multitask CNN with shared feature extraction "
         "and task-specific branches for maturity (3 classes) and disease (4 classes) classification."),
        ("Calibration", f"Temperature scaling (T={TEMPERATURE}) with label smoothing "
         f"(eps={SMOOTH_EPS}) and confidence cap ({CONF_CAP}) applied to raw logits."),
        ("Grad-CAM", "Class Activation Maps computed from the final convolutional block "
         "(m_conv4 / d_conv4). Gamma correction (0.6) applied for visual clarity."),
        ("Integrated Gradients", "50-step path integration from zero baseline. "
         "Satisfies sensitivity and implementation invariance axioms."),
        ("Occlusion Sensitivity", "16x16 pixel sliding patches with stride 8. "
         "Confidence drop averaged over overlapping patches with Gaussian smoothing (sigma=7)."),
        ("Advisory", "Developed a LangGraph agent pipeline with automated fallback for low-confidence outputs and execution failures."),
    ]

    for term, desc in methodology:
        row_tbl = Table([[
            Paragraph(term, ParagraphStyle(
                "mt", fontName="Helvetica-Bold", fontSize=8, textColor=C_DARK
            )),
            Paragraph(desc, S["body"]),
        ]], colWidths=[38 * mm, W - 38 * mm])
        row_tbl.setStyle(TableStyle([
            ("VALIGN",       (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING",   (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
            ("LINEBELOW",    (0, 0), (-1, 0),  0.3, C_BORDER),
        ]))
        story.append(row_tbl)

    # ── Build ──────────────────────────────────────────────────
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hdr">
    <span class="hdr-title">GuavaXAI</span>
    <span class="hdr-badge">Explainability Dashboard</span>
    <span class="hdr-sub">Cross-Stitch Multitask · Grad-CAM · IG · Occlusion · LangGraph Advisory</span>
</div>
""", unsafe_allow_html=True)

left, center, right = st.columns([1.05, 2.6, 1.55], gap="small")

# ══ LEFT ════════════════════════════════════════════════════════
with left:
    st.markdown('<p class="slbl">📂 Input Image</p>', unsafe_allow_html=True)
    file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"],
                            label_visibility="collapsed")
    if file:
        img         = Image.open(file).convert("RGB")
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        img_np      = np.array(img_resized)
        x           = transform(img).unsqueeze(0).to(device)

        st.markdown("""<div class="img-wrap">
            <div class="img-wrap-hdr">
                <span class="dot" style="background:#39d98a;"></span> Original
            </div>""", unsafe_allow_html=True)
        st.image(img_np, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="margin-top:50px;text-align:center;color:var(--muted);">
            <div style="font-size:2.5rem;">🍈</div>
            <p style="font-size:.75rem;margin-top:8px;line-height:1.6;">
                Upload a guava image<br>to begin analysis</p>
        </div>""", unsafe_allow_html=True)

# ══ CENTER ══════════════════════════════════════════════════════
with center:
    if file:
        with torch.no_grad():
            m_out_raw, d_out_raw = model(x)

        m_probs = calibrate_probs(m_out_raw, n_classes=len(MATURITY_CLASSES))
        d_probs = calibrate_probs(d_out_raw, n_classes=len(DISEASE_CLASSES))

        m_cls  = int(np.argmax(m_probs))
        d_cls  = int(np.argmax(d_probs))
        m_conf = float(m_probs[m_cls])
        d_conf = float(d_probs[d_cls])

        # ── Store in session_state so right panel can access ──
        st.session_state["m_cls"]   = m_cls
        st.session_state["d_cls"]   = d_cls
        st.session_state["m_conf"]  = m_conf
        st.session_state["d_conf"]  = d_conf
        st.session_state["m_probs"] = m_probs
        st.session_state["d_probs"] = d_probs
        st.session_state["img_np"]  = img_np
        st.session_state["x"]       = x

        bc, ec = st.columns([1, 2])
        with bc:
            st.markdown('<p class="slbl">Task branch</p>', unsafe_allow_html=True)
            branch_label = st.radio("branch", ["Maturity", "Disease"],
                                    horizontal=True, label_visibility="collapsed")
        branch       = 'm' if branch_label == "Maturity" else 'd'
        xai_target   = m_cls if branch == 'm' else d_cls
        xai_cls_name = MATURITY_CLASSES[m_cls] if branch == 'm' else DISEASE_CLASSES[d_cls]
        acc_color    = "var(--green)" if branch == 'm' else "var(--amber)"
        with ec:
            st.markdown(f"""
            <div style="padding:4px 0 0 4px;">
                <span style="font-family:'JetBrains Mono',monospace;font-size:.6rem;
                             letter-spacing:.1em;text-transform:uppercase;color:var(--muted);">
                    Explaining →</span>
                <span style="font-family:'DM Serif Display',serif;font-style:italic;
                             font-size:1.1rem;color:{acc_color};margin-left:6px;">
                    {xai_cls_name}</span>
            </div>""", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["⚡ Grad-CAM", "∫ Integrated Gradients", "🔲 Occlusion"])

        with tab1:
            st.markdown("""<div class="xai-desc"><b>Grad-CAM</b> — weighted combination of the
            final conv feature maps. Bright regions = where the network focused attention.</div>""",
            unsafe_allow_html=True)
            with st.spinner("Computing…"):
                cam = gradcam(model, x.clone(), xai_target, branch)
                # Also compute both branches for PDF
                if "cam_mat" not in st.session_state:
                    st.session_state["cam_mat"] = gradcam(model, x.clone(), m_cls, 'm')
                if "cam_dis" not in st.session_state:
                    st.session_state["cam_dis"] = gradcam(model, x.clone(), d_cls, 'd')
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<p class="slbl">Overlay</p>', unsafe_allow_html=True)
                st.image(overlay_cam(img_np, cam, cv2.COLORMAP_INFERNO), use_container_width=True)
            with c2:
                st.markdown('<p class="slbl">Heatmap</p>', unsafe_allow_html=True)
                st.image(make_pure_heatmap(cam, cv2.COLORMAP_INFERNO), use_container_width=True)

        with tab2:
            st.markdown("""<div class="xai-desc"><b>Integrated Gradients</b> — accumulates gradients
            along a path from a black baseline. Satisfies completeness axiom (50 steps).</div>""",
            unsafe_allow_html=True)
            with st.spinner("Computing…"):
                ig = integrated_gradients(model, x.clone(), xai_target, branch, steps=50)
                if "ig_mat" not in st.session_state:
                    st.session_state["ig_mat"] = integrated_gradients(model, x.clone(), m_cls, 'm', steps=50)
                if "ig_dis" not in st.session_state:
                    st.session_state["ig_dis"] = integrated_gradients(model, x.clone(), d_cls, 'd', steps=50)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<p class="slbl">Overlay</p>', unsafe_allow_html=True)
                st.image(overlay_cam(img_np, ig, cv2.COLORMAP_VIRIDIS), use_container_width=True)
            with c2:
                st.markdown('<p class="slbl">Heatmap</p>', unsafe_allow_html=True)
                st.image(make_pure_heatmap(ig, cv2.COLORMAP_VIRIDIS), use_container_width=True)

        with tab3:
            st.markdown("""<div class="xai-desc"><b>Occlusion Sensitivity</b> — patches out 16×16
            regions and measures the confidence drop. High intensity = critical region.</div>""",
            unsafe_allow_html=True)
            with st.spinner("Computing…"):
                occ = occlusion(model, x.clone(), xai_target, branch, patch=16, stride=8)
                if "occ_mat" not in st.session_state:
                    st.session_state["occ_mat"] = occlusion(model, x.clone(), m_cls, 'm', patch=16, stride=8)
                if "occ_dis" not in st.session_state:
                    st.session_state["occ_dis"] = occlusion(model, x.clone(), d_cls, 'd', patch=16, stride=8)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<p class="slbl">Overlay</p>', unsafe_allow_html=True)
                st.image(overlay_cam(img_np, occ, cv2.COLORMAP_HOT), use_container_width=True)
            with c2:
                st.markdown('<p class="slbl">Heatmap</p>', unsafe_allow_html=True)
                st.image(make_pure_heatmap(occ, cv2.COLORMAP_HOT), use_container_width=True)

    else:
        st.markdown("""
        <div style="margin-top:90px;text-align:center;color:var(--muted);padding:20px;">
            <p style="font-family:'DM Serif Display',serif;font-style:italic;
                      font-size:1.1rem;color:var(--txt);">XAI maps appear here</p>
            <p style="font-size:.75rem;margin-top:6px;line-height:1.6;">
                Upload an image on the left to run inference<br>
                and generate explainability visualisations.</p>
        </div>""", unsafe_allow_html=True)

# ══ RIGHT ═══════════════════════════════════════════════════════
with right:
    if file:
        st.markdown('<p class="slbl">🔍 Predictions</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="pred-card mat">
            <span class="pred-lbl">Maturity Stage</span>
            <span class="pred-val">{MATURITY_CLASSES[m_cls]}</span>
            <span class="pred-conf">Confidence · {m_conf*100:.1f}%</span>
            <div class="bar-wrap">
                <div class="bar-fill" style="width:{m_conf*100:.1f}%;
                     background:linear-gradient(90deg,#39d98a,#4ecdc4);"></div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="pred-card dis" style="margin-top:8px;">
            <span class="pred-lbl">Disease Type</span>
            <span class="pred-val">{DISEASE_CLASSES[d_cls]}</span>
            <span class="pred-conf">Confidence · {d_conf*100:.1f}%</span>
            <div class="bar-wrap">
                <div class="bar-fill" style="width:{d_conf*100:.1f}%;
                     background:linear-gradient(90deg,#f5a623,#ffe08a);"></div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<p class="slbl" style="margin-top:12px;">Maturity distribution</p>',
                    unsafe_allow_html=True)
        st.markdown(prob_bars_html(m_probs, MATURITY_CLASSES, MATURITY_COLORS),
                    unsafe_allow_html=True)

        st.markdown('<p class="slbl" style="margin-top:10px;">Disease distribution</p>',
                    unsafe_allow_html=True)
        st.markdown(prob_bars_html(d_probs, DISEASE_CLASSES, DISEASE_COLORS),
                    unsafe_allow_html=True)

        graph_mode = _infer_graph_mode(DISEASE_CLASSES[d_cls], m_conf, d_conf)

        st.markdown(f"""
        <div class="adv-header" style="margin-top:12px;">
            <p class="slbl" style="margin:0;">🤖 AI Advisory</p>
            {_mode_pill_html(graph_mode)}
        </div>""", unsafe_allow_html=True)

        if st.button("Generate Expert Advice", use_container_width=True):
            with st.spinner("Running LangGraph pipeline…"):
                advice, used_llm = generate_advisory(
                    MATURITY_CLASSES[m_cls],
                    DISEASE_CLASSES[d_cls],
                    float(m_conf),
                    float(d_conf),
                )
            st.session_state["advice"]     = advice
            st.session_state["graph_mode"] = graph_mode

            if graph_mode == "uncertain":
                st.markdown("""
                <div class="advisory-uncertain">
                    ⚠ Low confidence — uncertainty branch activated.
                    Manual inspection recommended.
                </div>""", unsafe_allow_html=True)

            if not used_llm:
                st.markdown('', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="advisory-box">
                {_format_advice_html(advice)}
            </div>""", unsafe_allow_html=True)

        # ══════════════════════════════════════
        # PDF DOWNLOAD BUTTON
        # ══════════════════════════════════════
        st.markdown('<p class="slbl" style="margin-top:14px;">📄 Report</p>',
                    unsafe_allow_html=True)

        # Ensure all XAI maps exist in session_state before allowing download
        xai_ready = all(k in st.session_state for k in
                        ("cam_mat", "cam_dis", "ig_mat", "ig_dis", "occ_mat", "occ_dis"))
        adv_ready = "advice" in st.session_state

        if not xai_ready:
            st.markdown(
                '<p style="font-size:.68rem;color:var(--muted);line-height:1.5;">'
                'Open all three XAI tabs (Grad-CAM, IG, Occlusion) and generate advice '
                'to unlock the full PDF report.</p>',
                unsafe_allow_html=True
            )
        else:
            advisory_for_pdf = st.session_state.get(
                "advice",
                "Advisory not generated yet. Click 'Generate Expert Advice' first."
            )
            gmode_for_pdf = st.session_state.get("graph_mode", graph_mode)

            with st.spinner("Building PDF…"):
                pdf_bytes = generate_pdf_report(
                    img_np          = st.session_state["img_np"],
                    maturity_class  = MATURITY_CLASSES[m_cls],
                    disease_class   = DISEASE_CLASSES[d_cls],
                    m_conf          = m_conf,
                    d_conf          = d_conf,
                    m_probs         = m_probs,
                    d_probs         = d_probs,
                    cam_mat         = st.session_state["cam_mat"],
                    cam_dis         = st.session_state["cam_dis"],
                    ig_mat          = st.session_state["ig_mat"],
                    ig_dis          = st.session_state["ig_dis"],
                    occ_mat         = st.session_state["occ_mat"],
                    occ_dis         = st.session_state["occ_dis"],
                    advisory_text   = advisory_for_pdf,
                    graph_mode      = gmode_for_pdf,
                )

            ts_fn = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label           = "⬇ Download Full PDF Report",
                data            = pdf_bytes,
                file_name       = f"guavaxai_report_{ts_fn}.pdf",
                mime            = "application/pdf",
                use_container_width=True,
            )

    else:
        st.markdown("""
        <div style="margin-top:90px;text-align:center;color:var(--muted);">
            <p style="font-size:.75rem;line-height:1.6;">
                Prediction results<br>will appear here.</p>
        </div>""", unsafe_allow_html=True)