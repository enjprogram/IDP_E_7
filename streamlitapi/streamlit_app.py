"""
streamlitapi/streamlit_app.py
==============================
AI Services dashboard — two sections, five pages.

Run from inside the streamlitapi/ folder:
    streamlit run streamlit_app.py

The app communicates with the FastAPI backend (fastapi/app.py) over HTTP.
All heavy ML dependencies live in the fastapi environment — this file
only requires: streamlit, requests, pandas.

Sidebar layout
--------------
  ── Computer Vision ──
    Species Classifier

  ── NLP Services ──
    Ticket Classifier
    Named Entity Recognition
    Draft Response

  ── Pipelines ──
    Full NLP Pipeline
"""
import os
import streamlit as st
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


# -- Page config -------------------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Services Hub",
    page_icon="AI/ML",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE   = os.getenv("API_URL", "http://127.0.0.1:8000")
#"http://127.0.0.1:8000"
CATEGORIES = ["Delivery", "Refund", "Account", "Product Issue", "Other"]

CATEGORY_COLORS = {
    "Delivery":      "#3B82F6",
    "Refund":        "#F59E0B",
    "Account":       "#10B981",
    "Product Issue": "#EF4444",
    "Other":         "#8B5CF6",
}
CATEGORY_ICONS = {
    "Delivery":      "📦",
    "Refund":        "💰",
    "Account":       "👤",
    "Product Issue": "⚙️",
    "Other":         "💬",
}
ENTITY_COLORS = {
    "ORDER_ID": "#3B82F6",
    "DATE":     "#10B981",
    "EMAIL":    "#F59E0B",
}

# --- CSS --------------------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

[data-testid="stSidebar"] { background: #0F172A !important; border-right: 1px solid #1E293B; }
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }

.sidebar-section {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px; font-weight: 600;
    letter-spacing: 0.15em; text-transform: uppercase; color: #475569 !important;
    padding: 16px 0 6px 0; border-top: 1px solid #1E293B; margin-top: 8px;
}

.page-header { border-bottom: 2px solid #E2E8F0; padding-bottom: 12px; margin-bottom: 24px; }
.page-header h1 { font-family: 'IBM Plex Mono', monospace; font-size: 22px; font-weight: 600; color: #0F172A; margin: 0; }
.page-header p  { font-size: 13px; color: #64748B; margin: 4px 0 0 0; }

.metric-card { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px 20px; }

.cat-pill {
    display: inline-block; padding: 4px 12px; border-radius: 999px;
    font-size: 13px; font-weight: 600; font-family: 'IBM Plex Sans', sans-serif;
}
.ent-chip {
    display: inline-block; padding: 3px 10px; border-radius: 4px;
    font-size: 12px; font-weight: 600; margin: 2px;
    font-family: 'IBM Plex Mono', monospace;
}
.draft-box {
    background: #F0F9FF; border-left: 4px solid #0EA5E9; border-radius: 0 8px 8px 0;
    padding: 16px 20px; font-size: 14px; line-height: 1.7; color: #0F172A;
}
.mono { font-family: 'IBM Plex Mono', monospace; }

.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 2px solid #E2E8F0; }
.stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Sans', sans-serif; font-size: 13px; padding: 8px 16px; }

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- Session state -------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "cnn"

# --- Sidebar --------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 0 16px 0;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:18px;font-weight:600;color:#F1F5F9;">⚡ AI Services</div>
        <div style="font-size:11px;color:#475569;margin-top:4px;">Unified ML Platform v2.0</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Computer Vision</div>', unsafe_allow_html=True)
    if st.button("Species Classifier", key="nav_cnn", use_container_width=True,
                 type="primary" if st.session_state.page == "cnn" else "secondary"):
        st.session_state.page = "cnn"; st.rerun()

    st.markdown('<div class="sidebar-section">NLP Services</div>', unsafe_allow_html=True)
    for page_key, label in [("nlp_classify", "Ticket Classifier"),
                              ("nlp_ner",      "Named Entity Recognition"),
                              ("nlp_draft",    "Draft Response")]:
        if st.button(label, key=f"nav_{page_key}", use_container_width=True,
                     type="primary" if st.session_state.page == page_key else "secondary"):
            st.session_state.page = page_key; st.rerun()

    st.markdown('<div class="sidebar-section">Pipelines</div>', unsafe_allow_html=True)
    if st.button("Full NLP Pipeline", key="nav_pipeline", use_container_width=True,
                 type="primary" if st.session_state.page == "pipeline" else "secondary"):
        st.session_state.page = "pipeline"; st.rerun()


    # show the current model used by the page
    st.markdown('<div class="sidebar-section">System</div>', unsafe_allow_html=True)
    try:
        info   = requests.get(f"{API_BASE}/health", timeout=2).json()
        models = info.get("models", {})
        nlp    = models.get("nlp", {})
        cnn    = models.get("cnn", {})
        page   = st.session_state.get("page", "")

        st.success("API online")

        if page == "cnn":
            st.caption(f"CNN: `{'loaded' if cnn.get('loaded') else 'not loaded'}`")
            st.caption(f"Classes: `{cnn.get('labels', 0)}`")
        elif page in ["nlp_classify", "nlp_ner", "nlp_draft", "pipeline"]:
            st.caption(f"NLP: `{nlp.get('classifier', '?')}`")
        else:
            st.caption(f"CNN: `{'loaded' if cnn.get('loaded') else 'not loaded'}`")
            st.caption(f"NLP: `{nlp.get('classifier', '?')}`")

    except Exception:
        st.error("API offline")
        st.caption(f"Expected at {API_BASE}")


# ── Helpers ────────────────────────────────────────────────────────────────────
def api_post(endpoint, **kwargs):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", timeout=30, **kwargs)
        r.raise_for_status(); return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API — is uvicorn running on port 8000?"
    except Exception as e:
        return None, str(e)

def api_get(endpoint):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=10)
        r.raise_for_status(); return r.json(), None
    except Exception as e:
        return None, str(e)

def render_category_badge(category, confidence=None):
    color    = CATEGORY_COLORS.get(category, "#6B7280")
    icon     = CATEGORY_ICONS.get(category, "💬")
    conf_str = f"  {confidence:.0%}" if confidence is not None else ""
    st.markdown(
        f'<span class="cat-pill" style="background:{color}20;color:{color};border:1px solid {color}40;">'
        f'{icon} {category}{conf_str}</span>', unsafe_allow_html=True)

def render_entities(entities):
    if not entities:
        st.caption("No entities detected."); return
    for e in entities:
        color = ENTITY_COLORS.get(e["label"], "#6B7280")
        st.markdown(
            f'<span class="ent-chip" style="background:{color}18;color:{color};border:1px solid {color}40;">'
            f'{e["label"]}</span> '
            f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:13px;">{e["text"]}</span>'
            f'<span style="color:#94A3B8;font-size:11px;margin-left:8px;">pos {e["start"]}–{e["end"]}</span>',
            unsafe_allow_html=True)
        st.markdown("")

def annotated_html(text, entities):
    if not entities:
        return f'<span style="font-size:14px;">{text}</span>'
    result, prev = "", 0
    for e in sorted(entities, key=lambda x: x["start"]):
        color  = ENTITY_COLORS.get(e["label"], "#6B7280")
        label  = e["label"]
        etext  = e["text"]
        start  = e["start"]
        end    = e["end"]
        result += text[prev:start]
        result += (
            '<mark style="background:' + color + '25;color:' + color +
            ';border-bottom:2px solid ' + color + ';padding:1px 3px;border-radius:3px;'
            'font-family:\'IBM Plex Mono\',monospace;font-size:13px;" title="' +
            label + '">' + etext + '</mark>'
        )
        prev = end
    result += text[prev:]
    return f'<div style="font-size:14px;line-height:1.8;">{result}</div>'


# --- MLflow artifact helpers ----------------------------------------------------------
import mlflow
import tempfile
from mlflow.tracking import MlflowClient

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_URI)


@st.cache_resource
def _get_all_bert_artifacts():
    """Download ALL BERT artifacts in one call — cached for entire session."""
    try:
        client   = MlflowClient()
        versions = client.get_latest_versions("bert-ticket-classifier", stages=["Production"])
        if not versions:
            return {}
        run_id = versions[0].run_id
        dst    = os.path.join(tempfile.gettempdir(), "streamlit_mlflow_bert")

        # Check which artifacts actually exist before downloading
        try:
            existing = mlflow.artifacts.list_artifacts(run_id=run_id)
            existing_names = {a.path for a in existing}
        except Exception:
            existing_names = set()

        results = {}
        for name in ["bert_model_card.json", "bert_history.json",
                     "classification_report.csv", "confusion_matrix.csv",
                     "test_pred_probs.npy", "test_y_true.npy"]:
            # Skip if we know the file doesn't exist
            if existing_names and name not in existing_names:
                results[name] = None
                continue
            try:
                results[name] = mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path=name, dst_path=dst
                )
            except Exception:
                results[name] = None
        return results
    except Exception:
        return {}


@st.cache_resource
def _download_cnn_artifact(run_id, artifact_path):
    """Download CNN artifact from MLflow — cached for session."""
    try:
        if not run_id:
            return None
        return mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
            dst_path=os.path.join(tempfile.gettempdir(), "streamlit_mlflow_cnn")
        )
    except Exception:
        return None
    
# ==============================================================================
# PAGE: CNN Classifier
# ==============================================================================
if st.session_state.page == "cnn":
    import os, json
    import numpy as np

    st.markdown("""<div class="page-header">
        <h1>Species Classifier</h1>
        <p>CNN · TensorFlow/Keras · 128×128 input · <span class="mono">POST /predict</span></p>
    </div>""", unsafe_allow_html=True)

    # ---Locate fastapi/ root and fastapi/data/ ----------------------------------
    # Inside Docker: FASTAPI_BASE=/app/fastapi (set in docker-compose.yml)
    # Local dev: falls back to relative path from streamlitapi/
    _FASTAPI_BASE  = os.getenv("FASTAPI_BASE",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fastapi"))
    )

    _FASTAPI_DATA  = os.path.join(_FASTAPI_BASE, "data")

    # --Load versions from MLflow registry --------------------------------
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(MLFLOW_URI)

    _registry     = {}   # version -> model info dict from MLflow
    _versions     = []   # list of version numbers as strings
    _prod_version = None

    try:
        _client   = MlflowClient()
        _all_vers = _client.search_model_versions("name='cnn-species-classifier'")
        for _v in _all_vers:
            _registry[str(_v.version)] = {
                "status":     _v.current_stage,
                "trained_on": str(_v.creation_timestamp),
                "run_id":     _v.run_id,
            }
        _versions     = sorted(_registry.keys(), key=lambda x: int(x))
        _prod_entries = [v for v, d in _registry.items() if d.get("status") == "Production"]
        _prod_version = _prod_entries[0] if _prod_entries else (_versions[-1] if _versions else None)
    except Exception as _e:
        st.warning(f"Could not connect to MLflow: {_e}")

    _has_insights = len(_versions) > 0

    tab_labels = ["Classify", "Model Insights"] if _has_insights else ["Classify"]
    tabs = st.tabs(tab_labels)

    # ======================================================================
    # TAB 1 — Classify
    # ======================================================================
    with tabs[0]:
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.markdown("#### Input Image")

            # ---Input mode toggle ---------------------------------------------------------
            _input_mode = st.radio(
                "Input source",
                ["Upload image", "Sample from dataset"],
                horizontal=True,
                help="Use 'Sample from dataset' to test with in-distribution training data"
            )

            _img_bytes  = None   # will hold the final bytes to send to API
            _img_display = None  # will hold image to preview

            if _input_mode == "Upload image":
                uploaded = st.file_uploader("Drag and drop or click to browse", type=["jpg","jpeg","png"])
                if uploaded:
                    _img_display = uploaded
                    uploaded.seek(0)
                    _img_bytes = uploaded.read()

            else:
                # ── Dataset sampler ───────────────────────────────────────
                # Default to fastapi/data/images_plant.npy relative to this file
                _npy_default = os.path.join(_FASTAPI_BASE, "data", "images_plant.npy")
                _npy_path_input = st.text_input(
                    "Path to images .npy file (relative to fastapi/data/ or absolute)",
                    value="data/images_plant.npy",
                    help="Relative path from fastapi/ folder, e.g. data/images_plant.npy"
                )
                # Resolve: if relative, anchor to fastapi/; if absolute use as-is
                _npy_path = _npy_path_input if os.path.isabs(_npy_path_input) \
                            else os.path.join(_FASTAPI_BASE, _npy_path_input)
                _npy_loaded = None
                _npy_labels = None

                # Try loading labels for display
                _label_file = os.path.join(_FASTAPI_BASE, "data", "labels_plant.csv")  # relative to fastapi/data/
                if os.path.exists(_label_file):
                    try:
                        import pandas as _pd_npy
                        _npy_labels = _pd_npy.read_csv(_label_file).squeeze().tolist()
                    except Exception:
                        _npy_labels = None

                if os.path.exists(_npy_path):
                    try:
                        _npy_loaded = np.load(_npy_path)
                        st.caption(f"Dataset loaded — {len(_npy_loaded)} images  ·  shape {_npy_loaded.shape}")
                    except Exception as _e:
                        st.error(f"Could not load .npy file: {_e}")

                if _npy_loaded is not None:
                    _max_idx = len(_npy_loaded) - 1
                    _col_idx, _col_rnd = st.columns([3, 1])
                    with _col_rnd:
                        if st.button("Random", use_container_width=True):
                            st.session_state["cnn_npy_idx"] = int(np.random.randint(0, _max_idx + 1))
                    with _col_idx:
                        _npy_idx = st.number_input(
                            f"Image index (0 – {_max_idx})",
                            min_value=0, max_value=_max_idx,
                            value=st.session_state.get("cnn_npy_idx", 0),
                            step=1, key="cnn_npy_idx"
                        )

                    # Show true label if available
                    if _npy_labels and _npy_idx < len(_npy_labels):
                        st.caption(f"True label: **{_npy_labels[_npy_idx]}**")

                    # Convert npy array -> PNG bytes for display and API
                    _arr = _npy_loaded[int(_npy_idx)]   # (128,128,3) float or uint8
                    if _arr.max() <= 1.0:
                        _arr_uint8 = (_arr * 255).astype(np.uint8)
                    else:
                        _arr_uint8 = _arr.astype(np.uint8)

                    from PIL import Image as _PILImg
                    import io as _io
                    _pil_img = _PILImg.fromarray(_arr_uint8, mode="RGB")
                    _buf = _io.BytesIO()
                    _pil_img.save(_buf, format="PNG")
                    _img_bytes   = _buf.getvalue()
                    _img_display = _buf

                else:
                    if _npy_path:
                        st.info("File not found — check the path above.")

            # --- Preview & classify button------------------------------------------------
            if _img_display is not None:
                st.image(_img_display, width="stretch")
            if _img_bytes is not None:
                if st.button("Run Classification", type="primary", use_container_width=True):
                    with st.spinner("Running CNN inference..."):
                        import io as _io2
                        result, err = api_post("/predict", files={"file": ("image.png", _io2.BytesIO(_img_bytes), "image/png")})
                    if err: st.error(err)
                    elif result: st.session_state["cnn_result"] = result

        with col2:
            st.markdown("#### Result")
            res = st.session_state.get("cnn_result")
            if res:
                if "error" in res:
                    st.error(res["error"])
                else:
                    conf       = res["confidence"]
                    class_name = res.get("class_name", f"Class {res['class_id']}")
                    class_id   = res["class_id"]
                    color      = "#10B981" if conf > 0.8 else "#F59E0B" if conf > 0.5 else "#EF4444"
                    st.markdown(f"""
                    <div class="metric-card" style="margin-bottom:12px;">
                        <div style="font-size:11px;color:#64748B;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;">Predicted Species</div>
                        <div style="font-size:28px;font-weight:700;color:#0F172A;">{class_name}</div>
                        <div style="font-size:12px;color:#94A3B8;font-family:'IBM Plex Mono',monospace;margin-top:4px;">class_id: {class_id}</div>
                    </div>
                    <div class="metric-card" style="margin-bottom:12px;">
                        <div style="font-size:11px;color:#64748B;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;">Confidence</div>
                        <div style="font-size:36px;font-weight:700;font-family:'IBM Plex Mono',monospace;color:#0F172A;">{conf:.1%}</div>
                        <div style="height:8px;background:#E2E8F0;border-radius:4px;overflow:hidden;margin-top:12px;">
                            <div style="height:100%;width:{conf*100:.1f}%;background:{color};border-radius:4px;"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                    # ---Per-class scores (same pattern as BERT classifier)-----
                    scores = res.get("scores", {})
                    if scores:
                        st.markdown("**All species scores**")
                        # Sort by score descending
                        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                        for _species, _score in sorted_scores:
                            _is_top = _species == class_name
                            _bar_color = color if _is_top else "#94A3B8"
                            st.markdown(
                                f'<div style="display:flex;align-items:center;gap:10px;margin:4px 0;">'
                                f'<div style="width:140px;font-size:12px;color:{"#0F172A" if _is_top else "#64748B"};'
                                f'font-weight:{"600" if _is_top else "400"};white-space:nowrap;overflow:hidden;'
                                f'text-overflow:ellipsis;">🌿 {_species}</div>'
                                f'<div style="flex:1;background:#F1F5F9;border-radius:3px;height:10px;overflow:hidden;">'
                                f'<div style="width:{_score*100:.1f}%;height:100%;background:{_bar_color};border-radius:3px;"></div></div>'
                                f'<div style="width:44px;font-size:12px;color:#64748B;font-family:\'IBM Plex Mono\',monospace;">{_score:.3f}</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
            else:
                st.markdown("""<div style="text-align:center;padding:48px 24px;color:#94A3B8;border:2px dashed #E2E8F0;border-radius:8px;">
                    <div style="font-size:32px;">📷</div>
                    <div style="font-size:13px;margin-top:8px;">Upload an image and click Run Classification</div>
                </div>""", unsafe_allow_html=True)

    # ====================================================================
    # TAB 2 — Model Insights
    # Only rendered when at least one versioned artefact directory exists.
    # Every section checks for its own file individually — missing files
    # show an info banner rather than raising an error.
    # ===================================================================
    if _has_insights:
        with tabs[1]:

            # -- Version picker-----------------------------------------
            def _version_label(v):
                status  = _registry.get(v, {}).get("status", "unknown")
                tag     = " * PRODUCTION" if status == "Production" else f"   {status}"
                return f"v{v}{tag}"

            _default_idx = _versions.index(_prod_version) if _prod_version in _versions else len(_versions) - 1
            _sel_label   = st.selectbox(
                "Model version",
                options=_versions,
                index=_default_idx,
                format_func=_version_label,
                key="cnn_version_picker",
            )
            _sel = _sel_label   # selected version string e.g. "2"

            import tempfile

            _run_id          = _registry.get(_sel, {}).get("run_id")

            with st.spinner("Loading model artifacts from MLflow..."):
                _HISTORY_PATH    = _download_cnn_artifact(_run_id, "artefacts/cnn_history.json")
                _MODEL_INFO_PATH = _download_cnn_artifact(_run_id, "model_card/cnn_model_info.json")
                _CLF_REPORT_PATH = _download_cnn_artifact(_run_id, "artefacts/cnn_classification_report.csv")
                _CONF_MAT_PATH   = _download_cnn_artifact(_run_id, "artefacts/cnn_confusion_matrix.csv")
                _PRED_PROBS_PATH = _download_cnn_artifact(_run_id, "artefacts/cnn_pred_probs.npy")
                _Y_TEST_PATH     = _download_cnn_artifact(_run_id, "artefacts/cnn_y_test.npy")
                _CLASSES_PATH    = _download_cnn_artifact(_run_id, "labels/cnn_label_classes.json")
                _CV_PATH         = None  # CV results not logged to MLflow yet

            st.divider()

            # -- Model info card ---------------------------------------------------------------------------------
            # Reading from mlflow
            minfo = {}
            if _MODEL_INFO_PATH and os.path.exists(_MODEL_INFO_PATH):
                try:
                    with open(_MODEL_INFO_PATH) as f:
                        minfo = json.load(f)
                except Exception as e:
                    st.warning(f"Could not load model info: {e}")

            if minfo:
                try:
                    _status = _registry.get(_sel, {}).get("status", "unknown")
                    if _status == "Production":
                        badge = '<span style="background:#D1FAE5;color:#065F46;padding:2px 10px;border-radius:4px;font-size:11px;font-weight:700;">★ PRODUCTION</span>'
                    else:
                        badge = '<span style="background:#FEF3C7;color:#92400E;padding:2px 10px;border-radius:4px;font-size:11px;font-weight:700;">CANDIDATE</span>'

                    def _fmt_metric(val, as_pct=True):
                        if val is None or val == 0 and not isinstance(val, float):
                            return "n/a"
                        return f"{val:.1%}" if as_pct else f"{val:.4f}"

                    st.markdown(f"""
                    <div class="metric-card" style="margin-bottom:16px;">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                            <span style="font-size:16px;font-weight:700;color:#0F172A;">Model v{_sel}</span>
                            {badge}
                        </div>
                        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;text-align:center;margin-bottom:10px;">
                            <div style="background:#F0FDF4;border-radius:6px;padding:10px 4px;">
                                <div style="font-size:10px;color:#64748B;text-transform:uppercase;letter-spacing:.06em;">Train Acc</div>
                                <div style="font-size:22px;font-weight:700;color:#0F172A;">{_fmt_metric(minfo.get('train_acc'))}</div>
                                <div style="font-size:11px;color:#94A3B8;">loss {_fmt_metric(minfo.get('train_loss'), as_pct=False)}</div>
                            </div>
                            <div style="background:#EFF6FF;border-radius:6px;padding:10px 4px;">
                                <div style="font-size:10px;color:#64748B;text-transform:uppercase;letter-spacing:.06em;">Val Acc</div>
                                <div style="font-size:22px;font-weight:700;color:#0F172A;">{_fmt_metric(minfo.get('val_acc'))}</div>
                                <div style="font-size:11px;color:#94A3B8;">loss {_fmt_metric(minfo.get('val_loss'), as_pct=False)}</div>
                            </div>
                            <div style="background:#FFF7ED;border-radius:6px;padding:10px 4px;">
                                <div style="font-size:10px;color:#64748B;text-transform:uppercase;letter-spacing:.06em;">Test Acc</div>
                                <div style="font-size:22px;font-weight:700;color:#0F172A;">{_fmt_metric(minfo.get('test_acc'))}</div>
                                <div style="font-size:11px;color:#94A3B8;">loss {_fmt_metric(minfo.get('test_loss'), as_pct=False)}</div>
                            </div>
                        </div>
                        <div style="font-size:11px;color:#94A3B8;text-align:center;">
                            Trained {minfo.get('trained_on', _registry.get(_sel,{}).get('trained_on','unknown'))} ·
                            {minfo.get('architecture','CNN')} ·
                            Epochs: {minfo.get('epochs_trained', '?')}
                        </div>
                    </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Could not render model info card: {e}")
            else:
                st.info(f"No model info found for v{_sel}.")

            # -- Cross-validation results ---------------------------------------------------------------------------------------------
            if _CV_PATH and os.path.exists(_CV_PATH):
                try:
                    with open(_CV_PATH) as _f:
                        _cv = json.load(_f)
                    st.markdown("#### Cross-Validation Results")
                    _ca, _cs = _cv["mean_accuracy"], _cv["std_accuracy"]
                    _fa, _fs = _cv["mean_f1"],       _cv["std_f1"]
                    st.markdown(
                        f'<div class="metric-card" style="margin-bottom:12px;">'
                        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;text-align:center;">'
                        f'<div><div style="font-size:10px;color:#64748B;text-transform:uppercase;">{_cv["n_folds"]}-Fold CV Accuracy</div>'
                        f'<div style="font-size:26px;font-weight:700;color:#0F172A;">{_ca:.1%}</div>'
                        f'<div style="font-size:12px;color:#94A3B8;">± {_cs:.1%}</div></div>'
                        f'<div><div style="font-size:10px;color:#64748B;text-transform:uppercase;">{_cv["n_folds"]}-Fold CV F1</div>'
                        f'<div style="font-size:26px;font-weight:700;color:#0F172A;">{_fa:.1%}</div>'
                        f'<div style="font-size:12px;color:#94A3B8;">± {_fs:.1%}</div></div>'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )
                    # Per-fold breakdown table
                    _fold_df = pd.DataFrame(_cv["folds"])[["fold","accuracy","f1","loss","epochs"]]
                    _fold_df.columns = ["Fold", "Accuracy", "F1", "Loss", "Epochs"]
                    st.dataframe(
                        _fold_df.style.format({"Accuracy": "{:.4f}", "F1": "{:.4f}", "Loss": "{:.4f}"}),
                        width="stretch", hide_index=True
                    )
                except Exception as _e:
                    st.warning(f"Could not load CV results: {_e}")
            else:
                st.caption("! Cross-validation not yet run for this version. Set `RUN_CV = True` in the notebook to generate results.")

            # ----- Training curves ------------------------------------------------------------------------------------------------
            if _HISTORY_PATH and os.path.exists(_HISTORY_PATH):
                try:
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    with open(_HISTORY_PATH) as f:
                        hist = json.load(f)

                    epochs = list(range(1, len(hist["loss"]) + 1))
                    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"))
                    fig.add_trace(go.Scatter(x=epochs, y=hist["loss"],         name="Train Loss", line=dict(color="#6366F1")),            row=1, col=1)
                    fig.add_trace(go.Scatter(x=epochs, y=hist["val_loss"],     name="Val Loss",   line=dict(color="#F59E0B", dash="dash")),row=1, col=1)
                    fig.add_trace(go.Scatter(x=epochs, y=hist["accuracy"],     name="Train Acc",  line=dict(color="#10B981")),            row=1, col=2)
                    fig.add_trace(go.Scatter(x=epochs, y=hist["val_accuracy"], name="Val Acc",    line=dict(color="#EF4444", dash="dash")),row=1, col=2)
                    # ── Test reference lines (single point — not a curve) ─────────
                    # Dotted orange horizontal line so it's visually distinct from
                    # the epoch curves. minfo is in scope from the model card above.
                    _t_acc  = minfo.get("test_acc")  if isinstance(minfo, dict) else None
                    _t_loss = minfo.get("test_loss") if isinstance(minfo, dict) else None
                    if isinstance(_t_acc, float):
                        fig.add_hline(y=_t_acc,  line_dash="dot", line_color="#F97316", line_width=2,
                                      annotation_text=f"Test acc {_t_acc:.3f}",
                                      annotation_position="bottom right", row=1, col=2)
                    if isinstance(_t_loss, float):
                        fig.add_hline(y=_t_loss, line_dash="dot", line_color="#F97316", line_width=2,
                                      annotation_text=f"Test loss {_t_loss:.3f}",
                                      annotation_position="top right", row=1, col=1)
                    fig.update_layout(height=320, margin=dict(t=40, b=20, l=20, r=20),
                                      legend=dict(orientation="h", y=-0.15),
                                      paper_bgcolor="white", plot_bgcolor="#F8FAFC")
                    fig.update_xaxes(title_text="Epoch")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not plot training curves: {e}")
            else:
                st.info(f"No training history found for v{_sel} — `cnn_history.json` missing.")

            # -- Classification report -----------------------------------------------------------------------------------------------
            if _CLF_REPORT_PATH and os.path.exists(_CLF_REPORT_PATH):
                try:
                    st.markdown("#### Per-class Metrics (Test Set)")
                    clf_df    = pd.read_csv(_CLF_REPORT_PATH, index_col=0)
                    skip_rows = ["accuracy", "macro avg", "weighted avg"]
                    class_rows = clf_df.loc[~clf_df.index.isin(skip_rows)].copy()
                    class_rows = class_rows[["precision","recall","f1-score","support"]].round(3)
                    st.dataframe(
                        class_rows.style.background_gradient(subset=["f1-score"], cmap="Greens"),
                        width="stretch"
                    )
                    summ = clf_df.loc[clf_df.index.isin(["macro avg","weighted avg"])]
                    st.dataframe(summ[["precision","recall","f1-score"]].round(3), width="stretch")
                except Exception as e:
                    st.warning(f"Could not load classification report: {e}")
            else:
                st.info(f"No classification report found for v{_sel}.")

            # -- Confusion matrix heatmap -----------------------------------------------------------------------------------------
            if _CONF_MAT_PATH and os.path.exists(_CONF_MAT_PATH):
                try:
                    import plotly.figure_factory as ff

                    st.markdown("#### Confusion Matrix (Test Set)")
                    cm_df  = pd.read_csv(_CONF_MAT_PATH, index_col=0)
                    labels = list(cm_df.columns)
                    z      = cm_df.values.tolist()
                    fig_cm = ff.create_annotated_heatmap(
                        z, x=labels, y=labels, colorscale="Blues", showscale=True
                    )
                    fig_cm.update_layout(
                        height=520, margin=dict(t=20, b=80, l=120, r=20),
                        xaxis=dict(title="Predicted", tickangle=45),
                        yaxis=dict(title="Actual", autorange="reversed"),
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not plot confusion matrix: {e}")
            else:
                st.info(f"No confusion matrix found for v{_sel}.")

            # -- Confidence distribution --------------------------------------------------------------------------------
            _probs_ok = (_PRED_PROBS_PATH is not None and os.path.exists(_PRED_PROBS_PATH) and _Y_TEST_PATH is not None and os.path.exists(_Y_TEST_PATH) and _CLASSES_PATH is not None and os.path.exists(_CLASSES_PATH))
            if _probs_ok:
                try:
                    import plotly.graph_objects as go

                    st.markdown("#### Classification Confidence Distribution (Test Set)")
                    pred_probs = np.load(_PRED_PROBS_PATH)
                    y_test_arr = np.load(_Y_TEST_PATH)
                    max_conf   = pred_probs.max(axis=1)
                    correct    = pred_probs.argmax(axis=1) == y_test_arr

                    fig_dens = go.Figure()
                    fig_dens.add_trace(go.Histogram(
                        x=max_conf[correct],  name="Correct",
                        opacity=0.65, marker_color="#10B981",
                        xbins=dict(start=0, end=1, size=0.02)
                    ))
                    fig_dens.add_trace(go.Histogram(
                        x=max_conf[~correct], name="Incorrect",
                        opacity=0.65, marker_color="#EF4444",
                        xbins=dict(start=0, end=1, size=0.02)
                    ))
                    fig_dens.update_layout(
                        barmode="overlay", height=300,
                        xaxis_title="Predicted confidence", yaxis_title="Count",
                        margin=dict(t=20, b=40, l=40, r=20),
                        legend=dict(orientation="h", y=1.1),
                        paper_bgcolor="white", plot_bgcolor="#F8FAFC",
                    )
                    st.plotly_chart(fig_dens, use_container_width=True)
                    st.caption(f"Correct: {correct.sum()} · Incorrect: {(~correct).sum()} · Total: {len(correct)}")
                except Exception as e:
                    st.warning(f"Could not plot confidence distribution: {e}")
            else:
                st.info(f"No prediction data found for v{_sel} — `cnn_pred_probs.npy` or `cnn_y_test.npy` missing.")

            # ----- ROC-AUC & Precision-Recall ----------------------------------------------------------------------------------------
            if _PRED_PROBS_PATH and _Y_TEST_PATH and _CLASSES_PATH and os.path.exists(_PRED_PROBS_PATH) and os.path.exists(_Y_TEST_PATH) and os.path.exists(_CLASSES_PATH):
                try:
                    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
                    from sklearn.preprocessing import label_binarize
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    _roc_probs   = np.load(_PRED_PROBS_PATH)
                    _roc_ytrue   = np.load(_Y_TEST_PATH)
                    with open(_CLASSES_PATH) as _f:
                        _roc_classes = json.load(_f)
                    _n_classes   = len(_roc_classes)
                    _y_bin       = label_binarize(_roc_ytrue, classes=list(range(_n_classes)))

                    # colour palette — one per class, cycling if needed
                    _PALETTE = ["#6366F1","#10B981","#F59E0B","#EF4444","#8B5CF6",
                                "#06B6D4","#F97316","#84CC16","#EC4899","#14B8A6",
                                "#A855F7","#FB923C"]

                    st.markdown("#### ROC Curves (One-vs-Rest, Test Set)")
                    _fig_roc = go.Figure()
                    _roc_aucs = {}
                    for _i, _cls in enumerate(_roc_classes):
                        _fpr, _tpr, _ = roc_curve(_y_bin[:, _i], _roc_probs[:, _i])
                        _roc_auc      = auc(_fpr, _tpr)
                        _roc_aucs[_cls] = _roc_auc
                        _col          = _PALETTE[_i % len(_PALETTE)]
                        _fig_roc.add_trace(go.Scatter(
                            x=_fpr, y=_tpr, mode="lines", name=f"{_cls}  (AUC={_roc_auc:.3f})",
                            line=dict(color=_col, width=2)
                        ))
                    # diagonal chance line
                    _fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                        line=dict(color="#CBD5E1", dash="dash", width=1), showlegend=True
                    ))
                    _fig_roc.update_layout(
                        height=420,
                        xaxis=dict(title="False Positive Rate", range=[0, 1]),
                        yaxis=dict(title="True Positive Rate",  range=[0, 1.02]),
                        legend=dict(x=1.01, y=1, xanchor="left", font=dict(size=11)),
                        margin=dict(t=20, b=50, l=50, r=180),
                        paper_bgcolor="white", plot_bgcolor="#F8FAFC",
                    )
                    st.plotly_chart(_fig_roc, use_container_width=True)
                    # macro-average AUC summary
                    _macro_auc = sum(_roc_aucs.values()) / len(_roc_aucs)
                    st.caption(f"Macro-average AUC: **{_macro_auc:.3f}**  ·  "
                               + "  ·  ".join(f"{c}: {v:.3f}" for c, v in _roc_aucs.items()))

                    st.markdown("#### Precision-Recall Curves (One-vs-Rest, Test Set)")
                    _fig_pr = go.Figure()
                    _ap_scores = {}
                    for _i, _cls in enumerate(_roc_classes):
                        _prec, _rec, _ = precision_recall_curve(_y_bin[:, _i], _roc_probs[:, _i])
                        _ap            = average_precision_score(_y_bin[:, _i], _roc_probs[:, _i])
                        _ap_scores[_cls] = _ap
                        _col           = _PALETTE[_i % len(_PALETTE)]
                        _fig_pr.add_trace(go.Scatter(
                            x=_rec, y=_prec, mode="lines", name=f"{_cls}  (AP={_ap:.3f})",
                            line=dict(color=_col, width=2)
                        ))
                    _fig_pr.update_layout(
                        height=420,
                        xaxis=dict(title="Recall",    range=[0, 1]),
                        yaxis=dict(title="Precision", range=[0, 1.02]),
                        legend=dict(x=1.01, y=1, xanchor="left", font=dict(size=11)),
                        margin=dict(t=20, b=50, l=50, r=180),
                        paper_bgcolor="white", plot_bgcolor="#F8FAFC",
                    )
                    st.plotly_chart(_fig_pr, use_container_width=True)
                    _mean_ap = sum(_ap_scores.values()) / len(_ap_scores)
                    st.caption(f"Mean Average Precision (mAP): **{_mean_ap:.3f}**  ·  "
                               + "  ·  ".join(f"{c}: {v:.3f}" for c, v in _ap_scores.items()))

                except Exception as e:
                    st.warning(f"Could not plot ROC / PR curves: {e}")
            else:
                st.info(f"ROC and PR curves require `cnn_pred_probs.npy` and `cnn_y_test.npy` — run the notebook to generate them.")

            # ---TensorBoard embed-------------------------
            st.markdown("#### Architecture & Training Logs")

            TB_HOST = "http://localhost:6006" # iframe
            TB_CHECK_HOST = os.getenv("TB_HOST", "http://tensorboard:6006")  # container check


            # Check whether TensorBoard is actually reachable before embedding
            _tb_alive = False
            try:
                _tb_resp = requests.get(TB_CHECK_HOST, timeout=1.5)
                _tb_alive = _tb_resp.status_code == 200
            except Exception:
                _tb_alive = False

            #_tb_logdir = _abs(_registry.get(_sel, {}).get("tensorboard_logdir", ""))

            # Get logdir from model info card (downloaded from MLflow)
            _tb_logdir = minfo.get("tensorboard_logdir", "") if minfo else ""

            if _tb_alive:
                # Open directly on the Graphs tab for the architecture view.
                # The smoothing=0 param keeps scalar curves unsmoothed by default.
                _tb_url = f"{TB_HOST}/#graphs"
                st.success(
                    f"TensorBoard is running — embedded below. "
                    f"Switch tabs inside the frame to view Scalars, Histograms, etc."
                )
                if _tb_logdir:
                    st.caption(f"Log dir for v{_sel}: `{_tb_logdir}`")
                import streamlit.components.v1 as _components
                _components.iframe(_tb_url, height=800, scrolling=True)
            else:
                st.warning(
                    "TensorBoard is not running — start it to see the CNN architecture graph "
                    "and training scalars."
                )
                if _tb_logdir:
                    st.caption(f"Log dir for v{_sel}: `{_tb_logdir}`")
                st.code(
                    "# Run from the fastapi/ folder\n"
                    ".venv\\Scripts\\python.exe -m tensorboard.main --logdir logs\n"
                    "# logs/ contains both logs/fit/ (CNN) and logs/bert/ (BERT)\n"
                    "# Then refresh this page — the graph will embed automatically.",
                    language="bash"
                )
                st.caption(f"Expected at: {TB_HOST}")


# ===============================================================================
# PAGE: Ticket Classifier
# ===============================================================================
elif st.session_state.page == "nlp_classify":
    import os, json
    import numpy as np

    st.markdown("""<div class="page-header">
        <h1>Ticket Classifier</h1>
        <p>BERT fine-tuned on 400 labeled tickets · MLflow tracked · <span class="mono">POST /nlp/classify</span></p>
    </div>""", unsafe_allow_html=True)

    # -- Locate BERT artefacts saved by train.py ----------------------------------
    # Inside Docker: FASTAPI_BASE=/app/fastapi (set in docker-compose.yml)
    # Local dev: falls back to relative path from streamlitapi/
    _FASTAPI_DIR   = os.getenv("FASTAPI_BASE",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fastapi"))
    )

    import mlflow
    from mlflow.tracking import MlflowClient
    import tempfile

    MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(MLFLOW_URI)

    # -- Don't download artifacts until user opens Model Insights tab ------
    _has_insights = True  # always show the tab
    _tab_labels   = ["Classify", "Batch CSV", "Model Insights"]
    _tabs         = st.tabs(_tab_labels)

    # ======================================================================
    # TAB 1 — Single Ticket
    # ======================================================================
    with _tabs[0]:
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            examples = [
                "My order ORD-12345678 hasn't arrived — it's been 10 days.",
                "I returned the laptop stand on 2026-01-15. Where is my refund?",
                "I can't log in. My password reset email never arrived.",
                "The coffee grinder stopped working after 3 days.",
                "Do you offer a student discount?",
            ]
            example   = st.selectbox("Load example", [""] + examples)
            ticket_id = st.text_input("Ticket ID (optional)", placeholder="TKT-0001")
            text      = st.text_area("Ticket text", value=example, height=130,
                                     placeholder="Paste support ticket text here...")
            if st.button("Classify", type="primary", disabled=not text.strip(), use_container_width=True):
                with st.spinner("Classifying..."):
                    result, err = api_post("/nlp/classify", json={"ticket_id": ticket_id or None, "text": text})
                if err: st.error(err)
                elif result: st.session_state["clf_result"] = result

        with col2:
            st.markdown("#### Prediction")
            res = st.session_state.get("clf_result")
            if res:
                render_category_badge(res["predicted_category"], res["confidence"])
                st.markdown("")
                st.markdown("**Category scores**")
                for c in CATEGORIES:
                    score  = res.get("scores", {}).get(c, 0)
                    color  = CATEGORY_COLORS[c]
                    is_top = c == res["predicted_category"]
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:10px;margin:4px 0;">'
                        f'<div style="width:120px;font-size:12px;color:{"#0F172A" if is_top else "#64748B"};font-weight:{"600" if is_top else "400"};">{CATEGORY_ICONS[c]} {c}</div>'
                        f'<div style="flex:1;background:#F1F5F9;border-radius:3px;height:10px;overflow:hidden;">'
                        f'<div style="width:{score*100:.1f}%;height:100%;background:{color};border-radius:3px;"></div></div>'
                        f'<div style="width:40px;font-size:12px;color:#64748B;font-family:\'IBM Plex Mono\',monospace;">{score:.2f}</div></div>',
                        unsafe_allow_html=True)
                st.caption(f"Method: `{res.get('method','')}` · Latency: {res.get('latency_ms','')} ms")
            else:
                st.markdown("""<div style="text-align:center;padding:48px 24px;color:#94A3B8;border:2px dashed #E2E8F0;border-radius:8px;">
                    <div style="font-size:28px;">Ticket</div>
                    <div style="font-size:13px;margin-top:8px;">Enter a ticket and click Classify</div>
                </div>""", unsafe_allow_html=True)

    # ======================================================================
    # TAB 2 — Batch CSV
    # ======================================================================
    with _tabs[1]:
        st.markdown("Upload a CSV with a `text` column (and optionally `ticket_id`).")
        csv_file = st.file_uploader("Choose CSV", type="csv", key="batch_csv")
        if csv_file:
            df_batch = pd.read_csv(csv_file)
            if "text" not in df_batch.columns:
                st.error("CSV must contain a `text` column.")
            else:
                st.dataframe(df_batch.head(5), width="stretch")
                if st.button("Run Batch Classification", type="primary"):
                    tickets = [{"ticket_id": str(row.get("ticket_id", "")), "text": row["text"]}
                               for _, row in df_batch.iterrows()]
                    with st.spinner(f"Classifying {len(tickets)} tickets..."):
                        result, err = api_post("/nlp/classify/batch", json={"tickets": tickets[:100]})
                    if err: st.error(err)
                    elif result:
                        res_df = pd.DataFrame([{
                            "ticket_id": r["ticket_id"],
                            "category":  r["predicted_category"],
                            "confidence": r["confidence"],
                        } for r in result["results"]])
                        st.dataframe(res_df, width="stretch")
                        st.bar_chart(res_df["category"].value_counts())
                        st.download_button("Download labeled CSV",
                                           res_df.to_csv(index=False).encode(),
                                           "labeled_tickets.csv", "text/csv")

    # =====================================================================
    # TAB 3 — Model Insights
    # Only rendered when bert_model_card.json exists (i.e. after training).
    # Every section independently checks for its own file.
    # =====================================================================
    if _has_insights:
        with _tabs[2]:

            #-----Download artifacts from MLflow when tab is opened
            with st.spinner("Loading model insights from MLflow..."):
                _artifacts    = _get_all_bert_artifacts()
                _CARD_PATH    = _artifacts.get("bert_model_card.json")
                _HISTORY_PATH = _artifacts.get("bert_history.json")
                _CLF_PATH     = _artifacts.get("classification_report.csv")
                _CM_PATH      = _artifacts.get("confusion_matrix.csv")
                _PROBS_PATH   = _artifacts.get("test_pred_probs.npy")
                _YTRUE_PATH   = _artifacts.get("test_y_true.npy")

            # ---- Model card ------------------------------------------------------------------------
            try:
                if _CARD_PATH and os.path.exists(_CARD_PATH):
                    with open(_CARD_PATH) as _f:
                        _card = json.load(_f)
                else:
                    _card = {}
                    st.warning("Model card not found in MLflow — run train.py to register the model.")

                _hp   = _card.get("hyperparameters", {})
                _met  = _card.get("metrics", {})
                _spl  = _card.get("data_split", {})
                _cats = _card.get("categories", CATEGORIES)

                st.markdown(f"""
                <div class="metric-card" style="margin-bottom:16px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                        <span style="font-size:16px;font-weight:700;color:#0F172A;">{_card.get('model','bert-base-uncased')}</span>
                        <span style="background:#EDE9FE;color:#5B21B6;padding:2px 10px;border-radius:4px;font-size:11px;font-weight:700;">BERT</span>
                    </div>
                    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;text-align:center;margin-bottom:12px;">
                        <div><div style="font-size:11px;color:#64748B;text-transform:uppercase;">Best Val F1</div>
                             <div style="font-size:22px;font-weight:700;color:#0F172A;">{_met.get('best_val_f1',0):.1%}</div></div>
                        <div><div style="font-size:11px;color:#64748B;text-transform:uppercase;">Test Acc</div>
                             <div style="font-size:22px;font-weight:700;color:#0F172A;">{_met.get('test_accuracy',0):.1%}</div></div>
                        <div><div style="font-size:11px;color:#64748B;text-transform:uppercase;">Test F1</div>
                             <div style="font-size:22px;font-weight:700;color:#0F172A;">{_met.get('test_f1',0):.1%}</div></div>
                        <div><div style="font-size:11px;color:#64748B;text-transform:uppercase;">Epochs</div>
                             <div style="font-size:22px;font-weight:700;color:#0F172A;">{_hp.get('epochs','?')}</div></div>
                    </div>
                    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;text-align:center;padding-top:10px;border-top:1px solid #E2E8F0;">
                        <div><div style="font-size:10px;color:#94A3B8;">Batch size</div><div style="font-size:13px;font-weight:600;">{_hp.get('batch_size','?')}</div></div>
                        <div><div style="font-size:10px;color:#94A3B8;">Max length</div><div style="font-size:13px;font-weight:600;">{_hp.get('max_len','?')}</div></div>
                        <div><div style="font-size:10px;color:#94A3B8;">LR</div><div style="font-size:13px;font-weight:600;">{_hp.get('lr','?')}</div></div>
                        <div><div style="font-size:10px;color:#94A3B8;">Warmup</div><div style="font-size:13px;font-weight:600;">{_hp.get('warmup_ratio','?')}</div></div>
                    </div>
                    <div style="margin-top:10px;font-size:11px;color:#94A3B8;">
                        Trained {_card.get('trained_on','unknown')} ·
                        Split: {_spl.get('train',0)} train / {_spl.get('val',0)} val / {_spl.get('test',0)} test ·
                        MLflow run: <span style="font-family:monospace;">{_card.get('mlflow_run_id','')[:8]}…</span>
                    </div>
                </div>""", unsafe_allow_html=True)
            except Exception as _e:
                st.warning(f"Could not load model card: {_e}")
                _card  = {}
                _cats  = CATEGORIES

            st.divider()

            # --- Training curves ---------------------------------------------------
            if _HISTORY_PATH and os.path.exists(_HISTORY_PATH):
                try:
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    with open(_HISTORY_PATH) as _f:
                        _hist = json.load(_f)

                    _epochs = list(range(1, len(_hist["train_loss"]) + 1))
                    _fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=("Loss (train vs val)", "Val Accuracy & F1")
                    )
                    _fig.add_trace(go.Scatter(x=_epochs, y=_hist["train_loss"],   name="Train Loss",   line=dict(color="#6366F1")),            row=1, col=1)
                    _fig.add_trace(go.Scatter(x=_epochs, y=_hist["val_loss"],     name="Val Loss",     line=dict(color="#F59E0B", dash="dash")),row=1, col=1)
                    _fig.add_trace(go.Scatter(x=_epochs, y=_hist["val_accuracy"], name="Val Accuracy", line=dict(color="#10B981")),            row=1, col=2)
                    _fig.add_trace(go.Scatter(x=_epochs, y=_hist["val_f1"],       name="Val F1",       line=dict(color="#8B5CF6", dash="dash")),row=1, col=2)
                    _fig.update_layout(height=320, margin=dict(t=40, b=20, l=20, r=20),
                                       legend=dict(orientation="h", y=-0.18),
                                       paper_bgcolor="white", plot_bgcolor="#F8FAFC")
                    _fig.update_xaxes(title_text="Epoch", dtick=1)
                    st.plotly_chart(_fig, use_container_width=True)
                except Exception as _e:
                    st.warning(f"Could not plot training curves: {_e}")
            else:
                st.info("No training history found — `bert_history.json` missing from `models/`.")

            # ---- Per-class F1 bar chart --------------------------------------------------------
            _per_class = _card.get("metrics", {}).get("per_class_f1", {})
            if _per_class:
                try:
                    import plotly.graph_objects as go
                    st.markdown("#### Per-class F1 (Test Set)")
                    _labels_f1 = list(_per_class.keys())
                    _values_f1 = list(_per_class.values())
                    _colors_f1 = [CATEGORY_COLORS.get(c, "#6B7280") for c in _labels_f1]
                    _fig_f1 = go.Figure(go.Bar(
                        x=_labels_f1, y=_values_f1,
                        marker_color=_colors_f1, text=[f"{v:.3f}" for v in _values_f1],
                        textposition="outside",
                    ))
                    _fig_f1.update_layout(
                        height=280, yaxis=dict(range=[0, 1.1], title="F1 Score"),
                        margin=dict(t=20, b=40, l=40, r=20),
                        paper_bgcolor="white", plot_bgcolor="#F8FAFC",
                    )
                    st.plotly_chart(_fig_f1, use_container_width=True)
                except Exception as _e:
                    st.warning(f"Could not plot per-class F1: {_e}")

            # --- Classification report table --------------------------------------------
            if _CLF_PATH and os.path.exists(_CLF_PATH):
                try:
                    st.markdown("#### Classification Report (Test Set)")
                    _clf_df    = pd.read_csv(_CLF_PATH, index_col=0)
                    _skip_rows = ["accuracy", "macro avg", "weighted avg"]
                    _class_rows = _clf_df.loc[~_clf_df.index.isin(_skip_rows)].copy()
                    _class_rows = _class_rows[["precision","recall","f1-score","support"]].round(3)
                    st.dataframe(
                        _class_rows.style.background_gradient(subset=["f1-score"], cmap="Purples"),
                        width="stretch"
                    )
                    _summ = _clf_df.loc[_clf_df.index.isin(["macro avg","weighted avg"])]
                    st.dataframe(_summ[["precision","recall","f1-score"]].round(3), width="stretch")
                except Exception as _e:
                    st.warning(f"Could not load classification report: {_e}")
            else:
                st.info("No classification report found — run `scripts/train.py` to generate it.")

            # --- Confusion matrix ------------------------------------------------------------------
            if _CM_PATH and os.path.exists(_CM_PATH):
                try:
                    import plotly.figure_factory as ff
                    st.markdown("#### Confusion Matrix (Test Set)")
                    _cm_df  = pd.read_csv(_CM_PATH, index_col=0)
                    _labels = list(_cm_df.columns)
                    _z      = _cm_df.values.tolist()
                    _fig_cm = ff.create_annotated_heatmap(
                        _z, x=_labels, y=_labels,
                        colorscale="Purples", showscale=True
                    )
                    _fig_cm.update_layout(
                        height=400, margin=dict(t=20, b=80, l=140, r=20),
                        xaxis=dict(title="Predicted", tickangle=30),
                        yaxis=dict(title="Actual", autorange="reversed"),
                    )
                    st.plotly_chart(_fig_cm, use_container_width=True)
                except Exception as _e:
                    st.warning(f"Could not plot confusion matrix: {_e}")
            else:
                st.info("No confusion matrix found — run `scripts/train.py` to generate it.")

            # --- Confidence distribution -------------------------------------------
            if _PROBS_PATH and _YTRUE_PATH and os.path.exists(_PROBS_PATH) and os.path.exists(_YTRUE_PATH):
                try:
                    import plotly.graph_objects as go
                    st.markdown("#### Classification Confidence Distribution (Test Set)")
                    _probs   = np.load(_PROBS_PATH)
                    _ytrue   = np.load(_YTRUE_PATH)
                    _maxconf = _probs.max(axis=1)
                    _correct = _probs.argmax(axis=1) == _ytrue

                    _fig_dens = go.Figure()
                    _fig_dens.add_trace(go.Histogram(
                        x=_maxconf[_correct],  name="Correct",
                        opacity=0.65, marker_color="#8B5CF6",
                        xbins=dict(start=0, end=1, size=0.05)
                    ))
                    _fig_dens.add_trace(go.Histogram(
                        x=_maxconf[~_correct], name="Incorrect",
                        opacity=0.65, marker_color="#EF4444",
                        xbins=dict(start=0, end=1, size=0.05)
                    ))
                    _fig_dens.update_layout(
                        barmode="overlay", height=280,
                        xaxis_title="Predicted confidence", yaxis_title="Count",
                        margin=dict(t=20, b=40, l=40, r=20),
                        legend=dict(orientation="h", y=1.12),
                        paper_bgcolor="white", plot_bgcolor="#F8FAFC",
                    )
                    st.plotly_chart(_fig_dens, use_container_width=True)
                    st.caption(f"Correct: {_correct.sum()} · Incorrect: {(~_correct).sum()} · Total: {len(_correct)}")
                except Exception as _e:
                    st.warning(f"Could not plot confidence distribution: {_e}")
            else:
                st.info("No prediction data found — `test_pred_probs.npy` missing from `models/`.")

            # ---- ROC-AUC & Precision-Recall -----------------------------------------------------------------------
            if _PROBS_PATH and _YTRUE_PATH and os.path.exists(_PROBS_PATH) and os.path.exists(_YTRUE_PATH):
                try:
                    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
                    from sklearn.preprocessing import label_binarize
                    import plotly.graph_objects as _go2
                    from plotly.subplots import make_subplots as _msp2

                    _b_probs    = np.load(_PROBS_PATH)
                    _b_ytrue    = np.load(_YTRUE_PATH)
                    _b_classes  = _cats   # from model card loaded above
                    _b_n        = len(_b_classes)
                    _b_ybin     = label_binarize(_b_ytrue, classes=list(range(_b_n)))

                    _B_PALETTE  = ["#8B5CF6","#06B6D4","#10B981","#F59E0B","#EF4444",
                                   "#6366F1","#F97316","#84CC16","#EC4899","#14B8A6"]

                    st.markdown("#### ROC Curves (One-vs-Rest, Test Set)")
                    _fig_b_roc  = _go2.Figure()
                    _b_aucs     = {}
                    for _bi, _bcls in enumerate(_b_classes):
                        _bfpr, _btpr, _ = roc_curve(_b_ybin[:, _bi], _b_probs[:, _bi])
                        _bauc           = auc(_bfpr, _btpr)
                        _b_aucs[_bcls]  = _bauc
                        _bcol           = _B_PALETTE[_bi % len(_B_PALETTE)]
                        _fig_b_roc.add_trace(_go2.Scatter(
                            x=_bfpr, y=_btpr, mode="lines",
                            name=f"{_bcls}  (AUC={_bauc:.3f})",
                            line=dict(color=_bcol, width=2)
                        ))
                    _fig_b_roc.add_trace(_go2.Scatter(
                        x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                        line=dict(color="#CBD5E1", dash="dash", width=1), showlegend=True
                    ))
                    _fig_b_roc.update_layout(
                        height=420,
                        xaxis=dict(title="False Positive Rate", range=[0, 1]),
                        yaxis=dict(title="True Positive Rate",  range=[0, 1.02]),
                        legend=dict(x=1.01, y=1, xanchor="left", font=dict(size=11)),
                        margin=dict(t=20, b=50, l=50, r=180),
                        paper_bgcolor="white", plot_bgcolor="#F8FAFC",
                    )
                    st.plotly_chart(_fig_b_roc, use_container_width=True)
                    _b_macro_auc = sum(_b_aucs.values()) / len(_b_aucs)
                    st.caption(f"Macro-average AUC: **{_b_macro_auc:.3f}**  ·  "
                               + "  ·  ".join(f"{c}: {v:.3f}" for c, v in _b_aucs.items()))

                    st.markdown("#### Precision-Recall Curves (One-vs-Rest, Test Set)")
                    _fig_b_pr  = _go2.Figure()
                    _b_aps     = {}
                    for _bi, _bcls in enumerate(_b_classes):
                        _bprec, _brec, _ = precision_recall_curve(_b_ybin[:, _bi], _b_probs[:, _bi])
                        _bap             = average_precision_score(_b_ybin[:, _bi], _b_probs[:, _bi])
                        _b_aps[_bcls]    = _bap
                        _bcol            = _B_PALETTE[_bi % len(_B_PALETTE)]
                        _fig_b_pr.add_trace(_go2.Scatter(
                            x=_brec, y=_bprec, mode="lines",
                            name=f"{_bcls}  (AP={_bap:.3f})",
                            line=dict(color=_bcol, width=2)
                        ))
                    _fig_b_pr.update_layout(
                        height=420,
                        xaxis=dict(title="Recall",    range=[0, 1]),
                        yaxis=dict(title="Precision", range=[0, 1.02]),
                        legend=dict(x=1.01, y=1, xanchor="left", font=dict(size=11)),
                        margin=dict(t=20, b=50, l=50, r=180),
                        paper_bgcolor="white", plot_bgcolor="#F8FAFC",
                    )
                    st.plotly_chart(_fig_b_pr, use_container_width=True)
                    _b_map = sum(_b_aps.values()) / len(_b_aps)
                    st.caption(f"Mean Average Precision (mAP): **{_b_map:.3f}**  ·  "
                               + "  ·  ".join(f"{c}: {v:.3f}" for c, v in _b_aps.items()))

                except Exception as _e:
                    st.warning(f"Could not plot ROC / PR curves: {_e}")
            else:
                st.info("ROC and PR curves require `test_pred_probs.npy` and `test_y_true.npy` — run `scripts/train.py` to generate them.")

            # ---- TensorBoard embed ----------------------------------------------------------------------------
            st.markdown("#### Training Logs & Model Graph")
            st.caption(
                "TensorBoard shows: **Scalars** (loss/acc/F1 curves) · "
                "**Graphs** (BERT computation graph) · **HParams** (hyperparameter comparison across runs)"
            )

            TB_HOST_BERT = "http://localhost:6006" # iframe
            TB_CHECK_HOST_BERT = os.getenv("TB_HOST", "http://tensorboard:6006")  # container check


            _tb_alive = False
            try:
                _tb_resp  = requests.get(TB_CHECK_HOST_BERT, timeout=1.5)
                _tb_alive = _tb_resp.status_code == 200
            except Exception:
                _tb_alive = False

            # Resolve logdir from model card — may be absolute or relative.
            # Always display as relative to fastapi/ for readability.
            _tb_logdir_raw = _card.get("tensorboard_logdir", "")
            if _tb_logdir_raw:
                _tb_logdir_abs = _tb_logdir_raw if os.path.isabs(_tb_logdir_raw) \
                                 else os.path.join(_FASTAPI_DIR, _tb_logdir_raw)
                # Show relative path in caption — cleaner than full absolute path
                try:
                    _tb_logdir_display = os.path.relpath(_tb_logdir_abs, _FASTAPI_DIR).replace("\\", "/")
                except ValueError:
                    _tb_logdir_display = _tb_logdir_raw
            else:
                _tb_logdir_display = ""

            if _tb_alive:
                st.success(
                    "TensorBoard is running — embedded below. "
                    "Use the **Scalars** tab for loss/F1 curves and **HParams** to compare runs."
                )
                if _tb_logdir_display:
                    st.caption(f"Log dir: `{_tb_logdir_display}`")
                import streamlit.components.v1 as _components
                _components.iframe(f"{TB_HOST_BERT}/#scalars", height=800, scrolling=True)
            else:
                st.warning(
                    "TensorBoard is not running — start it to see training curves and the BERT graph."
                )
                if _tb_logdir_display:
                    st.caption(f"Log dir: `{_tb_logdir_display}`")
                st.code(
                    "# Run from the fastapi/ folder — logs/bert/ contains BERT runs,\n"
                    "# logs/fit/ contains CNN runs. Using logs/ shows both together.\n"
                    ".venv\\Scripts\\python.exe -m tensorboard.main --logdir logs\n"
                    "# Then refresh this page — the frame will appear automatically.",
                    language="bash"
                )
                st.caption(f"Expected at: {TB_HOST_BERT}")


# =================================================================================
# PAGE: NER
# =================================================================================
elif st.session_state.page == "nlp_ner":
    st.markdown("""<div class="page-header">
        <h1>Named Entity Recognition</h1>
        <p>Regex NER · ORDER_ID · DATE · EMAIL · <span class="mono">POST /nlp/ner</span></p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Entity Extraction", "Model Evaluation"])

    with tab1:
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            NER_EXAMPLES = {
                "TKT-0007": "I returned laptop stand from order ORD-76842684 on 2025-11-27. When will I get my refund?",
                "TKT-0015": "My order ORD-29729975 was supposed to arrive on 2026-02-06 but it hasn't shown up.",
                "TKT-0010": "Please help me reset my password. I no longer have access to my email sam128@company.co.uk.",
                "TKT-0044": "I was charged twice for order ORD-90084745. You can reach me at jamie916@company.co.uk.",
            }

            # Initialise session state keys if not present
            if "ner_tid" not in st.session_state:
                st.session_state["ner_tid"] = ""
            if "ner_text" not in st.session_state:
                st.session_state["ner_text"] = ""
            if "ner_last_ex" not in st.session_state:
                st.session_state["ner_last_ex"] = ""

            example_choice = st.selectbox("Load example", [""] + list(NER_EXAMPLES.keys()), key="ner_ex")

            # Only update fields when a NEW example is selected
            if example_choice and example_choice != st.session_state["ner_last_ex"]:
                st.session_state["ner_tid"]     = example_choice
                st.session_state["ner_text"]    = NER_EXAMPLES[example_choice]
                st.session_state["ner_last_ex"] = example_choice

            # Widgets read directly from session state keys
            st.text_input("Ticket ID (optional)", key="ner_tid")
            st.text_area("Ticket text", height=130,
                         placeholder="Paste ticket text here...", key="ner_text")

            ticket_id = st.session_state["ner_tid"]
            text      = st.session_state["ner_text"]

            if st.button("Extract Entities", type="primary",
                         disabled=not text.strip(), use_container_width=True):
                with st.spinner("Extracting..."):
                    result, err = api_post("/nlp/ner", json={
                        "ticket_id": ticket_id.strip() or None,
                        "text": text.strip()
                    })
                if err:
                    st.error(err)
                elif result:
                    st.session_state["ner_result"] = result

        with col2:
            st.markdown("#### Detected Entities")
            res = st.session_state.get("ner_result")
            if res:
                # Show ticket ID if present
                if res.get("ticket_id"):
                    st.caption(f"Ticket: `{res['ticket_id']}`")

                entities = res.get("entities", [])

                # Show annotated ticket text
                st.markdown("**Ticket text**")
                st.markdown(annotated_html(res["text"], entities), unsafe_allow_html=True)
                st.markdown("")

                # Entity colour legend
                legend = " ".join(
                    f'<span class="ent-chip" style="background:{c}18;color:{c};border:1px solid {c}40;">{l}</span>'
                    for l, c in ENTITY_COLORS.items()
                )
                st.markdown(legend, unsafe_allow_html=True)
                st.markdown("")

                if entities:
                    st.markdown("**Entity details**")
                    render_entities(entities)
                    st.caption(f"Total: {res['entity_count']} entities detected")
                else:
                    st.info("No ORDER_ID, DATE, or EMAIL entities found in this ticket.")
            else:
                st.markdown("""<div style="text-align:center;padding:48px 24px;color:#94A3B8;border:2px dashed #E2E8F0;border-radius:8px;">
                    <div style="font-size:28px;">Entity</div>
                    <div style="font-size:13px;margin-top:8px;">Enter ticket text and click Extract Entities</div>
                </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        Evaluates regex NER against the **50-ticket gold-standard** (`data/ner_annotations.jsonl`).
        Metric: exact-match (label + character span) precision, recall, F1.

        **Why regex?** ORDER_ID, DATE, and EMAIL follow rigid surface patterns — regex achieves
        near-perfect scores with zero training data or model weight overhead.
        """)
        if st.button("Run Evaluation", type="primary"):
            with st.spinner("Computing metrics..."):
                result, err = api_get("/nlp/ner/evaluate")
            if err: st.error(err)
            elif result:
                rows = []
                for label in ["ORDER_ID", "DATE", "EMAIL"]:
                    m = result["evaluation"][label]
                    rows.append({"Entity": label, "Precision": m["precision"], "Recall": m["recall"],
                                 "F1": m["f1"], "TP": m["tp"], "FP": m["fp"], "FN": m["fn"]})
                m = result["evaluation"]["micro_avg"]
                rows.append({"Entity": "micro_avg", "Precision": m["precision"], "Recall": m["recall"],
                             "F1": m["f1"], "TP": None, "FP": None, "FN": None})
                st.dataframe(
                    pd.DataFrame(rows).style
                      .format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"})
                      .highlight_max(subset=["F1"], color="#D1FAE5"),
                    width="stretch"
                )


# ==============================================================================
# PAGE: LLM Draft
# ==============================================================================
elif st.session_state.page == "nlp_draft":
    st.markdown("""<div class="page-header">
        <h1>Draft Response Generator</h1>
        <p>OpenAI gpt-4o-mini · prompt-engineered · template fallback · <span class="mono">POST /nlp/draft</span></p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        DRAFT_EXAMPLES = [
            ("TKT-0007", "Refund",        "Hi team, I returned laptop stand from order ORD-76842684 on 2025-11-27. When will I get my refund?"),
            ("TKT-0011", "Delivery",      "Can you update me on shipping for order ORD-21547280? It's been 19 days."),
            ("TKT-0004", "Account",       "Hi team, Please help me reset my password. I no longer have access to my email sam130@company.co.uk."),
            ("TKT-0028", "Product Issue", "Hi team, Missing parts in my coffee grinder package from order ORD-53043520. You can reach me at sam157@company.co.uk."),
            ("TKT-0050", "Other",         "How do I apply a discount code at checkout?"),
        ]

        if "draft_tid" not in st.session_state:
            st.session_state["draft_tid"]     = ""
        if "draft_text" not in st.session_state:
            st.session_state["draft_text"]    = ""
        if "draft_cat" not in st.session_state:
            st.session_state["draft_cat"]     = "Delivery"
        if "draft_last_ex" not in st.session_state:
            st.session_state["draft_last_ex"] = ""

        selected = st.selectbox("Load example", [""] + [f"{t} ({c})" for t, c, _ in DRAFT_EXAMPLES],
                                key="draft_ex")

        # Only update fields when a NEW example is selected
        if selected and selected != st.session_state["draft_last_ex"]:
            for t, c, tx in DRAFT_EXAMPLES:
                if selected == f"{t} ({c})":
                    st.session_state["draft_tid"]     = t
                    st.session_state["draft_text"]    = tx
                    st.session_state["draft_cat"]     = c
                    st.session_state["draft_last_ex"] = selected

        st.text_input("Ticket ID", key="draft_tid")
        st.text_area("Ticket text", height=110, key="draft_text")
        category = st.selectbox("Category", CATEGORIES,
                                index=CATEGORIES.index(st.session_state["draft_cat"])
                                if st.session_state["draft_cat"] in CATEGORIES else 0,
                                key="draft_cat")
        api_key = st.text_input("OpenAI API Key", placeholder="sk-... (or set OPENAI_API_KEY env var)", type="password")

        ticket_id = st.session_state["draft_tid"]
        text      = st.session_state["draft_text"]

        st.markdown("""<div style="background:#FFF7ED;border:1px solid #FED7AA;border-radius:6px;padding:10px 14px;font-size:12px;color:#92400E;margin-top:8px;">
        API key sent only to your local FastAPI server. Template fallback used if no key provided.
        </div>""", unsafe_allow_html=True)

        if st.button("Generate Draft", type="primary", disabled=not text.strip(), use_container_width=True):
            payload = {"ticket_id": ticket_id or None, "text": text, "category": category}
            if api_key.strip(): payload["openai_api_key"] = api_key.strip()
            with st.spinner("Generating..."):
                result, err = api_post("/nlp/draft", json=payload)
            if err: st.error(err)
            elif result: st.session_state["draft_result"] = result

    with col2:
        st.markdown("#### Generated Draft")
        res = st.session_state.get("draft_result")
        if res:
            render_category_badge(res["category"])
            st.markdown("")
            if res.get("entities"):
                st.markdown("**Detected entities**")
                render_entities(res["entities"])
                st.markdown("")
            draft = res.get("draft_response", "")
            st.markdown("**Draft response**")
            st.markdown(f'<div class="draft-box">{draft}</div>', unsafe_allow_html=True)
            st.markdown("")
            if res.get("error"):
                st.info(f"Template fallback — {res['error']}")
            else:
                st.caption(f"Model: `{res.get('model','')}` · Tokens: {res.get('prompt_tokens',0)} prompt / {res.get('completion_tokens',0)} completion")
            st.text_area("Copy draft", value=draft, height=80)
            with st.expander("Prompt engineering notes"):
                st.markdown("""
                - **Concise** — 3–5 sentences max
                - **No hallucination** — only confirmed entities referenced
                - **Category-specific guidance** — tone/CTA adapted per category
                - **Missing info detection** — asks for order ID / email if absent
                - **JSON output** — `response_format: json_object` for reliable parsing
                - **Temperature 0.3** — deterministic, professional tone
                """)
        else:
            st.markdown("""<div style="text-align:center;padding:48px 24px;color:#94A3B8;border:2px dashed #E2E8F0;border-radius:8px;">
                <div style="font-size:28px;">Draft</div>
                <div style="font-size:13px;margin-top:8px;">Enter a ticket, select category, and click Generate</div>
            </div>""", unsafe_allow_html=True)


# ===============================================================================
# PAGE: Full NLP Pipeline
# ===============================================================================
elif st.session_state.page == "pipeline":
    st.markdown("""<div class="page-header">
        <h1>Full NLP Pipeline</h1>
        <p>Classify → Extract entities → Generate draft · single API call · <span class="mono">POST /nlp/analyse</span></p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        PIPE_EXAMPLES = {
            "TKT-0007": "Hi team, I returned laptop stand from order ORD-76842684 on 2025-11-27. When will I get my refund?",
            "TKT-0044": "I was charged twice for order ORD-90084745. I need a refund. You can reach me at jamie916@company.co.uk.",
            "TKT-0011": "Can you update me on shipping for order ORD-21547280? It's been 19 days.",
        }

        if "pipe_tid" not in st.session_state:
            st.session_state["pipe_tid"]      = ""
        if "pipe_text" not in st.session_state:
            st.session_state["pipe_text"]     = ""
        if "pipe_last_ex" not in st.session_state:
            st.session_state["pipe_last_ex"]  = ""

        example = st.selectbox("Load example", [""] + list(PIPE_EXAMPLES.keys()), key="pipe_ex")

        # Only update fields when a NEW example is selected
        if example and example != st.session_state["pipe_last_ex"]:
            st.session_state["pipe_tid"]     = example
            st.session_state["pipe_text"]    = PIPE_EXAMPLES[example]
            st.session_state["pipe_last_ex"] = example

        st.text_input("Ticket ID", key="pipe_tid")
        st.text_area("Ticket text", height=130, key="pipe_text")
        api_key = st.text_input("OpenAI API Key (optional)", type="password", key="pipe_key")

        ticket_id = st.session_state["pipe_tid"]
        text      = st.session_state["pipe_text"]

        if st.button("Run Full Pipeline", type="primary", disabled=not text.strip(), use_container_width=True):
            payload = {"ticket_id": ticket_id or None, "text": text}
            if api_key.strip(): payload["openai_api_key"] = api_key.strip()
            with st.spinner("Running classify → NER → draft..."):
                result, err = api_post("/nlp/analyse", json=payload)
            if err: st.error(err)
            elif result: st.session_state["pipeline_result"] = result

    with col2:
        res = st.session_state.get("pipeline_result")
        if res:
            st.markdown("**1 - Classification**")
            render_category_badge(res["category"], res["confidence"])
            st.caption(f"Method: `{res.get('classifier_method','')}`")
            st.markdown("")
            st.markdown("**2 - Entities**")
            entities = res.get("entities", [])
            if entities: render_entities(entities)
            else: st.caption("No entities detected.")
            st.markdown("")
            st.markdown("**3 - Draft Response**")
            draft = res.get("draft_response", "")
            st.markdown(f'<div class="draft-box">{draft}</div>', unsafe_allow_html=True)
            if res.get("draft_error"):
                st.info(f"Template fallback: {res['draft_error']}")
            st.caption(f"Total latency: {res.get('total_latency_ms', 0)} ms")
            with st.expander("Raw JSON response"):
                st.json(res)
        else:
            st.markdown("""<div style="text-align:center;padding:64px 24px;color:#94A3B8;border:2px dashed #E2E8F0;border-radius:8px;">
                <div style="font-size:36px;">Pipeline</div>
                <div style="font-size:13px;margin-top:8px;">Enter a ticket and click Run Full Pipeline<br>
                <span style="font-size:11px;">All three NLP steps run in a single API call</span></div>
            </div>""", unsafe_allow_html=True)
