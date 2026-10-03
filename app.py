import numpy as np
import streamlit as st

from inference import (
    CLASS_NAMES,
    DISPLAY_NAMES,
    MODEL_REGISTRY,
    SEVERITY_ORDER,
    build_gradcam_model,
    compute_gradcam,
    load_keras_model,
    overlay_heatmap,
    predict_probabilities,
)
from preprocessing import InvalidImageError, prepare_image, validate_and_load

st.set_page_config(page_title="DR Severity Classifier", page_icon="👁️", layout="wide")


# Only one model is kept in memory at a time (max_entries=1). Switching models
# evicts the previous one, which keeps RAM low on Streamlit Community Cloud.
@st.cache_resource(show_spinner=False, max_entries=1)
def get_model(model_key: str):
    return load_keras_model(model_key)


@st.cache_resource(show_spinner=False, max_entries=1)
def get_gradcam_model(model_key: str):
    return build_gradcam_model(get_model(model_key))


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    model_key = st.selectbox("Model", list(MODEL_REGISTRY.keys()))
    show_cam = st.checkbox("Show Grad-CAM heatmap", value=True)
    st.divider()
    st.markdown("**Classes**")
    st.markdown(
        "\n".join(f"- {DISPLAY_NAMES[name]}" for name in SEVERITY_ORDER)
    )

# --------------------------------------------------------------------- main
st.title("Diabetic Retinopathy Severity Classifier")
st.caption("Upload a retinal fundus photograph, choose a model, and get a severity prediction.")
st.warning(
    "For research and education only. This is not a medical device and its output "
    "is not a diagnosis. Consult a qualified eye-care professional."
)

uploaded = st.file_uploader("Retinal fundus image (JPEG)", type=["jpg", "jpeg"])
if uploaded is None:
    st.info("Upload a .jpg or .jpeg image to begin.")
    st.stop()

# 1) Validate: real JPEG, not corrupted
try:
    pil_image = validate_and_load(uploaded.getvalue(), uploaded.name)
except InvalidImageError as exc:
    st.error(str(exc))
    st.stop()

# 2) Resize + preprocess (same as training)
prepared = prepare_image(pil_image, uploaded.getvalue())

# 3) Load the selected model
try:
    with st.spinner(f"Loading {model_key}..."):
        model = get_model(model_key)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # noqa: BLE001 - show the real reason to the developer
    st.error(
        f"Could not load {model_key}. This usually means the TensorFlow/Keras version "
        f"in requirements.txt differs from the one used for training.\n\n`{exc}`"
    )
    st.stop()

# 4) Predict
probs = predict_probabilities(model, prepared.batch)
top = int(np.argmax(probs))
top_name = CLASS_NAMES[top]
confidence = float(probs[top])

# 5) Grad-CAM (optional, never blocks the prediction)
cam_image = None
cam_note = None
if show_cam:
    try:
        with st.spinner("Computing Grad-CAM..."):
            grad_model = get_gradcam_model(model_key)
            heatmap = compute_gradcam(grad_model, prepared.batch, top)
        if heatmap.max() <= 0:
            cam_note = "Grad-CAM produced no signal for this image."
        else:
            cam_image = overlay_heatmap(prepared.resized, heatmap)
    except Exception as exc:  # noqa: BLE001
        cam_note = f"Grad-CAM is unavailable for this model: {exc}"

# ------------------------------------------------------------------ results
st.subheader("Result")
res_left, res_right = st.columns([1, 2])
with res_left:
    st.metric("Predicted class", DISPLAY_NAMES[top_name])
    st.metric("Confidence", f"{confidence * 100:.1f}%")
    st.caption(f"Model: {model_key}")
    if confidence < 0.5:
        st.info("The model is not very confident about this image.")
with res_right:
    for name in SEVERITY_ORDER:
        idx = CLASS_NAMES.index(name)
        p = float(probs[idx])
        label = DISPLAY_NAMES[name]
        text = f"**{label}: {p * 100:.1f}%**" if idx == top else f"{label}: {p * 100:.1f}%"
        st.progress(min(max(p, 0.0), 1.0), text=text)

st.subheader("Images")
cols = st.columns(3 if show_cam else 2)
with cols[0]:
    st.image(prepared.resized, caption="Input (resized to 224x224)")
with cols[1]:
    st.image(
        np.clip(prepared.preprocessed, 0, 255).astype(np.uint8),
        caption="After preprocessing (model input)",
    )
if show_cam:
    with cols[2]:
        if cam_image is not None:
            st.image(cam_image, caption=f"Grad-CAM for '{DISPLAY_NAMES[top_name]}'")
        else:
            st.info(cam_note)

with st.expander("What the app did to your image"):
    st.markdown(
        "1. Checked the file is a real, intact JPEG\n"
        "2. Resized to 224x224 (RGB)\n"
        "3. Brightness normalised to a target mean of 75\n"
        "4. Contrast enhanced with CLAHE\n"
        "5. Noise reduced with a bilateral filter\n"
        "6. Sharpened with an unsharp mask\n"
        "7. Passed to the model with no extra scaling"
    )
