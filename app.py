"""Frontend (Streamlit) for the Pneumonia Detection decision-support demo."""
import streamlit as st

from backend.inference import load_config, load_model, predict

st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁", layout="centered")


@st.cache_resource(show_spinner="Loading model ...")
def get_model():
    return load_model()


st.title("🫁 Pneumonia Detection from Chest X-rays")
st.caption("AI decision-support tool - a second opinion for clinicians, not a diagnosis.")

cfg = load_config()
with st.sidebar:
    st.header("About the model")
    st.write(f"**Model:** {cfg.get('model_name', 'n/a')}")
    st.write(f"**Decision threshold:** {cfg.get('threshold', 0.5):.2f}")
    if "test_auc" in cfg:
        st.write(f"**Test ROC-AUC:** {cfg['test_auc']:.2f}")
    if "test_recall" in cfg:
        st.write(f"**Test recall (pneumonia):** {cfg['test_recall']:.2f}")
    if "test_precision" in cfg:
        st.write(f"**Test precision (pneumonia):** {cfg['test_precision']:.2f}")
    st.info("Trained on the RSNA Pneumonia Detection Challenge data. The model is tuned to "
            "catch most pneumonia cases (high recall), so some healthy scans will be flagged.")

get_model()

uploaded = st.file_uploader("Upload a chest X-ray (DICOM, PNG or JPG)", type=["dcm", "png", "jpg", "jpeg"])

if uploaded is not None:
    data = uploaded.getvalue()
    try:
        with st.spinner("Analysing image ..."):
            result = predict(data, uploaded.name)
    except Exception as exc:  # unreadable / unsupported file
        st.error(f"Could not read this file: {exc}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.image(result["image"], caption="Pre-processed input (224 x 224, grayscale)", clamp=True)
        with col2:
            if result["is_positive"]:
                st.error(f"Predicted class: **{result['label']}**")
            else:
                st.success(f"Predicted class: **{result['label']}**")
            st.metric("Pneumonia probability", f"{result['probability']:.2f}")
            st.progress(min(max(result["probability"], 0.0), 1.0))
            st.caption(f"Flagged as pneumonia when probability >= {result['threshold']:.2f}.")
        st.warning("For decision support only. A qualified radiologist must confirm every result.")
else:
    st.write("Upload an image to see the predicted class and probability.")
