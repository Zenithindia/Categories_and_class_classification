import streamlit as st
from PIL import Image

from src.infer_category import load_category_model, predict_category_topk
from src.kids_explainer import explain_animal_ollama

st.set_page_config(page_title="Category Classifier", layout="centered")

st.title("🧠 Model 1: Category Classifier")
st.caption("Upload an image or take a photo. The model will predict the broad category first.")

MODEL_DIR = "outputs/models/category_model"


@st.cache_resource
def load():
    model, idx_to_class, cfg = load_category_model(MODEL_DIR)
    return model, idx_to_class, cfg


model, idx_to_class, cfg = load()

tab1, tab2 = st.tabs(["📷 Camera", "📁 Upload"])
img = None

with tab1:
    cam = st.camera_input("Take a photo")
    if cam is not None:
        img = Image.open(cam)

with tab2:
    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"]
    )
    if uploaded is not None:
        img = Image.open(uploaded)

if img is not None:
    st.image(img, caption="Input Image", use_container_width=True)

    preds = predict_category_topk(
        model,
        idx_to_class,
        img,
        image_size=int(cfg["image_size"]),
        k=3
    )

    st.markdown(f"## Predicted Category: 🏷️ {preds[0]['label']}")

    st.subheader("Top-3 category guesses")
    for i, p in enumerate(preds, start=1):
        st.write(f"{i}. **{p['label']}** — {p['confidence']*100:.1f}%")

    st.divider()
    st.subheader("📚 Explanation")

    top_label = preds[0]["label"]
    model_name = "qwen2.5:3b"

    @st.cache_data(show_spinner=False)
    def cached_explain(lbl: str, llm_name: str):
        prompt_label = (
            f"{lbl} category. Explain to a child what this category means. "
            f"Give simple examples and fun facts."
        )
        return explain_animal_ollama(prompt_label, model=llm_name)

    with st.spinner("Generating explanation (offline)..."):
        try:
            info = cached_explain(top_label, model_name)

            st.markdown(f"### {info.get('title', top_label)}")
            if info.get("short"):
                st.write(info["short"])

            facts = info.get("facts", [])
            if facts:
                st.markdown("**Fun facts:**")
                for f in facts:
                    st.write("•", f)

            if info.get("habitat"):
                st.markdown(f"**Where we see it:** {info['habitat']}")
            if info.get("food"):
                st.markdown(f"**Related examples:** {info['food']}")
            if info.get("quiz"):
                st.markdown(f"**Quiz:** {info['quiz']}")
            if info.get("safety_note"):
                st.info(info["safety_note"])

        except Exception as e:
            st.warning(str(e))
            st.info("Tip: Ensure Ollama is installed, running, and the model is pulled.")