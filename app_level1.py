import streamlit as st
from PIL import Image

from src.infer_category import load_category_model, predict_category_topk
from src.infer_submodel import load_submodel, predict_subclass_topk
from src.kids_explainer import explain_animal_ollama

st.set_page_config(page_title="Hybrid Image Classifier", layout="centered")

st.title("🧠 Hybrid Image Classifier")
st.caption("Upload an image or take a photo. The app predicts the category first, then the exact class, then explains it.")

CATEGORY_MODEL_DIR = "outputs/models/category_model"

CATEGORY_TO_FOLDER = {
    "Animal": "animal_model",
    "animal": "animal_model",
    "flags": "flags_model",
    "flower": "flower_model",
    "food": "food_model",
    "fruits_and_vegetables": "fruits_and_vegetables_model",
    "fruits and vegetables": "fruits_and_vegetables_model",
    "monument": "monument_model",
    "shapes": "shapes_model",
    "sports_equipments": "sports_equipments_model",
    "sports equipments": "sports_equipments_model",
    "vehicle": "vehicle_model",
    "weather": "weather_model",
}

CATEGORY_DISPLAY = {
    "Animal": "Animal",
    "animal": "Animal",
    "flags": "Flags",
    "flower": "Flower",
    "food": "Food",
    "fruits_and_vegetables": "Fruits and Vegetables",
    "fruits and vegetables": "Fruits and Vegetables",
    "monument": "Monument",
    "shapes": "Shapes",
    "sports_equipments": "Sports Equipments",
    "sports equipments": "Sports Equipments",
    "vehicle": "Vehicle",
    "weather": "Weather",
}


@st.cache_resource
def load_category():
    model, idx_to_class, cfg = load_category_model(CATEGORY_MODEL_DIR)
    return model, idx_to_class, cfg


@st.cache_resource
def load_child_model(folder_name: str):
    model_dir = f"outputs/models/{folder_name}"
    model, idx_to_class, cfg = load_submodel(model_dir)
    return model, idx_to_class, cfg


@st.cache_data(show_spinner=False)
def cached_explain(final_label: str, category_label: str, llm_name: str):
    prompt = (
        f"Detected object: {final_label}. "
        f"Detected category: {category_label}. "
        f"Explain this to a child in simple language. "
        f"Return a short intro, 5 fun facts, one quiz question, and one useful or safe note."
    )
    return explain_animal_ollama(prompt, model=llm_name)


category_model, category_idx_to_class, category_cfg = load_category()

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

    # ----------------------------
    # Step 1: Category prediction
    # ----------------------------
    category_preds = predict_category_topk(
        category_model,
        category_idx_to_class,
        img,
        image_size=int(category_cfg["image_size"]),
        k=3
    )

    top_category = category_preds[0]["label"]
    display_category = CATEGORY_DISPLAY.get(top_category, top_category)

    st.markdown(f"## Step 1: Category → 🏷️ {display_category}")

    with st.expander("See top category guesses"):
        for i, p in enumerate(category_preds, start=1):
            pretty_label = CATEGORY_DISPLAY.get(p["label"], p["label"])
            st.write(f"{i}. **{pretty_label}** — {p['confidence']*100:.1f}%")

    # ----------------------------
    # Step 2: Load correct child model
    # ----------------------------
    folder_name = CATEGORY_TO_FOLDER.get(top_category)

    if folder_name is None:
        st.error(f"No child model mapping found for category: {top_category}")
        st.stop()

    try:
        child_model, child_idx_to_class, child_cfg = load_child_model(folder_name)
    except Exception as e:
        st.error(f"Could not load child model for category '{top_category}'.")
        st.exception(e)
        st.stop()

    # ----------------------------
    # Step 3: Final class prediction
    # ----------------------------
    class_preds = predict_subclass_topk(
        child_model,
        child_idx_to_class,
        img,
        image_size=int(child_cfg["image_size"]),
        k=3
    )

    top_class = class_preds[0]["label"]

    st.markdown(f"## Step 2: Final Class → ✅ {top_class}")

    with st.expander("See top class guesses"):
        for i, p in enumerate(class_preds, start=1):
            st.write(f"{i}. **{p['label']}** — {p['confidence']*100:.1f}%")

    # ----------------------------
    # Step 4: LLM explanation
    # ----------------------------
    st.divider()
    st.subheader("📚 Explanation")

    llm_name = "qwen2.5:3b"

    with st.spinner("Generating explanation..."):
        try:
            info = cached_explain(top_class, display_category, llm_name)

            st.markdown(f"### {info.get('title', top_class)}")

            if info.get("short"):
                st.write(info["short"])

            facts = info.get("facts", [])
            if facts:
                st.markdown("**Fun facts:**")
                for fact in facts:
                    st.write("•", fact)

            if info.get("habitat"):
                st.markdown(f"**Where we see it:** {info['habitat']}")
            if info.get("food"):
                st.markdown(f"**Related details:** {info['food']}")
            if info.get("quiz"):
                st.markdown(f"**Quiz:** {info['quiz']}")
            if info.get("safety_note"):
                st.info(info["safety_note"])

        except Exception as e:
            st.warning(str(e)) 
            st.info("Ensure Ollama is running and qwen2.5:3b is available")