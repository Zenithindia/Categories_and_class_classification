# Hybrid Multi-Category Image Classification System with Hierarchical CNN + Local LLM

A hierarchical computer vision project that classifies an input image in **two stages**:

1. **Model 1 – Category Classifier**  
   Predicts the broad category of the image such as:
   - Animal
   - Vehicle
   - Flower
   - Food
   - Fruits and Vegetables
   - Monument
   - Sports Equipments
   - Flags
   - Weather

2. **Model 2 – Category-Specific Classifier**  
   Once the category is predicted, the corresponding specialized model is loaded dynamically to predict the **final class**.

3. **LLM Explanation Layer**  
   A local LLM (via **Ollama**) generates a simple, child-friendly explanation of the final prediction.

The project also includes a **Streamlit web app** with:
- image upload
- camera input
- category prediction
- final class prediction
- explanation generation

---

# Project Objective

The aim of this project is to build an **educational and scalable image classification system** that can recognize objects from multiple domains instead of only one dataset.

Unlike a traditional single-model classifier, this project uses a **hybrid hierarchical approach**:

- first classify the **category**
- then classify the **specific class within that category**
- then explain the result in simple language using a **local LLM**

This architecture is more scalable, easier to maintain, and better suited for real-world multi-domain image classification.

---

# Why Hierarchical Classification?

If we train one flat model on all classes together, it must distinguish between very different domains like:
> ALL CATEGORIES AND CLASSES ARE IN FILE "all_categories_and_classes.txt"
- motorcycle
- pizza
- rectangle
- Eiffel Tower
- rose
- India flag

This creates:
- a very large class space
- more class confusion
- lower accuracy
- poor scalability

To solve this, the project uses:

## Stage 1 – Category Model
Predicts the broad domain.

## Stage 2 – Class Model
Loads the correct category-specific model and predicts the final class.

This reduces the classification search space and improves specialization.

---

# Current Project Scope

At this stage, the system supports the following categories:

- Animal
- Flags
- Flower
- Food
- Fruits and Vegetables
- Monument
- Sports Equipments
- Vehicle
- Weather

Each category has its own submodel trained on its category-specific classes.

---

# System Workflow

```text
Input Image
   ↓
Model 1: Category Classifier
   ↓
Predicted Category
   ↓
Load Corresponding Submodel
   ↓
Model 2: Class Prediction
   ↓
Final Class
   ↓
Local LLM Explanation (Ollama)
   ↓
Streamlit UI Output
```

# Project Folder Structure
> U HAVE TO SETUP OUTPUTS BY OWN BEACUSE WE AERE UNABLE TO UPLOAD MODELS BECAUSE OF THE SIZE ISSUE
```
Animal-species-classifier/
│
├── src/
│   ├── infer_category.py
│   ├── infer_submodel.py
│   ├── infer_level1.py
│   ├── kids_explainer.py
│   ├── train.py
│   ├── datasets.py
│   ├── models.py
│   ├── evaluate.py
│   ├── utils.py
│   └── config.py
│
├── outputs/
│   └── models/
│       ├── category_model/
│       ├── animal_model/
│       ├── vehicle_model/
│       ├── flower_model/
│       ├── food_model/
│       ├── monument_model/
│       ├── weather_model/
│       ├── shapes_model/
│       ├── sports_equipments_model/
│       └── fruits_and_vegetables_model/
│
├── data/
├── notebooks/
├── scripts/
├── app_level1.py
├── requirements.txt
├── config.yaml
└── README.md

```

Run command - streamlit run app_level_1.py
