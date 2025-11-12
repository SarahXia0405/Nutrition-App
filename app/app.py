# app/app.py
# -------------------------------------------------------------
# Import path fix so we can do `from src.nutrition ...`
# -------------------------------------------------------------
import os, sys
from pathlib import Path

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent           # .../food_nutrition_app_full_with_data
SRC_DIR = PROJECT_ROOT / "src"

# add both the project root and the src folder to sys.path
for p in (str(PROJECT_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# now imports will work whether we run from root or IDE
try:
    from src.nutrition.model import NutritionScorer
    from src.nutrition.detection import stub_detect
    from src.nutrition.mapping import NUTRIENTS_PER_100G
except ModuleNotFoundError:
    # fallback if environment strips the leading package name
    from nutrition.model import NutritionScorer
    from nutrition.detection import stub_detect
    from nutrition.mapping import NUTRIENTS_PER_100G

# -------------------------------------------------------------
# app proper
# -------------------------------------------------------------
import io
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Nutrition Density App", page_icon="🍎", layout="wide")
st.title("🍎 Nutrition Density App")
st.caption("Two options: upload an image (Detect) or enter foods manually. Edit names & portions before scoring.")

# Sidebar: model upload (optional; if skipped, baseline scorer is used)
st.sidebar.header("Model")
model_file = st.sidebar.file_uploader("Upload trained model (.pkl/.joblib)", type=["pkl", "joblib"])
scorer = NutritionScorer(model_file=model_file)

def make_table(rows=None):
    cols = ["food_name", "portion_g", "protein_g","fat_g","carbs_g","fiber_g","sugar_g","sodium_mg"]
    if not rows:
        data = [{c: "" for c in cols} for _ in range(3)]
        for r in data:
            r["portion_g"] = 100
            r["protein_g"]=0; r["fat_g"]=0; r["carbs_g"]=0; r["fiber_g"]=0; r["sugar_g"]=0; r["sodium_mg"]=0
        return pd.DataFrame(data, columns=cols)
    return pd.DataFrame(rows, columns=cols)

tab1, tab2 = st.tabs(["📷 Image Mode", "⌨️ Manual Mode"])

# ---------------- Image Mode ----------------
with tab1:
    st.subheader("📷 Upload & Detect")
    img_file = st.file_uploader("Upload a meal photo", type=["jpg","jpeg","png"], key="img_upl")
    detect_btn = st.button("Detect", disabled=(img_file is None), key="detect_btn")

    if "img_table" not in st.session_state:
        st.session_state["img_table"] = make_table()

    if detect_btn and img_file is not None:
        image = Image.open(io.BytesIO(img_file.read())).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        det = stub_detect(image)  # list of {food_name, portion_g, confidence}

        if len(det) == 0:
            st.info("No auto-detections yet. Fill the table manually or switch to Manual Mode.")

        rows = []
        for item in det:
            base = NUTRIENTS_PER_100G.get(item.get("food_name","").lower(), {})
            rows.append({
                "food_name": item.get("food_name",""),
                "portion_g": item.get("portion_g", 100),
                "protein_g": base.get("protein_g", 0.0),
                "fat_g": base.get("fat_g", 0.0),
                "carbs_g": base.get("carbs_g", 0.0),
                "fiber_g": base.get("fiber_g", 0.0),
                "sugar_g": base.get("sugar_g", 0.0),
                "sodium_mg": base.get("sodium_mg", 0.0),
            })
        st.session_state["img_table"] = make_table(rows)

    st.write("### Detected/Editable Items")
    edited = st.data_editor(st.session_state["img_table"], num_rows="dynamic", use_container_width=True, key="img_editor")

    if st.button("Estimate Nutrition Density (Image Mode)"):
        df = edited.copy()
        scale = df["portion_g"].astype(float) / 100.0
        features = pd.DataFrame({
            "protein_g": df["protein_g"].astype(float) * scale,
            "fat_g": df["fat_g"].astype(float) * scale,
            "carbs_g": df["carbs_g"].astype(float) * scale,
            "fiber_g": df["fiber_g"].astype(float) * scale,
            "sugar_g": df["sugar_g"].astype(float) * scale,
            "sodium_mg": df["sodium_mg"].astype(float) * scale,
            "portion_g": df["portion_g"].astype(float)
        })
        score = scorer.predict(features)
        st.success(f"Estimated Nutrition Density: **{score:.1f} / 100**")

# ---------------- Manual Mode ----------------
with tab2:
    st.subheader("⌨️ Enter Foods Manually")
    if "manual_table" not in st.session_state:
        st.session_state["manual_table"] = make_table()
    st.write("Enter each food's name, portion (g), and per-100g nutrient values.")

    edited2 = st.data_editor(st.session_state["manual_table"], num_rows="dynamic", use_container_width=True, key="manual_editor")

    if st.button("Estimate Nutrition Density (Manual Mode)"):
        df = edited2.copy()
        scale = df["portion_g"].astype(float) / 100.0
        features = pd.DataFrame({
            "protein_g": df["protein_g"].astype(float) * scale,
            "fat_g": df["fat_g"].astype(float) * scale,
            "carbs_g": df["carbs_g"].astype(float) * scale,
            "fiber_g": df["fiber_g"].astype(float) * scale,
            "sugar_g": df["sugar_g"].astype(float) * scale,
            "sodium_mg": df["sodium_mg"].astype(float) * scale,
            "portion_g": df["portion_g"].astype(float)
        })
        score = scorer.predict(features)
        st.success(f"Estimated Nutrition Density: **{score:.1f} / 100**")

st.divider()
st.caption("Upload your trained model to replace the baseline. Plug YOLO/CLIP into detection stub for auto-fill.")
