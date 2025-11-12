# 🍎 Nutrition Density App (Dual Mode)

Estimate a **Nutrition Density Score (0–100)** from either an **uploaded image** or **manual entry**.
🌐 **Play with it:** [Web](https://nutrition-app-ty4zym6xnthsrntdftp7n6.streamlit.app/)

## Modes
1. **Image Mode** — Upload a photo → click **Detect** → edit foods & portions → **Estimate**.  
   - Detection is a **stub** (no fine-tuning required). Plug in YOLO/CLIP later in `src/nutrition/detection.py`.
2. **Manual Mode** — Enter foods and portion sizes directly, with per-100g nutrient values.

## Model
Upload a trained model (`.pkl` / `.joblib`) in the sidebar.  
If omitted, a **rule-based baseline** is used.

## Train your own model
Open `notebooks/train_model.ipynb` and run the cells to:
- load `data/raw/nutrition_raw.csv`
- preprocess features
- train a baseline ElasticNet or RandomForest
- export to `models/nutrition_density_model.pkl`
Then relaunch the app and upload the model in the sidebar.

## Run
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```
