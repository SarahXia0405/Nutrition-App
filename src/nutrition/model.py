import io, joblib, numpy as np, pandas as pd

class NutritionScorer:
    """Loads a trained model if provided, else uses a rule-based baseline."""
    def __init__(self, model_file=None):
        self.model = None
        if model_file is not None:
            try:
                raw = model_file.read()
                model_file.seek(0)
                self.model = joblib.load(io.BytesIO(raw))
            except Exception:
                self.model = None

    def _baseline_score(self, X: pd.DataFrame) -> float:
        protein = X.get("protein_g", 0).sum()
        fiber   = X.get("fiber_g", 0).sum()
        fat     = X.get("fat_g", 0).sum()
        carbs   = X.get("carbs_g", 0).sum()
        sugar   = X.get("sugar_g", 0).sum()
        sodium  = X.get("sodium_mg", 0).sum() / 1000.0
        score = 50 + 1.2*protein + 2.0*fiber - 0.5*fat - 0.3*carbs - 1.5*sugar - 2.0*sodium
        return float(np.clip(score, 0, 100))

    def predict(self, features: pd.DataFrame) -> float:
        if self.model is not None:
            try:
                y = self.model.predict(features)
                return float(np.clip(np.mean(y), 0, 100))
            except Exception:
                pass
        return self._baseline_score(features)