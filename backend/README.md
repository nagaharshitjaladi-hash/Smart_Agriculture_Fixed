# Backend setup

Requires Python 3.10+.

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

API docs: http://127.0.0.1:8000/docs

## Train crop model
Obtain a properly licensed crop recommendation CSV and place it at:
`backend/data/Crop_recommendation.csv`

Required columns: `N,P,K,temperature,humidity,ph,rainfall,label`.

Then run:
```bash
python train_model.py
```
The model is saved to `backend/models/crop_model.joblib`. Until this model exists, crop recommendations return HTTP 503 rather than inventing predictions.

Fertilizer advice is explicitly rule-based screening, not an ML model. Plant disease and live weather endpoints intentionally report that they are not configured; they do not fabricate results.
