from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager
import sqlite3
import json

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "agriculture.db"
MODEL_PATH = ROOT / "models" / "crop_model.joblib"

app = FastAPI(
    title="Smart Agriculture Using AI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature TEXT NOT NULL,
                input_json TEXT NOT NULL,
                result TEXT NOT NULL,
                confidence REAL,
                created_at TEXT NOT NULL
            )
        """)


init_db()


class CropInput(BaseModel):
    nitrogen: float = Field(ge=0, le=300)
    phosphorus: float = Field(ge=0, le=300)
    potassium: float = Field(ge=0, le=300)
    temperature: float = Field(ge=-10, le=60)
    humidity: float = Field(ge=0, le=100)
    ph: float = Field(ge=0, le=14)
    rainfall: float = Field(ge=0, le=1000)


class FertilizerInput(BaseModel):
    crop: str = Field(min_length=1, max_length=80)
    soil_type: str = Field(min_length=1, max_length=80)
    nitrogen: float = Field(ge=0, le=300)
    phosphorus: float = Field(ge=0, le=300)
    potassium: float = Field(ge=0, le=300)
    ph: float = Field(ge=0, le=14)


def record(feature, data, result, confidence=None):
    with db() as conn:
        conn.execute(
            """
            INSERT INTO history
            (feature, input_json, result, confidence, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                feature,
                json.dumps(data),
                result,
                confidence,
                datetime.now(timezone.utc).isoformat()
            )
        )


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_ready": MODEL_PATH.exists()
    }


@app.post("/api/crop-recommendation")
def crop_recommendation(x: CropInput):
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Crop model is not trained. "
                "Follow backend/README.md to train the model."
            )
        )

    try:
        model = joblib.load(MODEL_PATH)

        values = pd.DataFrame(
            [[
                x.nitrogen,
                x.phosphorus,
                x.potassium,
                x.temperature,
                x.humidity,
                x.ph,
                x.rainfall
            ]],
            columns=[
                "N", "P", "K", "temperature",
                "humidity", "ph", "rainfall"
            ]
        )

        prediction = str(model.predict(values)[0])

        confidence = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(values)[0]
            confidence = round(
                float(np.max(probabilities)) * 100, 2
            )

        data = x.model_dump()

        record("crop", data, prediction, confidence)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "note": (
                "This is a model-based recommendation, "
                "not a guarantee. Verify with local "
                "agricultural guidance."
            )
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Crop prediction failed: {str(e)}"
        )


@app.post("/api/fertilizer-recommendation")
def fertilizer_recommendation(x: FertilizerInput):
    nutrients = {
        "N": x.nitrogen,
        "P": x.phosphorus,
        "K": x.potassium
    }

    low = [
        name for name, value in nutrients.items()
        if value < 40
    ]

    if low:
        result = (
            "Review soil-test results for low "
            + ", ".join(low)
            + " levels. Consider a fertilizer "
              "supplying the deficient nutrient."
        )

    elif x.ph < 5.5:
        result = (
            "Soil is acidic. Seek local soil-test "
            "advice about pH correction before "
            "selecting fertilizer."
        )

    elif x.ph > 7.5:
        result = (
            "Soil is alkaline. Ask an agricultural "
            "expert to interpret nutrient availability "
            "and select an appropriate fertilizer."
        )

    else:
        result = (
            "Nutrient values are not below the app's "
            "screening threshold. Use a crop-specific "
            "soil-test recommendation."
        )

    data = x.model_dump()
    record("fertilizer", data, result)

    return {
        "recommendation": result,
        "method": "Rule-based screening",
        "warning": (
            "This is not a dosage prescription. "
            "Confirm with a soil test and local "
            "agricultural guidance."
        )
    }


@app.post("/api/plant-disease")
async def plant_disease(file: UploadFile = File(...)):
    allowed_types = {
        "image/jpeg",
        "image/png"
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPG or PNG image."
        )

    content = await file.read()
    await file.close()

    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be 5 MB or smaller."
        )

    raise HTTPException(
        status_code=503,
        detail=(
            "Plant disease model is not configured. "
            "No prediction was made."
        )
    )


@app.get("/api/history")
def history():
    with db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM history
            ORDER BY id DESC
            LIMIT 200
            """
        ).fetchall()

    return [
        {
            **dict(row),
            "input": json.loads(row["input_json"])
        }
        for row in rows
    ]


@app.get("/api/stats")
def stats():
    with db() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM history"
        ).fetchone()[0]

        crops = conn.execute(
            "SELECT COUNT(*) FROM history WHERE feature='crop'"
        ).fetchone()[0]

        fertilizer = conn.execute(
            "SELECT COUNT(*) FROM history WHERE feature='fertilizer'"
        ).fetchone()[0]

        diseases = conn.execute(
            "SELECT COUNT(*) FROM history WHERE feature='disease'"
        ).fetchone()[0]

        crop_counts = conn.execute(
            """
            SELECT result, COUNT(*) AS n
            FROM history
            WHERE feature='crop'
            GROUP BY result
            ORDER BY n DESC
            """
        ).fetchall()

    return {
        "total": total,
        "crops": crops,
        "fertilizer": fertilizer,
        "diseases": diseases,
        "crop_counts": [
            {
                "crop": row["result"],
                "count": row["n"]
            }
            for row in crop_counts
        ]
    }


@app.delete("/api/history")
def clear_history():
    with db() as conn:
        conn.execute("DELETE FROM history")

    return {"deleted": True}


@app.get("/api/weather")
def weather():
    raise HTTPException(
        status_code=501,
        detail=(
            "Live weather is not configured. "
            "Add a weather provider to enable this feature."
        )
    )