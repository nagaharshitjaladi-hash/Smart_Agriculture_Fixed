"""Train a crop classifier from a labelled CSV dataset.

Expected file: backend/data/Crop_recommendation.csv
Required columns: N, P, K, temperature, humidity, ph, rainfall, label
The common Crop Recommendation dataset uses these column names.
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

HERE=Path(__file__).resolve().parent
data_path=HERE/"data"/"Crop_recommendation.csv"
if not data_path.exists():
    raise SystemExit(f"Dataset not found: {data_path}\nPlace a properly licensed Crop_recommendation.csv there.")
df=pd.read_csv(data_path)
features=["N","P","K","temperature","humidity","ph","rainfall"]
target="label"
missing=[c for c in features+[target] if c not in df.columns]
if missing:
    raise SystemExit("Missing required columns: "+", ".join(missing))
X=df[features].apply(pd.to_numeric, errors="coerce")
y=df[target].astype(str)
valid=X.notna().all(axis=1) & y.notna()
X,y=X[valid],y[valid]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
model=Pipeline([("imputer",SimpleImputer(strategy="median")),("classifier",RandomForestClassifier(n_estimators=250,random_state=42,class_weight="balanced"))])
model.fit(X_train,y_train)
pred=model.predict(X_test)
print("Accuracy:",round(accuracy_score(y_test,pred),4))
print(classification_report(y_test,pred))
print("Confusion matrix:\n",confusion_matrix(y_test,pred))
out=HERE/"models"/"crop_model.joblib"
out.parent.mkdir(parents=True,exist_ok=True)
joblib.dump(model,out)
print("Saved model:",out)
