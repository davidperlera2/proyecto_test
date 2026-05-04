import rasterio
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# =========================
# 1. CARGAR DATOS
# =========================

sentinel2_path = "../Imagenes_Satelitales/Sentinel2_2025.tif"
sentinel1_path = "../Imagenes_Satelitales/SAR_2025_1.tif"
gt_path = "../gt.tif"


with rasterio.open(sentinel2_path) as src:
    img = src.read()
    profile = src.profile


with rasterio.open(sentinel1_path) as src:
    sar = src.read()

with rasterio.open(gt_path) as src:
    gt = src.read(1)



print("Shape Sentinel:", img.shape)
print("Sahpe SAR:", sar.shape)
print("Shape GroundTruth:", gt.shape)

# =========================
# 2. PREPARAR DATOS
# =========================

B2 = img[0]
B3 = img[1]
B4 = img[2]
B8 = img[3]
B11 = img[4] 

epsilon = 1e-10 

NDVI = (B8 - B4) / (B8 + B4 + epsilon)
NDBI = (B11 - B8) / (B11 + B8 + epsilon)
NDWI = (B3 - B8) / (B3 + B8 + epsilon)

VH = sar[0]
VV = sar[1]
ratio = sar[2]

stack = np.stack([
    B2, B3, B4, B8, B11,
    NDVI, NDBI, NDWI,
    VV, VH, ratio
], axis=0)

n_features, rows, cols = stack.shape

X_full = stack.reshape(n_features, rows * cols).T

# etiquetas
y_full = gt.reshape(rows * cols)

# máscara (solo datos válidos)
mask = y_full != -1

# datos de entrenamiento
X = X_full[mask]
y = y_full[mask]

print("Datos entrenamiento:", X.shape)

# =========================
# 3. DIVIDIR DATASET
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================
# 4. ENTRENAR MODELO
# =========================

model = RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# =========================
# 5. EVALUAR MODELO
# =========================

y_pred_test = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred_test))
print("\nReporte de clasificación:\n")
print(classification_report(y_test, y_pred_test))

# =========================
# 6. PREDECIR TODO EL MAPA
# =========================

print("\nPrediciendo mapa completo...")

y_pred_full = model.predict(X_full)

# reconstruir imagen
pred = y_pred_full.reshape(rows, cols)

# =========================
# 7. GUARDAR RASTER
# =========================

profile.update(
    dtype=rasterio.int16,
    count=1,
    nodata=-1
)

output_path = "clasificacion_final.tif"

with rasterio.open(output_path, "w", **profile) as dst:
    dst.write(pred.astype(rasterio.int16), 1)

print("\nMapa guardado en:", output_path)

features = [
    "B2","B3","B4","B8","B11",
    "NDVI","NDBI","NDWI",
    "VV","VH","VV/VH"
]

importances = model.feature_importances_

for f, imp in zip(features, importances):
    print(f, imp)

joblib.dump(model, "modelo_rf_2.pkl")