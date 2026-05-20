import rasterio
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.ndimage import generic_filter, label
import joblib

# =========================
# 1. CARGAR DATOS
# =========================

sentinel2_path = "../Imagenes_Satelitales/Sentinel2_2025.tif"
sentinel1_path = "../Imagenes_Satelitales/SAR_2025_1.tif"
gt_path = "../gt_2.tif"

with rasterio.open(sentinel2_path) as src:
    img = src.read()
    profile = src.profile

with rasterio.open(sentinel1_path) as src:
    sar = src.read()

with rasterio.open(gt_path) as src:
    gt = src.read(1)

print("Shape Sentinel:", img.shape)
print("Shape SAR:", sar.shape)
print("Shape GroundTruth:", gt.shape)

# =========================
# 2. PREPARAR FEATURES
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

# =========================
# 3. SPLIT ESPACIAL POR BLOQUES
# =========================


block_size = 256

train_mask = np.zeros((rows, cols), dtype=bool)
test_mask = np.zeros((rows, cols), dtype=bool)


for i in range(0, rows, block_size):
    for j in range(0, cols, block_size):

        block_row = i // block_size
        block_col = j // block_size

        if (block_row + block_col) % 2 == 0:
            train_mask[i:i+block_size, j:j+block_size] = True
        else:
            test_mask[i:i+block_size, j:j+block_size] = True

# =========================
# 4. PREPARAR TRAIN Y TEST
# =========================

X_full = stack.reshape(n_features, rows * cols).T
y_full = gt.reshape(rows * cols)

train_mask_flat = train_mask.reshape(rows * cols)
test_mask_flat = test_mask.reshape(rows * cols)

train_valid = (y_full != -1) & train_mask_flat

X_train = X_full[train_valid]
y_train = y_full[train_valid]

test_valid = (y_full != -1) & test_mask_flat

X_test = X_full[test_valid]
y_test = y_full[test_valid]

print("Train:", X_train.shape)
print("Test:", X_test.shape)

# =========================
# 5. ENTRENAR MODELO
# =========================

model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# =========================================
# 6. EVALUAR MODELO y PREDECIR TODO EL MAPA
# =========================================

y_pred_test = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred_test))

print("\nReporte de clasificación:\n")
print(classification_report(y_test, y_pred_test))

print("\nPrediciendo mapa completo...")

y_pred_full = model.predict(X_full)

pred = y_pred_full.reshape(rows, cols)


# =========================
# 7. POSTPROCESAMIENTO
# =========================

print("\nAplicando filtro espacial...")

# -------------------------
# Majority Filter
# -------------------------

def majority_filter(values):
    values = values.astype(int)

   
    values = values[values != -1]

    if len(values) == 0:
        return -1

    return np.bincount(values).argmax()

filtered = generic_filter(
    pred,
    function=majority_filter,
    size=3
)

# -------------------------
# Eliminar regiones pequeñas
# -------------------------

min_pixels = 20

cleaned = filtered.copy()

classes = np.unique(filtered)

for cls in classes:

    if cls == -1:
        continue

    mask = filtered == cls

    labeled, num_features = label(mask)

    for region_id in range(1, num_features + 1):

        region = labeled == region_id

        if np.sum(region) < min_pixels:
            cleaned[region] = 0

# =========================
# 8. GUARDAR CLASIFICACIÓN
# =========================

profile.update(
    dtype=rasterio.int16,
    count=1,
    nodata=-1
)

output_path = "clasificacion_final_4.tif"

with rasterio.open(output_path, "w", **profile) as dst:
    dst.write(cleaned.astype(rasterio.int16), 1)

print("\nMapa guardado en:", output_path)

# =========================
# 9. GUARDAR MAPA DE SPLIT
# =========================

split_map = np.full((rows, cols), -1)

split_map[train_mask] = 1
split_map[test_mask] = 2

split_map[gt == -1] = -1

split_path = "split_blocks.tif"

with rasterio.open(split_path, "w", **profile) as dst:
    dst.write(split_map.astype(rasterio.int16), 1)

print("Mapa de split guardado en:", split_path)

# =========================
# 10. IMPORTANCIA FEATURES
# =========================

features = [
    "B2","B3","B4","B8","B11",
    "NDVI","NDBI","NDWI",
    "VV","VH","VV/VH"
]

importances = model.feature_importances_

for f, imp in zip(features, importances):
    print(f, imp)

# =========================
# 11. GUARDAR MODELO
# =========================
joblib.dump(model, "modelo_rf_2.pkl")