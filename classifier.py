import rasterio
import numpy as np
import joblib

model = joblib.load("modelo_rf_2.pkl")

sentinel2_path = "../Imagenes_Satelitales/Sentinel2Apopa_2018.tif"
sentinel1_path = "../Imagenes_Satelitales/SAR_2018_a.tif"

with rasterio.open(sentinel2_path) as src:
    img = src.read()
    profile = src.profile

with rasterio.open(sentinel1_path) as src:
    sar = src.read()

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


print("Prediciendo nueva área...")

y_pred_full = model.predict(X_full)

pred = y_pred_full.reshape(rows, cols)


profile.update(
    dtype=rasterio.int16,
    count=1,
    nodata=-1
)

output_path = "../Results/clasificacion_2018_ap.tif"

with rasterio.open(output_path, "w", **profile) as dst:
    dst.write(pred.astype(rasterio.int16), 1)

print("Mapa guardado:", output_path)