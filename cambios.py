import rasterio
import numpy as np

map_2017 = "../Results/clasificacion_2018_ap_split.tif"
map_2025 = "../Results/clasificacion_2025_ap_split.tif"

with rasterio.open(map_2017) as src:
    lulc_2017 = src.read(1)
    profile = src.profile

with rasterio.open(map_2025) as src:
    lulc_2025 = src.read(1)



urban_expansion = np.where(
    (lulc_2017 == 0) &
    (lulc_2025 == 1),
    1,
    0
)


profile.update(
    dtype=rasterio.uint8,
    count=1,
    nodata=0
)

output = "urban_expansion_0_1_ap.tif"

with rasterio.open(output,"w",**profile) as dst:
    dst.write(urban_expansion.astype(rasterio.uint8),1)


# métricas
pixels = np.sum(urban_expansion==1)

pixel_area=100

area_ha=(pixels*pixel_area)/10000

print("Pixeles expansión urbana:",pixels)
print("Área expansión urbana (ha):",area_ha)

print("Mapa generado:",output)