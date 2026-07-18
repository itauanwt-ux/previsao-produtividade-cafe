import numpy as np
import pandas as pd

np.random.seed(7)
n = 500

chuva_mm = np.random.normal(1200, 250, n).clip(400, 2200)
temp_media = np.random.normal(21, 2.5, n).clip(14, 30)
adubacao_kg_ha = np.random.normal(180, 50, n).clip(40, 350)
altitude_m = np.random.normal(900, 250, n).clip(300, 1600)
idade_lavoura = np.random.randint(1, 20, n)

produtividade = (
    10
    + 0.015 * chuva_mm
    - 0.35 * (temp_media - 21) ** 2
    + 0.04 * adubacao_kg_ha
    + 0.006 * altitude_m
    - 0.005 * (idade_lavoura - 8) ** 2
    + np.random.normal(0, 4, n)
).clip(5, 60)

df = pd.DataFrame({
    "chuva_mm_ano": chuva_mm.round(1),
    "temperatura_media_c": temp_media.round(1),
    "adubacao_kg_ha": adubacao_kg_ha.round(1),
    "altitude_m": altitude_m.round(0),
    "idade_lavoura_anos": idade_lavoura,
    "produtividade_sacas_ha": produtividade.round(1),
})
df.to_csv("produtividade_cafe.csv", index=False)
print(df.describe())
