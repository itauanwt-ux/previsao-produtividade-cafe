import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("produtividade_cafe.csv")

X = df.drop(columns=["produtividade_sacas_ha"])
y = df["produtividade_sacas_ha"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo baseline
lr = LinearRegression().fit(X_train, y_train)
pred_lr = lr.predict(X_test)

# Modelo principal
rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

print("== Regressão Linear (baseline) ==")
print(f"MAE: {mean_absolute_error(y_test, pred_lr):.2f} sacas/ha")
print(f"R²:  {r2_score(y_test, pred_lr):.3f}")

print("\n== Random Forest ==")
print(f"MAE: {mean_absolute_error(y_test, pred_rf):.2f} sacas/ha")
print(f"R²:  {r2_score(y_test, pred_rf):.3f}")

# Importância das variáveis
importancias = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nImportância das variáveis:")
print(importancias.round(3))

plt.figure(figsize=(7,5))
importancias.plot(kind="barh", color="#3b6e3b")
plt.gca().invert_yaxis()
plt.title("Importância das variáveis na produtividade do café")
plt.xlabel("Importância (Random Forest)")
plt.tight_layout()
plt.savefig("importancia_variaveis.png", dpi=120)
plt.close()

plt.figure(figsize=(6,6))
plt.scatter(y_test, pred_rf, alpha=0.6, color="#2c5f8a")
lims = [y.min(), y.max()]
plt.plot(lims, lims, "r--", label="Previsão perfeita")
plt.xlabel("Produtividade real (sacas/ha)")
plt.ylabel("Produtividade prevista (sacas/ha)")
plt.title("Real vs. Previsto - Random Forest")
plt.legend()
plt.tight_layout()
plt.savefig("real_vs_previsto.png", dpi=120)
plt.close()
