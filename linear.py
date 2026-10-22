import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pandas as pd

# 1. Загрузка датасета
penguins = sns.load_dataset('penguins')

# 2. Предобработка: удаляем пропуски (в датасете есть NaN)
penguins_clean = penguins.dropna()

# 3. Построение модели классической линейной регрессии (OLS)
# Используем формулу: целевая_переменная ~ признак1 + признак2 + ...
# C() используется для явного указания, что переменная категориальная
formula = 'body_mass_g ~ bill_length_mm + bill_depth_mm + flipper_length_mm + C(species) + C(island) + C(sex)'

model = smf.ols(formula=formula, data=penguins_clean)

# 4. Обучение (подгонка) модели
results = model.fit()

# 5. Вывод статистической сводки модели
print("="*60)
print("РЕЗУЛЬТАТЫ КЛАССИЧЕСКОЙ ЛИНЕЙНОЙ РЕГРЕССИИ (OLS)")
print("="*60)
print(results.summary())

# 6. Визуализация: Фактические значения против предсказанных
plt.figure(figsize=(8, 6))
plt.scatter(results.fittedvalues, results.model.endog, alpha=0.6, edgecolors='k')

# Идеальная линия (если бы предсказания были 100% точными)
min_val = min(results.model.endog.min(), results.fittedvalues.min())
max_val = max(results.model.endog.max(), results.fittedvalues.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', color='red', label='Идеальное совпадение')

plt.xlabel('Предсказанная масса тела (г)')
plt.ylabel('Фактическая масса тела (г)')
plt.title('Фактические vs Предсказанные значения (Penguins OLS)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()