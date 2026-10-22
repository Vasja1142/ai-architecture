import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
import re
import warnings

warnings.filterwarnings('ignore')

# 1. Загрузка и очистка данных
penguins = sns.load_dataset('penguins')
penguins_clean = penguins.dropna()

cont_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']
cat_features = ['species', 'sex']

# 2. Формируем ПОЛНУЮ формулу
formula_parts = []
formula_parts.extend(cont_features)

for col in cont_features:
    formula_parts.append(f"I({col}**2)")

for i in range(len(cont_features)):
    for j in range(i + 1, len(cont_features)):
        formula_parts.append(f"{cont_features[i]}:{cont_features[j]}")

formula_parts.extend([f"C({c})" for c in cat_features])

current_formula = "body_mass_g ~ " + " + ".join(formula_parts)
threshold = 0.05
max_iterations = 50

print("=" * 90)
print("BACKWARD ELIMINATION (ИСПРАВЛЕННАЯ ВЕРСИЯ)")
print(f"Начальная формула:\n{current_formula}")
print("=" * 90)

for iteration in range(1, max_iterations + 1):
    model = smf.ols(formula=current_formula, data=penguins_clean)
    results = model.fit()

    p_values = results.pvalues.drop('Intercept', errors='ignore')
    max_p_value = p_values.max()
    max_p_feature = p_values.idxmax()

    print(f"Итерация {iteration:2d}: Макс. p-value = {max_p_value:.4f} у '{max_p_feature}'")

    if max_p_value <= threshold:
        print(f"\n✅ ОСТАНОВКА: Все оставшиеся параметры значимы (p-value <= {threshold})")
        break

    # --- ИСПРАВЛЕННАЯ ЛОГИКА УДАЛЕНИЯ (с учетом пробелов) ---
    if "C(" in max_p_feature and "[" in max_p_feature:
        match = re.search(r'C\((.*?)\)', max_p_feature)
        term_to_remove = f"C({match.group(1)})" if match else max_p_feature
    else:
        term_to_remove = max_p_feature

    left_side, right_side = current_formula.split(' ~ ')
    terms = [t.strip() for t in right_side.split(' + ')]

    # Нормализуем строки для сравнения (удаляем все пробелы)
    normalized_terms = [t.replace(" ", "") for t in terms]
    normalized_target = term_to_remove.replace(" ", "")

    if normalized_target in normalized_terms:
        # Находим индекс и удаляем ОРИГИНАЛЬНЫЙ элемент (с его оригинальными пробелами)
        actual_index = normalized_terms.index(normalized_target)
        terms.pop(actual_index)
        current_formula = left_side + " ~ " + " + ".join(terms)
    else:
        print(f"\n⚠️ ОШИБКА: Не удалось найти и удалить '{term_to_remove}'. Прерывание.")
        break

# 3. Финальный результат
print("\n" + "=" * 90)
print("🏆 АБСОЛЮТНО ФИНАЛЬНАЯ МОДЕЛЬ ПОСЛЕ ПОЛНОГО ОТБОРА")
print("=" * 90)
print(f"Итоговая формула:\n{current_formula}\n")

model_final = smf.ols(formula=current_formula, data=penguins_clean)
results_final = model_final.fit()

print("СТАТИСТИКА КОЭФФИЦИЕНТОВ:")
print(results_final.summary().tables[1])

print("\n" + "=" * 90)
print("ОБЩИЕ МЕТРИКИ ФИНАЛЬНОЙ МОДЕЛИ")
print("=" * 90)
print(f"R-squared:       {results_final.rsquared:.4f}")
print(f"Adj R-squared:   {results_final.rsquared_adj:.4f}")
print(f"BIC:             {results_final.bic:.2f}")
print(f"Кол-во параметров: {len(results_final.params)}")

# Сравнение с самой первой базовой моделью
baseline_formula = "body_mass_g ~ bill_length_mm + bill_depth_mm + flipper_length_mm + C(species) + C(sex)"
results_baseline = smf.ols(formula=baseline_formula, data=penguins_clean).fit()

print("\n" + "=" * 90)
print("СРАВНЕНИЕ: Базовая линейная vs Финальная нелинейная")
print("=" * 90)
print(f"Базовая R²:   {results_baseline.rsquared:.4f} | BIC: {results_baseline.bic:.2f}")
print(f"Финальная R²: {results_final.rsquared:.4f} | BIC: {results_final.bic:.2f}")
print(f"Улучшение BIC: {results_baseline.bic - results_final.bic:.2f} пунктов (чем больше, тем лучше)")