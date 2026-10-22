import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings('ignore')

# 1. Загрузка и подготовка данных
penguins = sns.load_dataset('penguins')
penguins_clean = penguins.dropna()

# 2. Формируем формулу из признаков, отобранных Lasso
# (Берем список из вашего вывода и возвращаем его к исходным именам признаков)
lasso_selected_features = [
    'flipper_length_mm',
    'bill_depth_mm * flipper_length_mm',
    'bill_length_mm * bill_depth_mm',
    'flipper_length_mm**2',
    'flipper_length_mm**3',
    'bill_length_mm**2',
    'flipper_length_mm**2 * bill_depth_mm',
    'bill_depth_mm**2 * flipper_length_mm',
    'bill_length_mm**3',
    'bill_length_mm * flipper_length_mm',
    'bill_depth_mm**3',
    'bill_length_mm**2 * bill_depth_mm',
    'bill_depth_mm**2 * bill_length_mm'
]

# Оборачиваем сложные члены в I(...) для statsmodels
formula_parts = []
for feat in lasso_selected_features:
    # Если есть операция (умножение или степень), оборачиваем в I()
    if any(op in feat for op in ['**', '*']):
        formula_parts.append(f"I({feat})")
    else:
        formula_parts.append(feat)

# Добавляем обязательные категориальные признаки
formula_parts.extend(['C(species)', 'C(sex)'])

# Итоговая формула
formula = "body_mass_g ~ " + " + ".join(formula_parts)

print("="*90)
print("ФИНАЛЬНАЯ СТАТИСТИЧЕСКАЯ МОДЕЛЬ НА ОСНОВЕ ПРИЗНАКОВ LASSO")
print("="*90)
print(f"Формула:\n{formula}\n")

# 3. Обучение модели OLS
model = smf.ols(formula=formula, data=penguins_clean)
results = model.fit()

# 4. Полный статистический отчет
print(results.summary())

# 5. Ключевые выводы
print("\n" + "="*90)
print("КЛЮЧЕВЫЕ ВЫВОДЫ")
print("="*90)
print(f"R-squared:         {results.rsquared:.4f} ({results.rsquared*100:.2f}% объясненной дисперсии)")
print(f"Adj R-squared:     {results.rsquared_adj:.4f}")
print(f"BIC:               {results.bic:.2f}")
print(f"Condition Number:  {results.condition_number:.1f}")

# Считаем количество значимых признаков (p < 0.05)
p_values = results.pvalues.drop('Intercept', errors='ignore')
significant = (p_values < 0.05).sum()
total = len(p_values)
print(f"\nЗначимых параметров (p < 0.05): {significant} из {total}")

# 6. Список значимых и незначимых признаков
print("\nЗНАЧИМЫЕ ПРИЗНАКИ:")
print("-" * 60)
sig_df = pd.DataFrame({'Признак': p_values.index, 'p-value': p_values.values})
sig_df = sig_df.sort_values('p-value')
print(sig_df[sig_df['p-value'] < 0.05].to_string(index=False))

print("\nНЕЗНАЧИМЫЕ ПРИЗНАКИ (кандидаты на удаление):")
print("-" * 60)
print(sig_df[sig_df['p-value'] >= 0.05].to_string(index=False))