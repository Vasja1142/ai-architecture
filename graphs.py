import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 1. Подготовка данных и финальной модели
penguins = sns.load_dataset('penguins')
penguins_clean = penguins.dropna()

final_formula = "body_mass_g ~ bill_length_mm + bill_depth_mm + I(flipper_length_mm**2) + bill_length_mm:bill_depth_mm + C(species) + C(sex)"
model_final = smf.ols(formula=final_formula, data=penguins_clean)
results_final = model_final.fit()

# Добавляем предсказания и остатки в датафрейм для удобства
penguins_clean['fitted'] = results_final.fittedvalues
penguins_clean['residuals'] = results_final.resid

# 2. Настройка стиля графиков
sns.set_theme(style="whitegrid", font_scale=1.1)
fig = plt.figure(figsize=(16, 12))

# ==========================================
# График 1: Остатки vs Предсказанные значения
# ==========================================
ax1 = plt.subplot(2, 2, 1)
sns.scatterplot(x='fitted', y='residuals', data=penguins_clean, alpha=0.6, ax=ax1, color='steelblue')
ax1.axhline(0, color='red', linestyle='--', linewidth=2)
ax1.set_title('1. Остатки vs Предсказанные значения\n(Проверка на гетероскедастичность)')
ax1.set_xlabel('Предсказанная масса тела (г)')
ax1.set_ylabel('Остатки (Факт - Предсказание)')

# ==========================================
# График 2: Q-Q Plot остатков
# ==========================================
ax2 = plt.subplot(2, 2, 2)
sm.qqplot(results_final.resid, line='s', ax=ax2, fit=True, color='steelblue')
# Делаем линию красной для наглядности
ax2.lines[1].set_color('red')
ax2.lines[1].set_linestyle('--')
ax2.set_title('2. Q-Q Plot остатков\n(Проверка на нормальность распределения)')
ax2.set_xlabel('Теоретические квантили (Нормальное распр.)')
ax2.set_ylabel('Выборочные квантили (Наши остатки)')

# ==========================================
# График 3: Фактические vs Предсказанные значения
# ==========================================
ax3 = plt.subplot(2, 2, 3)
sns.scatterplot(x='fitted', y='body_mass_g', data=penguins_clean, hue='species', alpha=0.7, ax=ax3)
# Линия идеального предсказания (y = x)
min_val = min(penguins_clean['fitted'].min(), penguins_clean['body_mass_g'].min())
max_val = max(penguins_clean['fitted'].max(), penguins_clean['body_mass_g'].max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Идеальное совпадение')
ax3.set_title('3. Фактические vs Предсказанные значения')
ax3.set_xlabel('Предсказанная масса (г)')
ax3.set_ylabel('Фактическая масса (г)')
ax3.legend(title='Вид пингвина')

# ==========================================
# График 4: Визуализация нелинейного эффекта (Квадрат ласта)
# ==========================================
ax4 = plt.subplot(2, 2, 4)

# Создаем синтетические данные для построения гладкой кривой предсказания
flipper_range = np.linspace(penguins_clean['flipper_length_mm'].min(), penguins_clean['flipper_length_mm'].max(), 100)

# Фиксируем остальные переменные на средних/базовых значениях
mean_bill_len = penguins_clean['bill_length_mm'].mean()
mean_bill_dep = penguins_clean['bill_depth_mm'].mean()

pred_df = pd.DataFrame({
    'bill_length_mm': mean_bill_len,
    'bill_depth_mm': mean_bill_dep,
    'flipper_length_mm': flipper_range,
    'species': 'Adelie',      # Базовый вид
    'sex': 'Female'           # Базовый пол
})

# Получаем предсказания модели для этого синтетического ряда
pred_df['predicted_mass'] = results_final.predict(pred_df)

# Рисуем: 1) все реальные точки (серым фоном), 2) кривую предсказания модели (красным)
sns.scatterplot(x='flipper_length_mm', y='body_mass_g', data=penguins_clean, color='gray', alpha=0.3, ax=ax4, label='Все реальные данные')
sns.lineplot(x='flipper_length_mm', y='predicted_mass', data=pred_df, color='red', linewidth=3, ax=ax4, label='Предсказание модели\n(при средних прочих параметрах)')

ax4.set_title('4. Нелинейный эффект длины ласта\n(Квадратичная зависимость)')
ax4.set_xlabel('Длина ласта (мм)')
ax4.set_ylabel('Масса тела (г)')
ax4.legend()

# Финальная компоновка
plt.tight_layout()
plt.show()