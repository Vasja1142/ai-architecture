import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 1. Загрузка и очистка данных
penguins = sns.load_dataset('penguins')
penguins_clean = penguins.dropna()

# 2. Подготовка данных
# Целевая переменная
y = penguins_clean['species']

# Матрица признаков: непрерывные переменные + бинарный пол
X = penguins_clean[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']].copy()
X['sex_Male'] = (penguins_clean['sex'] == 'Male').astype(int)

# 3. Обучение модели
print("="*80)
print("КЛАССИЧЕСКАЯ МУЛЬТИНОМИАЛЬНАЯ ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ (с L2-регуляризацией)")
print("="*80)

# solver='lbfgs' и multi_class='multinomial' — это стандарт для многоклассовой логистической регрессии
# C=1.0 — это сила регулязации (по умолчанию). Она предотвращает "взрыв" коэффициентов.
model = LogisticRegression( solver='lbfgs', max_iter=1000, random_state=42)
model.fit(X, y)

# 4. Вывод коэффициентов
print("\nКОЭФФИЦИЕНТЫ МОДЕЛИ:")
print("-" * 80)
# Создаем красивую таблицу коэффициентов
coef_df = pd.DataFrame({
    'Признак': X.columns,
    'vs Adelie (Chinstrap)': model.coef_[1], # Индекс 1 обычно Chinstrap (зависит от алфавита)
    'vs Adelie (Gentoo)': model.coef_[2]     # Индекс 2 обычно Gentoo
})
# Привяжем индексы к реальным именам классов для надежности
classes = model.classes_
coef_df = pd.DataFrame(model.coef_, columns=X.columns, index=classes)
coef_df['Intercept'] = model.intercept_
print(coef_df.T.to_string())
print("-" * 80)

# 5. Оценка качества предсказаний
print("\n" + "="*80)
print("ОЦЕНКА КАЧЕСТВА КЛАССИФИКАЦИИ")
print("="*80)

# Предсказываем класс
penguins_clean['predicted_species'] = model.predict(X)

# Матрица ошибок
cm = confusion_matrix(penguins_clean['species'], penguins_clean['predicted_species'],
                      labels=model.classes_)

fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues', ax=ax, colorbar=False)
plt.title('Матрица ошибок (Логистическая регрессия с регуляризацией)')
plt.xlabel('Предсказанный вид')
plt.ylabel('Истинный вид')
plt.tight_layout()
plt.show()

# Текстовый отчет
print("\nОтчет о классификации:")
print(classification_report(penguins_clean['species'], penguins_clean['predicted_species']))

# Точность модели
accuracy = (penguins_clean['species'] == penguins_clean['predicted_species']).mean()
print(f"Общая точность (Accuracy): {accuracy:.2%}")