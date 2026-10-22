import seaborn as sns
import pandas as pd

# Загружаем и чистим данные (как мы делали ранее)
penguins = sns.load_dataset('penguins')
penguins_clean = penguins.dropna().reset_index(drop=True)

# Сохраняем в CSV (кодировка utf-8-sig нужна для корректного отображения кириллицы в Excel/DataLens)
penguins_clean.to_csv('penguins_clean.csv', index=False, encoding='utf-8-sig')
print("✅ Файл 'penguins_clean.csv' успешно сохранен в папке проекта!")