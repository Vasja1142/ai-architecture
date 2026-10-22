import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import warnings

warnings.filterwarnings('ignore')

# 1. Данные
penguins = sns.load_dataset('penguins')
penguins_clean = penguins.dropna()

y = penguins_clean['species']
X = penguins_clean[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']]

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Обучение
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)

# 3. Извлекаем структуру
tree = tree_model.tree_
feature_names = X.columns
class_names = tree_model.classes_

# 4. Рисуем дерево СЛЕВА НАПРАВО
fig, ax = plt.subplots(figsize=(22, 14))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')


def draw_tree_horizontal(node_id, x, y, y_span):
    n_samples = tree.n_node_samples[node_id]

    # Берём значения (это могут быть доли или целые числа)
    raw_values = tree.value[node_id, 0]

    # ✅ ИСПРАВЛЕНИЕ: Если это доли (сумма ~1.0), умножаем на n_samples и округляем
    if np.isclose(raw_values.sum(), 1.0):
        counts = np.round(raw_values * n_samples).astype(int)
    else:
        counts = raw_values.astype(int)

    class_idx = int(counts.argmax())
    class_name = class_names[class_idx]

    # Цвет блока
    if tree.children_left[node_id] != -1:
        face_color = '#AED6F1'
        edge_color = '#2874A6'
    else:
        colors_map = {'Adelie': '#F5CBA7', 'Chinstrap': '#ABEBC6', 'Gentoo': '#D7BDE2'}
        face_color = colors_map.get(class_name, '#F9E79F')
        edge_color = '#B9770E'

    # Блок
    box_width = 16
    box_height = min(10, y_span * 0.8)
    rect = patches.FancyBboxPatch(
        (x - box_width / 2, y - box_height / 2), box_width, box_height,
        boxstyle="round,pad=0.3", facecolor=face_color, edgecolor=edge_color, linewidth=2
    )
    ax.add_patch(rect)

    # Текст: условие + распределение классов (теперь точно не нули!)
    if tree.children_left[node_id] != -1:
        feature_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        text = (f"{feature_names[feature_idx]} ≤ {threshold:.1f}\n"
                f"[A:{counts[0]}, C:{counts[1]}, G:{counts[2]}]")
    else:
        text = (f"{class_name}\n"
                f"[A:{counts[0]}, C:{counts[1]}, G:{counts[2]}]")

    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

    # Стрелки и дети
    if tree.children_left[node_id] != -1:
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        dx = 20
        next_x = x + dx
        left_y = y + y_span / 2
        right_y = y - y_span / 2

        # True (вверх)
        arrow_left = FancyArrowPatch(
            (x + box_width / 2, y), (next_x - box_width / 2, left_y),
            arrowstyle='->', color='green', mutation_scale=20, linewidth=2.5
        )
        ax.add_patch(arrow_left)
        ax.text(x + dx / 2, y + y_span / 4, 'True', fontsize=10, color='green', fontweight='bold')

        # False (вниз)
        arrow_right = FancyArrowPatch(
            (x + box_width / 2, y), (next_x - box_width / 2, right_y),
            arrowstyle='->', color='red', mutation_scale=20, linewidth=2.5
        )
        ax.add_patch(arrow_right)
        ax.text(x + dx / 2, y - y_span / 4, 'False', fontsize=10, color='red', fontweight='bold')

        draw_tree_horizontal(left_child, next_x, left_y, y_span / 2)
        draw_tree_horizontal(right_child, next_x, right_y, y_span / 2)


draw_tree_horizontal(0, 12, 50, 40)
plt.title('Дерево решений (СЛЕВА → НАПРАВО)\n[A=Adelie, C=Chinstrap, G=Gentoo]',
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 5. Матрица ошибок
y_test_pred = tree_model.predict(X_test)
cm = confusion_matrix(y_test, y_test_pred, labels=['Adelie', 'Chinstrap', 'Gentoo'])

fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Adelie', 'Chinstrap', 'Gentoo'])
disp.plot(cmap='Blues', ax=ax, colorbar=False)
plt.title('Матрица ошибок (Тестовая выборка)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()