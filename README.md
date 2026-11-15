# Лабораторные работы по машинному обучению

Жданович Елизавета Таймуразовна, группа М8О-407Б-23.

Пять лабораторных на одних и тех же данных: от KNN до градиентного бустинга. В каждой работе есть бейзлайн из sklearn, улучшенный вариант и своя реализация алгоритма. В конце отдельный ноутбук, где все модели стоят рядом на одних графиках.

## Датасеты

Классификация: Wine recognition data, файл `data/wine.data`. Химический анализ вин из трёх сортов винограда, выращенных в одном регионе Италии. 178 образцов, 13 признаков, 3 класса.

Регрессия: Parkinsons Telemonitoring, файл `data/parkinsons_updrs.data`. Записи голоса пациентов с болезнью Паркинсона, цель предсказать `total_UPDRS`. 5875 измерений, 42 пациента.

## Что внутри

- `notebooks/lab1_knn.ipynb` — метод ближайших соседей
- `notebooks/lab2_linear_logistic.ipynb` — линейная и логистическая регрессия
- `notebooks/lab3_decision_tree.ipynb` — решающее дерево
- `notebooks/lab4_random_forest.ipynb` — случайный лес
- `notebooks/lab5_gradient_boosting.ipynb` — градиентный бустинг
- `notebooks/final_comparison.ipynb` — сравнение всех алгоритмов
- `src/` — собственные реализации
- `data/` — оба датасета

## Как запустить локально

```bash
git clone https://github.com/lizabeth2616/Artificial-Intelligence-Labs.git
cd Artificial-Intelligence-Labs
pip install -r requirements.txt
jupyter notebook
```

Откроется браузер. Зайди в папку `notebooks` и открой нужный файл. В ноутбуке нажми Run, затем Run All Cells. Графики и результаты появятся прямо в ячейках.

Если команда `jupyter notebook` не находится:

```bash
python -m jupyter notebook
```
