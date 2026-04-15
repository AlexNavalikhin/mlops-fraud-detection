# Документация MLOps-конвейера обнаружения мошеннических транзакций

## 1. Постановка задачи

Бинарная классификация банковских транзакций. Целевая переменная: `Is_Fraud` (0 - легитимная, 1 - мошенническая). Основная метрика: F1-score.

Источник данных: Kaggle [Bank Transaction Fraud Detection](https://www.kaggle.com/datasets/marusagar/bank-transaction-fraud-detection/data). Формат: CSV, 24 признака.

## 2. Обработка данных

**Сбор данных**: Источник - `data/raw_source/bank_transactions.csv`. Размер батча: 2000 строк (настраивается в config.yaml).

**Контроль качества**:
- Пропуски
- Дубликаты

**Очистка**: удаление дубликатов, колонок с пропусками >30%, строк с отрицательными суммами и некорректным возрастом. Заполнение пропусков: числовые - медианой, категориальные - модой.

**Ассоциативные правила**: Apriori (min_support=0.05, min_confidence=0.3, min_lift=1.0). Результаты в `reports/apriori/`.

**Data Drift**: сравнение текущего батча с первым. Порог отклонения: 0.2. Отчёты в `reports/drift/`.

**EDA**: статистики, распределения, корреляционная матрица, fraud rate по категориям. Результаты в `reports/eda/`.

## 3. Предобработка

**Удаляемые признаки**: Customer_ID, Transaction_ID, Merchant_ID, Transaction_Location, Transaction_Time, Transaction_Currency.

**Числовые признаки** (Transaction_Amount, Account_Balance, Age): импутация медианой, масштабирование (StandardScaler или MinMaxScaler - настраивается).

**Категориальные признаки** (Transaction_Type, Merchant_Category, Device_Type, Account_Type, State, Gender):кодирование (OrdinalEncoder или OneHotEncoder).

Препроцессор обучается на первом батче и сохраняется в `models/preprocessor/`.

## 4. Модели

**Random Forest**:
- n_estimators: число деревьев
- max_depth: максимальная глубина
- new_trees: новые деревья на батч

**MLP**:
- hidden_layer_sizes: слои

## 5. Валидация

**Метод**: TimeSeriesCrossValidation
**Метрики**: F1-score, ROC-AUC, Precision, Recall.

## 6. Сохранение артефактов

| Артефакт | Путь |
|----------|------|
| Сырые батчи | `data/raw/batch_XXXX/` |
| Препроцессор | `models/preprocessor/` |
| Версии моделей | `models/trainer/` |
| Лучшая модель | `models/validator/best_model.pkl` |
| Продуктовая модель | `models/serving/production_model.pkl` |
| Отчёты качества | `reports/quality/` |
| Правила Apriori | `reports/apriori/` |
| EDA | `reports/eda/` |
| Drift | `reports/drift/` |
| Логи | `logs/` |

## Команды для запуска
```bash
# Обучить на одном батче
python run.py -mode "update"

# Обучить на 3 батчах
python run.py -mode "update" -batches 3

# Обучить на всех батчах сразу
python run.py -mode "update" -all

# Сбросить и начать заново
python run.py -mode "reset"

# Применить модель к новым данным
python run.py -mode "inference" -file "./path_to_file.csv"

# Отчёт о работе системы
python run.py -mode "summary"
```

## Тесты
```bash
python run_tests.py
```
