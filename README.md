## Описание проекта

Данный проект реализует **MLOps-конвейер** для автоматического построения и обновления ML-моделей на потоковых табличных данных. Система эмулирует работу в продуктовой среде: получает данные батчами, анализирует их качество, обучает и дообучает модели, валидирует и обслуживает лучшую из них.

**Задача:** бинарная классификация мошеннических банковских транзакций (`Is_Fraud`).  
**Датасет:** [Bank Transaction Fraud Detection — LOL Bank Pvt. Ltd.](https://www.kaggle.com/datasets/marusagar/bank-transaction-fraud-detection/data)

---

## Команды
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

---

**Подробное описание работы:**  
[`how_it_works`](doc/how_it_works.md)

**Выполненные пункты:**  
[`grade`](doc/grade.md)

---