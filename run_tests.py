import unittest
import sys
import os

MODULES = [
    "tests.collection_data.test_collection_data",
    "tests.analysis_data.test_analysis_data",
    "tests.preparation_data.test_preparation_data",
    "tests.model_training.test_model_training",
    "tests.model_validation.test_model_validation",
    "tests.model_serving.test_model_serving",
]

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    for module_name in MODULES:
        try:
            tests = loader.loadTestsFromName(module_name)
            suite.addTests(tests)
            print(f"Загружен: {module_name}")
        except Exception as e:
            print(f"Пропущен: {module_name}  ({e})")

    print()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
