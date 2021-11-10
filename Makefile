TEST_PATH=./tests
MODULES_PATH=./noise2same
TRAIN_PATH=train.py evaluate.py

.PHONY: format lint test commit

format:
	isort $(MODULES_PATH) $(TRAIN_PATH) $(TEST_PATH)
	black $(MODULES_PATH) $(TRAIN_PATH) $(TEST_PATH)

lint:
	isort -c $(MODULES_PATH) $(TRAIN_PATH) $(TEST_PATH)
	black --check $(MODULES_PATH) $(TRAIN_PATH) $(TEST_PATH)
	mypy $(MODULES_PATH) $(TRAIN_PATH) $(TEST_PATH)

test:
	python3 -m unittest discover -s $(TEST_PATH) -t $(TEST_PATH)
