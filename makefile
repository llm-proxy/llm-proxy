.PHONY: test format all

all: format test

test:
	PYTHONPATH=. pytest
format:
	poetry run isort .
	poetry run black .
