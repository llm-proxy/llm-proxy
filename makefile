.PHONY: test format all

all: format test

test:
	poetry run pytest
format:
	poetry run isort .
	poetry run black .
