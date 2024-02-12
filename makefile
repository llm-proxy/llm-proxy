.PHONY: test format all

all: format test

test:
	poetry run pytest llmproxy/tests
format:
	poetry run isort .
	poetry run black .
