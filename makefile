.PHONY: test format all clean_imports

all: clean_imports format test

test:
	poetry run pytest tests

format:
	poetry run isort .
	poetry run black .

clean_imports:
	poetry run autoflake --remove-all-unused-imports --recursive --in-place --exclude=__init__.py .

coverage_report:
	poetry run coverage run -m pytest tests
	poetry run coverage report -i

coverage_html:
	poetry run coverage run -m pytest tests
	poetry run coverage html -i
	open htmlcov/index.html
