.PHONY: test format all

all: format test

test:
	poetry run pytest proxyllm/tests
format:
	poetry run isort .
	poetry run black .

coverage_report:
	poetry run coverage run -m pytest proxyllm/tests
	poetry run coverage report -i

coverage_html:
	poetry run coverage run -m pytest proxyllm/tests
	poetry run coverage html -i
	open htmlcov/index.html
