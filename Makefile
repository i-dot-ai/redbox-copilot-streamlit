.PHONY: app reqs up down clean lint format safe checktypes precommit

-include .env
export


reqs:
	poetry install

up:
	docker compose up -d elasticsearch minio streamlit-app --build

down:
	docker compose down

clean:
	docker compose down -v --rmi all --remove-orphans

lint:
	poetry run ruff check .

format:
	poetry run ruff format .
	poetry run ruff format **/*.ipynb

safe:
	poetry run bandit -ll -r ./redbox
	poetry run bandit -ll -r ./streamlit_app
	poetry run mypy ./redbox --ignore-missing-imports
	poetry run mypy ./streamlit_app --ignore-missing-imports

checktypes:
	poetry run mypy redbox streamlit_app --ignore-missing-imports

precommit:
	pre-commit run -a
