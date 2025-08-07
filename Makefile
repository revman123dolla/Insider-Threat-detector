.PHONY: setup index run test

setup:
	pip install -r requirements.txt

index:
	python build_index.py

run:
	python app.py

test:
	pytest -q
