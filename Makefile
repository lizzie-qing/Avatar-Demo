.PHONY: setup run test clean

setup:
	python -m pip install -U pip
	python -m pip install -r requirements.txt

run:
	python -m app run --input data --out outputs --config configs/default.yaml

test:
	pytest -q

clean:
	rm -rf outputs
