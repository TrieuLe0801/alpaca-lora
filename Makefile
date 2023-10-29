.DEFAULT_GOAL := build
.PHONY: build quality format clean

CHECK_DIRS := . utils
build:
	pip install -e .[all]

# Check that source code meets quality standards

quality:
	black --check $(CHECK_DIRS)
	isort --check-only $(CHECK_DIRS)
	flake8 $(CHECK_DIRS)

# Format source code automatically and check is there are any problems left that need manual fixing

format:
	black --fast $(CHECK_DIRS)
	isort $(CHECK_DIRS)

clean:
	rm -rf .pytest_cache/
	rm -rf dcc.egg-info/
	rm -rf dist/
	rm -rf build/
	find . | grep -E '(\.mypy_cache|__pycache__|\.pyc|\.pyo$$)' | xargs rm -rf
