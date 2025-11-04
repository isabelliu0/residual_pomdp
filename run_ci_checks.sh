#!/bin/bash
./run_autoformat.sh
mypy .
python -m pytest . --pylint -m pylint --pylint-rcfile=.pylintrc
python -m pytest tests/
