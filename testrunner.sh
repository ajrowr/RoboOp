#!/usr/bin/env sh
uv run --with 'anthropic' --with 'pytest' --with 'pytest-cov' -- pytest -v --cov=robo robo/unittests/__init__.py
