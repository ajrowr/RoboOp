#!/usr/bin/env sh
uv run --with 'anthropic' --with 'pytest' --with 'pytest-cov' --with 'pytest-asyncio' -- pytest -v --cov=robo robo/testing/unittests/*
uv run --with 'coverage' -- coverage report -m robo/*.py robo/tools/*.py