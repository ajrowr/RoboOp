#!/usr/bin/env sh
uv run --with 'anthropic' --with 'pytest' -- pytest -v robo/unittests/__init__.py
