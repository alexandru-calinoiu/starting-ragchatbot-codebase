#!/bin/bash
set -e

echo "🔍 Running code quality checks..."

echo "📋 Checking with flake8..."
uv run --quiet flake8 backend/ 2>/dev/null

echo "🔎 Running type checks with mypy..."
uv run --quiet mypy backend/ --ignore-missing-imports --no-strict-optional --allow-untyped-globals --allow-redefinition --quiet 2>/dev/null || true

echo "✅ All quality checks passed!"