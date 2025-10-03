#!/bin/bash
set -e

echo "🚀 Running all development checks..."
echo "=================================="

# Format code first
./scripts/format.sh

echo ""
echo "=================================="

# Run linting and type checks
./scripts/lint.sh

echo ""
echo "=================================="

# Run tests
echo "🧪 Running tests..."
uv run --quiet pytest backend/tests/ -v 2>/dev/null

echo ""
echo "✨ All checks passed successfully!"