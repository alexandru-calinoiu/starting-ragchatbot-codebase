#!/bin/bash
set -e

echo "🎨 Running code formatters..."

echo "📦 Sorting imports with isort..."
uv run --quiet isort backend/ 2>/dev/null

echo "✨ Formatting code with black..."
uv run --quiet black backend/ 2>/dev/null

echo "✅ Code formatting complete!"