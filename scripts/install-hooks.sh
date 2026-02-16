#!/usr/bin/env bash
# =============================================================================
# Install Loka git hooks
# =============================================================================
# Points git's core.hooksPath to the .githooks/ directory so that
# pre-commit (and any future hooks) are automatically active.
#
# Usage:
#   ./scripts/install-hooks.sh
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Setting git core.hooksPath → .githooks/ …"
git config core.hooksPath .githooks
chmod +x .githooks/*

echo "✓ Git hooks installed.  The pre-commit hook will now run before every commit."
echo "  To skip once:  SKIP_PRE_COMMIT=1 git commit …"
echo "  To uninstall:  git config --unset core.hooksPath"
