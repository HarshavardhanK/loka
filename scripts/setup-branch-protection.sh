#!/usr/bin/env bash
# =============================================================================
# Setup GitHub branch-protection rules for the main branch.
#
# Prerequisites:
#   - GitHub CLI (`gh`) installed and authenticated
#   - Repo admin permissions
#
# Usage:
#   ./scripts/setup-branch-protection.sh              # uses current repo
#   REPO=HarshavardhanK/loka ./scripts/setup-branch-protection.sh
#
# What this configures:
#   ✓ Require PRs (no direct pushes to main)
#   ✓ Require at least 1 approving review
#   ✓ Dismiss stale reviews when new commits are pushed
#   ✓ Require status checks to pass before merging:
#       - Quality Gate  (fan-in of lint + unit-tests + integration-tests)
#   ✓ Require branches to be up-to-date before merging
#   ✓ Block force-pushes to main
#   ✓ Block deletions of main
# =============================================================================
set -euo pipefail

REPO="${REPO:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}"
BRANCH="main"

echo "🔒 Configuring branch protection for ${REPO} → ${BRANCH}"
echo ""

# ---------------------------------------------------------------------------
# Use the GitHub REST API via gh to set branch protection.
# Docs: https://docs.github.com/en/rest/branches/branch-protection
# ---------------------------------------------------------------------------
gh api \
  --method PUT \
  "repos/${REPO}/branches/${BRANCH}/protection" \
  --input - <<'EOF'
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "Quality Gate"
    ]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false,
    "required_approving_review_count": 1
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_linear_history": false,
  "required_conversation_resolution": false
}
EOF

echo ""
echo "✅ Branch protection configured successfully!"
echo ""
echo "Protected branch: ${BRANCH}"
echo "Required checks:  Quality Gate (lint + unit-tests + integration-tests)"
echo "Required reviews:  1 approving review"
echo "Stale reviews:     dismissed on new pushes"
echo "Force push:        blocked"
echo "Branch deletion:   blocked"
echo "Up-to-date:        required before merging"
