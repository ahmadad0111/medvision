#!/usr/bin/env bash
# Push EVERY branch from a bundle-clone to a GitHub repo.
# A bundle clone only creates ONE local branch; the rest are remote-tracking
# refs, so `git push --all` misses them. This pushes them all.
# Usage: bash scripts/push_all_branches.sh https://github.com/<you>/<repo>.git
set -e
REPO_URL="${1:?Usage: push_all_branches.sh <github-repo-url>}"
git remote remove gh 2>/dev/null || true
git remote add gh "$REPO_URL"
git remote set-head origin --delete 2>/dev/null || true
git push gh "refs/remotes/origin/*:refs/heads/*"
echo "Pushed all branches to $REPO_URL"
git ls-remote --heads gh | awk '{print "  "$2}'
