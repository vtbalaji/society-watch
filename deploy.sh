#!/bin/sh
# Commit every change in this project and push it to GitHub (origin/main).
#   ./deploy.sh                 # commit message defaults to a timestamp
#   ./deploy.sh "Tune dog rule" # your own commit message
# Videos (samples/), results (output/) and the virtualenv are never pushed;
# .gitignore keeps them out.
set -e
cd "$(dirname "$0")"

msg="${1:-Update $(date '+%Y-%m-%d %H:%M')}"
branch="$(git branch --show-current)"

git add -A
if git diff --cached --quiet; then
  echo "Nothing new to commit."
else
  git commit -m "$msg"
fi

# Retry on network hiccups.
for wait in 0 2 4 8 16; do
  sleep "$wait"
  if git push -u origin "$branch"; then
    echo "Pushed $branch to $(git remote get-url origin)"
    exit 0
  fi
  echo "Push failed, retrying..."
done
echo "Push failed after 5 attempts." >&2
exit 1
