#!/bin/sh

current_branch=$(git rev-parse --abbrev-ref HEAD)
if [ $current_branch != "stable" ]; then
    echo "The current branch must be 'stable'."
    exit 1
fi

uv build
uv publish
