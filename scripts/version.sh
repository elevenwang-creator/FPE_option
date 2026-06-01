#!/usr/bin/env bash
# Extract version from git tag, fallback to 0.0.0-g<commit>
set -euo pipefail

git_describe=$(git describe --tags --always --dirty 2>/dev/null || echo "0.0.0")
version="${git_describe#v}"
echo "$version"
