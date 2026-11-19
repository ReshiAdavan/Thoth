#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
python3 -m thoth compare --input "${1:-data/taylorSwift.txt}"
