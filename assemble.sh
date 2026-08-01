#!/bin/zsh

# Check all the code

setopt err_exit null_glob pipefail

ROOT=$0:A:h
cd $ROOT

eval "$(mamba shell hook --shell zsh)"
mamba activate standard

# Format
ruff format --line-length 80 sigma tests | \
	grep -vE "^([0-9]+ files? reformatted, )?[0-9]+ files? left unchanged$" \
	|| true

# Pending work markers
find . -name "*.sh" -o -name "*.py" -o -name "*.txt" -o -name "*.toml" | \
	grep -vE "/\.fork/|/\.ruff_cache/|/\.mypy_cache/|/\.egg-info/|/build/" | \
	{ xargs grep -niE "todo|xxx" || true; } | \
	grep -vE "assemble\.sh:" | \
	sed 's|^\./||' | \
	sort | \
	sed -E 's/^([^:]+):([0-9]+):/todo \1 \2 /' \
	|| true

# Lint
ruff check --output-format concise --fix sigma tests 2>&1 | \
	grep -vxE -e "All checks passed!|Found [0-9]+ errors?\." \
	    -e "^No fixes available \(.*\)\.$" | \
	sed -E 's/^([^:]+):([0-9]+): /ruff \1 \2 /' \
	|| true

# Type check (ty)
ty check --output-format concise --no-progress --extra-search-path tests sigma tests | \
  grep -vxE "All checks passed!|Found [0-9]+ diagnostics?" | \
  sed -E 's/^([^:]+):([0-9]+):[0-9]+: /ty \1 \2 /' \
  || true

# Type check (mypy)
mypy --ignore-missing-imports --no-color sigma tests | \
	grep -vxE -e "Success: no issues found in [0-9]+ source files?" \
	    -e "Found [0-9]+ errors? in [0-9]+ files? \(.*\)" | \
	sed -E 's/^([^:]+):([0-9]+): /mypy \1 \2 /' \
	|| true

# Unit tests, one worker per module
TEST_LOGS=$(mktemp -d)
print -l tests/test_*.py(:t) | \
	xargs -P 8 -I '{}' \
	sh -c "python -m unittest discover tests -q -p '{}' > $TEST_LOGS/'{}'.log 2>&1" \
	|| true
cat $TEST_LOGS/*.log | \
    grep -vE \
        -e '^$|^-+$|^Ran [0-9]+ tests? in' -e '^OK( \(.*\))?$' \
    || true
rm -rf $TEST_LOGS

# Reinstall
pip install -e . -q
