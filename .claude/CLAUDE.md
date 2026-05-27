# Agent Instructions

The `sigma` Python library provides conditional inference trees for Python.

@CLAUDE.local.md

## Development Executables

- `assemble.sh`: format code, lint, type check, tests.

## Architecture

To be completed.

## Documentation

- For public declarations, use the Google Python Style Docstring standard, including the `Args`, `Returns` and `Raises` sections for public declarations.
- Wrap docstring paragraphs as close to 80 columns as possible without exceeding 80. Do not break lines earlier than needed.
- Private declarations, i.e., functions, classes, methods or constants whose names are prefixed with an underscore (_) should be documented with a single line explaining their purpose only.
- Do not use double backticks in docstrings.
- Do not leave an empty line at the end of a docstring.
- Do not use line feeds to break same paragraph content.

## Code

- Use module imports with dot-chaining, e.g. `import my_lib` and `my_lib.do_something()` instead of `from my_lib import do_something` and `do_something()`.
- As the only exceptions, you can still use `from . import ...` to import local modules, or to import and export objects in `__init__.py` files.
- Do not rename modules on import, e.g. write `import some_lib`, not `import some_lib as sl`.
- Do not use `assert` for type validation. Use properly typed data structures instead, or if really not possible, `typing.cast` statement, or if not relevant (e.g., the data type may be different), `isinstance` checks with `raise` guards.
- Always place `None` first in type annotations, e.g., `None | MyClass` instead of `MyClass | None`.
- Use `os.path` instead of `pathlib`.
- Name Python exception variables with a complete name, e.g., `exception`, not `e` or `exc`.
