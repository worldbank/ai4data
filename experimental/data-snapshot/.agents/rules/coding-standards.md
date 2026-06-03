---
trigger: always_on
---

# Coding Standards — Data Snapshot Extraction

To ensure consistency and maintainability across the codebase, all contributors (including AI agents) must adhere to the following standards.

## 1. Python Formatting

- **Formatter**: Mandatory use of **Black**.
- **Line Length**: Default Black behavior (88 characters).

## 2. Documentation

- **Standard**: All Python code must use the **NumPy docstring standard**.
- **Requirement**: Public modules, classes, and functions must include a descriptive docstring.
- **Content**:
    - **Summary**: A concise one-line summary of the object's purpose.
    - **Parameters**: (If applicable) Names, types, and descriptions of all arguments.
    - **Returns**: (If applicable) Type and description of the return value.
    - **Raises**: (If applicable) Any exceptions that are explicitly raised.

## 3. Type Hinting

- **Mandatory**: All function signatures should include type hints for parameters and return values.
- **Style**: Use modern Python 3.10+ type hinting (e.g., `list[str]` instead of `List[str]`, `str | None` instead of `Optional[str]`).

## 4. Code Structure

- **Naming**: Follow PEP 8 (snake_case for functions/variables, PascalCase for classes).
- **Constants**: Define project-wide constants in `src/dsa/constants.py`.
