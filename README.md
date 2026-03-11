# Image-Captioning-Architectures

## Prerequisites

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/) - Fast Python package installer and resolver
- pre-commit - Git hook scripts for identifying simple issues before submission

## Usage

### Installing Development Packages

Install all development dependencies using uv:

```bash
uv sync --dev
```

This will install all dependencies including development tools like mypy and pre-commit.

### Setting Up Pre-commit Hooks

After installing the development packages, set up pre-commit hooks:

```bash
pre-commit install
```

This will configure the following hooks to run automatically before each commit:
- **ruff** - Fast Python linter and formatter (with auto-fix)
- **mypy** - Static type checker

### Running Pre-commit Manually

To manually run pre-commit on all files before committing:

```bash
pre-commit run --all-files
```

Or to run on staged files only:

```bash
pre-commit run
```