---
title: UV & Modern Python Dependency Management
description: A beginner-friendly guide to UV, the modern Python package and dependency manager.

date: February 24, 2026
modified: April 25, 2026

author: Arpon Kapuria
category: Dev Journal
tags: AWS, Tooling, Infrastructure
---

## What is uv ?

**UV** is a fast Python package installer and environment manager. It replaces traditional workflows involving:

- `pip`
- `venv`
- `requirements.txt`

It is designed to handle environment creation and dependency management automatically.

```bash
# Run this command to create a new project with uv
uv init <project_name>
```

After successful run, the project should be initialised with the following files:

- `.gitignore`
- `main.py`
- `README.md`
- `.python-version` - Specific version of Python used in this project
- `pyproject.toml` - Main configuration file with project metadata & dependencies
- `uv.lock` - This file is created when a package is installed, maintains a proper dependencies structure

## Handling Virtual Environments

When working on Python projects, we usually create a virtual environment manually using:

```bash
python -m venv venv
```

But, in most cases, `uv` automatically creates and manages a virtual environment for you. You do not always need to manually create one.

### 2.1 Creating an Environment

You can manually create a virtual environment using the following command. This creates a `.venv/` folder in your project directory.

```bash
uv venv
```

### 2.2 Installing Packages with `uv` 

If you run:

```bash
uv add pandas
```

`uv` will:

- Automatically create a virtual environment if one does not exist
- Install packages inside that environment

You do not need to create it manually beforehand.

### 2.3 Running Code with `uv` 

If you run:

```bash
uv run python script.py
```

`uv` ensures:

- An isolated environment exists
- Dependencies are resolved
- Script runs inside that environment

This makes project setup clean and reproducible. `UV` manages and uses the virtual environment automatically when running commands, so manual activation is not required. 


Activation is required only if you want to:

- Use the environment manually
- Run `python` directly without `uv run`

Then you can activate:

```bash
# windows
.venv\Scripts\activate 

# macos
source .venv/bin/activate 
```

But with normal `uv` workflow → activation is unnecessary.

## Handling Packages

`uv` provides a simple and modern way to manage dependencies in your Python project. Unlike traditional `pip`, it integrates directly with your project configuration and lock files, ensuring reproducibility.

### 3.1 Installing Package

To add a package to your project:

```bash
uv add <package_name>
```

This command:

- Adds the dependency to `pyproject.toml`
- Updates the lock file
- Installs it inside the managed virtual environment

Install multiple packages:

```bash
uv add pandas numpy scikit-learn
```

Install a specific version:

```bash
uv add pandas==2.2.2
```

Add a package to a particular group while installing:

```bash
uv add --group "group_name" pandas

# ex: uv add --group dataViz pandas (here dataViz is a group)
```

### 3.2 Removing Package

To remove a dependency:

```bash
uv remove <package_name>
```

This:
- Deletes the package from `pyproject.toml`
- Updates the lock file
- Uninstalls it from the environment

### 3.3 When a Package Is Not Found

If a package is not available through the default registry, you can fall back to pip-compatible mode:

```bash
uv pip install <package_name>
```

If necessary, you can also directly use:

```bash
pip install <package_name>
```

However, this is not recommended for structured projects, because `uv add` properly tracks dependencies in `pyproject.toml`.

Best practice:

- Use `uv add` for project dependencies.
- Use `uv pip` only for temporary or quick installs.

### 3.4 Viewing the Dependency Tree

To inspect the full dependency graph:

```bash
uv tree
```

This shows:
- Direct dependencies
- Sub-dependencies
- Version relationships

It is extremely helpful when debugging version conflicts.

### 3.5 Updating Dependencies

To update a specific package:

```bash
uv add <package_name>--upgrade
```

To generate or update the lock file:

```bash
uv lock
```

To update all dependencies to the latest allowed versions:

```bash
uv lock --upgrade

uv sync
```

### 3.6 Syncing the Environment

If you clone a project and want to install all dependencies:

```bash
uv sync

# if there are multple groups in pyproject.toml
uv sync --all-groups
```

To enforce strict lock consistency (e.g., in CI/CD):

```bash
uv sync --locked
```

This reads from the lock file and installs exact versions.

### 3.7. Managing Python Versions

List available Python versions:

```bash
uv python list
```

Install a specific version:

```bash
uv python install 3.13
```

This allows consistent Python environments across machines.

### 3.8 Updating uv

To update `uv` to the latest version:

```bash
uv self update
```

<hr>

## Appendix: A Quick Demo

```bash
# To create a new FastAPI project and pin it to python 3.13

uv init <project_name>
uv python pin 3.13
uv add fastapi 
uv run [main.py](http://main.py) # automatically installs python 3.13 and fastapi into venv
```