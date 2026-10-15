# Installation

## Requirements
repobrief needs Python 3.11 or 3.12 and git. Python 3.13 is not supported yet.

## Windows
Open PowerShell, create a virtual environment with py -3.12 -m venv .venv, activate it
with .venv\Scripts\Activate.ps1 and run pip install repobrief. If PowerShell says that
repobrief is not recognized, the virtual environment is not active in that window.

## Linux and macOS
Create a virtual environment with python3 -m venv .venv, activate it with
source .venv/bin/activate and run pip install repobrief.

## Upgrading
Run pip install --upgrade repobrief inside the same virtual environment. Check the
installed version with repobrief --version.

## Using a local model
To run without a hosted model, install Ollama, run ollama pull llama3.2 and set
backend = "ollama" in the [llm] section of config.toml.
