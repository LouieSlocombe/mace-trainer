# Conda environment

This file creates the lightweight development environment. It does not install
the molecular-dynamics backend or ORCA. For actual dynamics, follow the
openmmqmmm environment instructions in the root `README.md` and install
mace-trainer there.

To create a separate development environment from the repository root:

```bash
conda env create -f build_tools/environment.yml
conda activate mace-trainer
python -m pip install --group dev -e .
```

This path resolves the latest versions allowed by `pyproject.toml` rather than
the exact versions pinned in `uv.lock`.
