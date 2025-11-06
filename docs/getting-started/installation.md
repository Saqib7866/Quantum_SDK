# Installation

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- (Optional) A virtual environment (recommended)

## Installing with pip

The easiest way to install QX-IR is using pip:

```bash
pip install qx-ir
```

## Installing from source

If you want to install the latest development version:

```bash
# Clone the repository
git clone https://github.com/zenaquantum/qx-ir.git
cd qx-ir

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

## Verifying the installation

After installation, you can verify that QX-IR is installed correctly by running:

```python
import qx_ir
print(f"QX-IR version: {qx_ir.__version__}")
```

## Optional Dependencies

Some features require additional dependencies:

- For visualization: `pip install matplotlib`
- For advanced simulation: `pip install numpy scipy`
- For documentation building: `pip install mkdocs mkdocs-material mkdocstrings-python`

## Development Setup

For contributing to QX-IR, you'll need to set up the development environment:

```bash
# Clone the repository
git clone https://github.com/zenaquantum/qx-ir.git
cd qx-ir

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
pip install -e .[dev]

# Run tests
pytest
```

## Troubleshooting

If you encounter any issues during installation:

1. Make sure you have the latest version of pip:
   ```bash
   pip install --upgrade pip
   ```

2. If you get permission errors, try using the `--user` flag:
   ```bash
   pip install --user qx-ir
   ```

3. For other issues, please check the [GitHub issues](https://github.com/zenaquantum/qx-ir/issues) or open a new one.
