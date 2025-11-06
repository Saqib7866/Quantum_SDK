# Setting up a development environment on Windows (WSL2)

This guide shows the minimal steps to get a reproducible dev environment for the ZenaQuantum SDK using WSL2 + Ubuntu and VS Code.

1) Install WSL2 (Windows 10/11)
   - In PowerShell (as Administrator):
     - `wsl --install -d ubuntu`  (this installs WSL2 and Ubuntu)
   - Reboot if prompted, then run the Ubuntu terminal from the Start menu.

2) Update Ubuntu and install system packages
   - In your WSL terminal:
     ```bash
     sudo apt update && sudo apt upgrade -y
     sudo apt install -y build-essential python3 python3-venv python3-pip git
     ```

3) Install Rust toolchain (optional)
   - If you plan to add Rust-based native modules:
     ```bash
     curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
     source $HOME/.cargo/env
     rustup toolchain install stable
     ```

4) Create a Python virtual environment and install dependencies
   - From the project root inside WSL (`/home/<you>/projects/zena_quantum`):
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     pip install --upgrade pip
     pip install -r requirements-dev.txt
     pip install -e .
     ```
   - The `-e .` (editable) install will make the `qx` package importable in the venv.

5) Recommended VS Code extensions (in Windows):
   - Remote - WSL (ms-vscode-remote.remote-wsl)
   - Python (ms-python.python)
   - Rust (rust-lang.rust) — optional
   - Pylance (ms-python.vscode-pylance)

6) Opening the project in VS Code
   - In Windows VS Code: press F1 → "Remote-WSL: New Window" and open the project folder in WSL.
   - Select the Python interpreter from `.venv/bin/python` in the status bar.

7) Try the "Hello QX" example
   - With the venv activated, run:
     ```bash
     python examples/bell.py --backend sim-local --shots 512
     ```
   - You should see a JSON artifact with counts and metadata.

Notes
 - If you need GPU or other native accelerators, follow additional setup steps and document them here.
 - The `requirements-dev.txt` holds test and plotting deps; you can pin exact versions if you need reproducible builds.
