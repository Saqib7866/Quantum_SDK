
file_path = r"\\wsl.localhost\Ubuntu\home\saqib\projects\Quantum_SDK\streamlit_app.py"

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

start_line = 535
end_line = 550

print(f"Printing lines {start_line} to {end_line} (0-indexed):")
for i in range(start_line, min(end_line, len(lines))):
    print(f"{i}: {repr(lines[i])}")
