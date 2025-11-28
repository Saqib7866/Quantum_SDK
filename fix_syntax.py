
import os

file_path = r"\\wsl.localhost\Ubuntu\home\saqib\projects\Quantum_SDK\streamlit_app.py"

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the anchor lines
start_idx = -1
end_idx = -1


for i, line in enumerate(lines):
    if "st.table" in line:
        print(f"Found st.table at {i}: {repr(line)}")
        start_idx = i
    if "_display_execution_summary" in line:
        print(f"Found _display_execution_summary at {i}: {repr(line)}")
        if i > start_idx and start_idx != -1:
            end_idx = i
            break

if start_idx != -1 and end_idx != -1:
    print(f"Found block from {start_idx} to {end_idx}")
    
    # Construct new lines
    new_block = [
        '                st.table(gate_data)\n',
        '\n',
        'def display_results(result):\n',
        '    """Displays the results of a quantum circuit execution with enhanced visualization."""\n',
        '    _display_execution_summary(result)\n'
    ]
    
    # Replace lines
    lines[start_idx:end_idx+1] = new_block
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print("File updated successfully.")
else:
    print("Could not find anchor lines.")
