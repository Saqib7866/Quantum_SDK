
file_path = r"\\wsl.localhost\Ubuntu\home\saqib\projects\Quantum_SDK\streamlit_app.py"

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Keep lines up to 738 (index 738 is line 739, so slice up to 738 means 0..737)
# Wait, line numbers are 1-based.
# Line 739 is the start of duplicate `def display_results`.
# So I want to keep lines 1 to 738.
# In 0-based index, that is 0 to 737.
# lines[738] would be the 739th line.

cutoff_index = 738

# Verify the line at cutoff is indeed the start of duplication or close to it
print(f"Line at {cutoff_index}: {repr(lines[cutoff_index])}")

# Truncate
new_lines = lines[:cutoff_index]

# Append entry point
new_lines.append('\n')
new_lines.append('if __name__ == "__main__":\n')
new_lines.append('    main()\n')

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"Truncated file to {len(new_lines)} lines.")
