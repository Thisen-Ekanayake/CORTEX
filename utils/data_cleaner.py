"""
Data cleaning utility script.

Cleans training data files by:
- Removing empty lines and separator lines (e.g., =====)
- Removing leading numbers (e.g., "2. ", "10. ")
- Writing cleaned output to a new file
"""

import re

input_file = "dataset/meta.txt"
output_file = "meta.txt"

cleaned_lines = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        # Skip empty lines and separator lines like =====
        if not line or set(line) == {"="}:
            continue

        # Remove leading numbers like "2. ", "10. "
        line = re.sub(r"^\d+\.\s*", "", line)

        cleaned_lines.append(line)

with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(cleaned_lines))

print("Cleaning complete.")