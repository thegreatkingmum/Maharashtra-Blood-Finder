from pathlib import Path
from pypdf import PdfReader
import re

PDF_FILE = Path("snapshots/blood-avail-data-10-9-26.pdf")

reader = PdfReader(str(PDF_FILE))

print("Pages:", len(reader.pages))

all_text = []

for i, page in enumerate(reader.pages, start=1):

    text = page.extract_text()

    if text:
        all_text.append(text)

        print(
            f"Page {i}: {len(text)} characters"
        )

full_text = "\n".join(all_text)

print("\nTotal extracted characters:", len(full_text))

print("\n--- FIRST 3000 CHARACTERS ---\n")
print(full_text[:3000])

print("\n--- hospitalCode COUNT ---")

codes = re.findall(
    r'"hospitalCode"\s*:',
    full_text
)

print(
    "hospitalCode occurrences:",
    len(codes)
)

print("\n--- ENTRYDATE COUNT ---")

dates = re.findall(
    r'"entrydate"\s*:',
    full_text
)

print(
    "entrydate occurrences:",
    len(dates)
)

print("\n--- AVAILABLE_WITHQTY COUNT ---")

stock = re.findall(
    r'"available_WithQty"\s*:',
    full_text
)

print(
    "available_WithQty occurrences:",
    len(stock)
)