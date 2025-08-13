import csv
import json
import os
from typing import List

CSV_PATH = os.path.join(os.path.dirname(__file__), 'inventory.csv')

def add_inventory_item(canonical_name: str, display_name: str, expiry_ts: int, embedding: List[float]) -> None:
    """Append a new inventory record to CSV.

    Loads existing rows, appends the new record, and writes the CSV back to disk.
    The embedding list is stored as a JSON-encoded string.
    """
    rows = []
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
    rows.append([canonical_name, display_name, str(expiry_ts), json.dumps(embedding)])
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
