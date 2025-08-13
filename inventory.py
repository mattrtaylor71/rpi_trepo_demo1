import csv
import json
import os
from typing import List, Dict, Optional

import numpy as np

INVENTORY_CSV = 'inventory.csv'


def load_inventory() -> List[Dict]:
    """Load inventory rows from CSV. Embeddings are stored as JSON arrays."""
    rows: List[Dict] = []
    if not os.path.exists(INVENTORY_CSV):
        return rows
    with open(INVENTORY_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['quantity'] = int(row.get('quantity') or 0)
            emb_txt = row.get('embedding') or ''
            if emb_txt:
                try:
                    row['embedding'] = np.array(json.loads(emb_txt), dtype=float)
                except Exception:
                    row['embedding'] = None
            else:
                row['embedding'] = None
            rows.append(row)
    return rows


def save_inventory(rows: List[Dict]):
    fieldnames = ['canonical_name', 'quantity', 'embedding']
    with open(INVENTORY_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = r.copy()
            emb = out.get('embedding')
            if isinstance(emb, np.ndarray):
                out['embedding'] = json.dumps(emb.tolist())
            elif emb is None:
                out['embedding'] = ''
            writer.writerow(out)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def manual_select(rows: List[Dict]) -> Optional[int]:
    if not rows:
        print("[inventory] Inventory is empty")
        return None
    print("Select item to discard:")
    for idx, r in enumerate(rows, 1):
        print(f" {idx}. {r['canonical_name']} (qty={r['quantity']})")
    choice = input("Enter number (or blank to cancel): ").strip()
    if not choice:
        return None
    try:
        n = int(choice)
        if 1 <= n <= len(rows):
            return n - 1
    except Exception:
        pass
    return None


def match_inventory(canonical_name: str, embedding: Optional[np.ndarray],
                    rows: List[Dict], client=None,
                    threshold: float = 0.80) -> Optional[int]:
    """Return index of best match in rows or None."""
    # Use embeddings if available
    if embedding is not None and any(r.get('embedding') is not None for r in rows):
        best_idx = None
        best_score = -1.0
        for i, r in enumerate(rows):
            emb = r.get('embedding')
            if emb is None:
                continue
            score = cosine_sim(embedding, emb)
            if score > best_score:
                best_idx, best_score = i, score
        if best_idx is not None and best_score >= threshold:
            return best_idx
        print(f"[inventory] Low confidence {best_score:.2f}; threshold {threshold}")
        return manual_select(rows)

    # Fallback to OpenAI selection if client provided
    if client is not None and rows:
        try:
            names = [r['canonical_name'] for r in rows]
            prompt = (
                f"From this list: {', '.join(names)}\n"
                f"Which item best matches '{canonical_name}'? Respond with the exact name."
            )
            resp = client.chat.completions.create(
                model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                messages=[{"role": "user", "content": prompt}],
            )
            reply = (resp.choices[0].message.content or '').strip()
            for i, name in enumerate(names):
                if name.lower() == reply.lower():
                    return i
            print(f"[inventory] OpenAI reply '{reply}' not matched")
        except Exception as e:
            print(f"[inventory] OpenAI selection error: {e}")
    # Manual selection as last resort
    return manual_select(rows)


def handle_discard(canonical_name: str, embedding: Optional[np.ndarray], client=None):
    rows = load_inventory()
    if not rows:
        print("[inventory] No inventory entries")
        return
    idx = match_inventory(canonical_name, embedding, rows, client)
    if idx is None:
        print("[inventory] Discard cancelled")
        return
    row = rows[idx]
    if row['quantity'] > 1:
        row['quantity'] -= 1
        print(f"[inventory] Decremented {row['canonical_name']} to {row['quantity']}")
    else:
        print(f"[inventory] Removed {row['canonical_name']}")
        rows.pop(idx)
    save_inventory(rows)
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
