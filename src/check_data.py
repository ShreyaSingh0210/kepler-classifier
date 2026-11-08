from pathlib import Path

import numpy as np

files = sorted((Path("data") / "processed").glob("*.npz"))
rows = []
for f in files:
    d = np.load(f)
    lv = d["local_view"]
    rows.append((int(d["label"]), int(np.argmin(lv)), bool(np.isnan(lv).any())))

for label, name in [(1, "planets"), (0, "false positives")]:
    mins = np.array([r[1] for r in rows if r[0] == label])
    centered = np.mean(np.abs(mins - 100) <= 15)
    print(f"{name}: {len(mins)} stars, deepest point near the center (+-15 bins): {centered:.0%}")

print("Stars with NaN values:", sum(r[2] for r in rows))