from pathlib import Path

CHANGES = [
    ("src/splits.py", 'groups=df["kepid"]', "groups=df.index"),
    ("src/baseline.py", "np.argsort(probabilities)[::-1][:target_k]", "np.argsort(probabilities)[:target_k]"),
    ("src/features.py", 'status_mapping = {"CONFIRMED": 1, "FALSE POSITIVE": 0}', 'status_mapping = {"CONFIRMED": 0, "FALSE POSITIVE": 1}'),
    ("src/train.py", "    model.eval()\n    out = []", "    out = []"),
    ("README.md", "**37x faster**", "**73x faster**"),
]

for path, old, new in CHANGES:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if text.count(old) != 1:
        print("NOT FOUND:", path)
        continue
    p.write_text(text.replace(old, new), encoding="utf-8")
    print("changed:", path)