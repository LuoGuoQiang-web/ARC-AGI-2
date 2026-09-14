import json, glob, os, re, html
from lxml import html as LH

SRC = r"C:\Users\LRDC07\Desktop\Kaggle\_forums\arc-prize-2026-arc-agi-2"
OUT = r"C:\Users\LRDC07\Desktop\Kaggle\_forums\_txt"
os.makedirs(OUT, exist_ok=True)

def to_text(h):
    if not h:
        return ""
    # keep link hrefs
    h = re.sub(r'<a\b[^>]*href="([^"]+)"[^>]*>(.*?)</a>', lambda m: f'{m.group(2)} <{m.group(1)}>', h, flags=re.S|re.I)
    try:
        doc = LH.fromstring("<div>"+h+"</div>")
        t = doc.text_content()
    except Exception:
        t = re.sub(r'<[^>]+>', ' ', h)
    t = html.unescape(t)
    t = t.replace('\u00a0', ' ')
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\n\s*\n\s*\n+', '\n\n', t)
    return t.strip()

index = []
for f in sorted(glob.glob(os.path.join(SRC, "*.json"))):
    d = json.load(open(f, encoding="utf-8"))
    tp = d["topic"]
    tid = tp["id"]
    lines = []
    lines.append(f"===== TOPIC {tid} | votes={tp.get('votes')} | comments={tp.get('commentCount')} | date={tp.get('postDate')}")
    lines.append(f"TITLE: {tp.get('title')}")
    lines.append("")
    for i, m in enumerate(d["messages"]):
        role = "OP" if i == 0 else f"reply#{i}"
        lines.append(f"--- [{tid}] {role} msgid={m.get('id')} date={m.get('postDate')} votes={m.get('votes')}")
        lines.append(to_text(m.get("content")))
        lines.append("")
    body = "\n".join(lines)
    open(os.path.join(OUT, f"{tid}.txt"), "w", encoding="utf-8").write(body)
    index.append((tid, tp.get("votes"), tp.get("commentCount"), tp.get("postDate"), tp.get("title"), len(body)))

index.sort(key=lambda r: -(r[1] or 0))
with open(os.path.join(OUT, "_INDEX.txt"), "w", encoding="utf-8") as fh:
    fh.write("topic_id | votes | comments | date | bytes | title\n")
    for r in index:
        fh.write(f"{r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[5]} | {r[4]}\n")

total = sum(r[5] for r in index)
print(f"wrote {len(index)} txt files, total {total} chars")
