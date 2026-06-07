"""
Generate Tableau-ready CSV extracts for the SentiYelp dashboard (Project 5).
Outputs tidy (long-format) CSVs into tableau/data/ — ideal for Tableau Public.
Run:  python generate_tableau_data.py
"""
import csv, json, os, collections
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "tableau", "data")
os.makedirs(OUT, exist_ok=True)

SAMPLE = os.path.join(ROOT, "data", "processed", "sample_50k.json")
BUSINESS = os.path.join(ROOT, "data", "raw", "yelp_academic_dataset_business.json")
EVAL = os.path.join(ROOT, "reports", "evaluation_results.json")


def write_csv(name, header, rows):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  {name:<28} {len(rows):>6} rows")


def label(s):
    return {"negative": "N\u00e9gatif", "neutral": "Neutre", "positive": "Positif"}.get(s, s)


# ── 1-4. From evaluation_results.json ──────────────────────────────────────────────
ev = json.load(open(EVAL, encoding="utf-8"))

comp = ev.get("comparison", [])
write_csv("model_comparison.csv",
          ["Model", "Accuracy", "F1_Macro", "F1_Negative", "F1_Neutral", "F1_Positive"],
          [[m["Model"], m["Accuracy"], m["F1 Macro"], m["F1 Negative"], m["F1 Neutral"], m["F1 Positive"]] for m in comp])

perclass = []
for m in comp:
    perclass += [[m["Model"], "N\u00e9gatif", m["F1 Negative"]],
                 [m["Model"], "Neutre", m["F1 Neutral"]],
                 [m["Model"], "Positif", m["F1 Positive"]]]
write_csv("model_f1_by_class.csv", ["Model", "Class", "F1_Score"], perclass)

burn = []
for s in ev.get("sprints", []):
    for d, ideal, actual in zip(s["days"], s["ideal"], s["actual"]):
        burn.append([s["name"], d, "Id\u00e9al", round(ideal, 2)])
        burn.append([s["name"], d, "R\u00e9el", actual])
write_csv("sprint_burndown.csv", ["Sprint", "Day", "Type", "Points_Remaining"], burn)

velocity = [["Sprint 0", 6, 6, 6], ["Sprint 1", 14, 11, 11], ["Sprint 2", 16, 16, 16], ["Sprint 3", 24, 18, 18]]
write_csv("sprint_velocity.csv", ["Sprint", "Planned", "Delivered", "Velocity"], velocity)

# ── 5-7,9. Stream the review sample ─────────────────────────────────────────────────────────────────────────────────────────
dist = collections.Counter()
temporal = collections.Counter()          # (year, sentiment) -> count
texts_by_sent = collections.defaultdict(list)
reviews_rows = []
biz_to_sent = collections.defaultdict(collections.Counter)  # business_id -> sentiment counts

with open(SAMPLE, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        sent = r.get("sentiment", "")
        dist[sent] += 1
        date = (r.get("date") or "")[:10]
        year = date[:4]
        month = date[5:7]
        if year.isdigit():
            temporal[(year, sent)] += 1
        txt = r.get("text", "") or ""
        if len(texts_by_sent[sent]) < 8000:
            texts_by_sent[sent].append(txt)
        reviews_rows.append([r.get("review_id", ""), date, year, month,
                             r.get("stars", ""), label(sent), len(txt.split())])
        biz_to_sent[r.get("business_id", "")][sent] += 1

write_csv("sentiment_distribution.csv", ["Sentiment", "Count"],
          [[label(s), c] for s, c in sorted(dist.items())])

write_csv("temporal_sentiment.csv", ["Year", "Sentiment", "Count"],
          [[y, label(s), c] for (y, s), c in sorted(temporal.items())])

write_csv("reviews_sample.csv",
          ["Review_ID", "Date", "Year", "Month", "Stars", "Sentiment", "Word_Count"],
          reviews_rows)

try:
    from sklearn.feature_extraction.text import CountVectorizer
    wf_rows = []
    for sent, docs in texts_by_sent.items():
        if not docs:
            continue
        cv = CountVectorizer(stop_words="english", max_features=60, token_pattern=r"[A-Za-z]{3,}")
        X = cv.fit_transform(docs)
        freqs = X.sum(axis=0).A1
        for word, idx in cv.vocabulary_.items():
            wf_rows.append([label(sent), word, int(freqs[idx])])
    wf_rows.sort(key=lambda r: (r[0], -r[2]))
    write_csv("word_frequencies.csv", ["Sentiment", "Word", "Frequency"], wf_rows)
except Exception as e:
    print("  word_frequencies skipped:", e)

needed = set(biz_to_sent.keys())
cat_sent = collections.defaultdict(collections.Counter)
with open(BUSINESS, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        b = json.loads(line)
        bid = b.get("business_id")
        if bid not in needed:
            continue
        cats = b.get("categories") or ""
        for cat in [c.strip() for c in cats.split(",") if c.strip()]:
            for sent, n in biz_to_sent[bid].items():
                cat_sent[cat][sent] += n

totals = {c: sum(v.values()) for c, v in cat_sent.items()}
top = sorted(totals, key=totals.get, reverse=True)[:20]
cat_rows = []
for c in top:
    for sent, n in cat_sent[c].items():
        cat_rows.append([c, label(sent), n])
write_csv("category_sentiment.csv", ["Category", "Sentiment", "Count"], cat_rows)

print("\nAll Tableau extracts written to:", OUT)
