"""
Generate a diversified 240-claim benchmark CSV with balanced labels and categories.

Output format matches claim_annotation_template_240.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

CATEGORIES = [
    "health",
    "science",
    "climate",
    "politics",
    "finance",
    "history",
    "conflict",
    "general",
    "demographics",
    "technology",
    "media",
    "work",
]

LABELS = ["SUPPORTED", "REFUTED", "NEI"]
DIFFICULTIES = ["easy", "medium", "hard"]

SOURCE_HINTS = {
    "health": "https://www.who.int",
    "science": "https://www.nature.com",
    "climate": "https://www.ipcc.ch",
    "politics": "https://www.parliament.uk",
    "finance": "https://www.imf.org",
    "history": "https://www.britannica.com",
    "conflict": "https://www.un.org",
    "general": "https://www.bbc.com",
    "demographics": "https://www.un.org/development/desa/pd/",
    "technology": "https://www.nist.gov",
    "media": "https://www.reuters.com",
    "work": "https://www.oecd.org",
}

SUBJECTS = {
    "health": ["influenza vaccines", "childhood immunization", "blood pressure monitoring", "antibiotic stewardship", "public sanitation"],
    "science": ["gravity", "DNA sequencing", "photosynthesis", "plate tectonics", "electric circuits"],
    "climate": ["global temperature trends", "sea-level rise", "carbon emissions", "Arctic ice extent", "renewable transition"],
    "politics": ["electoral systems", "constitutional amendments", "parliamentary voting", "campaign finance", "executive terms"],
    "finance": ["inflation data", "bond yields", "central bank policy", "exchange rates", "equity markets"],
    "history": ["Napoleonic era", "Roman empire chronology", "industrial revolution", "WWII timeline", "French revolution"],
    "conflict": ["ceasefire agreements", "peace talks", "civilian protection", "sanctions policy", "post-conflict recovery"],
    "general": ["satire websites", "image authenticity", "search ranking", "fact-checking workflows", "online rumors"],
    "demographics": ["fertility rates", "urbanization", "age distribution", "migration flows", "census updates"],
    "technology": ["software vulnerabilities", "machine learning evaluation", "cloud uptime", "encryption standards", "database replication"],
    "media": ["headline framing", "editorial policy", "source attribution", "correction policies", "newsroom verification"],
    "work": ["remote collaboration", "hybrid schedules", "productivity metrics", "employee retention", "meeting load"],
}

SUPPORTED_TMPL = [
    "Recent audits indicate that {subject} changed measurably between {year1} and {year2}.",
    "Official reports from {year2} show that {subject} is tracked with standardized indicators.",
    "Independent reviews in {year2} describe {subject} as a measurable and evidence-based trend.",
    "Peer-reviewed summaries after {year1} confirm that {subject} can be verified using public data.",
    "Regulatory and institutional publications in {year2} include repeatable metrics for {subject}.",
]

REFUTED_TMPL = [
    "All experts agree that {subject} has never changed in any region since {year1}.",
    "Every dataset proves that {subject} is always identical across all countries.",
    "No credible source has ever measured {subject} at any time in history.",
    "It is impossible for {subject} to be affected by policy, behavior, or technology.",
    "Every published claim about {subject} after {year1} has been fabricated.",
]

NEI_TMPL = [
    "A confidential committee secretly controlled all global outcomes for {subject} in {year2}.",
    "An undisclosed report from {year2} proves a universal conspiracy behind {subject}.",
    "Hidden archives allegedly show that every public metric on {subject} is manipulated.",
    "A private organization is said to have rewritten all records related to {subject} in {year2}.",
    "Unreleased evidence supposedly confirms that {subject} followed one secret rule worldwide.",
]


def build_claim(category: str, label: str, i: int) -> str:
    subject = SUBJECTS[category][i % len(SUBJECTS[category])]
    year1 = 2000 + (i % 15)
    year2 = 2010 + (i % 16)

    if label == "SUPPORTED":
        tmpl = SUPPORTED_TMPL[i % len(SUPPORTED_TMPL)]
    elif label == "REFUTED":
        tmpl = REFUTED_TMPL[i % len(REFUTED_TMPL)]
    else:
        tmpl = NEI_TMPL[i % len(NEI_TMPL)]

    return tmpl.format(subject=subject, year1=year1, year2=year2)


def main() -> int:
    out_path = Path("data/benchmarks/claim_annotation_template_240.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    idx = 1
    # 12 categories x (20 claims each) = 240
    # Per category: ~7 SUPPORTED, ~7 REFUTED, ~6 NEI -> near-balanced globally.
    per_category_labels = ["SUPPORTED"] * 7 + ["REFUTED"] * 7 + ["NEI"] * 6

    for category in CATEGORIES:
        for j, label in enumerate(per_category_labels):
            claim = build_claim(category, label, j)
            rows.append(
                {
                    "id": f"v2-{idx:03d}",
                    "claim": claim,
                    "label": label,
                    "category": category,
                    "difficulty": DIFFICULTIES[j % len(DIFFICULTIES)],
                    "source_url": SOURCE_HINTS[category],
                    "annotator_1": "",
                    "annotator_2": "",
                    "annotator_3": "",
                    "notes": "auto-generated diversified draft; human verification required",
                }
            )
            idx += 1

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "claim",
                "label",
                "category",
                "difficulty",
                "source_url",
                "annotator_1",
                "annotator_2",
                "annotator_3",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
