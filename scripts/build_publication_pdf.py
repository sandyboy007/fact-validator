from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "publication_paper_final.pdf"


def body(text: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph(text, styles["Body"])


def heading(text: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph(text, styles["Heading2"])


def build_pdf() -> None:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCenter",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubTitleCenter",
            parent=styles["Normal"],
            alignment=TA_CENTER,
            fontName="Helvetica",
            fontSize=10.5,
            leading=13,
            textColor=colors.HexColor("#4a4a4a"),
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.2,
            leading=13.8,
            spaceAfter=8,
        )
    )

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
        title="Fact Validator Publication Paper",
        author="Fact Validator Project Team",
    )

    story: list = []
    story.append(Spacer(1, 0.25 * inch))
    story.append(
        Paragraph(
            "Fact Validator: A Transparent and Deployment-Ready Fact-Checking Architecture",
            styles["TitleCenter"],
        )
    )
    story.append(
        Paragraph(
            "Auditable credibility scoring, tool-augmented retrieval, and versioned 5000-claim evaluation",
            styles["SubTitleCenter"],
        )
    )
    story.append(Paragraph("Publication Preview PDF | July 2026", styles["SubTitleCenter"]))
    story.append(
        body(
            "This PDF is a ReportLab preview generated from repository metadata and manuscript-aligned content. It is not the final IEEE-rendered layout; the true submission source is docs/publication_paper_final.tex.",
            styles,
        )
    )

    abstract = (
        "Fact Validator is an end-to-end fact-checking system for fake-news and misinformation analysis over text and URL-derived content. "
        "The system extracts claims, retrieves web evidence, applies an auditable source-credibility prior, reranks evidence semantically, produces claim-level verdicts, and aggregates a final misinformation-likelihood score. "
        "On the frozen 5000-claim benchmark family stored in the repository, the default full pipeline reaches 50.82% accuracy, the best tuned deployment variant reaches 50.98%, and the strongest frozen ablation reaches 51.34%. "
        "The full pipeline exceeds majority and length baselines in the strict-validation snapshot, while caching lowers estimated monthly evidence-retrieval cost by 71.43%. "
        "The strongest novelty relative to Zero-shot Faithful Factual Error Correction and FacTool is not universal factuality superiority, but a stronger combination of source-aware transparency, operational accountability, and full-stack deployment readiness."
    )
    story.append(heading("Abstract", styles))
    story.append(body(abstract, styles))

    story.append(heading("Project Analysis", styles))
    story.append(
        body(
            "The repository contains multiple evaluation tracks, but the strongest publication-safe surface is the frozen 5000-claim result family dated 2026-07-02. That family is internally versioned, includes strict-validation snapshots, and supports a modest but significant default full-pipeline advantage over simple baselines.",
            styles,
        )
    )
    story.append(
        body(
            "Architecturally, the project is stronger as a systems paper than as a pure leaderboard paper. Its main strengths are explicit source-credibility modeling, persistent evidence workflows, caching, fallback behavior, and a working full-stack deployment.",
            styles,
        )
    )

    story.append(heading("Architecture", styles))
    story.append(
        body(
            "Fact Validator uses a nine-stage pipeline: content extraction, claim decomposition, web retrieval, domain credibility scoring, semantic reranking, verdict inference, optional debate arbitration, sentiment and bias analysis, and final misinformation scoring with persistence.",
            styles,
        )
    )
    story.append(
        body(
            "The client layer is implemented in Next.js, the orchestration layer in FastAPI, and the persistence layer through SQLite and JSON caches. This separation makes the verification flow inspectable and repeatable, which is one of the main publication strengths of the project.",
            styles,
        )
    )

    story.append(heading("Main Results", styles))
    main_table = Table(
        [
            ["System", "Accuracy", "95% CI", "Avg. Conf.", "Calib. Err."],
            ["ablate_debate", "0.5134", "[0.4995, 0.5273]", "88.37", "0.3703"],
            ["tune_fever", "0.5098", "[0.4959, 0.5237]", "88.53", "0.3755"],
            ["full_proxy", "0.5082", "[0.4943, 0.5221]", "88.31", "0.3749"],
            ["length", "0.4942", "[0.4803, 0.5081]", "48.83", "0.0059"],
            ["majority", "0.4926", "[0.4787, 0.5065]", "50.00", "0.0074"],
            ["random", "0.3320", "[0.3189, 0.3451]", "50.00", "0.1680"],
        ],
        repeatRows=1,
        colWidths=[1.55 * inch, 0.8 * inch, 1.35 * inch, 0.9 * inch, 0.95 * inch],
    )
    main_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16324f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#eef3f8")]),
                ("FONTSIZE", (0, 0), (-1, -1), 8.8),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(main_table)
    story.append(Spacer(1, 0.12 * inch))
    story.append(
        body(
            "In the strict-validation snapshot, the default full pipeline beats majority by +1.56 percentage points and length by +1.40 percentage points. The best tuned variant reaches 50.98% accuracy, while the highest-accuracy frozen ablation indicates that debate should be selectively triggered rather than always enabled.",
            styles,
        )
    )

    story.append(heading("Comparison to the Reference Papers", styles))
    compare_table = Table(
        [
            ["Dimension", "Zero-shot Factual Correction", "FacTool", "Fact Validator"],
            ["Primary scope", "faithful error correction", "tool-augmented factuality", "misinformation risk and claim verification"],
            ["Trust modeling", "implicit via correction process", "tool selection and evidence", "explicit source-credibility prior"],
            ["Deployment surface", "research method", "research framework", "full-stack application and stored runs"],
            ["Operational metrics", "not central", "not central", "latency, throughput, cost, cache savings"],
        ],
        repeatRows=1,
        colWidths=[1.45 * inch, 1.55 * inch, 1.35 * inch, 1.85 * inch],
    )
    compare_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16324f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#eef3f8")]),
                ("FONTSIZE", (0, 0), (-1, -1), 8.4),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(compare_table)
    story.append(Spacer(1, 0.12 * inch))
    story.append(
        body(
            "The strongest honest claim is that Fact Validator is better positioned for practical deployment, source-trust transparency, and operational accountability. It is not yet proven better than the reference papers on their native tasks or benchmarks.",
            styles,
        )
    )

    story.append(heading("Novelty", styles))
    story.append(
        body(
            "The research novelty comes from combining auditable source credibility, live evidence retrieval, optional debate inside a deterministic pipeline, persistent evidence storage, and versioned large-scale benchmarking. This combination is the main reason the project is publication-capable.",
            styles,
        )
    )

    story.append(heading("Venue Fit", styles))
    story.append(
        body(
            "The strongest fit for the current evidence is an applied NLP, AI systems, or fact-checking workshop submission. The benchmark advantage is modest, but the transparency, deployment realism, and cost-aware evaluation make the work stronger as a systems paper than as a pure leaderboard paper.",
            styles,
        )
    )

    story.append(heading("Limitations", styles))
    story.append(
        body(
            "The full pipeline is still overconfident relative to its observed accuracy, the debate layer does not help when enabled by default in the frozen benchmark, and the health-domain slot uses a normalized substitute source that must be documented explicitly.",
            styles,
        )
    )

    story.append(PageBreak())
    story.append(heading("Conclusion", styles))
    story.append(
        body(
            "Fact Validator should be framed as a transparent systems contribution for practical fact checking. Its strongest result family supports a modest but positive comparative story, while its more important value lies in architecture, explainability, reproducibility, and cost-aware deployment.",
            styles,
        )
    )

    doc.build(story)


if __name__ == "__main__":
    build_pdf()