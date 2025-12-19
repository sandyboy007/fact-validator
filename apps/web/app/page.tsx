"use client";

import { useState } from "react";

type Verdict = "SUPPORTED" | "REFUTED" | "NEI";

export default function Home() {
  const [url, setUrl] = useState("");
  const [text, setText] = useState("");
  const [mode, setMode] = useState<"live" | "snapshot">("live");

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  async function analyze() {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url: url.trim() || null,
          text: text.trim() || null,
          mode,
        }),
      });

      if (!res.ok) {
        const msg = await res.text();
        throw new Error(`API ${res.status}: ${msg}`);
      }

      setResult(await res.json());
    } catch (e: any) {
      setError(e?.message ?? "Failed to fetch");
    } finally {
      setLoading(false);
    }
  }

  const claims = (result?.claims ?? []) as Array<{
    claim_text: string;
    verdict: Verdict;
    confidence: number;
    evidence: Array<{ url: string; snippet: string; domain: string; domain_score: number }>;
    debate_summary?: string | null;
  }>;

  const canAnalyze = url.trim().length > 0 || text.trim().length > 0;

  return (
    <main style={{ padding: 24, maxWidth: 980, margin: "0 auto", fontFamily: "Arial, sans-serif" }}>
      <h1 style={{ fontSize: 22, fontWeight: 700 }}>Fact Validator (Working Demo)</h1>
      <p style={{ marginTop: 8, opacity: 0.8 }}>
        Paste a URL or text and click Analyze. Right now the backend returns a mock report, but it should correctly
        detect the domain from your URL (Step 3.1).
      </p>

      {/* URL input */}
      <div style={{ marginTop: 16 }}>
        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>URL (optional)</div>
        <input
          style={{ width: "100%", padding: 12 }}
          placeholder="https://... or www.example.com"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
        />
      </div>

      {/* Text input */}
      <div style={{ marginTop: 16 }}>
        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>Text (optional)</div>
        <textarea
          style={{ width: "100%", padding: 12, minHeight: 140 }}
          placeholder="Paste a claim or article text..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
      </div>

      {/* Mode */}
      <div style={{ marginTop: 12, display: "flex", gap: 10, alignItems: "center" }}>
        <div style={{ fontSize: 13, fontWeight: 600 }}>Mode</div>
        <select value={mode} onChange={(e) => setMode(e.target.value as any)} style={{ padding: 8 }}>
          <option value="live">live</option>
          <option value="snapshot">snapshot</option>
        </select>

        <button
          style={{ marginLeft: "auto", padding: "10px 14px", cursor: "pointer" }}
          disabled={loading || !canAnalyze}
          onClick={analyze}
        >
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </div>

      {error && (
        <div style={{ marginTop: 12, color: "crimson" }}>
          <b>Error:</b> {error}
        </div>
      )}

      {result && (
        <div style={{ marginTop: 18 }}>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            <Card title="Input Type" value={String(result.input_type)} />
            <Card title="Domain" value={String(result.domain ?? "-")} />
            <Card title="Domain Score" value={`${result.domain_score} (${result.domain_label})`} />
            <Card title="Misinformation Likelihood" value={String(result.final_misinformation_likelihood)} />
          </div>

          <h2 style={{ marginTop: 18, fontSize: 18, fontWeight: 700 }}>Claims</h2>

          {claims.length === 0 ? (
            <p style={{ opacity: 0.8 }}>No claims returned.</p>
          ) : (
            <div style={{ marginTop: 10, display: "grid", gap: 10 }}>
              {claims.map((c, idx) => (
                <div key={idx} style={{ border: "1px solid #ddd", borderRadius: 8, padding: 12 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                    <div style={{ fontWeight: 600 }}>{c.claim_text}</div>
                    <div style={{ fontFamily: "monospace" }}>
                      {c.verdict} ({c.confidence})
                    </div>
                  </div>

                  {c.debate_summary && (
                    <div style={{ marginTop: 8, opacity: 0.9 }}>
                      <b>Debate:</b> {c.debate_summary}
                    </div>
                  )}

                  <div style={{ marginTop: 10 }}>
                    <b>Evidence</b>
                    <ul style={{ marginTop: 6 }}>
                      {c.evidence.map((e, i) => (
                        <li key={i} style={{ marginBottom: 6 }}>
                          <div>
                            <a href={e.url} target="_blank" rel="noreferrer">
                              {e.domain}
                            </a>{" "}
                            (score: {e.domain_score})
                          </div>
                          <div style={{ opacity: 0.85 }}>{e.snippet}</div>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              ))}
            </div>
          )}

          <details style={{ marginTop: 18 }}>
            <summary style={{ cursor: "pointer" }}>Raw JSON (debug)</summary>
            <pre style={{ marginTop: 8, padding: 12, background: "#f5f5f5", overflow: "auto" }}>
              {JSON.stringify(result, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </main>
  );
}

function Card({ title, value }: { title: string; value: string }) {
  return (
    <div style={{ border: "1px solid #ddd", borderRadius: 8, padding: 12, minWidth: 210 }}>
      <div style={{ fontSize: 12, opacity: 0.75 }}>{title}</div>
      <div style={{ marginTop: 6, fontWeight: 700 }}>{value}</div>
    </div>
  );
}
