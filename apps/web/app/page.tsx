"use client";

import { useEffect, useMemo, useState } from "react";

type RunListItem = {
  id: number;
  created_utc: string;
  input_type: string;
  input_url?: string | null;
  input_domain?: string | null;
  extracted_text_chars?: number;
};

function downloadJson(filename: string, obj: any) {
  const json = JSON.stringify(obj, null, 2);
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export default function Home() {
  const API_BASE = useMemo(() => "http://127.0.0.1:8000", []);

  const [url, setUrl] = useState("");
  const [text, setText] = useState("");
  const [mode, setMode] = useState<"live" | "snapshot">("live");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [result, setResult] = useState<any>(null);

  // History
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [historyOpen, setHistoryOpen] = useState(true);
  const [historyLoading, setHistoryLoading] = useState(false);

  async function analyze() {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const payload: any = { mode };
      if (url.trim().length > 0) payload.url = url.trim();
      if (text.trim().length > 0) payload.text = text.trim();

      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data = await res.json();
      setResult(data);

      // If snapshot, refresh history automatically
      if (mode === "snapshot") {
        await loadRuns();
      }
    } catch (e: any) {
      setError(e?.message ?? "Failed to fetch");
    } finally {
      setLoading(false);
    }
  }

  async function loadRuns() {
    setHistoryLoading(true);
    try {
      const res = await fetch(`${API_BASE}/runs?limit=25`);
      if (!res.ok) throw new Error(`Runs API error: ${res.status}`);
      const data = await res.json();
      setRuns(Array.isArray(data) ? data : []);
    } catch {
      setRuns([]);
    } finally {
      setHistoryLoading(false);
    }
  }

  async function openRun(runId: number) {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/runs/${runId}`);
      if (!res.ok) throw new Error(`Run load error: ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (e: any) {
      setError(e?.message ?? "Failed to load run");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadRuns();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const runId = result?.metadata?.run_id as number | undefined;
  const sharePath = runId ? `/run/${runId}` : null;

  return (
    <main className="min-h-screen p-8 max-w-6xl mx-auto">
      {/* Print helpers */}
      <style jsx global>{`
        @media print {
          .no-print {
            display: none !important;
          }
        }
      `}</style>

      <h1 className="text-2xl font-semibold">Fact Validator</h1>
      <p className="text-sm text-gray-600 mt-1">
        Live analysis + Snapshot run history. Use Snapshot mode for thesis demos so you can reopen reports later.
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
        {/* Left: Analyzer */}
        <section className="lg:col-span-2">
          <div className="p-4 border rounded no-print">
            <label className="block text-sm font-medium">URL (optional)</label>
            <input
              className="w-full mt-2 p-2 border rounded"
              placeholder="https://example.com/article"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
            />

            <label className="block text-sm font-medium mt-4">Text (optional)</label>
            <textarea
              className="w-full mt-2 p-2 border rounded min-h-[120px]"
              placeholder="Paste a claim or article text..."
              value={text}
              onChange={(e) => setText(e.target.value)}
            />

            <div className="flex items-center gap-3 mt-4 flex-wrap">
              <label className="text-sm font-medium">Mode</label>
              <select
                className="p-2 border rounded"
                value={mode}
                onChange={(e) => setMode(e.target.value as any)}
              >
                <option value="live">live</option>
                <option value="snapshot">snapshot</option>
              </select>

              <button
                className="ml-auto px-4 py-2 rounded bg-black text-white disabled:opacity-50"
                disabled={loading || (url.trim().length === 0 && text.trim().length === 0)}
                onClick={analyze}
              >
                {loading ? "Analyzing..." : "Analyze"}
              </button>
            </div>

            {error && (
              <div className="mt-4 p-3 rounded border border-red-300 bg-red-50 text-red-700">
                {error}
              </div>
            )}
          </div>

          {/* Report */}
          {result && (
            <div className="mt-5 p-4 border rounded bg-white">
              <div className="flex items-start justify-between gap-4 flex-wrap">
                <div className="flex flex-wrap gap-6 text-sm">
                  <div>
                    <div className="text-gray-500">Input Type</div>
                    <div className="font-medium">{result.input_type}</div>
                  </div>
                  <div>
                    <div className="text-gray-500">Domain</div>
                    <div className="font-medium">{result.domain ?? "-"}</div>
                  </div>
                  <div>
                    <div className="text-gray-500">Extracted Text Chars</div>
                    <div className="font-medium">{result.extracted_text_chars}</div>
                  </div>
                  <div>
                    <div className="text-gray-500">Domain Score</div>
                    <div className="font-medium">
                      {result.domain_score} ({result.domain_label})
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-500">Misinformation Likelihood</div>
                    <div className="font-medium">{result.final_misinformation_likelihood}</div>
                  </div>
                  {runId && (
                    <div>
                      <div className="text-gray-500">Run ID</div>
                      <div className="font-medium">{runId}</div>
                    </div>
                  )}
                </div>

                {/* Thesis buttons */}
                <div className="no-print flex gap-2">
                  <button
                    className="px-3 py-2 border rounded text-sm"
                    onClick={() => window.print()}
                  >
                    Print / Save PDF
                  </button>
                  <button
                    className="px-3 py-2 border rounded text-sm"
                    onClick={() => downloadJson(`fact-validator-report-${runId ?? "live"}.json`, result)}
                  >
                    Download JSON
                  </button>
                  {sharePath && (
                    <a className="px-3 py-2 border rounded text-sm underline" href={sharePath}>
                      Open Share Page
                    </a>
                  )}
                </div>
              </div>

              {sharePath && (
                <div className="no-print mt-3 text-sm text-gray-700">
                  Shareable link (local):{" "}
                  <a className="underline" href={sharePath}>
                    {`http://localhost:3000${sharePath}`}
                  </a>
                </div>
              )}

              <div className="mt-4">
                <div className="text-gray-500 text-sm">Extracted Text Preview</div>
                <div className="text-sm mt-1">{result.extracted_text_preview}</div>
              </div>

              <div className="mt-6">
                <div className="text-gray-800 font-semibold">Claims</div>
                <div className="mt-2 space-y-3">
                  {(result.claims ?? []).map((c: any, idx: number) => (
                    <div key={idx} className="p-3 border rounded">
                      <div className="text-sm">{c.claim_text}</div>
                      <div className="mt-2 text-sm">
                        <span className="font-medium">{c.verdict}</span>{" "}
                        <span className="text-gray-500">({c.confidence})</span>
                      </div>

                      {c.debate_summary && (
                        <div className="mt-2 text-sm text-gray-700">
                          <span className="text-gray-500">Debate:</span> {c.debate_summary}
                        </div>
                      )}

                      <div className="mt-3 text-sm text-gray-800 font-medium">Evidence</div>
                      <ul className="mt-1 space-y-2">
                        {(c.evidence ?? []).map((e: any, j: number) => (
                          <li key={j} className="text-sm">
                            <div className="font-medium">
                              {e.domain} <span className="text-gray-500">(score: {e.domain_score})</span>
                            </div>
                            <div className="text-gray-700">{e.snippet}</div>
                            <div className="no-print">
                              <a className="text-blue-700 underline" href={e.url} target="_blank" rel="noreferrer">
                                Open source
                              </a>
                            </div>
                            <div className="hidden print:block text-xs text-gray-700">
                              Source URL: {e.url}
                            </div>
                          </li>
                        ))}
                        {(c.evidence ?? []).length === 0 && (
                          <li className="text-sm text-gray-500">No evidence retrieved.</li>
                        )}
                      </ul>
                    </div>
                  ))}
                </div>
              </div>

              <details className="mt-6 no-print">
                <summary className="cursor-pointer text-sm text-gray-700">Raw JSON (debug)</summary>
                <pre className="mt-2 text-xs overflow-auto p-3 border rounded bg-gray-50">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </details>
            </div>
          )}
        </section>

        {/* Right: History */}
        <aside className="lg:col-span-1">
          <div className="p-4 border rounded no-print">
            <div className="flex items-center justify-between">
              <div className="font-semibold">Run History</div>
              <div className="flex items-center gap-2">
                <button
                  className="text-sm underline"
                  onClick={() => loadRuns()}
                  disabled={historyLoading}
                >
                  {historyLoading ? "Loading..." : "Refresh"}
                </button>
                <button
                  className="text-sm underline"
                  onClick={() => setHistoryOpen(!historyOpen)}
                >
                  {historyOpen ? "Hide" : "Show"}
                </button>
              </div>
            </div>

            {historyOpen && (
              <div className="mt-3">
                {runs.length === 0 ? (
                  <div className="text-sm text-gray-500">
                    No saved runs yet. Use Snapshot mode and Analyze.
                  </div>
                ) : (
                  <ul className="space-y-2">
                    {runs.map((r) => (
                      <li key={r.id} className="p-2 border rounded">
                        <div className="text-sm font-medium">Run #{r.id}</div>
                        <div className="text-xs text-gray-600 mt-1">{r.created_utc}</div>
                        <div className="text-xs text-gray-600 mt-1">
                          {r.input_type === "url" ? (r.input_url ?? "-") : "text"}
                        </div>

                        <div className="mt-2 flex gap-3">
                          <button className="text-sm underline" onClick={() => openRun(r.id)}>
                            Open
                          </button>
                          <a className="text-sm underline" href={`/run/${r.id}`}>
                            Share Page
                          </a>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </div>
        </aside>
      </div>
    </main>
  );
}
