"use client";

import { useEffect, useMemo, useState } from "react";
import { cx } from "../components/ui";

type EvidenceItem = {
  url: string;
  snippet?: string;
  domain?: string;
  domain_score?: number;
  score?: number;
  title?: string;
  quality_score?: number;
  stance?: "support" | "refute" | "neutral" | string;
  source_type?: string;
  primary_source?: boolean;
  primary_source_reason?: string;
  published_year?: number;
  recency_score?: number;
  directness_score?: number;
  quote_grounded?: boolean;
  expertise_match?: number;
  numeric_match?: boolean;
  entity_match?: boolean;
  manipulation_flags?: string[];
};

type ClaimItem = {
  claim_text: string;
  verdict: "SUPPORTED" | "REFUTED" | "NEI" | string;
  confidence: number;
  debate_summary?: string;
  evidence: EvidenceItem[];
  structured_verdict?: string;
  uncertainty_reasons?: string[];
  needs_human_review?: boolean;
  human_review_reason?: string;
  claim_profile?: {
    atomic_claims?: string[];
    entities?: string[];
    numbers?: string[];
    years?: number[];
    expertise_profile?: string;
    loaded_language_terms?: string[];
  };
  evidence_summary?: {
    evidence_count?: number;
    high_credibility_sources?: number;
    primary_source_count?: number;
    primary_source_present?: boolean;
    supporting_items?: number;
    refuting_items?: number;
    conflict_level?: string;
    distinct_domains?: number;
    oldest_citation_year?: number;
    newest_citation_year?: number;
    average_quality_score?: number;
  };
};

type AnalyzeResponse = {
  input_type: "url" | "text" | string;
  domain?: string;
  extracted_text_chars?: number;
  extracted_text_preview?: string;
  domain_score?: number;
  domain_label?: string;
  final_misinformation_likelihood?: number;
  claims?: ClaimItem[];
  timestamp_utc?: string;
  metadata?: Record<string, unknown>;
};

type RunRow = {
  id: number | string;
  time_utc?: string;
  timestamp_utc?: string;
  input_type?: string;
  type?: string;
  domain?: string;
  url?: string;
};

function fmtPct(x?: number) {
  if (typeof x !== "number") return "-";
  const v = Math.max(0, Math.min(1, x));
  return `${Math.round(v * 100)}%`;
}

function safeHostFromUrl(u?: string) {
  try {
    if (!u) return "-";
    return new URL(u).hostname;
  } catch {
    return "-";
  }
}

function evScore(ev: EvidenceItem): number {
  if (typeof ev.domain_score === "number") return ev.domain_score;
  if (typeof ev.score === "number") return ev.score;
  return 0;
}

function evQuality(ev: EvidenceItem): number {
  if (typeof ev.quality_score === "number") return ev.quality_score;
  return evScore(ev);
}

function sortEvidence(evs: EvidenceItem[]) {
  return [...(evs || [])].sort((a, b) => {
    const qa = evQuality(a);
    const qb = evQuality(b);
    if (qb !== qa) return qb - qa;
    const sa = evScore(a);
    const sb = evScore(b);
    if (sb !== sa) return sb - sa;
    return (a.domain ?? "").localeCompare(b.domain ?? "");
  });
}

function sentimentColor(v?: string) {
  const verdict = (v || "").toUpperCase();
  if (verdict === "SUPPORTED") return "text-emerald-300 border-emerald-300/35 bg-emerald-400/10";
  if (verdict === "REFUTED") return "text-rose-300 border-rose-300/35 bg-rose-400/10";
  return "text-amber-200 border-amber-300/30 bg-amber-300/10";
}

function staggerStyle(index: number, stepMs = 70) {
  return { animationDelay: `${index * stepMs}ms` };
}

export default function Page() {
  const API_BASE = "http://127.0.0.1:8000";

  const [tab, setTab] = useState<"url" | "text">("url");
  const [url, setUrl] = useState("");
  const [text, setText] = useState("");

  const [mode, setMode] = useState<"live" | "snapshot">("live");
  const [verifier, setVerifier] = useState<"baseline" | "debate">("baseline");

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [maxClaims, setMaxClaims] = useState(6);
  const [maxEvidence, setMaxEvidence] = useState(5);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);

  const [runs, setRuns] = useState<RunRow[]>([]);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [runsLoading, setRunsLoading] = useState(false);

  const [openClaimIdx, setOpenClaimIdx] = useState<number | null>(0);

  const canAnalyze = useMemo(() => {
    if (tab === "url") return url.trim().length > 8;
    return text.trim().length > 10;
  }, [tab, url, text]);

  async function analyze() {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const payload: {
        mode: "live" | "snapshot";
        verifier: "baseline" | "debate";
        max_claims: number;
        max_evidence_per_claim: number;
        url?: string;
        text?: string;
      } = {
        mode,
        verifier,
        max_claims: maxClaims,
        max_evidence_per_claim: maxEvidence,
      };
      if (tab === "url") payload.url = url.trim();
      else payload.text = text.trim();

      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const t = await res.text();
        throw new Error(`API error ${res.status}: ${t}`);
      }

      const json = (await res.json()) as AnalyzeResponse;
      if (json.claims) {
        json.claims = json.claims.map((c) => ({ ...c, evidence: sortEvidence(c.evidence || []) }));
      }
      setResult(json);
      setOpenClaimIdx(0);
      fetchRuns();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to fetch");
    } finally {
      setLoading(false);
    }
  }

  async function fetchRuns() {
    setRunsLoading(true);
    setRunsError(null);

    try {
      const res = await fetch(`${API_BASE}/runs?limit=10`);
      if (!res.ok) {
        const t = await res.text();
        throw new Error(`Runs API error ${res.status}: ${t}`);
      }
      const json = await res.json();
      const items: RunRow[] = Array.isArray(json) ? json : (json?.items ?? []);
      setRuns(items);
    } catch (e: unknown) {
      setRuns([]);
      setRunsError(e instanceof Error ? e.message : "Could not load run history.");
    } finally {
      setRunsLoading(false);
    }
  }

  useEffect(() => {
    fetchRuns();
  }, []);

  const examples = [
    "https://ourworldindata.org/co2-and-greenhouse-gas-emissions",
    "https://www.bbc.com/news",
    "https://en.wikipedia.org/wiki/Naples",
    "https://www.who.int",
  ];

  const showResultSkeleton = loading && !result;

  return (
    <main className="min-h-screen text-slate-100 relative overflow-hidden">
      <div className="pointer-events-none absolute inset-0 grid-overlay opacity-30" />
      <div className="pointer-events-none absolute -top-24 left-[-5rem] h-80 w-80 rounded-full bg-cyan-400/20 blur-3xl" />
      <div className="pointer-events-none absolute top-10 right-0 h-96 w-96 rounded-full bg-blue-500/20 blur-3xl" />
      <div className="pointer-events-none absolute bottom-0 left-1/3 h-80 w-80 rounded-full bg-emerald-400/10 blur-3xl" />

      <div className="relative mx-auto max-w-7xl px-4 py-8 md:py-10 section-fade-in">
        <header className="glass-panel rounded-3xl p-5 md:p-7 border mb-6">
          <div className="flex items-start justify-between gap-4 flex-wrap">
            <div>
              <div className="inline-flex items-center gap-2 rounded-full border border-cyan-300/30 bg-cyan-400/10 px-3 py-1 text-[11px] uppercase tracking-wider text-cyan-200">
                <span className="h-2 w-2 rounded-full bg-cyan-300 shadow-[0_0_12px_#42d9ff]" />
                AI Verification Console
              </div>
              <h1 className="mt-4 text-3xl md:text-4xl font-semibold leading-tight text-white">
                Fact Validator AI
              </h1>
              <p className="mt-2 max-w-2xl text-sm md:text-base text-slate-300">
                Professional-grade claim intelligence with credibility scoring, counter-evidence analysis, uncertainty disclosure, and explainable verdicts.
              </p>
            </div>
            <div className="grid grid-cols-2 gap-3 text-sm w-full sm:w-auto">
              <a href="/source" className="glass-panel rounded-xl px-4 py-3 text-slate-100 hover:border-cyan-300/40 transition border">Source Checker</a>
              <a href={`${API_BASE}/docs`} target="_blank" rel="noreferrer" className="glass-panel rounded-xl px-4 py-3 text-slate-100 hover:border-cyan-300/40 transition border">API Docs</a>
              <a href={`${API_BASE}/evaluation/benchmark`} target="_blank" rel="noreferrer" className="glass-panel rounded-xl px-4 py-3 text-slate-100 hover:border-cyan-300/40 transition border col-span-2">Benchmark Dataset</a>
            </div>
          </div>
        </header>

        <section className="grid xl:grid-cols-[1.05fr_1.4fr] gap-6">
          <div className="glass-panel rounded-3xl border p-5 md:p-6 section-fade-in">
            <div className="flex items-center justify-between mb-4">
              <div className="text-sm uppercase tracking-wider text-slate-300">Input</div>
              <button className="text-xs px-3 py-1.5 rounded-lg border border-slate-300/20 hover:border-cyan-300/40" onClick={() => setShowAdvanced((v) => !v)}>
                {showAdvanced ? "Hide advanced" : "Advanced"}
              </button>
            </div>

            <div className="grid grid-cols-2 gap-2 mb-4">
              <button className={cx("rounded-xl py-2.5 text-sm border transition", tab === "url" ? "bg-cyan-400/15 border-cyan-300/40 text-cyan-100" : "border-slate-300/20 hover:border-slate-300/40")} onClick={() => setTab("url")}>URL analysis</button>
              <button className={cx("rounded-xl py-2.5 text-sm border transition", tab === "text" ? "bg-cyan-400/15 border-cyan-300/40 text-cyan-100" : "border-slate-300/20 hover:border-slate-300/40")} onClick={() => setTab("text")}>Text analysis</button>
            </div>

            <div className="grid sm:grid-cols-2 gap-3">
              <label className="text-xs text-slate-300">Mode
                <select className="mt-1 w-full rounded-xl border border-slate-300/20 bg-slate-900/60 px-3 py-2 text-sm" value={mode} onChange={(e) => setMode(e.target.value as "live" | "snapshot")}>
                  <option value="live">Live</option>
                  <option value="snapshot">Snapshot</option>
                </select>
              </label>
              <label className="text-xs text-slate-300">Verifier
                <select className="mt-1 w-full rounded-xl border border-slate-300/20 bg-slate-900/60 px-3 py-2 text-sm" value={verifier} onChange={(e) => setVerifier(e.target.value as "baseline" | "debate")}>
                  <option value="baseline">Baseline</option>
                  <option value="debate">Debate</option>
                </select>
              </label>
            </div>

            {showAdvanced && (
              <div className="grid sm:grid-cols-2 gap-3 mt-3">
                <label className="text-xs text-slate-300">Max claims
                  <input type="number" min={1} max={12} className="mt-1 w-full rounded-xl border border-slate-300/20 bg-slate-900/60 px-3 py-2 text-sm" value={maxClaims} onChange={(e) => setMaxClaims(parseInt(e.target.value || "6", 10))} />
                </label>
                <label className="text-xs text-slate-300">Evidence per claim
                  <input type="number" min={1} max={10} className="mt-1 w-full rounded-xl border border-slate-300/20 bg-slate-900/60 px-3 py-2 text-sm" value={maxEvidence} onChange={(e) => setMaxEvidence(parseInt(e.target.value || "5", 10))} />
                </label>
              </div>
            )}

            {tab === "url" ? (
              <div className="mt-3">
                <label className="text-xs text-slate-300">Target URL</label>
                <input className="mt-1 w-full rounded-xl border border-slate-300/20 bg-slate-900/60 px-3 py-3 text-sm" placeholder="https://example.com/article" value={url} onChange={(e) => setUrl(e.target.value)} />
                <div className="mt-2 flex flex-wrap gap-2">
                  {examples.map((x) => (
                    <button key={x} className="rounded-full border border-slate-300/20 px-3 py-1 text-xs text-slate-200 hover:border-cyan-300/40" onClick={() => setUrl(x)}>
                      Use sample
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="mt-3">
                <label className="text-xs text-slate-300">Claim or paragraph</label>
                <textarea className="mt-1 w-full min-h-[140px] rounded-xl border border-slate-300/20 bg-slate-900/60 px-3 py-3 text-sm" placeholder="Paste a claim to verify" value={text} onChange={(e) => setText(e.target.value)} />
              </div>
            )}

            <button className={cx("mt-4 w-full rounded-xl px-4 py-3 text-sm font-semibold transition", !canAnalyze || loading ? "bg-slate-700/60 text-slate-400 cursor-not-allowed" : "bg-gradient-to-r from-cyan-300 to-blue-400 text-slate-950 hover:brightness-110")} onClick={analyze} disabled={!canAnalyze || loading}>
              {loading ? "Running AI verification..." : "Run verification"}
            </button>

            {error && (
              <div className="mt-4 rounded-xl border border-rose-300/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-100">
                {error}
              </div>
            )}
          </div>

          <div className="grid gap-6 section-fade-in">
            <div className="grid md:grid-cols-3 gap-4">
              <div className="glass-panel rounded-2xl border p-4 stagger-card hover-lift" style={staggerStyle(0)}>
                <div className="text-xs uppercase tracking-wide text-slate-300">Input domain</div>
                <div className="mt-2 font-mono text-lg text-white">{result?.domain ?? "-"}</div>
                <div className="mt-2 text-xs text-slate-400">{result?.timestamp_utc ?? "Ready"}</div>
              </div>
              <div className="glass-panel rounded-2xl border p-4 stagger-card hover-lift" style={staggerStyle(1)}>
                <div className="text-xs uppercase tracking-wide text-slate-300">Credibility score</div>
                <div className="mt-2 text-2xl font-semibold text-white">{typeof result?.domain_score === "number" ? result.domain_score : "-"}</div>
                <div className="text-xs text-slate-300">{result?.domain_label ?? "No label"}</div>
              </div>
              <div className="glass-panel rounded-2xl border p-4 stagger-card hover-lift" style={staggerStyle(2)}>
                <div className="text-xs uppercase tracking-wide text-slate-300">Misinformation risk</div>
                <div className="mt-2 text-2xl font-semibold text-white">{fmtPct(result?.final_misinformation_likelihood)}</div>
                <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-700/80">
                  <div className="h-2 meter-fill bg-gradient-to-r from-emerald-300 via-amber-300 to-rose-400" style={{ width: typeof result?.final_misinformation_likelihood === "number" ? `${Math.round(result.final_misinformation_likelihood * 100)}%` : "0%" }} />
                </div>
              </div>
            </div>

            <div className="glass-panel rounded-2xl border p-4 md:p-5 stagger-card hover-lift" style={staggerStyle(3)}>
              <div className="text-sm font-semibold text-white">Extracted context</div>
              <div className="mt-1 text-xs text-slate-400">Characters: {result?.extracted_text_chars ?? 0}</div>
              <p className="mt-3 text-sm leading-relaxed text-slate-200">{result?.extracted_text_preview ?? "Run a verification to see extraction preview."}</p>
            </div>

            <div className="glass-panel rounded-2xl border p-4 md:p-5 stagger-card" style={staggerStyle(4)}>
              <div className="flex items-center justify-between">
                <h2 className="text-base font-semibold text-white">Claim intelligence</h2>
                <span className="text-xs text-slate-300">{(result?.claims || []).length} claims</span>
              </div>

              <div className="mt-4 grid gap-3">
                {showResultSkeleton && (
                  <div className="grid gap-3">
                    {Array.from({ length: 3 }).map((_, i) => (
                      <div key={`claim-skeleton-${i}`} className="rounded-xl border border-slate-300/20 bg-slate-900/45 p-4">
                        <div className="skeleton h-3 w-24 rounded" />
                        <div className="skeleton mt-3 h-4 w-full rounded" />
                        <div className="skeleton mt-2 h-4 w-11/12 rounded" />
                        <div className="mt-3 flex gap-2">
                          <div className="skeleton h-6 w-28 rounded-full" />
                          <div className="skeleton h-6 w-36 rounded-full" />
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {(result?.claims || []).map((c, idx) => {
                  const open = openClaimIdx === idx;
                  return (
                    <article key={idx} className="rounded-xl border border-slate-300/20 bg-slate-900/45 stagger-card hover-lift" style={staggerStyle(idx, 64)}>
                      <button className="w-full p-4 text-left" onClick={() => setOpenClaimIdx(open ? null : idx)}>
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <div className="text-xs uppercase tracking-wider text-slate-400">Claim #{idx + 1}</div>
                            <div className="mt-1 text-sm text-white leading-relaxed">{c.claim_text}</div>
                            <div className="mt-2 flex flex-wrap items-center gap-2">
                              <span className={cx("rounded-full border px-3 py-1 text-xs", sentimentColor(c.verdict))}>{(c.verdict || "NEI").toUpperCase()} {fmtPct(c.confidence)}</span>
                              {c.structured_verdict && <span className="rounded-full border border-cyan-300/30 bg-cyan-400/10 px-3 py-1 text-xs text-cyan-100">{c.structured_verdict}</span>}
                              {c.needs_human_review && <span className="rounded-full border border-amber-300/30 bg-amber-300/10 px-3 py-1 text-xs text-amber-100">Human review</span>}
                            </div>
                          </div>
                          <div className="text-slate-400 text-xs">{open ? "Hide" : "Details"}</div>
                        </div>
                      </button>

                      {open && (
                        <div className="border-t border-slate-300/20 p-4">
                          {c.debate_summary && <p className="text-sm text-slate-200">{c.debate_summary}</p>}

                          <div className="mt-3 grid md:grid-cols-2 gap-3 text-xs text-slate-300">
                            <div className="rounded-lg border border-slate-300/20 bg-slate-800/40 p-3">
                              <div className="font-semibold text-slate-100">Claim profile</div>
                              <div className="mt-2">Expertise: {c.claim_profile?.expertise_profile ?? "general"}</div>
                              <div>Entities: {(c.claim_profile?.entities || []).join(", ") || "-"}</div>
                              <div>Numbers: {(c.claim_profile?.numbers || []).join(", ") || "-"}</div>
                            </div>
                            <div className="rounded-lg border border-slate-300/20 bg-slate-800/40 p-3">
                              <div className="font-semibold text-slate-100">Trust diagnostics</div>
                              <div className="mt-2">High credibility: {c.evidence_summary?.high_credibility_sources ?? 0}</div>
                              <div>Primary sources: {c.evidence_summary?.primary_source_count ?? 0}</div>
                              <div>Conflict level: {c.evidence_summary?.conflict_level ?? "low"}</div>
                              <div>Average quality: {c.evidence_summary?.average_quality_score ?? "-"}</div>
                            </div>
                          </div>

                          {!!(c.uncertainty_reasons && c.uncertainty_reasons.length) && (
                            <div className="mt-3 rounded-lg border border-amber-300/30 bg-amber-300/10 p-3 text-xs text-amber-100">
                              <div className="font-semibold">Uncertainty signals</div>
                              <ul className="mt-1 list-disc pl-5 space-y-1">
                                {c.uncertainty_reasons?.map((reason, i) => <li key={i}>{reason}</li>)}
                              </ul>
                            </div>
                          )}

                          <div className="mt-4 grid gap-2">
                            {sortEvidence(c.evidence || []).map((ev, j) => {
                              const score = evScore(ev);
                              return (
                                <div key={j} className="rounded-lg border border-slate-300/20 bg-slate-900/60 p-3 stagger-card hover-lift" style={staggerStyle(j, 42)}>
                                  <div className="flex items-center justify-between gap-3 flex-wrap">
                                    <div className="flex flex-wrap items-center gap-2">
                                      <span className="text-sm font-semibold text-slate-100">{ev.domain ?? safeHostFromUrl(ev.url)}</span>
                                      <span className="rounded-full border border-slate-300/25 px-2 py-0.5 text-[11px] text-slate-200">Score {score}</span>
                                      {typeof ev.quality_score === "number" && <span className="rounded-full border border-cyan-300/30 px-2 py-0.5 text-[11px] text-cyan-100">Q {Math.round(ev.quality_score)}</span>}
                                      {ev.primary_source && <span className="rounded-full border border-emerald-300/30 px-2 py-0.5 text-[11px] text-emerald-100">Primary</span>}
                                    </div>
                                    <a href={ev.url} target="_blank" rel="noreferrer" className="rounded-md border border-slate-300/25 px-2 py-1 text-xs text-slate-200 hover:border-cyan-300/40">Open</a>
                                  </div>
                                  {ev.snippet && <p className="mt-2 text-sm text-slate-200 leading-relaxed">{ev.snippet}</p>}
                                  <div className="mt-2 grid sm:grid-cols-3 gap-2 text-[11px] text-slate-400">
                                    <span>Type: {ev.source_type ?? "news"}</span>
                                    <span>Recency: {typeof ev.recency_score === "number" ? `${Math.round(ev.recency_score * 100)}%` : "-"}</span>
                                    <span>Directness: {typeof ev.directness_score === "number" ? `${Math.round(ev.directness_score * 100)}%` : "-"}</span>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}
                    </article>
                  );
                })}
              </div>
            </div>
          </div>
        </section>

        <section className="glass-panel rounded-2xl border p-5 mt-6 section-fade-in">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <h3 className="text-sm font-semibold text-white">Recent verification runs</h3>
            <button className="rounded-lg border border-slate-300/20 px-3 py-1.5 text-xs hover:border-cyan-300/40" onClick={fetchRuns} disabled={runsLoading}>
              {runsLoading ? "Refreshing..." : "Refresh"}
            </button>
          </div>

          {runsLoading && runs.length === 0 ? (
            <div className="mt-3 grid gap-2">
              {Array.from({ length: 4 }).map((_, i) => (
                <div key={`run-skeleton-${i}`} className="skeleton h-9 rounded-lg" />
              ))}
            </div>
          ) : runsError ? (
            <div className="mt-3 text-sm text-rose-200">{runsError}</div>
          ) : runs.length === 0 ? (
            <div className="mt-3 text-sm text-slate-300">No runs yet.</div>
          ) : (
            <div className="mt-3 overflow-auto rounded-xl border border-slate-300/20">
              <table className="min-w-full text-sm">
                <thead className="bg-slate-900/70 text-slate-300">
                  <tr>
                    <th className="px-3 py-2 text-left">ID</th>
                    <th className="px-3 py-2 text-left">Time</th>
                    <th className="px-3 py-2 text-left">Type</th>
                    <th className="px-3 py-2 text-left">Domain</th>
                    <th className="px-3 py-2 text-left">URL</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((r, i) => (
                    <tr key={`${r.id}_${i}`} className="border-t border-slate-300/15">
                      <td className="px-3 py-2 text-slate-100">{r.id}</td>
                      <td className="px-3 py-2 font-mono text-xs text-slate-300">{r.time_utc ?? r.timestamp_utc ?? "-"}</td>
                      <td className="px-3 py-2 text-slate-200">{r.input_type ?? r.type ?? "-"}</td>
                      <td className="px-3 py-2 text-slate-200">{r.domain ?? safeHostFromUrl(r.url)}</td>
                      <td className="px-3 py-2">{r.url ? <a className="rounded-md border border-slate-300/20 px-2 py-1 text-xs text-slate-200 hover:border-cyan-300/40" href={r.url} target="_blank" rel="noreferrer">Open</a> : "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
