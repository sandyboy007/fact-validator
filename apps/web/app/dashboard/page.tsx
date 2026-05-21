"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

type DomainCount = {
  domain: string;
  count: number;
};

type DashboardSummary = {
  limit: number;
  total_runs: number;
  last_run_utc: string | null;
  claims_analyzed: number;
  claims_requiring_human_review: number;
  avg_misinformation_likelihood: number | null;
  input_type_counts: Record<string, number>;
  verifier_counts: Record<string, number>;
  verdict_counts: Record<string, number>;
  top_domains: DomainCount[];
};

const API_BASE = "http://127.0.0.1:8000";

function formatPercent(value: number | null | undefined): string {
  if (typeof value !== "number") return "-";
  return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
}

function formatUtc(value: string | null | undefined): string {
  if (!value) return "-";
  return value.replace("T", " ").replace("Z", "").slice(0, 19);
}

function CountList({ data, emptyLabel }: { data: Record<string, number>; emptyLabel: string }) {
  const rows = Object.entries(data).sort((a, b) => b[1] - a[1]);
  if (!rows.length) {
    return <p className="text-sm text-slate-400">{emptyLabel}</p>;
  }

  return (
    <div className="space-y-2">
      {rows.map(([label, count]) => (
        <div key={label} className="flex items-center justify-between rounded-lg border border-slate-700/70 bg-slate-900/60 px-3 py-2">
          <span className="text-sm capitalize text-slate-300">{label.replace(/_/g, " ")}</span>
          <span className="rounded bg-cyan-500/15 px-2 py-0.5 text-xs font-semibold text-cyan-200">{count}</span>
        </div>
      ))}
    </div>
  );
}

export default function DashboardPage() {
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadSummary = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/dashboard/summary?limit=250`, {
        method: "GET",
      });
      if (!res.ok) {
        throw new Error(`Request failed (${res.status})`);
      }
      const payload = (await res.json()) as DashboardSummary;
      setSummary(payload);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unable to load dashboard summary";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadSummary();
  }, [loadSummary]);

  const verdictTotal = useMemo(() => {
    if (!summary) return 0;
    return Object.values(summary.verdict_counts || {}).reduce((acc, value) => acc + value, 0);
  }, [summary]);

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-6xl space-y-6 px-4 py-8 md:py-12">
        <header className="rounded-2xl border border-slate-700/70 bg-slate-900/70 p-5 md:p-7">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-cyan-200">Fact Validator</p>
              <h1 className="mt-2 text-2xl font-semibold text-slate-100 md:text-3xl">Backend + Frontend Dashboard</h1>
              <p className="mt-2 max-w-2xl text-sm text-slate-300">
                This page is served by Next.js and reads summary analytics from the new FastAPI endpoint
                <span className="ml-1 font-mono text-cyan-200">/dashboard/summary</span>.
              </p>
            </div>
            <div className="flex gap-2">
              <Link href="/" className="rounded-lg border border-slate-600 px-3 py-2 text-sm text-slate-200 hover:border-cyan-300/60">
                Main App
              </Link>
              <button
                type="button"
                onClick={loadSummary}
                className="rounded-lg border border-cyan-400/60 bg-cyan-400/10 px-3 py-2 text-sm text-cyan-100 hover:bg-cyan-400/20"
              >
                Refresh
              </button>
            </div>
          </div>
        </header>

        {loading && (
          <section className="rounded-2xl border border-slate-700/70 bg-slate-900/70 p-6 text-sm text-slate-300">
            Loading dashboard summary...
          </section>
        )}

        {error && (
          <section className="rounded-2xl border border-rose-500/40 bg-rose-500/10 p-6 text-sm text-rose-100">
            Failed to load backend summary: {error}
          </section>
        )}

        {!loading && !error && summary && (
          <>
            <section className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
              <article className="rounded-xl border border-slate-700/70 bg-slate-900/70 p-4">
                <p className="text-xs uppercase tracking-wider text-slate-400">Runs</p>
                <p className="mt-2 text-2xl font-semibold">{summary.total_runs.toLocaleString()}</p>
                <p className="mt-1 text-xs text-slate-400">Last run: {formatUtc(summary.last_run_utc)}</p>
              </article>
              <article className="rounded-xl border border-slate-700/70 bg-slate-900/70 p-4">
                <p className="text-xs uppercase tracking-wider text-slate-400">Claims analyzed</p>
                <p className="mt-2 text-2xl font-semibold">{summary.claims_analyzed.toLocaleString()}</p>
                <p className="mt-1 text-xs text-slate-400">Human review: {summary.claims_requiring_human_review.toLocaleString()}</p>
              </article>
              <article className="rounded-xl border border-slate-700/70 bg-slate-900/70 p-4">
                <p className="text-xs uppercase tracking-wider text-slate-400">Avg misinformation</p>
                <p className="mt-2 text-2xl font-semibold text-amber-200">{formatPercent(summary.avg_misinformation_likelihood)}</p>
                <p className="mt-1 text-xs text-slate-400">Computed across sampled runs</p>
              </article>
              <article className="rounded-xl border border-slate-700/70 bg-slate-900/70 p-4">
                <p className="text-xs uppercase tracking-wider text-slate-400">Claim verdicts</p>
                <p className="mt-2 text-2xl font-semibold">{verdictTotal.toLocaleString()}</p>
                <p className="mt-1 text-xs text-slate-400">From {summary.limit} most recent runs</p>
              </article>
            </section>

            <section className="grid gap-4 lg:grid-cols-3">
              <article className="rounded-xl border border-slate-700/70 bg-slate-900/70 p-4">
                <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-300">Input Types</h2>
                <CountList data={summary.input_type_counts || {}} emptyLabel="No input-type stats yet." />
              </article>
              <article className="rounded-xl border border-slate-700/70 bg-slate-900/70 p-4">
                <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-300">Verifiers</h2>
                <CountList data={summary.verifier_counts || {}} emptyLabel="No verifier stats yet." />
              </article>
              <article className="rounded-xl border border-slate-700/70 bg-slate-900/70 p-4">
                <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-300">Verdicts</h2>
                <CountList data={summary.verdict_counts || {}} emptyLabel="No verdict stats yet." />
              </article>
            </section>

            <section className="rounded-xl border border-slate-700/70 bg-slate-900/70 p-4">
              <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-300">Top Domains</h2>
              {!summary.top_domains?.length && <p className="text-sm text-slate-400">No domain stats yet.</p>}
              {Boolean(summary.top_domains?.length) && (
                <div className="space-y-2">
                  {summary.top_domains.map((entry) => (
                    <div key={entry.domain} className="flex items-center justify-between rounded-lg border border-slate-700/70 bg-slate-900/60 px-3 py-2">
                      <span className="text-sm text-slate-200">{entry.domain}</span>
                      <span className="rounded bg-blue-500/15 px-2 py-0.5 text-xs font-semibold text-blue-200">{entry.count}</span>
                    </div>
                  ))}
                </div>
              )}
            </section>
          </>
        )}
      </div>
    </main>
  );
}
