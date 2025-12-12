"use client";

import { useState } from "react";

export default function Home() {
  const [text, setText] = useState("");
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
        body: JSON.stringify({ text, mode: "live" }),
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

  return (
    <main style={{ padding: 24, maxWidth: 900, margin: "0 auto" }}>
      <h1 style={{ fontSize: 22, fontWeight: 600 }}>Fact Validator</h1>
      <p style={{ marginTop: 8, opacity: 0.8 }}>
        Paste text and click Analyze. You should see JSON from the backend.
      </p>

      <textarea
        style={{ width: "100%", marginTop: 16, padding: 12, minHeight: 140 }}
        placeholder="Paste a claim or text..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <button
        style={{ marginTop: 12, padding: "10px 14px" }}
        disabled={loading || text.trim().length === 0}
        onClick={analyze}
      >
        {loading ? "Analyzing..." : "Analyze"}
      </button>

      {error && (
        <div style={{ marginTop: 12, color: "crimson" }}>
          <b>Error:</b> {error}
        </div>
      )}

      {result && (
        <pre style={{ marginTop: 16, padding: 12, background: "#f5f5f5", overflow: "auto" }}>
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </main>
  );
}
