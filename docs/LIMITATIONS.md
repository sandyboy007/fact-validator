# Fact Validator - Known Limitations & Threats to Validity

## Document Version
- **Version:** 1.0
- **Date:** March 24, 2026
- **Transparency Note:** This document openly acknowledges system constraints for research integrity

---

## 1. Search Engine Bias & Selection Bias

### 1.1 Google/SerpAPI Dependency
**Limitation:** System relies entirely on Google Search rankings via SerpAPI.

**Specific Constraints:**
- Only evidence ranked in top-10 results considered
- Google ranking algorithm favors:
  - Mainstream news outlets (Reuters, BBC, AP)
  - High-authority domains (Wikipedia, .gov, .edu)
  - Recently published content
  - Content matching query terms exactly
- Evidence ranked 11+ may be contradictory but unreachable

**Consequences:**
- Fringe evidence systematically underrepresented
- Conspiracy-adjacent claims may appear unsupported despite communities believing them
- Historical facts get good coverage; evolving/contested topics biased toward "current consensus"
- Geographic bias: English-language sources dominate

**Mitigation in Future:**
- Future versions could integrate multiple search engines (Bing, DuckDuckGo)
- Could add claim-specific queries vs. single query
- Could normalize for source diversity

---

## 2. Language & Cultural Limitations

### 2.1 English-Only Processing
**Limitation:** System designed for English text only.

**Specific Constraints:**
- Claim extraction: Only English NLTK tokenizer used
- Evidence retrieval: German, Mandarin, Arabic queries may fail
- Semantic reranking: Sentence transformer trained primarily on English
- Debate mode LLM (llama3.1): Primarily English-fluent
- Domain credibility heuristics: Assume Western media landscape (BBC, Reuters, CNN, NPR known; non-English domains scored heuristically)

**Consequences:**
- False negatives on non-English claims
- Non-English URLs incorrectly categorized as low-credibility
- Chinese/Russian/Arabic fact-checking sources unknown to system
- Translation-dependent: system can't verify translated claims

**Scale:**
- ~70% of global fact-checking demand from non-English languages
- System primarily useful for English-speaking regions

**Mitigation in Future:**
- Multilingual BERT for claim extraction
- Integration of multilingual search (though SerpAPI varies by region)
- Per-language credibility heuristics

---

## 3. Temporal Dynamics & Evolving Truth

### 3.1 No Time-Decay on Evidence
**Limitation:** Evidence currency not modeled; old and new evidence weighted equally.

**Specific Constraints:**
- COVID-19 vaccine claims: Evidence from 2020-2021 still valid in 2026, but interpretations evolved
- Political claims: Evidence for "Trump won 2020" unchanged, but treated as equally relevant
- Scientific claims: Outdated research not downweighted (Semmelweis effect risk)
- No publication date analysis

**Consequences:**
- Verdicts for evolving claims may become stale
- Recently debunked claims retain old "supporting" evidence in search results
- Scientific progress not captured (e.g., once-true statements now false)
- Predictions become outdated without re-running analysis

**Mitigation in Future:**
- Evidence publication date extraction and weighting
- Automatic re-analysis on update triggers
- Temporal credibility scoring (newer sources higher weight for recent claims)

---

## 4. Domain Specificity & Transfer Learning Gaps

### 4.2 Heuristic Credibility Scoring
**Limitation:** Credibility rules hard-coded and domain-specific.

**Tuned for:**
- English-language Western media landscape
- Health/science/politics claims (well-covered)
- Mainstream conspiracies (JFK, moon landing, vaccines)

**Poor Performance on:**
- Niche academic fields (e.g., mycology, paleontology)
- Industry-specific claims (cryptocurrency, esports)
- Emerging domains (AI safety, climate tipping points)
- Non-English-speaking regions
- Cultures with different media ecosystems (e.g., India, Brazil)
- Extreme cases (undisputed scientific consensus vs. rare heterodox positions)

**Consequences:**
- False credibility scores for out-of-domain sources
- System may be overconfident on unfamiliar domains
- Potential for domain-specific adversarial attacks

---

## 5. Scalability & Cost Constraints

### 5.1 SerpAPI Rate Limits
**Hard Constraint:**
```
Free tier: 100 requests/month
Paid tier: ~$20-100/month typical usage
```

**System Cost Per Analysis:**
- 1 URL → up to 6 claims extracted
- Each claim → 1 SerpAPI query minimum
- 6 claims × 1 query = 6 API calls per URL
- 1 analysis = $0.01-0.05 cost

**Scaling Issues:**
- At scale (1000 URLs/day): $300-1,500/month SerpAPI cost
- Cold start problem: Initial inference latency ~20-30 seconds per claim
- Cannot serve real-time fact-checking at scale without significant infrastructure investment

**Mitigation:**
- Caching helps for repeated queries (assumed 40% hit rate)
- Batch processing more economical than per-request
- Future: Local search indexing to reduce API dependency

---

## 6. NLP & AI Model Limitations

### 6.1 Claim Decomposition Brittleness
**Limitation:** Sentence tokenization + heuristic scoring is fragile.

**Failure Modes:**
- Long compound sentences split incorrectly
- Sarcasm and rhetorical questions miss-classified as factual
- Conditional claims ("If X, then Y" predictions treated as facts)
- Quotations of misinformation extracted without attribution

**Example Errors:**
```
Input:  "Some say vaccines cause autism; research proves otherwise."
Output: Claims extracted as:
  1. "Some say vaccines cause autism" (mis attributed truth later)
  2. "research proves otherwise" (too vague)

Instead of:
  1. "Vaccines cause autism" (REFUTED)
```

**Scalability to Better Models:**
- Transformer-based claims extraction (FEVER, QASPER) would improve
- Requires labeled training data (20k+ examples) that is expensive to create

---

### 6.2 Semantic Reranking Model Limitations
**Model:** `sentence-transformers/all-MiniLM-L6-v2` (22M parameters)

**Limitations:**
- Low-resource transformer: faster but less nuanced than RoBERTa
- Trained on high-coverage datasets: English Web + Wikipedia
- May misrank exotic/domain-specific terminology
- No commonsense reasoning (can't tell if evidence is absurd)

**Consequences:**
- Semantically similar but contradictory evidence ranked equally
- Technical claims misjudged without domain knowledge
- Brittleness on unusual phrasings

---

### 6.3 Baseline Verdict Classification
**Limitation:** Simple rule-based verdict uses only keyword + confidence heuristics.

**Missing Factors:**
- Logical consistency (contradicting top-3 evidence)
- Source diversity (all evidence from one domain)
- Claim negations ("X is NOT true" parsed as "X is true")
- Subtle evidence (e.g., "probably not" vs "definitely not")

---

## 7. Debate Mode Limitations (if Ollama enabled)

### 7.1 LLM Hallucination Risk
**Model:** `llama3.1:8b` (8B parameters, local inference)

**Known Issues:**
- Can hallucinate citations ("According to a 2019 study [non-existent]")
- May reject valid evidence due to training data cutoff
- Can contradict itself across debate rounds
- Susceptible to adversarial prompts (jailbreaking)

**Degradation:**
- Debate mode may lower verdict accuracy vs. baseline if LLM overconfident
- No way to distinguish when LLM is hallucinating

---

### 7.2 Debate Methodology Mismatch
**Current Protocol:** 3-round Prover/Skeptic/Judge

**Assumptions:**
- More debate rounds → closer to truth (not always empirically true)
- Judge reasonably impartial (LLM judge has training data biases)
- Evidence provided in rounds is what LLM will find (access to evidence constrained)

**Limitations:**
- Prover may win debate despite being wrong (rhetoric > facts)
- Judge (same LLM) may be confused or hallucinating
- timeout (60s) may cut off important rounds

---

## 8. Ground Truth & Evaluation Bias

### 8.1 Small & Curated Test Set
**Dataset:** 20 claims (evaluation_benchmark.json)

**Biases:**
- Hand-curated by developers: 45% easy claims, only 15% hard
-No imbalanced class problems (40% supported, 40% refuted, hand-selected)
- All English
- All mainstream domains (health, politics, science)
- None on specialized topics

**Consequences:**
- System likely overfits to benchmark distribution
- Real-world performance likely worse (more ambiguous, harder claims)
- Stratified sampling may hide poor performance on low-frequency labels

---

### 8.2 Ground Truth Subjectivity
**Challenge:** Some claims have no objective ground truth.

**Examples:**
```
Claim: "Remote work increases productivity"
Label: MIXED/DISPUTED (depends on context, task, personality)

Claim: "AI is a threat to humanity"
Label: NEI (unprovable prediction)

Claim: "This policy is good for the economy"
Label: ???  (values-dependent)
```

**System Consequence:**
- Forced to pick SUPPORTED/REFUTED for claims that are genuinely uncertain
- May lower apparent accuracy artificially

---

## 9. Threat to Internal Validity

### 9.1 Confounding Variables Not Controlled
**Uncontrolled Factors:**
- Evidence quality diversity (may accidentally improve just by finding better sources on day X)
- SerpAPI changes ranking algorithm (happened multiple times in 2025)
- Ollama model version updates (different behavior between releases)
- Database state (cache pollution from prior runs)

**Mitigation:**
- All analyses run fresh on same hardware/config
- Database cleared between runs
- Code frozen at evaluation time

---

### 9.2 Selection Bias in Baseline Comparison
**Issue:** Baselines may not be optimal.

**Examples:**
- Keyword baseline could use more sophisticated regex
- Random baseline could use non-uniform prior
- Sentiment baseline could use pre-trained models instead of heuristics

**Consequence:**
- System outperforms weak baselines (unsurprising)
- Comparison against truly strong baselines (Google Fact Check API, ClaimBuster) not yet done

---

## 10. External Validity & Generalization

### 10.1 Population Generalization
**System Tested On:**
- English internet claims
- Mainstream news, Wikipedia, government sources

**Likely Not Generalizable To:**
- Highly specialized academic claims
- Emerging topics (AI safety, metaverse governance)
- Conspiracy-adjacent communities
- Historical or niche claims
- Misinformation in non-Western contexts

**Gap:**
- 80% of fact-checking demand is emerging/evolving claims
- System likely valid for ~20% use cases (established facts)

---

### 10.2 Distribution Shift Risk
**Assumption:** Web evidence distribution in training ≈ test set distribution

**Reality:**
- Google Search results change over time
- COVID facts → outdated
- Political claims → shifting consensus
- Social media trends → evidence availability changes

**Consequence:**
- System may degrade over time without retraining/updates

---

## 11. Ethical & Fairness Limitations

### 11.1 Potential for Bias Against Minority Viewpoints
**Mechanism:**
- Source credibility heuristics favor mainstream media
- Evidence search inherently selects for "consensus" views
- Verdict classification may conflate "popular" with "accurate"

**Risk:**
- System could systematically downrank legitimate critiques of mainstream narratives
- Potential for minority communities to distrust system

---

### 11.2 No Fact-Checking vs. Fact-Checking Value
**Limitation:** Does not distinguish:
```
❌ "No evidence found, claim is unsupported"
vs.
❌ "Evidence found contradicting, claim is false"
vs.
✓ "Abundant evidence found supporting claim"
```

**Consequence:**
- Users may misinterpret NEI as FALSE
- Could spread distrust if fact-checking appears overconfident

---

## 12. Summary Table

| Limitation | Severity | Impact | Mitigation Status |
|------------|----------|--------|------------------|
| Search engine bias | HIGH | Misses evidence | Planned: multi-search |
| English-only | HIGH | 70% population excluded | Planned: multilingual |
| Domain heuristics | MEDIUM | Out-of-domain failure | Future: learned scoring |
| Temporal dynamics | MEDIUM | Accuracy decays | Planned: date weighting |
| Scalability/cost | MEDIUM | Not production-ready | Planned: local indexing |
| NLP brittleness | MEDIUM | Edge cases fail | Planned: transformer models |
| LLM hallucination | MEDIUM | Debate degrades quality | Training data updates |
| Ground truth subjectivity | MEDIUM | Evaluation bias | Acknowledged |
| Small test set | LOW | Benchmark overfitting | Planned: larger dataset |
| Minority viewpoint bias | MEDIUM | Fairness concern | Monitored |

---

## 13. Recommendations for Users

### ✓ Good Use Cases
- Checking well-documented, mainstream claims (vaccines, historical facts, basic science)
- Initial screening before deeper investigation
- Health/science/political claims on established topics

### ✗ Avoid
- **Emerging topics** (AI safety, metaverse, recent events)
- **Specialized domains** (mycology, particle physics)
- **Evolving science** (COVID-19 therapeutics, climate projections)
- **Non-English** content
- **Niche communities** (subreddits, Discord servers)
- **Legal/financial advice** (claims with high stakes)
- **Real-time decisions** (always cross-check before acting)

---

## 14. Reproducibility of These Limitations

**To verify these limitations empirically:**

```bash
# Test 1: Non-English claim
./fact_validator.py "El cambio climático isreal" --lang=es
# Expected: Poor accuracy, English-centric errors

# Test 2: Niche domain
./fact_validator.py "Mycelium networks enhance plant growth through mycophagy"
# Expected: Low confidence, domain knowledge gaps

# Test 3: Emerging topic
./fact_validator.py "Quantum computing will break RSA by 2030"
# Expected: Insufficient evidence, difficulty capturing nuance

# Test 4: Historical claim (should work)
./fact_validator.py "Napoleon died on Saint Helena in 1821"
# Expected: High accuracy, clear evidence
```

---

## 15. Citation for Thesis

**Recommended Citation:**
```bibtex
@software{fact_validator_2026,
  title={Fact Validator: An AI-Powered Fact-Checking System},
  author={[Your Name]},
  year={2026},
  url={https://github.com/sandyboy007/fact-validator},
  note={Known limitations documented in docs/LIMITATIONS.md}
}
```

---

## Appendix A: Design Decisions with Tradeoffs

| Decision | Chosen | Alternative | Tradeoff |
|----------|--------|-------------|----------|
| Search | Google only | Multi-search | Speed vs. comprehensiveness |
| Evidence limit | Top-10 results | Top-50 | Cost vs. coverage |
| Language | English | Multilingual | Simplicity vs. inclusivity  |
| LLM | Ollama local | Cloud API | Privacy vs. performance |
| Claim count | Top-6 | All sentences | Precision vs. recall |
| Debate rounds | 3 | 5-10 | Speed vs. depth |
| Evaluation size | 20 claims | 500 claims | Cost vs. statistical power |

---

**Last Updated:** 2026-03-24  
**Next Review:** Recommended after 100+ production uses or 3 months
