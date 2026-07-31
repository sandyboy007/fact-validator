# Fact Validator - Legacy 20-Claim Methods

> Historical document retained for provenance. It does not describe the final
> 5,000-claim thesis experiment. See `docs/METHODS.md` for the final protocol.

## Document Version
- **Version:** 1.0
- **Date:** March 24, 2026
- **Author:** Research Team
- **Status:** Finalized

---

## 1. System Overview

### 1.1 System Architecture
Fact Validator is a full-stack fact-checking system that performs automated claim analysis and verdict generation. The pipeline consists of 9 sequential processing stages:

```
Input URL/Text
    ↓
1. Content Extraction (Trafilatura)
    ↓
2. Claim Decomposition (NLP)
    ↓
3. Evidence Retrieval (SerpAPI)
    ↓
4. Source Credibility Scoring
    ↓
5. Semantic Reranking
    ↓
6. Baseline Verdict Classification
    ↓
7. [Optional] LLM Debate Mode
    ↓
8. Sentiment & Bias Analysis
    ↓
9. Final Misinformation Scoring
```

**Key Technologies:**
- **Frontend:** Next.js 16, React 19, TypeScript 5
- **Backend:** FastAPI (Python 3.10+)
- **NLP:** Trafilatura, NLTK, Sentence Transformers
- **Search:** SerpAPI (Google Search)
- **Storage:** SQLite, JSON cache
- **Optional LLM:** Ollama (llama3.1:8b)

---

## 2. Dataset & Data Splitting

### 2.1 Primary Evaluation Dataset

**Source:** `docs/evaluation_benchmark.json`

**Composition:**
```
Total Claims: 20
├─ Climate: 1 (5%)
├─ Health: 4 (20%)
├─ Finance: 2 (10%)
├─ Politics: 2 (10%)
├─ Science: 4 (20%)
├─ History: 2 (10%)
├─ Conflict: 1 (5%)
├─ Media/General: 2 (10%)
└─ Mixed/Other: 2 (10%)

By Verdict Label:
├─ Supported: 8 (40%)
├─ Refuted/False: 8 (40%)
├─ NEI (Insufficient Evidence): 2 (10%)
└─ Mixed/Disputed: 2 (10%)

By Difficulty:
├─ Easy: 9 (45%)
├─ Medium: 8 (40%)
└─ Hard: 3 (15%)
```

### 2.2 Data Splitting Strategy

For comprehensive evaluation, we employ **stratified random splitting**:

```
Total: 20 claims (seed=42)
├─ Training Set: 12 claims (60%)
│  └─ Used for: Hyperparameter tuning, feature development
│
├─ Validation Set: 4 claims (20%)
│  └─ Used for: Model selection, threshold tuning
│
└─ Test Set: 4 claims (20%)
   └─ Used for: Final performance reporting
```

**Stratification Criteria:**
- Verdict label distribution maintained
- Difficulty distribution maintained
- Domain category distribution balanced

**Seed:** 42 (for reproducibility)

### 2.3 Baseline Evaluation

All baselines evaluated on **held-out test set only** (no training/validation):
- Random baseline: 4 predictions
- Keyword baseline: 4 predictions
- Length heuristic: 4 predictions
- Sentiment heuristic: 4 predictions
- Majority class: 4 predictions

---

## 3. Experimental Protocol

### 3.1 Evaluation Metrics

#### 3.1.1 Primary Metrics
```
Overall Accuracy = (TP + TN) / (TP + TN + FP + FN)

For Each Verdict Class (SUPPORTED, REFUTED, NEI):
├─ Precision = TP / (TP + FP)
├─ Recall = TP / (TP + FN)
└─ F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

#### 3.1.2 Secondary Metrics
```
Per-Category Accuracy:
├─ Health accuracy
├─ Political accuracy
├─ Science accuracy
└─ Other category accuracy

Difficulty Stratification:
├─ Easy claims accuracy
├─ Medium claims accuracy
└─ Hard claims accuracy

Confidence Calibration:
├─ Bin-wise (0-10%, 10-20%, ..., 90-100%)
├─ Calibration error: |avg_confidence - avg_accuracy|
└─ Expected Calibration Error (ECE)

AUC-ROC: Binary classification (Supported vs Others)
```

#### 3.1.3 Error Analysis
```
Error Types:
├─ Extraction Error: Claim extraction failed
├─ Retrieval Error: No evidence found (<30% confidence)
├─ Ranking Error: Evidence rankingissue (30-60% confidence)
├─ Verdict Error: Classification failure (>60% confidence)
└─ Confidence Error: Correct verdict, wrong confidence

Severities:
├─ HIGH: Verdict completely wrong (opposite label)
├─ MEDIUM: Verdict borderline/uncertain
└─ LOW: Minor errors, mostly correct
```

### 3.2 Ablation Study Design

Each ablation removes one component and measures impact:

```
Component 1: Domain Credibility Scoring
├─ With: Full system with credibility adjustment
└─ Without: Using only base verdict, no credibility weighting
└─ Impact Metric: Accuracy drop %

Component 2: Semantic Reranking
├─ With: Evidence ranked by semantic similarity
└─ Without: Evidence in SerpAPI order
└─ Impact Metric: Accuracy drop %

Component 3: Debate Mode
├─ With: LLM debate processing enabled
└─ Without: Debate disabled, baseline verdict only
└─ Impact Metric: Accuracy drop %

Component 4: Sentiment Adjustment
├─ With: Misinformation score adjusted by sentiment
└─ Without: Sentiment analysis disabled
└─ Impact Metric: Accuracy drop %

Component 5: Source Filtering
├─ With: Evidence from reputable sources only
└─ Without: All sources included
└─ Impact Metric: Accuracy drop %
```

---

## 4. Baseline Implementations

### 4.1 Random Baseline
**Type:** Lower bound  
**Algorithm:** Uniform random label assignment  
**Confidence:** Uniform random (30-70%)  
**Expected Accuracy:** ~33%

**Code:** `services/api/app/baselines.py::RandomBaseline`

### 4.2 Keyword Matching Baseline
**Type:** Strawman  
**Algorithm:** Keyword-based verdict determination  
**Keywords:**
- SUPPORTED: "confirmed", "verified", "proven", "supported", "research shows"
- REFUTED: "false", "debunked", "myth", "hoax", "refuted"
- NEI: "unclear", "uncertain", "may", "perhaps", "insufficient evidence"

**Confidence:** Based on keyword match count

**Code:** `services/api/app/baselines.py::KeywordBaseline`

### 4.3 Length Heuristic Baseline
**Type:** Simple heuristic  
**Algorithm:** Claim length based prediction  
- Short (<100 chars) → SUPPORTED
- Medium (100-200 chars) → NEI
- Long (>200 chars) → REFUTED

**Code:** `services/api/app/baselines.py::LengthHeuristic`

### 4.4 Sentiment Heuristic Baseline
**Type:** Linguistic heuristic  
**Algorithm:** Emotional language detection  
- High negative words → REFUTED
- High positive words → SUPPORTED
- Neutral → SUPPORTED

**Code:** `services/api/app/baselines.py::SentimentHeuristic`

### 4.5 Majority Class Baseline
**Type:** Trivial baseline  
**Algorithm:** Always predict most common label  
- Always predicts: SUPPORTED (40% of training set)

**Code:** `services/api/app/baselines.py::MajorityClassBaseline`

---

## 5. Statistical Analysis

### 5.1 Significance Testing

**Tests Used:**
- Paired t-test (one-tailed): System vs each baseline
- Confidence levels: 95% (α = 0.05)
- Test statistic: t = (mean_diff) / (std_err)

**Null Hypothesis:** System accuracy ≤ Baseline accuracy  
**Alternative:** System accuracy > Baseline accuracy

### 5.2 Effect Size Calculation

**Cohen's d formula:**
```
d = (mean_system - mean_baseline) / pooled_std

Interpretation:
├─ d < 0.2: Negligible effect
├─ 0.2 ≤ d < 0.5: Small effect
├─ 0.5 ≤ d < 0.8: Medium effect
└─ d ≥ 0.8: Large effect
```

### 5.3 Confidence Intervals

**95% Confidence Interval:**
```
CI = mean ± (t_alpha × SE)

Where:
├─ t_alpha = t-distribution critical value (df = n-1)
├─ SE = std_dev / sqrt(n)
└─ n = sample size
```

---

## 6. System Configuration

### 6.1 Environment Variables

**Required:**
```bash
FACT_VALIDATOR_DB=./data/fact_validator.db
SERPAPI_API_KEY=<your-api-key>
```

**Optional:**
```bash
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.1:8b
DEBATE_ENABLED=true
CACHING_ENABLED=true
```

### 6.2 Feature Flags

```python
# services/api/app/config.py
FEATURE_DEBATE_MODE = True
FEATURE_CACHING = True
FEATURE_RATE_LIMITING = False
FEATURE_STRUCTURED_LOGGING = False
```

### 6.3 Model Hyperparameters

```python
# Claim extraction
TOP_CLAIMS = 6
MIN_CLAIM_LENGTH = 20

# Evidence retrieval
TOP_SEARCH_RESULTS = 10
TOP_EVIDENCE = 3

# Semantic reranking
SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_RERANKED = 3

# Debate (if enabled)
DEBATE_MAX_ROUNDS = 3
DEBATE_TIMEOUT_SEC = 60

# Sentiment adjustment
SENTIMENT_WEIGHT = 0.2
```

---

## 7. Reproducibility Checklist

- [x] Dataset splits fixed (seed=42)
- [x] All baselines deterministic (seed controlled)
- [x] Metrics calculation documented
- [x] Environment variables specified
- [x] Hyperparameters frozen in config
- [x] Tests automated (pytest)
- [x] Results logged to database
- [x] Version control (git)
- [x] Code frozen at evaluation time
- [x] Results reproducible within statistical noise

---

## 8. Running Baseline Comparison

```bash
# Run all baselines
cd services/api
python -m pytest tests/test_evaluation.py::TestBaselineComparison -v

# Generate baseline comparison report
python -c "from app.evaluation import *; from app.baselines import *; ..."
```

---

## 9. References

- Dataset: `docs/evaluation_benchmark.json`
- Code: `services/api/app/evaluation.py`
- Baselines: `services/api/app/baselines.py`
- Tests: `services/api/tests/test_evaluation.py`
