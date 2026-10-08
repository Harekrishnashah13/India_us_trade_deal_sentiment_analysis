# India_us_trade_deal_sentiment_analysis
Social-media sentiment analysis of the US–India trade deal: TF-IDF + VADER vs an LLM (Claude) on Reddit and X posts, with topic modelling, a side-by-side comparison and charts.
# TF-IDF vs LLM: social-media sentiment on the US–India trade deal

Compares a **classical NLP pipeline** (TF-IDF, NMF topics, VADER, logistic regression) with **LLM labelling** (Claude) on the same Reddit and X posts. It reports where the two agree, where they diverge, and what the community is talking about.

The default topic is the 2026 US–India trade deal. Edit one config file to analyse something else, such as cricket.

> **Data notice.** The datasets bundled in `data/raw/` are **synthetic**: written to mirror the real 2026 storyline, not scraped from Reddit or X. Use `collect_real.py` (below) to gather real posts. Every result in this README comes from the 110-post demo corpus.

---

## What it does

| Step | Classical lane | LLM lane |
|---|---|---|
| Features | TF-IDF (1–2 word phrases, domain stop-words) | — |
| Sentiment | VADER lexicon score → pos / neu / neg | Claude: sentiment **and** stance, sarcasm aware |
| Topics | NMF clusters over TF-IDF | Claude: one label from an 11-topic list |
| Learning | Logistic regression on TF-IDF, trained on the LLM labels (5-fold CV) | — |
| Compare | Agreement, Cohen's κ, macro-F1, confusion matrix, topic match (ARI), biggest disagreements | |

## Architecture

```
 COLLECT            collect_real.py / src/collectors/   Reddit (PRAW) · Bluesky · Hacker News · X API v2
    │               or data/raw/*.jsonl (demo, synthetic, your own export)
    ▼               schema: id, source, channel, created_at, score, text
 PREPROCESS         src/preprocess.py
    │               de-dup · two text views:
    │                 text_vader  – keeps case, punctuation, emoji
    │                 text_tfidf  – lowercase, tokens like 18pct, usd500bn, h1bvisa
    ├──────────────────────────────┐
    ▼                              ▼
 CLASSICAL  src/classical.py     LLM  src/llm_analyzer.py
  TfidfVectorizer                 Claude Messages API + forced tool schema
  NMF (6 topics)                  batches of 20 → {sentiment, stance, topic, sarcasm}
  VADER compound                  cached to JSONL · optional narrative briefing
  LogReg on LLM labels (CV)
    └──────────────┬───────────────┘
                   ▼
 COMPARE    src/compare.py    agreement · Cohen's κ · macro-F1 · confusion · ARI · disagreements
                   ▼
 REPORT     src/report.py     results/metrics.json · CSVs · results/figures/*.png
```

## Quick start

```bash
pip install -r requirements.txt

# Offline demo: no API keys needed (uses cached LLM labels)
python run_pipeline.py --source demo --llm cache

pytest -q tests
```

## Using real data

`collect_real.py` gathers on-topic posts into `data/raw/real_posts.csv/.jsonl`, with a link to each original. Run it on your own machine:

```bash
# Reddit (free): create a "script" app at https://www.reddit.com/prefs/apps
export REDDIT_CLIENT_ID=...  REDDIT_CLIENT_SECRET=...
export REDDIT_USER_AGENT="trade-sentiment/0.1 by u/<you>"
# Bluesky (free): Settings > Privacy and security > App passwords
export BSKY_HANDLE=you.bsky.social  BSKY_APP_PASSWORD=...
# X (optional, paid API tier with search)
export X_BEARER_TOKEN=...

python collect_real.py --target 2500
```

Hacker News needs no key. Sources without credentials are skipped. Posts are kept only if they mention India **and** trade or tariffs.

Then analyse them with the Claude API:

```bash
export ANTHROPIC_API_KEY=...
python run_pipeline.py --source file --input data/raw/real_posts.jsonl --llm api --insights
```

## CLI

| Flag | Values | Meaning |
|---|---|---|
| `--source` | `demo` · `file` · `live` | Bundled 110 posts, a JSONL you supply (`--input`), or the live collectors |
| `--llm` | `cache` · `api` | Reuse saved labels (`--labels`) or call Claude |
| `--insights` | | With `--llm api`, also write `results/llm_insights.md` |

## Datasets

| File | Rows | Notes |
|---|---|---|
| `data/raw/demo_corpus.jsonl` | 110 | Synthetic, hand-written posts; LLM labels in `data/processed/llm_labels_insession.jsonl` |
| `data/raw/synthetic_2400.csv/.jsonl` | 2,400 | Synthetic, from ~65 templates (`scripts/generate_synthetic.py`). Has `gen_*` labels taken from each template. Good for training/practice; flatters TF-IDF |
| `data/raw/real_posts.csv/.jsonl` | — | Created by `collect_real.py` |

## Switching topic (e.g. cricket)

In `src/config.py`, change `topic_name`, `reddit_query`, `reddit_subreddits`, `x_query` and `TOPIC_TAXONOMY`. In `collect_real.py`, change `QUERY_TERMS` and the `RELEVANT` filter. Nothing else needs to change.

## Demo results (110 posts)

| | VADER | TF-IDF + LogReg (CV) | LLM |
|---|---|---|---|
| Positive | 46% | 9% | 28% |
| Neutral | 23% | 55% | 38% |
| Negative | 31% | 35% | 34% |
| Agreement with LLM | 51% (κ 0.28) | 49% (κ 0.21) | — |

- **Sarcasm:** VADER matches the LLM on 25% of sarcastic posts, vs 53% on the rest.
- **Topics:** NMF clusters vs LLM topics give ARI 0.15 and purity 0.44. NMF groups posts by shared words (e.g. every "99% settled" post); the LLM groups them by meaning.
- **Classifier:** TF-IDF + LogReg needs far more than 110 labelled posts; here it falls back to "neutral". A common pattern is to label 1–2k posts with the LLM and train the cheap model on those.

**Community themes (demo data):** relief among exporters (Tiruppur, Surat, Andhra shrimp); growing frustration with delays and the unpublished text; Russian oil seen as a loss of independence; farmers split on both sides.

## Project structure

```
run_pipeline.py              orchestrator / CLI
collect_real.py              real-data collector (Reddit, Bluesky, HN, X)
scripts/generate_synthetic.py
src/
  config.py                  queries, model settings, topic list
  preprocess.py              cleaning, two text views
  classical.py               TF-IDF, NMF, VADER, LogReg
  llm_analyzer.py            prompt, tool schema, batching, insights
  compare.py                 metrics, confusion, disagreements
  report.py                  charts
  collectors/                reddit_collector.py, x_collector.py
tests/test_pipeline.py
results/                     metrics.json, labelled_posts.csv, top_disagreements.csv, figures/
```

## Glossary

- **TF-IDF:** weights a word by how often it appears in a post, discounted by how many posts contain it, so distinctive words rank high.
- **VADER:** rule-based sentiment from a ~7,500-entry human-scored word list, plus rules for caps, `!`, "very", "not" and "but". Output is a compound score from −1 to +1; ≥ 0.05 is positive and ≤ −0.05 is negative.
- **NMF:** splits the TF-IDF matrix into topic clusters, each described by its top words.
- **Cohen's κ:** agreement corrected for chance, κ = (p_o − p_e) / (1 − p_e). 0.21–0.40 is "fair".
- **Stance vs sentiment:** sentiment is the tone of a post; stance is whether the author supports the deal. "Deadline missed again" is negative in tone but neutral in stance.

## Limitations

- Bundled data is synthetic. The demo LLM labels were produced by Claude in a chat session using the same schema as the API code.
- There are no human gold labels, so "agreement with the LLM" is not accuracy. Hand-label ~200 posts to measure both methods properly.
- X recent search covers about 7 days and needs a paid tier. Reddit search is relevance-limited. Follow each platform's API terms.
- The LLM costs money per post and can drift between model versions, so pin the model and cache its outputs.

## Requirements

Python 3.10+, `pandas`, `scikit-learn`, `vaderSentiment`, `matplotlib`, `anthropic`, `praw`, `requests`, `pytest` (see `requirements.txt`).
