# US–India Trade Deal: What Is the Community Saying?
 
Sentiment and topic analysis of real X (Twitter) posts about the 2026 US–India trade talks. The same posts are analysed twice, once with classic NLP (TF-IDF, VADER, logistic regression, NMF) and once with a large language model (Llama via the Groq API), and the two readings are compared.
 
**Questions**
1. What is the overall sentiment towards the deal: positive, negative or neutral?
2. Which themes dominate the conversation?
3. Does an LLM read the conversation differently from a word-counting model, and where?
---
 
## Pipeline
 
```
 Raw posts ─► Cleaning ─┬─► Part A: Classic NLP ──────────────┐
 (X, 2,804)  (1,666)    │   TF-IDF + 12 hand-crafted features │
                        │   VADER sentiment                   ├─► Part C: Compare ─► Insights
                        │   Logistic regression, NMF topics   │
                        └─► Part B: LLM (Llama via Groq) ─────┘
                            sentiment + topic + reason per post
```
 
| Stage | What happens | Tools |
|---|---|---|
| 1. Collect | X search for posts that mention India **and** the US **and** trade or tariffs | Python, `twscrape` |
| 2. Store | 2,804 raw posts in one CSV | pandas |
| 3. Clean | 2,804 → 1,666 posts (see below) | pandas |
| 4A. Classic NLP | TF-IDF features, VADER sentiment, logistic regression, NMF topics | scikit-learn, vaderSentiment |
| 4B. LLM | Posts sent in batches of 30; the model returns sentiment, topic and a one-line reason as JSON | Groq API, `llama-3.1-8b-instant` |
| 5. Compare | Agreement, Cohen's kappa, confusion matrix, topic overlap, sentiment over time | scikit-learn, matplotlib |
 
## Data
 
- **Source:** public X posts found by keyword search, 20 September – 8 October 2026.
- **Raw:** 2,804 posts.
- **Cleaning** (one row = one real voice):
  1. Keep posts from 20 September 2026 onwards
  2. Drop exact duplicate texts
  3. Drop posts longer than 1,000 characters (newsletters where the topic is a passing mention)
  4. Keep at most 3 posts per author (their most-liked)
  5. Drop near-duplicates (same text with different @mentions or links)
- **After cleaning:** 1,666 posts.
- **Privacy:** author handles are replaced with a hash (`author_hash`).
## Method
 
### Part A: Classic NLP
- **TF-IDF features:** unigrams and bigrams (`ngram_range=(1, 2)`), `min_df=5`, `max_df=0.6`. Stop words and the search words (india, us, trade, tariff…) are removed because every post contains them. Negations are kept for the sentiment model.
- **12 hand-crafted features** describe *how* a post is written: length, `!` count, CAPS ratio, negations, likes (log-scaled) and similar.
- **VADER** gives each post a lexicon-based sentiment score.
- **Logistic regression** (80/20 train–test split, `class_weight="balanced"`) is trained on VADER's labels as weak supervision. Two versions are compared: TF-IDF only, and TF-IDF plus the hand-crafted features.
- **NMF** finds 8 topics from the TF-IDF matrix.
### Part B: LLM labelling
- Model: `llama-3.1-8b-instant` on Groq (falls back to another available model if needed).
- Each request sends 30 posts and gets back JSON with a **sentiment** (Positive / Negative / Neutral), a **topic** and a short **reason**.
- The prompt tells the model to read sarcasm as the author means it and to label plain news as Neutral.
- Topics: Deal negotiations · Russian oil & sanctions · Tariffs & economic impact · Geopolitics & diplomacy · Domestic politics · Not about India–US trade.
- `temperature=0`; every label is checked in Python against the allowed values. Results are cached to `llm_labels.csv`, so a re-run continues where it stopped.
### Part C: Comparison
- Agreement % and **Cohen's kappa** between VADER, the TF-IDF model and the LLM.
- The posts where the methods disagree most, with the LLM's reason.
- NMF topics against LLM topics.
- Daily net sentiment (% positive − % negative) for each method.
- **Better teacher:** the same TF-IDF model is retrained on the LLM's labels instead of VADER's, to show that label quality matters more than model choice.
## Results
 
Run the notebook and fill in this table with your own numbers:
 
| | VADER | TF-IDF model (VADER labels) | TF-IDF model (LLM labels) | LLM |
|---|---|---|---|---|
| Positive | | | | |
| Neutral | | | | |
| Negative | | | | |
| Agreement with LLM | | | | — |
| Cohen's kappa vs LLM | | | | — |
 
### What the conversation is about
1. **The talks have stalled.** The most-shared news is the Finance Minister saying talks have reached a "plateau", and USTR saying a deal is "not imminent". Most of these posts are factual news; the opinions around them are pessimistic.
2. **Russian oil is the biggest pressure point.** A US sanctions law allows tariffs of up to 100% on buyers of Russian energy. The tone is tense and often defiant.
3. **Sovereignty splits the conversation.** Some posts read the stalemate as India standing firm; others accuse the government of weakness.
### Classic NLP vs LLM
| | TF-IDF + VADER + logistic regression | LLM (Groq) |
|---|---|---|
| Sarcasm, irony | Takes words literally | Reads the intended meaning |
| News vs opinion | "BIG RELIEF?" scores as positive | Plain news is labelled Neutral |
| Off-topic posts | Cannot tell | Flags them |
| Topics | Keyword groups, one large catch-all | Named, meaningful categories |
| Explainability | Exact word weights | A short reason per post |
| Speed and cost | Seconds, free | Minutes; free tier has daily limits |
| Reproducibility | Same result every run | Can vary slightly between runs |
 
**Takeaway:** TF-IDF is fast, free and transparent, which makes it good for finding keywords and storylines. The LLM is better at understanding what people mean, which matters on political social media where sarcasm and headlines are common.
 
## How to run
 
The notebook is written for Google Colab.
 
1. Open `US_India_Trade_Sentiment_Analysis.ipynb` in Colab.
2. Upload `x_all.csv` when asked (or place it next to the notebook).
3. Get a free API key at [console.groq.com](https://console.groq.com) and add it as a Colab secret named `GROQ_API_KEY` (the notebook asks for it if it isn't set).
4. Run all cells. For a quick test, set `LLM_LIMIT = 300` in section 5.3.
Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `vaderSentiment`, `groq`.
 
## Files
 
| File | Description |
|---|---|
| `US_India_Trade_Sentiment_Analysis.ipynb` | Full analysis: cleaning, Part A, Part B, comparison, insights |
| `x_all.csv` | Raw posts (input) |
 
Created when the notebook runs: `posts_classic_nlp.csv`, `llm_labels.csv`, `posts_with_all_labels.csv`, `architecture.png`.
 
## Limitations
- X only, about three weeks, English posts found by keyword search. Not a representative sample of all opinion.
- The TF-IDF model first learns from VADER labels, so it inherits VADER's mistakes.
- There are no human-labelled posts, so agreement with the LLM is not the same as accuracy.
## Next steps
- Hand-label about 200 posts to measure the true accuracy of each method.
- Add Reddit as a second community.
- Re-run weekly to track how sentiment changes as the talks move.
## Author
**Harekrishna Shah** · Data Analyst · [LinkedIn](https://www.linkedin.com/in/hshah13) · [GitHub](https://github.com/Harekrishnashah13)
