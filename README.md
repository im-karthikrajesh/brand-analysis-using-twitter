# brand-analysis-using-twitter

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](#)

> Brand analysis of Twitter (X) conversations for **Adidas** and **Lululemon** with robust cleaning, sentiment & topic mining, perceptual mapping, mention‑network centrality, and **micro‑influencer discovery**.

---

## Overview
This repository contains a complete Python workflow (script/notebook equivalent) to compare brand conversations on **Twitter/X** for two focal brands (default: *Adidas* and *Lululemon*). The pipeline loads tweet CSVs, performs **text cleaning & normalization**, extracts **hashtags/mentions**, builds **EDA and time‑series views**, computes **sentiment** (VADER & TextBlob), runs **topic modeling** (TF‑IDF + NMF), constructs a **mention network** (NetworkX) to measure centrality, and aggregates user‑level metrics to shortlist **micro‑influencer candidates** using a transparent **composite score**. It then applies **skew/outlier handling**, **PCA**, and **KMeans clustering** to segment candidates and highlight the **top pick per brand**.

## Data
- **Input files (CSV):**
  - `/content/adidas_tweets.csv`
  - `/content/lululemon_tweets.csv`

## Pipeline
1. **Load & Summarize** — read brand CSVs, print dataset shapes, column info, basic stats (users, date window, location coverage).
2. **Clean & Prepare**
   - Deduplicate rows; parse `created_at` → datetime; flag `location_available` and fill missing locations.
   - Build `processed_text` via URL/emoji/mention stripping, tokenization, stopword removal, and **WordNet lemmatization**.
3. **Hashtags, Mentions & Word Clouds**
   - Regex‑based extraction → frequency tables & **seaborn** bar charts, plus **WordCloud** visualizations.
4. **EDA & Time Series**
   - Engagement distributions (retweets/favorites); follower count comparisons.
   - Daily & weekly tweet volume with **7‑day rolling averages** (interactive **Plotly** lines).
5. **Sentiment & Topics**
   - **VADER** and **TextBlob** sentiment scores; histograms.
   - **TF‑IDF + NMF** topics (`n_topics=5`) with top‑word tables; assign each tweet a **dominant_topic**.
6. **Perceptual Map**
   - Interactive **Plotly scatter**: x = VADER sentiment, y = **engagement_ratio** (log scale), color = topic, symbol = brand.
7. **Mention Network & Centrality**
   - Build directed weighted graph of `@mentions` (screen_name → mentioned screen_name) with **NetworkX**; compute degree centrality.
8. **User‑Level Aggregation**
   - Aggregate to **user.id** (sentiment means, activity, hashtag diversity, dominant topic, centrality).
9. **Outliers & Skewness**
   - IQR capping; **log1p** for followers/statuses; **sqrt** for engagement and skewed metrics; **Yeo‑Johnson** for tweet counts.
10. **Composite Score & Micro‑Influencers**
    - Normalize features to [0,1]; **composite_score** = `0.30·engage + 0.25·reach + 0.05·activity + 0.20·centrality + 0.20·sentiment`.
    - **Follower band** filter: **1k–100k** → *micro*.
11. **Correlation • PCA • Clustering**
    - Correlation heatmap of transformed features; PCA explained variance; **KMeans** clustering (k=2…10, SSE + silhouette).
    - Final **k=5**; pick **top candidate per cluster** and **overall top** per brand.
12. **Exports**
    - **`micro_influencers_final.csv`**

## Key Formulas & Design Choices
- **Engagement Ratio (tweet‑level):**  
  `engagement_ratio = (retweet_count + favorite_count) / user.followers_count`
- **Skew handling:** log1p (counts), sqrt (proportions & skewed non‑negatives), Yeo‑Johnson (tweet_count).
- **Composite Score (user‑level):**  
  `score = 0.30·engage_norm + 0.25·reach_norm + 0.05·tweet_norm + 0.20·cent_norm + 0.20·sent_norm`
- **Micro band:** 1,000 ≤ followers ≤ 100,000 (tune per brief).
- **Topics:** NMF on TF‑IDF (stopwords=english, max_df=0.95, min_df=2), `n_topics=5`, `n_top_words=10`.

## Outputs
- **CSV:** `micro_influencers_final.csv` — filtered, normalized, scored & clustered candidates with brand labels.

## Repository Structure
```
.
├─ Brand_Analysis_using_Twitter_X.ipynb
├─ outputs/
│  └─ micro_influencers_final.csv
└─ README.md
```

## Running
- **Colab/Jupyter (recommended):** open the notebook, set file paths, run cells in order.

## Requirements
Create `requirements.txt` with:
```txt
pandas
numpy
seaborn
matplotlib
plotly
networkx
nltk
wordcloud
tabulate
textblob
scikit-learn
scipy
```

### NLTK Resources

```python
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('omw-1.4')
# 'punkt_tab' is included in the code for safety in some environments
```
> If a tokenization LookupError occurs, re‑run the cell or ensure internet access for NLTK downloads.


---

**Author:** Karthik Rajesh  
**Environment:** Google Colab
