# Project Gutenberg NLP

**Sentiment analysis and topic modeling across 57,000 books from the Project Gutenberg library.**

> **This repository is source-available, not open source.**
> Viewing for educational and reference purposes is permitted. All other use is prohibited.
> See [LICENSE](LICENSE) for full terms. Violations will be pursued.

---

## Overview

Large-scale NLP pipeline that processes the entire Project Gutenberg digital library to extract narrative structure, sentiment dynamics, and thematic fingerprints from classic literature. The system models how stories work at a structural level, tracking emotional arcs and topic shifts from exposition through resolution.

## Technical Architecture

### Data Pipeline
- **57,000 books** ingested via R-programming Gutenberg module
- Text cleaning and concatenation pipeline for consistent corpus preparation
- Paragraph-level segmentation for granular narrative analysis

### Topic Modeling
- **NMF** (Non-negative Matrix Factorization) for subgenre decomposition
- Feature extraction via **CountVectorizer** and **TF-IDF**
- Per-book topic evolution tracking across narrative segments
- Topic change distance metric for narrative "complexity" scoring

### Sentiment Analysis
- **TextBlob** polarity analysis across narrative arc
- Binned sentiment trajectories (Exposition -> Rising -> Climax -> Falling -> Resolution)
- Cross-book sentiment pattern comparison

### Clustering & Visualization
- **K-Means** clustering for genre grouping
- **t-SNE** dimensionality reduction for visual exploration
- Interactive **Dash/Plotly** web application for result exploration

### Recommendation Engine
- Recommendations derived from topic similarity scores (NMF vectors)
- Genre-aware clustering for "books like this" suggestions
- LLM-powered recommendation generation (see `recommend.py`)

## Quick Start

```bash
# LLM-powered recommendations (requires vLLM, Ollama, or OpenAI API key)
python recommend.py "The Yellow Wallpaper"
python recommend.py "Moby Dick" --n 5
```

## Technology Stack

`R` `TextBlob` `NLTK` `CountVectorizer` `TF-IDF` `NMF` `K-Means` `t-SNE` `Flask` `Dash` `Plotly` `Python`

## Related Work

This research directly informs the literary analysis and recommendation engine in [Readify](https://www.ireadifybooks.com), an AI-powered interactive reading platform for schools and institutions.

---

## Legal Notice

**Copyright (c) 2018-2026 Clarence Stephen. All rights reserved.**

This software is provided under a **Source Available License**. It is **not** open source. You may view this code for educational and reference purposes only. Commercial use, redistribution, modification, derivative works, and incorporation into other products or services are **strictly prohibited** without prior written authorization.

Unauthorized use will result in legal action, including DMCA takedowns, injunctive relief, and claims for damages. See [LICENSE](LICENSE) for complete terms.

For licensing inquiries or institutional partnerships: **clarence@ireadifybooks.com**
