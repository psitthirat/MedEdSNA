# MedEdSNA

**Decolonizing Health Professions Education Research through Social Network Analysis**

This repository hosts the data and scripts used in the study *“Decolonizing Health Professions Education Research: An Analysis of Global Network Patterns and Equity Implications”*. We apply **Social Network Analysis (SNA)** to examine authorship and collaboration patterns in health professions education (HPE) research, highlighting inequities and the global decolonization process.

---

## Background
Colonial legacies persist in medical education research, dominated by high-income countries (HICs). Traditional bibliometrics fail to capture power relations. SNA offers structural insights into inequities, identifying central actors, peripheral regions, and emerging knowledge brokers.

---

## Live Dashboard
An interactive dashboard (choropleth map + co-authorship network explorer) is published via GitHub Pages:

**https://psitthirat.github.io/MedEdSNA/**

It's a static D3.js app (`dashboard/`) that reads precomputed data bundles, so it needs no server or API calls. Any push to `dashboard/` on `main` redeploys it automatically via GitHub Actions (`.github/workflows/pages.yml`).

---

## Repository Structure
```
main.ipynb            # Main analytical notebook (bibliometrics + SNA)
subanalysis_thai.ipynb # Thailand-focused subanalysis

data/
  journal/             # Raw per-journal bibliometric exports (Scopus)
  world-map/           # Country grouping + world boundary/language reference data
  metadata/            # OpenAlex works/authorships metadata (gitignored — regenerate locally)

output/
  map/                 # Geocoded author/affiliation/institution coordinates
  scraped/             # Raw OpenAlex scrape output (gitignored — regenerate locally)

scripts/
  scrp_article.py       # Scrapes works/authorships from the OpenAlex API
  process.py             # Cleaning, parsing, geocode imputation utilities
  geocoder.py             # Affiliation NLP + geocoding (Google Maps API)
  network.py               # Co-authorship graph construction, centrality, community detection
  visualization.py          # Choropleth/bump-chart/network plotting helpers
  export_dashboard_data.py   # Exports notebook aggregations to dashboard/data.js & world.js

dashboard/             # Static D3.js dashboard, published via GitHub Pages
```

---

## Methods
- **Data Source**: HPE journals indexed via the OpenAlex API (2015–2026).
- **Unit**: Country-level co-authorship.
- **Analysis**: Bibliometrics, SNA (density, modularity, centrality), visualizations with NetworkX and D3.js.

---

## Key Findings
- HICs dominate the network core; LICs remain peripheral.
- South–South collaboration is minimal.
- Some LMICs (e.g., Kenya, Cambodia, Sudan) show emerging centrality.
- The field is in a transitional phase of decolonization.

---

## Usage
```bash
git clone https://github.com/psitthirat/MedEdSNA.git
cd MedEdSNA
pip install -r scripts/requirements.txt
```
Then open `main.ipynb` to reproduce the analysis, or run the pipeline scripts directly:
```bash
python scripts/scrp_article.py           # scrape OpenAlex works/authorships
python scripts/export_dashboard_data.py  # regenerate dashboard/data.js + world.js
```

---

## Citation
Sitthirat P, et al. *Decolonizing Health Professions Education Research: An Analysis of Global Network Patterns and Equity Implications.* (Preprint / Under Review).

---

## Equitable Partnership
This project was conceived, conducted, and written by a Thailand-based LMIC team with shared leadership, fair attribution, and inclusive dissemination.

---

## License
Released under the **MIT License**.
