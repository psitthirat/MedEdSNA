# MedEdSNA

**Decolonizing Health Professions Education Research through Social Network Analysis**

This repository hosts the data and scripts used in the study *“Decolonizing Health Professions Education Research: An Analysis of Global Network Patterns and Equity Implications”*. We apply **Social Network Analysis (SNA)** to examine authorship and collaboration patterns in health professions education (HPE) research, highlighting inequities and the global decolonization process.

---

## Background
Colonial legacies persist in medical education research, dominated by high-income countries (HICs). Traditional bibliometrics fail to capture power relations. SNA offers structural insights into inequities, identifying central actors, peripheral regions, and emerging knowledge brokers.

---

## Repository Structure
```
data/       # Contain bibliometric from included journal and countries' data
scripts/    # Python scripts (data processing, geocoding, network analysis)
main.ipynb  # Main analystical notebook
```

---

## Methods
- **Data Source**: Scopus-indexed Q1 HPE journals (2015–2024).
- **Unit**: Country-level co-authorship.
- **Analysis**: Bibliometrics, SNA (density, modularity, centrality), visualizations with NetworkX.

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
pip install -r requirements.txt
python scripts/run_analysis.py
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
