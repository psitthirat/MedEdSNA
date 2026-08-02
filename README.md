# OpenKnowledgeAtlas

**Mapping decolonization in academic knowledge networks, one field at a time**

This repository applies **Social Network Analysis (SNA)** to authorship and collaboration
patterns across academic fields, to surface structural inequities in global knowledge
production: who sits at the core of a field's collaboration network, who remains peripheral,
and who is structurally left out. Each field is analysed independently with the same method
and pipeline, so the project scales by adding fields rather than by rewriting itself.

It started as a single-field study — *"Decolonizing Health Professions Education Research: An
Analysis of Global Network Patterns and Equity Implications"* — and has since expanded to a
second field (Economics & Business), with the pipeline and dashboard now built to take a third,
fourth, or Nth field without restructuring.

---

## Background
Colonial legacies persist in academic research, dominated by high-income countries (HICs).
Traditional bibliometrics fail to capture power relations. SNA offers structural insights into
inequities, identifying central actors, peripheral regions, and emerging knowledge brokers —
a lens that applies as much to economics or clinical science as it does to health professions
education.

---

## Live Dashboards
Interactive dashboards (choropleth map + co-authorship network explorer) are published via
GitHub Pages, one per field plus a hub page:

- **https://psitthirat.github.io/OpenKnowledgeAtlas/** — hub, links to every field
- **https://psitthirat.github.io/OpenKnowledgeAtlas/meded/** — Health Professions Education
- **https://psitthirat.github.io/OpenKnowledgeAtlas/econ/** — Economics & Business

Each is a static D3.js app that reads precomputed data bundles, so it needs no server or API
calls. Any push to `dashboard/` on `main` redeploys it automatically via GitHub Actions
(`.github/workflows/pages.yml`).

---

## Repository Structure
```
data/
  source_id.csv        # Field registry: which journals/sources belong to which field (the `group` column)
  world-map/            # Reference data shared by every field: country boundaries, languages, income groups
  metadata/               # Per-field cleaned works/authorships (gitignored — regenerate via each field's pipeline notebook)
    meded/, econ/, ...

output/
  scraped/              # Per-field raw OpenAlex scrape output (gitignored — regenerate locally)
    meded/, econ/, ...
  map/                  # Global institution/affiliation/author -> country-code crosswalks (shared across every field, see below)
    pending/             # Scratch exports from scripts/geocode_workflow.py, consumed by merge_labels()

fields/
  meded/pipeline.ipynb   # Scrape -> clean -> geocode -> SNA for Health Professions Education
  econ/pipeline.ipynb    # Same pipeline, parameterized for Economics & Business

scripts/
  scrp_article.py         # Scrapes works/authorships from the OpenAlex API
  geocode_workflow.py       # Fills institution_country_code from the shared crosswalk; export/merge loop for new institutions (see below)
  process.py                 # Affiliation-string parsing utilities
  geocoder.py                  # NLP + Google Maps geocoding (legacy fallback, not used by the current pipeline — see legacy/README.md)
  network.py                     # Co-authorship graph construction, centrality, community detection
  visualization.py                # Choropleth/bump-chart/network plotting helpers (used inside each field's notebook)
  export_dashboard_data.py         # Exports one field's aggregations to dashboard/<field>/data.js + dashboard/shared/world.js

dashboard/
  index.html            # Hub page, links to each field
  shared/                # style.css, app.js, world.js, vendor/ -- identical across every field, not duplicated
  meded/, econ/           # One index.html + data.js per field

legacy/                # Pre-OpenAlex, Scopus-era pipeline and the original Thailand subanalysis -- kept for reference, not maintained (see legacy/README.md)
```

---

## What's global vs. per-field, and why
- **Per-field** (namespaced under a field slug like `meded` or `econ`): scraped/cleaned data,
  the pipeline notebook, the dashboard page and its `data.js`. These are specific to one field's
  journals and can't be shared.
- **Global** (flat, no field namespace, intentionally): `data/world-map/` (country boundaries,
  income groups), `data/source_id.csv` (the field registry itself), and the three crosswalks in
  `output/map/` (`geocode_ins.csv`, `geocode_aff.csv`, `geocode_author.csv`). An institution
  doesn't change country because a new field started scraping — a code learned once for
  "Harvard University" should never need re-labeling for the next field. Keeping these global is
  what makes adding a field cheap after the first one or two.

---

## The geocoding workflow
Bibliometric records don't always carry a clean institution country code. The pipeline resolves
`institution_country_code` in four stages, run automatically by
`geocode_workflow.impute_country_codes()` inside each field's pipeline notebook:
1. Look up `institution_id` in the shared `output/map/geocode_ins.csv` crosswalk.
2. Fall back to `raw_affiliation` text in `output/map/geocode_aff.csv`.
3. Fall back to `author_id` in `output/map/geocode_author.csv` (per-author manual override).
4. Iteratively infer from the same author's other works (nearest publication year) and
   co-authors on the same work (majority country).

Whatever's still missing after that is a genuinely new institution/affiliation the crosswalk
hasn't seen before. Rather than hand-exporting a CSV, labeling it in a separate chat, and
hand-merging it back in (which is how this repo used to do it, and how it drifted — see
`legacy/README.md`), the loop is now:

```bash
python3 -m scripts.geocode_workflow export --field econ --kind institution
python3 -m scripts.geocode_workflow export --field econ --kind affiliation
# Fill in the institution_country_code column of the printed pending file --
# e.g. by asking Claude Code to label it, in a session against this repo,
# the same way this tooling itself was built and exercised.
python3 -m scripts.geocode_workflow merge --kind institution \
    --pending output/map/pending/geocode_institution_econ.csv
python3 -m scripts.geocode_workflow merge --kind affiliation \
    --pending output/map/pending/geocode_affiliation_econ.csv
# Re-run the pipeline notebook's geocoding cell to pick up the newly-labeled rows.
```

`export_pending` only ever returns institutions/affiliations *not already* in the crosswalk, so
re-running it doesn't regenerate a bloated list of things you've already labeled.

---

## Adding a new field
1. Add the field's journal/source IDs to `data/source_id.csv` under a new `group` value.
2. Copy `fields/econ/pipeline.ipynb` to `fields/<field>/pipeline.ipynb`, change `group = "econ"`
   to `group = "<field>"`, and run it top to bottom (scrape → clean → geocode → SNA).
3. If rows are still missing a country code after the notebook's automatic imputation, run the
   geocode export/label/merge loop above for the new institutions/affiliations, then re-run the
   notebook's geocoding cell.
4. Run `python3 -m scripts.export_dashboard_data --field <field>` to generate
   `dashboard/<field>/data.js`.
5. Copy `dashboard/econ/` to `dashboard/<field>/`, edit the hero copy and meta tags in
   `index.html` (paths already point at `../shared/`, no need to touch those).
6. Add a card for the new field on `dashboard/index.html`'s hub page.

---

## Methods
- **Data Source**: Field-specific journals indexed via the OpenAlex API (2015–2026).
- **Unit**: Country-level co-authorship.
- **Analysis**: Bibliometrics, SNA (density, modularity, centrality), visualizations with NetworkX and D3.js.

---

## Key Findings — Health Professions Education
- HICs dominate the network core; LICs remain peripheral.
- South–South collaboration is minimal.
- Some LMICs (e.g., Kenya, Cambodia, Sudan) show emerging centrality.
- The field is in a transitional phase of decolonization.

Economics & Business findings are still being written up — see the [live dashboard](https://psitthirat.github.io/OpenKnowledgeAtlas/econ/) for the current data.

---

## Usage
```bash
git clone https://github.com/psitthirat/OpenKnowledgeAtlas.git
cd OpenKnowledgeAtlas
pip install -r scripts/requirements.txt
```
Then open a field's notebook (e.g. `fields/meded/pipeline.ipynb`) to reproduce that field's
analysis, or run the pipeline scripts directly for a given field:
```bash
python3 -m scripts.export_dashboard_data --field meded   # regenerate dashboard/meded/data.js
```

---

## Citation
**Health Professions Education study**: Sitthirat P, et al. *Decolonizing Health Professions
Education Research: An Analysis of Global Network Patterns and Equity Implications.* (Preprint /
Under Review).

---

## Equitable Partnership
This project was conceived, conducted, and written by a Thailand-based LMIC team with shared leadership, fair attribution, and inclusive dissemination.

---

## License
Released under the **MIT License**.
