"""
Export a small cross-field comparison bundle for the hub page.

Every field's own dashboard already computes collaboration-equity buckets,
year-by-year network stats (incl. small-world coefficient), and per-country
centrality (`dashboard/<field>/data.js`, written by export_dashboard_data.py).
This script doesn't recompute any of that -- it just reads each field's
already-published data.js and re-aggregates the handful of numbers the hub
page's comparison section needs into one small shared bundle
(`dashboard/shared/hub_comparison.js` -> `window.HUB_COMPARISON`).

Run after export_dashboard_data.py has been run for every field you want
included:
    python3 scripts/export_hub_comparison.py
"""

import json
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD_DIR = os.path.join(REPO_ROOT, "dashboard")
SHARED_DIR = os.path.join(DASHBOARD_DIR, "shared")

# Display name for each field slug -- kept here rather than derived, same as
# the field cards on dashboard/index.html (see README's "Adding a new field"
# step 6): a field's name is authored once, not computed. Color is NOT baked
# in here -- hub_compare.js assigns it by field position from the page's own
# CSS theme tokens, so it stays correct across light/dark mode.
FIELD_NAMES = {
    "meded": "Health Professions Education",
    "econ": "Economics & Business",
}

TOP_N_LEADERS = 5


def load_field_data(field):
    path = os.path.join(DASHBOARD_DIR, field, "data.js")
    with open(path) as f:
        content = f.read()
    match = re.match(r"window\.DASHBOARD_DATA\s*=\s*(.*);\s*$", content, re.DOTALL)
    return json.loads(match.group(1))


def build_field_summary(field, data):
    name = FIELD_NAMES.get(field, field)

    # Collaboration equity: normalize each year's High-only / Developing-only /
    # Mixed counts to shares, so fields of very different size are comparable.
    equity_series = []
    for row in sorted(data["collaboration_equity"], key=lambda r: r["year"]):
        total = sum(row.get(k, 0) for k in ("High only", "Developing only", "Mixed", "Unclassified"))
        if total == 0:
            continue
        equity_series.append({
            "year": row["year"],
            "high_only_share": row.get("High only", 0) / total,
            "developing_only_share": row.get("Developing only", 0) / total,
            "mixed_share": row.get("Mixed", 0) / total,
        })

    # Small-world coefficient: one value per cumulative-network year.
    small_world_series = [
        {"year": int(row["year"]), "value": row.get("Small-World Coefficient")}
        for row in sorted(data["network_yearly_stats"], key=lambda r: r["year"])
        if row.get("Small-World Coefficient") is not None
    ]

    # Top-N leaders by degree centrality in the full cumulative network.
    economy_by_iso = {c["ISO_A2_EH"]: c.get("Economy", c["ISO_A2_EH"]) for c in data["country_stats"]}
    leaders = sorted(
        (n for n in data["network_full"]["nodes"] if n.get("degree_centrality") is not None),
        key=lambda n: n["degree_centrality"],
        reverse=True,
    )[:TOP_N_LEADERS]
    top_leaders = [
        {
            "iso": n["id"],
            "economy": economy_by_iso.get(n["id"], n["id"]),
            "income_group": n.get("income_group"),
            "degree_centrality": n["degree_centrality"],
        }
        for n in leaders
    ]

    return {
        "slug": field,
        "name": name,
        "total_publications": data["summary"]["total_publications"],
        "total_countries": data["summary"]["total_countries"],
        "collaboration_equity_series": equity_series,
        "small_world_series": small_world_series,
        "top_leaders": top_leaders,
    }


def main():
    fields = sorted(
        d for d in os.listdir(DASHBOARD_DIR)
        if os.path.isfile(os.path.join(DASHBOARD_DIR, d, "data.js"))
    )
    if not fields:
        raise SystemExit("No dashboard/<field>/data.js found -- run export_dashboard_data.py per field first.")

    print(f"Aggregating comparison bundle for: {', '.join(fields)}")
    payload = {
        "fields": [build_field_summary(f, load_field_data(f)) for f in fields],
    }

    out_path = os.path.join(SHARED_DIR, "hub_comparison.js")
    with open(out_path, "w") as f:
        f.write("window.HUB_COMPARISON = ")
        f.write(json.dumps(payload, separators=(",", ":")))
        f.write(";\n")
    print(f"Wrote {out_path} ({os.path.getsize(out_path) / 1e3:.1f} KB)")


if __name__ == "__main__":
    main()
