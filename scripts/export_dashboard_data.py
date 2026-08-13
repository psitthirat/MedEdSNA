"""
Export precomputed dashboard data for one field.

Reproduces the aggregations built in each field's `fields/<field>/pipeline.ipynb`
(sections 3-5: country-level analysis, co-authorship network, and
concentration) and writes them to a per-field JS file
(`dashboard/<field>/data.js`) that assigns the payload to
`window.DASHBOARD_DATA`, plus a simplified world boundary file shared by every
field (`dashboard/shared/world.js` -> `window.WORLD_GEOJSON`, identical
regardless of field so it's written once, not duplicated). Both are loaded
via <script> tags so each field's dashboard page works from a plain `file://`
URL with no server and no fetch/CORS concerns.

Run from the project root with the project venv active:
    python3 scripts/export_dashboard_data.py --field meded
    python3 scripts/export_dashboard_data.py --field econ
"""

import argparse
import json
import os

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import community as community_louvain
from shapely.geometry.polygon import orient
from shapely.geometry import MultiPolygon, Polygon

from scripts import network

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD_DIR = os.path.join(REPO_ROOT, "dashboard")
SHARED_DIR = os.path.join(DASHBOARD_DIR, "shared")
os.makedirs(SHARED_DIR, exist_ok=True)

EARLY_WINDOW = range(2015, 2020)   # 2015-2019 snapshot for the map's "early period" network
LATEST_FULL_YEAR = 2025            # 2026 is a partial year in the source data
TOP_N_BUMP = 10
BUMP_START_YEAR = 2020             # rank-trajectory bump charts start here, not at the network's true first year

# World Bank economies whose ISO3 code has no Natural Earth ADM0_A3 match, so
# the crosswalk below would leave them with a NaN ISO_A2_EH. See the comment at
# the merge for why a NaN there is not merely a missing label but a data bug.
ISO2_PATCH = {
    "BHS": "BS", "CHI": "JE", "COG": "CG", "CPV": "CV", "CUW": "CW",
    "CZE": "CZ", "GNB": "GW", "MAC": "MO", "MKD": "MK", "SSD": "SS",
    "STP": "ST", "SWZ": "SZ", "TZA": "TZ", "VIR": "VI", "XKX": "XK",
}

INCOME_COLORS = {
    "High income": "#4C9F70",
    "Upper middle income": "#D9A441",
    "Lower middle income": "#C1666B",
    "Low income": "#7B4B94",
    "Other": "#9AA5B1",
}


def to_records(df):
    return json.loads(df.to_json(orient="records"))


def gini(values):
    """Gini coefficient of a 1-D array of non-negative values.

    Used instead of the small-world coefficient to answer "is production
    distributed?": sigma is confounded by density and reads a star topology --
    maximal centralisation -- as low, so it cannot separate an equitable
    network from a hub-dominated one.
    """
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return None
    return float((2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum())) - (n + 1) / n)


def main(field):
    out_dir = os.path.join(DASHBOARD_DIR, field)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading base map + language data...")
    world = gpd.read_file(os.path.join(REPO_ROOT, "data/world-map/ne_10m_admin_0_countries.zip"))
    world = world[world["ADMIN"] != "Antarctica"]
    language = gpd.read_file(os.path.join(REPO_ROOT, "data/world-map/world_languages.zip"))
    world = world.merge(language[["COUNTRY", "FIRST_OFFI"]], how="left", left_on="GEOUNIT", right_on="COUNTRY")

    print(f"Loading works/authorships for field '{field}'...")
    works_df = pd.read_csv(os.path.join(REPO_ROOT, "data/metadata", field, "works.csv"))
    authorships_df = pd.read_csv(os.path.join(REPO_ROOT, "data/metadata", field, "authorships.csv"))

    income_gr = pd.read_csv(os.path.join(REPO_ROOT, "data/world-map/country_group.csv"))
    income_gr = income_gr.merge(
        world[["ADM0_A3", "FIRST_OFFI", "ISO_A2_EH"]], how="left", left_on="Code", right_on="ADM0_A3"
    )
    # pandas joins NaN to NaN on a merge key. Without this, every authorship row
    # whose country code never resolved would match all 15 unmatched economies
    # and be duplicated once per spurious income group -- which inflated the
    # low-income series roughly threefold. Patch the codes we can recover, then
    # drop the rest so the key is unique and non-null before it is joined on.
    income_gr["ISO_A2_EH"] = income_gr["ISO_A2_EH"].fillna(income_gr["Code"].map(ISO2_PATCH))
    income_gr = income_gr.dropna(subset=["ISO_A2_EH"]).drop_duplicates("ISO_A2_EH")

    df = authorships_df.merge(
        income_gr[["ISO_A2_EH", "Income group", "Region", "Economy"]],
        how="left", left_on="institution_country_code", right_on="ISO_A2_EH",
    )
    df = df.merge(
        world[["ISO_A2_EH", "FIRST_OFFI"]].dropna(subset=["ISO_A2_EH"]).drop_duplicates("ISO_A2_EH"),
        how="left", on="ISO_A2_EH",
    )
    # These merges are lookups, not joins -- they must never change the row count.
    assert len(df) == len(authorships_df), "income/language merge duplicated authorship rows"
    df = df.merge(
        works_df[["work_id", "publication_year", "journal_name"]].drop_duplicates("work_id"),
        how="left", on="work_id", suffixes=("", "_w"),
    )
    df["publication_year"] = df["publication_year"].fillna(df["publication_year_w"])
    df = df[df["publication_year"] <= LATEST_FULL_YEAR + 1]  # keep the partial latest year for trend context

    payload = {}

    # ------------------------------------------------------------------
    # 1. Summary tiles
    # ------------------------------------------------------------------
    print("Building summary...")
    payload["summary"] = {
        "total_publications": int(df["work_id"].nunique()),
        "total_authors": int(df["author_id"].nunique()),
        "total_countries": int(df["ISO_A2_EH"].dropna().nunique()),
        "total_institutions": int(df["institution_id"].dropna().nunique()),
        "total_journals": int(works_df["journal_name"].nunique()),
        "journal_names": sorted(works_df["journal_name"].dropna().unique().tolist()),
        "year_min": int(df["publication_year"].min()),
        "year_max": int(df["publication_year"].max()),
        "date_min": works_df["publication_date"].min(),
        "date_max": works_df["publication_date"].max(),
    }

    # ------------------------------------------------------------------
    # 2. Yearly trends by income group
    # ------------------------------------------------------------------
    print("Building yearly trends...")
    pubs = (
        df.groupby(["publication_year", "Income group"])["work_id"]
        .nunique().reset_index(name="publications")
    )
    authors = (
        df.groupby(["publication_year", "Income group"])["author_id"]
        .nunique().reset_index(name="authors")
    )
    payload["yearly_trends"] = {
        "publications": to_records(pubs.rename(columns={"publication_year": "year"})),
        "authors": to_records(authors.rename(columns={"publication_year": "year"})),
    }

    # ------------------------------------------------------------------
    # 3. Country-level publication counts (choropleth)
    # ------------------------------------------------------------------
    print("Building country stats...")
    country_pubs = (
        df.dropna(subset=["ISO_A2_EH"])
        .groupby("ISO_A2_EH")
        .agg(
            publications=("work_id", "nunique"),
            authors=("author_id", "nunique"),
        )
        .reset_index()
    )
    country_meta = income_gr.dropna(subset=["ISO_A2_EH"]).drop_duplicates("ISO_A2_EH")[
        ["ISO_A2_EH", "Economy", "Region", "Income group"]
    ]
    country_stats = country_pubs.merge(country_meta, on="ISO_A2_EH", how="left")
    country_stats["log_publications"] = np.log1p(country_stats["publications"])
    payload["country_stats"] = to_records(country_stats)

    country_year = (
        df.dropna(subset=["ISO_A2_EH"])
        .groupby(["ISO_A2_EH", "publication_year"])["work_id"]
        .nunique().reset_index(name="publications")
        .rename(columns={"publication_year": "year"})
    )
    payload["country_year_stats"] = to_records(country_year)

    # ------------------------------------------------------------------
    # 4. Collaboration equity buckets over time
    # ------------------------------------------------------------------
    print("Building collaboration-equity buckets...")
    work_income = df.dropna(subset=["work_id"]).groupby("work_id").agg(
        year=("publication_year", "first"),
        income_groups=("Income group", lambda x: set(x.dropna())),
        n_countries=("ISO_A2_EH", lambda x: x.dropna().nunique()),
    ).reset_index()

    def classify(groups):
        if not groups:
            return "Unclassified"
        if groups == {"High income"}:
            return "High only"
        if "High income" not in groups:
            return "Developing only"
        return "Mixed"

    work_income["bucket"] = work_income["income_groups"].apply(classify)
    equity = work_income.pivot_table(index="year", columns="bucket", aggfunc="size", fill_value=0).reset_index()
    payload["collaboration_equity"] = to_records(equity)

    # ------------------------------------------------------------------
    # 5. Multi-country collaboration (UpSet-style combination counts)
    # ------------------------------------------------------------------
    # Includes both cross-income-group collaborations (e.g. High + Low) and
    # within-group ones (e.g. two High-income countries, or two Low-income
    # countries, co-authoring) -- any work with more than one country.
    print("Building UpSet combination counts...")
    income_levels = ["High income", "Upper middle income", "Lower middle income", "Low income"]
    multi = work_income[work_income["n_countries"] > 1].copy()
    for lvl in income_levels:
        multi[lvl] = multi["income_groups"].apply(lambda s, lvl=lvl: lvl in s)
    combo_counts = multi.groupby(income_levels).size().reset_index(name="count")
    combo_counts = combo_counts[combo_counts["count"] > 0]
    payload["upset_combinations"] = to_records(combo_counts)

    # ------------------------------------------------------------------
    # 6. Collaboration topology + within-bloc concentration
    #
    #    The equity buckets above count a "Developing only" work the same
    #    whether it joined four countries or none. Splitting each bucket by
    #    country spread separates South-South *collaboration* from parallel
    #    domestic production, and comparing concentration inside each bloc
    #    tests whether the non-high-income bloc is any more evenly
    #    distributed than the one it is usually contrasted with.
    # ------------------------------------------------------------------
    print("Building collaboration topology + bloc concentration...")
    topo = work_income[work_income["bucket"] != "Unclassified"].copy()
    topo["is_multi"] = topo["n_countries"] > 1
    topology = (
        topo.groupby("bucket")
        .agg(works=("is_multi", "size"), multi_country=("is_multi", "sum"))
        .reset_index()
    )
    topology["single_country"] = topology["works"] - topology["multi_country"]
    topology["multi_country_pct"] = (100 * topology["multi_country"] / topology["works"]).round(1)
    payload["collaboration_topology"] = {
        "buckets": to_records(topology),
        "total_works": int(len(topo)),
    }

    economy_of = country_meta.set_index("ISO_A2_EH")["Economy"]
    bloc_masks = {
        "High income": df["Income group"] == "High income",
        "Non-high income": df["Income group"].notna() & (df["Income group"] != "High income"),
    }
    bloc_rows = []
    for label, mask in bloc_masks.items():
        counts = df[mask].groupby("ISO_A2_EH")["work_id"].nunique().sort_values(ascending=False)
        if counts.empty:
            continue
        bloc_rows.append({
            "bloc": label,
            "countries": int(len(counts)),
            "works": int(counts.sum()),
            "top1_code": counts.index[0],
            "top1_economy": economy_of.get(counts.index[0], counts.index[0]),
            "top1_pct": round(100 * counts.iloc[0] / counts.sum(), 1),
            "top3_pct": round(100 * counts.head(3).sum() / counts.sum(), 1),
            "top5_pct": round(100 * counts.head(5).sum() / counts.sum(), 1),
            "gini": round(gini(counts.values), 3),
        })
    payload["bloc_concentration"] = bloc_rows

    # ------------------------------------------------------------------
    # 7. Country-level co-authorship network (cumulative, all years)
    # ------------------------------------------------------------------
    print("Building co-authorship network (this runs centrality + community detection, may take a bit)...")
    df_net = df

    def build_network_payload(df_subset, label):
        G = network.network_coauthorship(df_subset, "institution_country_code", node_label=["Region", "Income group", "FIRST_OFFI"])
        deg = nx.degree_centrality(G)
        btw = nx.betweenness_centrality(G)
        cls = nx.closeness_centrality(G)
        try:
            eig = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)
        except (nx.PowerIterationFailedConvergence, nx.NetworkXPointlessConcept):
            # PowerIterationFailedConvergence: didn't converge on a real graph.
            # NetworkXPointlessConcept: the null graph (e.g. a field with zero
            # low-income-first-author works has no nodes at all to build a
            # subnetwork from) -- eigenvector centrality is undefined either way.
            eig = {n: np.nan for n in G.nodes}
        partition = community_louvain.best_partition(G, random_state=42) if G.number_of_nodes() else {}
        pc = network.participation_coefficient(G, partition) if partition else {}

        nodes = []
        for node, data in G.nodes(data=True):
            nodes.append({
                "id": node,
                "size": data.get("size", 1),
                "region": data.get("Region"),
                "income_group": data.get("Income group"),
                "language": data.get("FIRST_OFFI"),
                "degree_centrality": deg.get(node),
                "betweenness_centrality": btw.get(node),
                "closeness_centrality": cls.get(node),
                "eigenvector_centrality": eig.get(node),
                "participation_coefficient": pc.get(node),
                "community": partition.get(node),
            })
        edges = [{"source": u, "target": v, "weight": d.get("weight", 1)} for u, v, d in G.edges(data=True)]

        stats_df = network.network_params(
            df_subset, G, homophily_attr=["Region", "Income group", "FIRST_OFFI"]
        )
        stats = json.loads(stats_df.to_json(orient="records"))[0]
        stats.pop("None", None)

        print(f"  {label}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return {"nodes": nodes, "edges": edges, "stats": stats}

    payload["network_full"] = build_network_payload(df_net, "full network")

    # LIC-first-author-only subnetwork
    lic_first_dois = set(
        df_net[(df_net["Income group"] == "Low income") & (df_net["author_position"] == "first")]["work_id"]
    )
    df_lic = df_net[df_net["work_id"].isin(lic_first_dois)]
    payload["network_lic_first_author"] = build_network_payload(df_lic, "LIC-first-author network")

    df_early_full = df_net[df_net["publication_year"].isin(EARLY_WINDOW)]
    payload["network_early_period"] = build_network_payload(df_early_full, "early-period network (2015-2019)")

    # ------------------------------------------------------------------
    # 8. Network structure over time + rank trajectories: cumulative
    #    network rebuilt through each year from the first year to the
    #    last (mirrors the notebook's year-by-year G_all accumulation).
    #    One pass builds both the graph-level stat time series and the
    #    per-country centrality used by the rank-trajectory bump charts.
    # ------------------------------------------------------------------
    print("Building yearly cumulative network stats + rank trajectories...")
    country_lookup = income_gr.dropna(subset=["ISO_A2_EH"]).drop_duplicates("ISO_A2_EH").set_index("ISO_A2_EH")
    first_year = int(df_net["publication_year"].min())
    last_year = int(df_net["publication_year"].max())

    yearly_rows = []
    rank_rows = []
    concentration_rows = []
    for y in range(first_year, last_year + 1):
        df_cum = df_net[df_net["publication_year"].between(first_year, y)]
        G_cum = network.network_coauthorship(df_cum, "institution_country_code", node_label=["Region", "Income group"])
        stats_row = network.network_params(df_cum, G_cum, homophily_attr=["Region", "Income group"])
        stats_row["year"] = y
        yearly_rows.append(stats_row)

        deg = nx.degree_centrality(G_cum)
        btw = nx.betweenness_centrality(G_cum)
        cls = nx.closeness_centrality(G_cum)
        for country in G_cum.nodes:
            rank_rows.append({
                "year": y,
                "country": country,
                "economy": country_lookup["Economy"].get(country, country),
                "income_group": country_lookup["Income group"].get(country, "Other"),
                "degree": deg.get(country),
                "betweenness": btw.get(country),
                "closeness": cls.get(country),
            })

        # Freeman degree centralisation, on a COPY that also carries countries
        # publishing only domestically. network_coauthorship creates nodes from
        # edges, so those countries are absent from G_cum -- leaving them out
        # would hide exactly the periphery a centralisation measure is for.
        # The copy keeps G_cum itself untouched for the stats/centrality above.
        if y <= LATEST_FULL_YEAR:  # a partial year would read as a collapse
            G_with_isolates = G_cum.copy()
            G_with_isolates.add_nodes_from(df_cum["institution_country_code"].dropna().unique())
            degrees = np.array([d for _, d in G_with_isolates.degree()])
            n_nodes = len(degrees)

            df_year = df_net[df_net["publication_year"] == y]
            nonhic = df_year[df_year["Income group"].notna() & (df_year["Income group"] != "High income")]
            hic = df_year[df_year["Income group"] == "High income"]

            def works_per_country(d):
                return d.groupby("institution_country_code")["work_id"].nunique().values

            concentration_rows.append({
                "year": y,
                "countries": int(n_nodes),
                "freeman_centralisation": round(float((degrees.max() - degrees).sum() / ((n_nodes - 1) * (n_nodes - 2))), 3) if n_nodes > 2 else None,
                "gini_all_cumulative": round(gini(works_per_country(df_cum)), 3),
                "gini_non_high_income": round(gini(works_per_country(nonhic)), 3),
                "gini_high_income": round(gini(works_per_country(hic)), 3),
            })

        print(f"  {y}: {G_cum.number_of_nodes()} nodes, {G_cum.number_of_edges()} edges")

    payload["concentration_timeseries"] = concentration_rows

    yearly_stats_df = pd.concat(yearly_rows, ignore_index=True)
    payload["network_yearly_stats"] = json.loads(yearly_stats_df.to_json(orient="records"))

    # ------------------------------------------------------------------
    # 9. Rank-trajectory bump charts: each top-10 country's rank across
    #    every year from first to last (a full rise/fall trajectory, not
    #    just a two-point before/after comparison).
    # ------------------------------------------------------------------
    print("Building rank-trajectory bump charts...")
    rank_df = pd.DataFrame(rank_rows)
    bump_first_year = max(BUMP_START_YEAR, first_year)
    bump_charts_multi = {"first_year": bump_first_year, "last_year": last_year}
    for metric in ["degree", "betweenness", "closeness"]:
        m = (
            rank_df[["year", "country", "economy", "income_group", metric]]
            .rename(columns={metric: "score"})
            .dropna(subset=["score"])
        )
        m["rank"] = m.groupby("year")["score"].rank(ascending=False, method="min").astype(int)
        m = m[m["year"] >= bump_first_year]

        first_ranks = m[m["year"] == bump_first_year].set_index("country")["rank"].nsmallest(TOP_N_BUMP)
        last_ranks = m[m["year"] == last_year].set_index("country")["rank"].nsmallest(TOP_N_BUMP)
        keep = set(first_ranks.index) | set(last_ranks.index)

        series = m[m["country"].isin(keep)].sort_values(["country", "year"])
        bump_charts_multi[metric] = {
            "first_top10": first_ranks.index.tolist(),
            "last_top10": last_ranks.index.tolist(),
            "series": to_records(series),
        }

    payload["bump_charts_multi"] = bump_charts_multi

    payload["income_colors"] = INCOME_COLORS
    payload["generated_at"] = pd.Timestamp.utcnow().isoformat()

    # ------------------------------------------------------------------
    # Write data.js
    # ------------------------------------------------------------------
    data_path = os.path.join(out_dir, "data.js")
    with open(data_path, "w") as f:
        f.write("window.DASHBOARD_DATA = ")
        f.write(json.dumps(payload, separators=(",", ":")))
        f.write(";\n")
    print(f"Wrote {data_path} ({os.path.getsize(data_path) / 1e6:.2f} MB)")

    # ------------------------------------------------------------------
    # Write simplified world boundaries -- identical across every field, so
    # this lives in dashboard/shared/ and is simply overwritten (not
    # duplicated per field) each time this script runs.
    # ------------------------------------------------------------------
    print("Simplifying world boundaries...")
    world_geo = world[["ADMIN", "ADM0_A3", "ISO_A2_EH", "GEOUNIT", "geometry"]].copy()
    world_geo["geometry"] = world_geo.geometry.simplify(0.05, preserve_topology=True)

    def drop_degenerate_parts(geom):
        # Simplification can collapse a tiny island down to a near-degenerate
        # ring (e.g. 3 unique points). Its signed area becomes numerically
        # unreliable, so d3-geo's antimeridian clipping can render it as
        # covering the whole globe instead of a few invisible pixels. Cheaper
        # and safer to drop slivers than to trust their winding.
        if geom is None or geom.is_empty:
            return geom
        polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
        kept = [p for p in polys if len(p.exterior.coords) >= 5 and p.area > 1e-4]
        if not kept:
            kept = max(polys, key=lambda p: p.area, default=None)
            return kept
        return kept[0] if len(kept) == 1 else MultiPolygon(kept)

    world_geo["geometry"] = world_geo["geometry"].apply(drop_degenerate_parts)

    def rewind(geom):
        # d3-geo's antimeridian clipping expects exterior rings wound CLOCKWISE
        # in (lon, lat) plane -- the opposite of the RFC 7946 GeoJSON spec.
        # Without this every polygon renders as its own complement (the whole
        # map minus a hole where the country actually is).
        if geom is None or geom.is_empty:
            return geom
        if isinstance(geom, Polygon):
            return orient(geom, sign=-1.0)
        if isinstance(geom, MultiPolygon):
            return MultiPolygon([orient(p, sign=-1.0) for p in geom.geoms])
        return geom

    world_geo["geometry"] = world_geo["geometry"].apply(rewind)
    gj = json.loads(world_geo.to_json())

    def round_coords(obj):
        if isinstance(obj, list):
            if obj and isinstance(obj[0], (int, float)):
                return [round(x, 3) for x in obj]
            return [round_coords(x) for x in obj]
        return obj

    for feat in gj["features"]:
        feat["geometry"]["coordinates"] = round_coords(feat["geometry"]["coordinates"])

    world_path = os.path.join(SHARED_DIR, "world.js")
    with open(world_path, "w") as f:
        f.write("window.WORLD_GEOJSON = ")
        f.write(json.dumps(gj, separators=(",", ":")))
        f.write(";\n")
    print(f"Wrote {world_path} ({os.path.getsize(world_path) / 1e6:.2f} MB)")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--field", required=True, help="Field slug matching a data/metadata/<field>/ folder, e.g. meded, econ")
    main(parser.parse_args().field)
