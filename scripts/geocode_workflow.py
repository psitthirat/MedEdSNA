"""
Geocode crosswalk workflow, shared across every field.

The three crosswalks in output/map/ (geocode_ins.csv, geocode_aff.csv,
geocode_author.csv) map institution_id / raw_affiliation / author_id to an
institution_country_code. They are global and NOT field-namespaced on
purpose: an institution's country doesn't change because a new field started
scraping, so a code learned once should never need re-labeling.

This module replaces the old per-notebook, copy/paste-into-a-chat workflow
with three steps:

    1. impute_country_codes(authorships_df) -- fills institution_country_code
       from the crosswalks plus same-author/same-work inference. Call this
       right after scraping a field.
    2. export_pending(authorships_df, kind, field) -- for whatever's still
       missing, writes ONLY the institutions/affiliations not already in the
       crosswalk (deduplicated) to output/map/pending/geocode_{kind}_{field}.csv.
    3. Fill in the institution_country_code column of that pending file --
       e.g. by asking Claude (in a chat or a Claude Code session, same as
       this repo's own history) to label the institution/affiliation names --
       then call merge_labels(kind, pending_path) to fold the labeled rows
       back into the global crosswalk. Re-run impute_country_codes to pick
       them up.

CLI (from the repo root):
    python3 -m scripts.geocode_workflow export --field econ --kind institution
    python3 -m scripts.geocode_workflow merge --kind institution \
        --pending output/map/pending/geocode_institution_econ.csv
"""

import argparse
import os

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP_DIR = os.path.join(REPO_ROOT, "output", "map")
PENDING_DIR = os.path.join(MAP_DIR, "pending")

CROSSWALKS = {
    "institution": {
        "path": os.path.join(MAP_DIR, "geocode_ins.csv"),
        "key": "institution_id",
        "extra_cols": ["institution_name"],
    },
    "affiliation": {
        "path": os.path.join(MAP_DIR, "geocode_aff.csv"),
        "key": "raw_affiliation",
        "extra_cols": [],
    },
    "author": {
        "path": os.path.join(MAP_DIR, "geocode_author.csv"),
        "key": "author_id",
        "extra_cols": [],
    },
}


def _load_crosswalk(kind):
    spec = CROSSWALKS[kind]
    cols = [spec["key"]] + spec["extra_cols"] + ["institution_country_code"]
    if os.path.exists(spec["path"]):
        df = pd.read_csv(spec["path"])
        return df[[c for c in cols if c in df.columns]]
    return pd.DataFrame(columns=cols)


def impute_country_codes(authorships_df, max_iterations=100):
    """
    Fill missing institution_country_code values, in order: institution_id
    crosswalk, raw_affiliation crosswalk, author_id crosswalk, then
    iterative same-author (nearest publication year) / same-work (majority
    country) inference. Returns a filled copy; does not mutate the input.
    """
    df = authorships_df.copy()

    def missing():
        return int(df["institution_country_code"].isna().sum())

    print(f"There are {missing():,} rows missing the country code before crosswalk imputation.")

    ins = _load_crosswalk("institution")
    if not ins.empty:
        ins_map = (
            ins.dropna(subset=["institution_id", "institution_country_code"])
            .drop_duplicates(subset=["institution_id"], keep="first")
            .set_index("institution_id")["institution_country_code"]
        )
        df["institution_country_code"] = df["institution_country_code"].fillna(df["institution_id"].map(ins_map))
        print(f"  after institution_id mapping: {missing():,} rows missing.")

    aff = _load_crosswalk("affiliation")
    if not aff.empty:
        aff_map = (
            aff.dropna(subset=["raw_affiliation", "institution_country_code"])
            .drop_duplicates(subset=["raw_affiliation"], keep="first")
            .set_index("raw_affiliation")["institution_country_code"]
        )
        df["institution_country_code"] = df["institution_country_code"].fillna(df["raw_affiliation"].map(aff_map))
        print(f"  after raw_affiliation mapping: {missing():,} rows missing.")

    author = _load_crosswalk("author")
    if not author.empty:
        author_map = (
            author.dropna(subset=["author_id", "institution_country_code"])
            .drop_duplicates(subset=["author_id"], keep="first")
            .set_index("author_id")["institution_country_code"]
        )
        df["institution_country_code"] = df["institution_country_code"].fillna(df["author_id"].map(author_map))
        print(f"  after author_id mapping: {missing():,} rows missing.")

    print("Starting iterative country-code inference (same-author nearest-year, same-work majority)...")
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        missing_before = missing()

        missing_mask = df["institution_country_code"].isna() & df["author_id"].notna()
        missing_rows = (
            df.loc[missing_mask, ["author_id", "publication_year"]]
            .reset_index()
            .rename(columns={"index": "row_idx", "publication_year": "missing_year"})
        )
        known_rows = (
            df.loc[
                df["institution_country_code"].notna() & df["author_id"].notna(),
                ["author_id", "publication_year", "institution_country_code"],
            ]
            .rename(columns={"publication_year": "known_year", "institution_country_code": "known_country_code"})
            .drop_duplicates()
        )
        if not missing_rows.empty and not known_rows.empty:
            candidates = missing_rows.merge(known_rows, on="author_id", how="inner")
            if not candidates.empty:
                candidates["year_diff"] = (candidates["missing_year"] - candidates["known_year"]).abs().fillna(float("inf"))
                candidates = candidates.sort_values(
                    by=["row_idx", "year_diff", "known_year"], ascending=[True, True, False], kind="stable"
                )
                nearest_country = (
                    candidates.drop_duplicates(subset=["row_idx"], keep="first").set_index("row_idx")["known_country_code"]
                )
                df.loc[nearest_country.index, "institution_country_code"] = nearest_country.to_numpy()

        most_frequent_country_per_work = (
            df.dropna(subset=["work_id", "institution_country_code"])
            .groupby("work_id")["institution_country_code"]
            .agg(lambda x: x.value_counts().index[0])
        )
        same_work_country = df["work_id"].map(most_frequent_country_per_work)
        df["institution_country_code"] = df["institution_country_code"].fillna(same_work_country)

        filled_this_iteration = missing_before - missing()
        print(f"  iteration {iteration}: filled {filled_this_iteration:,}, {missing():,} rows still missing.")
        if filled_this_iteration == 0:
            break

    print(f"Done. {missing():,} rows still missing a country code after {iteration} iteration(s).")
    return df


def export_pending(authorships_df, kind, field):
    """
    Write the institutions/affiliations/authors from authorships_df that
    still lack a country code AND aren't already in the global crosswalk to
    output/map/pending/geocode_{kind}_{field}.csv, with an empty
    institution_country_code column ready to be filled in before calling
    merge_labels.
    """
    spec = CROSSWALKS[kind]
    key = spec["key"]
    crosswalk = _load_crosswalk(kind)
    # A key only counts as "known" once it has an actual country code -- a
    # crosswalk row can exist but still be blank (e.g. carried over from a
    # partially-labeled export), and that should still come back as pending.
    known_keys = (
        set(crosswalk.loc[crosswalk["institution_country_code"].notna(), key]) if not crosswalk.empty else set()
    )

    candidate_cols = [key] + spec["extra_cols"]
    if kind == "institution":
        mask = authorships_df["institution_country_code"].isna() & authorships_df["institution_id"].notna()
    elif kind == "affiliation":
        mask = (
            authorships_df["institution_country_code"].isna()
            & authorships_df["institution_id"].isna()
            & authorships_df["raw_affiliation"].notna()
        )
    else:
        mask = authorships_df["institution_country_code"].isna() & authorships_df["author_id"].notna()

    candidates = authorships_df.loc[mask, candidate_cols].drop_duplicates()
    pending = candidates[~candidates[key].isin(known_keys)].copy()
    pending["institution_country_code"] = None

    os.makedirs(PENDING_DIR, exist_ok=True)
    out_path = os.path.join(PENDING_DIR, f"geocode_{kind}_{field}.csv")
    pending.to_csv(out_path, index=False)
    print(f"Wrote {len(pending):,} new (not-yet-labeled) {kind} rows to {out_path}")
    return pending


def merge_labels(kind, pending_path):
    """
    Fold labeled rows from a pending CSV (as written by export_pending, with
    institution_country_code filled in) into the global crosswalk for
    `kind`, deduplicated by key. Rows still missing institution_country_code
    are left out (they'll show up again next time export_pending runs).
    """
    spec = CROSSWALKS[kind]
    key = spec["key"]
    cols = [key] + spec["extra_cols"] + ["institution_country_code"]

    pending = pd.read_csv(pending_path)
    pending = pending[[c for c in cols if c in pending.columns]]
    labeled = pending.dropna(subset=["institution_country_code"])
    skipped = len(pending) - len(labeled)

    crosswalk = _load_crosswalk(kind)
    merged = pd.concat([crosswalk, labeled], ignore_index=True)
    merged = merged.drop_duplicates(subset=[key], keep="last")

    merged.to_csv(spec["path"], index=False)
    print(
        f"Merged {len(labeled):,} labeled rows into {spec['path']} ({len(merged):,} total rows). "
        f"{skipped:,} still-unlabeled rows were skipped."
    )


def _cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    export_p = sub.add_parser("export", help="Export not-yet-labeled institutions/affiliations/authors for a field")
    export_p.add_argument("--field", required=True, help="Field slug, e.g. meded, econ")
    export_p.add_argument("--kind", required=True, choices=list(CROSSWALKS))

    merge_p = sub.add_parser("merge", help="Merge a labeled pending file into the global crosswalk")
    merge_p.add_argument("--kind", required=True, choices=list(CROSSWALKS))
    merge_p.add_argument("--pending", required=True)

    args = parser.parse_args()

    if args.command == "export":
        metadata_path = os.path.join(REPO_ROOT, "data", "metadata", args.field, "authorships.csv")
        authorships_df = pd.read_csv(metadata_path)
        export_pending(authorships_df, args.kind, args.field)
    elif args.command == "merge":
        merge_labels(args.kind, args.pending)


if __name__ == "__main__":
    _cli()
