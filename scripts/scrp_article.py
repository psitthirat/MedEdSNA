from __future__ import annotations

import os
from dotenv import load_dotenv
import time
from typing import Any

import pandas as pd
import requests

load_dotenv()
API_KEY = os.getenv("OPENALEX_API_KEY")


def get_json(
    url: str,
    params: dict[str, Any],
    max_retries: int = 5,
) -> dict[str, Any]:
    """Request JSON from OpenAlex with basic retry handling."""

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=60)

            if response.status_code == 429:
                wait_seconds = min(2 ** attempt, 30)
                print(f"Rate limited. Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            return response.json()

        except requests.RequestException:
            if attempt == max_retries - 1:
                raise

            wait_seconds = min(2 ** attempt, 30)
            time.sleep(wait_seconds)

    raise RuntimeError("OpenAlex request failed.")

def retrieve_journal_works(
    source_id: str,
    start_date: str,
    end_date: str,
) -> list[dict[str, Any]]:
    """Retrieve all works from a source within a publication-date range."""

    short_source_id = source_id.rsplit("/", 1)[-1]
    cursor = "*"
    all_works: list[dict[str, Any]] = []

    while cursor:
        params = {
            "api_key": API_KEY,
            "filter": (
                f"primary_location.source.id:{short_source_id},"
                f"from_publication_date:{start_date},"
                f"to_publication_date:{end_date}"
            ),
            "per_page": 100,
            "cursor": cursor,
        }

        data = get_json(f"https://api.openalex.org/works", params=params)
        results = data.get("results", [])

        if not results:
            break

        all_works.extend(results)
        cursor = data.get("meta", {}).get("next_cursor")

        print(
            f"Retrieved {len(all_works):,} of "
            f"{data['meta']['count']:,} records"
        )

    return all_works


def flatten_works(
    works: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create publication-level and authorship-affiliation-level tables."""

    work_rows: list[dict[str, Any]] = []
    authorship_rows: list[dict[str, Any]] = []

    for work in works:
        primary_location = work.get("primary_location") or {}
        source = primary_location.get("source") or {}
        open_access = work.get("open_access") or {}
        biblio = work.get("biblio") or {}

        work_rows.append(
            {
                "work_id": work.get("id"),
                "doi": work.get("doi"),
                "title": work.get("display_name"),
                "publication_year": work.get("publication_year"),
                "publication_date": work.get("publication_date"),
                "work_type": work.get("type"),
                "cited_by_count": work.get("cited_by_count"),
                "is_retracted": work.get("is_retracted"),
                "is_paratext": work.get("is_paratext"),
                "language": work.get("language"),
                "journal_id": source.get("id"),
                "journal_name": source.get("display_name"),
                "issn_l": source.get("issn_l"),
                "volume": biblio.get("volume"),
                "issue": biblio.get("issue"),
                "first_page": biblio.get("first_page"),
                "last_page": biblio.get("last_page"),
                "is_open_access": open_access.get("is_oa"),
                "oa_status": open_access.get("oa_status"),
                "oa_url": open_access.get("oa_url"),
                "referenced_works_count": work.get(
                    "referenced_works_count"
                ),
            }
        )

        for authorship in work.get("authorships", []):
            author = authorship.get("author") or {}
            institutions = authorship.get("institutions") or []
            raw_affiliations = authorship.get(
                "raw_affiliation_strings", []
            )

            base_author_row = {
                "work_id": work.get("id"),
                "doi": work.get("doi"),
                "title": work.get("display_name"),
                "publication_year": work.get("publication_year"),
                "author_id": author.get("id"),
                "author_name": author.get("display_name"),
                "orcid": author.get("orcid"),
                "author_position": authorship.get("author_position"),
                "is_corresponding": authorship.get(
                    "is_corresponding"
                ),
                "raw_affiliation": " | ".join(raw_affiliations),
            }

            if institutions:
                for institution in institutions:
                    authorship_rows.append(
                        {
                            **base_author_row,
                            "institution_id": institution.get("id"),
                            "institution_name": institution.get(
                                "display_name"
                            ),
                            "institution_country_code": institution.get(
                                "country_code"
                            ),
                            "institution_type": institution.get("type"),
                            "institution_ror": institution.get("ror"),
                        }
                    )
            else:
                authorship_rows.append(
                    {
                        **base_author_row,
                        "institution_id": None,
                        "institution_name": None,
                        "institution_country_code": None,
                        "institution_type": None,
                        "institution_ror": None,
                    }
                )

    return pd.DataFrame(work_rows), pd.DataFrame(authorship_rows)
