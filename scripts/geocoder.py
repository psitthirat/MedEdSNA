"""
Geocoding and Affiliation Location Extraction Utility

This script provides functionality to extract geographic information from textual affiliations (e.g., author institutions) 
and convert them into structured geographic outputs. It integrates NLP for entity recognition and geocoding services 
to enrich bibliometric or collaboration network datasets with spatial attributes.

Key functionalities include:
- Splitting and extracting the last parts of affiliation strings.
- Using NLP (stanza) to identify organization names and location entities.
- Geocoding text-based locations into latitude and longitude coordinates with the Google Maps Geocoding API.
- Mapping coordinates to countries and continents using geopandas and shapely.
- Performing reverse geocoding to retrieve city or locality information.

Dependencies:
- stanza
- geopy
- geopandas
- shapely
- pandas
- requests
- tqdm

Requires a Google Maps Geocoding API key (billing-enabled Google Cloud project) set in the
`GOOGLE_MAPS_API_KEY` environment variable, e.g. `export GOOGLE_MAPS_API_KEY=your-key-here`
before importing this module.

Usage:
1. Set the `GOOGLE_MAPS_API_KEY` environment variable, then initialize the NLP pipeline and geocoder.
2. Provide input text (affiliation strings or locations).
3. Use the provided functions:
   - `extract_last_parts(text, number_of_parts)`
   - `extract_location(text)`
   - `extract_coordination(text)`
   - `extract_country(world, lat, lon)`
   - `extract_city(lat, lon)`
   - `geocode_affiliations(unique_affiliations, world, temp_file, final_file)`

Example:
    from geocoder import extract_location, extract_coordination, extract_country

    text = "Department of Medicine, University of Oxford, Oxford, UK"
    loc, org = extract_location(text)
    lat, lon = extract_coordination(loc)
    country, continent = extract_country(world, lat, lon)
    print(loc, org, lat, lon, country, continent)

Authors: P. Sitthirat et al
Version: 1.0
License: MIT License
"""

import os
import pandas as pd
from tqdm import tqdm
import stanza
from dotenv import load_dotenv
from geopy.geocoders import GoogleV3
from geopy.extra.rate_limiter import RateLimiter
import geopandas as gpd
from shapely.geometry import Point
from IPython.display import clear_output

# Initialize pre-trained NLP model
stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

# Initialize geocoder (requires a billing-enabled Google Cloud project).
# Reads GOOGLE_MAPS_API_KEY from the environment, or from a .env file at the project root
# (copy .env.example to .env and fill in your key — .env is gitignored).
load_dotenv()
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
if not GOOGLE_MAPS_API_KEY:
    raise RuntimeError(
        "GOOGLE_MAPS_API_KEY is not set. Copy .env.example to .env and fill in a key from "
        "https://console.cloud.google.com/google/maps-apis/credentials (with the Geocoding API "
        "enabled), or export GOOGLE_MAPS_API_KEY directly before importing this module."
    )
geolocator = GoogleV3(api_key=GOOGLE_MAPS_API_KEY)
# Google's default quota is far higher than Nominatim's 1 req/s; error_wait_seconds/max_retries
# give automatic backoff if a burst still trips OVER_QUERY_LIMIT.
geocode = RateLimiter(geolocator.reverse, min_delay_seconds=0.1, error_wait_seconds=5, max_retries=3)
geocode_forward = RateLimiter(geolocator.geocode, min_delay_seconds=0.1, error_wait_seconds=5, max_retries=3)

# Function to extract some parts of text
def extract_last_parts(text, number_of_parts=0):
    if not isinstance(text, str):
        return str(text)  # Handle NaN values safely
    parts = text.split(",")  # Split by commas
    return ", ".join(parts[-number_of_parts:]) if len(parts) >= number_of_parts else text

# Function to extract locations and organization from the affiliation
def extract_location(text):
    if not isinstance(text, str):  # Ensure input is a string
        text = str(text)
        
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.type in ("GPE", "LOC")]  # Extract locations
    organizations = [ent.text for ent in doc.ents if ent.type == "ORG"]  # Extract organizations
    
    if locations or organizations:
        location_text = ", ".join(locations) if locations else None
        organization_text = ", ".join(organizations) if organizations else None
    else:
        location_text = None
        organization_text = None
    
    return location_text, organization_text

# Function to search for coordinations (LAT,LON)
def extract_coordination(text):
    if not isinstance(text, str):  # Ensure input is a string
        text = str(text)
    try:
        if text.lower() in ["university", "hospital", "center"]:
            print(f"Skipping ambiguous location: {text}")
            return None, None
                
        geo = geocode_forward(text, timeout=10)
        if geo:
            lat, lon = geo.latitude, geo.longitude
        else:
            lat, lon = None, None
    except Exception as e:
        print(f"Error geocoding {text}: {e}")
        lat, lon = None, None
    
    return lat, lon

# Function to extract the country from coordinations (LAT,LON)
def extract_country(world, lat, lon):
    
    world_proj = world.to_crs(epsg=3857)
    
    # Create a Point object
    point = Point(lon, lat)  # Note: (longitude, latitude) format
    point_proj = gpd.GeoSeries([point], crs=world.crs).to_crs(epsg=3857).iloc[0]

    # Find the country that contains the point
    country_row = world[world.contains(point)]

    if not country_row.empty:
        country = country_row.iloc[0]["ADM0_A3"]
        continent = country_row.iloc[0]["CONTINENT"]
    else:
        # Calculate distances without modifying the original dataframe
        distances = world_proj.geometry.distance(point_proj)
        nearest_row = world.loc[distances.idxmin()]
        
        if not nearest_row.empty:
            country = nearest_row["ADM0_A3"]
            continent = nearest_row["CONTINENT"]           
        
        else:
            country = None
            continent = None

    return country, continent

# Function to extract the city from coordinations (LAT,LON)
def extract_city(lat, lon):
    try:
        location = geocode((lat, lon), language='en', exactly_one=True)
        if location and location.raw and "address_components" in location.raw:
            components = location.raw["address_components"]

            def component(*types):
                for comp in components:
                    if any(t in comp["types"] for t in types):
                        return comp["long_name"]
                return None

            city = component("locality", "postal_town", "administrative_area_level_3")
            country = component("country")
            return f"{city}, {country}" if city else country
        else:
            return None
    except Exception as e:
        print(f"Error in reverse geocoding: {e}")
        return None

# Function to geocode a batch of unique affiliations with checkpointing
def geocode_affiliations(unique_affiliations, world, temp_file, final_file):
    """
    Geocode a list of unique affiliation strings into coordinates, country,
    and continent, with checkpointing so long runs can resume.

    For each affiliation, extracts a location/organization pair via
    `extract_location`, then geocodes it through a fallback cascade (organization
    name, then full location, then progressively truncated tails of the location
    string) using `extract_coordination`, and finally resolves the country and
    continent via `extract_country`. Affiliations already present in `temp_file`
    or `final_file` are skipped, so an interrupted run can be resumed by calling
    this again with the same paths. Progress is saved to `temp_file` every 100
    newly processed affiliations; `temp_file` is deleted once complete.

    Parameters:
    - unique_affiliations (array-like): Affiliation strings to geocode.
    - world (geopandas.GeoDataFrame): Country boundaries used by `extract_country`.
    - temp_file (str): Path to the checkpoint CSV used to resume interrupted runs.
    - final_file (str): Path to write (and, if present, resume from) the completed
      geocoding results.

    Returns:
    - pd.DataFrame: One row per affiliation with columns 'Affiliation', 'Location',
      'Organization', 'Latitude', 'Longitude', 'Country', 'Continent'.

    Example:
        >>> unique_affiliations = authors_df['Affiliation'].dropna().unique()
        >>> geo_df = geocode_affiliations(unique_affiliations, world,
        ...     temp_file="output/geocode/aff_geocoded_temp.csv",
        ...     final_file="output/geocode/aff_geocoded.csv")
    """

    print(f"Total Unique Affiliations to Process: {len(unique_affiliations)}")

    if os.path.exists(temp_file):
        geo_df = pd.read_csv(temp_file, index_col=0)
        geo_df = geo_df.dropna(subset=['Latitude', 'Longitude']).reset_index(drop=True)
        processed_affiliations = set(geo_df["Affiliation"])
        print(f"Resuming from previous progress: {len(processed_affiliations)} affiliations already processed.")
    elif os.path.exists(final_file):
        geo_df = pd.read_csv(final_file, index_col=0)
        geo_df = geo_df.dropna(subset=['Latitude', 'Longitude']).reset_index(drop=True)
        processed_affiliations = set(geo_df["Affiliation"])
        print(f"Resuming from previous progress: {len(processed_affiliations)} affiliations already processed.")
    else:
        geo_df = pd.DataFrame()
        processed_affiliations = set()

    geo_results = []
    geo_forcoding = set(unique_affiliations) - processed_affiliations
    print(f"Affiliations remaining to geocode: {len(geo_forcoding)}")

    # Iterate through unique affiliations with progress bar
    for affil in tqdm(geo_forcoding):

        location, organization = extract_location(affil)
        
        if location is None and organization is None:
            lat, lon, country, continent = None, None, None, None
        else:
            # Try geocoding organization first, fallback to location
            lat, lon = None, None
            if organization:
                lat, lon = extract_coordination(organization)
                if lat is None and lon is None and location:
                    lat, lon = extract_coordination(location)
                    if lat is None and lon is None:
                        lat, lon = extract_coordination(extract_last_parts(location, 3))
                        if lat is None and lon is None:
                            lat, lon = extract_coordination(extract_last_parts(location, 2))
                            if lat is None and lon is None:
                                lat, lon = extract_coordination(extract_last_parts(location, 1))

                if location is None and lat and lon:
                    location = extract_city(lat, lon)

            else:
                lat, lon = extract_coordination(location)
                if lat is None and lon is None:
                    lat, lon = extract_coordination(extract_last_parts(location, 3))
                    if lat is None and lon is None:
                        lat, lon = extract_coordination(extract_last_parts(location, 2))
                        if lat is None and lon is None:
                            lat, lon = extract_coordination(extract_last_parts(location, 1))

            country, continent = None, None
            if lat and lon:
                country, continent = extract_country(world, lat, lon)

        geo_results.append({
            "Affiliation": affil,
            "Location": location,
            "Organization": organization,
            "Latitude": lat,
            "Longitude": lon,
            "Country": country,
            "Continent": continent
        })

        # Save progress every 100 records
        if len(geo_results) % 100 == 0:
            print(f"Saving progress at {len(geo_results)} new records...")
            temp_df = pd.DataFrame(geo_results)
            combined = pd.concat([geo_df, temp_df], ignore_index=True)
            combined.to_csv(temp_file, index=True)

    clear_output()

    # Create a DataFrame from the final results
    final_df = pd.concat([geo_df, pd.DataFrame(geo_results)], ignore_index=True)
    condition = final_df['Location'].isna() & final_df['Organization'].isna()
    final_df.loc[condition, ['Latitude', 'Longitude', 'Country', 'Continent']] = None
    final_df.to_csv(final_file, index=True)
    print("Geocoding complete. Final file saved.")

    # Remove the temporary file
    if os.path.exists(temp_file):
        os.remove(temp_file)
        print("Temporary file deleted.")

    return final_df