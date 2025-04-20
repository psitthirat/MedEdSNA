import stanza
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import geopandas as gpd
from shapely.geometry import Point

# Initialize pre-trained NLP model
stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

# Initialize geocoder
geolocator = Nominatim(user_agent="geo_coauthorship")
geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1)

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
                
        geo = geolocator.geocode(text, timeout=10, )
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
        if location and location.raw and "address" in location.raw:
            address = location.raw["address"]
            city = address.get("city") or address.get("town") or address.get("village") or address.get("hamlet")
            country = address.get("country")
            return f"{city}, {country}" if city else country
        else:
            return None
    except Exception as e:
        print(f"Error in reverse geocoding: {e}")
        return None