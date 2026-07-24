from pyproj import Proj

UTM_ZONE = Proj(proj="utm", zone=29, ellps="WGS84")

def latlon_to_meters(lat:float , lon:float) -> tuple:
    """
    Convert latitude and longitude to UTM coordinates in meters.

    Args:
        lat (float): Latitude in decimal degrees.
        lon (float): Longitude in decimal degrees.  
    """
    x, y = UTM_ZONE(lon, lat)
    return x, y
    