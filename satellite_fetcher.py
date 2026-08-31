import io
import math
import requests
from PIL import Image

class CoordinateSatelliteFetcher:
    """
    Fetches real-time overhead satellite tiles given geographic (lat, lon) coordinates
    and zoom levels without requiring mandatory paid API accounts.
    """
    @staticmethod
    def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
        """Converts latitude and longitude to Slippy Map tile numbers (X, Y)."""
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (xtile, ytile)

    @classmethod
    def fetch_tile_by_coordinates(cls, lat: float, lon: float, zoom: int = 16) -> tuple[Image.Image, dict]:
        """
        Retrieves an overhead satellite imagery tile centered at (lat, lon).
        """
        xtile, ytile = cls.deg2num(lat, lon, zoom)
        
        # High-resolution Esri World Imagery Tile Server
        tile_url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ytile}/{xtile}"
        headers = {"User-Agent": "SatQuery-AI-Geospatial-Assistant/1.0"}

        response = requests.get(tile_url, headers=headers, timeout=10)
        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            metadata = {
                "filename": f"Tile_Z{zoom}_{xtile}_{ytile}.jpg",
                "format": "Remote Web-Fetched Tile",
                "center_coords": (lat, lon),
                "zoom_level": zoom,
                "bands": 3
            }
            return image, metadata
        else:
            raise RuntimeError(f"Satellite tile fetch failed with HTTP status {response.status_code}")