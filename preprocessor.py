import io
import json
import numpy as np
from PIL import Image, ImageDraw

def load_satellite_image(file_obj, composite_mode="RGB"):
    """
    Pure-Python satellite image loader that works without C++ DLL dependencies.
    Generates True Color RGB, False Color NIR, or Colormapped NDVI composites.
    """
    filename = file_obj.filename.lower()
    file_bytes = file_obj.read()
    file_obj.seek(0)

    metadata = {
        "filename": file_obj.filename,
        "format": "GeoTIFF" if filename.endswith((".tif", ".tiff")) else "Standard Image",
        "bands": 3,
        "mode": composite_mode
    }

    try:
        img = Image.open(io.BytesIO(file_bytes))
        metadata["bands"] = len(img.getbands())
        metadata["size"] = img.size

        arr = np.array(img, dtype=np.float32)

        if len(arr.shape) == 2:
            valid = arr[arr > 0]
            p2, p98 = np.percentile(valid, (2, 98)) if valid.size > 0 else (0, 1)
            clipped = np.clip(arr, p2, p98)
            norm = ((clipped - p2) / (p98 - p2 + 1e-8) * 255.0).astype(np.uint8)
            pil_image = Image.fromarray(norm).convert("RGB")

        elif len(arr.shape) == 3:
            num_bands = arr.shape[2]

            if composite_mode == "NDVI":
                nir = arr[:, :, -1]
                red = arr[:, :, 0]
                ndvi = (nir - red) / (nir + red + 1e-8)
                ndvi_clipped = np.clip(ndvi, -0.2, 0.8)
                norm_ndvi = ((ndvi_clipped + 0.2) / 1.0 * 255.0).astype(np.uint8)
                
                h, w = norm_ndvi.shape
                ndvi_rgb = np.zeros((h, w, 3), dtype=np.uint8)
                ndvi_rgb[:, :, 0] = 255 - norm_ndvi
                ndvi_rgb[:, :, 1] = norm_ndvi
                ndvi_rgb[:, :, 2] = (norm_ndvi * 0.2).astype(np.uint8)
                return Image.fromarray(ndvi_rgb), metadata

            elif composite_mode == "NIR" and num_bands >= 3:
                channels = [arr[:, :, -1], arr[:, :, 0], arr[:, :, 1]]
            else:
                channels = [arr[:, :, i] for i in range(min(3, num_bands))]

            scaled = []
            for b in channels:
                valid = b[b > 0]
                p2, p98 = np.percentile(valid, (2, 98)) if valid.size > 0 else (0, 1)
                clipped = np.clip(b, p2, p98)
                norm = ((clipped - p2) / (p98 - p2 + 1e-8) * 255.0).astype(np.uint8)
                scaled.append(norm)

            while len(scaled) < 3:
                scaled.append(scaled[0])

            rgb_array = np.stack(scaled[:3], axis=-1)
            pil_image = Image.fromarray(rgb_array)
        else:
            pil_image = img.convert("RGB")

    except Exception:
        pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    return pil_image, metadata


def draw_bounding_boxes(image: Image.Image, boxes_normalized: list) -> Image.Image:
    """Draws normalized bounding boxes [ymin, xmin, ymax, xmax] onto the image."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size

    for item in boxes_normalized:
        try:
            box = item.get("box_2d", [])
            label = item.get("label", "Target")
            if len(box) == 4:
                ymin, xmin, ymax, xmax = box
                left = int((xmin / 1000) * w)
                top = int((ymin / 1000) * h)
                right = int((xmax / 1000) * w)
                bottom = int((ymax / 1000) * h)

                draw.rectangle([left, top, right, bottom], outline="#ef4444", width=3)
                draw.rectangle([left, max(0, top - 18), left + len(label) * 8 + 10, top], fill="#ef4444")
                draw.text((left + 3, max(0, top - 16)), label, fill="#ffffff")
        except Exception:
            continue

    return annotated


def generate_change_diff_map(img1: Image.Image, img2: Image.Image) -> Image.Image:
    """Generates spatial difference overlay for bi-temporal image pairs."""
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)

    arr1 = np.array(img1.convert("L"), dtype=np.int16)
    arr2 = np.array(img2.convert("L"), dtype=np.int16)

    diff = np.abs(arr1 - arr2)
    change_mask = diff > 45

    overlay = np.array(img2.copy())
    overlay[change_mask] = [239, 68, 68]

    return Image.fromarray(overlay)


def boxes_to_geojson(boxes_normalized: list, center_lat=20.5937, center_lon=78.9629) -> dict:
    """Converts pixel bounding boxes to GIS-standard GeoJSON FeatureCollection."""
    features = []
    delta = 0.015  # Coordinate bounding radius for normalized projection

    for idx, item in enumerate(boxes_normalized):
        box = item.get("box_2d", [0, 0, 1000, 1000])
        label = item.get("label", f"Object_{idx+1}")
        
        ymin, xmin, ymax, xmax = box
        min_lon = center_lon + ((xmin - 500) / 1000.0) * delta
        max_lon = center_lon + ((xmax - 500) / 1000.0) * delta
        max_lat = center_lat - ((ymin - 500) / 1000.0) * delta
        min_lat = center_lat - ((ymax - 500) / 1000.0) * delta

        polygon_coords = [[
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat]
        ]]

        features.append({
            "type": "Feature",
            "properties": {
                "id": idx + 1,
                "label": label,
                "model": "SatQuery-Agentic-Vision"
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": polygon_coords
            }
        })

    return {
        "type": "FeatureCollection",
        "features": features
    }