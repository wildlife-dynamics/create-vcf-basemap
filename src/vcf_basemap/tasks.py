from __future__ import annotations

import math
import os
import tempfile
from typing import Annotated, Any

import ee
import requests
from ecoscope.platform.connections import EarthEngineClient
from pydantic import Field
from wt_registry import register

# Endmember RGB colours (sampled from the Gourma reference map legend).
# MOD44B has three percent-cover bands that sum to 100 per pixel — we render
# each pixel as a linear (convex) mix of three endmember colours weighted by
# the three band values, giving a continuous fractional-cover composite.
_DEFAULT_BARE_HEX = "B89070"  # bare ground   (sandy reddish brown)
_DEFAULT_HERB_HEX = "B3B2A0"  # non-tree veg  (warm light olive)
_DEFAULT_TREE_HEX = "4F7B47"  # tree cover    (forest green)

# Water overlay defaults. Inland water now comes from JRC Global Surface Water
# (`JRC/GSW1_4/GlobalSurfaceWater`, `occurrence` band), thresholded by a
# user-configurable percentage. Ocean is everything outside any LSIB country
# polygon. (MOD44W was the previous inland source, dropped 2026-06-04 in favour
# of GSW's higher resolution and rivers; MOD10A1 snow detection also dropped
# same day — too many false positives from cloud cover.)
_DEFAULT_WATER_HEX = "4A7BA8"  # inland water — muted cartographic blue
_DEFAULT_OCEAN_HEX = "2A5A82"  # ocean — darker blue
_DEFAULT_WATER_OCCURRENCE_MIN = 50  # GSW occurrence % threshold for inland water

# Fallback for MOD44B no-data pixels the multi-year mosaic can't fill: render the
# SRTM hillshade through a neutral mid-gray. Same darkening curve as the VCF mix
# so the terrain reads continuously across no-data ↔ valid-VCF boundaries.
_NODATA_HILLSHADE_GRAY = 180  # 0–255 base value, modulated by hillshade

_HEX_PATTERN = r"^#?[0-9A-Fa-f]{6}$"


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    s = hex_str.lstrip("#")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


# ── shared GEE helper ─────────────────────────────────────────────────────────


def _build_basemap_image(
    year: int,
    hillshade_weight: float,
    scale_m: int,
    bbox: list[float],
    bare_rgb: tuple[int, int, int],
    herb_rgb: tuple[int, int, int],
    tree_rgb: tuple[int, int, int],
    water_rgb: tuple[int, int, int],
    ocean_rgb: tuple[int, int, int],
    water_occurrence_min: int,
) -> ee.Image:
    """Build the blended VCF + hillshade + water + ocean image over the given bbox.

    Does not trigger any export — safe to call from both the export and preview tasks.
    """
    # MODIS native projection (sinusoidal, SR-ORG:6974) can't reliably transform
    # complex polygon edges to EPSG:4326. Use a rectangular bbox for all intermediate
    # clipping and filtering; the precise region clip happens in the download step.
    aoi_bbox = ee.Geometry.Rectangle(bbox, proj="EPSG:4326", geodesic=False)

    # Temporal mosaic across every MOD44B year up to and including `year`, sorted
    # so the user's selected year ends up on top. `.mosaic()` walks top-to-bottom
    # and fills each pixel with the topmost valid value — so an earlier year
    # backfills wherever the latest year has a sensor/cloud gap. This dramatically
    # reduces no-data holes that were previously rendering as the neutral fill.
    mod44b = (
        ee.ImageCollection("MODIS/061/MOD44B")
        .filterDate("2000-01-01", f"{year + 1}-01-01")
        .filterBounds(aoi_bbox)
        .select(
            ["Percent_Tree_Cover", "Percent_NonTree_Vegetation", "Percent_NonVegetated"]
        )
        .sort("system:time_start")  # ascending → latest is last → on top in mosaic
        .mosaic()
    )
    # Force MODIS sinusoidal → EPSG:4326 here. Without this, downstream clip
    # operations fail to transform clip-geometry edges back to MODIS sinusoidal
    # at tile-render time: `Image.clip: Unable to transform edge ... from SR-ORG:6974`.
    cover = mod44b.reproject(crs="EPSG:4326", scale=scale_m)

    tree = cover.select("Percent_Tree_Cover")
    herb = cover.select("Percent_NonTree_Vegetation")
    bare = cover.select("Percent_NonVegetated")
    # MOD44B masks water (no valid percent value over lakes/ocean), so on its own
    # it can't tell water apart from genuine no-data (glacier, persistent cloud).
    # Mask those holes out of the land mix here; resolve them below with MOD44W.
    valid = tree.lte(100).And(herb.lte(100)).And(bare.lte(100))

    # Per-channel linear mix: each output channel is the percent-weighted
    # average of the three endmember channel values. Since the three percents
    # sum to 100, dividing by 100 produces a true convex combination.
    def _mix(channel_idx: int) -> ee.Image:
        return (
            bare.multiply(bare_rgb[channel_idx])
            .add(herb.multiply(herb_rgb[channel_idx]))
            .add(tree.multiply(tree_rgb[channel_idx]))
            .divide(100.0)
        )

    vcf_rgb = (
        ee.Image.cat([_mix(0), _mix(1), _mix(2)])
        .updateMask(valid)
        .rename(["R", "G", "B"])
    )

    srtm = ee.Image("USGS/SRTMGL1_003")
    # Azimuth 315° (NW) and elevation 35° are standard cartographic conventions.
    hillshade = ee.Terrain.hillshade(srtm, azimuth=315, elevation=35)

    vcf_01 = vcf_rgb.divide(255.0)
    # Mid-tone multiply: factor = hs_norm * weight + (1 - weight)
    # Single-band hs_factor broadcasts across all 3 RGB channels in GEE arithmetic.
    hs_factor = (
        hillshade.divide(255.0).multiply(hillshade_weight).add(1.0 - hillshade_weight)
    )
    vcf_hillshade = (
        vcf_01.multiply(hs_factor).multiply(255).toUint8().rename(["R", "G", "B"])
    )

    # LSIB country polygons within the bbox, used to split MOD44W water into
    # inland vs ocean. Pixels outside all country polygons are treated as ocean
    # (robust to MOD44W's patchy open-sea coverage); trans-border lakes (Victoria
    # across KE/UG/TZ) stay inland because they sit inside *some* country polygon.
    landmass = (
        ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filterBounds(aoi_bbox).geometry()
    )
    # sameFootprint=False is essential: without it, unmask only fills within the
    # clipped landmass footprint, so open-ocean pixels stay masked and the ocean
    # layer never gets painted there.
    inside_land = ee.Image.constant(1).clip(landmass).unmask(0, False)

    # Inland water from JRC Global Surface Water (1984–2021 Landsat-derived).
    # `occurrence` is 0–100% — the fraction of valid observations classified as
    # water. The user's `water_occurrence_min` chooses how permanent: 50 catches
    # rivers/permanent lakes, lower values include seasonal water. Confined to
    # land — open sea is the ocean layer's job.
    #
    # We explicitly aggregate from GSW's native 30 m to scale_m via reduceResolution
    # (mean) before thresholding. Without that, `gte(threshold).reproject(scale_m)`
    # produces a fractional aggregate (mean of 30 m booleans) and EE's `.And()`
    # treats *any* nonzero float as truthy — so a cell with even one
    # high-occurrence subpixel would get flagged as water.
    gsw_occurrence_agg = (
        ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        .select("occurrence")
        .unmask(0)  # GSW is masked over never-wet land; treat as 0% occurrence.
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
        .reproject(crs="EPSG:4326", scale=scale_m)
    )
    inland_water_mask = gsw_occurrence_agg.gte(water_occurrence_min).And(inside_land)

    # Ocean: purely geographic. Anywhere outside every country polygon = sea.
    ocean_mask = inside_land.Not()

    # Each overlay painted flat (no hillshade) and masked to its pixel set.
    def _flat_layer(rgb: tuple[int, int, int], mask: ee.Image) -> ee.Image:
        return (
            ee.Image.constant(list(rgb))
            .toUint8()
            .rename(["R", "G", "B"])
            .updateMask(mask)
        )

    ocean_layer = _flat_layer(ocean_rgb, ocean_mask)
    inland_layer = _flat_layer(water_rgb, inland_water_mask)

    # Fallback for residual MOD44B no-data (after the multi-year mosaic): show
    # the SRTM hillshade through a neutral mid-gray instead of a flat patch.
    # Uses the same `hs_factor` darkening curve as vcf_hillshade, so the terrain
    # reads continuously across no-data ↔ valid-VCF boundaries.
    hillshade_neutral = (
        ee.Image.constant(_NODATA_HILLSHADE_GRAY).multiply(hs_factor).toUint8()
    )
    hillshade_fill = ee.Image.cat(
        [hillshade_neutral, hillshade_neutral, hillshade_neutral]
    ).rename(["R", "G", "B"])
    basemap = (
        vcf_hillshade.unmask(hillshade_fill)
        .blend(ocean_layer)
        .blend(inland_layer)
        .toUint8()
        .clip(aoi_bbox)
    )
    return basemap


# ── tiled getDownloadURL helpers ──────────────────────────────────────────────

# GEE's getDownloadURL caps each request at ~50 MB of uncompressed pixels. For
# Kenya, that limit is hit anywhere below ~400 m/px. We tile the request, pull
# each tile synchronously, and mosaic locally with rasterio.
# 6 MB per tile — tighter than the 50 MB request-size cap because the limiting
# factor is now EE's *compute memory*, not bytes: the multi-year MOD44B mosaic +
# GSW reduceResolution + landmass polygon rasterization together exceeded memory
# at the previous 30 MB budget when the whole Kenya bbox sat in a single tile.
# 6 MB forces ≥2 tiles even at the coarse 500 m default and gives 9 tiles at 250 m.
_GETDOWNLOADURL_TARGET_BYTES = 6_000_000


def _choose_tile_grid(
    bbox: list[float], scale_m: int, bytes_per_pixel: int = 3
) -> tuple[int, int]:
    """Pick (nx, ny) so each rectangular tile stays under the per-request cap.

    Crude metre estimate: ~111 km/deg latitude. Kenya straddles the equator so
    this is conservative for longitude too. Always returns at least 1×1.
    """
    minx, miny, maxx, maxy = bbox
    width_m = (maxx - minx) * 111_000.0
    height_m = (maxy - miny) * 111_000.0
    est_bytes = (width_m * height_m / (scale_m**2)) * bytes_per_pixel
    n_tiles = max(1, math.ceil(est_bytes / _GETDOWNLOADURL_TARGET_BYTES))
    aspect = width_m / height_m
    nx = max(1, math.ceil(math.sqrt(n_tiles * aspect)))
    ny = max(1, math.ceil(n_tiles / nx))
    return nx, ny


def _bbox_grid(bbox: list[float], nx: int, ny: int):
    minx, miny, maxx, maxy = bbox
    dx = (maxx - minx) / nx
    dy = (maxy - miny) / ny
    for i in range(nx):
        for j in range(ny):
            yield (
                minx + i * dx,
                miny + j * dy,
                minx + (i + 1) * dx,
                miny + (j + 1) * dy,
            )


def _download_basemap_tiled(
    image: ee.Image,
    bbox: list[float],
    scale_m: int,
    crs: str,
    clip_geometry=None,
) -> bytes:
    """Download an image as a grid of tiles via getDownloadURL, mosaic, and return COG bytes.

    Produces a Cloud Optimized GeoTIFF: internal 512×512 tiles, deflate compression,
    and a pyramid of overview levels. If clip_geometry is provided (a list of shapely
    geometries), the merged raster is clipped to that polygon boundary before the COG
    step, so the output matches the exact AOI shape rather than its bounding rectangle.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.merge import merge
    from rasterio.shutil import copy as rio_copy

    nx, ny = _choose_tile_grid(bbox, scale_m)
    tmp_paths: list[str] = []
    try:
        for tile_bbox in _bbox_grid(bbox, nx, ny):
            tile_region = ee.Geometry.Rectangle(
                list(tile_bbox), proj=crs, geodesic=False
            )
            url = image.getDownloadURL(
                {
                    "region": tile_region,
                    "scale": scale_m,
                    "format": "GEO_TIFF",
                    "crs": crs,
                }
            )
            r = requests.get(url, timeout=600)
            r.raise_for_status()
            fd, path = tempfile.mkstemp(suffix=".tif")
            os.close(fd)
            with open(path, "wb") as f:
                f.write(r.content)
            tmp_paths.append(path)

        datasets = [rasterio.open(p) for p in tmp_paths]
        try:
            mosaic, transform = merge(datasets)
            profile = datasets[0].profile.copy()
        finally:
            for d in datasets:
                d.close()

        profile.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "driver": "GTiff",
                "compress": "deflate",
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
            }
        )

        # COG requires overviews built before the copy with COPY_SRC_OVERVIEWS=YES.
        # Step 1: write merged data with internal tiling.
        # Step 2: build overviews in a second pass (r+ avoids rewriting all pixels).
        # Step 3: rio_copy with copy_src_overviews=True writes the final COG layout.
        merged_fd, merged_path = tempfile.mkstemp(suffix="_merged.tif")
        os.close(merged_fd)
        cog_fd, cog_path = tempfile.mkstemp(suffix=".tif")
        os.close(cog_fd)
        try:
            with rasterio.open(merged_path, "w", **profile) as dst:
                dst.write(mosaic)

            if clip_geometry is not None:
                from rasterio.mask import mask as rio_mask

                with rasterio.open(merged_path) as src:
                    clipped, clipped_transform = rio_mask(
                        src, clip_geometry, crop=True, filled=True, nodata=0
                    )
                    clipped_profile = src.profile.copy()
                    clipped_profile.update(
                        {
                            "height": clipped.shape[1],
                            "width": clipped.shape[2],
                            "transform": clipped_transform,
                        }
                    )
                with rasterio.open(merged_path, "w", **clipped_profile) as dst:
                    dst.write(clipped)

            with rasterio.open(merged_path, "r+") as dst:
                dst.build_overviews([2, 4, 8, 16, 32], Resampling.average)
                dst.update_tags(ns="rio_overview", resampling="average")

            rio_copy(
                merged_path,
                cog_path,
                copy_src_overviews=True,
                tiled=True,
                blockxsize=512,
                blockysize=512,
                compress="deflate",
                driver="GTiff",
            )
            with open(cog_path, "rb") as f:
                return f.read()
        finally:
            for p in (merged_path, cog_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass


# ── tasks ─────────────────────────────────────────────────────────────────────


@register()
def set_vcf_palette(
    bare_hex: Annotated[
        str,
        Field(
            default=_DEFAULT_BARE_HEX,
            pattern=_HEX_PATTERN,
            title="Bare Ground Colour (hex)",
            description="RGB hex for the 100% bare-ground endmember (e.g. B89070 = sandy brown).",
        ),
    ] = _DEFAULT_BARE_HEX,
    herb_hex: Annotated[
        str,
        Field(
            default=_DEFAULT_HERB_HEX,
            pattern=_HEX_PATTERN,
            title="Herbaceous Colour (hex)",
            description="RGB hex for the 100% non-tree-vegetation endmember (e.g. B3B2A0 = warm light olive).",
        ),
    ] = _DEFAULT_HERB_HEX,
    tree_hex: Annotated[
        str,
        Field(
            default=_DEFAULT_TREE_HEX,
            pattern=_HEX_PATTERN,
            title="Tree Cover Colour (hex)",
            description="RGB hex for the 100% tree-cover endmember (e.g. 4F7B47 = forest green).",
        ),
    ] = _DEFAULT_TREE_HEX,
    water_hex: Annotated[
        str,
        Field(
            default=_DEFAULT_WATER_HEX,
            pattern=_HEX_PATTERN,
            title="Inland Water Colour (hex)",
            description="RGB hex for inland water (lakes, rivers) from MOD44W (e.g. 4A7BA8 = muted blue).",
        ),
    ] = _DEFAULT_WATER_HEX,
    ocean_hex: Annotated[
        str,
        Field(
            default=_DEFAULT_OCEAN_HEX,
            pattern=_HEX_PATTERN,
            title="Ocean Colour (hex)",
            description="RGB hex for ocean pixels (outside any country polygon).",
        ),
    ] = _DEFAULT_OCEAN_HEX,
    water_occurrence_min: Annotated[
        int,
        Field(
            default=_DEFAULT_WATER_OCCURRENCE_MIN,
            ge=0,
            le=100,
            title="Water Occurrence Min (%)",
            description=(
                "JRC GSW `occurrence` threshold for inland water — the minimum % "
                "of valid Landsat observations classified as water for a pixel "
                "to count as inland water. 50 catches permanent rivers/lakes; "
                "lower includes seasonal water bodies."
            ),
        ),
    ] = _DEFAULT_WATER_OCCURRENCE_MIN,
) -> dict[str, Any]:
    """Bundle the basemap style — VCF endmembers plus inland water and ocean colours.

    Returns a dict consumed by every downstream basemap task so the form-driven
    style is applied consistently across export, preview, and legend.
    """
    return {
        "bare": bare_hex.lstrip("#"),
        "herb": herb_hex.lstrip("#"),
        "tree": tree_hex.lstrip("#"),
        "water": water_hex.lstrip("#"),
        "ocean": ocean_hex.lstrip("#"),
        "water_occurrence_min": water_occurrence_min,
    }


@register()
def create_vcf_basemap(
    # Type-annotating as EarthEngineClient triggers the Pydantic BeforeValidator
    # on EarthEngineConnection.client_from_named_connection, which constructs
    # EarthEngineIO(...) and thereby calls ee.Initialize(). Without this type,
    # ee.* calls below fail with "Earth Engine client library not initialized".
    client: Annotated[EarthEngineClient, Field(exclude=True)],
    palette: Annotated[dict[str, Any], Field(exclude=True)],
    roi: Annotated[Any, Field(exclude=True)],
    root_path: Annotated[
        str,
        Field(
            title="Results directory",
            description="Local directory where the GeoTIFF is written; wired from ECOSCOPE_WORKFLOWS_RESULTS.",
        ),
    ],
    year: Annotated[
        int,
        Field(
            default=2020,
            ge=2000,
            le=2024,
            title="VCF Year",
            description=(
                "Latest MODIS MOD44B year (Collection 6.1) to include. The basemap "
                "is a temporal mosaic from 2000 through this year, with this year "
                "on top — earlier years backfill any sensor/cloud no-data gaps."
            ),
        ),
    ] = 2020,
    scale_m: Annotated[
        int,
        Field(
            default=250,
            ge=250,
            le=5000,
            title="Export Resolution (m)",
            description="Pixel size in metres. MOD44B native resolution is ~250 m.",
        ),
    ] = 250,
    hillshade_weight: Annotated[
        float,
        Field(
            default=0.60,
            ge=0.0,
            le=1.0,
            title="Hillshade Blend Weight",
            description=(
                "Controls terrain texture strength. "
                "0 = flat colour only, 1 = full multiply darkening. "
                "0.60 matches the Gourma reference map style."
            ),
        ),
    ] = 0.60,
    output_name: Annotated[
        str,
        Field(
            default="vcf_hillshade_basemap",
            title="Output Filename Prefix",
            description="The product year is appended automatically, e.g. vcf_hillshade_basemap_2020.tif",
        ),
    ] = "vcf_hillshade_basemap",
) -> str:
    """Download a MODIS VCF + SRTM hillshade RGB basemap as a local COG GeoTIFF.

    Derives the AOI bbox from the input ROI GeoDataFrame, tiles the GEE request,
    mosaics locally, clips to the ROI polygon boundary, then writes a COG.
    Returns the on-disk path.
    """
    from ecoscope.platform.serde import _persist_bytes

    roi_4326 = roi.to_crs("EPSG:4326")
    bbox = list(roi_4326.total_bounds)
    clip_geoms = list(roi_4326.geometry)

    basemap = _build_basemap_image(
        year,
        hillshade_weight,
        scale_m,
        bbox,
        _hex_to_rgb(palette["bare"]),
        _hex_to_rgb(palette["herb"]),
        _hex_to_rgb(palette["tree"]),
        _hex_to_rgb(palette["water"]),
        _hex_to_rgb(palette["ocean"]),
        int(palette["water_occurrence_min"]),
    )

    tif_bytes = _download_basemap_tiled(basemap, bbox, scale_m, "EPSG:4326", clip_geoms)

    fname = f"{output_name}_{year}.tif"
    return _persist_bytes(tif_bytes, root_path, fname)


@register()
def draw_vcf_preview(
    client: Annotated[EarthEngineClient, Field(exclude=True)],
    palette: Annotated[dict[str, Any], Field(exclude=True)],
    roi: Annotated[Any, Field(exclude=True)],
    year: Annotated[
        int,
        Field(
            default=2020,
            ge=2000,
            le=2024,
            title="VCF Year",
            description="Must match the year set on the export task.",
        ),
    ] = 2020,
    hillshade_weight: Annotated[
        float,
        Field(
            default=0.60,
            ge=0.0,
            le=1.0,
            title="Hillshade Blend Weight",
            description="Must match the weight set on the export task.",
        ),
    ] = 0.60,
) -> Annotated[str, Field()]:
    """Live-tile map preview of the VCF basemap via GEE tile streaming.

    Streams tiles directly from Earth Engine — no Drive export needed.
    Returns an HTML string compatible with create_map_widget_single_view.
    """
    from ecoscope.mapping import EcoMap

    roi_4326 = roi.to_crs("EPSG:4326")
    bbox = list(roi_4326.total_bounds)

    basemap = _build_basemap_image(
        year,
        hillshade_weight,
        250,
        bbox,
        _hex_to_rgb(palette["bare"]),
        _hex_to_rgb(palette["herb"]),
        _hex_to_rgb(palette["tree"]),
        _hex_to_rgb(palette["water"]),
        _hex_to_rgb(palette["ocean"]),
        int(palette["water_occurrence_min"]),
    )

    m = EcoMap(static=False, default_widgets=False)
    m.add_scale_bar()
    m.add_north_arrow()
    m.add_save_image()

    # EcoMap.ee_layer() calls getMapId() and returns a lonboard BitmapTileLayer
    layer = EcoMap.ee_layer(
        basemap,
        {"bands": ["R", "G", "B"], "min": 0, "max": 255},
    )
    m.add_layer(layer)

    minx, miny, maxx, maxy = bbox
    center_lon = (minx + maxx) / 2
    center_lat = (miny + maxy) / 2
    max_extent = max(maxx - minx, maxy - miny)
    zoom = max(3, min(10, round(math.log2(360 / max_extent))))
    m.set_view_state(
        longitude=center_lon, latitude=center_lat, zoom=zoom, pitch=0, bearing=0
    )

    return m.to_html()


def _build_legend_png(
    bare_rgb: tuple[int, int, int],
    herb_rgb: tuple[int, int, int],
    tree_rgb: tuple[int, int, int],
    water_rgb: tuple[int, int, int],
    ocean_rgb: tuple[int, int, int],
) -> bytes:
    """Render the endmember ternary triangle plus inland-water / ocean swatches as a PNG.

    Style notes: typography hierarchy (bold title / italic subtitle / regular labels),
    softened borders (`#2a2a2a` at 0.6–0.8 px instead of pure black at 1 px), generous
    whitespace, and 200 dpi output. Falls back to DejaVu Sans if the preferred fonts
    aren't installed in the rendering env.
    """
    import io

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    INK = "#2a2a2a"  # primary text / outlines (darker than #444, softer than #000)
    SUB = "#6f6f6f"  # secondary / caption text
    LINE = 0.7  # default line width for outlines

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            # Picks the first available on the rendering env. Falls back to DejaVu.
            "font.sans-serif": [
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "Liberation Sans",
                "DejaVu Sans",
            ],
        }
    )

    # === ternary gradient (barycentric mix of endmember RGBs) ===
    size = 600
    margin = 80
    h = (size - 2 * margin) * np.sqrt(3) / 2
    v_tree = np.array([size / 2, margin])
    v_bare = np.array([margin, margin + h])
    v_herb = np.array([size - margin, margin + h])

    yy, xx = np.mgrid[:size, :size]
    pts = np.stack([xx, yy], axis=-1).astype(np.float32)

    # Barycentric weights for each pixel (u for tree, v for bare, w for herb).
    v0 = v_bare - v_tree
    v1 = v_herb - v_tree
    v2 = pts - v_tree
    denom = v0[0] * v1[1] - v0[1] * v1[0]
    v = (v2[..., 0] * v1[1] - v2[..., 1] * v1[0]) / denom
    w = (v0[0] * v2[..., 1] - v0[1] * v2[..., 0]) / denom
    u = 1.0 - v - w
    inside = (u >= 0) & (v >= 0) & (w >= 0)

    bare_arr = np.array(bare_rgb)
    herb_arr = np.array(herb_rgb)
    tree_arr = np.array(tree_rgb)
    rgb = u[..., None] * tree_arr + v[..., None] * bare_arr + w[..., None] * herb_arr
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    alpha = inside.astype(np.uint8) * 255
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)

    # === figure ===
    fig = plt.figure(figsize=(5, 5.6), dpi=200, facecolor="white")
    ax = fig.add_axes([0.08, 0.16, 0.84, 0.66])  # leave room for title + swatches
    ax.imshow(rgba, extent=(0, size, size, 0), interpolation="bilinear")
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    # subtle outline around the triangle to define its edge
    ax.add_patch(
        plt.Polygon(
            [v_tree, v_bare, v_herb],
            fill=False,
            edgecolor=INK,
            linewidth=LINE,
            joinstyle="round",
        )
    )

    # vertex markers: small white dots with a thin ring — much less visually heavy
    # than the old filled-coloured 80pt scatter points.
    for vertex, label, ha, va, dx, dy in [
        (v_tree, "Tree cover (100%)", "center", "bottom", 0, -10),
        (v_bare, "Bare ground (100%)", "right", "top", -10, 14),
        (v_herb, "Herbaceous (100%)", "left", "top", 10, 14),
    ]:
        ax.plot(
            vertex[0],
            vertex[1],
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=INK,
            markeredgewidth=LINE,
            zorder=3,
        )
        ax.text(
            vertex[0] + dx, vertex[1] + dy, label, ha=ha, va=va, fontsize=10, color=INK
        )

    # === title + subtitle (figure coords) ===
    fig.text(
        0.5,
        0.94,
        "VCF Endmember Mix",
        ha="center",
        va="center",
        fontsize=14,
        color=INK,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.895,
        "MOD44B fractional cover",
        ha="center",
        va="center",
        fontsize=9.5,
        color=SUB,
        style="italic",
    )

    # === inland-water + ocean swatches (figure coords) ===
    def _hex(c: tuple[int, int, int]) -> str:
        return "#{:02x}{:02x}{:02x}".format(*c)

    swatch_y = 0.05
    swatch_w, swatch_h = 0.038, 0.026
    col_x = (0.13, 0.55)  # two columns
    for x, (rgb_tuple, label) in zip(
        col_x,
        [
            (water_rgb, "Inland water (JRC GSW)"),
            (ocean_rgb, "Ocean"),
        ],
    ):
        fig.add_artist(
            plt.Rectangle(
                (x, swatch_y),
                swatch_w,
                swatch_h,
                facecolor=_hex(rgb_tuple),
                edgecolor=INK,
                linewidth=LINE,
                transform=fig.transFigure,
            )
        )
        fig.text(
            x + swatch_w + 0.012,
            swatch_y + swatch_h / 2,
            label,
            ha="left",
            va="center",
            fontsize=10,
            color=INK,
            transform=fig.transFigure,
        )

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.25,
        dpi=200,
    )
    plt.close(fig)
    return buf.getvalue()


@register()
def collect_dashboard_widgets(
    map_widget: Annotated[Any, Field(exclude=True)],
    legend_widget: Annotated[Any, Field(exclude=True)],
) -> list:
    """Collect dashboard widgets, filtering out SkipSentinel values.

    When the ROI is empty the map widget is a SkipSentinel. In that case the
    legend is also excluded — the PRD specifies the legend should appear
    alongside the map, not independently — and gather_dashboard receives an
    empty list rather than crashing on an unserializable sentinel value.

    When data is present both widgets are included; if either is unexpectedly
    a SkipSentinel it is silently dropped so the dashboard still renders.
    """
    from wt_task.skip import SkipSentinel

    if isinstance(map_widget, SkipSentinel):
        return []
    widgets = [map_widget]
    if not isinstance(legend_widget, SkipSentinel):
        widgets.append(legend_widget)
    return widgets


@register()
def draw_vcf_legend(
    palette: Annotated[dict[str, Any], Field(exclude=True)],
    root_path: Annotated[
        str,
        Field(
            title="Results directory",
            description="Where to write the legend PNG; wired from ECOSCOPE_WORKFLOWS_RESULTS.",
        ),
    ],
) -> Annotated[str, Field()]:
    """Render the endmember ternary triangle as a PNG, persist it, and return HTML wrapping it.

    The returned HTML is base64-self-contained so it renders independently of how Desktop
    serves the persisted PNG. The PNG itself is also written to `root_path` as the export.
    """
    import base64

    from ecoscope.platform.serde import _persist_bytes

    png_bytes = _build_legend_png(
        _hex_to_rgb(palette["bare"]),
        _hex_to_rgb(palette["herb"]),
        _hex_to_rgb(palette["tree"]),
        _hex_to_rgb(palette["water"]),
        _hex_to_rgb(palette["ocean"]),
    )
    _persist_bytes(png_bytes, root_path, "vcf_legend.png")

    b64 = base64.b64encode(png_bytes).decode("ascii")
    return (
        '<!doctype html><html><body style="margin:0;padding:0;'
        'display:flex;align-items:center;justify-content:center;height:100vh;">'
        f'<img src="data:image/png;base64,{b64}" '
        'alt="VCF endmember triangle" '
        'style="max-width:100%;max-height:100%;object-fit:contain;">'
        "</body></html>"
    )
