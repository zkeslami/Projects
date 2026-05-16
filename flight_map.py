"""Animated world map showing sequential flight legs with great-circle routes."""
from __future__ import annotations

import os
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import offset_copy
from PIL import Image
from pyproj import Geod

GEOD = Geod(ellps="WGS84")

CITIES = {
    "Austin":        (-97.7431, 30.2672),
    "London":        ( -0.1276, 51.5074),
    "Bangalore":     ( 77.5946, 12.9716),
    "Singapore":     (103.8198,  1.3521),
    "San Francisco": (-122.4194, 37.7749),
}

# Label-offset hints (text-anchor, dx_pts, dy_pts)
CITY_LABEL_OFFSET = {
    "Austin":        ("right",  -8, -10),
    "London":        ("left",    8,  10),
    "Bangalore":     ("left",   10,   2),
    "Singapore":     ("left",   10,  -2),
    "San Francisco": ("right",  -8,  10),
}

LEGS = [
    {"date": "May 11", "from": "Austin",        "to": "London"},
    {"date": "May 12", "from": "London",        "to": "Bangalore"},
    {"date": "May 15", "from": "Bangalore",     "to": "Singapore"},
    {"date": "May 19", "from": "Singapore",     "to": "San Francisco"},
    {"date": "May 19", "from": "San Francisco", "to": "Austin"},
]

# Hint where each leg's label sits — (anchor_lon, anchor_lat) in degrees.
# Placed manually so labels don't overlap each other or city dots.
LEG_LABEL_ANCHOR = {
    0: ( -45,  62),   # May 11: Austin -> London — over N Atlantic / Greenland
    1: (  20,  30),   # May 12: London -> Bangalore — Mediterranean / N Africa
    2: (  95, -18),   # May 15: Bangalore -> Singapore — Indian Ocean (south)
    3: ( 175,  55),   # May 19: Singapore -> SF — over N Pacific
    4: (-105,  18),   # May 19: SF -> Austin — over Mexico
}

FRAMES_PER_LEG = 26     # plane in motion
HOLD_FRAMES = 8         # pause after each leg lands (label settles in)
END_PAUSE_FRAMES = 36   # final pause before loop restarts
FPS = 18

ROUTE_COLOR = "#d7263d"
PLANE_COLOR = "#ffcf00"
CITY_COLOR  = "#1a1a1a"


def great_circle(lon1: float, lat1: float, lon2: float, lat2: float, n: int = 200) -> np.ndarray:
    """Return (n+2, 2) array of (lon, lat) along the great circle."""
    inner = GEOD.npts(lon1, lat1, lon2, lat2, n)
    return np.array([(lon1, lat1), *inner, (lon2, lat2)])


# Pre-compute routes
for leg in LEGS:
    lo1, la1 = CITIES[leg["from"]]
    lo2, la2 = CITIES[leg["to"]]
    leg["points"] = great_circle(lo1, la1, lo2, la2, 240)


def setup_map(ax) -> None:
    ax.set_global()
    ax.add_feature(cfeature.OCEAN,    facecolor="#cfe4f5", zorder=0)
    ax.add_feature(cfeature.LAND,     facecolor="#f1e3c6", zorder=1)
    ax.add_feature(cfeature.LAKES,    facecolor="#cfe4f5", edgecolor="none", zorder=1.1)
    ax.add_feature(cfeature.COASTLINE, edgecolor="#6f6f6f", linewidth=0.45, zorder=2)
    ax.add_feature(cfeature.BORDERS,   edgecolor="#9c9c9c", linewidth=0.3, alpha=0.7, zorder=2)


def draw_cities(ax) -> None:
    for name, (lon, lat) in CITIES.items():
        ax.plot(lon, lat, marker="o", markersize=6,
                markerfacecolor=CITY_COLOR, markeredgecolor="white",
                markeredgewidth=1.2, transform=ccrs.PlateCarree(), zorder=6)
        anchor, dx, dy = CITY_LABEL_OFFSET[name]
        ha = "right" if anchor == "right" else "left"
        text_tf = offset_copy(ccrs.PlateCarree()._as_mpl_transform(ax),
                              fig=ax.figure, x=dx, y=dy, units="points")
        ax.text(lon, lat, name, transform=text_tf,
                ha=ha, va="center", fontsize=10, fontweight="bold",
                color="#1a1a1a",
                path_effects=[],
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.75),
                zorder=7)


def draw_route_label(ax, leg_idx: int, leg: dict) -> None:
    label_lon, label_lat = LEG_LABEL_ANCHOR[leg_idx]
    text = f"{leg['date']}: {leg['from']} → {leg['to']}"
    ax.text(label_lon, label_lat, text,
            transform=ccrs.PlateCarree(),
            ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#222",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff8d6",
                      edgecolor=ROUTE_COLOR, linewidth=1.0, alpha=0.95),
            zorder=8)


def draw_completed_route(ax, leg: dict) -> None:
    pts = leg["points"]
    ax.plot(pts[:, 0], pts[:, 1], color=ROUTE_COLOR, linewidth=2.4,
            transform=ccrs.Geodetic(), zorder=4, solid_capstyle="round")


def draw_partial_route(ax, leg: dict, progress: float) -> None:
    pts = leg["points"]
    n = max(2, int(len(pts) * progress))
    sub = pts[:n]
    ax.plot(sub[:, 0], sub[:, 1], color=ROUTE_COLOR, linewidth=2.4,
            transform=ccrs.Geodetic(), zorder=4, solid_capstyle="round")
    # Plane marker at leading edge
    head = sub[-1]
    prev = sub[-2]
    # Compute heading for triangle rotation
    dlon = head[0] - prev[0]
    dlat = head[1] - prev[1]
    angle = np.degrees(np.arctan2(dlat, dlon))
    ax.plot(head[0], head[1], marker=(3, 0, angle - 90), markersize=14,
            markerfacecolor=PLANE_COLOR, markeredgecolor="#1a1a1a",
            markeredgewidth=1.0, transform=ccrs.PlateCarree(), zorder=9)


def render_frame(frame_idx: int, total_anim_frames: int, out_path: Path) -> None:
    fig = plt.figure(figsize=(10, 10), dpi=108)
    fig.patch.set_facecolor("#0d1b2a")
    proj = ccrs.Robinson(central_longitude=30)
    ax = plt.axes([0.02, 0.06, 0.96, 0.88], projection=proj)
    setup_map(ax)

    # Title strip
    fig.text(0.5, 0.955, "Around the World — May 2026",
             ha="center", va="center", fontsize=18, fontweight="bold",
             color="#f5f5f5")
    fig.text(0.5, 0.028, "Austin → London → Bangalore → Singapore → SFO → Austin",
             ha="center", va="center", fontsize=10.5, color="#cfd8e3")

    draw_cities(ax)

    slot = FRAMES_PER_LEG + HOLD_FRAMES
    animation_total = len(LEGS) * slot
    in_end_pause = frame_idx >= animation_total

    if in_end_pause:
        for i, leg in enumerate(LEGS):
            draw_completed_route(ax, leg)
            draw_route_label(ax, i, leg)
    else:
        active_leg = frame_idx // slot
        sub = frame_idx % slot
        in_flight = sub < FRAMES_PER_LEG
        progress = ((sub + 1) / FRAMES_PER_LEG) if in_flight else 1.0

        for i, leg in enumerate(LEGS):
            if i < active_leg:
                draw_completed_route(ax, leg)
                draw_route_label(ax, i, leg)
            elif i == active_leg:
                if in_flight:
                    draw_partial_route(ax, leg, progress)
                else:
                    # Hold: leg has landed — show full route + label
                    draw_completed_route(ax, leg)
                    draw_route_label(ax, i, leg)

    fig.savefig(out_path, dpi=108, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    out_dir = Path("frames")
    out_dir.mkdir(exist_ok=True)

    total_anim = len(LEGS) * (FRAMES_PER_LEG + HOLD_FRAMES)
    total_frames = total_anim + END_PAUSE_FRAMES

    frame_paths: list[Path] = []
    for i in range(total_frames):
        path = out_dir / f"frame_{i:04d}.png"
        render_frame(i, total_anim, path)
        frame_paths.append(path)
        if (i + 1) % 10 == 0 or i == total_frames - 1:
            print(f"  rendered {i + 1}/{total_frames}")

    # Build GIF (with optimized palette)
    print("Building GIF...")
    images = []
    for p in frame_paths:
        img = Image.open(p).convert("RGB")
        # Downscale to 720 for smaller GIF size (still very Instagram-friendly)
        img = img.resize((720, 720), Image.LANCZOS)
        images.append(img)
    duration_ms = int(1000 / FPS)
    images[0].save(
        "flight_map.gif",
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )

    # Build MP4 too (Instagram converts GIFs to video on upload — MP4 = best quality)
    print("Building MP4...")
    with imageio.get_writer("flight_map.mp4", fps=FPS, codec="libx264",
                            quality=8, macro_block_size=1) as writer:
        for p in frame_paths:
            writer.append_data(imageio.imread(p))

    print("Done.")
    print("Outputs: flight_map.gif, flight_map.mp4")


if __name__ == "__main__":
    main()
