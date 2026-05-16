"""Animated world map of May 2026 flight legs — reference-style.

Inspired stylistically by a flight itinerary infographic: Natural Earth
shaded-relief background, dashed leg-colored routes, numbered callout
badges, and a rotating airplane icon at the leading edge.
"""
from __future__ import annotations

from pathlib import Path as FsPath

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import offset_copy
from PIL import Image
from pyproj import Geod

GEOD = Geod(ellps="WGS84")

# (lon, lat, country-label)
CITIES = {
    "Austin":        (-97.7431, 30.2672, "USA"),
    "London":        ( -0.1276, 51.5074, "UK"),
    "Bangalore":     ( 77.5946, 12.9716, "INDIA"),
    "Singapore":     (103.8198,  1.3521, "SINGAPORE"),
    "San Francisco": (-122.4194, 37.7749, "USA"),
}

CITY_LABEL_OFFSET = {
    "Austin":        ("left",    10,  -10),
    "London":        ("left",    10,   -2),
    "Bangalore":     ("left",    10,   -2),
    "Singapore":     ("left",    10,  -10),
    "San Francisco": ("left",    10,    2),
}

# One entry per "leg" the user described (4 legs, 4 dates, 4 colors)
LEGS = [
    {"date": "May 11", "label": "Austin to London",
     "color": "#2e7dd1", "segments": [("Austin", "London")]},
    {"date": "May 12", "label": "London to Bangalore",
     "color": "#8e44ad", "segments": [("London", "Bangalore")]},
    {"date": "May 15", "label": "Bangalore to Singapore",
     "color": "#1f9d55", "segments": [("Bangalore", "Singapore")]},
    {"date": "May 19", "label": "Singapore to San Francisco\nto Austin",
     "color": "#c0392b", "segments": [("Singapore", "San Francisco"),
                                      ("San Francisco", "Austin")]},
]

# Placement of each numbered badge/callout: (lon, lat, side)
# side controls which way the box extends from the badge dot.
LEG_BADGE = {
    0: (-55,  72, "right"),
    1: ( 18,  38, "right"),
    2: ( 95, -22, "right"),
    3: (175,  52, "left"),
}

OCEAN_LABELS = [
    ("ARCTIC OCEAN",         -10,  78),
    ("NORTH ATLANTIC\nOCEAN", -40,  30),
    ("NORTH PACIFIC OCEAN",  -160, 35),
    ("SOUTH PACIFIC\nOCEAN", -130, -25),
    ("SOUTH ATLANTIC\nOCEAN", -25, -30),
    ("INDIAN OCEAN",          80, -25),
]

FRAMES_PER_SEGMENT = 26
HOLD_FRAMES = 10        # pause after each leg lands
INTER_SEGMENT_PAUSE = 4 # pause at intermediate stop within a multi-segment leg
END_PAUSE_FRAMES = 38
FPS = 18

# Map projection — center around 30°E so Asia is on the right and the
# Singapore→SF great circle exits the right edge and re-enters at the left.
PROJ = ccrs.PlateCarree(central_longitude=30)


def great_circle(lon1: float, lat1: float, lon2: float, lat2: float, n: int = 200) -> np.ndarray:
    inner = GEOD.npts(lon1, lat1, lon2, lat2, n)
    return np.array([(lon1, lat1), *inner, (lon2, lat2)])


# Build the per-segment animation timeline
SEGMENTS = []
for leg_idx, leg in enumerate(LEGS):
    for seg_idx, (c1, c2) in enumerate(leg["segments"]):
        lo1, la1 = CITIES[c1][:2]
        lo2, la2 = CITIES[c2][:2]
        SEGMENTS.append({
            "leg_idx": leg_idx,
            "from": c1, "to": c2,
            "geo": great_circle(lo1, la1, lo2, la2, 240),
            "color": leg["color"],
            "is_last_in_leg": seg_idx == len(leg["segments"]) - 1,
            "is_first_in_leg": seg_idx == 0,
        })


def heading_deg(prev_pt: np.ndarray, curr_pt: np.ndarray) -> float:
    dlon = curr_pt[0] - prev_pt[0]
    # Handle antimeridian wrap
    if dlon > 180:   dlon -= 360
    if dlon < -180:  dlon += 360
    dlat = curr_pt[1] - prev_pt[1]
    return float(np.degrees(np.arctan2(dlat, dlon)))


def setup_map(ax) -> None:
    ax.set_global()
    ax.stock_img()
    # Boost ocean color with a translucent bright-blue overlay
    ax.add_feature(cfeature.OCEAN, facecolor=(0.49, 0.75, 0.92, 0.45),
                   edgecolor="none", zorder=1)
    ax.add_feature(cfeature.LAKES, facecolor=(0.49, 0.75, 0.92, 0.55),
                   edgecolor="none", zorder=1.1)
    ax.add_feature(cfeature.COASTLINE, edgecolor="#5a5a5a", linewidth=0.4, zorder=2)


def draw_ocean_labels(ax) -> None:
    for text, lon, lat in OCEAN_LABELS:
        ax.text(lon, lat, text,
                transform=ccrs.PlateCarree(),
                ha="center", va="center",
                fontsize=7.5, color="#1a4f7a",
                fontstyle="italic", alpha=0.85,
                zorder=3)


def draw_cities(ax) -> None:
    fig = ax.figure
    for name, (lon, lat, country) in CITIES.items():
        ax.plot(lon, lat, marker="o", markersize=5,
                markerfacecolor="#111", markeredgecolor="white",
                markeredgewidth=1.0, transform=ccrs.PlateCarree(), zorder=8)
        anchor, dx, dy = CITY_LABEL_OFFSET[name]
        ha = {"left": "left", "right": "right", "center": "center"}[anchor]
        text_tf = offset_copy(ccrs.PlateCarree()._as_mpl_transform(ax),
                              fig=fig, x=dx, y=dy, units="points")
        label = f"{name.upper()}\n({country})"
        ax.text(lon, lat, label, transform=text_tf,
                ha=ha, va="center", fontsize=8.5, fontweight="bold",
                color="#111", linespacing=1.05,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.55),
                zorder=9)


def draw_completed_segment(ax, seg: dict) -> None:
    pts = seg["geo"]
    ax.plot(pts[:, 0], pts[:, 1], color=seg["color"], linewidth=1.8,
            linestyle=(0, (6, 4)), transform=ccrs.Geodetic(),
            zorder=4, solid_capstyle="round")


def draw_partial_segment(ax, seg: dict, progress: float) -> None:
    pts = seg["geo"]
    n = max(2, int(len(pts) * progress))
    sub = pts[:n]
    ax.plot(sub[:, 0], sub[:, 1], color=seg["color"], linewidth=1.8,
            linestyle=(0, (6, 4)), transform=ccrs.Geodetic(),
            zorder=4, solid_capstyle="round")
    head = sub[-1]
    prev = sub[-2]
    angle = heading_deg(prev, head)
    # ✈ glyph default points right; rotation == heading aligns nose with travel
    ax.text(head[0], head[1], "✈",
            transform=ccrs.PlateCarree(),
            ha="center", va="center",
            fontsize=20, color=seg["color"],
            rotation=angle, rotation_mode="anchor",
            zorder=10)


def draw_leg_badge(ax, leg_idx: int, leg: dict) -> None:
    """Numbered colored circle + callout box, reference-style."""
    fig = ax.figure
    lon, lat, side = LEG_BADGE[leg_idx]
    color = leg["color"]

    # Badge circle (numbered)
    ax.plot(lon, lat, marker="o", markersize=18,
            markerfacecolor=color, markeredgecolor="white",
            markeredgewidth=1.5, transform=ccrs.PlateCarree(), zorder=11)
    ax.text(lon, lat, str(leg_idx + 1),
            transform=ccrs.PlateCarree(),
            ha="center", va="center",
            fontsize=10, fontweight="bold", color="white", zorder=12)

    # Callout text box, offset to one side of the badge
    if side == "right":
        offx, ha = 26, "left"
    else:
        offx, ha = -26, "right"
    text_tf = offset_copy(ccrs.PlateCarree()._as_mpl_transform(ax),
                          fig=fig, x=offx, y=0, units="points")
    text = f"{leg['date']}\n{leg['label']}"
    ax.text(lon, lat, text, transform=text_tf,
            ha=ha, va="center",
            fontsize=9, color="#111",
            linespacing=1.15,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=color, linewidth=1.6, alpha=0.95),
            zorder=11)


def draw_title(fig) -> None:
    fig.text(0.5, 0.94, "FLIGHT ITINERARY: MAY 11 – MAY 19",
             ha="center", va="center",
             fontsize=16, fontweight="bold", color="#111",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                       edgecolor="#111", linewidth=1.2))


def compute_timeline() -> list[dict]:
    """Return per-frame state: list of dicts describing what to render."""
    frames = []
    # Segments are animated in order. Each segment: FRAMES_PER_SEGMENT.
    # Between segments within a leg: INTER_SEGMENT_PAUSE (hold partial route + plane).
    # After last segment of a leg: HOLD_FRAMES (show badge + label).
    for seg_idx, seg in enumerate(SEGMENTS):
        # In-flight frames
        for i in range(FRAMES_PER_SEGMENT):
            frames.append({"kind": "flight", "seg_idx": seg_idx,
                           "progress": (i + 1) / FRAMES_PER_SEGMENT})
        if seg["is_last_in_leg"]:
            for _ in range(HOLD_FRAMES):
                frames.append({"kind": "hold", "seg_idx": seg_idx, "progress": 1.0})
        else:
            for _ in range(INTER_SEGMENT_PAUSE):
                frames.append({"kind": "hold", "seg_idx": seg_idx, "progress": 1.0})
    # End pause: everything visible
    for _ in range(END_PAUSE_FRAMES):
        frames.append({"kind": "end_pause"})
    return frames


def state_at(frame: dict) -> tuple[set[int], int | None, float, set[int]]:
    """Return (completed_segment_ids, active_segment_id_or_none,
              active_progress, completed_leg_ids_with_visible_badge)."""
    if frame["kind"] == "end_pause":
        completed_segs = set(range(len(SEGMENTS)))
        completed_legs = set(range(len(LEGS)))
        return completed_segs, None, 1.0, completed_legs

    seg_idx = frame["seg_idx"]
    prog = frame["progress"]

    if frame["kind"] == "flight":
        completed_segs = set(range(seg_idx))
        active = seg_idx
    else:  # hold
        completed_segs = set(range(seg_idx + 1))
        active = None

    completed_legs = set()
    for li, leg in enumerate(LEGS):
        seg_count = len(leg["segments"])
        # Determine the first/last seg-index in SEGMENTS for this leg
        offset = sum(len(LEGS[k]["segments"]) for k in range(li))
        last_seg_for_leg = offset + seg_count - 1
        if last_seg_for_leg in completed_segs:
            completed_legs.add(li)
    return completed_segs, active, prog, completed_legs


def render_frame(frame: dict, out_path: FsPath) -> None:
    # 1080 x 608 → 1.78:1 landscape (Instagram-friendly)
    fig = plt.figure(figsize=(10.0, 5.63), dpi=108)
    fig.patch.set_facecolor("#e8f1f7")
    ax = plt.axes([0.0, 0.0, 1.0, 0.88], projection=PROJ)
    setup_map(ax)
    draw_ocean_labels(ax)
    draw_cities(ax)

    completed_segs, active, progress, completed_legs = state_at(frame)

    # Draw completed segments
    for i, seg in enumerate(SEGMENTS):
        if i in completed_segs:
            draw_completed_segment(ax, seg)

    # Draw active in-flight segment
    if active is not None:
        draw_partial_segment(ax, SEGMENTS[active], progress)

    # Draw badges for legs that have completed all their segments
    for li in sorted(completed_legs):
        draw_leg_badge(ax, li, LEGS[li])

    draw_title(fig)

    fig.savefig(out_path, dpi=108, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    out_dir = FsPath("frames")
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()

    frames = compute_timeline()
    total = len(frames)

    paths: list[FsPath] = []
    for i, fr in enumerate(frames):
        p = out_dir / f"frame_{i:04d}.png"
        render_frame(fr, p)
        paths.append(p)
        if (i + 1) % 15 == 0 or i == total - 1:
            print(f"  rendered {i + 1}/{total}")

    print("Building GIF...")
    images = []
    for p in paths:
        img = Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=128)
        w, h = img.size
        new_w = 640
        new_h = int(h * new_w / w)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        images.append(img)
    duration_ms = int(1000 / FPS)
    images[0].save(
        "flight_map.gif",
        save_all=True, append_images=images[1:],
        duration=duration_ms, loop=0, optimize=True, disposal=2,
    )

    print("Building MP4...")
    with imageio.get_writer("flight_map.mp4", fps=FPS, codec="libx264",
                            quality=8, macro_block_size=1) as writer:
        for p in paths:
            writer.append_data(imageio.imread(p))

    print("Done. Outputs: flight_map.gif, flight_map.mp4")


if __name__ == "__main__":
    main()
