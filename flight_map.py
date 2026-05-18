"""Animated 3D-globe flight tracker — portrait 9:16 for Instagram Reels.

For each flight segment, the camera (NearsidePerspective) sits over the
departure city zoomed in, then rises to a wide view as the plane flies along
its great-circle route while the globe spins to follow, then zooms back in
on the arrival city.
"""
from __future__ import annotations

from pathlib import Path as FsPath

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio.v2 as imageio
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pyproj import Geod

GEOD = Geod(ellps="WGS84")

CITIES = {
    "Austin":        (-97.7431, 30.2672, "USA"),
    "London":        ( -0.1276, 51.5074, "UK"),
    "Bangalore":     ( 77.5946, 12.9716, "INDIA"),
    "Singapore":     (103.8198,  1.3521, "SINGAPORE"),
    "San Francisco": (-122.4194, 37.7749, "USA"),
}

# Animation segments (the May 19 leg has two sub-segments)
SEGMENTS = [
    {"from": "Austin",        "to": "London",        "color": "#3aa0ff", "leg": 0},
    {"from": "London",        "to": "Bangalore",     "color": "#b06ee6", "leg": 1},
    {"from": "Bangalore",     "to": "Singapore",     "color": "#2bd47d", "leg": 2},
    {"from": "Singapore",     "to": "San Francisco", "color": "#ff5b5b", "leg": 3},
    {"from": "San Francisco", "to": "Austin",        "color": "#ff5b5b", "leg": 3},
]

DISPLAY_LEGS = [
    {"date": "May 11", "destination": "London",    "label": "AUSTIN  →  LONDON",           "color": "#3aa0ff"},
    {"date": "May 12", "destination": "Bangalore", "label": "LONDON  →  BANGALORE",        "color": "#b06ee6"},
    {"date": "May 15", "destination": "Singapore", "label": "BANGALORE  →  SINGAPORE",     "color": "#2bd47d"},
    {"date": "May 19", "destination": "Austin",    "label": "SINGAPORE  →  SF  →  AUSTIN", "color": "#ff5b5b"},
]

LOW_ALT  =  4_000_000   # zoomed in, ~50° visible
HIGH_ALT = 22_000_000   # zoomed out, globe view

DEPARTURE_FRAMES = 8    # camera holds at departure city
FLIGHT_FRAMES    = 56   # plane in motion + camera follow
ARRIVAL_FRAMES   = 14   # camera holds at arrival city
END_SPIN_FRAMES  = 96   # final 360° rotation showing all legs
FPS = 24

WIDTH, HEIGHT = 1080, 1920

BG_COLOR    = "#04081a"
TEXT_COLOR  = "#f0f0f0"
ACCENT      = "#ffcb00"
MUTED       = "#5a6c75"


def great_circle_points(lon1, lat1, lon2, lat2, n=240):
    inner = GEOD.npts(lon1, lat1, lon2, lat2, n)
    return np.array([(lon1, lat1), *inner, (lon2, lat2)])


for seg in SEGMENTS:
    lo1, la1 = CITIES[seg["from"]][:2]
    lo2, la2 = CITIES[seg["to"]][:2]
    seg["geo"] = great_circle_points(lo1, la1, lo2, la2, 240)


def interp_along(geo: np.ndarray, t: float) -> tuple[float, float]:
    if t <= 0: return float(geo[0][0]), float(geo[0][1])
    if t >= 1: return float(geo[-1][0]), float(geo[-1][1])
    idx_f = t * (len(geo) - 1)
    idx = int(idx_f)
    frac = idx_f - idx
    if idx >= len(geo) - 1:
        return float(geo[-1][0]), float(geo[-1][1])
    lo1, la1 = geo[idx]
    lo2, la2 = geo[idx + 1]
    dlon = lo2 - lo1
    if dlon > 180: dlon -= 360
    if dlon < -180: dlon += 360
    return float(lo1 + frac * dlon), float(la1 + frac * (la2 - la1))


def heading_along(geo: np.ndarray, t: float) -> float:
    idx = max(1, min(len(geo) - 1, int(t * (len(geo) - 1))))
    p = geo[idx - 1]
    c = geo[idx]
    dlon = float(c[0] - p[0])
    if dlon > 180: dlon -= 360
    if dlon < -180: dlon += 360
    dlat = float(c[1] - p[1])
    return float(np.degrees(np.arctan2(dlat, dlon)))


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def altitude_factor(t: float) -> float:
    """0 at start/end, 1 in middle, smooth ramps over first/last 30%."""
    if t < 0.30:
        return smoothstep(t / 0.30)
    if t > 0.70:
        return smoothstep((1 - t) / 0.30)
    return 1.0


def setup_globe(ax) -> None:
    ax.stock_img()
    # Slight blue overlay to richen the ocean
    ax.add_feature(cfeature.OCEAN, facecolor=(0.30, 0.55, 0.85, 0.35),
                   edgecolor='none', zorder=0.5)
    ax.add_feature(cfeature.COASTLINE, edgecolor='#333', linewidth=0.4, zorder=2)
    ax.add_feature(cfeature.BORDERS, edgecolor='#444', linewidth=0.25, alpha=0.55, zorder=2)


def is_last_subsegment(seg_idx: int) -> bool:
    if seg_idx == len(SEGMENTS) - 1:
        return True
    return SEGMENTS[seg_idx]["leg"] != SEGMENTS[seg_idx + 1]["leg"]


def render_frame(state: dict, out_path: FsPath) -> None:
    fig = plt.figure(figsize=(WIDTH / 120, HEIGHT / 120), dpi=120)
    fig.patch.set_facecolor(BG_COLOR)

    proj = ccrs.NearsidePerspective(
        central_longitude=state["cam_lon"],
        central_latitude=state["cam_lat"],
        satellite_height=state["altitude"],
    )
    # Square globe axes centered in portrait frame
    # Frame is 1080×1920. Make axes 1080×1080 with 300px footer, 540px header
    ax = plt.axes([0.0, 300 / HEIGHT, 1.0, 1080 / HEIGHT], projection=proj)
    setup_globe(ax)

    route_halo = [pe.withStroke(linewidth=7.5, foreground="white", alpha=0.85)]

    # Completed routes (full dashed line)
    for r in state["completed_routes"]:
        pts = r["geo"]
        ax.plot(pts[:, 0], pts[:, 1], color=r["color"],
                linewidth=4.2, linestyle=(0, (7, 4)),
                transform=ccrs.Geodetic(), zorder=5,
                solid_capstyle="round",
                path_effects=route_halo)

    # Active partial route
    if state["active_route"] is not None:
        r = state["active_route"]
        pts = r["geo"]
        n = max(2, int(len(pts) * state["active_progress"]))
        sub = pts[:n]
        ax.plot(sub[:, 0], sub[:, 1], color=r["color"],
                linewidth=4.2, linestyle=(0, (7, 4)),
                transform=ccrs.Geodetic(), zorder=5,
                solid_capstyle="round",
                path_effects=route_halo)

    # City dots (drawn always; cartopy hides ones on the far side)
    for name, (lon, lat, _) in CITIES.items():
        ax.plot(lon, lat, marker='o', markersize=7,
                markerfacecolor=ACCENT, markeredgecolor='white',
                markeredgewidth=1.4, transform=ccrs.PlateCarree(), zorder=8)

    # Focus city label (only during departure/arrival pause)
    if state.get("focus_city"):
        city = state["focus_city"]
        lon, lat, country = CITIES[city]
        from matplotlib.transforms import offset_copy
        tf = offset_copy(ccrs.PlateCarree()._as_mpl_transform(ax),
                         fig=fig, x=0, y=26, units='points')
        ax.text(lon, lat, f"{city.upper()}\n{country}",
                transform=tf, ha='center', va='bottom',
                fontsize=13, fontweight='bold', color='white',
                linespacing=1.15,
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor=(0, 0, 0, 0.75),
                          edgecolor=ACCENT, linewidth=1.4),
                zorder=12)

    # Plane (rotation = heading in degrees; ✈ glyph default points right)
    if state.get("plane"):
        plon, plat, pheading, pcolor = state["plane"]
        ax.text(plon, plat, "✈",
                transform=ccrs.PlateCarree(),
                ha='center', va='center',
                fontsize=38, color=pcolor,
                rotation=pheading, rotation_mode='anchor',
                zorder=11,
                path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

    # ---- HEADER (top): static trip title ----
    fig.text(0.5, 0.955, "Globetrotting",
             ha='center', va='top', fontsize=38,
             color=TEXT_COLOR, fontweight='bold',
             family='serif', fontstyle='italic')
    fig.text(0.5, 0.918, "May 11  –  May 19",
             ha='center', va='top', fontsize=20,
             color=MUTED, fontweight='bold',
             family='monospace')

    # ---- FOOTER (bottom): destination + date strip ----
    n = len(DISPLAY_LEGS)
    y_dot      = 0.118
    y_loc      = 0.075
    y_date     = 0.040
    edge_pad   = 0.10
    span       = 1.0 - 2 * edge_pad
    for i, dleg in enumerate(DISPLAY_LEGS):
        x = edge_pad + i * (span / (n - 1))
        completed = i in state["completed_display_legs"]
        active    = i == state["active_display_leg"]
        lit       = completed or active
        if active and not completed:
            color, dot_size = dleg["color"], 30
        elif completed:
            color, dot_size = dleg["color"], 24
        else:
            color, dot_size = "#3a3f4a", 20
        fig.text(x, y_dot, "●", ha='center', va='center',
                 fontsize=dot_size, color=color)
        loc_color  = dleg["color"] if lit else "#555"
        date_color = dleg["color"] if lit else "#444"
        loc_weight = 'bold' if lit else 'normal'
        fig.text(x, y_loc, dleg["destination"],
                 ha='center', va='center',
                 fontsize=15, color=loc_color, fontweight=loc_weight,
                 family='sans-serif')
        fig.text(x, y_date, dleg["date"],
                 ha='center', va='center',
                 fontsize=13, color=date_color,
                 fontweight='normal', alpha=0.95 if lit else 0.7)
        # Connecting line between dots — solid when the leg into the *next* stop is done
        if i < n - 1:
            x_next = edge_pad + (i + 1) * (span / (n - 1))
            done_to_next = (i + 1) in state["completed_display_legs"]
            line_color = "#cfd5dc" if done_to_next else "#2a2f38"
            fig.add_artist(plt.Line2D([x + 0.022, x_next - 0.022],
                                       [y_dot, y_dot],
                                       color=line_color, linewidth=1.8,
                                       transform=fig.transFigure))

    fig.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def compute_timeline() -> list[dict]:
    frames = []

    for seg_idx, seg in enumerate(SEGMENTS):
        dep_lon, dep_lat = CITIES[seg["from"]][:2]
        arr_lon, arr_lat = CITIES[seg["to"]][:2]
        leg_idx = seg["leg"]
        completed_before = set(range(leg_idx))
        completed_after  = (completed_before | {leg_idx}
                            if is_last_subsegment(seg_idx) else completed_before)

        completed_routes_before = [SEGMENTS[k] for k in range(seg_idx)]
        completed_routes_after  = [SEGMENTS[k] for k in range(seg_idx + 1)]

        first_heading = heading_along(seg["geo"], 0.02)
        last_heading  = heading_along(seg["geo"], 0.98)

        # Departure pause
        for _ in range(DEPARTURE_FRAMES):
            frames.append({
                "cam_lon": dep_lon, "cam_lat": dep_lat,
                "altitude": LOW_ALT,
                "completed_routes": completed_routes_before,
                "active_route": None,
                "active_progress": 0.0,
                "plane": (dep_lon, dep_lat, first_heading, seg["color"]),
                "focus_city": seg["from"],
                "active_display_leg": leg_idx,
                "completed_display_legs": completed_before,
            })

        # Flight
        for i in range(FLIGHT_FRAMES):
            t = (i + 0.5) / FLIGHT_FRAMES
            cam_t = smoothstep(t)
            cam_lon, cam_lat = interp_along(seg["geo"], cam_t)
            alt_f = altitude_factor(t)
            altitude = LOW_ALT + (HIGH_ALT - LOW_ALT) * alt_f
            plane_lon, plane_lat = interp_along(seg["geo"], t)
            plane_heading = heading_along(seg["geo"], t)
            frames.append({
                "cam_lon": cam_lon, "cam_lat": cam_lat,
                "altitude": altitude,
                "completed_routes": completed_routes_before,
                "active_route": seg,
                "active_progress": t,
                "plane": (plane_lon, plane_lat, plane_heading, seg["color"]),
                "focus_city": None,
                "active_display_leg": leg_idx,
                "completed_display_legs": completed_before,
            })

        # Arrival pause
        for _ in range(ARRIVAL_FRAMES):
            frames.append({
                "cam_lon": arr_lon, "cam_lat": arr_lat,
                "altitude": LOW_ALT,
                "completed_routes": completed_routes_after,
                "active_route": None,
                "active_progress": 1.0,
                "plane": (arr_lon, arr_lat, last_heading, seg["color"]),
                "focus_city": seg["to"],
                "active_display_leg": leg_idx,
                "completed_display_legs": completed_after,
            })

    # End spin: wide globe view doing a full 360° rotation, all routes visible
    last_leg = len(DISPLAY_LEGS) - 1
    completed_all = set(range(len(DISPLAY_LEGS)))
    aus_lon, aus_lat = CITIES["Austin"][:2]
    for i in range(END_SPIN_FRAMES):
        rot = (i / END_SPIN_FRAMES) * 360.0
        frames.append({
            "cam_lon": aus_lon + rot,
            "cam_lat": aus_lat,
            "altitude": HIGH_ALT,
            "completed_routes": list(SEGMENTS),
            "active_route": None,
            "active_progress": 1.0,
            "plane": None,
            "focus_city": None,
            "active_display_leg": last_leg,
            "completed_display_legs": completed_all,
        })

    return frames


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
        if (i + 1) % 20 == 0 or i == total - 1:
            print(f"  rendered {i + 1}/{total}")

    print("Building MP4...")
    with imageio.get_writer("flight_map.mp4", fps=FPS, codec="libx264",
                            quality=8, macro_block_size=1) as writer:
        for p in paths:
            writer.append_data(imageio.imread(p))

    print("Building GIF...")
    images = []
    for p in paths:
        img = Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=96)
        # Downscale heavily — portrait 9:16 GIFs at full res would be huge
        new_w = 432
        new_h = int(img.height * new_w / img.width)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        images.append(img)
    images[0].save(
        "flight_map.gif",
        save_all=True, append_images=images[1:],
        duration=int(1000 / FPS), loop=0, optimize=True, disposal=2,
    )

    print("Done. Outputs: flight_map.mp4, flight_map.gif")


if __name__ == "__main__":
    main()
