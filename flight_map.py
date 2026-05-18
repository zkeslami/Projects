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
    # (lon, lat, country, IATA code)
    "Austin":        (-97.7431, 30.2672, "USA",       "AUS"),
    "London":        ( -0.1276, 51.5074, "UK",        "LHR"),
    "Bangalore":     ( 77.5946, 12.9716, "INDIA",     "BLR"),
    "Singapore":     (103.8198,  1.3521, "SINGAPORE", "SIN"),
    "San Francisco": (-122.4194, 37.7749, "USA",      "SFO"),
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

# Pre-generate a deterministic set of "stars" for the space background
_rng = np.random.default_rng(42)
N_STARS = 240
_STAR_X = _rng.uniform(0.0, 1.0, N_STARS)
_STAR_Y = _rng.uniform(0.0, 1.0, N_STARS)
_STAR_SZ = _rng.uniform(0.4, 1.8, N_STARS)
_STAR_AL = _rng.uniform(0.3, 0.95, N_STARS)


def great_circle_points(lon1, lat1, lon2, lat2, n=240):
    inner = GEOD.npts(lon1, lat1, lon2, lat2, n)
    return np.array([(lon1, lat1), *inner, (lon2, lat2)])


for seg in SEGMENTS:
    lo1, la1 = CITIES[seg["from"]][:2]
    lo2, la2 = CITIES[seg["to"]][:2]
    seg["geo"] = great_circle_points(lo1, la1, lo2, la2, 240)
    _, _, dist_m = GEOD.inv(lo1, la1, lo2, la2)
    seg["distance_km"] = dist_m / 1000.0
    # Estimate duration: 850 km/h cruise + 30 min for taxi/climb/descent
    seg["duration_hours"] = seg["distance_km"] / 850.0 + 0.5
    seg["from_code"] = CITIES[seg["from"]][3]
    seg["to_code"]   = CITIES[seg["to"]][3]

# Cumulative distance at the END of each segment (for the running km counter)
_cum = 0.0
for seg in SEGMENTS:
    _cum += seg["distance_km"]
    seg["cum_km_end"] = _cum

TOTAL_DISTANCE_KM    = sum(s["distance_km"] for s in SEGMENTS)
TOTAL_FLIGHT_HOURS   = sum(s["duration_hours"] for s in SEGMENTS)
COUNTRIES_VISITED    = len({CITIES[c][2] for c in CITIES})  # 4 unique
# Going east the entire trip is +24h around the world, but for display we
# show the unique time zones crossed: AUS(CDT,-5) LHR(+1) BLR(+5.5) SIN(+8) SFO(-7) = 5
TIMEZONES_TOUCHED    = 5


def format_hours(h: float) -> str:
    total_min = int(round(h * 60))
    return f"{total_min // 60}h {total_min % 60:02d}m"


def format_km(km: float) -> str:
    return f"{int(round(km)):,} km"


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

    # Star field across the entire figure (will be partially covered by the globe)
    for i in range(N_STARS):
        fig.add_artist(plt.Circle(
            (_STAR_X[i], _STAR_Y[i]), radius=_STAR_SZ[i] * 0.0014,
            transform=fig.transFigure,
            facecolor=(1, 1, 1, _STAR_AL[i]),
            edgecolor='none', zorder=0,
        ))

    proj = ccrs.NearsidePerspective(
        central_longitude=state["cam_lon"],
        central_latitude=state["cam_lat"],
        satellite_height=state["altitude"],
    )
    # Square globe axes centered in portrait frame
    # Frame is 1080×1920. Make axes 1080×1080 with 300px footer, 540px header
    ax = plt.axes([0.0, 300 / HEIGHT, 1.0, 1080 / HEIGHT], projection=proj)
    ax.set_facecolor((0, 0, 0, 0))  # transparent so stars show through
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
    for name, (lon, lat, _, _) in CITIES.items():
        ax.plot(lon, lat, marker='o', markersize=7,
                markerfacecolor=ACCENT, markeredgecolor='white',
                markeredgewidth=1.4, transform=ccrs.PlateCarree(), zorder=8)

    # Focus city label (during departure/arrival pause)
    if state.get("focus_city"):
        city = state["focus_city"]
        lon, lat, country, code = CITIES[city]
        from matplotlib.transforms import offset_copy
        tf = offset_copy(ccrs.PlateCarree()._as_mpl_transform(ax),
                         fig=fig, x=0, y=30, units='points')
        ax.text(lon, lat, f"{city.upper()}\n{code}  ·  {country}",
                transform=tf, ha='center', va='bottom',
                fontsize=13, fontweight='bold', color='white',
                linespacing=1.25,
                bbox=dict(boxstyle='round,pad=0.45',
                          facecolor=(0, 0, 0, 0.78),
                          edgecolor=ACCENT, linewidth=1.4),
                zorder=12)

    # Landing pulse: expanding white-on-color rings + "ARRIVED ✓" pop
    if state.get("phase") == "arrival" and state.get("landing_pulse", 99) < 10:
        i = state["landing_pulse"]
        arr_lon, arr_lat = CITIES[state["active_segment"]["to"]][:2]
        pulse_color = state["active_segment"]["color"]
        for k in range(3):
            phase = i - k * 2.5
            if 0 <= phase < 6:
                t = phase / 6.0
                ring_size = 35 + t * 120
                alpha = (1.0 - t) ** 1.3 * 0.95
                # White outer halo
                ax.plot(arr_lon, arr_lat, marker='o',
                        markersize=ring_size,
                        markerfacecolor='none',
                        markeredgecolor='white',
                        markeredgewidth=4.5,
                        alpha=alpha * 0.55,
                        transform=ccrs.PlateCarree(), zorder=13)
                # Colored ring inside
                ax.plot(arr_lon, arr_lat, marker='o',
                        markersize=ring_size,
                        markerfacecolor='none',
                        markeredgecolor=pulse_color,
                        markeredgewidth=3.0,
                        alpha=alpha,
                        transform=ccrs.PlateCarree(), zorder=14)
        # "ARRIVED" pop near top of frame for first ~6 frames
        if i < 6:
            pop_a = max(0.0, (1.0 - i / 6.0) ** 0.7) * 0.95
            fig.text(0.5, 0.84, "✓  ARRIVED",
                     ha='center', va='center', fontsize=18,
                     color=(*plt.matplotlib.colors.to_rgb(pulse_color), pop_a),
                     fontweight='bold', family='monospace',
                     bbox=dict(boxstyle='round,pad=0.4',
                               facecolor=(0, 0, 0, 0.7 * pop_a),
                               edgecolor=(*plt.matplotlib.colors.to_rgb(pulse_color), pop_a),
                               linewidth=1.5))

    # Contrail: a thin wisp behind the plane during flight
    if state.get("phase") == "flight" and state.get("active_route") is not None:
        seg = state["active_segment"]
        pts = seg["geo"]
        t = state["flight_t"]
        n = max(2, int(len(pts) * t))
        # Last ~12% of route is the contrail (the freshest)
        tail_n = max(2, int(0.12 * len(pts)))
        start = max(0, n - tail_n)
        sub = pts[start:n]
        # Draw as several segments with increasing alpha towards the head
        for i in range(len(sub) - 1):
            local_t = (i + 1) / len(sub)
            ax.plot(sub[i:i + 2, 0], sub[i:i + 2, 1],
                    color=(1, 1, 1, 0.15 + 0.65 * local_t),
                    linewidth=2.4, solid_capstyle='round',
                    transform=ccrs.Geodetic(), zorder=6)

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

    # ---- HEADER (top) ----
    fig.text(0.5, 0.965, "Globetrotting",
             ha='center', va='top', fontsize=38,
             color=TEXT_COLOR, fontweight='bold',
             family='serif', fontstyle='italic')
    fig.text(0.5, 0.928, "May 11  –  May 19",
             ha='center', va='top', fontsize=18,
             color=MUTED, fontweight='bold',
             family='monospace')

    # Mid-flight info card (codes · distance · duration)
    seg = state.get("active_segment")
    if seg is not None and state.get("phase") in ("departure", "flight", "arrival"):
        info = (f"{seg['from_code']}  →  {seg['to_code']}     "
                f"{format_km(seg['distance_km'])}     "
                f"{format_hours(seg['duration_hours'])}")
        fig.text(0.5, 0.892, info,
                 ha='center', va='top', fontsize=14,
                 color=seg["color"], fontweight='bold',
                 family='monospace',
                 bbox=dict(boxstyle='round,pad=0.35',
                           facecolor=(0, 0, 0, 0.55),
                           edgecolor=seg["color"], linewidth=1.0))

    # Cumulative-km odometer (top-right corner)
    cum_km = state.get("cum_km", 0.0)
    fig.text(0.94, 0.972, "TOTAL FLOWN",
             ha='right', va='top', fontsize=9,
             color=MUTED, fontweight='bold',
             family='monospace')
    fig.text(0.94, 0.955, format_km(cum_km).upper(),
             ha='right', va='top', fontsize=16,
             color=ACCENT, fontweight='bold',
             family='monospace')

    # End-spin big-stats reveal (fades in over the rotating globe)
    if state.get("phase") == "end_spin":
        a = state.get("stats_alpha", 0.0)
        if a > 0:
            # Translucent dark scrim across the globe area for legibility
            scrim = plt.Rectangle((0.0, 300 / HEIGHT), 1.0, 1080 / HEIGHT,
                                  transform=fig.transFigure,
                                  facecolor=(0, 0, 0, 0.45 * a),
                                  edgecolor='none', zorder=20)
            fig.add_artist(scrim)
            # Big stats stacked vertically inside the globe area
            cx = 0.5
            base_y = 0.66
            line_h = 0.07
            stats = [
                ("TOTAL DISTANCE", format_km(TOTAL_DISTANCE_KM).upper(),  ACCENT),
                ("TIME IN THE AIR", format_hours(TOTAL_FLIGHT_HOURS).upper(),  "#ff5b5b"),
                ("COUNTRIES",       f"{COUNTRIES_VISITED}",                "#2bd47d"),
                ("TIME ZONES",      f"{TIMEZONES_TOUCHED}",                "#b06ee6"),
            ]
            for i, (lbl, val, col) in enumerate(stats):
                y = base_y - i * line_h
                fig.text(cx, y, lbl,
                         ha='center', va='center', fontsize=11,
                         color=(1, 1, 1, 0.55 * a), fontweight='bold',
                         family='monospace', zorder=21)
                fig.text(cx, y - 0.028, val,
                         ha='center', va='center', fontsize=26,
                         color=(*plt.matplotlib.colors.to_rgb(col), a),
                         fontweight='bold', family='monospace', zorder=21)
            fig.text(cx, base_y - 4 * line_h - 0.01, "AROUND THE WORLD",
                     ha='center', va='center', fontsize=15,
                     color=(*plt.matplotlib.colors.to_rgb(ACCENT), a),
                     fontweight='bold', family='serif', fontstyle='italic',
                     zorder=21)

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

        cum_before = seg["cum_km_end"] - seg["distance_km"]
        cum_end    = seg["cum_km_end"]

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
                "cum_km": cum_before,
                "active_segment": seg,
                "phase": "departure",
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
                "cum_km": cum_before + (cum_end - cum_before) * t,
                "active_segment": seg,
                "phase": "flight",
                "flight_t": t,
            })

        # Arrival pause
        for i in range(ARRIVAL_FRAMES):
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
                "cum_km": cum_end,
                "active_segment": seg,
                "phase": "arrival",
                "landing_pulse": i,  # 0,1,2,... — render shows expanding ring for small values
            })

    # End spin: wide globe view doing a full 360° rotation, all routes visible
    last_leg = len(DISPLAY_LEGS) - 1
    completed_all = set(range(len(DISPLAY_LEGS)))
    aus_lon, aus_lat = CITIES["Austin"][:2]
    for i in range(END_SPIN_FRAMES):
        rot = (i / END_SPIN_FRAMES) * 360.0
        # Stats reveal: fade in after first 8 frames, fully visible by frame 24, hold
        stats_t = max(0.0, min(1.0, (i - 8) / 16.0))
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
            "cum_km": TOTAL_DISTANCE_KM,
            "active_segment": None,
            "phase": "end_spin",
            "stats_alpha": stats_t,
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
    # Encode via ffmpeg directly so we control all the Instagram-required bits:
    # H.264 main profile, yuv420p, +faststart (moov at front), and a silent
    # AAC audio track — without audio Instagram sometimes treats the upload
    # as a still photo.
    import subprocess
    pat = str(out_dir / "frame_%04d.png")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS), "-i", pat,
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-c:v", "libx264", "-profile:v", "main", "-level", "4.0",
        "-pix_fmt", "yuv420p", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k", "-shortest",
        "-movflags", "+faststart",
        "flight_map.mp4",
    ]
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
