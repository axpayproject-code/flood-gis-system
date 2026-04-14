import streamlit as st
import numpy as np
import folium
from streamlit.components.v1 import html
from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN

# Optional reverse geocoding
from geopy.geocoders import Nominatim

st.set_page_config(page_title="Flood Catchment GIS", layout="wide")

CENTER = [10.67, 122.95]
CELL = 0.00012

geolocator = Nominatim(user_agent="flood_gis_app")

# =========================
# GET PLACE NAME (SAFE)
# =========================
def get_place(lat, lon):
    try:
        loc = geolocator.reverse((lat, lon), language="en")
        return loc.raw.get("display_name", "Unknown") if loc else "Unknown"
    except:
        return "Unknown"

# =========================
# DEM SIMULATION
# =========================
def load_dem():
    size = 120
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    xx, yy = np.meshgrid(x, y)

    return np.sin(xx)*np.cos(yy)*50 + np.random.rand(size, size)*5 + 100

# =========================
# HYDROLOGY MODEL
# =========================
def slope(dem):
    gy, gx = np.gradient(dem)
    return np.sqrt(gx**2 + gy**2)

def flow(dem):
    acc = np.ones_like(dem)
    for i in range(1, dem.shape[0]-1):
        for j in range(1, dem.shape[1]-1):
            if dem[i, j] < np.mean(dem[i-1:i+2, j-1:j+2]):
                acc[i, j] += 2
    return acc

def score(acc, slp):
    a = acc / (acc.max() + 1e-9)
    s = 1 - slp / (slp.max() + 1e-9)
    return a*0.75 + s*0.25

# =========================
# EXTRACT HOTSPOTS
# =========================
def extract(sc):
    return np.argwhere(sc > 0.82)

def cluster(points):
    if len(points) == 0:
        return {}

    labels = DBSCAN(eps=5, min_samples=4).fit_predict(points)

    zones = {}
    for p, l in zip(points, labels):
        if l == -1:
            continue
        zones.setdefault(l, []).append(p)

    return zones

def polygons(zones):
    out = []

    for zid, pts in zones.items():
        pts = np.array(pts)

        if len(pts) < 3:
            continue

        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pts)
            pts = pts[hull.vertices]
        except:
            pass

        poly = Polygon([(p[1], p[0]) for p in pts])

        # center point for labeling
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])

        out.append({
            "id": zid,
            "geometry": poly,
            "size": len(pts),
            "center": (cx, cy)
        })

    return out

# =========================
# ENGINEERING ANALYSIS
# =========================
def analyze(polys):
    zones = []

    for p in polys:
        size = p["size"]

        reduction = min(0.95, size / 600)

        lat = CENTER[0] + p["center"][0] * CELL
        lon = CENTER[1] + p["center"][1] * CELL

        place_name = get_place(lat, lon)

        zones.append({
            "id": p["id"],
            "area_ha": size * 0.6,
            "capacity_m3": size * 120,
            "flood_reduction": reduction,
            "priority": "HIGH" if reduction > 0.6 else "MEDIUM",
            "lat": lat,
            "lon": lon,
            "place": place_name,
            "geometry": p["geometry"]
        })

    return sorted(zones, key=lambda x: x["flood_reduction"], reverse=True)

# =========================
# MAP
# =========================
def make_map(zones):
    m = folium.Map(location=CENTER, zoom_start=11)

    for z in zones:

        coords = []
        for x, y in z["geometry"].exterior.coords:
            lat = CENTER[0] + y * CELL
            lon = CENTER[1] + x * CELL
            coords.append([lat, lon])

        color = "red" if z["flood_reduction"] > 0.6 else "orange"

        folium.Polygon(
            locations=coords,
            color="black",
            fill=True,
            fill_opacity=0.5,
            fill_color=color,
            popup=f"""
            <b>Catchment Zone {z['id']}</b><br>
            Place: {z['place']}<br>
            Area: {z['area_ha']:.2f} ha<br>
            Capacity: {z['capacity_m3']:.0f} m³<br>
            Flood Reduction: {z['flood_reduction']*100:.1f}%
            """
        ).add_to(m)

        # label marker
        folium.Marker(
            location=[z["lat"], z["lon"]],
            popup=z["place"],
            icon=folium.DivIcon(html=f"""
                <div style="
                    font-size:10px;
                    color:black;
                    background:white;
                    padding:2px;
                    border:1px solid black;
                ">
                Z{z['id']}
                </div>
            """)
        ).add_to(m)

    return m

# =========================
# UI
# =========================
st.title("🌊 Flood Catchment Planning System (With Place Labels)")

dem = load_dem()
slp = slope(dem)
acc = flow(dem)
sc = score(acc, slp)

pts = extract(sc)
zones_raw = cluster(pts)
polys = polygons(zones_raw)
zones = analyze(polys)

# =========================
# ANSWERS
# =========================
if zones:
    st.subheader("📊 Engineering Answers")

    st.write("### 1. Where should we build catchments?")
    st.success("Red zones + labeled locations on map")

    st.write("### 2. How big should they be?")
    st.info(f"Average size: {np.mean([z['area_ha'] for z in zones]):.2f} ha")

    st.write("### 3. Flood reduction potential?")
    st.warning(f"Average reduction: {np.mean([z['flood_reduction'] for z in zones])*100:.1f}%")

# =========================
# MAP
# =========================
st.subheader("🗺️ Interactive Flood Map")

m = make_map(zones)
html(m._repr_html_(), height=650)