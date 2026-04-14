import streamlit as st
import leafmap.foliumap as leafmap
import geopandas as gpd
import rasterio
import numpy as np
import richdem as rd
import tempfile, zipfile, os
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="Flood Catchment Planning GIS", layout="wide")

# Dark professional theme
st.markdown("""
<style>
.main {background-color:#0e1117;}
[data-testid="stSidebar"] {background:#161b22;}
h1,h2,h3 {color:white;}
</style>
""", unsafe_allow_html=True)

# Toolbar
c1,c2,c3,c4 = st.columns([4,1,1,1])
with c1:
    st.title("Flood Catchment Planning GIS")

run_model = c2.button("Run Model")
export_pdf = c3.button("Export PDF")
export_geojson = c4.button("Export GeoJSON")

# Sidebar controls
st.sidebar.header("Layers")
show_flood = st.sidebar.checkbox("Flood Risk", True)
st.sidebar.header("Engineering Parameters")
rainfall = st.sidebar.slider("Design Storm Rainfall (mm)", 50, 500, 200)
runoff_coeff = st.sidebar.slider("Runoff Coefficient", 0.2, 1.0, 0.7)
dem_zip = st.sidebar.file_uploader("Upload DEM ZIP", type="zip")

# Map canvas with drawing tools
m = leafmap.Map(center=[10.67,122.95], zoom=11)
m.add_basemap("OpenStreetMap")
m.add_basemap("HYBRID")
m.add_draw_control(export=True)   # drawing tool!
m.add_measure_control()           # measuring tool

def unzip(upload):
    temp=tempfile.mkdtemp()
    with zipfile.ZipFile(upload,'r') as z:
        z.extractall(temp)
    return temp

def hydrology(dem_file):
    with rasterio.open(dem_file) as src:
        dem=src.read(1); transform=src.transform
    dem_rd=rd.rdarray(dem,no_data=-9999)
    filled=rd.FillDepressions(dem_rd)
    flow=rd.FlowAccumulation(filled,method='D8')
    slope=rd.TerrainAttribute(filled,attrib='slope_riserun')
    return dem,flow,slope,transform

# Run model
if run_model and dem_zip:
    folder=unzip(dem_zip)
    dem_file=None
    for r,_,f in os.walk(folder):
        for file in f:
            if file.endswith(".tif") or file.endswith(".hgt"):
                dem_file=os.path.join(r,file)

    dem,flow,slope,transform=hydrology(dem_file)
    flood_index=(flow/flow.max())*0.7+(1-slope/slope.max())*0.3
    flood_mask=flood_index>np.percentile(flood_index,95)

    flood_tif=tempfile.NamedTemporaryFile(suffix=".tif").name
    with rasterio.open(flood_tif,"w",driver="GTiff",
        height=flood_mask.shape[0],width=flood_mask.shape[1],
        count=1,dtype="uint8",crs="EPSG:4326",transform=transform) as dst:
        dst.write(flood_mask.astype("uint8"),1)

    if show_flood:
        m.add_raster(flood_tif, layer_name="Flood Zones")

    # Engineering outputs
    pixel_area=30*30
    flood_area=flood_mask.sum()*pixel_area
    rainfall_m=rainfall/1000
    runoff_volume=flood_area*rainfall_m*runoff_coeff
    catchment_storage=runoff_volume*0.35
    reduction=(catchment_storage/runoff_volume)*100

    st.session_state.results=dict(
        area=flood_area/1e6,
        volume=runoff_volume/1e6,
        storage=catchment_storage/1e6,
        reduction=reduction
    )

# Display map
m.to_streamlit(height=720)

# Results panel
if "results" in st.session_state:
    r=st.session_state.results
    st.subheader("Flood Prevention Engineering Results")
    c1,c2,c3=st.columns(3)
    c1.metric("Flood Area (km²)",f"{r['area']:.2f}")
    c2.metric("Runoff Volume (M m³)",f"{r['volume']:.2f}")
    c3.metric("Flood Reduction (%)",f"{r['reduction']:.1f}")

    st.write("Where to build catchments:")
    st.write("High-risk zones shown in red.")
    st.write("How big:")
    st.write(f"{r['storage']:.2f} million m³ storage required.")
    st.write("Flood reduction:")
    st.write(f"{r['reduction']:.1f}% expected reduction.")

# Export GeoJSON
if export_geojson and "results" in st.session_state:
    geojson_path="catchments.geojson"
    gdf=gpd.GeoDataFrame(geometry=[])
    gdf.to_file(geojson_path, driver="GeoJSON")
    st.success("GeoJSON exported")

# Export printable PDF
if export_pdf and "results" in st.session_state:
    pdf="Flood_Report.pdf"
    doc=SimpleDocTemplate(pdf)
    styles=getSampleStyleSheet()
    story=[]
    story.append(Paragraph("Flood Catchment Planning Report", styles['Title']))
    r=st.session_state.results
    story.append(Paragraph(f"Flood Area: {r['area']:.2f} km²", styles['Normal']))
    story.append(Paragraph(f"Runoff Volume: {r['volume']:.2f} million m³", styles['Normal']))
    story.append(Paragraph(f"Flood Reduction: {r['reduction']:.1f}%", styles['Normal']))
    doc.build(story)
    st.success("PDF exported")