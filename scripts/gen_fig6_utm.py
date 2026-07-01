"""Regenerate Figure 6 from the UTM recall run so it matches Table 1."""
from __future__ import annotations
import os
import sys, math
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from yazoo.labeling.dem_usgs3dep import Usgs3depImageServerSource
from yazoo.labeling.dem_sources import WindowRequest
from demo_terrain_query import classify_geomorphon_simple, make_hillshade
from scripts.earthwork_query import detect_earthworks

UTM="EPSG:26915"; RES=1.0; HALF=150
GOLD = os.environ.get("EARTHWORK_GOLD_LIST", "data/reference/located_mounds.csv")  # RESTRICTED, not shipped
res=pd.read_csv("data/refind_utm/refind_utm.csv")
res["dist_m"]=pd.to_numeric(res["dist_m"],errors="coerce")
gold=pd.read_csv(GOLD).set_index("mound_id")
rec=res[(res.status=="ok")&(res.dist_m<=30)].sort_values("dist_m")
# one mound per distinct site, spread across the offset range (avoids a gallery
# dominated by a single mound group like 22-N-13)
rec=rec.assign(site=rec.mound_id.str.replace(r"_m\d+$","",regex=True))
uniq=rec.drop_duplicates("site").sort_values("dist_m").reset_index(drop=True)
import numpy as _np
idx=_np.linspace(0,len(uniq)-1,8).round().astype(int)
picks=list(dict.fromkeys(uniq.loc[idx,"mound_id"]))[:8]

def fetch(cx,cy):
    src=Usgs3depImageServerSource()
    req=WindowRequest(center_x=cx,center_y=cy,utm_crs=UTM,resolution_m=RES,size_px=2*HALF)
    return src.fetch_window(req)

fig=plt.figure(figsize=(13,7.2)); gs=fig.add_gridspec(2,5)
for i,mid in enumerate(picks):
    ax=fig.add_subplot(gs[i//4,i%4])
    cx=float(gold.loc[mid,"utm15n_easting_m"]); cy=float(gold.loc[mid,"utm15n_northing_m"])
    dem=fetch(cx,cy); dem=np.where(np.isfinite(dem),dem,np.nanmedian(dem[np.isfinite(dem)])).astype("float32")
    geo=classify_geomorphon_simple(dem); cands=detect_earthworks(geo,dem,"Find pre-European earthwork mounds")
    gpx=HALF; gpy=HALF  # mound at tile centre
    best=None; bd=1e9
    for c in cands:
        d=math.hypot(c["x"]-gpx,c["y"]-gpy)
        if d<bd: bd=d; best=c
    ax.imshow(make_hillshade(dem),cmap="gray",origin="lower")
    # note: make_hillshade origin lower -> flip y for plotting markers
    H=dem.shape[0]
    ax.plot(gpx,H-gpy,marker="*",ms=15,mfc="cyan",mec="k",mew=1.0)
    d_m=bd*RES
    if best is not None and d_m<=30:
        ax.add_patch(plt.Circle((best["x"],H-best["y"]),12,fill=False,ec="lime",lw=2))
    ax.set_title(f"{mid}  {d_m:.0f} m",fontsize=9,color="green"); ax.set_xticks([]); ax.set_yticks([])
# histogram + summary from UTM offsets
d=rec["dist_m"]
axh=fig.add_subplot(gs[0,4]); axh.hist(d,bins=np.arange(0,32,3),color="#2e7d32",edgecolor="k")
axh.axvline(d.median(),color="red",ls="--",lw=1.5)
axh.set_title("Offset, all 31 recovered",fontsize=8)
axh.set_xlabel("distance (m)",fontsize=8); axh.set_ylabel("mounds",fontsize=8); axh.tick_params(labelsize=7)
axt=fig.add_subplot(gs[1,4]); axt.axis("off")
# no in-image caption text: captions belong to the manuscript, and the summary
# statistics live in Table 1
fig.tight_layout()
fig.savefig("docs/figures/fig_refind_known_mounds.png",dpi=110,bbox_inches="tight")
print("regenerated Figure 6 from UTM run; picks:",picks)
