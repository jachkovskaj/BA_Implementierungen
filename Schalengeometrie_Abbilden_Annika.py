import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ===== 1) RPT einlesen & Koordinaten extrahieren =====
filename = "spannung.rpt"

coord_data = []
reading_coords = False

with open(filename, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if "Orig. Coords" in line:
            reading_coords = True
            continue
        if "S, Max. In-Plane Principal" in line:
            reading_coords = False
            break
        if reading_coords:
            if (not line.strip()) or ("-----" in line) or ("Part Instance" in line):
                continue
            parts = line.strip().split()
            try:
                node_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                coord_data.append([node_id, x, y, z])
            except (IndexError, ValueError):
                continue

df = pd.DataFrame(coord_data, columns=["NodeID", "X", "Y", "Z"])
print("Knoten geladen:", len(df))
if df.empty:
    raise ValueError("Keine Koordinaten gefunden. Prüfe, ob 'Orig. Coords' im RPT vorhanden ist.")

# ===== 2) Koordinaten vorbereiten (Z negativ spiegeln) =====
x = df["X"].to_numpy()
y = df["Y"].to_numpy()
z = -df["Z"].to_numpy()   # negative Z-Achse gemäß Wunsch

# ===== 3) Triangulation in XY =====
triang = tri.Triangulation(x, y)

# ===== 4) Freie Kanten (Rand/Lochkonturen) bestimmen =====
edge_count = {}
for t in triang.triangles:
    for (i, j) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]:
        e = tuple(sorted((int(i), int(j))))
        edge_count[e] = edge_count.get(e, 0) + 1
boundary_edges = [e for e, c in edge_count.items() if c == 1]

# ===== 5) Hilfsfunktion: gleiche Skalen =====
def set_axes_equal(ax):
    x_limits, y_limits, z_limits = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    x_range = abs(x_limits[1]-x_limits[0])
    y_range = abs(y_limits[1]-y_limits[0])
    z_range = abs(z_limits[1]-z_limits[0])
    max_range = max([x_range, y_range, z_range, 1e-9])
    x_mid, y_mid, z_mid = np.mean(x_limits), np.mean(y_limits), np.mean(z_limits)
    ax.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
    ax.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
    ax.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])

# ===== 6) Optionale Hervorhebung von Knoten =====
highlight_nodes = [563, 1309, 2055, 3022, 3174, 2775, 2011, 1296, 357, 233]         # <-- HIER deine NodeIDs eintragen, z.B. [2607, 2656, 3210]
show_labels = False             # True, wenn du die IDs neben die Punkte schreiben willst
hl_size = 50                    # Markergröße der hervorgehobenen Knoten

if highlight_nodes:
    mask_hl = df["NodeID"].isin(highlight_nodes)
    x_h, y_h, z_h = df.loc[mask_hl, ["X", "Y", "Z"]].to_numpy().T
    z_h = -z_h  # auch hier negative Z, damit es zur Szene passt
    ids_h = df.loc[mask_hl, "NodeID"].to_numpy()

# ===== 7) Plot =====
fig = plt.figure(figsize=(12, 10), dpi=300, facecolor="none")   # hochauflösend + transparenter Hintergrund
ax = fig.add_subplot(111, projection="3d", facecolor="none")

# Dezente transparente Fläche (optional) – bei Bedarf auskommentieren
#ax.plot_trisurf(
#    x, y, z, triangles=triang.triangles,
#    linewidth=0.3, edgecolor="0.6", antialiased=True,
#    shade=False, alpha=0.15
#)

# Netzlinien (Drahtgitter)
for tri_idx in triang.triangles:
    loop = np.r_[tri_idx, tri_idx[0]]
    ax.plot(x[loop], y[loop], z[loop], linewidth=0.5, color="0.5", alpha=0.7)

# Randkonturen deutlich
for (i, j) in boundary_edges:
    ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]],
            linewidth=1.0, color="0.1")

# Hervorgehobene Knoten
if highlight_nodes:
    ax.scatter(x_h, y_h, z_h, s=hl_size, c="red", marker="o", depthshade=True)
    if show_labels:
        for xi, yi, zi, nid in zip(x_h, y_h, z_h, ids_h):
            ax.text(xi, yi, zi, f"{nid}", fontsize=8, color="red", zorder=10)

# Ansicht/Kamera
ax.view_init(elev=20, azim=-60)
set_axes_equal(ax)

# Koordinatensystem ausblenden
ax.set_axis_off()

# ENGES Layout
plt.tight_layout()

# ===== 8) Hochauflösend speichern (transparent) – vor dem Show! =====
plt.savefig("bauteil_3d_model_2.png", dpi=600, bbox_inches="tight", transparent=True)
# Zusätzlich vektorisiert (falls für Dokumente gewünscht)
plt.savefig("bauteil_3d_model_2.svg", bbox_inches="tight", transparent=True)

# Anzeigen
plt.show()


