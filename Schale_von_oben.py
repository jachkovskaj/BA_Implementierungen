import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from adjustText import adjust_text


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
            # ab hier würden Spannungen kommen – abbrechen
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
                z = float(parts[4])  # für 2D nicht benötigt
                coord_data.append([node_id, x, y, z])
            except (IndexError, ValueError):
                continue

df = pd.DataFrame(coord_data, columns=["NodeID", "X", "Y", "Z"])
print("Knoten geladen:", len(df))
if df.empty:
    raise ValueError("Keine Koordinaten gefunden. Prüfe, ob 'Orig. Coords' im RPT vorhanden ist.")

# ===== 2) XY-Koordinaten =====
x = df["X"].to_numpy()
y = df["Y"].to_numpy()

# Optional: Spiegelungen/Drehungen (bei Bedarf aktivieren)
# y = -y                      # nur Y spiegeln
# x, y = -x, -y               # 180° um Z-Achse

# ===== 3) Triangulation in XY =====
triang = tri.Triangulation(x, y)

# ===== 4) Freie Kanten (Rand/Lochkonturen) bestimmen =====
edge_count = {}
for t in triang.triangles:
    for (i, j) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]:
        e = tuple(sorted((int(i), int(j))))
        edge_count[e] = edge_count.get(e, 0) + 1
boundary_edges = [e for e, c in edge_count.items() if c == 1]

# ===== 5) Hervorhebung von Knoten in 4 Gruppen (Versuchspläne) =====

show_labels = False       # True => NodeIDs als Text anzeigen
hl_size = 80              # Markergröße in pt^2

"""# Modell 1
marker_styles = {
    "Halton":  "o",   # Kreis
    "LHS":     "s",   # Quadrat
    "Sobol":   "^",   # Dreieck
    "Taguchi": "X",   # X-Stern
}

colors = {
    "Halton":  "tab:blue",
    "LHS":     "tab:orange",
    "Sobol":   "tab:green",
    "Taguchi": "tab:red",
}

highlight_nodes_Halton  = [1884, 3290, 377]
highlight_nodes_LHS     = [1557, 54, 2647]
highlight_nodes_Sobol   = [1744, 54, 2930]
highlight_nodes_Taguchi = [1600, 3284, 160]

groups = [
    {"label": "Halton", "nodes": highlight_nodes_Halton},
    {"label": "LHS", "nodes": highlight_nodes_LHS},
    {"label": "Sobol", "nodes": highlight_nodes_Sobol},
    {"label": "Taguchi", "nodes": highlight_nodes_Taguchi},
]"""

# (Falls du Modell 2 weiter nutzen willst, kannst du das separat lassen)
# Modell 2
colors = {
    "Knoten mit 0en": 'tab:red',
    "Knoten ohne 0en": 'tab:green',
}

highlight_nodes_model2_ohne_0en = [563, 1309, 3022, 3174, 2011, 1296]
highlight_nodes_model2_mit_0en = [2055, 2775, 357, 233]

groups = [
    {"label": "Knoten mit 0en", "nodes": highlight_nodes_model2_mit_0en},
    {"label": "Knoten ohne 0en", "nodes": highlight_nodes_model2_ohne_0en},
]

# ===== 6) Plot (2D) =====
fig = plt.figure(figsize=(12, 10), dpi=300, facecolor="none")
ax = fig.add_subplot(111, facecolor="none")

# (a) Netz dezent
ax.triplot(triang, color="0.75", linewidth=0.6, alpha=0.9)

# (b) Rand-/Lochkonturen hervorheben
for (i, j) in boundary_edges:
    ax.plot([x[i], x[j]], [y[i], y[j]], linewidth=1.0, color="0.2")

# (c) Hervorgehobene Knoten für jede Gruppe (Modell 1)
idx = df.set_index("NodeID")

for g in groups:
    nodes = [nid for nid in g["nodes"] if nid in idx.index]
    if not nodes:
        continue

    x_h = [idx.at[nid, "X"] for nid in nodes]
    y_h = [idx.at[nid, "Y"] for nid in nodes]

    ax.scatter(
        x_h, y_h,
        s=hl_size,
        c=colors[g["label"]],
        marker="o",
        zorder=3,
        label=g["label"]
    )

    if show_labels:
        for Xi, Yi, nid in zip(x_h, y_h, nodes):
            ax.text(
                Xi, Yi,
                f"{nid}",
                fontsize=8,
                color=g["color"],
                ha="left", va="bottom",
                zorder=4
            )

"""idx = df.set_index("NodeID")
hl_size = 100     # Punktgröße
show_labels = True   # Nummerierung innerhalb der Gruppe

for g in groups:
    label = g["label"]
    nodes = g["nodes"]

    # nur vorhandene Knoten übernehmen
    valid_nodes = [nid for nid in nodes if nid in idx.index]

    x_g = [idx.at[nid, "X"] for nid in valid_nodes]
    y_g = [idx.at[nid, "Y"] for nid in valid_nodes]

    # Punkte zeichnen
    ax.scatter(
        x_g, y_g,
        s=hl_size,
        c=colors[label],
        marker=marker_styles[label],
        edgecolor="black",
        linewidth=0.6,
        zorder=10,
        label=label
    )

    # Punkte innerhalb des Versuchsplans unterscheiden: 1, 2, 3
    if show_labels:
        for i, (Xi, Yi) in enumerate(zip(x_g, y_g), start=1):
            ax.text(
                Xi, Yi,
                str(i),            # Index: 1, 2, 3
                fontsize=8,
                color="black",
                ha="center", va="center",
                zorder=11,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.2)
            )"""

# Achsen unsichtbar & Seitenverhältnis
ax.set_aspect("equal", adjustable="box")
ax.set_axis_off()

# Legende der Versuchspläne
ax.legend(loc="lower right", frameon=True)

plt.tight_layout()

# Hochauflösend speichern (PNG transparent + SVG vektor)
plt.savefig("bauteil_2d_model_2.png", dpi=600, bbox_inches="tight", transparent=True)

plt.show()