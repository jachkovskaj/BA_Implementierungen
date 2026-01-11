import matplotlib.pyplot as plt
import numpy as np

# Design Space Grenzen
x_min, x_max = -350, 350
y_min, y_max = -150, 150
x_range = 700
y_range = 300
ratio = x_range / y_range

# Weißer Rand
margin = 30
frame_xmin, frame_xmax = x_min - margin, x_max + margin
frame_ymin, frame_ymax = y_min - margin, y_max + margin
frame_width = frame_xmax - frame_xmin
frame_height = frame_ymax - frame_ymin

# Zentrum
center = np.array([-180, 30])

# Vektorlängen (z. B. Beispielwerte – können ersetzt werden)
vector_lengths = [120, 60, 30, 80, 100, 160, 40, 100]

# Berechne gleichmäßig verteilte Winkel mit min. 45° Abstand
n_vectors = len(vector_lengths)
angles = np.linspace(0, 2*np.pi, n_vectors, endpoint=False)

# Endpunkte berechnen
polygon_points = []
for L, a in zip(vector_lengths, angles):
    endpoint = center + L * np.array([np.cos(a), np.sin(a)])
    polygon_points.append(endpoint)
polygon_points = np.array(polygon_points)

# Plot vorbereiten
fig, ax = plt.subplots(figsize=(10, 10/ratio))

# Weißer Hintergrundrahmen
ax.add_patch(plt.Rectangle((frame_xmin, frame_ymin),
                           frame_width, frame_height,
                           facecolor="white", zorder=0))

# Hintergrund Design-Space-Rechteck
ax.add_patch(plt.Rectangle((x_min, y_min),
                           x_range, y_range,
                           facecolor="steelblue", alpha=0.3,
                           edgecolor="none", zorder=1))

# Orangene Linie vom Ursprung zum Zentrum
ax.plot([0, center[0]], [0, center[1]], color="orange", linewidth=2, label="Positionsvektor", zorder=2)

# Polygonfläche
ax.fill(polygon_points[:, 0], polygon_points[:, 1],
        color="yellowgreen", alpha=0.4, label="Geometrieparameter", zorder=2)

# Geometric Vectors
for point in polygon_points:
    ax.arrow(center[0], center[1],
             point[0] - center[0], point[1] - center[1],
             head_width=10, head_length=15, color="yellowgreen", length_includes_head=True, zorder=3)

# Polygonränder
ax.plot(np.append(polygon_points[:, 0], polygon_points[0, 0]),
        np.append(polygon_points[:, 1], polygon_points[0, 1]),
        color="green", zorder=3)

# Achsen und Beschriftungen
ax.set_xlim(frame_xmin, frame_xmax)
ax.set_ylim(frame_ymin, frame_ymax)
ax.set_xticks([-350, -250, -150, -50, 0, 50, 150, 250, 350])
ax.set_xlabel("Länge (mm)")
ax.set_ylabel("Breite (mm)")
ax.set_title("Designvektoren (45°)")
ax.legend()

plt.savefig("design_space_vectors.png", dpi=300, bbox_inches="tight")
plt.show()