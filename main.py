import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon
import math
import numpy
# --- USER PARAMETERS: set triangle by two angles and side length c = AB ---
angleA_deg = 60    # angle at vertex A in degrees
angleB_deg = 60    # angle at vertex B in degrees
side_c = 1.0       # length of side AB (between A and B)
interval_ms = 30   # animation frame interval in milliseconds

# Compute third angle
angleC_deg = 180.0 - angleA_deg - angleB_deg
if angleC_deg <= 0:
    raise ValueError(f"Angles must sum to < 180°, got angleC = {angleC_deg}")

# Convert to radians
A_rad = np.deg2rad(angleA_deg)
B_rad = np.deg2rad(angleB_deg)
C_rad = np.deg2rad(angleC_deg)

# Law of sines to get other sides
circum = side_c / np.sin(C_rad)
side_a = circum * np.sin(A_rad)  # BC
side_b = circum * np.sin(B_rad)  # CA

# Define triangle vertices
A = np.array([0.0, 0.0])
B = np.array([side_c, 0.0])
C = np.array([side_b * np.cos(A_rad), side_b * np.sin(A_rad)])
verts = np.vstack([A, B, C])

# Compute incenter I and inradius r_in
a, b, c = side_a, side_b, side_c
P = a + b + c
I = (a*A + b*B + c*C) / P
# Triangle area via scalar cross
t_area = 0.5 * abs((B-A)[0]*(C-A)[1] - (B-A)[1]*(C-A)[0])
r_in = 2 * t_area / P

# Precompute inward normals for circle solver
sides = [(A, B), (B, C), (C, A)]
params = []
for P1, P2 in sides:
    d = (P2 - P1) / np.linalg.norm(P2 - P1)
    N = np.array([-d[1], d[0]])
    if np.dot(I - (P1 + P2) / 2, N) < 0:
        N = -N
    params.append(N)

# Compute midpoints & axis vectors for angle bisectors
mid_A = (B + C) / 2; axis_A = A - mid_A
mid_B = (C + A) / 2; axis_B = B - mid_B
mid_C = (A + B) / 2; axis_C = C - mid_C
len2_A = axis_A.dot(axis_A)
len2_B = axis_B.dot(axis_B)
len2_C = axis_C.dot(axis_C)

# Solver: find center & radius along ray angle theta
def find_center_and_radius(theta):
    u = np.array([math.cos(theta), math.sin(theta)])
    best_t = np.inf
    for N in params:
        ni = N.dot(u)
        Aq = 1 - ni**2
        if abs(Aq) < 1e-8:
            continue
        Bq = -2 * r_in * ni
        Cq = -r_in**2
        disc = Bq**2 - 4 * Aq * Cq
        if disc < 0:
            continue
        sd = math.sqrt(disc)
        for t in [(-Bq + sd) / (2 * Aq), (-Bq - sd) / (2 * Aq)]:
            if 1e-6 < t < best_t:
                best_t = t
    center = I + best_t * u
    radius = math.dist(center, I)
    return center, radius

# Precompute samples
thetas = np.linspace(0, 2 * np.pi, 600, endpoint=False)
centers = np.array([find_center_and_radius(th)[0] for th in thetas])
radii = np.array([find_center_and_radius(th)[1] for th in thetas])
# Bisector-axis coordinates
sA = (centers - mid_A) @ axis_A / len2_A
sB = (centers - mid_B) @ axis_B / len2_B
sC = (centers - mid_C) @ axis_C / len2_C
# Circle area array
circle_area = math.pi * radii**2

# Compute functions:
s_max = np.maximum.reduce([sA, sB, sC])
s_min = np.minimum.reduce([sA, sB, sC])
s_med = (sA + sB + sC) - s_max - s_min  # the middle projection
f1 = s_max * s_min * (math.sqrt(3)/2)
# f2: product of max, min, and twice the medium projection
# Setup first figure: original animation
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
pad = max(side_a, side_b, side_c) * 0.3
# Left: triangle & moving circle
ax1.set_aspect('equal', 'box')
ax1.set_xlim(verts[:,0].min()-pad, verts[:,0].max()+pad)
ax1.set_ylim(verts[:,1].min()-pad, verts[:,1].max()+pad)
ax1.add_patch(Polygon(verts, fill=False, lw=2))
ax1.plot(*I, 'ro', label='Incenter')
circ = Circle(centers[0], radii[0], fill=False, color='tab:orange', lw=2)
ax1.add_patch(circ)
for mid, axis, name in [(mid_A,axis_A,'A-axis'),(mid_B,axis_B,'B-axis'),(mid_C,axis_C,'C-axis')]:
    ax1.plot([mid[0],mid[0]+axis[0]],[mid[1],mid[1]+axis[1]],'--',lw=1,color='gray')
    ax1.text(mid[0]+axis[0]*1.05, mid[1]+axis[1]*1.05, name, color='gray')
ax1.text(0.02,0.02, f"Triangle Area = {t_area:.5f}", transform=ax1.transAxes, fontsize=10, color='blue')
area_text = ax1.text(0.02,0.07, f"Circle Area = {circle_area[0]:.5f}", transform=ax1.transAxes, fontsize=10, color='green')
coord_text = ax1.text(0.02,0.12, f"sA = {sA[0]:.5f}\nsB = {sB[0]:.5f}\nsC = {sC[0]:.5f}", transform=ax1.transAxes, fontsize=10, color='purple', verticalalignment='top')
ax1.set_title(f"Triangle & Moving Circle (A={angleA_deg}°,B={angleB_deg}°)")
ax1.legend()
# Middle: bisector-axis coords
frames = np.arange(len(thetas))
lineA, = ax2.plot(frames, sA, label='x (A-axis)')
lineB, = ax2.plot(frames, sB, label='y (B-axis)')
lineC, = ax2.plot(frames, sC, label='z (C-axis)')
markerA, = ax2.plot([],[], 'o', color=lineA.get_color())
markerB, = ax2.plot([],[], 'o', color=lineB.get_color())
markerC, = ax2.plot([],[], 'o', color=lineC.get_color())
ax2.set_xlim(0,len(thetas)); ax2.set_ylim(0,1)
ax2.set_xlabel('Frame'); ax2.set_ylabel('Axis Coordinate')
ax2.set_title('Bisector-Axis Coordinates'); ax2.legend()
# Right: circle area evolution
lineArea, = ax3.plot(frames, circle_area, label='Circle Area', color='green')
markerArea, = ax3.plot([],[], 'o', color='green')
ax3.set_xlim(0,len(thetas)); ax3.set_ylim(circle_area.min()*0.9, circle_area.max()*1.1)
ax3.set_xlabel('Frame'); ax3.set_ylabel('Area')
ax3.set_title('Circle Area Evolution'); ax3.legend()
# Pause state management
paused = False
pause_text = fig.text(0.5, 0.95, "", fontsize=16, ha='center', va='top', color='red', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'), visible=False)
# Animation update
def update(frame):
    circ.center = centers[frame]
    circ.radius = radii[frame]
    markerA.set_data([frame], [sA[frame]])
    markerB.set_data([frame], [sB[frame]])
    markerC.set_data([frame], [sC[frame]])
    markerArea.set_data([frame], [circle_area[frame]])
    area_text.set_text(f"Circle Area = {circle_area[frame]:.5f}")
    coord_text.set_text(f"sA = {sA[frame]:.5f}\nsB = {sB[frame]:.5f}\nsC = {sC[frame]:.5f}")
    return circ, markerA, markerB, markerC, markerArea, area_text, coord_text, pause_text
# Toggle pause function
def toggle_pause(event):
    global paused
    if event.key == ' ':
        if paused:
            anim.resume()
            pause_text.set_visible(False)
            paused = False
        else:
            anim.pause()
            pause_text.set_text("PAUSED (Press SPACE to resume)")
            pause_text.set_visible(True)
            paused = True
        fig.canvas.draw_idle()
# Create animation
anim = FuncAnimation(fig, update, frames=len(thetas), interval=interval_ms, blit=False, repeat=True)
FuncAnimation.pause = lambda self: self.event_source.stop()
FuncAnimation.resume = lambda self: self.event_source.start()
fig.canvas.mpl_connect('key_press_event', toggle_pause)
plt.tight_layout()

# --- SECOND FIGURE: plotting f1, f2 & circle area ---
fig2, ax4 = plt.subplots(figsize=(6,4))
ax4.plot(frames, f1, label='f1 = max·min·√3/2', color='tab:blue')
ax4.plot(frames, circle_area, label='Circle Area', color='tab:green')
ax4.set_xlabel('Frame')
ax4.set_ylabel('Value')
ax4.set_title('Functions vs Circle Area')
ax4.legend()

plt.show()
