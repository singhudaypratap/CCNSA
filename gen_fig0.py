"""
CCNSA architecture diagram -- monochrome (black/white), PowerPoint-style
rebuild.

REBUILT (post-review: "figure 1 tex are going out of the box... make it
like it is made in power point white black figure only"). The previous
color version went through several rounds of hand-tuned box/font sizes
and kept re-breaking (text overflowing box borders) every time the print
size was adjusted, because box widths and font sizes were independently
guessed rather than derived from each other. This version fixes that
class of bug structurally instead of by further guessing:

  1. Every box's width is computed from the ACTUAL measured pixel width
     of its longest line of text (via a throwaway Matplotlib renderer
     pass, `_measure()` below), not a hand-picked constant. If a line is
     too long, the box grows to fit it -- it is not possible for text to
     overflow horizontally by construction.
  2. Every box's height is computed from the actual number of lines it
     contains times a fixed line-height, plus fixed title/padding space.
     Same guarantee vertically.
  3. Column widths and row heights are then the MAX over every box in
     that column/row, so the grid stays aligned while every box is still
     individually big enough for its own content.
  4. Colour is dropped entirely: white fill, black 1.5pt border, black
     text, black arrows -- a plain schematic look instead of the
     pastel-coded version, per the request.

This trades a small amount of visual density (text is plainer, no colour
coding by branch) for a hard guarantee against the recurring overflow
bug.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": "#000000",
})

FIG_W_IN, FIG_H_IN, DPI = 12.5, 11.5, 280

# ---- text measurement helpers -------------------------------------------
_measure_fig = plt.figure(figsize=(1, 1))
_renderer = _measure_fig.canvas.get_renderer()

def _measure(s, fontsize, weight="normal", style="normal"):
    """Return (width_px, height_px) of `s` rendered at `fontsize` pt, at
    this figure's DPI -- a real measurement, not an estimate."""
    t = _measure_fig.text(0, 0, s, fontsize=fontsize, fontweight=weight, fontstyle=style)
    bbox = t.get_window_extent(renderer=_renderer)
    t.remove()
    return bbox.width, bbox.height

PX_PER_IN = _measure_fig.dpi  # matplotlib text measurement is dpi-dependent;
# convert through this figure's own dpi, then re-express in the OUTPUT
# figure's data-unit system (defined below) so box sizes are correct
# regardless of the two figures having different dpi.

TITLE_FS, BODY_FS, ITAL_FS = 15, 12.5, 11
LINE_H = 1.55  # line-height multiplier on fontsize, in points
PAD_X = 16  # left AND right inner margin, in points -- text is drawn starting
# at box_x + PAD_X, so the box width must include 2*PAD_X (both sides), not
# just the raw measured text width. Missing this was the exact cause of the
# v1 overflow (box-1 title bled into box-2, box-9's last line touched its
# own right border): box_size() previously returned the bare text width and
# the +PAD_X left offset at draw time silently ate into the right margin
# with nothing added back.

def box_size(title, lines, ital_lines=()):
    """Compute (width, height) in POINTS needed for a box with this
    title + body lines + trailing italic caption lines, from real text
    measurements, INCLUDING left+right padding. Returns points (1/72 in),
    independent of any figure's dpi -- points-to-data-unit conversion
    happens once, globally, below."""
    all_widths = [_measure(title, TITLE_FS, weight="bold")[0]]
    for l in lines:
        all_widths.append(_measure(l, BODY_FS)[0])
    for l in ital_lines:
        all_widths.append(_measure(l, ITAL_FS, style="italic")[0])
    w_px = max(all_widths)
    w_pt = w_px / PX_PER_IN * 72 + 2 * PAD_X
    n_lines = 1 + len(lines) + len(ital_lines)
    h_pt = (TITLE_FS * 1.4) + n_lines * (BODY_FS * LINE_H) + 18  # +18pt title gap
    return w_pt, h_pt

# ---- box content (kept factually identical to the manuscript; wording
#      shortened so it reads like a slide, not a paragraph) --------------
BOXES = {
    1: dict(title="1. Network Trauma Environment",
            lines=["20 nodes - 100 users - horizon T=100",
                   "Trauma: physical / cyber / congestion",
                   "Cyber trauma uses UNSW-NB15 signatures"]),
    2: dict(title="2. Observed State",
            lines=[r"$s_t \in \mathbb{R}^{32}$",
                   "Throughput, loss, latency, severity,",
                   "queue depth, topology"]),
    3: dict(title="3. Reward",
            lines=["Weighted satisfaction + emergency",
                   "satisfaction - power cost (Eq. 9)",
                   "Feeds critic update (Eq. 3)"]),
    4: dict(title="4. Causal Branch",
            lines=["PC algorithm (Fisher-Z test)",
                   "Learns DAG over state features",
                   r"$\varphi_C(s)=(A_G \odot M_C)\varphi(s)$"],
            ital=["Fixed offline mask, not interventional"]),
    5: dict(title="5. Symbolic Branch (policy rules)",
            lines=["8 differentiable rules " + r"$R_1 \ldots R_8$",
                   "Softmax-gated combination"],
            ital=["Distinct from the 6 audit rules (Track B)"]),
    6: dict(title="6. Neural Branch (twin-critic)",
            lines=[r"Actor: MLP $32{\to}256{\to}256{\to}4$",
                   "Twin critics + conservative penalty",
                   "BC warm-start from symbolic policy"]),
    7: dict(title="7. Hybrid Action Fusion",
            lines=["Concatenates causal + symbolic reps",
                   "Smooth-clip limits per-step change"],
            ital=["Fusion reconciles branch disagreement"]),
    8: dict(title="8. Action",
            lines=["Reroute, patch priority, throttle,",
                   "restoration order"],
            ital=["Applied to environment (dotted, left)"]),
    9: dict(title="9. Faithfulness Audit (offline)",
            lines=["SHAP / LIME / Integrated Gradients",
                   "agreement + deletion-insertion test"],
            ital=["Every K=10 episodes; no gradient to actor"]),
}

for k, d in BOXES.items():
    d["ital"] = d.get("ital", [])
    w, h = box_size(d["title"], d["lines"], d["ital"])
    d["w"], d["h"] = w, h

# ---- grid layout: 3 columns x 4 rows, column width = max box width in
#      that column, row height = max box height in that row ------------
GRID = {
    (0, 0): 1, (0, 1): 2, (0, 2): 3,
    (1, 0): 4, (1, 1): 5, (1, 2): 6,
}
col_w = [0, 0, 0]
row_h = [0, 0]
for (r, c), idx in GRID.items():
    col_w[c] = max(col_w[c], BOXES[idx]["w"])
    row_h[r] = max(row_h[r], BOXES[idx]["h"])

COL_GAP, ROW_GAP = 34, 46
row7_h = BOXES[7]["h"]
row89_w = [BOXES[8]["w"], BOXES[9]["w"]]
row89_h = max(BOXES[8]["h"], BOXES[9]["h"])

total_w_pt = sum(col_w) + 2 * COL_GAP
total_h_pt = (row_h[0] + row_h[1] + row7_h + row89_h) + 3 * ROW_GAP + 90  # +90 for title block
strip_h_pt = 108

total_h_pt += strip_h_pt + ROW_GAP

# points -> inches for the actual output figure
IN_PER_PT = 1 / 72
fig_w_in = total_w_pt * IN_PER_PT + 0.6
fig_h_in = total_h_pt * IN_PER_PT + 0.6
plt.close(_measure_fig)

fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=DPI)
# data units == points for this figure, 1:1, with a small margin
MARGIN = 20
ax.set_xlim(0, total_w_pt + 2 * MARGIN)
ax.set_ylim(0, total_h_pt + 2 * MARGIN)
ax.axis("off")
ax.invert_yaxis()

def box(x, y, w, h, num=None):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=4,rounding_size=6",
                        linewidth=1.6, edgecolor="#000000", facecolor="#ffffff", zorder=2)
    ax.add_patch(b)

def draw_box_content(x, y, w, h, d):
    box(x, y, w, h)
    tx = x + PAD_X
    ty = y + TITLE_FS * 1.1
    ax.text(tx, ty, d["title"], fontsize=TITLE_FS, fontweight="bold", va="top", zorder=4)
    ty += TITLE_FS * 0.9
    for l in d["lines"]:
        ty += BODY_FS * LINE_H
        ax.text(tx, ty, l, fontsize=BODY_FS, va="top", zorder=4)
    for l in d["ital"]:
        ty += BODY_FS * LINE_H
        ax.text(tx, ty, l, fontsize=ITAL_FS, style="italic", va="top", zorder=4)

def arrow(p0, p1, rad=0.0, ls="-"):
    a = FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                         arrowstyle="-|>", mutation_scale=13, linewidth=1.3,
                         color="#000000", linestyle=ls, zorder=1)
    ax.add_patch(a)

# ---- title block ----
cx = MARGIN + total_w_pt / 2
ax.text(cx, MARGIN + 20, "CCNSA - Conservative-Critic Neuro-Symbolic Agent",
        ha="center", va="top", fontsize=20, fontweight="bold")
ax.text(cx, MARGIN + 52,
        "Environment -> State -> Causal + Symbolic + Neural fusion -> Action -> Faithfulness audit (offline)",
        ha="center", va="top", fontsize=12, style="italic")

y0 = MARGIN + 90
x_positions = [MARGIN, MARGIN + col_w[0] + COL_GAP, MARGIN + col_w[0] + col_w[1] + 2 * COL_GAP]

# Row 0 (boxes 1,2,3)
row0_boxes = {}
for c in range(3):
    idx = GRID[(0, c)]
    d = BOXES[idx]
    x = x_positions[c]
    row0_boxes[idx] = (x, y0, col_w[c], row_h[0])
    draw_box_content(x, y0, col_w[c], row_h[0], d)

y1 = y0 + row_h[0] + ROW_GAP
row1_boxes = {}
for c in range(3):
    idx = GRID[(1, c)]
    d = BOXES[idx]
    x = x_positions[c]
    row1_boxes[idx] = (x, y1, col_w[c], row_h[1])
    draw_box_content(x, y1, col_w[c], row_h[1], d)

# Row for box 7 (spans full width, centered)
y2 = y1 + row_h[1] + ROW_GAP
box7_w = max(BOXES[7]["w"], col_w[0] + col_w[1] + col_w[2] + 2 * COL_GAP)
x7 = MARGIN
draw_box_content(x7, y2, box7_w, row7_h, BOXES[7])

# Row for boxes 8, 9
y3 = y2 + row7_h + ROW_GAP
box8_x = MARGIN + (col_w[0] - BOXES[8]["w"]) / 2 if col_w[0] > BOXES[8]["w"] else MARGIN
box8_x = x_positions[0]
box9_x = x_positions[1]
draw_box_content(box8_x, y3, BOXES[8]["w"], row89_h, BOXES[8])
draw_box_content(box9_x, y3, BOXES[9]["w"], row89_h, BOXES[9])

# ---- arrows: row 0 chain, row0->row1, row1->row7, row7->row8/9, feedback ----
b1 = row0_boxes[1]; b2 = row0_boxes[2]; b3 = row0_boxes[3]
arrow((b1[0] + b1[2], b1[1] + b1[3] * 0.35), (b2[0], b2[1] + b2[3] * 0.35))
arrow((b2[0] + b2[2], b2[1] + b2[3] * 0.35), (b3[0], b3[1] + b3[3] * 0.35))

b4 = row1_boxes[4]; b5 = row1_boxes[5]; b6 = row1_boxes[6]
arrow((b2[0] + b2[2] * 0.3, b2[1] + b2[3]), (b4[0] + b4[2] * 0.7, b4[1]), rad=-0.15)
arrow((b2[0] + b2[2] * 0.5, b2[1] + b2[3]), (b5[0] + b5[2] * 0.5, b5[1]))
arrow((b6[0] + b6[2], b6[1] + b6[3] * 0.15), (b3[0] + b3[2] * 0.2, b3[1] + b3[3]), rad=0.15, ls="--")
ax.text(b6[0] + b6[2] * 0.5, b6[1] - 18, r"$r_t \to$ critic update",
        fontsize=10, style="italic", ha="center", va="bottom")

arrow((b4[0] + b4[2] * 0.75, b4[1] + b4[3]), (x7 + box7_w * 0.22, y2), rad=-0.12)
arrow((b5[0] + b5[2] * 0.5, b5[1] + b5[3]), (x7 + box7_w * 0.5, y2))
arrow((b6[0] + b6[2] * 0.25, b6[1] + b6[3]), (x7 + box7_w * 0.78, y2), rad=0.12)

arrow((x7 + box7_w * 0.3, y2 + row7_h), (box8_x + BOXES[8]["w"] * 0.5, y3), rad=0.1)
arrow((box8_x + BOXES[8]["w"], y3 + row89_h * 0.5), (box9_x, y3 + row89_h * 0.5))

# feedback: Action -> Environment (dotted, left margin)
feedback_x = MARGIN - 14
ax.plot([box8_x, feedback_x, feedback_x, b1[0]],
        [y3 + row89_h * 0.3, y3 + row89_h * 0.3, b1[1] + b1[3] * 0.6, b1[1] + b1[3] * 0.6],
        color="#000000", linestyle=(0, (2, 2)), linewidth=1.1, zorder=1)
ax.annotate("", xy=(b1[0], b1[1] + b1[3] * 0.6), xytext=(b1[0] - 12, b1[1] + b1[3] * 0.6),
            arrowprops=dict(arrowstyle="-|>", color="#000000", lw=1.1))
ax.text(feedback_x - 12, (y3 + b1[1]) / 2, "apply action to environment", fontsize=9.5,
        style="italic", rotation=90, ha="center", va="center")

# ---- bottom strip: four propositions, monochrome ----
# Each column's width is measured from its own head+body text (same
# discipline as the boxes above), NOT an equal 1/4 split of strip_w --
# the equal-split version is what let P1's second line run into P2's
# column in the first monochrome draft.
PROP_FS_HEAD, PROP_FS_BODY = 12.5, 10.6
props = [
    ("P1 - Convergence", ["SAC convergence transfers to the", "symbolic-augmented MDP (Robbins-Monro)."]),
    ("P2 - Significance", [r"Bonferroni $\alpha/k$=0.00625 ($k$=8);", r"$d_{paired}$=2.42 vs CQL, $n$=10 seeds."]),
    ("P3 - Attribution bound", ["Under approx. independence of the", r"3 ranks, $\mathcal{F}\geq0.70$ w.p. $\geq0.95$."]),
    ("P4 - Complexity", ["181,760 FLOPs/pass; 0.99 ms mean", "latency - within the 5 ms URLLC budget."]),
]
PROP_GAP = 26
prop_widths = []
for head, body in props:
    wpx = max([_measure(head, PROP_FS_HEAD, weight="bold")[0]] +
              [_measure(l, PROP_FS_BODY)[0] for l in body])
    prop_widths.append(wpx / PX_PER_IN * 72)
strip_w = max(box7_w, sum(prop_widths) + (len(props) - 1) * PROP_GAP + 2 * PAD_X)

y4 = y3 + row89_h + ROW_GAP
ax.add_patch(FancyBboxPatch((MARGIN, y4), strip_w, strip_h_pt,
                             boxstyle="round,pad=4,rounding_size=5",
                             linewidth=1.2, edgecolor="#000000", facecolor="#ffffff", zorder=1))
px = MARGIN + PAD_X
for (head, body), w in zip(props, prop_widths):
    ax.text(px, y4 + 24, head, fontsize=PROP_FS_HEAD, fontweight="bold", va="top")
    py = y4 + 24
    for l in body:
        py += PROP_FS_BODY * LINE_H
        ax.text(px, py, l, fontsize=PROP_FS_BODY, va="top")
    px += w + PROP_GAP

plt.tight_layout()
plt.savefig("figures/fig0_architecture.png", dpi=DPI, facecolor="white")
print("saved")
print(f"canvas: {fig_w_in:.2f}in x {fig_h_in:.2f}in @ {DPI}dpi -> "
      f"{fig_w_in*DPI:.0f}x{fig_h_in*DPI:.0f}px")
