import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FONT = "Arial"
plt.rcParams["font.family"] = FONT

NEUTRAL = dict(fc="#FFFFFF", ec="#212121")
BLUE = dict(fc="#E8F0FE", ec="#4285F4")
TEAL = dict(fc="#E0F2F1", ec="#0097A7")
PURPLE = dict(fc="#F1E9FB", ec="#8E24AA")
CORAL = dict(fc="#FFEDE8", ec="#E64A19")

PARAM_BOXES = {
    "tf": (0.3, 5.7, 2.6, 0.62, "TF parameters", "Sets the velocity-luminosity relation", NEUTRAL),
    "color": (3.2, 5.7, 3.0, 0.62, "Color parameters", "How each color tracks velocity\nand shifts luminosity", PURPLE),
    "kcorr": (6.5, 5.7, 3.0, 0.62, "K-correction", "Redshift-dependent shift per band", CORAL),
}

NODES = {
    "xtrue": (1.6, 4.15, 1.5, 0.55, "x_true", "latent true velocity", NEUTRAL),
    "yTF": (4.3, 3.15, 1.6, 0.55, "y_TF", "latent TF magnitude", NEUTRAL),
    "ec": (6.2, 3.15, 1.4, 0.55, "ε_c (r−z)", "latent color, r−z", BLUE),
    "eg": (8.1, 3.15, 1.4, 0.55, "ε_g (g−r)", "latent color, g−r", TEAL),
    "xhat": (1.6, 1.4, 1.4, 0.55, "x̂", "observed velocity", NEUTRAL),
    "yhat": (3.5, 1.4, 1.4, 0.55, "ŷ (r-band)", "observed r-mag", NEUTRAL),
    "zhat": (6.2, 1.4, 1.4, 0.55, "ẑ (z-band)", "observed z-mag", BLUE),
    "ghat": (8.1, 1.4, 1.4, 0.55, "ĝ (g-band)", "observed g-mag", TEAL),
}

ALL_BOXES = {**PARAM_BOXES, **NODES}
WHITE_KEYS = {key for key, spec in ALL_BOXES.items() if spec[-1] is NEUTRAL}

SOLID_ARROWS = [
    ("tf", "yTF"), ("color", "ec"), ("color", "eg"),
    ("xtrue", "xhat"), ("yTF", "xtrue"), ("xtrue", "ec"), ("xtrue", "eg"),
    ("yTF", "yhat"), ("yTF", "zhat"), ("yTF", "ghat"),
    ("ec", "yhat"), ("ec", "zhat"), ("eg", "yhat"), ("eg", "ghat"),
]

DASHED_ARROWS = [("kcorr", "yhat", 0.25), ("kcorr", "zhat", 0.15), ("kcorr", "ghat", 0.0)]

FADE_ALPHA = 0.2


def draw_box(ax, x, y, w, h, title, sub, style, centers, key, alpha=1.0):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                          linewidth=1.1, facecolor=style["fc"], edgecolor=style["ec"],
                          alpha=alpha, zorder=3)
    ax.add_patch(box)
    cx, cy = x + w / 2, y + h / 2
    centers[key] = (cx, cy, x, y, w, h)
    n_lines = sub.count("\n") + 1
    title_y = cy + (0.16 if n_lines == 1 else 0.20)
    ax.text(cx, title_y, title, ha="center", va="center", fontsize=11.5, fontweight="bold",
            color="#212121", alpha=alpha, zorder=4)
    for i, line in enumerate(sub.split("\n")):
        sub_y = cy - 0.14 + (0 if n_lines == 1 else (0.05 - i * 0.19))
        ax.text(cx, sub_y, line, ha="center", va="center", fontsize=8.0, color="#595959",
                alpha=alpha, zorder=4)


def render(mode, out_path):
    # mode: "full" (everything solid), "white" (white chain solid, rest faded),
    # "non_white" (rest solid, white chain faded) — the reverse of "white".
    def node_alpha(key):
        if mode == "full":
            return 1.0
        is_white = key in WHITE_KEYS
        highlight = is_white if mode == "white" else not is_white
        return 1.0 if highlight else FADE_ALPHA

    def edge_alpha(a, b):
        if mode == "full":
            return 1.0
        both_white = a in WHITE_KEYS and b in WHITE_KEYS
        highlight = both_white if mode == "white" else not both_white
        return 1.0 if highlight else FADE_ALPHA

    fig, ax = plt.subplots(figsize=(10, 6.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.6, 6.6)
    ax.axis("off")

    centers = {}
    for key, (x, y, w, h, title, sub, style) in PARAM_BOXES.items():
        draw_box(ax, x, y, w, h, title, sub, style, centers, key, alpha=node_alpha(key))
    for key, (x, y, w, h, title, sub, style) in NODES.items():
        draw_box(ax, x, y, w, h, title, sub, style, centers, key, alpha=node_alpha(key))

    for a, b in SOLID_ARROWS:
        (x1, y1, *_), (x2, y2, *_) = centers[a], centers[b]
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                 linewidth=0.9, color="#78909C", alpha=edge_alpha(a, b),
                                 shrinkA=22, shrinkB=22, zorder=2,
                                 connectionstyle="arc3,rad=0.05")
        ax.add_patch(arrow)

    for a, b, rad in DASHED_ARROWS:
        (x1, y1, *_), (x2, y2, *_) = centers[a], centers[b]
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                 linewidth=0.9, color="#78909C", linestyle=(0, (5, 3)), alpha=edge_alpha(a, b),
                                 shrinkA=22, shrinkB=22, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}")
        ax.add_patch(arrow)

    plate = FancyBboxPatch((0.6, 0.9), 9.0, 4.35, boxstyle="round,pad=0.02,rounding_size=0.08",
                            linewidth=1.0, linestyle=(0, (5, 3)), facecolor="none", edgecolor="#78909C", zorder=1)
    ax.add_patch(plate)
    ax.text(0.85, 5.02, "galaxies, n = 1…N", fontsize=9, color="#595959", ha="left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


HERE = os.path.dirname(os.path.abspath(__file__))
render(mode="full", out_path=os.path.join(HERE, "dag_v2.png"))
render(mode="white", out_path=os.path.join(HERE, "dag_v2_highlight.png"))
render(mode="non_white", out_path=os.path.join(HERE, "dag_v2_highlight_inverted.png"))
print("done")
