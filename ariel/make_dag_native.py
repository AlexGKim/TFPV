import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# DAG that follows the native equations (C1)-(C10) node-for-node and
# term-for-term, with no marginalization: every latent (y_TF, x, c, eta,
# y, z, Delta) and every measurement (xhat, yhat, zhat) gets its own node,
# and every additive term in the conditional means gets its own edge.
# The z-band and g-band chains are structurally identical (same equations,
# with b in {z, g}), so they are drawn as single combined boxes.

FONT = "Arial"
plt.rcParams["font.family"] = FONT

NEUTRAL = dict(fc="#FFFFFF", ec="#212121")   # pure TF chain: yTF, x, xhat
BLUE = dict(fc="#E8F0FE", ec="#4285F4")      # color latents: c, eta
PURPLE = dict(fc="#F1E9FB", ec="#8E24AA")    # color parameters box
CORAL = dict(fc="#FFEDE8", ec="#E64A19")     # redshift / k-correction chain
GREEN = dict(fc="#E8F5E9", ec="#43A047")     # combined per-band magnitudes: y, z/g, yhat, zhat/ghat

PARAM_BOXES = {
    "tf": (0.3, 8.5, 2.4, 0.62, "TF parameters", "α, β, σ_int,x", NEUTRAL),
    "color": (3.0, 8.5, 2.9, 0.62, "Color parameters", "μ_c,δ,τ_c  &  μ_g,δ_g,τ_g", PURPLE),
    "kcorr": (9.1, 8.5, 2.4, 0.62, "K-correction", "α_k,b  (b = r,z,g)", CORAL),
}

NODES = {
    # eq label included in the subtitle so each node is traceable to (C1)-(C10)
    "yTF":   (1.1, 7.1, 1.7, 0.62, "y_TF", "Uniform(y_min, y_max)  (C1)", NEUTRAL),
    "xtrue": (1.1, 5.7, 1.7, 0.62, "x", "(C2)", NEUTRAL),
    "c":     (3.5, 5.7, 1.9, 0.62, "c , c_g", "intrinsic color: r−z, g−r  (C3)", BLUE),
    "eta":   (6.0, 5.7, 1.9, 0.62, "η , η_g", "residual color: r−z, g−r  (C4)", BLUE),
    "zobs":  (9.3, 7.1, 1.9, 0.62, "z_obs", "observed redshift", CORAL),
    "Delta": (9.3, 5.7, 1.9, 0.62, "Δ", "k-correction shift  (C10)", CORAL),
    "y":     (2.6, 4.15, 1.7, 0.62, "y", "true r-mag  (C5)", GREEN),
    "w":     (6.4, 4.15, 1.9, 0.62, "z , g", "true z/g-mag  (C6)", GREEN),
    "xhat":  (1.1, 2.6, 1.6, 0.62, "x̂", "observed velocity  (C7)", NEUTRAL),
    "yhat":  (2.6, 2.6, 1.7, 0.62, "ŷ", "observed r-mag  (C8)", GREEN),
    "what":  (6.4, 2.6, 1.9, 0.62, "ẑ , ĝ", "observed z/g-mag  (C9)", GREEN),
}

ALL_BOXES = {**PARAM_BOXES, **NODES}

# solid = stochastic sampling step; dashed = deterministic / defining relation
SOLID_ARROWS = [
    ("tf", "xtrue"), ("yTF", "xtrue"),
    ("color", "c"), ("xtrue", "c"),
    ("yTF", "y"), ("eta", "y"), ("Delta", "y"),
    ("yTF", "w"), ("eta", "w"), ("xtrue", "w"), ("color", "w"), ("Delta", "w"),
    ("xtrue", "xhat"), ("y", "yhat"), ("w", "what"),
]

DASHED_ARROWS = [
    ("c", "eta", 0.0), ("xtrue", "eta", -0.15), ("color", "eta", 0.15),
    ("kcorr", "Delta", -0.1), ("zobs", "Delta", 0.1),
]

# Groups used for the "alpha" (highlight) renders. Overlap is allowed:
# x_true is relevant to both the TF chain and the color chain.
TF_GROUP = {"tf", "yTF", "xtrue", "xhat"}
COLOR_GROUP = {"color", "xtrue", "c", "eta"}
KCORR_GROUP = {"kcorr", "zobs", "Delta"}
OUTPUTS = {"y", "w", "yhat", "what"}

FADE_ALPHA = 0.15


def draw_box(ax, x, y, w, h, title, sub, style, centers, key, alpha=1.0):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                          linewidth=1.1, facecolor=style["fc"], edgecolor=style["ec"],
                          alpha=alpha, zorder=3)
    ax.add_patch(box)
    cx, cy = x + w / 2, y + h / 2
    centers[key] = (cx, cy)
    ax.text(cx, cy + 0.14, title, ha="center", va="center", fontsize=12, fontweight="bold",
            color="#212121", alpha=alpha, zorder=4)
    ax.text(cx, cy - 0.15, sub, ha="center", va="center", fontsize=7.6, color="#595959",
            alpha=alpha, zorder=4)


def render(out_path, highlight=None):
    # highlight: None -> full-opacity "current" version.
    # Otherwise a set of keys to keep at full opacity; everything else fades.
    def node_alpha(key):
        if highlight is None:
            return 1.0
        return 1.0 if key in highlight else FADE_ALPHA

    def edge_alpha(a, b):
        if highlight is None:
            return 1.0
        return 1.0 if (a in highlight and b in highlight) else FADE_ALPHA

    fig, ax = plt.subplots(figsize=(13.2, 9.6))
    ax.set_xlim(0, 11.9)
    ax.set_ylim(2.0, 9.4)
    ax.axis("off")

    centers = {}
    for key, (x, y, w, h, title, sub, style) in PARAM_BOXES.items():
        draw_box(ax, x, y, w, h, title, sub, style, centers, key, alpha=node_alpha(key))
    for key, (x, y, w, h, title, sub, style) in NODES.items():
        draw_box(ax, x, y, w, h, title, sub, style, centers, key, alpha=node_alpha(key))

    for a, b in SOLID_ARROWS:
        (x1, y1), (x2, y2) = centers[a], centers[b]
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                 linewidth=0.9, color="#78909C", alpha=edge_alpha(a, b),
                                 shrinkA=24, shrinkB=24, zorder=2,
                                 connectionstyle="arc3,rad=0.05")
        ax.add_patch(arrow)

    for a, b, rad in DASHED_ARROWS:
        (x1, y1), (x2, y2) = centers[a], centers[b]
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                 linewidth=0.9, color="#78909C", linestyle=(0, (5, 3)),
                                 alpha=edge_alpha(a, b),
                                 shrinkA=24, shrinkB=24, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}")
        ax.add_patch(arrow)

    plate = FancyBboxPatch((0.7, 2.1), 8.2, 5.85, boxstyle="round,pad=0.02,rounding_size=0.08",
                            linewidth=1.0, linestyle=(0, (5, 3)), facecolor="none", edgecolor="#78909C", zorder=1)
    ax.add_patch(plate)
    ax.text(0.9, 7.72, "galaxies, n = 1…N", fontsize=9, color="#595959", ha="left")

    # legend for solid vs dashed
    ax.plot([9.5, 10.3], [2.55, 2.55], color="#78909C", linewidth=0.9, zorder=2)
    ax.text(10.4, 2.55, "stochastic (sampled)", fontsize=8, color="#595959", va="center")
    ax.plot([9.5, 10.3], [2.25, 2.25], color="#78909C", linewidth=0.9, linestyle=(0, (5, 3)), zorder=2)
    ax.text(10.4, 2.25, "deterministic (defined)", fontsize=8, color="#595959", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


HERE = os.path.dirname(os.path.abspath(__file__))
render(os.path.join(HERE, "dag_native.png"), highlight=None)
render(os.path.join(HERE, "dag_native_highlight_tf.png"), highlight=TF_GROUP | OUTPUTS)
render(os.path.join(HERE, "dag_native_highlight_color.png"), highlight=COLOR_GROUP | OUTPUTS)
render(os.path.join(HERE, "dag_native_highlight_kcorr.png"), highlight=KCORR_GROUP | OUTPUTS)
print("done")
