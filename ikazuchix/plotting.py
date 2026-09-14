"""Matplotlib plotting helpers (optional dependency)."""

import numpy as np

# ===========================================================================
# Plotting helpers
# ===========================================================================

_DEFECT_COLORS = [
    "#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED",
    "#0891B2", "#DB2777", "#65A30D", "#EA580C", "#4B5563",
    "#92400E", "#0F766E",
]
_CHARGE_COLORS = [
    "#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED",
    "#0891B2", "#DB2777", "#65A30D", "#EA580C", "#4B5563",
]

def _defect_color(idx):
    return _DEFECT_COLORS[idx % len(_DEFECT_COLORS)]

def _charge_color(q, charges):
    return _CHARGE_COLORS[charges.index(q) % len(_CHARGE_COLORS)]


def _draw_band_edges(ax, gap, x_lo, x_hi):
    """Draw VBM/CBM lines; shade out-of-gap regions."""
    ax.axvline(0,   color="#64748B", lw=1.0, zorder=2)
    ax.axvline(gap, color="#64748B", lw=1.0, zorder=2)
    ax.text(0,   0.01, "VBM", transform=ax.get_xaxis_transform(),
            ha="right", va="bottom", fontsize=8, color="#475569")
    ax.text(gap, 0.01, "CBM", transform=ax.get_xaxis_transform(),
            ha="left",  va="bottom", fontsize=8, color="#475569")
    if x_lo < 0:
        ax.axvspan(x_lo, 0,   color="#E2E8F0", alpha=0.55, zorder=0,
                   label="Below VBM")
    if x_hi > gap:
        ax.axvspan(gap, x_hi, color="#FEF9C3", alpha=0.55, zorder=0,
                   label="Above CBM")


def plot_single_defect(data, res, fermi_ref, out_path):
    """Two-panel figure: E_f vs E_F + E_KO bar chart (per defect)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("[WARNING] matplotlib not installed. Skipping plot.")
        return

    charges = res["charges"]
    grid    = res["fermi_grid"]
    gap     = data["band_structure"]["cbm"] - data["band_structure"]["vbm"]
    x_lo, x_hi = grid[0], grid[-1]

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(8, 9),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.40})

    for q in charges:
        ax.plot(grid, res["ef_by_charge"][q],
                lw=1.6, ls="--", color=_charge_color(q, charges),
                alpha=0.70, label=f"q = {q:+d}", zorder=3)

    ax.plot(grid, res["envelope"],
            lw=2.8, color="black", label="Stable envelope", zorder=5)
    ax.axhline(0, color="#94A3B8", lw=0.7, ls=":", zorder=2)
    ax.axvline(fermi_ref, color="#F59E0B", lw=1.3, ls="-.",
               label=f"E$_F$ = {fermi_ref:.3f} eV", zorder=4)

    _draw_band_edges(ax, gap, x_lo, x_hi)

    # Mark only stable CTLs
    ctl_count = 0
    for ctl in sorted(res["stable_ctls"], key=lambda x: x["E_F_CTL"]):
        ef_ctl = ctl["E_F_CTL"]
        ax.axvline(ef_ctl, color="#94A3B8", lw=0.9, ls=":", zorder=3)
        y_pos = 0.95 - 0.09 * ctl_count
        ax.text(ef_ctl, y_pos,
                f"$\\varepsilon$({ctl['q1']:+d}/{ctl['q2']:+d})\n"
                f"{ef_ctl:.3f} eV",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7, color="#475569",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))
        ctl_count += 1

    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("Fermi level relative to VBM  (eV)", fontsize=12)
    ax.set_ylabel("Formation energy  (eV)", fontsize=12)
    ax.set_title("Defect Formation Energy", fontsize=11, fontweight="bold")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which="major", alpha=0.20, zorder=1)
    ax.legend(fontsize=9, loc="best", framealpha=0.88)

    ko_vals = [res["ko"][q]["E_KO"] for q in charges]
    colors  = [_charge_color(q, charges) for q in charges]
    x_pos   = np.arange(len(charges))
    bars = ax2.bar(x_pos, ko_vals, color=colors, edgecolor="white",
                   linewidth=0.8, width=0.55, zorder=3)
    y_span = max(ko_vals) - min(ko_vals) if len(ko_vals) > 1 else 1.0
    y_off  = 0.03 * abs(y_span) if y_span != 0 else 0.05
    for bar, val in zip(bars, ko_vals):
        va = "bottom" if val >= 0 else "top"
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 val + (y_off if val >= 0 else -y_off),
                 f"{val:+.3f}", ha="center", va=va,
                 fontsize=8, color="#1E293B")
    ax2.axhline(0, color="black", lw=0.8, zorder=2)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"q = {q:+d}" for q in charges], fontsize=10)
    ax2.set_ylabel("E$_{KO}$  (eV)", fontsize=11)
    ax2.set_title("Kumagai-Oba Correction per Charge State", fontsize=11)
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.grid(True, axis="y", which="major", alpha=0.22, zorder=1)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Per-defect plot saved to: {out_path}")
    plt.close(fig)


def plot_comparison(data, all_results, fermi_ref, out_path):
    """
    Comparison figure: stable envelopes of all defects overlaid.
    Only the minimum-energy envelope per defect is drawn (no individual
    charge-state curves) to keep the figure clean.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("[WARNING] matplotlib not installed. Skipping comparison plot.")
        return

    gap  = data["band_structure"]["cbm"] - data["band_structure"]["vbm"]
    grid = all_results[0]["fermi_grid"]
    x_lo, x_hi = grid[0], grid[-1]

    fig, ax = plt.subplots(figsize=(9, 6))

    all_envelopes = []
    for idx, res in enumerate(all_results):
        color = _defect_color(idx)
        ax.plot(grid, res["envelope"],
                lw=2.4, ls="-", color=color, zorder=5,
                label=res["name"])
        all_envelopes.append(res["envelope"])
        # D-4a: stable CTL vertical lines (thin, colour-matched)
        for ctl in res["stable_ctls"]:
            if 0 <= ctl["E_F_CTL"] <= gap:
                ax.axvline(ctl["E_F_CTL"], color=color,
                           lw=0.8, ls=":", alpha=0.55, zorder=3)

    ax.axvline(fermi_ref, color="#F59E0B", lw=1.3, ls="-.",
               label=f"E$_F$ = {fermi_ref:.3f} eV", zorder=6)
    ax.axhline(0, color="#94A3B8", lw=0.7, ls=":", zorder=2)
    _draw_band_edges(ax, gap, x_lo, x_hi)

    # D-4b: auto-clip y-axis -- focus on the range covered by envelopes
    #        within the sweep window, with a +/-15 % margin.
    env_stack = np.vstack(all_envelopes)           # shape (n_defects, N_pts)
    in_gap    = (grid >= 0) & (grid <= gap)
    if in_gap.any():
        ymin_data = env_stack[:, in_gap].min()
        ymax_data = env_stack[:, in_gap].max()
    else:
        ymin_data, ymax_data = env_stack.min(), env_stack.max()
    yspan = max(ymax_data - ymin_data, 1.0)
    ax.set_ylim(ymin_data - 0.15 * yspan, ymax_data + 0.15 * yspan)

    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("Fermi level relative to VBM  (eV)", fontsize=12)
    ax.set_ylabel("Formation energy  (eV)", fontsize=12)
    ax.set_title("Defect Formation Energy", fontsize=11, fontweight="bold")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which="major", alpha=0.20, zorder=1)
    ax.legend(fontsize=10, loc="best", framealpha=0.88)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Comparison plot saved to: {out_path}")
    plt.close(fig)


def plot_stability_region(data, stability_result, out_path):
    """Plot the 2D chemical-potential stability region (ternary hosts
    only -- see ikazuchix.stability_region.feasible_region_2d), with the
    input file's 'chemical_potentials' point marked. Skipped (with a
    console message, not an error) for non-ternary hosts, where no 2D
    region is computed in the first place."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not installed. Skipping stability plot.")
        return

    if "region_vertices" not in stability_result:
        print("[INFO] Stability region plot requires exactly 3 host "
              "species (ternary system); skipping plot.")
        return

    sx = stability_result["region_species_x"]
    sy = stability_result["region_species_y"]
    vertices = stability_result["region_vertices"]

    fig, ax = plt.subplots(figsize=(7, 6))

    if vertices:
        xs = [v[0] for v in vertices] + [vertices[0][0]]
        ys = [v[1] for v in vertices] + [vertices[0][1]]
        ax.fill(xs, ys, color="#2563EB", alpha=0.25, zorder=2,
                label="Stability region")
        ax.plot(xs, ys, color="#2563EB", lw=1.6, zorder=3)
    else:
        ax.text(0.5, 0.5,
                "No feasible region\n(host unstable against the given "
                "competing phases)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="#DC2626")

    point = stability_result["point_deltas"]
    px, py = point.get(sx, 0.0), point.get(sy, 0.0)
    color = "#16A34A" if stability_result["feasible"] else "#DC2626"
    ax.scatter([px], [py], color=color, s=70, zorder=5,
               label="Input chemical potentials",
               edgecolor="white", linewidth=1.0)

    ax.axhline(0, color="#94A3B8", lw=0.7, ls=":", zorder=1)
    ax.axvline(0, color="#94A3B8", lw=0.7, ls=":", zorder=1)
    ax.set_xlabel(f"$\\Delta\\mu_{{{sx}}}$  (eV)", fontsize=12)
    ax.set_ylabel(f"$\\Delta\\mu_{{{sy}}}$  (eV)", fontsize=12)
    host_formula = data.get("bulk", {}).get("formula", "")
    title = "Chemical Potential Stability Region"
    ax.set_title(f"{title}  [{host_formula}]" if host_formula else title,
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.20, zorder=0)
    ax.legend(fontsize=9, loc="best", framealpha=0.88)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Stability region plot saved to: {out_path}")
    plt.close(fig)
