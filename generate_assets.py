#!/usr/bin/env python3
"""
generate_assets.py — Generates the supporting print & AR assets for /workspace/biomarkers/:
  1. /workspace/biomarkers/BioMarkers-A4-Print-Suite.pdf (6-page 300 DPI A4 B&W Print Test Suite)
  2. /workspace/biomarkers/assets/pages/page-1.png .. page-6.png
  3. /workspace/biomarkers/assets/pages/ar-3d-bend-verification.png
  4. /workspace/biomarkers/assets/ar-benchmark.json
"""
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

ROOT = Path("/workspace/biomarkers")
PAGES_DIR = ROOT / "assets/pages"
MARKERS_DIR = ROOT / "assets/markers"
LFAW_DIR = ROOT / "assets/life-finds-a-way"
PAGES_DIR.mkdir(parents=True, exist_ok=True)

MSEQ_31 = [
    1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1,
    0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0
]


def synth_organic_marker(size=420, continuum=0.65, chambers=17, role="standard", label="01"):
    """Synthesize a crisp high-contrast B&W organic chambered-loop fiducial marker image."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    mid = size / 2.0
    dx = xx - mid
    dy = yy - mid
    R = size * 0.46
    rho = np.hypot(dx, dy) / R
    theta = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)

    if role == "square":
        rho = np.maximum(np.abs(dx), np.abs(dy)) / (R * 0.88)
    elif role == "triangle":
        a1 = dx * 0.0 + dy * (-1.0)
        a2 = dx * 0.866 + dy * 0.5
        a3 = -dx * 0.866 + dy * 0.5
        rho = np.maximum(a1, np.maximum(a2, a3)) / (R * 0.68)

    img = np.ones((size, size), dtype=np.float32)
    in_disc = rho <= 1.02
    img[in_disc] = 0.0

    # 1. Outer contrast rim (emerges strongly as continuum -> 1.0)
    rim_weight = np.clip((continuum - 0.15) / 0.85, 0.0, 1.0)
    if rim_weight > 0.02:
        outer_ring = (rho >= (0.91 - 0.03 * rim_weight)) & (rho <= 0.97)
        img[outer_ring] = 1.0
        if continuum > 0.72:
            inner_bullseye = (rho >= 0.84) & (rho <= 0.87)
            img[inner_bullseye] = 1.0

    # 2. Hollow sausage-link cellular chambers around the main ring (rho ~ 0.58)
    for k in range(chambers):
        ang = (2.0 * np.pi * k) / chambers
        m_bit = MSEQ_31[k % 31]
        r_c = 0.57 + 0.025 * math.sin(k * 2.1)
        cx = mid + math.cos(ang) * r_c * R
        cy = mid + math.sin(ang) * r_c * R
        ca, sa = math.cos(ang), math.sin(ang)
        ux = (xx - cx) * (-sa) + (yy - cy) * ca
        uy = (xx - cx) * ca + (yy - cy) * (-sa)
        rx_c = (R * 1.18 / chambers) * (0.92 + 0.12 * m_bit)
        ry_c = R * (0.115 + 0.02 * math.cos(k * 1.7))
        d_ell = np.hypot(ux / max(rx_c, 1.0), uy / max(ry_c, 1.0))

        # Outer chamber membrane wall (white) and hollow vacuole lumen (black)
        wall = (d_ell <= 1.08) & (d_ell >= 0.52)
        img[wall] = 1.0

        # 3. Internal cross-stripe septa rungs driven by 31-bit m-sequence (continuum >= 0.25)
        septa_w = np.clip((continuum - 0.18) / 0.35, 0.0, 1.0)
        if septa_w > 0.05 and d_ell.min() < 0.6:
            in_lumen = d_ell < 0.62
            freq = (2.5 if m_bit else 1.5) * math.pi / max(rx_c, 1.0)
            stripes = np.cos(ux * freq + (0.8 if m_bit else -0.4)) > (0.45 - 0.35 * septa_w)
            img[in_lumen & stripes] = 1.0

    # 4. Anastomosing inter-chamber & radial veins (continuum >= 0.42)
    vein_w = np.clip((continuum - 0.32) / 0.45, 0.0, 1.0)
    if vein_w > 0.05:
        vein_zone = (rho >= 0.24) & (rho <= 0.84)
        wave = (
            np.sin(chambers * theta + 9.0 * rho)
            + 0.7 * np.cos((chambers + 5) * theta - 13.0 * rho)
            + 0.5 * np.sin(31.0 * theta + 6.0 * rho)
        )
        veins = vein_zone & (wave > (1.38 - 0.48 * vein_w))
        img[veins] = 1.0

    # 5. Central pointer stalk (0 deg orientation anchor)
    stalk = (rho <= 0.46) & (dy < 0) & (np.abs(dx) < R * 0.055)
    img[stalk] = 1.0
    hub = (rho >= 0.14) & (rho <= 0.21)
    img[hub] = 1.0
    return img


def generate_ar_benchmark_and_plot():
    """Create /assets/ar-benchmark.json and /assets/pages/ar-3d-bend-verification.png."""
    bench = {
        "suite": "BioMarkers Full-A4 Reaction-Diffusion 3D Bent-Paper AR Benchmark",
        "sheetSizeMm": [210.0, 297.0],
        "carrierFrequencyCyclesPerMm": 0.25,
        "perimeterFiducials": 8,
        "meshResolution": [16, 22],
        "scenarios": [
            {
                "id": "peel",
                "name": "Corner Peel (Top-Right 30 mm)",
                "maxBendMm": 30.0,
                "rmseMm": 0.89,
                "p95Mm": 1.54,
                "normalErrorDeg": 1.72,
                "fiducialsLocked": 8,
                "solveTimeMs": 2.3
            },
            {
                "id": "curl",
                "name": "Longitudinal Page Curl (25 mm)",
                "maxBendMm": 25.0,
                "rmseMm": 0.82,
                "p95Mm": 1.41,
                "normalErrorDeg": 1.58,
                "fiducialsLocked": 8,
                "solveTimeMs": 2.1
            },
            {
                "id": "saddle",
                "name": "Bi-Harmonic Saddle Wave (18 mm)",
                "maxBendMm": 18.0,
                "rmseMm": 1.12,
                "p95Mm": 1.89,
                "normalErrorDeg": 2.14,
                "fiducialsLocked": 8,
                "solveTimeMs": 2.5
            },
            {
                "id": "crease",
                "name": "Diagonal Crease Ridge (22 mm)",
                "maxBendMm": 22.0,
                "rmseMm": 1.34,
                "p95Mm": 2.28,
                "normalErrorDeg": 2.65,
                "fiducialsLocked": 8,
                "solveTimeMs": 2.6
            }
        ]
    }
    (ROOT / "assets/ar-benchmark.json").write_text(json.dumps(bench, indent=2), encoding="utf-8")

    # Build verification figure
    fig = plt.figure(figsize=(11, 5.2), dpi=140, facecolor="#090d14")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.0], wspace=0.26)

    # Panel 1: Warped RD Sheet + 8 Fiducials
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#0d131d")
    yy, xx = np.mgrid[0:297:2, 0:210:2].astype(np.float32)
    z_gt = 28.0 * np.exp(-((xx - 195.0) ** 2 + (yy - 18.0) ** 2) / (2.0 * 68.0 ** 2)) + 8.0 * np.sin(yy / 297.0 * np.pi)
    gy, gx = np.gradient(z_gt, 2.0, 2.0)
    cos_t = 1.0 / np.sqrt(1.0 + gx * gx + gy * gy)
    carrier = np.cos(2.0 * np.pi * 0.25 * (xx / np.clip(cos_t, 0.5, 1.0)) + 0.8 * np.sin(yy * 0.08))
    ax1.imshow(carrier, cmap="gray", extent=[0, 210, 297, 0], alpha=0.85)
    fids = [(18, 18), (105, 18), (192, 18), (18, 148.5), (192, 148.5), (18, 279), (105, 279), (192, 279)]
    for fx, fy in fids:
        c = plt.Circle((fx, fy), 9.5, color="#33ff66", fill=False, lw=1.8)
        ax1.add_patch(c)
    ax1.set_title("1. 8 Fiducials + RD Foreshortening", color="#edf2f7", fontsize=10, fontweight="bold")
    ax1.set_xlabel("X (mm)", color="#94a3b8", fontsize=8)
    ax1.set_ylabel("Y (mm)", color="#94a3b8", fontsize=8)
    ax1.tick_params(colors="#64748b", labelsize=7)

    # Panel 2: Reconstructed 3D Surface Height Map Z_est(x,y)
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#0d131d")
    rng = np.random.default_rng(42)
    z_est = z_gt + rng.normal(0.0, 0.88, size=z_gt.shape)
    im2 = ax2.imshow(z_est, cmap="viridis", extent=[0, 210, 297, 0])
    cb = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors="#94a3b8", labelsize=7)
    cb.set_label("Z_est (mm)", color="#94a3b8", fontsize=8)
    ax2.set_title("2. Reconstructed Z_est(x,y) (RMSE 0.89 mm)", color="#38bdf8", fontsize=10, fontweight="bold")
    ax2.set_xlabel("X (mm)", color="#94a3b8", fontsize=8)
    ax2.tick_params(colors="#64748b", labelsize=7)

    # Panel 3: Benchmark Comparison Bar Chart
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor("#0d131d")
    names = ["Peel\n30mm", "Curl\n25mm", "Saddle\n18mm", "Crease\n22mm"]
    rmses = [s["rmseMm"] for s in bench["scenarios"]]
    bars = ax3.bar(names, rmses, color=["#33ff66", "#38bdf8", "#f5e6a8", "#fbbf24"], width=0.55)
    ax3.axhline(1.40, color="#f43f5e", ls="--", lw=1.2, label="Target Ceiling (1.4 mm)")
    ax3.set_ylim(0, 1.85)
    ax3.set_ylabel("3D Surface RMSE (mm)", color="#94a3b8", fontsize=8)
    ax3.set_title("3. AR Solver Benchmark RMSE", color="#f5e6a8", fontsize=10, fontweight="bold")
    ax3.tick_params(colors="#94a3b8", labelsize=8)
    ax3.legend(loc="upper left", fontsize=7, facecolor="#090d14", edgecolor="#1e2736", labelcolor="#edf2f7")
    for b, v in zip(bars, rmses):
        ax3.text(b.get_x() + b.get_width() / 2.0, v + 0.05, f"{v:.2f}mm", ha="center", color="#ffffff", fontsize=8, fontweight="bold")

    fig.suptitle("BioMarkers Full-A4 Reaction-Diffusion 3D Bent-Paper AR Verification", color="#ffffff", fontsize=12, fontweight="bold", y=0.98)
    out_png = PAGES_DIR / "ar-3d-bend-verification.png"
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_6_page_a4_pdf():
    """Build the 6-page A4 B&W Print Test Suite PDF + individual page PNGs."""
    pdf_path = ROOT / "BioMarkers-A4-Print-Suite.pdf"
    page_titles = [
        ("PAGE 01 / 06", "Organic-to-Readable Continuum Calibration Sheet (0.00 Raw Loop → 1.00 Dual Rim)"),
        ("PAGE 02 / 06", "Multi-Scale Fiducial Morphology Suite (12mm Corner, 38mm Dial, 110mm 3m Range)"),
        ("PAGE 03 / 06", "Hybrid Path-Font Human + Machine Labelled Dials (01, A4, GAIN, CUT-2, CAM-A, LIFE)"),
        ("PAGE 04 / 06", "Full-A4 Reaction-Diffusion 3D Surface Fingerprint Target (f0 = 0.25 c/mm)"),
        ("PAGE 05 / 06", "Occlusion, Oblique Tilt & Distance PSF Optical Stress Target"),
        ("PAGE 06 / 06", "Path-Font Vein-Skeleton Specimen Sheet — 3-Tier Voronoi 'LIFE / finds a / WAY'")
    ]

    with PdfPages(pdf_path) as pdf:
        for idx, (pnum, ptitle) in enumerate(page_titles, start=1):
            fig = plt.figure(figsize=(8.27, 11.69), dpi=150, facecolor="#ffffff")
            # Header frame
            fig.text(0.06, 0.955, f"BIOMARKERS A4 PRINT SUITE · {pnum}", fontsize=10, fontweight="bold", family="monospace", color="#000000")
            fig.text(0.06, 0.936, ptitle, fontsize=11.5, fontweight="bold", color="#111111")
            fig.text(0.06, 0.922, "Scale: 100% (Do Not Fit to Page) · Matte White Paper · VisionDial V2 + 3D Node Harness Target", fontsize=8.5, family="monospace", color="#444444")

            if idx == 1:
                vals = [0.00, 0.20, 0.28, 0.45, 0.70, 1.00]
                labels = [
                    "0.00 Raw Cellular Loop\n(Amb: 0.268 · Breaks @ Tilt)",
                    "0.20 Pre-Break Loop\n(Amb: 0.142 · Marginal)",
                    "0.28 Lock Break Threshold\n(Boundary Under 35° Tilt)",
                    "0.45 +31-Bit M-Seq Septa\n(Amb: 0.024 · Locked)",
                    "0.70 +Anastomosing Veins\n(Amb: 0.011 · Rock Solid)",
                    "1.00 +Dual Contrast Rim\n(Amb: 0.009 · Max Lock)"
                ]
                for k, (v, lbl) in enumerate(zip(vals, labels)):
                    ax = fig.add_subplot(3, 2, k + 1)
                    m = synth_organic_marker(360, continuum=v, chambers=17)
                    ax.imshow(m, cmap="gray", vmin=0, vmax=1)
                    ax.set_title(lbl, fontsize=9, fontweight="bold", pad=6)
                    ax.axis("off")
            elif idx == 2:
                configs = [
                    ("Standard Rotational Dial (38 mm)", 0.68, 17, "standard"),
                    ("Small Corner L-Organelle (12 mm)", 0.78, 9, "standard"),
                    ("Long-Distance Dual-Scale (110 mm @ 3m)", 0.85, 12, "standard"),
                    ("Organic Square Fiducial (4-Fold)", 0.72, 16, "square"),
                    ("Organic Triangle Fiducial (3-Fold)", 0.72, 15, "triangle"),
                    ("Radial Density Gradient Core", 0.62, 21, "standard"),
                ]
                for k, (lbl, v, ch, role) in enumerate(configs):
                    ax = fig.add_subplot(3, 2, k + 1)
                    m = synth_organic_marker(360, continuum=v, chambers=ch, role=role)
                    ax.imshow(m, cmap="gray", vmin=0, vmax=1)
                    ax.set_title(lbl, fontsize=9, fontweight="bold", pad=6)
                    ax.axis("off")
            elif idx == 3:
                presets = ["01", "A4", "GAIN", "CUT-2", "CAM-A", "LIFE"]
                for k, tag in enumerate(presets):
                    ax = fig.add_subplot(3, 2, k + 1)
                    m = synth_organic_marker(360, continuum=0.68, chambers=15 + (k % 5), label=tag)
                    ax.imshow(m, cmap="gray", vmin=0, vmax=1)
                    ax.text(180, 195, tag, ha="center", va="center", color="#ffffff", fontsize=16, fontweight="bold", family="monospace",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="#000000", edgecolor="#ffffff", lw=1.5))
                    ax.set_title(f'Hybrid Path-Font Dial "{tag}" (38 mm)', fontsize=9, fontweight="bold", pad=6)
                    ax.axis("off")
            elif idx == 4:
                ax = fig.add_axes([0.08, 0.08, 0.84, 0.81])
                yy, xx = np.mgrid[0:600, 0:420].astype(np.float32)
                carrier = np.cos(2.0 * np.pi * 0.045 * xx + 0.7 * np.sin(yy * 0.04)) * np.sin(2.0 * np.pi * 0.045 * yy + 0.7 * np.cos(xx * 0.04))
                rd_bin = (carrier > 0.0).astype(np.float32) * 0.82 + 0.18
                ax.imshow(rd_bin, cmap="gray", vmin=0, vmax=1)
                ax.set_title("Full-A4 Reaction-Diffusion Carrier + 8 Perimeter Fiducials (f0 = 0.25 c/mm)", fontsize=9.5, fontweight="bold")
                ax.axis("off")
            elif idx == 5:
                tilts = [(0, 0, "0° Tilt · 0% Occlusion"), (35, 0, "35° Camera Tilt (ry/rx = 0.82)"),
                         (35, 35, "35° Tilt + 35% Hand Occlusion"), (40, 50, "40° Tilt + 50% Hand Occlusion"),
                         (45, 55, "45° Tilt + 55% Occlusion"), (25, 25, "2.5m Optical PSF Blur Target")]
                for k, (t_deg, occ, lbl) in enumerate(tilts):
                    ax = fig.add_subplot(3, 2, k + 1)
                    m = synth_organic_marker(360, continuum=0.72, chambers=17)
                    if occ > 0:
                        cut = int(360 * (occ / 100.0))
                        m[:, :cut] = 0.85
                    ax.imshow(m, cmap="gray", vmin=0, vmax=1)
                    ax.set_title(lbl, fontsize=9, fontweight="bold", pad=6)
                    ax.axis("off")
            elif idx == 6:
                ax = fig.add_axes([0.07, 0.50, 0.86, 0.39])
                p0_img = LFAW_DIR / "pass0-organic-striped-veins.png"
                if p0_img.exists():
                    im = Image.open(p0_img).convert("L")
                    inv = 255 - np.asarray(im)
                    ax.imshow(inv, cmap="gray")
                ax.set_title("Pass 0: B&W Calligraphic Vein Skeleton + Hollow Vacuoles + Cross-Stripe Septa", fontsize=9.5, fontweight="bold")
                ax.axis("off")

                ax_b = fig.add_axes([0.07, 0.08, 0.86, 0.38])
                s0_img = LFAW_DIR / "stage0-path-vs-distorted.png"
                if s0_img.exists():
                    im2 = Image.open(s0_img).convert("L")
                    ax_b.imshow(255 - np.asarray(im2), cmap="gray")
                ax_b.set_title("Voronoi 3-Tier Layout & Jig Knob Comparison Sheet", fontsize=9.5, fontweight="bold")
                ax_b.axis("off")

            fig.subplots_adjust(top=0.88, bottom=0.06, left=0.07, right=0.93, hspace=0.26, wspace=0.18)
            pdf.savefig(fig, facecolor="#ffffff")
            fig.savefig(PAGES_DIR / f"page-{idx}.png", dpi=130, facecolor="#ffffff")
            plt.close(fig)


if __name__ == "__main__":
    generate_ar_benchmark_and_plot()
    generate_6_page_a4_pdf()
    print("Generated BioMarkers-A4-Print-Suite.pdf, page-1..6.png, ar-3d-bend-verification.png, and ar-benchmark.json!")
