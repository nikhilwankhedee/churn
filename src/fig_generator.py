# =============================================================================
# FINAL RESEARCH FIGURE GENERATOR
# SAVES EVERYTHING INTO:
# final-research-figs/
# =============================================================================

import os
from pathlib import Path

from PIL import Image

import matplotlib.pyplot as plt

# =============================================================================
# ROOT DIRECTORY — configurable via FIG_ROOT env var
# =============================================================================

ROOT = os.environ.get(
    "FIG_ROOT",
    str(Path(__file__).resolve().parent.parent.parent.parent / "IMP FIGURES"),
)

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

OUTPUT_DIR = os.path.join(
    ROOT,
    "final-research-figs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# V2 FIGURE PATHS ONLY
# =============================================================================

ROC_PATHS = {

    "Instacart":
    f"{ROOT}/Instacart/V2/figures(2)/roc_comparison.png",

    "Olist":
    f"{ROOT}/olist/v2/olist_roc.png",

    "RetailRocket":
    f"{ROOT}/RetailRocket/v2/retailrocket_roc.png",

    "REES46":
    f"{ROOT}/REES64/v2/rees46_roc.png"
}

SHAP_PATHS = {

    "Instacart":
    f"{ROOT}/Instacart/V2/figures(2)/shap_summary.png",

    "Olist":
    f"{ROOT}/olist/v2/olist_shap.png",

    "RetailRocket":
    f"{ROOT}/RetailRocket/v2/retailrocket_shap.png",

    "REES46":
    f"{ROOT}/REES64/v2/rees46_shap.png"
}

CALIBRATION_PATHS = {

    "Instacart":
    f"{ROOT}/Instacart/V2/figures(2)/calibration_comparison.png",

    "Olist":
    f"{ROOT}/olist/v2/olist_calibration.png",

    "RetailRocket":
    f"{ROOT}/RetailRocket/v2/retailrocket_calibration.png",

    "REES46":
    f"{ROOT}/REES64/v2/rees46_calibration.png"
}

# =============================================================================
# GENERIC COLLAGE FUNCTION
# =============================================================================

def create_collage(

    image_paths,
    title,
    save_path,
    figsize=(16,12)

):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=figsize
    )

    axes = axes.flatten()

    for ax, (dataset, path) in zip(
        axes,
        image_paths.items()
    ):

        if not os.path.exists(path):

            print(f"[MISSING] {path}")
            continue

        img = Image.open(path)

        ax.imshow(img)

        ax.set_title(
            dataset,
            fontsize=16,
            fontweight='bold'
        )

        ax.axis("off")

    plt.suptitle(
        title,
        fontsize=22,
        fontweight='bold'
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# =============================================================================
# FIGURE 2 — ROC COLLAGE
# =============================================================================

print("="*80)
print("GENERATING FIGURE 2 — ROC COLLAGE")
print("="*80)

create_collage(

    image_paths=ROC_PATHS,

    title="Figure 2 — Cross-Ecosystem ROC Comparison",

    save_path=os.path.join(
        OUTPUT_DIR,
        "figure2_roc_collage.png"
    )
)

print("[OK] Figure 2 generated")

# =============================================================================
# FIGURE 3 — SHAP COLLAGE
# =============================================================================

print("\n" + "="*80)
print("GENERATING FIGURE 3 — SHAP COLLAGE")
print("="*80)

create_collage(

    image_paths=SHAP_PATHS,

    title="Figure 3 — Cross-Ecosystem SHAP Comparison",

    save_path=os.path.join(
        OUTPUT_DIR,
        "figure3_shap_collage.png"
    )
)

print("[OK] Figure 3 generated")

# =============================================================================
# FIGURE 4 — CALIBRATION COLLAGE
# =============================================================================

print("\n" + "="*80)
print("GENERATING FIGURE 4 — CALIBRATION COLLAGE")
print("="*80)

create_collage(

    image_paths=CALIBRATION_PATHS,

    title="Figure 4 — Cross-Ecosystem Calibration Comparison",

    save_path=os.path.join(
        OUTPUT_DIR,
        "figure4_calibration_collage.png"
    )
)

print("[OK] Figure 4 generated")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*80)
print("ALL FINAL RESEARCH FIGURES GENERATED")
print("="*80)

print("\nSaved in:\n")
print(OUTPUT_DIR)

print("\nGenerated Files:\n")

for file in os.listdir(OUTPUT_DIR):

    print(f" - {file}")