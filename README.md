# Explainable AI with CNNs — Grad-CAM on PlantVillage

Tutorial code for **Explainable AI with CNNs: Visualizing Model Decisions using Grad-CAM**, trained on a subset of the **PlantVillage** plant disease dataset (downloaded via [KaggleHub](https://github.com/Kaggle/kagglehub) — dataset slug: `mohitsingh1804/plantvillage`).

## Repository link for markers

**Repository (public):** `https://github.com/Areeb-Gul1/individual-project.git`

## Quick start

1. Create a virtual environment (recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   If you use GPU, install a matching **PyTorch** build from [https://pytorch.org](https://pytorch.org).

3. Open and run `gradcam_plantvillage_tutorial.ipynb` from top to bottom.

First run may take time while `kagglehub` downloads the dataset.

### Notes for reproducibility (markers)

- **Determinism**: the notebook sets a fixed random seed and uses stratified splits.
- **Transfer learning weights**: ResNet-18 ImageNet weights are downloaded the first time you run training. If your network/proxy corrupts downloads, delete the cached file at `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth` and rerun (or switch networks).
- **Outputs**:
  - Figures are exported to `figures/` for the webpage.
  - Training artifacts (best checkpoint + metrics + classification report text) are exported to `artifacts/`.

## Project layout

| File | Purpose |
|------|---------|
| `gradcam_plantvillage_tutorial.ipynb` | Full pipeline: data loading → training → predictions → evaluation → Grad-CAM |
| `src/` | Modular research-quality functions used by the notebook |
| `requirements.txt` | Python dependencies |
| `LICENSE` | MIT — reuse conditions for assessors |
| `index.html` | Story-based webpage (main marking focus) |
| `figures/` | Exported figures referenced by the webpage |
| `artifacts/` | Saved checkpoints + metrics logs |

## Accessibility (for your write-up)

For the PDF/web/video, consider: colour-blind-safe figure palettes (this notebook uses perceptually uniform heatmaps where possible), figure captions describing what non-colour cues mean, and alt-text for each key figure.

## References (add to your tutorial document)

The notebook includes a **References** cell with starter citations; extend this list in your submitted tutorial PDF/web page as required by your course.
