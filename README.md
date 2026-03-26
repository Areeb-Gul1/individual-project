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

## Project layout

| File | Purpose |
|------|---------|
| `gradcam_plantvillage_tutorial.ipynb` | Full pipeline: data loading → training → predictions → evaluation → Grad-CAM |
| `requirements.txt` | Python dependencies |
| `LICENSE` | MIT — reuse conditions for assessors |

## Accessibility (for your write-up)

For the PDF/web/video, consider: colour-blind-safe figure palettes (this notebook uses perceptually uniform heatmaps where possible), figure captions describing what non-colour cues mean, and alt-text for each key figure.

## References (add to your tutorial document)

The notebook includes a **References** cell with starter citations; extend this list in your submitted tutorial PDF/web page as required by your course.
