# Parkinson's Spiral Drawing Screening Aid

CNN-based screening tool that classifies hand-drawn spiral images as healthy
or Parkinson's-screening-positive, with patient-level cross-validation and
bootstrap confidence intervals.

> **Note:** This is an MCA mini-project. The tool is a **screening aid**,
> not a diagnostic device. It flags spirals for clinical follow-up — it
> does not detect or diagnose Parkinson's disease.

---

## Project Status

🚧 **Pre-build phase** — Repository structure and methodology are finalized.
Implementation begins Week 1.

---

## Methodology

This project applies subject-independent cross-validation to spiral-PD
classification, following the protocol described in:

> (2025) "Finger drawing on smartphone screens enables early Parkinson's
> disease detection through hybrid 1D-CNN and BiGRU deep learning
> architecture." PMC12258557.
> https://pmc.ncbi.nlm.nih.gov/articles/PMC12258557

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Dataset | NewHandPD (122 subjects) | Larger than Zham (31), less saturated on GitHub |
| Evaluation | 5-fold GroupKFold (patient-level) | Prevents data leakage per PMC 2021 |
| Confidence intervals | BCa bootstrap (2000 resamples) | Quantifies uncertainty at small-N |
| Model comparison | McNemar's test (paired) | Correct test for same-image predictions |
| Explainability | Grad-CAM (plausibility check) | Paired with handcrafted-feature correlation |
| Demo | Upload-only (no live canvas) | Mouse input is OOD vs tablet-captured training |

---

## Why Not 98% Accuracy?

The most-replicated spiral-PD result in the literature is 98% accuracy on
the Zham/Kaggle dataset. This number is computed on a test set of 20-30
images from ~31 unique humans, using a pre-split that is not documented
as subject-level. The leakage literature (PMC 2021) shows non-subject-level
splits inflate accuracy by 30-55 percentage points.

This project re-evaluates under patient-level cross-validation. The honest
accuracy will likely land between 78-88%, with bootstrap confidence
intervals of ±4-5 percentage points.

---

## Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras** — CNN model
- **scikit-learn** — GroupKFold, McNemar, metrics
- **OpenCV / Pillow** — image preprocessing
- **Streamlit** — upload-only demo
- **tf-keras-vis** — Grad-CAM
- **Google Colab** — free GPU training

---

## Repository Structure

```
parkinsons-spiral-drawing-screening-aid/
├── README.md                  ← you are here
├── LICENSE                    ← MIT
├── .gitignore                 ← Python + data exclusions
├── requirements.txt           ← Python dependencies
│
├── data/
│   ├── raw/                   ← NewHandPD images (not committed)
│   │   └── README.md          ← download instructions
│   └── processed/             ← preprocessed arrays (not committed)
│       └── README.md
│
├── notebooks/                 ← Jupyter notebooks for EDA, experiments
│   └── README.md
│
├── src/                       ← source code
│   ├── __init__.py
│   ├── data_loader.py         ← load + split by patient ID
│   ├── model.py               ← CNN architecture
│   ├── train.py               ← 5-fold GroupKFold training
│   ├── evaluate.py            ← bootstrap CIs, McNemar, ROC
│   └── gradcam.py             ← Grad-CAM plausibility check
│
├── app/
│   └── streamlit_app.py       ← upload-only demo (no live canvas)
│
├── docs/
│   └── README.md              ← supplementary documentation
│
└── tests/
    └── README.md              ← unit tests (TBD)
```

---

## Timeline

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1 | Data audit + handcrafted features baseline | Working data pipeline + LR/SVM baseline |
| 2 | CNN training + patient-level 5-fold CV | First honest accuracy number |
| 3 | Bootstrap CIs + McNemar + Grad-CAM | Full evaluation suite |
| 4 | Streamlit demo + report + viva rehearsal | Submission-ready project |

---

## Limitations

- **Screening, not detection.** Cannot distinguish Parkinson's from
  essential tremor — both produce visually overlapping spiral pathology.
- **Static images only.** Discards pen pressure and speed, which the
  smart-pen literature identifies as the primary discriminative signal.
- **Small-N evaluation.** At n=122, confidence intervals are wider than
  clinically validated tools.
- **No live drawing.** Upload-only because mouse input is out-of-distribution
  relative to tablet-captured training data.
- **ET confound.** Model will have high false-positive rate on anyone with
  essential tremor, dystonic tremor, or a caffeinated hand.

---

## Installation (for reviewers)

```bash
# Clone the repository
git clone https://github.com/[your-username]/parkinsons-spiral-drawing-screening-aid.git
cd parkinsons-spiral-drawing-screening-aid

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Download dataset (see data/raw/README.md for instructions)

# Run demo (after model is trained)
streamlit run app/streamlit_app.py
```

---

## Citation

If you reference this project, please cite:

```bibtex
@misc{parkinsons_spiral_screening_2026,
  title        = {Parkinson's Spiral Drawing Screening Aid},
  author       = {[Your Name]},
  year         = {2026},
  howpublished = {\url{https://github.com/[your-username]/parkinsons-spiral-drawing-screening-aid}},
  note         = {MCA Mini-Project, [Your College Name]}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Acknowledgments

- **NewHandPD dataset:** Pereira et al. (2018)
- **Methodology reference:** PMC 2025 — subject-independent CV protocol
- **Leakage literature:** Honorio et al. (2021), PMC8604922
- **Grad-CAM:** Selvaraju et al. (2017), IEEE ICCV
