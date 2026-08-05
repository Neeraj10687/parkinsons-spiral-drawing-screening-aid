# Implementation Plan — Parkinson's Spiral Drawing Screening Aid

**Authors:** Neeraj N (25MCA042) & Roopak M (25MCA048)
**Dataset:** NewHandPD (66 subjects, 264 spiral images)
**Timeline:** 4 weeks, ~15-20 hours/week per person
**Started:** July 2026

---

## How to use this plan

- Each week has daily tasks with checkboxes. Check them off as you go.
- Each week has a "Definition of Done" — if you hit that, you're on track.
- Each week has a "If you're behind" section — what to cut if you're slipping.
- Don't skip ahead. Week 2 depends on Week 1, etc.
- When you hit a bug, paste it to your AI assistant (me) — don't rabbit-hole for hours.

---

## WEEK 1: Data Audit + Handcrafted Features Baseline

**Goal:** Get the data loading correctly, extract patient IDs, and train a simple handcrafted-features baseline with patient-level CV.

**Hours:** 18-20 total
**Deliverable:** Working data pipeline + logistic regression baseline accuracy number

### Day 1 (3-4 hours) — Dataset exploration

- [ ] Download NewHandPD from the official UNESP page:
      - `https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/NewHealthy/HealthySpiral.zip`
      - `https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/NewPatients/PatientSpiral.zip`
      - `https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/NewSpiral.csv`
- [ ] Unzip both image folders into `data/raw/`
- [ ] Open `NewSpiral.csv` in Excel — verify ~264 rows, 66 unique patient IDs
- [ ] Count image files in each folder — does it match the CSV?
- [ ] Look at 5 random images from each class — get a feel for the data
- [ ] Note: how many images per patient on average?

### Day 2 (4-5 hours) — Data loader

- [ ] Open Google Colab, create a new notebook called `01_data_loader.ipynb`
- [ ] Ask AI: "Write `src/data_loader.py` — load images, extract patient IDs from filenames, verify no leakage. Then explain every line."
- [ ] Run the code on Colab
- [ ] Verify outputs:
  - `images.shape` should be `(264, 256, 256, 1)` or similar
  - `labels.shape` should be `(264,)` with values 0 and 1
  - `patient_ids.shape` should be `(264,)` with 66 unique values
- [ ] **Sanity check:** Print 5 random filenames and their extracted patient IDs. Do they look right?
- [ ] Commit working code to GitHub

### Day 3 (4-5 hours) — Patient-level CV + EDA

- [ ] Ask AI: "Write the patient-level 5-fold StratifiedGroupKFold split function. Explain every line."
- [ ] Run it — verify zero patient overlap between any train and test fold
- [ ] **Critical assertion:** For each fold, `set(train_patients) & set(test_patients)` must be empty
- [ ] Create an EDA notebook (`02_eda.ipynb`):
  - Class distribution (how many healthy vs PD images?)
  - Images per patient distribution
  - Age/gender distribution (from CSV)
  - Plot 5 healthy + 5 PD spirals side by side
  - Plot image size distribution (are they all the same size?)
- [ ] Commit EDA notebook to GitHub

### Day 4 (4-5 hours) — Handcrafted features

- [ ] Ask AI: "Write `src/features.py` — extract 5 handcrafted features (radius deviation, stroke smoothness, entropy, intersection count, displacement). Explain every line."
- [ ] Run it on 10 sample images — do the feature values make sense?
  - PD spirals should have HIGHER radius deviation, HIGHER smoothness (jerkier), HIGHER entropy
  - If the values don't differentiate, something's wrong
- [ ] Compute features for all 264 images
- [ ] Save as `data/processed/features_handcrafted.npy`

### Day 5 (3-4 hours) — Handcrafted baseline training

- [ ] Ask AI: "Write the logistic regression training with patient-level 5-fold CV. Compute per-fold accuracy, then pooled OOF accuracy. Explain every line."
- [ ] Run it — get your first honest accuracy number
- [ ] **Write down the number.** This is your baseline.
- [ ] Expected: 70-80% accuracy with handcrafted features alone
- [ ] Commit results to GitHub
- [ ] Update the worklog with what you learned

### Week 1 — Definition of Done

- [ ] Data loads correctly with verified patient IDs
- [ ] Zero patient leakage confirmed across all 5 folds
- [ ] EDA notebook shows class distribution and sample images
- [ ] 5 handcrafted features extracted for all 264 images
- [ ] Logistic regression baseline trained with patient-level CV
- [ ] First honest accuracy number recorded (e.g., "75.2% ± 4.1%")

### If you're behind by end of Week 1

- If data loader is broken → cut the EDA notebook, just verify the loader works
- If handcrafted features are too complex → use only 2 features (radius deviation + smoothness) instead of 5
- If logistic regression is confusing → use sklearn's default LogisticRegression without worrying about class weights

---

## WEEK 2: CNN Training + Patient-Level CV

**Goal:** Train the CNN with patient-level 5-fold CV, get the honest CNN accuracy number, compare to the handcrafted baseline.

**Hours:** 18-20 total
**Deliverable:** CNN accuracy with confidence intervals, comparison to baseline

### Day 1 (4-5 hours) — CNN architecture

- [ ] Ask AI: "Write `src/model.py` — 3-block CNN (Conv-BN-ReLU-MaxPool ×3, GAP, Dense, Sigmoid). Explain every line."
- [ ] Test the model builds: `model = build_model(); model.summary()`
- [ ] Verify input shape (256, 256, 1) and output shape (1, 1) with sigmoid
- [ ] Count parameters — should be ~100K-500K (not millions)
- [ ] Commit to GitHub

### Day 2 (4-5 hours) — Training pipeline

- [ ] Ask AI: "Write `src/train.py` — train the CNN with 5-fold GroupKFold, collect OOF predictions. Use Adam, lr=1e-4, binary crossentropy, batch size 8, early stopping patience 10. Explain every line."
- [ ] Set up data augmentation: rotation ±10°, slight zoom, horizontal flip
- [ ] Run fold 1 only first — debug any errors before running all 5 folds
- [ ] **Common bug:** OOM error → reduce batch size to 4
- [ ] **Common bug:** Shape mismatch → check if images are grayscale (1 channel) vs RGB (3 channels)

### Day 3 (5-6 hours) — Full 5-fold training

- [ ] Run all 5 folds — this will take 1-3 hours on Colab GPU
- [ ] Save OOF predictions to `data/processed/oof_predictions_cnn.npy`
- [ ] Save per-fold metrics (accuracy, AUC, precision, recall, F1)
- [ ] **Write down the number.** This is your CNN accuracy.
- [ ] Expected: 75-85% accuracy
- [ ] Commit model weights and results to GitHub

### Day 4 (3-4 hours) — CNN vs handcrafted comparison

- [ ] Compute handcrafted OOF predictions (if not done in Week 1)
- [ ] Ask AI: "Write the McNemar test comparing CNN vs handcrafted predictions. Explain every line."
- [ ] Run it — is the difference significant?
- [ ] Expected: p > 0.05 (no significant difference) — the literature predicts handcrafted may match the CNN
- [ ] **If handcrafted beats CNN:** That's a publishable finding. Don't hide it.

### Day 5 (2-3 hours) — Buffer / catch-up

- [ ] Fix any bugs from Days 1-4
- [ ] Update worklog
- [ ] Commit everything to GitHub

### Week 2 — Definition of Done

- [ ] CNN builds and trains without errors
- [ ] 5-fold patient-level CV completed
- [ ] OOF predictions saved for both CNN and handcrafted
- [ ] McNemar test result recorded
- [ ] CNN accuracy number recorded (e.g., "82.4% ± 4.2%")

### If you're behind by end of Week 2

- If training takes too long → reduce epochs from 50 to 30
- If OOM errors persist → reduce image size from 256×256 to 128×128
- If model overfits (train acc >> test acc) → increase dropout to 0.7, add more augmentation
- If you can't get McNemar working → just report the two accuracies without the statistical test

---

## WEEK 3: Statistics + Grad-CAM + Evaluation Suite

**Goal:** Complete the full evaluation suite — bootstrap CIs, ROC/AUC, calibration, Grad-CAM. All numbers ready for the report.

**Hours:** 16-18 total
**Deliverable:** Complete evaluation results, ready for report writing

### Day 1 (3-4 hours) — Bootstrap confidence intervals

- [ ] Ask AI: "Write `bootstrap_ci()` in `src/evaluate.py` — BCa bootstrap with 2000 resamples at the patient level. Explain every line."
- [ ] Run it on CNN OOF predictions — get accuracy with 95% CI
- [ ] Run it on handcrafted OOF predictions — get accuracy with 95% CI
- [ ] Verify the format: "82.4% [76.1%, 86.7%]"
- [ ] Commit to GitHub

### Day 2 (3-4 hours) — Per-class metrics + ROC

- [ ] Compute per-class precision, recall, F1 with bootstrap CIs
- [ ] Generate confusion matrix (raw counts + row-normalized percentages)
- [ ] Plot ROC curve with AUC and Hanley-McNeil CI
- [ ] Plot precision-recall curve
- [ ] Save all plots to `reports/figures/`

### Day 3 (3-4 hours) — Calibration + Brier score

- [ ] Compute Brier score
- [ ] Generate reliability diagram (calibration plot)
- [ ] If Brier > 0.25, note that the model is poorly calibrated
- [ ] Commit plots to GitHub

### Day 4 (4-5 hours) — Grad-CAM

- [ ] Ask AI: "Write `src/gradcam.py` — Grad-CAM on the last conv layer using tf-keras-vis. Explain every line."
- [ ] Generate Grad-CAM heatmaps for 5 correctly classified + 5 incorrectly classified images
- [ ] Verify the heatmap is on the spiral stroke, not the corner
- [ ] **Common bug:** Shape mismatch error → use the raw Keras model, not a wrapper
- [ ] Compute correlation between Grad-CAM stroke activation and radius-deviation feature
- [ ] Save heatmap images to `reports/figures/gradcam/`

### Day 5 (2-3 hours) — Results compilation

- [ ] Create `03_results.ipynb` — compile all results in one place
- [ ] Generate the final results table for the report
- [ ] Update worklog with all numbers

### Week 3 — Definition of Done

- [ ] Bootstrap CIs computed for all metrics
- [ ] ROC curve, PR curve, confusion matrix generated
- [ ] Brier score and reliability diagram generated
- [ ] Grad-CAM heatmaps for 10 sample images
- [ ] All figures saved to `reports/figures/`
- [ ] Results table ready for the report

### If you're behind by end of Week 3

- If bootstrap is too slow → reduce resamples from 2000 to 1000
- If Grad-CAM is broken → skip it, just report accuracy + CIs + ROC
- If calibration is confusing → skip Brier score, just report the basics
- If you're really stuck → cut everything except accuracy + CIs + McNemar

---

## WEEK 4: Streamlit Demo + Report + Viva Rehearsal

**Goal:** Build the demo, write the report, rehearse the viva. No new code after Day 4.

**Hours:** 18-20 total
**Deliverable:** Submission-ready project

### Day 1 (3-4 hours) — Streamlit demo

- [ ] Plug trained model weights into `app/streamlit_app.py` (already written)
- [ ] Test locally — upload a spiral image, verify it returns a prediction
- [ ] Add Grad-CAM heatmap display
- [ ] Add disclaimer: "Screening aid, not diagnosis"
- [ ] Commit to GitHub

### Day 2 (3-4 hours) — Deploy to Streamlit Cloud (optional)

- [ ] Create account at share.streamlit.io
- [ ] Connect your GitHub repo
- [ ] Deploy the app
- [ ] Test the public URL
- [ ] If deployment fails → just run locally, it's fine

### Day 3 (5-6 hours) — Report writing

- [ ] Use the project specification as your outline
- [ ] Write sections in this order:
  1. Abstract (already done — copy from submission PDF)
  2. Introduction (problem statement, motivation)
  3. Literature Review (use the papers table from abstract PDF)
  4. Methodology (use Section 8 from project spec)
  5. Implementation (tech stack, architecture, training details)
  6. Results (use Week 3 results compilation)
  7. Limitations (use Section 10 from project spec)
  8. Conclusion + Future Work
  9. References (use the 7 papers from abstract PDF)
- [ ] Add figures (ROC, confusion matrix, Grad-CAM samples)
- [ ] Format per college guidelines

### Day 4 (3-4 hours) — Report editing + slide deck

- [ ] Edit report for clarity, remove fluff
- [ ] Verify all numbers are consistent (abstract matches results section)
- [ ] Update the 8-slide pitch deck with actual results (replace "expected 75-85%" with actual number)
- [ ] Create a 5-10 slide presentation for the final viva

### Day 5 (2-3 hours) — Viva rehearsal

- [ ] Read the Monday script PDF out loud once (update numbers if needed)
- [ ] Have Roopak read his crash course out loud
- [ ] Practice the handoff (Neeraj → Roopak at slide 5)
- [ ] Do a full timed run-through — aim for 8 minutes
- [ ] Review the 20 viva Q&A — make sure you can answer all Tier 1 questions

### Day 6-7 (2-3 hours) — Final polish

- [ ] Proofread the report
- [ ] Verify GitHub repo is clean (README, requirements.txt, LICENSE all present)
- [ ] Verify all figures are in the report
- [ ] Print final copies
- [ ] **NO NEW CODE.** Last-minute code changes cause 90% of demo failures.

### Week 4 — Definition of Done

- [ ] Streamlit demo works (locally or deployed)
- [ ] Report written (15-20 pages, all sections complete)
- [ ] Slide deck updated with actual results
- [ ] Viva rehearsed at least twice
- [ ] GitHub repo is clean and complete
- [ ] All deliverables ready for submission

---

## RISK FLAGS — Watch for these

### Red flags (stop and reassess)

- You're 2+ days behind on any week → cut scope immediately
- CNN accuracy is below 65% → something's wrong, debug before proceeding
- CNN accuracy is above 95% → likely data leakage, verify splits
- Grad-CAM heatmaps are all on the background → model is cheating, check augmentation
- Streamlit demo crashes on every input → test with known-good images first

### Yellow flags (slow down, but keep going)

- Handcrafted features take more than 2 days to implement → simplify to 2-3 features
- Bootstrap takes more than 10 minutes to run → reduce resamples to 1000
- Report writing is taking too long → use the project spec as a template, don't rewrite from scratch

---

## DAILY HABITS

### Every day you code

1. **Start by pulling latest from GitHub** (if working with Roopak)
2. **Write what you're about to do** in a comment or worklog
3. **Run code in small chunks** — don't write 100 lines then run
4. **When you hit a bug, paste it to AI immediately** — don't rabbit-hole
5. **Commit at end of day** — even if code is incomplete
6. **Update worklog with what you did and what's next**

### Every week

1. **Sunday night:** Review what you accomplished vs. plan
2. **If behind:** Decide what to cut BEFORE Monday, not during the week
3. **Message your partner** about progress and blockers
4. **Back up your work** — push to GitHub, save Colab notebooks to Drive

---

## WHAT TO ASK AI AT EACH STEP

Be specific. Don't say "help with the model." Say:

- ✅ "Write `src/data_loader.py` for NewHandPD. The folder structure is `data/raw/HealthySpiral/` and `data/raw/PatientSpiral/`. Filenames look like `sp1-H1.jpg`. Extract patient ID as everything after the first hyphen. Then explain every line."
- ✅ "I'm getting `ValueError: Input 0 of layer conv2d is incompatible with layer: expected ndim=4, found ndim=3` when I call model.predict(). Here's my code: [paste]. What's wrong?"
- ✅ "My CNN accuracy is 99% on training but 60% on test. It's overfitting. What regularization should I add?"

Don't ask:

- ❌ "Write the whole project for me"
- ❌ "Just fix it"
- ❌ "Is this good?" (without showing what "this" is)

---

## THE ONE RULE THAT MATTERS MOST

**If you can't explain a line of code, you will fail the viva.**

When AI writes code for you, read every line. If you don't understand a line, ask "what does this line do?" until you do. This is not optional. It is the difference between a B+ and an F.

---

## Final note

You've done the hard part — planning, getting approved, framing the project honestly. The implementation is mostly mechanical. The bugs are normal. The slow progress is normal. Just keep moving.

When in doubt, ask. When stuck, paste the error. When behind, cut scope. When tired, take a break.

You've got this.
