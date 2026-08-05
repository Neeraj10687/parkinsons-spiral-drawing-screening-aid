# Tests

Unit tests for the Parkinson's Spiral Drawing Screening Aid project.

## Planned Tests

- `test_data_loader.py` — patient ID extraction, image preprocessing, leakage verification
- `test_model.py` — model builds correctly, output shapes match expectations
- `test_evaluate.py` — bootstrap CI computation, McNemar test correctness
- `test_gradcam.py` — Grad-CAM heatmap shape and value range

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Critical Test: No Patient Leakage

The most important test verifies that patient-level splitting works:

```python
def test_no_patient_leakage():
    """Verify no patient appears in both train and test."""
    from src.data_loader import patient_level_split, verify_no_leakage
    # ... load test data ...
    for train_idx, test_idx in patient_level_split(images, labels, patient_ids):
        verify_no_leakage(patient_ids, train_idx, test_idx)  # asserts no overlap
```

This test MUST pass. If it fails, the entire project's evaluation is invalid.
