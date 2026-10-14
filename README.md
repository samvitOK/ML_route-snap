# Dynamic Adaptive HMM Map-Matching Pipeline

A Python implementation of real-time map matching on road networks using a Dynamic Adaptive Hidden Markov Model (HMM) with Viterbi decoding and machine learning feature weighting (Linear SVM and KNN).

## Overview

GPS traces collected in urban areas often suffer from Gaussian sensor noise and multipath reflections. This pipeline disambiguates vehicle road corridors by dynamically adjusting buffer delay ($N$) based on confidence margins:
- **High Confidence**: Minimal delay ($N = 2$) for near-instant real-time tracking.
- **High Ambiguity / Intersections**: Stalls output buffer ($N \le 7$) to accumulate future observations before resolving the path.

## Repository Contents

- `map_matching_pipeline.py`: Main executable pipeline containing:
  - Synthetic grid road network generation.
  - Realistic GPS trajectory simulation with noise and multipath jumps.
  - Feature extraction and classifier training (Linear SVM with L1 sparsity and KNN).
  - Dynamic Adaptive HMM Viterbi decoder.
  - Buffer delay sensitivity analysis ($N \in [1..9]$).
  - Final test phase comparing unbuffered ($N=0$) vs. dynamic adaptive matching.
- `map_matching_trajectory.png`: 2D visualization of the road network, raw GPS points, ground truth, and HMM matched path.
- `buffer_error_curve.png`: Convex error curve identifying the optimal fixed buffer delay.
- `instant_vs_adaptive_comparison.png`: Performance metrics (Accuracy, Precision, Recall, F1) comparing unbuffered vs. dynamic adaptive matching.

## Requirements

- Python 3.8+
- `numpy`
- `matplotlib`
- `scikit-learn`

Install dependencies:
```bash
pip install numpy matplotlib scikit-learn
```

## Usage

Run the pipeline:
```bash
python map_matching_pipeline.py
```

All figures and evaluation metrics will be generated in the root directory.
