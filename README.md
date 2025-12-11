# GaLAM

C++ implementation of the **GaLAM** (Geometric and Local Affine Matching) outlier detection algorithm from the paper:

**"GaLAM: Two-Stage Outlier Detection Algorithm"** by Xiaojun Lu, Zhe Yan, Ziyun Fan (IEEE Access 2025)

**Paper:** [https://ieeexplore.ieee.org/document/10967479](https://ieeexplore.ieee.org/document/10967479)

**Report:** [Implementing the GaLAM Outlier Detection Algorithm](report/Implementing th GaLAM Outlier Detection Algorithm.pdf)

**Demo Presentation:** [Watch on Google Drive](https://drive.google.com/file/d/1BtqGpWlnIshK1C5TMOqit3nsoO-nU3aY/view?usp=drive_link)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Usage](#usage)
- [Dataset](#dataset)
- [Output](#output)
- [Project Structure](#project-structure)
- [Results](#results)
- [References](#references)
- [Authors](#authors)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

GaLAM is a two-stage outlier detection algorithm for feature matching that combines local affine verification with global geometric consistency. It addresses limitations of existing methods like ratio test, RANSAC, and AdaLAM by introducing geometric constraints for filtering incorrect matches.

### Algorithm Pipeline

```
Input Images → Keypoint Detection → Descriptor Matching
                                          ↓
                            ┌─────────────────────────────┐
                            │          GaLAM              │
                            │  ┌─────────────────────┐    │
                            │  │ Local Affine        │    │
                            │  │ Verification        │    │
                            │  │ • Seed Selection    │    │
                            │  │ • Neighborhood      │    │
                            │  │   Selection         │    │
                            │  │ • Affine            │    │
                            │  │   Verification      │    │
                            │  └──────────┬──────────┘    │
                            │             ↓               │
                            │  ┌─────────────────────┐    │
                            │  │ Global Geometric    │    │
                            │  │ Consistency         │    │
                            │  │ • Fit Fundamental   │    │
                            │  │   Matrix            │    │
                            │  │ • Evaluate Seeds    │    │
                            │  │ • Filter Outliers   │    │
                            │  └──────────┬──────────┘    │
                            │             ↓               │
                            │  ┌─────────────────────┐    │
                            │  │ Final Threshold     │    │
                            │  └─────────────────────┘    │
                            └─────────────────────────────┘
                                          ↓
                                  Filtered Matches
```

---

## Key Features

- **Two-stage filtering** — Combines local affine verification with global geometric consistency
- **Seed point selection** — Uses bidirectional nearest neighbor matching with non-maximum suppression
- **Local neighborhood verification** — Validates matches using affine transformation constraints
- **Global geometric consistency** — Fits fundamental matrix using RANSAC across all seed points
- **Multiple detector support** — Works with SIFT, ORB, and AKAZE feature detectors
- **Comprehensive benchmarking** — Includes comparison against NN+RT, RANSAC, and GMS methods
---
## Usage

### Requirements
Requires C++17, OpenCV, and the OpenCV extra modules.

### Running Benchmark Tests

Run the full benchmark suite on the Oxford Affine Dataset:

```bash
cd build

# Run with Oxford dataset
./test_galam ../data 
```

### Demo Mode

Test GaLAM on two specific images:

```bash
./Galam match path/to/image1.jpg path/to/image2.jpg
```

This generates visualization images showing:

| Output File | Description |
|-------------|-------------|
| `galam_1_initial.jpg` | All initial matches |
| `galam_2_seeds.jpg` | Selected seed points |
| `galam_3_stage1.jpg` | Matches after local affine verification |
| `galam_4_final.jpg` | Final filtered matches |

## Dataset

This implementation uses the [Oxford Affine Dataset](http://www.robots.ox.ac.uk/~vgg/research/affine/), which includes 8 scene categories:

| Category | Scenes | Transformation Type |
|----------|--------|---------------------|
| Viewpoint | `graf`, `wall` | Camera angle changes |
| Blur | `bikes`, `trees` | Focus/motion blur |
| Zoom+Rotation | `bark`, `boat` | Scale and rotation |
| Illumination | `leuven` | Lighting changes |
| Compression | `ubc` | JPEG artifacts |

Each scene contains 6 images with ground truth homographies (`H1to2p`, `H1to3p`, etc.).

### Dataset Structure

```
data/
├── bark/
│   ├── img1.ppm
│   ├── img2.ppm
│   ├── ...
│   ├── H1to2p
│   ├── H1to3p
│   └── ...
├── bikes/
├── boat/
├── graf/
├── leuven/
├── trees/
├── ubc/
└── wall/
```

---

## Output

### Console Output
Console output is formatted as follows:

```
Scene   Pair    Method  Corr    AvgErr  Inlier% Time(ms)
------------------------------------------------------------
bark    1-2     NN+RT   1245    45.32   68.42   0.01
bark    1-2     RANSAC  1198    42.15   71.33   18.45
bark    1-2     GMS     1302    38.67   65.21   28.34
bark    1-2     GaLAM   534     8.92    91.24   156.78
...

===== HOMOGRAPHY ESTIMATION =====
Method      View %H.E   View AP   Light %H.E  Light AP  ...
----------------------------------------------------------------
NN+RT       30.62       56.19     76.62       85.97     ...
RANSAC      35.84       65.35     87.36       99.82     ...
GMS         33.59       64.30     67.12       76.53     ...
GaLAM       41.17       72.13     89.18       99.07     ...

===== SUMMARY =====
Method      Avg Corr    Avg Error   Inlier %    Runtime(ms)
------------------------------------------------------------
NN+RT       1121.0      67.26       72.49       0.00
RANSAC      1167.3      70.08       75.33       19.16
GMS         1272.5      34.59       70.84       31.68
GaLAM       547.6       11.21       89.59       180.23
```

### Metrics
The following metrics can be used for evaluation:

| Metric | Description |
|--------|-------------|
| **Correspondences** | Number of matches after filtering |
| **Average Error** | Mean reprojection error in pixels |
| **Inlier %** | Percentage of matches with reprojection error < 3px |
| **%H.E** | Percentage of matches with reprojection error < 1px |
| **Runtime** | Processing time in milliseconds |

### CSV Output

Results are saved to CSV for further analysis:

```csv
Scene,Pair,Method,Correspondences,AvgError,Inlier%,H.E%,Runtime_ms
bark,1-2,NN+RT,1245,45.32,68.42,32.15,0.01
bark,1-2,RANSAC,1198,42.15,71.33,35.67,18.45
...
```

---

## Project Structure

```
galam/
├── 📄 CMakeLists.txt      # Build configuration
├── 📄 README.md           # This file
├── 📄 galam.h             # GaLAM class declaration
├── 📄 galam.cpp           # GaLAM algorithm implementation
├── 📄 match_test.h        # Testing framework declaration
├── 📄 match_test.cpp      # Testing implementation
├── 📄 main.cpp            # Entry point and CLI
├── 📂 data/               # Oxford Affine Dataset
├── 📂 report/             # Final report
```

## Results

Our implementation achieves results fairly consistent with the paper:

| Metric | GaLAM | RANSAC | GMS | NN+RT |
|--------|-------|--------|-----|-------|
| **Avg Correspondences** | 547.6 | 1167.3 | 1272.5 | 1121 |
| **Avg Error (px)** | 11.21 | 70.08 | 34.59 | 67.26 |
| **Inlier %** | 89.59 | 75.33 | 70.84 | 72.49 |
| **Runtime (ms)** | 180.23 | 19.16 | 31.68 | 0.00 |

**Key Findings:**
- **Highest accuracy**: GaLAM achieves ~89% inlier rate vs ~72-75% for other methods
- **Lowest error**: Average projection error of ~11px vs ~35-70px for others
- **Cleaner matches**: Produces fewer but higher-quality correspondences
- **Trade-off**: Higher runtime (~180ms) due to two-stage verification

---

## References

```bibtex
@ARTICLE{10967479,
  author={Lu, Xiaojun and Yan, Zhe and Fan, Ziyun},
  journal={IEEE Access}, 
  title={GaLAM: Two-Stage Outlier Detection Algorithm}, 
  year={2025},
  volume={13},
  number={},
  pages={76135-76144},
  keywords={Matched filters;Image matching;Silicon;Optical filters;Filtering algorithms;Anomaly detection;Robustness;Accuracy;Three-dimensional displays;Estimation;Image matching;outlier detection;GaLAM},
  doi={10.1109/ACCESS.2025.3561823}}
```

**Additional Resources:**
- [Oxford Affine Dataset](http://www.robots.ox.ac.uk/~vgg/research/affine/)
- [OpenCV Documentation](https://docs.opencv.org/)

---

## Authors

**Implementation by:**

| Name | Role |
|------|------|
| **Neha Kotwal** | Developer |
| **Anoop Prasad** | Developer |
| **Yu Dinh** | Developer |

**University of Washington Bothell**  
**CSS 587: Computer Vision**  
**Winter 2025**

---

## License

This project is for educational purposes as part of CSS 587 coursework.

---

## Acknowledgments

- Original GaLAM paper authors: Xiaojun Lu, Zhe Yan, Ziyun Fan
- [Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/) for the Oxford Affine Dataset
- OpenCV library contributors