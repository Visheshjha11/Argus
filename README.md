# Argus

<p align="center">
  Machine-learning powered near-Earth asteroid risk analysis using NASA NEO data.
</p>

<p align="center">
  <a href="https://github.com/Visheshjha11/Argus">GitHub Repository</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NASA-NEO_API-0B3D91?style=for-the-badge&logo=nasa&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-Random_Forest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-Data_Processing-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Joblib-Model_Persistence-333333?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Open_Source-GitHub-181717?style=for-the-badge&logo=github&logoColor=white" />
</p>

---

## What is Argus?

**Argus** is a machine-learning based asteroid risk analysis system built around NASA's Near-Earth Object (NEO) data.

The system retrieves asteroid observations from NASA's NEO API, transforms the raw astronomical data into machine-learning features, trains and uses the trained model to generate an additional risk signal for newly retrieved asteroid observations.

Argus combines two sources of information:

* **NASA's potentially hazardous asteroid classification**
* **Argus's machine-learning risk probability**

Rather than replacing NASA's classification, the ML model acts as an additional analytical signal. This allows Argus to identify objects that may deserve additional review while keeping NASA's classification as the primary baseline.

---

## Why Argus?

Asteroid datasets contain multiple physical and observational attributes that can be used for classification.

Argus focuses on four primary features:

| Feature         | Description                                                   |
| --------------- | ------------------------------------------------------------- |
| `diameter`      | Estimated asteroid diameter                                   |
| `velocity`      | Relative velocity during close approach                       |
| `miss_distance` | Distance between the asteroid and Earth during close approach |
| `magnitude`     | Absolute magnitude of the asteroid                            |

The target variable is:

```text
hazardous
```

where:

```text
0 → Not potentially hazardous
1 → Potentially hazardous
```

The model learns relationships between these features and NASA's existing hazardous/non-hazardous classification.

---

## Core Workflow

```text
                    NASA NEO API
                         │
                         ▼
              ┌─────────────────────┐
              │   Data Collection   │
              │  Historical / Live  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Feature Extraction  │
              │                     │
              │ • Diameter          │
              │ • Velocity          │
              │ • Miss Distance     │
              │ • Magnitude         │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Data Preparation    │
              │                     │
              │ Train / Test Split  │
              │ 80% / 20%           │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Random Forest       │
              │ Classifier          │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Model Evaluation    │
              │                     │
              │ Accuracy            │
              │ Classification      │
              │ Report              │
              │ Confusion Matrix    │
              └──────────┬──────────┘
                         │
                         ▼
                  argus_model.pkl
                         │
                         ▼
              ┌─────────────────────┐
              │ Latest NEO Data     │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ ML Risk Probability │
              │+ NASA Classification│
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Argus Assessment    │
              │                     │
              │ LOW / MEDIUM / HIGH │
              │ REVIEW              │
              └─────────────────────┘
```

---

## Features

### NASA Data Integration

Argus retrieves Near-Earth Object data directly from NASA's NEO Feed API.

The project supports both:

* Historical dataset collection
* Latest available asteroid observations

The dataset collector processes NASA's seven-day API window limitation and progressively builds a larger local dataset.

### Machine Learning Classification

Argus uses a **Random Forest Classifier** implemented with scikit-learn.

The current training configuration uses:

```text
n_estimators     = 200
max_depth        = 8
min_samples_split = 4
class_weight     = balanced
random_state     = 42
```

The dataset is divided into:

```text
80% → Training
20% → Testing
```

with stratified splitting to preserve the class distribution between the training and test sets.

### Risk Probability

Instead of relying only on a binary prediction, Argus also uses the model's probability estimate:

```text
Hazardous Probability × 100
```

This produces an ML risk score between:

```text
0 – 100
```

The score is then mapped to an operational threat level.

### NASA + ML Assessment

Argus intentionally does **not** blindly override NASA's classification.

The current assessment logic treats NASA's classification as the baseline and uses the ML probability as an additional signal:

```text
NASA Classification
        +
ML Risk Probability
        ↓
Argus Assessment
```

This produces classifications such as:

```text
LOW
MEDIUM
HIGH
REVIEW
POTENTIALLY HAZARDOUS
```

This design makes the ML component an analytical decision-support layer rather than a replacement for NASA's scientific classification.

---

## Machine Learning Pipeline

### 1. Data Collection

`collect_dataset.py` retrieves historical NEO observations from NASA's API.

The collector is designed to build a large dataset and currently targets:

```text
100,000 records
```

It also resumes from an existing CSV if one is already available.

### 2. Feature Engineering

Raw NASA responses are converted into structured numerical features.

For asteroid diameter, Argus derives an average estimate:

```text
diameter = (minimum diameter + maximum diameter) / 2
```

The resulting model features are:

```python
[
    "diameter",
    "velocity",
    "miss_distance",
    "magnitude"
]
```

### 3. Model Training

`train.py` prepares the feature matrix `X` and target vector `y`, performs the train/test split, and trains the Random Forest classifier.

### 4. Model Evaluation

The training pipeline evaluates the model using:

* Accuracy
* Classification report
* Confusion matrix

This provides visibility into both overall performance and class-level behavior.

## Project Structure

```text
Argus/
│
│
├── collect_dataset.py     # builds nasa_neo_dataset.csv from the NASA feed
├── argus_core.py          # features, model bundle I/O, explanations (shared)
├── train.py               # trains + evaluates; never calls the API
├── predict.py             # live inference with per-asteroid reasoning
│
├── nasa_neo_dataset.csv
├── argus_model.pkl
├── argus_metrics.json     # written by train.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

### Clone the Repository

```bash
git clone https://github.com/Visheshjha11/Argus.git
```

### Navigate Into the Project

```bash
cd Argus
```

### Create a Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Configuration

Create a `.env` file in the project root:

```env
NASA_API_KEY=your_nasa_api_key
```

The results include:

* Asteroid name
* ML risk score
* Threat level
* Diameter
* Relative velocity
* Miss distance
* NASA hazardous classification
* Argus assessment

---

## Risk Assessment Model

Argus uses the following ML risk thresholds:

| ML Risk Score | Interpretation |
| ------------: | -------------- |
|        `< 40` | Low            |
|  `40 – 74.99` | Medium         |
|        `≥ 75` | High           |

This allows Argus to surface anomalous ML signals without treating the model as an authoritative replacement for NASA's classification.

---

## Example Output

```text
==============================================
                 ARGUS AI
==============================================

NASA DATA DATE: 2026-08-07
TOTAL ASTEROIDS DETECTED: ...

TOP RISK ASTEROIDS
----------------------------------------------

#1 Asteroid Name

    ML Risk Score    : 82.41/100
    Threat Level     : HIGH
    Diameter         : 0.42 m
    Velocity         : 98,421.32 km/h
    Miss Distance    : 2,184,321.42 km
    NASA Hazardous   : YES
    ARGUS Assessment : POTENTIALLY HAZARDOUS
```

*Example format only; values depend on the NASA data returned at runtime.*

---


## Data Source

Argus uses NASA's **Near Earth Object Web Service (NeoWs)** as its primary asteroid data source.

NASA NEO data provides the observational attributes used by Argus for dataset construction and inference.

For more information, visit the official NASA API documentation.

https://api.nasa.gov/

---

## Measured Results

Numbers below come from `python train.py` on the committed dataset
(72,701 approaches, 20,300 unique asteroids, 9.57% hazardous). The test set is
20% of **asteroids** (not rows); no asteroid appears in both train and test.
Full output is in `argus_metrics.json`.

| Metric (held-out)  | Value  |
| ------------------ | -----: |
| ROC-AUC            | 0.937  |
| PR-AUC             | 0.612 (chance 0.096) |
| Accuracy           | 92.1% (majority-class baseline 90.4%) |
| Calibration error  | 0.0065 |

Risk bands are calibrated: objects scored below 40 were NASA-flagged 6.5% of
the time, 40-75 flagged 55%, and 75+ flagged 90%.

### Known ceiling

NASA's label is defined by MOID (<= 0.05 AU) and H (<= 22). MOID is an orbital
property and is not in the NeoWs feed. For objects that are large enough but
pass far from Earth on the observed approach (28% of rows, 29% hazardous), the
model separates classes at only AUC 0.69. Swapping to gradient boosting or
adding more derived features did not change this. The way past it is more
information: orbital elements (eccentricity, semi-major axis, inclination,
perihelion) from the NeoWs lookup endpoint.

---

## Explanations

Every prediction includes sectioned reasoning:

* **NASA criteria**: size vs H <= 22 and miss distance vs 0.05 AU, with the
  diameter uncertainty range
* **Impact severity context**: velocity and size percentiles, hypothetical
  impact energy with Hiroshima / Chelyabinsk / Tunguska comparison
* **What drove the ML score**: exact Shapley attribution over Size, Speed and
  Distance, in percentage points relative to the population baseline
* **Verdict**: whether NASA and the model agree, and what to do when they do not

Velocity is shown as severity context, not as a hazard driver: it is not part
of NASA's definition and the model assigns it little weight.

---

## License

Just code, curiosity, and a few suspicious asteroids.
