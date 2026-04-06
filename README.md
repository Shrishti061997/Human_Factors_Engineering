# Wrist Exoskeleton — Ergonomic Assessment with ML & IMU Sensors


---
![image](https://github.com/user-attachments/assets/f34fce84-b71a-441b-9732-86fe699ede6c)

3D Printed Passive Exoskeleton - Back

![image](https://github.com/user-attachments/assets/3a05fccf-b7bd-4cb3-8435-466a5d88098b)

3D Printed Passive Exoskeleton - Front

<img width="161" height="149" alt="image" src="https://github.com/user-attachments/assets/e60a8623-5e5f-4cd2-b507-e54c2da15b81" />

Participant - 1 doing task wearing exoskeleton and dual imu sensors 

![image](https://github.com/user-attachments/assets/bc27814c-a599-464a-aad9-66506a1d72e0)

Exoskeleton with dual IMU sensor

## Overview

Repetitive manual assembly tasks are a leading cause of work-related musculoskeletal disorders (MSDs), especially in the wrist. This project investigates whether a **custom 3D-printed passive wrist exoskeleton** can reduce ergonomic risk during a standardized 20-peg insertion task.

We combined **digital human modeling (RULA analysis)**, **wearable IMU sensor data**, **statistical testing**, and **machine learning** to evaluate wrist strain across four experimental conditions:

| Condition | Task Layout | Support |
|-----------|-------------|---------|
| 1 | Left-to-Right | No Exoskeleton |
| 2 | Diagonal | No Exoskeleton |
| 3 | Left-to-Right | With Exoskeleton |
| 4 | Diagonal | With Exoskeleton |



---

## Key Results

| Metric | Finding |
|--------|---------|
| **ML Accuracy** | 84% — XGBoost classifier distinguishing diagonal vs. straight-line movements |
| **RULA Risk Reduction** | Exoskeleton reduced RULA score by 1 level across all conditions |
| **Rotational Load (GYRO_MAG)** | 10.4% reduction with exoskeleton during diagonal tasks |
| **Movement Smoothness (JERK)** | 8.7–12.4% reduction with exoskeleton |
| **Wrist Deviation** | Statistically significant reduction (paired t-test, p < 0.05) |

### Box Plot Results

<p align="center">
  <img src="Box_plots_ML/ACC_MAG_boxplot.png" width="32%" />
  <img src="Box_plots_ML/GYRO_MAG_boxplot.png" width="32%" />
  <img src="Box_plots_ML/JERK_boxplot.png" width="32%" />
</p>
<p align="center"><em>Acceleration magnitude, rotational load, and jerk across task orientations and conditions (Exo vs. No-Exo)</em></p>

---

## 3D-Printed Exoskeleton Design

- **Material:** PLA (Polylactic Acid)
- **Printer:** Original Prusa i3 MK3 (FDM)
- **Slicer:** PrusaSlicer
- **Fabrication Time:** ~1.5 weeks
- **Design Software:** SolidWorks
- **Fit Customization:** Heat-forming process for individual hand anthropometry

The exoskeleton uses elastic bands routed between dorsal finger components and a wrist anchor to replicate natural tendon mechanics. During flexion (gripping a peg), the bands store elastic energy; upon relaxation, they passively assist finger extension back to a neutral position — reducing ulnar deviation without restricting dexterity.



---

## Project Structure

```
ML_model_codes/
├── Feature_extraction_file.py    # IMU signal preprocessing & sliding-window feature extraction
├── Wrist_ergonomics_project_ML.py # XGBoost classification, ANOVA, kinematic analysis & plotting
├── imu_features.csv              # Extracted feature dataset (24 sensor features + labels)
├── Box_plots_ML/                 # Generated result visualizations
│   ├── ACC_MAG_boxplot.png
│   ├── GYRO_MAG_boxplot.png
│   └── JERK_boxplot.png                    
```

---

## Methodology

### 1. Data Collection
- Dual **IMU sensors** (tri-axial accelerometer + gyroscope) worn on the wrist
- 6 raw channels: ACC X/Y/Z (G-force), GYRO X/Y/Z (deg/s)
- Recorded at the Machine Learning and Safety Analytics Lab (MLSA), Santa Clara University

### 2. Feature Engineering (`Feature_extraction_file.py`)
- Z-score normalization across all sensor channels
- Sliding window segmentation (window = 100 samples, step = 50)
- 24 features extracted per window: mean, std, max, min for each of the 6 sensor axes
- 3 composite kinematic metrics derived:
  - **ACC_MAG** — overall acceleration magnitude
  - **GYRO_MAG** — rotational load
  - **JERK** — movement smoothness (rate of change of acceleration)

### 3. Machine Learning (`Wrist_ergonomics_project_ML.py`)
- **Model:** XGBoost Classifier (300 estimators, max depth 6, learning rate 0.05)
- **Task:** Binary classification — diagonal vs. straight-line peg insertion
- **Result:** 84% accuracy on held-out test set (80/20 stratified split)

### 4. Statistical Analysis
- **One-way ANOVA** on ACC_MAG, GYRO_MAG, and JERK across conditions
  - Significant differences found for GYRO_MAG and JERK (p < 0.001)
- **Paired t-tests** comparing Exo vs. No-Exo wrist deviation
  - Left-to-Right: t = 15.0, p < 0.05 (significant)
  - Diagonal: t = 17.0, p < 0.05 (significant)

### 5. RULA Analysis
- Digital human model created in **Siemens Technomatix Jack**
- Pegboard workstation modeled in **SolidWorks** and imported for simulation
- RULA scores computed for worst-case postures in each condition

---

## Tech Stack

| Category | Tools |
|----------|-------|
| ML / Data Science | Python, XGBoost, scikit-learn, pandas, NumPy, SciPy |
| Visualization | Matplotlib, Seaborn |
| 3D Design & Fabrication | SolidWorks, PrusaSlicer, Prusa i3 MK3 (FDM/PLA) |
| Ergonomic Analysis | Siemens Technomatix Jack (RULA), Wearable IMU Sensors |

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/Shrishti061997/Human_Factors_Engineering.git
cd Human_Factors_Engineering/ML_model_codes

# Install dependencies
pip install pandas numpy scikit-learn xgboost scipy matplotlib seaborn

# Run feature extraction (if starting from raw data)
python Feature_extraction_file.py

# Run ML classification + statistical analysis + generate plots
python Wrist_ergonomics_project_ML.py
```

---

## Conclusions

1. **Exoskeleton works:** The 3D-printed device significantly reduced wrist deviation, jerk, and RULA scores — confirming it as an effective engineering control for assembly tasks.
2. **Layout matters:** Diagonal peg orientations consistently produced higher ergonomic risk than linear left-to-right movements, suggesting workstation layouts should prioritize linear workflows.
3. **ML validates findings:** The XGBoost classifier objectively confirmed that diagonal and straight-line tasks produce distinct kinematic signatures, demonstrating that wearable sensors + ML can identify risky movement patterns better than observation alone.

---

## Future Work

- Increase sample size for broader anthropometric coverage
- Test exoskeleton comfort over full-shift durations
- Explore flexible materials (TPU) as an alternative to rigid PLA
- Add multi-joint sensor coverage (elbow, shoulder) to detect compensatory movements

---

## References

1. McAtamney & Corlett (1993) — RULA survey method for upper limb disorders
2. Kong, Lee & Kim (2018) — Ergonomic assessment tools for repetitive tasks
3. Zhu, Zhang & Zhang (2019) — Digital human models in manufacturing
4. Grazi et al. (2020) — Passive upper-limb exoskeleton evaluation
5. Nasr, Rashid & Bouguila (2020) — Jerk-based movement smoothness metrics
6. https://www.printables.com/model/870468-improving-fine-motor-skills-designing-a-3d-fdm-pri

---

## License

This project was developed as part of academic coursework at Santa Clara University.
