# SmartCart – Customer Segmentation (Clustering) for E‑Commerce

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)](#)
[![Project Type](https://img.shields.io/badge/Project-Customer%20Segmentation-success)](#)

SmartCart is a data-driven **customer segmentation** project that clusters e‑commerce customers into meaningful groups using **unsupervised learning**. The goal is to help marketing, product, and growth teams design **targeted campaigns**, improve **retention**, and optimize **conversion** by understanding different purchasing behaviors and customer profiles.

> Built in Python with **pandas**, **scikit‑learn**, and visualized using **matplotlib/seaborn**.

---

## Table of Contents
- [Project Highlights](#project-highlights)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [1) Data Preprocessing](#1-data-preprocessing)
  - [2) Feature Engineering](#2-feature-engineering)
  - [3) Encoding, Scaling, and PCA](#3-encoding-scaling-and-pca)
  - [4) Choosing the Best K](#4-choosing-the-best-k)
  - [5) Clustering Models](#5-clustering-models)
  - [6) Cluster Characterization](#6-cluster-characterization)
- [Results (Cluster Profiles)](#results-cluster-profiles)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Future Improvements](#future-improvements)
- [Contact](#contact)

---

## Project Highlights
- ✅ Cleaned and prepared real-world customer data (missing values + outliers)
- ✅ Engineered business-friendly features (age, tenure, total spending, children count)
- ✅ Encoded categorical variables and standardized numeric features
- ✅ Reduced dimensionality using PCA for explainable visualization
- ✅ Identified optimal clusters using **Elbow** + **Silhouette Score**
- ✅ Compared **K-Means** and **Agglomerative (Ward)** clustering
- ✅ Built cluster summaries to support marketing personas and actions

---

## Dataset
The notebook expects a CSV file named:

- `smartcart_customers.csv`

Typical columns used in the analysis include:
- **Demographics**: `Year_Birth`, `Education`, `Marital_Status`, `Kidhome`, `Teenhome`, `Income`
- **Customer history**: `Dt_Customer`, `Recency`
- **Channel behavior**: `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth`, `NumDealsPurchases`
- **Spending**: `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`
- **Other**: `Complain`, `Response`

> If your dataset has slightly different column names, update the notebook accordingly.

---

## Methodology

### 1) Data Preprocessing
- **Missing values**
  - Filled missing `Income` with the **median**.
- **Outlier handling**
  - Removed extreme values to stabilize clustering:
    - `Age < 90`
    - `Income < 600,000`

### 2) Feature Engineering
Created business-relevant features:
- **Age** = `2026 - Year_Birth`
- **Customer_Tenure_Days** = days since `Dt_Customer` (reference date: `2026-01-01`)
- **Total_Spending** = sum of all spend categories
- **Total_Children** = `Kidhome + Teenhome`
- Standardized categories:
  - `Education → Educ_Level` (Undergraduate / Graduate / Postgraduate)
  - `Marital_Status → Living_With` (Partner / Alone)

### 3) Encoding, Scaling, and PCA
- One-hot encoded:
  - `Education`, `Living_With`
- Standardized all features with **StandardScaler**
- Applied **PCA (3 components)** for:
  - Visualization in 3D
  - Reduced noise / improved clustering stability

### 4) Choosing the Best K
Used two common methods:
- **Elbow Method** (WCSS/Inertia)
- **Silhouette Score** (cluster separation quality)

The analysis indicated **4 clusters** as a strong choice.

### 5) Clustering Models
Two algorithms were tested:
- **K-Means** (`n_clusters=4`)
- **Agglomerative Clustering** (Ward linkage, `n_clusters=4`)

The final labeling and characterization used **Agglomerative (Ward)** clusters.

### 6) Cluster Characterization
- Added cluster labels back to the engineered dataset
- Created:
  - Cluster count plot
  - Income vs Total Spending scatter plot
  - Mean-based cluster summary table

---

## Results (Cluster Profiles)
Based on the cluster summary (mean values), the four segments broadly look like:

1. **Cluster 0 — Value-focused Families (Partner + Children)**
   - Lower income
   - Higher deal purchases, fewer catalog/web purchases
   - Higher web visits (research/browsing behavior)

2. **Cluster 1 — Affluent Families (Partner + Low Children)**
   - Higher income
   - High catalog/store/web purchases (strong omnichannel buyers)
   - Lower web visits (efficient shoppers)

3. **Cluster 2 — Value-focused Singles (Mostly Alone + Children)**
   - Lower income
   - Similar “value” behavior as Cluster 0, but primarily living alone
   - Higher web visits, lower catalog/store purchases

4. **Cluster 3 — Affluent Singles (Alone + Low Children)**
   - Higher income
   - High catalog/store/web purchases
   - Low deal purchases and lower web visits

> These profiles can be translated into actionable marketing personas (discount strategy, loyalty tiers, premium bundles, reactivation campaigns, etc.).

---

## Project Structure
A clean structure you can follow in your repository:

```
smartcart-clustering/
├─ smartcart.ipynb
├─ smartcart_customers.csv         # (place inside /data if preferred)
├─ README.md
├─ requirements.txt
└─ outputs/
   ├─ figures/
   └─ cluster_summary.csv
```

---

## How to Run

### 1) Clone the repo
```bash
git clone <your-repo-url>
cd smartcart-clustering
```

### 2) Create a virtual environment (recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Run the notebook
```bash
jupyter notebook
```
Open `smartcart.ipynb` and run all cells.

---

## Tech Stack
- **Language**: Python
- **Libraries**:
  - Data: `pandas`, `numpy`
  - ML: `scikit-learn`
  - Visualization: `matplotlib`, `seaborn`
  - K selection: `kneed` (KneeLocator)

---

## Future Improvements
- Add **cluster naming** and business rules (persona labels)
- Export cluster labels back to production systems (CRM / marketing tools)
- Include **model evaluation**: Davies–Bouldin, Calinski–Harabasz
- Add **pipeline** (`sklearn.pipeline`) for reproducibility
- Build a **Streamlit dashboard** for interactive segmentation
- Try additional clustering: **GMM**, **DBSCAN**, **HDBSCAN**, **K-Prototypes** (for mixed numeric/categorical)

---

## Contact
**Author:** Kritarth Srivastava  
- GitHub: `https://github.com/sriKritarth>`  
- LinkedIn: `https://www.linkedin.com/in/kritarthsrivastava/>`

If you like this project, consider giving it a ⭐ in the repo!
