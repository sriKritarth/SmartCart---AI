<div align="center">

# 🛒 SmartCart — AI Customer Segmentation

### Data-Driven Clustering for E-Commerce Marketing

**Understand your customers. Target smarter. Grow faster.**  
An interactive ML-powered dashboard that clusters e-commerce customers into actionable segments using unsupervised learning.

[![Live App](https://img.shields.io/badge/🚀%20Live%20App-smartcart--ai.streamlit.app-brightgreen?style=for-the-badge)](https://smartcart-ai.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter)](https://jupyter.org/)

</div>

---

## 🔗 Live Deployment

| Resource | URL |
|---|---|
| 🚀 **Live Streamlit App** | [smartcart-ai.streamlit.app](https://smartcart-ai.streamlit.app/) |
| 📓 **Research Notebook** | [`smartcart_clustering.ipynb`](https://github.com/sriKritarth/SmartCart---AI/blob/main/smartcart_clustering.ipynb) |
| 💻 **Source Code** | [`app.py`](https://github.com/sriKritarth/SmartCart---AI/blob/main/app.py) |

---

## 🧠 Overview

**SmartCart** is a data-driven **customer segmentation** project that clusters e-commerce customers into meaningful behavioral groups using **unsupervised machine learning**. It helps marketing, product, and growth teams design **targeted campaigns**, improve **retention**, and optimize **conversion** by revealing the distinct purchasing patterns and profiles that exist within a customer base.

The project ships as two complementary artifacts:

- **`smartcart_clustering.ipynb`** — the full research notebook with exploratory data analysis, feature engineering, and model evaluation
- **`app.py`** — a polished, production-ready **Streamlit dashboard** that brings the notebook pipeline to life with interactive controls and real-time Plotly visualizations, now **live at** [smartcart-ai.streamlit.app](https://smartcart-ai.streamlit.app/)

---

## ✨ Features

### 🖥️ Interactive Streamlit Dashboard
- **Upload your own CSV** from the sidebar, or run with the default dataset
- **Sidebar controls** — choose clustering method, number of clusters, max K to evaluate, and outlier cutoffs — all without touching code
- **Four tabbed views** — Overview, Cluster Analysis, K Selection, and Data Preview
- **Download results** — export the fully clustered dataset as a CSV with one click

### 📊 Visualizations (Plotly)
- **3D PCA scatter plot** — every customer plotted in PCA space, coloured by cluster, with hover tooltips showing income, spending, age, and recency
- **PCA explained variance bar chart** — see how much information each principal component captures
- **Cluster size bar chart** — customers-per-cluster at a glance
- **Income vs Total Spending scatter** — bubble chart revealing spending behaviour by segment
- **Per-metric cluster comparison** — bar chart for any numeric feature (income, age, tenure, spending, children, recency)
- **Elbow + Silhouette dual-axis chart** — combined WCSS and silhouette score curve to choose the ideal K

### 🤖 ML Pipeline (mirrors the notebook exactly)
- Missing value imputation → outlier filtering → feature engineering → one-hot encoding → StandardScaler → PCA → clustering → cluster profiling

---

## 🗂️ Project Structure

```
SmartCart---AI/
│
├── app.py                        # Streamlit dashboard (608 lines) — live at smartcart-ai.streamlit.app
├── smartcart_clustering.ipynb    # Research notebook: full EDA, feature engineering, model evaluation
├── requirements.txt              # Python dependencies
└── readme.md                     # This file
```

---

## 🔬 Methodology

The full ML pipeline is implemented identically in both the notebook and the Streamlit app.

### 1 — Data Preprocessing
- Parses `Income` as numeric; fills missing values with the **column median**
- Removes extreme outliers to stabilise clustering:
  - `Age < 90` (configurable in the dashboard sidebar)
  - `Income < 600,000` (configurable in the dashboard sidebar)

### 2 — Feature Engineering
New business-relevant features created from raw columns:

| Feature | Formula |
|---|---|
| `Age` | `2026 − Year_Birth` |
| `Customer_Tenure` | Days since `Dt_Customer` (reference: latest date in dataset) |
| `total_spendings` | Sum of all `Mnt*` spend columns |
| `total_children` | `Kidhome + Teenhome` |
| `Education → Educ_Level` | Undergraduate / Graduate / Postgraduate |
| `Marital_Status → living_with` | Partner / Alone |

### 3 — Encoding, Scaling & PCA
- **One-hot encoding** of `Education` and `living_with`
- **StandardScaler** applied to all features
- **PCA (3 components)** for dimensionality reduction — improves clustering stability and enables 3D visualisation

### 4 — Choosing the Best K
Two complementary methods are evaluated for every K from 2 to `max_k`:
- **Elbow Method** — plots WCSS/inertia; the "knee" is auto-detected using `KneeLocator`
- **Silhouette Score** — measures cluster separation quality; higher is better

The analysis recommended **4 clusters** as the optimal choice.

### 5 — Clustering Models
Both algorithms are available in the dashboard:

| Algorithm | Details |
|---|---|
| **Agglomerative (Ward)** | Default — hierarchical, Ward linkage, `n_clusters=4` |
| **K-Means** | Alternative — `n_clusters=4`, `n_init=10`, `random_state=42` |

### 6 — Cluster Profiling
After labelling, the dashboard builds a **mean-based profile table** per cluster across: customer count, income, recency, age, tenure, total spending, and total children.

---

## 👥 Cluster Profiles (Results)

Based on the 4-cluster Agglomerative (Ward) solution, four distinct customer personas emerge:

| Cluster | Persona | Income | Spending | Behaviour |
|---|---|---|---|---|
| **0** | Value-Focused Families | Lower | Lower | Higher deal purchases, more web visits, partner + children |
| **1** | Affluent Families | Higher | High | Strong omnichannel buyers (web + catalog + store), partner, low children |
| **2** | Value-Focused Singles | Lower | Lower | Similar to Cluster 0 but living alone; high web visits, fewer catalog/store purchases |
| **3** | Affluent Singles | Higher | High | High catalog/store/web purchases, low deal purchases, living alone |

> These personas translate directly into marketing actions: discount/loyalty strategies for Clusters 0 & 2, premium bundle campaigns for Clusters 1 & 3, reactivation campaigns based on recency, and channel-specific targeting (web vs. catalog vs. store).

---

## 🗃️ Dataset

The app expects a CSV named `smartcart_customers.csv` (or upload your own via the sidebar).

**Required columns:**

| Category | Columns |
|---|---|
| Demographics | `Year_Birth`, `Education`, `Marital_Status`, `Kidhome`, `Teenhome`, `Income` |
| Customer history | `Dt_Customer`, `Recency` |
| Channel behaviour | `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth`, `NumDealsPurchases` |
| Spending | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` |
| Other | `ID`, `Complain`, `Response` |

> If your dataset uses slightly different column names, update the `REQUIRED_COLUMNS` constant in `app.py` accordingly.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Dashboard** | [Streamlit](https://streamlit.io/) | Interactive web app framework |
| **Visualisation** | [Plotly Express & Graph Objects](https://plotly.com/) | Interactive 3D & 2D charts |
| **Machine Learning** | [scikit-learn](https://scikit-learn.org/) | KMeans, Agglomerative, PCA, StandardScaler, OneHotEncoder, Silhouette Score |
| **Elbow Detection** | [kneed](https://github.com/arvkevi/kneed) | Automatic knee/elbow point detection |
| **Data Processing** | [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) | Feature engineering & data wrangling |
| **Research** | [Jupyter Notebook](https://jupyter.org/) | EDA and prototype pipeline |
| **Deployment** | [Streamlit Community Cloud](https://streamlit.io/cloud) | Live app hosting |
| **Language** | Python 3.9+ | Entire codebase |

---

## 🚀 Deployment

### Live App — Streamlit Community Cloud

The app is deployed and publicly accessible at:  
👉 **[smartcart-ai.streamlit.app](https://smartcart-ai.streamlit.app/)**

No installation required — open the link and upload your CSV or use the bundled dataset.

To deploy your own instance:
1. Fork this repository on GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and connect your fork.
3. Set `app.py` as the entry point.
4. Optionally add `smartcart_customers.csv` to the repo so the app auto-loads data on launch.
5. Click **Deploy** — Streamlit Cloud installs `requirements.txt` automatically.

---

## ⚙️ Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sriKritarth/SmartCart---AI.git
cd SmartCart---AI
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Place `smartcart_customers.csv` in the same directory, or upload it from the sidebar.

### 5. Run the Research Notebook

```bash
jupyter notebook
```

Open `smartcart_clustering.ipynb` and run all cells to reproduce the full analysis.

---

## 📦 Dependencies

```txt
streamlit
pandas
numpy
plotly
scikit-learn
kneed
jupyter          # for the notebook only
```

---

## 🧭 Dashboard Walkthrough

### Sidebar Controls
| Control | Description |
|---|---|
| **Upload CSV** | Load your own customer dataset |
| **Clustering method** | Switch between Agglomerative (Ward) and KMeans |
| **Number of clusters** | Slider from 2–10 |
| **Max K for evaluation** | How many K values to test (4–15) |
| **Maximum age cutoff** | Outlier filter — removes customers older than this |
| **Maximum income cutoff** | Outlier filter — removes customers above this income |
| **Show raw / processed data** | Toggle dataset previews in the Data tab |

### Tab — Overview
- Top-level metrics: original customers, cleaned customers, cluster count, best silhouette K
- 3D PCA customer map coloured by cluster
- PCA explained variance chart
- Cluster size bar chart
- Data cleaning impact metrics

### Tab — Cluster Analysis
- Cluster profile summary table with formatted income & spending
- Income vs Total Spending scatter (bubble sized by spend)
- Per-metric average comparison bar chart (select any numeric feature)

### Tab — K Selection
- Combined WCSS / Silhouette dual-axis line chart
- Auto-detected elbow K and best silhouette K highlighted
- Full evaluation table for all tested K values

### Tab — Data Preview
- Toggle raw and processed dataset views
- One-click **Download clustered customers CSV**

---

## 🔮 Future Improvements

- Add **named persona labels** with business rule descriptions per cluster
- Export cluster labels back to CRM / marketing automation tools
- Include additional cluster quality metrics: **Davies–Bouldin**, **Calinski–Harabasz**
- Add an **`sklearn.Pipeline`** wrapper for full reproducibility
- Try additional algorithms: **GMM**, **DBSCAN**, **HDBSCAN**, **K-Prototypes** (mixed numeric/categorical)
- Add **time-series analysis** to track cluster migration over customer tenure

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: describe your change"`
4. Push to your fork: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is open source. See the repository for license details.

---

## 👨‍💻 Author

**Kritarth Srivastava (sriKritarth)**

- GitHub: [@sriKritarth](https://github.com/sriKritarth)
- LinkedIn: [kritarthsrivastava](https://www.linkedin.com/in/kritarthsrivastava/)

---

<div align="center">

🚀 [Live App](https://smartcart-ai.streamlit.app/) &nbsp;|&nbsp; 📓 [Notebook](https://github.com/sriKritarth/SmartCart---AI/blob/main/smartcart_clustering.ipynb)

⭐ Star this repo if you find it useful!

*Built with ❤️ using Python, scikit-learn, and Streamlit.*

</div>