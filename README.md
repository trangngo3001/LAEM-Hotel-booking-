# Hotel Booking Cancellation Prediction

This project uses Python to analyze hotel booking data, identify key factors
driving cancellation behavior, and build a binary classification model to
predict whether a booking will be canceled before arrival.

---

## Team Members & Responsibilities

| Full Name | Role | Responsibilities |
| :--- | :--- | :--- |
| **Ngô Thùy Trang** | Team Lead / Data Analyst / ML Engineer | Dataset sourcing, project planning, data cleaning, model training |
| **Nguyễn Quốc Đạt** | Data Engineer / ML Engineer | Dataset sourcing, data cleaning, model training |
| **Lê Tiến Hiếu** | ML Engineer / Data Engineer | Dataset sourcing, data understanding, model training |
| **Vũ Ngọc Liên** | Data Analyst / Data Engineer | Data understanding, discussion & insights analysis, README |

---

## Repository Structure
```
.
├── Data/
│   ├── hotel_bookings.csv                          <- Original dataset (raw data)
│   └── hotel_bookings_cleaned_model.csv            <- Cleaned dataset used for analysis and modeling
│
├── Notebook/
│   ├── 01_understanding_data.ipynb                 <- Step 1: Understanding dataset structure and EDA on raw data
│   ├── 02_cleaning_data.ipynb                      <- Step 2: Data cleaning and preprocessing
│   ├── 03_eda_modeling.ipynb                       <- Step 3: EDA on cleaned dataset
│   ├── 04_model_training_xgboost.ipynb             <- Step 4: Logistic Regression & XGBoost modeling
│   └── 05_conclusion_and_discussion.ipynb          <- Step 5: Final conclusions, recommendations, limitations
│
└── README.md                                       <- Project documentation
```

## Data Source

- **Platform:** Kaggle — [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- **Author:** Jesse Mostipak (2020)
- **Original paper:** Antonio, Almeida & Nunes (2019) — *Data in Brief*, Vol. 22, pp. 41–49
- **Description:** The dataset contains **119,390 rows and 32 columns**, covering hotel
  reservations from July 2015 to August 2017. Each row represents one booking and
  includes information on booking timing, guest profile, pricing, deposit policy,
  and reservation outcome.
- **Target variable:** `is_canceled` — 0 = Not Canceled (63.0%) | 1 = Canceled (37.0%)

---

## Workflow Overview

| Step | Notebook | Description |
| :--- | :--- | :--- |
| 1 | `Notebook/01_understanding_data.ipynb` | Explored dataset structure, checked data types, analyzed missing values, examined numerical and categorical distributions, and identified potential outliers from the original dataset |
| 2 | `Notebook/02_cleaning_data.ipynb` | Cleaned the raw dataset by handling missing values, removing invalid records and outliers, dropping leakage columns, and performing feature engineering |
| 3 | `Notebook/03_eda_modeling.ipynb` | Conducted exploratory data analysis on the cleaned dataset to validate data quality and uncover business insights related to booking cancellation behavior |
| 4 | `Notebook/04_model_training_xgboost.ipynb` | Built preprocessing pipelines and trained classification models including Logistic Regression and XGBoost, followed by model evaluation using Accuracy, F1-Score, and ROC-AUC |
| 5 | `Notebook/05_conclusion_and_discussion.ipynb` | Summarized findings, compared model performance, discussed business implications, limitations, and future improvements |

---

## Key Findings

**On the data:**
- **Deposit type** is the single strongest predictor: bookings with a Non Refund
  deposit cancel at **99.4%**, while No Deposit bookings cancel at only **28.4%**
- **Lead time** has a monotonic relationship with cancellation risk:
  bookings made 180+ days in advance cancel at **57.0%** vs only **20.9%** for
  bookings within 30 days
- **City Hotel** cancels at **41.7%** vs Resort Hotel at **27.8%** — reflecting
  the difference between business and leisure traveler behavior
- Guests with **0 special requests + lead time > 90 days** cancel at **63.9%**
  — the highest-volume risk profile in the dataset (32,507 bookings)
- Guests with **prior cancellation history** cancel at **94.4%** on their next booking
- **Returning guests** cancel at only **14.5%** vs **37.8%** for first-time guests

# Model Performance
## Logistic Regression

- Logistic Regression was used as the baseline classification model.
- The model achieved strong overall performance with good interpretability.
- Important predictive features included:
  - Deposit type
  - Previous cancellations
  - Lead time
  - Parking requests
  - Room type
  - Customer booking history

## XGBoost

- XGBoost was introduced as an advanced ensemble learning model to improve predictive performance.
- The model handled nonlinear relationships and feature interactions more effectively than Logistic Regression.
- XGBoost achieved better classification performance, especially in identifying canceled bookings.

## Evaluation Metrics

The models were evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC**

Because the dataset is moderately imbalanced (63/37), F1-Score and ROC-AUC were considered especially important for evaluating model quality.


---

## Important Notes

> **Data Leakage:** Columns `reservation_status` and `reservation_status_date`
> directly encode the target variable and were removed before any modeling step.
> Including them produces ~100% accuracy with no real predictive value.

> **Class Imbalance:** The target variable has a 63/37 class distribution. Therefore, F1-Score and ROC-AUC are used alongside accuracy to evaluate model performance more appropriately.

---
## How to Run

1. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib xgboost
```

2. Open the project in Jupyter Notebook, JupyterLab, or VS Code.

3. Run the notebooks in order:

```text
Notebook/01_understanding_data.ipynb
Notebook/02_cleaning_data.ipynb
Notebook/03_eda_modeling.ipynb
Notebook/04_model_training_.ipynb
Notebook/05_conclusion_and_discussion.ipynb
```

4. Make sure the dataset files are in the `Data/` folder:

```text
Data/hotel_bookings.csv
Data/hotel_bookings_cleaned_model.csv
```
___

## References

- Antonio, N., Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets.
  *Data in Brief*, 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126
- Mostipak, J. (2020). Hotel Booking Demand. *Kaggle*.
  https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python.
  *Journal of Machine Learning Research*, 12, 2825–2830.
