#  Pace – Machine Learning–Driven Candidate Attendance Forecasting  
### **AI Model for Predicting Exam Attendance & Optimizing Daily Capacity**

---

##  Project Overview  
Pace is an **AI-powered forecasting system** designed to predict *daily exam attendance* using historical behavioral data.  
The system models the full pipeline from **invitation → confirmation → attendance**, learning real-world participation patterns through supervised machine learning.

Its primary goal is to determine:  
> **“How many candidates should be invited today so that ~45 actually attend?”**

To answer this, Pace uses a multi-model ML architecture and a custom optimization layer that simulates different invitation scenarios to reach the highest possible accuracy.

---

##  Machine Learning Approach  
Pace uses **Random Forest Regression**, a powerful ensemble method that reduces variance and captures non-linear interactions within temporal data.

### ** Models Trained**
The system trains **4 separate ML models**, each learning a different stage of candidate behavior:

| Model | Predicts | Description |
|-------|----------|-------------|
| `model_davet` | Invitations | Baseline prediction for how many candidates are usually invited on similar days |
| `model_teyit` | Confirmations | Learns confirmation habits across weekdays, weeks, and months |
| `model_teyit_yok` | No-response | Predicts silent/no-feedback behavior |
| `model_katilim` | Attendance | Predicts actual show-up numbers (key operational metric) |

### ** Input Features (Feature Engineering)**
All models use:

- **Weekday (0–6)**  
- **Month (1–12)**  
- **ISO Week Number**  
- **Holiday indicator** (via Holidays API)  
- **Historical attendance distribution**

These temporal features allow the ML models to learn hidden patterns such as:
- Mid-week higher attendance trends  
- Lower attendance on Mondays  
- Holiday effect on participation  
- Monthly behavioral shifts  

---

##  AI Optimization Layer  
Machine learning alone predicts attendance based on “expected invitations.”  
But Pace introduces an additional AI-driven optimization mechanism:

### ** Dynamic Invitation Scaling**
For each date, the system **simulates invitation counts from 20 to 100** and evaluates:

```python
scaled_attendance = model_katilim * (invitation_candidate / predicted_invitation)
```

It selects the invitation number that brings predicted attendance **closest to the 45-seat capacity**.

> This transforms the system from a *pure forecast* tool into a **decision-support AI** that generates actionable operational recommendations.

---

##  Analytics & Visualization  
The backend automatically generates insights through data-driven charts:

- **Daily Attendance Breakdown**  
- **Cumulative Attendance Curve**  
- **Room Utilization Ratio (attendance/45)**  

These visualizations help non-technical users understand the model’s behavioral predictions intuitively.

---

##  Web Interface (AI-Powered Planning Dashboard)  
A minimal but functional Flask dashboard supports:

- **Date range selection**  
- **Real-time ML inference**  
- **Automatic invitation recommendation**  
- **Graph-based analysis**  
- **Excel export of predicted results**

The dashboard converts machine learning outputs into a **usable HR decision-making tool**.

---

##  Tech Stack  
### 🔹 Core AI / DS Technologies  
- **Scikit-learn** – ML modeling (Random Forest Regression)  
- **Pandas & NumPy** – Feature engineering, preprocessing  
- **Matplotlib** – Analytical visualization  
- **Holidays API** – External feature source (calendar intelligence)

### 🔹 Backend & Frontend  
- **Flask** – API + dashboard framework  
- **Jinja2** – templating  
- **HTML/CSS** – UI  
- **XlsxWriter** – exporting ML results  

---

##  Potential Enhancements (AI-Focused Roadmap)  
- **Gradient Boosting (XGBoost / LightGBM)–based models** for higher accuracy  
- **Deep learning sequence models (LSTM / Temporal CNN)** for long-term trend learning  
- **Anomaly detection** for unexpected attendance patterns  
- **Reinforcement Learning** for long-term optimal invitation policies  
- **AutoML hyperparameter tuning**  
- **Interactive HR analytics dashboard** with BI-level filtering  
- **Explainable AI (SHAP) integration**  



