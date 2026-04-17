<div align="center">

<br />

```
   ██████╗██████╗ ██╗ ██████╗███████╗ ██████╗ ██████╗ ██████╗ ███████╗
  ██╔════╝██╔══██╗██║██╔════╝██╔════╝██╔════╝██╔═══██╗██╔══██╗██╔════╝
  ██║     ██████╔╝██║██║     ███████╗██║     ██║   ██║██████╔╝█████╗
  ██║     ██╔══██╗██║██║     ╚════██║██║     ██║   ██║██╔═══╝ ██╔══╝
  ╚██████╗██║  ██║██║╚██████╗███████║╚██████╗╚██████╔╝██║     ███████╗
   ╚═════╝╚═╝  ╚═╝╚═╝ ╚═════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝     ╚══════╝
```

### **Precision Match Analytics for Modern Cricket**

<br />

![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-gold?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

<br />

> *A luxury-grade IPL match intelligence dashboard — real-time win probability powered by machine learning, wrapped in a fintech-inspired dark UI.*

<br />

---

</div>

<br />

## 📸 &nbsp;Preview

<div align="center">

| Dashboard | Match Analysis | Prediction Output |
|:---------:|:--------------:|:-----------------:|
| Team overview with live stats | Configure match state in real time | Gold-gradient win probability cards |

</div>

<br />

---

## ✨ &nbsp;Features

- 🏏 &nbsp;**Real-time Win Probability** — Logistic regression model trained on ball-by-ball IPL delivery data
- 📊 &nbsp;**Live Match State Inputs** — Target, score, overs, wickets — all configurable on the fly
- ⚡ &nbsp;**Instant Predictions** — CRR, RRR, balls remaining, wickets in hand — all computed and displayed
- 🎨 &nbsp;**Luxury Dark UI** — Glassmorphism cards, gold gradients, Cormorant Garamond typography
- 🏆 &nbsp;**All 8 IPL Teams** — CSK, MI, RCB, KKR, DC, RR, SRH, PBKS with team-colored glow effects
- 🔢 &nbsp;**Confidence Scoring** — High / Moderate / Close verdict labels with probability breakdown
- 👤 &nbsp;**Premium Sidebar** — Profile card with clickable email, LinkedIn, and GitHub links

<br />

---

## 🧠 &nbsp;How It Works

```
Ball-by-ball delivery data
        │
        ▼
Feature Engineering
  ├── Current Run Rate (CRR)
  ├── Required Run Rate (RRR)
  ├── Runs Left
  ├── Balls Remaining
  ├── Wickets in Hand
  └── Target Score
        │
        ▼
Logistic Regression Pipeline
  ├── OneHotEncoder (team names, city)
  └── LogisticRegression (max_iter=1000)
        │
        ▼
Win Probability [0.0 → 1.0]
```

The model is trained on **second-innings** deliveries only, where the chase context (target, required rate, wickets) is most predictive of match outcome.

<br />

---

## 🗂️ &nbsp;Project Structure

```
cricscope/
│
├── app.py                  # Main Streamlit application
├── matches.csv             # IPL match-level data
├── deliveries.csv          # Ball-by-ball delivery data
├── requirements.txt        # Python dependencies
└── README.md               # You are here
```

<br />

---

## 🚀 &nbsp;Getting Started

### Prerequisites

- Python 3.9 or higher
- `matches.csv` and `deliveries.csv` (IPL dataset — see [Kaggle](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020))

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/Arnav-Singh-5080/cricscope.git
cd cricscope
```

**2. Create a virtual environment** *(recommended)*

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Add the dataset**

Place `matches.csv` and `deliveries.csv` in the root directory alongside `app.py`.

**5. Run the app**

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

<br />

---

## 📦 &nbsp;Requirements

```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
```

Or install directly:

```bash
pip install streamlit pandas numpy scikit-learn
```

<br />

---

## 🏏 &nbsp;Supported Teams

| Team | Abbr | 
|------|------|
| Chennai Super Kings | CSK |
| Mumbai Indians | MI |
| Royal Challengers Bangalore | RCB |
| Kolkata Knight Riders | KKR |
| Delhi Capitals | DC |
| Rajasthan Royals | RR |
| Sunrisers Hyderabad | SRH |
| Punjab Kings | PBKS |

<br />

---

## 🎨 &nbsp;Design System

| Element | Choice |
|---------|--------|
| **Display Font** | Cormorant Garamond (serif) |
| **Body Font** | DM Sans |
| **Mono / Data Font** | DM Mono |
| **Primary Accent** | `#d4af37` Gold |
| **Background** | `#080808` Obsidian |
| **Card Style** | Glassmorphism — `backdrop-filter: blur` |
| **Logo Rendering** | `mix-blend-mode: screen` on dark container |

<br />

---

## 📈 &nbsp;Model Performance

| Metric | Value |
|--------|-------|
| **Algorithm** | Logistic Regression |
| **Training Data** | IPL 2008–2020 (ball-by-ball) |
| **Features** | 9 (6 numeric + 3 categorical) |
| **Innings Modelled** | 2nd innings only |
| **Preprocessing** | OneHotEncoder + passthrough |

> Model accuracy varies by match situation — probabilities near 50% indicate a genuinely close contest.

<br />

---

## 🔮 &nbsp;Roadmap

- [ ] Over-by-over probability graph (animated line chart)
- [ ] Historical head-to-head team stats panel
- [ ] Player performance impact overlay
- [ ] IPL 2021–2024 dataset integration
- [ ] Mobile-responsive layout
- [ ] Dark/light theme toggle

<br />

---

## 🤝 &nbsp;Contributing

Contributions are welcome. To contribute:

```bash
# Fork the repo, then:
git checkout -b feature/your-feature-name
git commit -m "feat: add your feature"
git push origin feature/your-feature-name
# Open a Pull Request
```

Please keep PRs focused — one feature or fix per PR.

<br />

---

## 📄 &nbsp;License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

<br />

---

<div align="center">

**Built by Arnav Singh**

[![Email](https://img.shields.io/badge/Email-itsarnav.singh80%40gmail.com-d4af37?style=flat-square&logo=gmail&logoColor=white)](mailto:itsarnav.singh80@gmail.com)
&nbsp;
[![LinkedIn](https://img.shields.io/badge/LinkedIn-arnav--singh-0077b5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/arnav-singh-a87847351)
&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-Arnav--Singh--5080-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Arnav-Singh-5080)

<br />

*If you found this useful, consider leaving a ⭐ on the repo.*

<br />

</div>
