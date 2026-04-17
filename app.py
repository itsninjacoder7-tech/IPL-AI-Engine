import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import random

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# -----------------------------------
# CONFIG
# -----------------------------------
st.set_page_config(page_title="IPL AI Engine", layout="wide")

# -----------------------------------
# 🎨 PREMIUM UI STYLE
# -----------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #0f172a, #020617 70%);
    color: #e2e8f0;
}

/* HERO */
.hero {
    text-align: center;
    padding: 60px 20px 30px;
}

.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 56px;
    font-weight: 700;
    background: linear-gradient(90deg,#ff416c,#ff4b2b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #94a3b8;
    font-size: 18px;
}

/* CARDS */
.card {
    background: rgba(15, 23, 42, 0.65);
    padding: 28px;
    border-radius: 22px;
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.06);
    margin-top: 25px;
    box-shadow: 0 0 40px rgba(255,75,43,0.08);
    transition: 0.3s ease;
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 0 60px rgba(255,75,43,0.15);
}

/* HEADINGS */
h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(135deg,#ff416c,#ff4b2b);
    color: white;
    border-radius: 14px;
    height: 50px;
    font-weight: 600;
    border: none;
    box-shadow: 0 4px 20px rgba(255,75,43,0.4);
    transition: 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 25px rgba(255,75,43,0.6);
}

/* INPUTS */
.stSelectbox div, .stNumberInput div {
    border-radius: 12px !important;
}

/* SLIDER */
.stSlider {
    padding-top: 15px;
}

/* HIDE HEADER */
header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# HERO
# -----------------------------------
st.markdown("""
<div class="hero">
<h1>IPL AI Match Engine</h1>
<p>Real-Time Prediction • Simulation • Advanced Analytics</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------
# MODEL
# -----------------------------------
@st.cache_resource
def train_model():
    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")

    df = deliveries.merge(matches, left_on='match_id', right_on='id')

    total_df = df[df['inning'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
    total_df.rename(columns={'total_runs': 'target'}, inplace=True)

    df = df.merge(total_df, on='match_id')
    df = df[df['inning'] == 2]

    df['current_score'] = df.groupby('match_id')['total_runs'].cumsum()
    df['runs_left'] = df['target'] - df['current_score']
    df['balls_left'] = 120 - (df['over'] * 6 + df['ball'])

    df['player_dismissed'] = df['player_dismissed'].notna().astype(int)
    df['wickets'] = df.groupby('match_id')['player_dismissed'].cumsum()
    df['wickets'] = 10 - df['wickets']

    df['over'] = df['over'].replace(0, 0.1)

    df['crr'] = df['current_score'] / (df['over'] + df['ball']/6)
    df['rrr'] = (df['runs_left'] * 6) / df['balls_left']

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df['result'] = np.where(df['batting_team'] == df['winner'], 1, 0)

    final_df = df[['batting_team','bowling_team','city',
                   'runs_left','balls_left','wickets',
                   'target','crr','rrr','result']]

    final_df.dropna(inplace=True)

    X = final_df.drop('result', axis=1)
    y = final_df['result']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['batting_team','bowling_team','city']),
        ('num', 'passthrough', ['runs_left','balls_left','wickets','target','crr','rrr'])
    ])

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000))
    ])

    pipe.fit(X, y)
    return pipe

pipe = train_model()

# -----------------------------------
# INPUT CARD
# -----------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Match Setup")

teams = [
    'Chennai Super Kings','Delhi Capitals','Kings XI Punjab',
    'Kolkata Knight Riders','Mumbai Indians',
    'Rajasthan Royals','Royal Challengers Bangalore',
    'Sunrisers Hyderabad'
]

cities = ['Mumbai','Chennai','Kolkata','Delhi','Bangalore','Hyderabad','Jaipur']

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Batting Team", teams)
    bowling_team = st.selectbox("Bowling Team", teams)
    city = st.selectbox("City", cities)

with col2:
    target = st.number_input("Target", 1, value=180)
    score = st.number_input("Score", 0, value=50)
    wickets = st.number_input("Wickets", 0, 10, value=2)

overs = st.slider("Match Progress", 1, 20, 10)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# CALCULATIONS
# -----------------------------------
runs_left = target - score
balls_left = 120 - (overs * 6)
wickets_remaining = 10 - wickets
crr = score / overs
rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

# -----------------------------------
# ANALYSIS CARD
# -----------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Match Analysis")

if st.button("Analyze Match"):

    input_df = pd.DataFrame({
        'batting_team':[batting_team],
        'bowling_team':[bowling_team],
        'city':[city],
        'runs_left':[runs_left],
        'balls_left':[balls_left],
        'wickets':[wickets_remaining],
        'target':[target],
        'crr':[crr],
        'rrr':[rrr]
    })

    win = pipe.predict_proba(input_df)[0][1]

    st.progress(float(win))
    st.write(f"Win Probability: {round(win*100)}%")

    st.subheader("Win Probability Curve")

    overs_range = list(range(1, 21))
    probs = []

    for o in overs_range:
        temp_df = input_df.copy()
        temp_df['balls_left'] = 120 - (o * 6)
        probs.append(pipe.predict_proba(temp_df)[0][1])

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#020617')
    ax.set_facecolor('#0f172a')

    ax.plot(overs_range, probs, linewidth=3)
    ax.fill_between(overs_range, probs, alpha=0.2)

    ax.set_xlabel("Overs", color="white")
    ax.set_ylabel("Win Probability", color="white")
    ax.tick_params(colors='white')

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.grid(alpha=0.2)

    st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# SIMULATION CARD
# -----------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Live Simulation")

if st.button("Start Simulation"):

    commentary = st.empty()
    prob = st.empty()

    current_score = score
    current_wickets = wickets
    balls = 0

    for i in range(int(balls_left)):

        if current_score >= target or current_wickets >= 10:
            break

        event = random.choice([0,1,2,4,6,"W"])

        if event == "W":
            current_wickets += 1
            text = f"WICKET! {current_score}/{current_wickets}"
        else:
            current_score += event
            text = f"{event} runs → {current_score}/{current_wickets}"

        balls += 1

        commentary.markdown(f"### {text}")
        prob.metric("Win %", f"{round(random.random()*100)}%")

        time.sleep(0.1)

st.markdown('</div>', unsafe_allow_html=True)
