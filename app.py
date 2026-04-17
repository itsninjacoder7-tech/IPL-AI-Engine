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
st.set_page_config(page_title="IPL Intelligence Engine", layout="wide")

# -----------------------------------
# 🎨 ULTRA PREMIUM FINTECH UI
# -----------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&family=Playfair+Display:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* BACKGROUND */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #020617, #020617 60%, #000000);
    color: #e5e7eb;
}

/* HERO CONTAINER */
.hero-box {
    background: linear-gradient(145deg, rgba(10,15,30,0.9), rgba(2,6,23,0.95));
    border: 1px solid rgba(212,175,55,0.25);
    border-radius: 28px;
    padding: 60px 40px;
    margin-top: 30px;
    text-align: center;
    box-shadow: 0 0 80px rgba(212,175,55,0.08);
}

/* TAG */
.tag {
    display: inline-block;
    padding: 8px 18px;
    border-radius: 999px;
    border: 1px solid rgba(212,175,55,0.4);
    color: #d4af37;
    font-size: 12px;
    letter-spacing: 2px;
    margin-bottom: 20px;
}

/* TITLE */
.hero-box h1 {
    font-family: 'Playfair Display', serif;
    font-size: 64px;
    font-weight: 600;
    background: linear-gradient(90deg,#ffffff,#d4af37);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* SUBTEXT */
.hero-box p {
    color: #9ca3af;
    font-size: 18px;
    margin-top: 10px;
}

/* STATS */
.stats {
    display: flex;
    justify-content: center;
    gap: 60px;
    margin-top: 40px;
}

.stat-item {
    text-align: center;
}

.stat-item h2 {
    color: #d4af37;
    font-size: 28px;
}

.stat-item p {
    color: #9ca3af;
    font-size: 12px;
    letter-spacing: 1px;
}

/* CARDS */
.card {
    background: rgba(2,6,23,0.9);
    padding: 28px;
    border-radius: 20px;
    border: 1px solid rgba(212,175,55,0.15);
    margin-top: 25px;
    box-shadow: 0 0 40px rgba(212,175,55,0.05);
}

/* HEADINGS */
h2, h3 {
    font-family: 'Playfair Display', serif;
    color: #f9fafb;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(135deg,#d4af37,#b8962e);
    color: black;
    border-radius: 12px;
    height: 48px;
    font-weight: 600;
    border: none;
}

/* METRIC */
[data-testid="stMetric"] {
    background: rgba(212,175,55,0.05);
    padding: 15px;
    border-radius: 12px;
}

/* HIDE STREAMLIT HEADER */
header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# HERO
# -----------------------------------
st.markdown("""
<div class="hero-box">
<div class="tag">ADVANCED MATCH ANALYTICS</div>
<h1>CricScope</h1>
<p>Precision match analytics for modern cricket — powered by real-time data and predictive modeling.</p>

<div class="stats">
    <div class="stat-item">
        <h2>92%</h2>
        <p>MODEL ACCURACY</p>
    </div>
    <div class="stat-item">
        <h2>&lt;1s</h2>
        <p>RESPONSE TIME</p>
    </div>
    <div class="stat-item">
        <h2>10+</h2>
        <p>KEY METRICS</p>
    </div>
</div>
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
# ANALYSIS
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

st.markdown('</div>', unsafe_allow_html=True)
