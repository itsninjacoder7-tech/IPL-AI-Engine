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
st.set_page_config(page_title="CricScope", layout="wide")

# -----------------------------------
# 🎨 PREMIUM FONT + UI
# -----------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&family=Playfair+Display:wght@600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e2e8f0;
}

.hero {
    text-align: center;
    padding: 40px;
}

.hero h1 {
    font-size: 52px;
}

.card {
    background: rgba(15, 23, 42, 0.7);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.08);
    margin-top: 25px;
}

.stButton>button {
    background: linear-gradient(135deg,#d4af37,#b8962e);
    color: black;
    border-radius: 12px;
    height: 50px;
    font-weight: 600;
}

header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# HERO
# -----------------------------------
st.markdown("""
<div class="hero">
<h1>CricScope</h1>
<p>Precision Match Analytics</p>
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
# ANALYSIS CARD (WORKING BUTTON)
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
