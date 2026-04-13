import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="IPL AI Engine", layout="wide")

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

    categorical_cols = ['batting_team','bowling_team','city']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', [col for col in X.columns if col not in categorical_cols])
    ])

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000))
    ])

    pipe.fit(X, y)
    return pipe


pipe = train_model()

# -----------------------------------
# UI HEADER
# -----------------------------------
st.title("🏏 IPL Live AI Match Simulator")

# -----------------------------------
# INPUTS
# -----------------------------------
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
    target = st.number_input("Target", min_value=1, value=180)
    score = st.number_input("Score", min_value=0, value=50)
    wickets = st.number_input("Wickets", min_value=0, max_value=10, value=2)

# 🎮 LIVE OVER SLIDER (KEY FEATURE)
overs = st.slider("🎮 Match Progress (Overs)", 1, 20, 10)

# -----------------------------------
# CALCULATIONS
# -----------------------------------
runs_left = target - score
balls_left = 120 - (overs * 6)
wickets_remaining = 10 - wickets
crr = score / overs
rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

# -----------------------------------
# PREDICTION BUTTON
# -----------------------------------
if st.button("🚀 Run Simulation"):

    progress_bar = st.progress(0)
    status = st.empty()

    # Animate loading
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    status.success("Simulation Complete")

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

    result = pipe.predict_proba(input_df)
    win = result[0][1]
    loss = result[0][0]

    # -----------------------------------
    # ANIMATED PROBABILITY
    # -----------------------------------
    st.subheader("📊 Live Win Probability")

    colA, colB = st.columns(2)

    bar1 = colA.progress(0)
    bar2 = colB.progress(0)

    for i in range(100):
        bar1.progress(min(win, i/100))
        bar2.progress(min(loss, i/100))
        time.sleep(0.01)

    colA.write(f"{batting_team}: {round(win*100)}%")
    colB.write(f"{bowling_team}: {round(loss*100)}%")

    # -----------------------------------
    # LIVE METRICS
    # -----------------------------------
    st.subheader("📈 Live Match Stats")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Runs Left", runs_left)
    m2.metric("Balls Left", balls_left)
    m3.metric("CRR", round(crr,2))
    m4.metric("RRR", round(rrr,2))

    # -----------------------------------
    # DYNAMIC GRAPH
    # -----------------------------------
    st.subheader("📉 Win Probability Trend")

    overs_range = list(range(1, overs+1))
    probs = []

    for o in overs_range:
        temp_balls_left = 120 - (o * 6)
        temp_rrr = (runs_left * 6) / temp_balls_left if temp_balls_left > 0 else 0

        temp_df = pd.DataFrame({
            'batting_team':[batting_team],
            'bowling_team':[bowling_team],
            'city':[city],
            'runs_left':[runs_left],
            'balls_left':[temp_balls_left],
            'wickets':[wickets_remaining],
            'target':[target],
            'crr':[crr],
            'rrr':[temp_rrr]
        })

        prob = pipe.predict_proba(temp_df)[0][1]
        probs.append(prob)

    fig, ax = plt.subplots()
    ax.plot(overs_range, probs)
    ax.set_xlabel("Overs")
    ax.set_ylabel("Win Probability")

    st.pyplot(fig)

    # -----------------------------------
    # SMART INSIGHT
    # -----------------------------------
    st.subheader("🧠 AI Commentary")

    if win > 0.75:
        st.success("🔥 Batting side dominating the game!")
    elif win < 0.3:
        st.error("💀 Bowling side in full control!")
    else:
        st.info("⚖️ Match is on a knife edge!")
