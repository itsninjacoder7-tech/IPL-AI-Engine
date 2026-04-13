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

st.title("🏏 IPL Live AI Match Engine")

# -----------------------------------
# MODEL
# -----------------------------------
@st.cache_resource
def train_model():

    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")

    df = deliveries.merge(matches, left_on='match_id', right_on='id')

    # Target
    total_df = df[df['inning'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
    total_df.rename(columns={'total_runs': 'target'}, inplace=True)

    df = df.merge(total_df, on='match_id')

    # 2nd innings only
    df = df[df['inning'] == 2]

    # Features
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

# 🎮 Slider
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
# BASIC PREDICTION
# -----------------------------------
if st.button("📊 Predict Outcome"):

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

    # Progress bars
    st.subheader("📊 Win Probability")

    colA, colB = st.columns(2)

    colA.progress(float(win))
    colB.progress(float(loss))

    colA.write(f"{batting_team}: {round(win*100)}%")
    colB.write(f"{bowling_team}: {round(loss*100)}%")

    # Graph
    st.subheader("📈 Win Probability Curve")

    overs_range = list(range(1, 21))
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
# LIVE SIMULATION
# -----------------------------------
if st.button("🎮 Start Live Simulation"):

    st.subheader("🏏 Live Match Simulation")

    commentary = st.empty()
    prob_box = st.empty()
    progress = st.progress(0)

    current_score = score
    current_wickets = wickets
    balls = 0

    events = [0,1,2,3,4,6,"W"]

    for i in range(int(balls_left)):

        if current_score >= target or current_wickets >= 10:
            break

        event = random.choice(events)

        if event == "W":
            current_wickets += 1
            text = f"❌ WICKET! {current_score}/{current_wickets}"
        else:
            current_score += event
            text = f"🏏 {event} runs! {current_score}/{current_wickets}"

        balls += 1

        runs_left_live = target - current_score
        balls_left_live = 120 - balls
        wickets_remaining_live = 10 - current_wickets

        crr_live = current_score / (balls/6 if balls > 0 else 1)
        rrr_live = (runs_left_live * 6) / balls_left_live if balls_left_live > 0 else 0

        temp_df = pd.DataFrame({
            'batting_team':[batting_team],
            'bowling_team':[bowling_team],
            'city':[city],
            'runs_left':[runs_left_live],
            'balls_left':[balls_left_live],
            'wickets':[wickets_remaining_live],
            'target':[target],
            'crr':[crr_live],
            'rrr':[rrr_live]
        })

        win_prob = pipe.predict_proba(temp_df)[0][1]

        commentary.markdown(f"### {text}")
        prob_box.metric("Win Probability", f"{round(win_prob*100)}%")

        progress.progress(min(balls/120,1.0))

        time.sleep(0.12)

    st.markdown("---")

    if current_score >= target:
        st.success(f"🎉 {batting_team} wins!")
    else:
        st.error(f"💀 {bowling_team} wins!")
