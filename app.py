import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# -----------------------------------
# CONFIG
# -----------------------------------
st.set_page_config(page_title="IPL AI Engine", layout="wide")
st.title("🏏 IPL AI Match Simulator")

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

overs = st.slider("🎮 Match Progress (Overs)", 1, 20, 10)

# -----------------------------------
# MATCH CONTROLS
# -----------------------------------
st.subheader("🎮 Match Controls")

colA, colB = st.columns(2)

with colA:
    aggression = st.slider("Batting Aggression", 0.0, 1.0, 0.5)

with colB:
    pitch = st.selectbox("Pitch Type", ["Normal", "Batting", "Bowling"])

# -----------------------------------
# CALCULATIONS
# -----------------------------------
runs_left = target - score
balls_left = 120 - (overs * 6)
wickets_remaining = 10 - wickets
crr = score / overs
rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

# -----------------------------------
# SMART OUTCOME ENGINE
# -----------------------------------
def smart_outcome(rrr, wickets, aggression):

    base_probs = {
        0: 0.30,
        1: 0.35,
        2: 0.10,
        3: 0.02,
        4: 0.12,
        6: 0.06,
        "W": 0.05
    }

    if rrr > 10:
        base_probs[4] += 0.05
        base_probs[6] += 0.05
        base_probs["W"] += 0.05

    if wickets <= 3:
        base_probs["W"] += 0.08

    base_probs[4] += aggression * 0.05
    base_probs[6] += aggression * 0.05
    base_probs[0] -= aggression * 0.05

    outcomes = list(base_probs.keys())
    probs = list(base_probs.values())

    return np.random.choice(outcomes, p=np.array(probs)/sum(probs))

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

    st.subheader("📊 Win Probability")

    colA, colB = st.columns(2)
    colA.progress(float(win))
    colB.progress(float(loss))

    colA.write(f"{batting_team}: {round(win*100)}%")
    colB.write(f"{bowling_team}: {round(loss*100)}%")

# -----------------------------------
# LIVE SIMULATION
# -----------------------------------
if st.button("🎮 Start Smart Simulation"):

    st.subheader("🏏 Live Simulation")

    commentary = st.empty()
    prob_box = st.empty()
    progress = st.progress(0)

    prob_history = []

    current_score = score
    current_wickets = wickets
    balls = 0

    for i in range(int(balls_left)):

        if current_score >= target or current_wickets >= 10:
            break

        event = smart_outcome(rrr, wickets_remaining, aggression)

        if event == "W":
            current_wickets += 1
            text = f"💥 BIG WICKET! {current_score}/{current_wickets}"
        else:
            current_score += event

            if event == 6:
                text = "🚀 MASSIVE SIX!"
            elif event == 4:
                text = "🔥 CRISP FOUR!"
            else:
                text = f"{event} run(s)"

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
        prob_history.append(win_prob)

        commentary.markdown(f"### {text} → {current_score}/{current_wickets}")
        prob_box.metric("Win Probability", f"{round(win_prob*100)}%")

        progress.progress(min(balls/120,1.0))

        # Momentum graph update
        if len(prob_history) > 5:
            fig, ax = plt.subplots()
            ax.plot(prob_history)
            ax.set_title("Momentum Shift")
            st.pyplot(fig)

        time.sleep(0.1)

    st.markdown("---")

    if current_score >= target:
        st.success(f"🎉 {batting_team} wins!")
    else:
        st.error(f"💀 {bowling_team} wins!")
