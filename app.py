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
# TEAM DATA (PNG LOGOS - FIXED)
# -----------------------------------
team_data = {
    "Chennai Super Kings": {"logo":"https://i.imgur.com/9R6F7Yk.png","abbr":"CSK","color":"#facc15"},
    "Delhi Capitals": {"logo":"https://i.imgur.com/jwXz5sK.png","abbr":"DC","color":"#2563eb"},
    "Kings XI Punjab": {"logo":"https://i.imgur.com/Y7v2YbR.png","abbr":"PBKS","color":"#ef4444"},
    "Kolkata Knight Riders": {"logo":"https://i.imgur.com/WF7mG6R.png","abbr":"KKR","color":"#7c3aed"},
    "Mumbai Indians": {"logo":"https://i.imgur.com/8J9pKkP.png","abbr":"MI","color":"#3b82f6"},
    "Rajasthan Royals": {"logo":"https://i.imgur.com/3k8sQ4G.png","abbr":"RR","color":"#ec4899"},
    "Royal Challengers Bangalore": {"logo":"https://i.imgur.com/6n6z9Yw.png","abbr":"RCB","color":"#dc2626"},
    "Sunrisers Hyderabad": {"logo":"https://i.imgur.com/q9k9z7C.png","abbr":"SRH","color":"#f97316"}
}

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
# SIDEBAR
# -----------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

with st.sidebar:
    st.title("CricScope")
    st.write("Arnav Singh\nAI Developer")

    if st.button("Dashboard"):
        st.session_state.page = "Dashboard"
    if st.button("Match Analysis"):
        st.session_state.page = "Analysis"
    if st.button("Simulation"):
        st.session_state.page = "Simulation"

# -----------------------------------
# DASHBOARD
# -----------------------------------
if st.session_state.page == "Dashboard":
    st.title("CricScope")
    st.write("Premium Cricket Analytics Platform")

# -----------------------------------
# ANALYSIS
# -----------------------------------
if st.session_state.page == "Analysis":

    teams = list(team_data.keys())

    batting_team = st.selectbox("Batting Team", teams)
    bowling_team = st.selectbox("Bowling Team", teams)

    target = st.number_input("Target", value=180)
    score = st.number_input("Score", value=50)
    wickets = st.number_input("Wickets", 0, 10, value=2)
    overs = st.slider("Overs", 1, 20, 10)

    # LOGOS
    col1, col2, col3 = st.columns([2,1,2])

    t1 = team_data[batting_team]
    t2 = team_data[bowling_team]

    with col1:
        st.image(t1["logo"], width=100)
        st.markdown(f"### {t1['abbr']}")

    with col2:
        st.markdown("## VS")

    with col3:
        st.image(t2["logo"], width=100)
        st.markdown(f"### {t2['abbr']}")

    # CALCULATIONS
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_remaining = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    if st.button("Analyze Match"):
        input_df = pd.DataFrame({
            'batting_team':[batting_team],
            'bowling_team':[bowling_team],
            'city':['Mumbai'],
            'runs_left':[runs_left],
            'balls_left':[balls_left],
            'wickets':[wickets_remaining],
            'target':[target],
            'crr':[crr],
            'rrr':[rrr]
        })

        win = pipe.predict_proba(input_df)[0][1]

        st.success(f"Win Probability: {round(win*100)}%")

# -----------------------------------
# SIMULATION
# -----------------------------------
if st.session_state.page == "Simulation":

    st.title("Live Simulation")

    if st.button("Start Simulation"):

        score = 0
        wickets = 0

        placeholder = st.empty()

        for i in range(30):

            event = random.choice([0,1,2,4,6,"W"])

            if event == "W":
                wickets += 1
            else:
                score += event

            placeholder.markdown(f"### Score: {score}/{wickets}")

            time.sleep(0.2)
