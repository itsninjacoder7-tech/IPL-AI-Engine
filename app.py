import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(page_title="IPL Intelligence Engine", layout="wide")

# -----------------------------------
# TRAIN MODEL (ROBUST VERSION)
# -----------------------------------
@st.cache_resource
def train_model():

    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")

    # ✅ Fix team names (important for consistency)
    teams = [
        'Chennai Super Kings','Delhi Capitals','Kings XI Punjab',
        'Kolkata Knight Riders','Mumbai Indians',
        'Rajasthan Royals','Royal Challengers Bangalore',
        'Sunrisers Hyderabad'
    ]

    matches = matches[matches['team1'].isin(teams)]
    matches = matches[matches['team2'].isin(teams)]

    # ✅ Merge datasets
    df = deliveries.merge(matches, left_on='match_id', right_on='id')

    # -----------------------------------
    # TARGET (1st innings total)
    # -----------------------------------
    total_df = df[df['inning'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
    total_df.rename(columns={'total_runs': 'target'}, inplace=True)

    df = df.merge(total_df, on='match_id')

    # -----------------------------------
    # SECOND INNINGS ONLY
    # -----------------------------------
    df = df[df['inning'] == 2]

    # -----------------------------------
    # FEATURE ENGINEERING
    # -----------------------------------

    # Current score
    df['current_score'] = df.groupby('match_id')['total_runs_x'].cumsum()

    # Runs left
    df['runs_left'] = df['target'] - df['current_score']

    # Balls left
    df['balls_left'] = 120 - (df['over'] * 6 + df['ball'])

    # Wickets fallen
    df['player_dismissed'] = df['player_dismissed'].notnull().astype(int)
    df['wickets'] = df.groupby('match_id')['player_dismissed'].cumsum()
    df['wickets'] = 10 - df['wickets']

    # Avoid division errors
    df['over'] = df['over'].replace(0, 0.1)

    # CRR
    df['crr'] = df['current_score'] / (df['over'] + df['ball']/6)

    # RRR
    df['rrr'] = (df['runs_left'] * 6) / df['balls_left']

    # Clean infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Result
    df['result'] = np.where(df['batting_team'] == df['winner'], 1, 0)

    # -----------------------------------
    # FINAL DATASET
    # -----------------------------------
    final_df = df[[
        'batting_team','bowling_team','city',
        'runs_left','balls_left','wickets',
        'target','crr','rrr','result'
    ]]

    final_df.dropna(inplace=True)

    # -----------------------------------
    # MODEL
    # -----------------------------------
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
# UI
# -----------------------------------
st.title("🏏 IPL Match Intelligence Engine")

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
    target = st.number_input("Target", min_value=1)
    score = st.number_input("Score", min_value=0)
    overs = st.number_input("Overs", min_value=0.1, max_value=20.0)
    wickets = st.number_input("Wickets", min_value=0, max_value=10)

# -----------------------------------
# CALCULATIONS
# -----------------------------------
runs_left = target - score
balls_left = 120 - (overs * 6)
wickets_remaining = 10 - wickets
crr = score / overs if overs > 0 else 0
rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

# -----------------------------------
# PREDICTION
# -----------------------------------
if st.button("🚀 Predict"):

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

    with colA:
        st.metric(batting_team, f"{round(win*100)}%")

    with colB:
        st.metric(bowling_team, f"{round(loss*100)}%")

    # Insight
    st.subheader("🧠 AI Insight")

    if rrr > crr + 2:
        st.error("High pressure situation")
    elif wickets_remaining <= 3:
        st.warning("Few wickets left")
    elif win > 0.7:
        st.success("Batting team dominating")
    else:
        st.info("Match balanced")

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown("---")
st.markdown("⚡ Built by Arnav Singh • AI-Powered Cricket Analytics")
