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
st.set_page_config(
    page_title="IPL Intelligence Engine",
    page_icon="🏏",
    layout="wide"
)

# -----------------------------------
# LOAD + PREPROCESS + TRAIN
# -----------------------------------
@st.cache_resource
def train_model():

    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")

    # Standardize team names
    teams = [
        'Chennai Super Kings','Delhi Capitals','Kings XI Punjab',
        'Kolkata Knight Riders','Mumbai Indians',
        'Rajasthan Royals','Royal Challengers Bangalore',
        'Sunrisers Hyderabad'
    ]

    matches = matches[matches['team1'].isin(teams)]
    matches = matches[matches['team2'].isin(teams)]

    # Merge datasets
    df = deliveries.merge(matches, left_on='match_id', right_on='id')

    # Total score per match
    total_score_df = df.groupby(['match_id','inning']).sum()['total_runs'].reset_index()
    total_score_df = total_score_df[total_score_df['inning'] == 1]

    # Merge target
    df = df.merge(total_score_df[['match_id','total_runs']], on='match_id')

    # Keep 2nd innings only
    df = df[df['inning'] == 2]

    # Features
    df['runs_left'] = df['total_runs'] - df['total_runs_y']
    df['balls_left'] = 126 - (df['over']*6 + df['ball'])

    df['wickets'] = df.groupby('match_id')['player_dismissed'].cumsum().fillna(0)
    df['wickets'] = 10 - df['wickets']

    df['crr'] = df['total_runs_y'] / (df['over'] + df['ball']/6)
    df['rrr'] = (df['runs_left']*6) / df['balls_left']

    # Result
    df['result'] = np.where(df['batting_team'] == df['winner'],1,0)

    # Final dataset
    final_df = df[['batting_team','bowling_team','city','runs_left',
                   'balls_left','wickets','total_runs_x','crr','rrr','result']]

    final_df = final_df.dropna()

    X = final_df.drop('result', axis=1)
    y = final_df['result']

    categorical_cols = ['batting_team','bowling_team','city']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', [col for col in X.columns if col not in categorical_cols])
    ])

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression())
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
if st.button("Predict"):

    input_df = pd.DataFrame({
        'batting_team':[batting_team],
        'bowling_team':[bowling_team],
        'city':[city],
        'runs_left':[runs_left],
        'balls_left':[balls_left],
        'wickets':[wickets_remaining],
        'total_runs_x':[target],
        'crr':[crr],
        'rrr':[rrr]
    })

    result = pipe.predict_proba(input_df)

    win = result[0][1]
    loss = result[0][0]

    st.subheader("Win Probability")

    st.write(f"{batting_team}: {round(win*100)}%")
    st.write(f"{bowling_team}: {round(loss*100)}%")
