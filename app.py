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

st.set_page_config(page_title="CricScope", layout="wide")

# ---------------- UI ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter&family=Playfair+Display&display=swap');

html, body {font-family: Inter;}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top,#020617,#000);
    color:white;
}

.hero-box {
    padding:50px;text-align:center;
}

.hero-box h1 {
    font-family:'Playfair Display';
    font-size:60px;
    background:linear-gradient(90deg,#fff,#d4af37);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.card {
    background:#020617;
    padding:25px;
    border-radius:15px;
    margin-top:20px;
    border:1px solid rgba(212,175,55,0.1);
}

section[data-testid="stSidebar"] {
    background:#020617;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

with st.sidebar:
    st.markdown("## CricScope")
    st.write("Arnav Singh")

    if st.button("Dashboard"):
        st.session_state.page = "Dashboard"
    if st.button("Analysis"):
        st.session_state.page = "Analysis"
    if st.button("Simulation"):
        st.session_state.page = "Simulation"

# ---------------- HERO ----------------
if st.session_state.page == "Dashboard":
    st.markdown("""
    <div class="hero-box">
    <h1>CricScope</h1>
    <p>Precision cricket analytics</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- TEAM DATA (PNG FIX) ----------------
team_data = {
    "Chennai Super Kings": {"logo":"https://img1.hscicdn.com/image/upload/f_auto,t_ds_square_w_240/lsci/db/PICTURES/CMS/317000/317000.png","abbr":"CSK","color":"#facc15"},
    "Mumbai Indians": {"logo":"https://img1.hscicdn.com/image/upload/f_auto,t_ds_square_w_240/lsci/db/PICTURES/CMS/317000/317002.png","abbr":"MI","color":"#3b82f6"},
    "Royal Challengers Bangalore": {"logo":"https://img1.hscicdn.com/image/upload/f_auto,t_ds_square_w_240/lsci/db/PICTURES/CMS/317000/317004.png","abbr":"RCB","color":"#dc2626"},
    "Kolkata Knight Riders": {"logo":"https://img1.hscicdn.com/image/upload/f_auto,t_ds_square_w_240/lsci/db/PICTURES/CMS/317000/317006.png","abbr":"KKR","color":"#7c3aed"},
    "Delhi Capitals": {"logo":"https://img1.hscicdn.com/image/upload/f_auto,t_ds_square_w_240/lsci/db/PICTURES/CMS/317000/317007.png","abbr":"DC","color":"#2563eb"},
    "Rajasthan Royals": {"logo":"https://img1.hscicdn.com/image/upload/f_auto,t_ds_square_w_240/lsci/db/PICTURES/CMS/317000/317008.png","abbr":"RR","color":"#ec4899"},
    "Sunrisers Hyderabad": {"logo":"https://img1.hscicdn.com/image/upload/f_auto,t_ds_square_w_240/lsci/db/PICTURES/CMS/317000/317009.png","abbr":"SRH","color":"#f97316"},
    "Kings XI Punjab": {"logo":"https://img1.hscicdn.com/image/upload/f_auto,t_ds_square_w_240/lsci/db/PICTURES/CMS/317000/317010.png","abbr":"PBKS","color":"#ef4444"}
}

# ---------------- MODEL ----------------
@st.cache_resource
def train_model():
    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")

    df = deliveries.merge(matches, left_on='match_id', right_on='id')
    total_df = df[df['inning']==1].groupby('match_id')['total_runs'].sum().reset_index()
    total_df.rename(columns={'total_runs':'target'}, inplace=True)

    df = df.merge(total_df,on='match_id')
    df = df[df['inning']==2]

    df['current_score'] = df.groupby('match_id')['total_runs'].cumsum()
    df['runs_left'] = df['target'] - df['current_score']
    df['balls_left'] = 120 - (df['over']*6 + df['ball'])

    df['player_dismissed'] = df['player_dismissed'].notna().astype(int)
    df['wickets'] = 10 - df.groupby('match_id')['player_dismissed'].cumsum()

    df['over'] = df['over'].replace(0,0.1)

    df['crr'] = df['current_score']/(df['over']+df['ball']/6)
    df['rrr'] = (df['runs_left']*6)/df['balls_left']

    df['result'] = np.where(df['batting_team']==df['winner'],1,0)

    final_df = df[['batting_team','bowling_team','city','runs_left','balls_left','wickets','target','crr','rrr','result']].dropna()

    X = final_df.drop('result',axis=1)
    y = final_df['result']

    pre = ColumnTransformer([
        ('cat',OneHotEncoder(handle_unknown='ignore'),['batting_team','bowling_team','city']),
        ('num','passthrough',['runs_left','balls_left','wickets','target','crr','rrr'])
    ])

    pipe = Pipeline([('pre',pre),('model',LogisticRegression(max_iter=1000))])
    pipe.fit(X,y)
    return pipe

pipe = train_model()

# ---------------- ANALYSIS ----------------
if st.session_state.page == "Analysis":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    teams = list(team_data.keys())
    cities = ['Mumbai','Chennai','Kolkata','Delhi','Bangalore','Hyderabad','Jaipur']

    batting_team = st.selectbox("Batting Team", teams)
    bowling_team = st.selectbox("Bowling Team", teams)
    city = st.selectbox("City", cities)

    target = st.number_input("Target", 1, value=180)
    score = st.number_input("Score", 0, value=50)
    wickets = st.number_input("Wickets", 0, 10, value=2)
    overs = st.slider("Overs", 1, 20, 10)

    # LOGOS
    c1,c2,c3 = st.columns([2,1,2])
    t1 = team_data[batting_team]
    t2 = team_data[bowling_team]

    with c1:
        st.image(t1["logo"], width=100)
        st.markdown(f"### {t1['abbr']}")

    with c2:
        st.markdown("## VS")

    with c3:
        st.image(t2["logo"], width=100)
        st.markdown(f"### {t2['abbr']}")

    # BUTTON
    if st.button("Analyze Match"):
        runs_left = target-score
        balls_left = 120-(overs*6)
        win = pipe.predict_proba(pd.DataFrame({
            'batting_team':[batting_team],
            'bowling_team':[bowling_team],
            'city':[city],
            'runs_left':[runs_left],
            'balls_left':[balls_left],
            'wickets':[10-wickets],
            'target':[target],
            'crr':[score/overs],
            'rrr':[runs_left*6/balls_left]
        }))[0][1]

        st.success(f"Win Probability: {round(win*100)}%")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- SIMULATION ----------------
if st.session_state.page == "Simulation":

    st.markdown("## Live Simulation")

    if st.button("Start Simulation"):
        score = 0
        wickets = 0

        for i in range(30):
            event = random.choice([0,1,2,4,6,"W"])

            if event=="W":
                wickets+=1
                st.write(f"WICKET! {score}/{wickets}")
            else:
                score+=event
                st.write(f"{event} runs → {score}/{wickets}")

            time.sleep(0.2)
