import streamlit as st
import pandas as pd
import numpy as np
import time
import random

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ---------------- CONFIG ----------------
st.set_page_config(page_title="CricScope", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
body {font-family: sans-serif;}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top,#020617,#000);
    color:white;
}
.card {
    background:#020617;
    padding:20px;
    border-radius:15px;
    border:1px solid rgba(255,255,255,0.05);
    margin-top:20px;
}
section[data-testid="stSidebar"] {
    background:#020617;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION ----------------
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("CricScope")
    st.write("Arnav Singh")

    if st.button("Dashboard"):
        st.session_state.page = "Dashboard"
    if st.button("Analysis"):
        st.session_state.page = "Analysis"
    if st.button("Simulation"):
        st.session_state.page = "Simulation"

# ---------------- HERO ----------------
if st.session_state.page == "Dashboard":
    st.markdown("## CricScope")
    st.write("Precision cricket analytics")

# ---------------- TEAM DATA ----------------
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

    # SAFE RRR
    df['rrr'] = np.where(df['balls_left']>0,(df['runs_left']*6)/df['balls_left'],0)

    df.replace([np.inf,-np.inf],np.nan,inplace=True)

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
        st.image(t1["logo"], width=90)
        st.markdown(f"### {t1['abbr']}")

    with c2:
        st.markdown("## VS")

    with c3:
        st.image(t2["logo"], width=90)
        st.markdown(f"### {t2['abbr']}")

    # ANALYZE BUTTON
    if st.button("Analyze Match"):

        runs_left = target-score
        balls_left = 120-(overs*6)
        wickets_rem = 10-wickets
        crr = score/overs
        rrr = (runs_left*6/balls_left) if balls_left>0 else 0

        input_df = pd.DataFrame({
            'batting_team':[batting_team],
            'bowling_team':[bowling_team],
            'city':[city],
            'runs_left':[runs_left],
            'balls_left':[balls_left],
            'wickets':[wickets_rem],
            'target':[target],
            'crr':[crr],
            'rrr':[rrr]
        })

        win = pipe.predict_proba(input_df)[0][1]
        lose = 1-win

        st.success(f"{t1['abbr']} Win Probability: {round(win*100)}%")

        # ---------------- WIN BAR ----------------
        st.markdown("### Win Probability")

        bar_html = f"""
        <div style="display:flex;height:30px;border-radius:20px;overflow:hidden;">
            <div style="width:{win*100}%;background:{t1['color']};text-align:center;">
                {round(win*100)}%
            </div>
            <div style="width:{lose*100}%;background:{t2['color']};text-align:center;">
                {round(lose*100)}%
            </div>
        </div>
        """

        st.markdown(bar_html, unsafe_allow_html=True)

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
