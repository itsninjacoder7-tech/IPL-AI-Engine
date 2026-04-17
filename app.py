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
# SESSION STATE
# -----------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# -----------------------------------
# 🎨 UI
# -----------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&family=Playfair+Display:wght@600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #020617, #000000);
    color: #e5e7eb;
}

.hero-box {
    border: 1px solid rgba(212,175,55,0.25);
    border-radius: 28px;
    padding: 60px;
    text-align: center;
    margin-top: 30px;
}

.hero-box h1 {
    font-family: 'Playfair Display', serif;
    font-size: 60px;
    background: linear-gradient(90deg,#ffffff,#d4af37);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

section[data-testid="stSidebar"] {
    background: #020617;
}

.sidebar-title {
    text-align:center;
    color:#d4af37;
    font-size:24px;
}

.stButton>button {
    background: linear-gradient(135deg,#d4af37,#b8962e);
    color:black;
    border-radius:12px;
    height:45px;
    font-weight:600;
}

header {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# SIDEBAR
# -----------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">CricScope</div>', unsafe_allow_html=True)

    if st.button("Dashboard"):
        st.session_state.page = "Dashboard"

    if st.button("Analysis"):
        st.session_state.page = "Analysis"

# -----------------------------------
# TEAM DATA
# -----------------------------------
team_data = {
    "Chennai Super Kings": {"logo": "http://assets.designhill.com/design-blog/wp-content/uploads/2025/03/1-5.jpg", "abbr": "CSK", "color": "#facc15"},
    "Delhi Capitals": {"logo": "https://sp-ao.shortpixel.ai/client/to_webp,q_glossy,ret_img,w_700/https://assets.designhill.com/design-blog/wp-content/uploads/2025/03/2-4.jpg", "abbr": "DC", "color": "#2563eb"},
    "Kings XI Punjab": {"logo": "https://sp-ao.shortpixel.ai/client/to_webp,q_glossy,ret_img,w_700/https://assets.designhill.com/design-blog/wp-content/uploads/2025/03/5-4.jpg", "abbr": "PBKS", "color": "#ef4444"},
    "Kolkata Knight Riders": {"logo": "http://assets.designhill.com/design-blog/wp-content/uploads/2025/03/3-4.jpg", "abbr": "KKR", "color": "#7c3aed"},
    "Mumbai Indians": {"logo": "http://assets.designhill.com/design-blog/wp-content/uploads/2025/03/4-4.jpg", "abbr": "MI", "color": "#3b82f6"},
    "Rajasthan Royals": {"logo": "https://sp-ao.shortpixel.ai/client/to_webp,q_glossy,ret_img,w_700/https://assets.designhill.com/design-blog/wp-content/uploads/2025/03/6-4.jpg", "abbr": "RR", "color": "#ec4899"},
    "Royal Challengers Bangalore": {"logo": "https://assets.designhill.com/design-blog/wp-content/uploads/2025/03/Untitled-4.jpg", "abbr": "RCB", "color": "#dc2626"},
    "Sunrisers Hyderabad": {"logo": "http://assets.designhill.com/design-blog/wp-content/uploads/2025/03/8-4.jpg", "abbr": "SRH", "color": "#f97316"}
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
# DASHBOARD
# -----------------------------------
if st.session_state.page == "Dashboard":
    st.markdown("""
    <div class="hero-box">
    <h1>CricScope</h1>
    <p>Precision match analytics</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------
# ANALYSIS PAGE
# -----------------------------------
if st.session_state.page == "Analysis":

    teams = list(team_data.keys())

    col1, col2 = st.columns(2)

    with col1:
        batting_team = st.selectbox("Batting Team", teams)
        bowling_team = st.selectbox("Bowling Team", teams)

    with col2:
        target = st.number_input("Target", value=180)
        score = st.number_input("Score", value=50)

    overs = st.slider("Overs", 1, 20, 10)
    wickets = st.number_input("Wickets", 0, 10, 2)

    team1 = team_data[batting_team]
    team2 = team_data[bowling_team]

    colA, colB, colC = st.columns([2,1,2])

    with colA:
        st.image(team1['logo'], width=120)
        st.write(team1['abbr'])

    with colC:
        st.image(team2['logo'], width=120)
        st.write(team2['abbr'])

    # BUTTON FIX
    analyze = st.button("Analyze Match", key="analyze_btn", use_container_width=True)

    if analyze:
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        crr = score / overs
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        input_df = pd.DataFrame({
            'batting_team':[batting_team],
            'bowling_team':[bowling_team],
            'city':['Mumbai'],
            'runs_left':[runs_left],
            'balls_left':[balls_left],
            'wickets':[10-wickets],
            'target':[target],
            'crr':[crr],
            'rrr':[rrr]
        })

        win = pipe.predict_proba(input_df)[0][1]

        st.metric(team1['abbr'], f"{round(win*100)}%")
        st.metric(team2['abbr'], f"{round((1-win)*100)}%")
        st.progress(float(win))
