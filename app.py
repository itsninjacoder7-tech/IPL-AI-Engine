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
# 🎨 PREMIUM UI
# -----------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&family=Playfair+Display:wght@600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* BACKGROUND */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #020617, #000000);
    color: #e5e7eb;
}

/* HERO */
.hero-box {
    background: linear-gradient(145deg, rgba(10,15,30,0.9), rgba(2,6,23,0.95));
    border: 1px solid rgba(212,175,55,0.25);
    border-radius: 28px;
    padding: 60px;
    text-align: center;
    margin-top: 30px;
}

.tag {
    color: #d4af37;
    letter-spacing: 2px;
    font-size: 12px;
}

.hero-box h1 {
    font-family: 'Playfair Display', serif;
    font-size: 64px;
    background: linear-gradient(90deg,#ffffff,#d4af37);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* CARDS */
.card {
    background: rgba(2,6,23,0.9);
    padding: 28px;
    border-radius: 20px;
    border: 1px solid rgba(212,175,55,0.15);
    margin-top: 25px;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid rgba(212,175,55,0.15);
}

.sidebar-title {
    font-family: 'Playfair Display', serif;
    color: #d4af37;
    text-align: center;
    font-size: 24px;
    margin-bottom: 20px;
}

.profile-box {
    text-align: center;
    margin-bottom: 25px;
}

.profile-name {
    color: white;
    font-weight: 600;
}

.profile-role {
    color: #9ca3af;
    font-size: 12px;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(135deg,#d4af37,#b8962e);
    color: black;
    border-radius: 12px;
    height: 45px;
    font-weight: 600;
}

header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# SIDEBAR
# -----------------------------------
with st.sidebar:

    st.markdown('<div class="sidebar-title">CricScope</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="profile-box">
        <div class="profile-name">Arnav Singh</div>
        <div class="profile-role">AI Developer</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🏠 Dashboard"):
        st.session_state.page = "Dashboard"

    if st.button("📊 Match Analysis"):
        st.session_state.page = "Analysis"

    if st.button("🎮 Simulation"):
        st.session_state.page = "Simulation"

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
# HERO
# -----------------------------------
if st.session_state.page == "Dashboard":
    st.markdown("""
    <div class="hero-box">
    <div class="tag">DATA-DRIVEN CRICKET ANALYTICS</div>
    <h1>CricScope</h1>
    <p>Precision match analytics for modern cricket.</p>
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
# ANALYSIS PAGE
# -----------------------------------
if st.session_state.page == "Analysis":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    teams = list(team_data.keys())
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

    # LOGO DISPLAY
    team1 = team_data[batting_team]
    team2 = team_data[bowling_team]

    colA, colB, colC = st.columns([2,1,2])

    with colA:
        st.markdown(f"""
        <div style="text-align:center;">
        <img src="{team1['logo']}" width="120"
        style="border-radius:50%; padding:10px; box-shadow:0 0 30px {team1['color']};">
        <h3 style="color:{team1['color']}">{team1['abbr']}</h3>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown("<h2 style='text-align:center;'>VS</h2>", unsafe_allow_html=True)

    with colC:
        st.markdown(f"""
        <div style="text-align:center;">
        <img src="{team2['logo']}" width="120"
        style="border-radius:50%; padding:10px; box-shadow:0 0 30px {team2['color']};">
        <h3 style="color:{team2['color']}">{team2['abbr']}</h3>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# SIMULATION PAGE
# -----------------------------------
if st.session_state.page == "Simulation":
    st.markdown("### Live Simulation Coming Soon...")
