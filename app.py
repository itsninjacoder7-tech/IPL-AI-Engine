import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
# PREMIUM CSS
# -----------------------------------
st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg,#020617,#020617,#0f172a);
    color: white;
}

/* HERO */
.hero {
    text-align: center;
    padding: 60px;
    border-radius: 20px;
    background: linear-gradient(135deg,#020617,#0f172a,#1e293b);
    box-shadow: 0px 20px 50px rgba(0,0,0,0.8);
}

.hero h1 {
    font-size: 50px;
}

.hero p {
    font-size: 18px;
    opacity: 0.8;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(135deg,#ff512f,#dd2476);
    color:white;
    border:none;
    border-radius:12px;
    height:50px;
    font-size:16px;
    width:100%;
}

/* CARDS */
.card {
    padding:18px;
    border-radius:15px;
    background: rgba(15,23,42,0.6);
    margin-top:20px;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#020617,#0f172a);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# HERO
# -----------------------------------
st.markdown("""
<div class="hero">
<h1>🏏 IPL Match Intelligence Engine</h1>
<p>AI-Powered Real-Time Win Prediction • Advanced Cricket Analytics</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------
# MODEL TRAINING
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
# SIDEBAR
# -----------------------------------
st.sidebar.title("🏏 IPL AI Engine")

st.sidebar.markdown("### 👤 Developer")
st.sidebar.write("Arnav Singh")
st.sidebar.markdown("[GitHub](https://github.com/Arnav-Singh-5080)")

# -----------------------------------
# INPUT SECTION
# -----------------------------------
st.markdown('<div class="card">🏏 Match Setup</div>', unsafe_allow_html=True)

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
    score = st.number_input("Current Score", min_value=0)
    overs = st.number_input("Overs Completed", min_value=0.1, max_value=20.0)
    wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10)

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
if st.button("🚀 Analyze Match"):

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

    st.markdown("---")

    # PROGRESS BARS
    st.subheader("📊 Win Probability")

    colA, colB = st.columns(2)

    with colA:
        st.write(f"**{batting_team}**")
        st.progress(float(win))

    with colB:
        st.write(f"**{bowling_team}**")
        st.progress(float(loss))

    st.write(f"{batting_team}: {round(win*100)}% | {bowling_team}: {round(loss*100)}%")

    # GRAPH
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

    # INSIGHT
    st.subheader("🧠 AI Match Insight")

    if rrr > crr + 3:
        st.error("Extreme pressure — unlikely chase")
    elif rrr > crr + 1.5:
        st.warning("Pressure building — required rate rising")
    elif wickets_remaining <= 3:
        st.warning("Few wickets left — risky situation")
    elif win > 0.75:
        st.success("Dominating performance")
    else:
        st.info("Match evenly balanced")

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown("---")
st.markdown("⚡ AI-Powered Cricket Analytics • Arnav Singh")
