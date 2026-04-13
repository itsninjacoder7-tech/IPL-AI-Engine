import random

if st.button("🎮 Start Live Simulation"):

    st.subheader("🏏 Live Match Simulation")

    commentary_box = st.empty()
    prob_box = st.empty()
    progress = st.progress(0)

    current_score = score
    current_wickets = wickets
    current_over = 0
    balls = 0

    events = [0,1,2,3,4,6,"W"]

    for i in range(int(balls_left)):

        # Stop if match ends
        if current_score >= target or current_wickets >= 10:
            break

        ball_outcome = random.choice(events)

        # Update state
        if ball_outcome == "W":
            current_wickets += 1
            text = f"❌ WICKET! Total: {current_score}/{current_wickets}"
        else:
            current_score += ball_outcome
            text = f"🏏 {ball_outcome} runs! Total: {current_score}/{current_wickets}"

        balls += 1
        current_over = balls // 6

        # Recalculate features
        runs_left_live = target - current_score
        balls_left_live = 120 - balls
        wickets_remaining_live = 10 - current_wickets
        crr_live = current_score / (balls/6 if balls > 0 else 1)
        rrr_live = (runs_left_live * 6) / balls_left_live if balls_left_live > 0 else 0

        input_df = pd.DataFrame({
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

        win_prob = pipe.predict_proba(input_df)[0][1]

        # Update UI
        commentary_box.markdown(f"### {text}")
        prob_box.metric("Win Probability", f"{round(win_prob*100)}%")

        progress.progress((balls / 120))

        time.sleep(0.15)

    # FINAL RESULT
    st.markdown("---")

    if current_score >= target:
        st.success(f"🎉 {batting_team} wins!")
    else:
        st.error(f"💀 {bowling_team} wins!")
