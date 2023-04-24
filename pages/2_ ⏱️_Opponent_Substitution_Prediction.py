import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from predict_functions import load_predictions, get_high_risk_players


st.set_page_config(layout="wide")

minute_to_filter = st.slider('Minute of the match', 45, 90, 45)  # min: 40, max: 90, default: 45

goal_diff_to_filter = st.slider('Goal differential', -2, 2, 0) 

# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

team_to_filter = st.selectbox(
    "Select Opponent Team",
    ("Arsenal WFC", "Leicester City WFC", "Aston Villa", "Tottenham Hotspur Women", "Liverpool WFC", "Brighton & Hove Albion WFC"),
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled,
)

players = get_high_risk_player(team_to_filter, minute_to_filter, goal_diff_to_filter)

def display_player(player):
    st.markdown("""
    #### <p height:200px><center>""" + player['player_name'] + '</center></p>', unsafe_allow_html=True)
    st.markdown("""
    #### <p height:100px><center>""" + player['team'] + '</center></p>', unsafe_allow_html=True)
    st.markdown("""
    #### <p height:100px><center>""" + player['player_out_position'] + '</center></p>', unsafe_allow_html=True)


st.markdown('***')
st.markdown('## Substitute Predictions')
for i in np.arange((len(players)//3)+1):
    with st.container():
        c = st.columns(3)
        for j in np.arange(min(len(players)-3*i, 3)):
            with c[j]:
                display_player(players.iloc[3*i+j])