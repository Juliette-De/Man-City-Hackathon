# predict which players will be substituted using static file
import pandas as pd
import numpy as np

def load_predictions():
    with open('StatsBomb/Data/predictions.csv') as data_file:    
        predictions = pd.read_csv(data_file)  
    return predictions

def get_high_risk_players(team: str, minutes: int, goal_diff: int, predictions: pd.DataFrame):
    filtered_predictions = predictions.loc[(predictions['team_name'].str.contains(team)) & (predictions['minutes'] >= minutes) & (np.abs(predictions['goal_diff'] - goal_diff) < 1.5)]
    return filtered_predictions[["player_name", "team", "player_out_position"]].drop_duplicates()[:3]


        
        
    



