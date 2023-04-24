import pandas as pd
import numpy as np

from load_data import events, total, matches, lineups, columns, categorical, ohe


lineups_AstonVilla = lineups[lineups['match_id'] == matches['AstonVilla']] # for build_test


def stats_match(m):
    """
    m: minute of the game
    """
    
    # matches['AstonVilla'] = 3856030
    events_AstonVilla = events[(events['match_id'] == 3856030) & (events['minute']<m)]


    ### Players in game

    # Every minute, compute the obv of all players on the pitch

    substitued = events_AstonVilla.loc[events_AstonVilla['type.id']==19, 'player.id']

    players_on_pitch = events_AstonVilla[(events_AstonVilla['team.id'] == 746) &
                                         (~events_AstonVilla['player.id'].isin(substitued))].groupby(
        'player.id').agg(position_off = ('position.id', pd.Series.mode),
                         obv_off_match = ('obv_total_net', 'sum'),
                         xg_off_match = ('shot.statsbomb_xg', 'sum'),
                         shots_off_match = ('type.id', lambda x: (x==16).sum()),
                         fouls_won_off_match = ('type.id', lambda x: (x==21).sum()),
                         fouls_committed_off_match = ('type.id', lambda x: (x==22).sum())
                        ).reset_index().fillna({'fouls_committed_off': 0})

    players_on_pitch = players_on_pitch.astype({'player.id': 'int'}) # to display the pictures
    
    
    return players_on_pitch



def build_test(m): 
    """
    m: minute of the game
    """
    
    players_on_pitch = stats_match(m)
    
     # Add total stats
    players_on_pitch = players_on_pitch.merge(total[['player.id', 'obv']].rename(columns={'obv': 'obv_off'}),
                                              how='left')
 
    
    
    
    ### Players on the bench

    
    # Exclude all players who appear in the events table
    bench = lineups_AstonVilla[(lineups_AstonVilla['team_id'] == 746) &
                               (~lineups_AstonVilla['player_id'].isin(
                                   events.loc[(events['match_id'] == 3856030) & (events['minute']<m), 'player.id'].unique()))].drop(columns=['player_name'])
    

    ## Add the position and OBV on the players on the bench (no need to add features that we don't use to train)
    
    bench = bench.merge(total[['player.id', 'position', 'obv']].rename(columns={'player.id': 'player_id',
                                                                                'position': 'position_in',
                                                                                'obv': 'sum_obv_in'}),
                        on='player_id',
                        how='left').fillna({'sum_obv_in': 0})


    bench = bench.rename(columns={'player_id': 'substitution.replacement.id'}) 



    ### Dataframes : Matrices players on the pitch x player on the bench

    X_test = players_on_pitch.merge(bench[['substitution.replacement.id', 'position_in', 'sum_obv_in']],
                                    how='cross')
    
    
    # Data about the game
    
    X_test = events.loc[(events['match_id'] == 3856030) &
                        (events['minute']<m) &
                        (events['team.id'] == 746),
                          ['minute', 'second', 'team.id', 'opponent.id', 'GF', 'GA', 'GD', 'status',
                           'xgF', 'xgA', 'xgD']].tail(1).merge(X_test, how='cross')
    
    return X_test




def preprocessing(array_hot_encoded, df):
    
    #Convert one_hot_encoded dataframe to df
    data_hot_encoded = pd.DataFrame(array_hot_encoded,
                                    index=df.index,
                                    columns = ohe.get_feature_names_out())
    
    #Extract only the columns that didnt need to be encoded: drop categorical columns and keep numerical ones
    data_other_cols = df.drop(columns=categorical)
    
    #Concatenate the two dataframes : 
    return pd.concat([data_hot_encoded, data_other_cols], axis=1)




def predict_best_subs(model, m):
    
    X_test = build_test(m)
    
    test_array_hot_encoded = ohe.transform(X_test[categorical])
    X_test_to_predict = preprocessing(test_array_hot_encoded, X_test[columns])
    
    
    # Predict on the model already trained
    train_predictions = model.predict(X_test_to_predict)
    X_test['predicted_obv_in'] = train_predictions
    #print(X_test[(X_test['player.id']==15570) & (X_test['substitution.replacement.id']==6818)])
    
    # Add the net difference between the two players in OBV
    X_test['predicted_obv'] = X_test['predicted_obv_in'] - X_test['obv_off_match']
    
    # Rank by best OBV
    best_subs = X_test.sort_values('predicted_obv', ascending=False)
    
    
    ## Filter by position
    
    # Look all the positions taken by a player in all avaible games
    best_subs = best_subs.merge(events.groupby('player.id')['position.id'].unique(),
                                how='left',
                                left_on = 'substitution.replacement.id', right_on = 'player.id').rename(
        columns={'position.id': 'possible_position_in'})
    best_subs['possible_position_in'] = best_subs['possible_position_in'].fillna("").apply(list)

    
    # Filter
    best_subs = best_subs[[x in y for x,y in zip(best_subs['position_off'], best_subs['possible_position_in'])]
                         & (best_subs['predicted_obv']>0)]

    
    # Add names and stats for the two players
    
    sub_colums = ['player.id', 'player_name', 'xg', 'shots', 'fouls_won', 'fouls_committed']

    best_subs = best_subs.merge(total[sub_colums].rename(
        columns = {i: i+'_off' for i in sub_colums[1:]}),
                                on='player.id',
                                how='left')
    
    best_subs = best_subs.merge(total[sub_colums].rename(
        columns = {'player.id': 'substitution.replacement.id',
                   **{i: i+'_in' for i in sub_colums[1:]}}),
                                how='left')
    
    best_subs = best_subs.replace('Hayley Emma Raso', 'Hayley Raso')

    return best_subs[:5] # 5 first



def stats_player(s):
    
    col = ['Game', 'Avg last 5 games']
    
    obv = [s['obv_off_match'], s['obv_off']]
    xg = [s['xg_off_match'], s['xg_off']]
    shots = [s['shots_off_match'], s['shots_off']]
    fouls_won = [s['fouls_won_off_match'], s['fouls_won_off']]
    fouls_committed = [s['fouls_committed_off_match'], s['fouls_committed_off']]
    
    obv_row='On-Ball Value'
    
    if s['position_off'] == 1: # Goalkeeper
        return pd.DataFrame(np.array([obv]),
                             [obv_row],
                             col)
    
    elif s['position_off'] in np.arange(2,9): # Back
        return pd.DataFrame(np.array([obv, fouls_committed]),
                             [obv_row, 'Fouls Committed'],
                             col)
        
    elif s['position_off'] in np.concatenate((np.arange(9,17),np.arange(18,21))): # Midfield
        return pd.DataFrame(np.array([obv, xg, shots, fouls_won, fouls_committed]),
                             [obv_row, 'xG', 'Shots', 'Fouls Won', 'Fouls Committed'],
                             col)
        
    else: #s['position_off'].isin(np.concatenate(([17],np.arange(21, 26)))): # Forward
        return pd.DataFrame(np.array([obv, xg, shots, fouls_won]),
                             [obv_row, 'xG', 'Shots', 'Fouls Won'],
                             col)
    
    
def highlight(x, m):
    """
    x: DataFrame to be styled
    m: minute of the game
    """

    df1 = pd.DataFrame(index=x.index, columns=x.columns) # background-color: 
    
    obv_row = df1.index[0]
    col1 = df1.columns[0]
    col2 = df1.columns[1]
    
    #rewrite values by boolean masks
    df1.loc[[obv_row], col1] = np.where((x.loc[[obv_row]][col1] > x.loc[[obv_row]][col2]),
                                                           'color: green;',
                                                           'font-weight: bold; color: red;')
    
    for i in df1.index[1:]:
        
        if i in ['xG', 'Shots', 'Fouls Won']:
            if x.loc[i, col1]*(90/m) < x.loc[i, col2]:
                df1.loc[[i], col1] = 'color: red'
            elif x.loc[i, col1] > x.loc[i, col2]:
                df1.loc[[i], col1] = 'color: green'
        
        if i in ['Fouls Committed']:
            if x.loc[i, col1] > x.loc[i, col2]:
                df1.loc[[i], col1] = 'color: red'
            elif x.loc[i, col1]*(90/m) < x.loc[i, col2]:
                df1.loc[[i], col1] = 'color: green'
  
    return df1