import pandas as pd

from load_data import events, total, lineups_AstonVilla, columns, categorical, ohe



def build_test(m):
    
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
                         obv_off = ('obv_total_net', 'sum'),
                         shots_off = ('type.id', lambda x: (x==16).sum()),
                         fouls_committed_off = ('type.id', lambda x: (x==22).sum())
                        ).reset_index().fillna({'fouls_committed_off': 0})

    players_on_pitch = players_on_pitch.astype({'player.id': 'int'}) # to display the pictures
    
    
    # Add total stats
    
    players_on_pitch = players_on_pitch.merge(total[['player.id', 'obv_total_net']].rename(
        columns={'obv_total_net': 'sum_obv_off'}),
                                              how='left')
    
    
    
    
    
    ### Players on the bench

    bench = lineups_AstonVilla[(lineups_AstonVilla['team_id'] == 746) &
                               (~lineups_AstonVilla['player_id'].isin(events_AstonVilla['player.id'].unique()))]


    ## Add the positions on the players on the bench

    bench = bench.merge(events.groupby('player.id')['position.id'].agg(pd.Series.mode), how='left',
                       left_on = 'player_id', right_on = 'player.id').rename(
        columns={'position.id': 'position_in'})

    # Players that didn't play during the 6 available games
    bench.loc[bench['player_nickname'] == 'Vicky Losada', 'position_in'] = 14
    bench.loc[bench['player_name'] == 'Alexandra MacIver', 'position_in'] = 1



    ## Add the OBV on the players on the bench (no need to add features that we don't use to train)

    bench = bench.merge(total.rename(columns={'player.id': 'player_id',
                                              'obv_total_net': 'sum_obv_in'}),
                        how='left').fillna({'sum_obv_in': 0})


    bench = bench.rename(columns={'player_id': 'substitution.replacement.id'}) 



    ### Dataframes : Matrices players on the pitch x player on the bench

    X_test = players_on_pitch.merge(bench[['substitution.replacement.id', 'position_in', 'sum_obv_in']],
                                    how='cross')

    X_test = events_AstonVilla.loc[(events_AstonVilla['team.id'] == 746),
                          ['minute', 'second', 'team.id', 'opponent.id', 'GF', 'GA', 'GD', 'status',
                           'xgF', 'xgA', 'xgD']].tail(1).merge(X_test, how='cross')
    
    return X_test




def preprocessing(array_hot_encoded, df):
    #Convert it to df
    data_hot_encoded = pd.DataFrame(array_hot_encoded,
                                    index=df.index,
                                    columns = ohe.get_feature_names_out())
    
    #Extract only the columns that didnt need to be encoded
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
    X_test['predicted_obv'] = X_test['predicted_obv_in'] - X_test['obv_off']

    
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
    
    #print(best_subs['fouls_committed'])
    
    #print(total[total['player.id'] == 6818])
    #print(best_subs[best_subs['substitution.replacement.id'] == 6818])
    
    #print(total['player.id'])
    #print(best_subs['substitution.replacement.id'])

    best_subs = best_subs.merge(total[['player.id', 'player_name', 'shots', 'fouls_committed']].rename(
        columns = {'player_name': 'player_name_off',
                   'shots': 'shots_off_total',
                   'fouls_committed': 'fouls_committed_off_total'}),
                                on='player.id',
                                how='left')
    
    best_subs = best_subs.merge(total[['player.id', 'player_name', 'shots', 'fouls_committed']].rename(
        columns = {'player.id': 'substitution.replacement.id',
                   'player_name': 'player_name_in',
                   'shots': 'shots_in',
                   'fouls_committed': 'fouls_committed_in'}),
                                how='left')
    
    print(best_subs[best_subs['substitution.replacement.id'] == 6818])

    return best_subs[:5] # 5 first
