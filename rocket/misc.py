import carball
import gzip
from carball.json_parser.game import Game
from carball.analysis.analysis_manager import AnalysisManager
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.animation as animation
from matplotlib import rc
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
rc('animation', html='html5')
sns.set_context('talk')
df = pd.read_csv('shots.csv')
def get_players(game):
    df_players = pd.DataFrame()
    for x in game.players:
        df_players = (df_players.append(pd.DataFrame([x.online_id,x.name,x.is_orange]
                                                    ).T,ignore_index=True))
    df_players.columns = ['id','name','is_orange']
        
    return df_players

def get_flat_df(players, df):
    flat_df = pd.DataFrame()
    for x in players:
        frames = df['frames'].unique()
        temp = df[x]
        temp['frames'] = frames
        temp['name'] = x
        flat_df = flat_df.append(temp)
    return flat_df

def get_replay_df(replay_file):
    _json = carball.decompile_replay(replay_file)
    game = Game()
    game.initialize(loaded_json=_json)

    analysis_manager = AnalysisManager(game)
    analysis_manager.create_analysis()
    t = analysis_manager.get_json_data()
    df = analysis_manager.get_data_frame()
    df = df.reset_index().rename(columns={'index':'frames'})
    df_ball = df['ball']
    df_ball['frames'] = df['frames'].values
    players = [x.name for x in game.players]
    flat_df = get_flat_df(players, df)
    print(game.name)
    game_info = pd.DataFrame([[format(game.datetime,"%Y%m%d%H%M%S"),game.game_info.match_guid,game.name[4],game.id]],columns=['match_date','match_id','game_number','game_id'])
    player_info = get_players(game)
    players = [x.name for x in game.players]
    game_info = pd.concat((game_info,player_info),axis=1)
    game_info = game_info.fillna(method='ffill')
    return df, flat_df, game_info, players, df_ball


