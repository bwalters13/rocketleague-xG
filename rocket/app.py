import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
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
from misc import get_replay_df


boost_locs = pd.DataFrame([[-3072.0, -4096.0, 73.0, 'big'],
                           [ 3072.0, -4096.0, 73.0, 'big'],
                           [-3584.0,     0.0, 73.0, 'big'],
                           [ 3584.0,     0.0, 73.0, 'big'],
                           [-3072.0,  4096.0, 73.0, 'big'],
                           [ 3072.0,  4096.0, 73.0, 'big'],
                           [    0.0, -4240.0, 70.0, 'small'],
                           [-1792.0, -4184.0, 70.0, 'small'],
                           [ 1792.0, -4184.0, 70.0, 'small'],
                           [- 940.0, -3308.0, 70.0, 'small'],
                           [  940.0, -3308.0, 70.0, 'small'],
                           [    0.0, -2816.0, 70.0, 'small'],
                           [-3584.0, -2484.0, 70.0, 'small'],
                           [ 3584.0, -2484.0, 70.0, 'small'],
                           [-1788.0, -2300.0, 70.0, 'small'],
                           [ 1788.0, -2300.0, 70.0, 'small'],
                           [-2048.0, -1036.0, 70.0, 'small'],
                           [    0.0, -1024.0, 70.0, 'small'],
                           [ 2048.0, -1036.0, 70.0, 'small'],
                           [-1024.0,     0.0, 70.0, 'small'],
                           [ 1024.0,     0.0, 70.0, 'small'],
                           [-2048.0,  1036.0, 70.0, 'small'],
                           [    0.0,  1024.0, 70.0, 'small'],
                           [ 2048.0,  1036.0, 70.0, 'small'],
                           [-1788.0,  2300.0, 70.0, 'small'],
                           [ 1788.0,  2300.0, 70.0, 'small'],
                           [-3584.0,  2484.0, 70.0, 'small'],
                           [ 3584.0,  2484.0, 70.0, 'small'],
                           [    0.0,  2816.0, 70.0, 'small'],
                           [- 940.0,  3310.0, 70.0, 'small'],
                           [  940.0,  3308.0, 70.0, 'small'],
                           [-1792.0,  4184.0, 70.0, 'small'],
                           [ 1792.0,  4184.0, 70.0, 'small'],
                           [    0.0,  4240.0, 70.0, 'small']], 
                          columns=['boost_pad_x', 'boost_pad_y', 
                                   'boost_pad_z', 'boost_pad_type'])

def show_play(fig, expected_goal, frame, is_goal):
    is_goal = 'Goal' if is_goal == 1 else "No Goal"
    fig = go.Figure(
    data=[
         go.Scatter(x=[-4096,-4096+1152], y=[5120-1152, 5120],
                     mode="lines",
                     line=dict(width=2, color="blue")),
        go.Scatter(x=[-4096,-4096+1152], y=[5120-1152, 5120],
                     mode="lines",
                     line=dict(width=2, color="blue")),
         go.Scatter(x=[4096,4096-1152], y=[5120-1152, 5120],
                     mode="lines",
                     line=dict(width=2, color="blue")),
          go.Scatter(x=[-4096,-4096+1152], y=[-5120+1152, -5120],
                     mode="lines",
                     line=dict(width=2, color="blue")),
         go.Scatter(x=[-4096,-4096+1152], y=[5120-1152, 5120],
                     mode="lines",
                     line=dict(width=2, color="blue")),
         go.Scatter(x=[4096,4096-1152], y=[-5120+1152, -5120],
                     mode="lines",
                     line=dict(width=2, color="blue")),
          
          go.Scatter(x=[-4096,-4096], y=[5120-1152, -5120+1152],
                     mode="lines",
                     line=dict(width=2, color="blue")),
          go.Scatter(x=[4096,4096], y=[5120-1152, -5120+1152],
                     mode="lines",
                     line=dict(width=2, color="blue")),
          
          go.Scatter(x=[-4096+1152,-893], y=[5120, 5120],
                     mode="lines",
                     line=dict(width=2, color="blue")),
          go.Scatter(x=[4096-1152,893], y=[5120, 5120],
                     mode="lines",
                     line=dict(width=2, color="blue")),
          go.Scatter(x=[-893,893], y=[5120, 5120],
                     mode="lines",
                     line=dict(width=2, color="blue")),
          
          go.Scatter(x=[-4096+1152,-893], y=[-5120, -5120],
                     mode="lines",
                     line=dict(width=2, color="blue")),
          go.Scatter(x=[4096-1152,893], y=[-5120, -5120],
                     mode="lines",
                     line=dict(width=2, color="blue")),
          go.Scatter(x=[-893,893], y=[-5120, -5120],
                     mode="lines",
                     line=dict(width=2, color="blue")),
          
          go.Scatter(x=[-893,893], y=[5120+880, 5120+880],
                     mode="lines",
                     line=dict(width=2, color="blue")),
          go.Scatter(x=[893,893], y=[5120+880, 5120],
                     mode="lines",
                     line=dict(width=2, color="blue")),
          go.Scatter(x=[-893,-893], y=[5120+880, 5120],
                     mode="lines",
                     line=dict(width=2, color="blue")),
          
          go.Scatter(x=[-893,893], y=[-5120-880, -5120-880],
                     mode="lines",
                     line=dict(width=2, color="blue")),
          go.Scatter(x=[893,893], y=[-5120-880, -5120],
                     mode="lines",
                     line=dict(width=2, color="blue")),
          go.Scatter(x=[-893,-893], y=[-5120-880, -5120],
                     mode="lines",
                     line=dict(width=2, color="blue")),
          go.Scatter(x=boost_locs['boost_pad_x'],y=boost_locs['boost_pad_y'], mode='markers', marker=dict(color='pink'))
         
         ],
    layout=go.Layout(
        xaxis=dict(range=[-5000, 5000], autorange=False, zeroline=False),
        yaxis=dict(range=[-6050, 6050], autorange=False, zeroline=False),
        title_text="Expected Goal: {}\n Result: {}".format(expected_goal, is_goal), hovermode="closest",
        width=700,
        height=800
        
        ),
    
    frames=[go.Frame(
        
        data=[go.Scatter(
            x=flat_df.loc[(flat_df.frames == k),'pos_x'].values,
            y=flat_df.loc[(flat_df.frames == k),'pos_y'].values,
            mode="markers",
            
            marker=dict(color=['orange' if game_info.loc[game_info.name == nm].is_orange.values[0] else 'blue' for nm in flat_df.name.unique()], 
                        size=10
                       )),
              go.Scatter(
            x=[df_ball.loc[df_ball.frames == k,'pos_x'].values[0]],
            y=[df_ball.loc[df_ball.frames == k,'pos_y'].values[0]],
            mode="markers",
            marker=dict(color='black', size=10)
                
              )
             ]
            )

        for k in df.loc[df.frames.between(frame-100,frame+150), 'frames'].values]
    )
    fig.update_layout(
    autosize=False,
    width=700,
    height=800,
    updatemenus=[dict(buttons = [dict(
                                               args = [None, {"frame": {"duration": 30, 
                                                                        "redraw": False},
                                                              "fromcurrent": True, 
                                                              "transition": {"duration": 0}}],
                                               label = "Play",
                                               method = "animate")],
                                type='buttons',
                                showactive=False,
                                y=1,
                                x=1.12,
                                xanchor='right',
                                yanchor='top')]
    )
    return fig

app = dash.Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
results_df = pd.read_csv("results.csv")
random_sample = results_df.sample(1)
replay = random_sample['replay_name'].values[0]
df, flat_df, game_info, players, df_ball = get_replay_df(replay)
is_goal = random_sample['is_goal'].values[0]
fig = show_play(go.Figure(), random_sample['expected_goal'].values[0], random_sample['frames'].values[0], is_goal)



app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)