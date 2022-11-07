
import pickle
import sys
from io import BytesIO
from PIL import Image
import streamlit as st
import sklearn.metrics as skm
from matplotlib.colors import ListedColormap
from mplsoccer import Pitch
from st_row_buttons import st_row_buttons
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path


path_root = Path(__file__).parents[1]
print(path_root)
sys.path.append(str(path_root))
from opta.epv.ml_builder import __add_features
from opta.epv.processor import create_base_dataset, feature_creation
from opta.epv.twelve_xg_model_old import xT_pass, get_EPV_at_location
from opta.settings import ROOT_DIR


#@st.experimental_memo
def get_pass_model():
    return pickle.load(open(f"{ROOT_DIR}/models/EPV_Log_Model.sav", 'rb')),  \
           pickle.load(open(f"{ROOT_DIR}/models/EPV_Lin_Model.sav", 'rb'))


def get_img_bytes(fig, custom=False, format='png', dpi=200):
    tmpfile = BytesIO()

    if custom:
        fig.savefig(tmpfile, format=format, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.35)
    else:
        fig.savefig(tmpfile, format=format, dpi=dpi,facecolor=fig.get_facecolor(),transparent=False, frameon=False)  # , transparent=False, bbox_inches='tight', pad_inches=0.35)

    tmpfile.seek(0)

    return tmpfile


st.set_page_config(page_title='Twelve Analytics Page',
                 #  page_icon='data/img/light-logo-small.png',
                   layout="wide"

                   )

#def app():
with st.spinner("Loading"):

    selected_sub_page = st_row_buttons(["Passing Model"])

    if selected_sub_page == 'Passing Model':

        test_old_model = st.selectbox("Select Passing Model", ["Trained model", "xT Positional" ,'xT Action Based'])

        columns = st.columns(6)
        assist = columns[0].checkbox('Assist')
        cross = columns[1].checkbox('Cross')
        cutback = columns[2].checkbox('Cutback')
        switch = columns[3].checkbox('Switch')
        through_pass = columns[4].checkbox('Through Pass')

        columns = st.columns(3)
        start_x = columns[0].slider('Start x',0, 100, 50)
        start_y = columns[1].slider('Start y', 0, 100, 50)

        # Load Models
        model_pass_log, model_pass_lin, =  get_pass_model()

        #For ROC/SCORE
        dfr = pd.read_parquet(f"{ROOT_DIR}/data/pass_xg.parquet")

        # Only Open play passes
        dfr = dfr[dfr['chain_type'] == 'open_play']

        # Only passes from attacking possessions
        dfr = dfr[dfr['possession_team_id'] == dfr['team_id']]

        # Only successful passes
        dfr = dfr[(dfr['outcome'])]

        dfr = feature_creation(dfr)

        PASS_LOG_MODEL_COLUMNS = [
            'const',
            'start_x', 'end_x',
            'end_y_adj', 'start_y_adj',
            'start_x*start_y_adj', 'start_x*start_x',
            'end_x*end_y_adj', 'end_x*end_x', 'end_x*end_x*end_x',
            'start_x*end_x', 'start_y_adj*end_y_adj', 'start_x*start_y_adj*end_x',
            'start_x*end_x*end_y_adj',
            'start_x*start_x*end_x',
            'end_x*end_x*start_x', 'start_x*start_x*start_y_adj', 'end_x*end_x*end_y_adj',
            'end_y_adj*end_y_adj*start_y_adj',

        ]

        PASS_LIN_MODEL_COLUMNS = [
            'const',
            'start_x', 'end_x',
            'end_y_adj', 'start_y_adj',
            'cross*end_x',
            'through_pass*end_x',
            'start_x*start_y_adj', 'start_x*start_x',
            'end_x*end_y_adj', 'end_x*end_x', 'end_x*end_x*end_x',
            'start_x*end_x', 'start_y_adj*end_y_adj', 'start_x*start_y_adj*end_x',
            'start_x*end_x*end_y_adj',
            'start_x*start_x*end_x',
            'end_x*end_x*start_x', 'start_x*start_x*start_y_adj', 'end_x*end_x*end_y_adj',
            'end_y_adj*end_y_adj*start_y_adj',
            'switch',
            'pass_length',
            'assist',
            'directness*end_x',
            'distance_start',
            'distance_end',
            'time_difference',
            'time_from_chain_start'
        ]
        dfr['const'] = 1
        for var in (PASS_LOG_MODEL_COLUMNS+PASS_LIN_MODEL_COLUMNS):
            if var not in dfr:
                dfr[var] = dfr.eval(var)
        X = dfr.copy()
        Ylog = X.loc[:, X.columns == 'chain_shot']
        Ylin = X.loc[:, X.columns == 'chain_xG']
        Xlog = X[PASS_LOG_MODEL_COLUMNS]
        Xlin = X[PASS_LIN_MODEL_COLUMNS]
        Ylog_pred = model_pass_log.predict(Xlog).values
        Ylin_pred = model_pass_lin.predict(Xlin).values

        fpr, tpr, weight = skm.roc_curve(Ylog, Ylog_pred)
        auc = skm.auc(fpr, tpr)

        # Merged
        def create_dataset_start(start_x,start_y,one_dim=True, simple=False):
            df = create_base_dataset(start_x, start_y,one_dim)
            if simple:
                return df

            return df

        starting_point = True
        row_x, row_y, starting_point = 'end_x','end_y', True

        #weighed = None
        if test_old_model == 'xT Action Based':
            df = create_dataset_start(start_x, start_y, not starting_point, simple=True)
            df['prob'] = [xT_pass(start_x,start_y, x2, y2, cross=cross, throughBall=through_pass, pullBack=False, chanceCreated=False, flickOn=False) for x2, y2 in df[['end_x', 'end_y']].values]
            df['prob'] *= 0.3

        if  test_old_model == "xT Positional":
            df = create_dataset_start(start_x, start_y, not starting_point, simple=True)
            df['prob'] = [get_EPV_at_location((x2,y2)) - get_EPV_at_location((start_x,start_y)) for x2, y2 in df[['end_x', 'end_y']].values]

        else:

            #weighed = st.checkbox("Weighted", True, help="multiplies final probability by 0.3")

            df = create_dataset_start(start_x, start_y, not starting_point, simple=False)
            df[['assist','cross','pull-back','switch','through_pass']] = assist, cross, cutback, switch, through_pass
            df[['time_from_chain_start','time_difference']] = 0, 0

            df = __add_features(df, model_pass_log.model.exog_names + model_pass_lin.model.exog_names)

            df['prob_log'] = model_pass_log.predict(df[model_pass_log.model.exog_names])
            df['prob_lin'] = model_pass_lin.predict(df[model_pass_lin.model.exog_names])

            df['prob'] =  df['prob_log'] * df['prob_lin']

            df = df.fillna(0)

        # Heatmap
        category_colors = plt.get_cmap('Greens')(np.linspace(0.10, 0.80, 50))
        newcmp = ListedColormap(category_colors)

        i= 0
        columns = st.columns(3)
        if 'prob_log' not in df:
            # Pitch
            pitch = Pitch(pitch_type='opta',
                          linewidth=1,
                          goal_type='box',
                          line_zorder=2)
            fig, axs = pitch.grid(figheight=8, title_height=0.05, endnote_space=0,
                                  # Turn off the endnote/title axis. I usually do this after
                                  # I am happy with the chart layout and text placement
                                  axis=False,
                                  title_space=0.01, grid_height=0.92, endnote_height=0.02)
            ax = axs['pitch']
            bin_statistic = pitch.bin_statistic(df[row_x], df[row_y], values=df['prob'], statistic='mean', bins=(13, 7))

            pcm = pitch.heatmap(bin_statistic, ax=ax, cmap='Greens', edgecolors='#22312b', vmax=0.3, vmin=0)
            labels = pitch.label_heatmap(bin_statistic, color='k', fontsize=10,  ax=ax, ha='center', va='center',  str_format='{:.3f}')

            fig.colorbar(pcm, ax=ax, shrink=0.75)  # ,orientation='horizontal',pad=0.05)

            # Starting point
            if starting_point:
                pitch.plot(start_x, start_y, zorder=10,
                           marker="o", markersize=16,
                           markeredgewidth=2,
                           markeredgecolor='k',
                           ax=ax)

            pitch.draw(ax)
            ax.set_title(f"{test_old_model}")

            columns[i].pyplot(fig, dpi=100, transparent=False, bbox_inches=None)
            columns[i].download_button('Download file', get_img_bytes(fig), f"{test_old_model}.png")
        else:
            columns = st.columns(3)
            for i, row in enumerate(['prob_log', 'prob_lin', 'prob']):
                # Pitch
                pitch = Pitch(pitch_type='opta',
                              linewidth=1,
                              goal_type='box',
                              line_zorder=2)
                fig, axs = pitch.grid(figheight=8, title_height=0.05, endnote_space=0,
                                      # Turn off the endnote/title axis. I usually do this after
                                      # I am happy with the chart layout and text placement
                                      axis=False,
                                      title_space=0.01, grid_height=0.92, endnote_height=0.02)
                ax = axs['pitch']
                # Add weight to probability like in twelve old model
                #if weighed and row == 'prob':
                #    df[row] *= 0.3

                bin_statistic = pitch.bin_statistic(df[row_x], df[row_y], values=df[row], statistic='mean',
                                                    bins=(13, 7))

                pcm = pitch.heatmap(bin_statistic, ax=ax, cmap='Greens', edgecolors='#22312b', vmax=0.3, vmin=0)
                labels = pitch.label_heatmap(bin_statistic, color='k', fontsize=10, ax=ax, ha='center', va='center',
                                             str_format='{:.3f}')

                fig.colorbar(pcm, ax=ax, shrink=0.75)  # ,orientation='horizontal',pad=0.05)

                # Starting point
                if starting_point:
                    pitch.plot(start_x, start_y, zorder=10,
                               marker="o", markersize=16,
                               markeredgewidth=2,
                               markeredgecolor='k',
                               ax=ax)

                pitch.draw(ax)
                ax.set_title(f"{test_old_model} ({row})")

                columns[i].pyplot(fig, dpi=100, transparent=False, bbox_inches=None)
                columns[i].download_button('Download file', get_img_bytes(fig), f"{test_old_model}.png")
            if test_old_model == 'Trained model':
                col1, col2, col3 = st.columns([1,6,1])
                with col1:
                    st.write("")
                with col2:
                    st.write("")
                with col3:
                    fig = plt.figure()
                    plt.plot(fpr, tpr, color="orange", label="ROC Curve")
                    plt.plot([0, 1], [0, 1], color="blue", label="Random Guess", linestyle="--")
                    plt.legend()
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("ROC curve")
                    st.pyplot(fig)
                    st.write(f"AUC = {auc:.3f}")


        #if i>=1:
        #    columns = st.columns(2)
        #    i = 0
        #else:
       #     i +=1
