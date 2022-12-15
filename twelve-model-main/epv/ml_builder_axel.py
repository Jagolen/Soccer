import numpy as np
import pandas as pd

# add epv. before processor if running streamlit for app.py
from epv.processor import distance_to_goal, pass_distance, feature_creation
import pickle

# import numexpr
from sklearn.model_selection import train_test_split
from math import sqrt
import random
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.discrete.discrete_model import Logit


def __add_features(df, model_variables):
    """
        Calculates the values for the variables
    :param df:
    :param model_variables:
    :return:
    """
    for var in model_variables:
        if var not in df:
            df[var] = df.eval(var)

    return df


def add_pass_data(df, short_edge=False):
    '''
    Create a new dataframe for passes with random start values, ending at the pitches line and an xG = 0
    '''
    df1 = pd.DataFrame(np.random.uniform(0.0, 100.0, size = (100000, 4)), columns = ['start_x', 'start_y', 'end_x', 'end_y'])
    edge = [0.0, 100.0]
    if short_edge:
        df1['end_x'] = pd.Series(np.tile(edge, len(df) // len(edge) + 1)[:len(df)]).sample(frac=1)
    df1['end_y'] = pd.Series(np.tile(edge, len(df) // len(edge) + 1)[:len(df)]).sample(frac=1)
    df2 = pd.DataFrame(np.ones((100000, 3), int), columns = ['type_id', 'chain_start_type_id', 'prev_event_type'])
    df3 = pd.DataFrame(np.zeros((100000, 6)), columns = ['time_difference', 'time_from_chain_start', 'pass_angle',
            'minute', 'second', 'match_state'])
    df4 = pd.DataFrame(np.full((100000, 24), False), columns = ['chain_goal', 'chain_shot','cross', 'head_pass',
            'through_pass', 'freekick_pass', 'corner_pass', 'throw-in', 'chipped', 'lay-off', 'launch', 'flick-on',
            'pull-back', 'switch', 'assist', '2nd_assist', 'in-swing', 'out-swing', 'straight', 'overhit_cross',
            'driven_cross', 'floated_cross', 'possession_goal', 'possession_shot'])
    df5 = pd.DataFrame(np.random.randint(1, 1000000000, size = (100000, 11)), columns = ['id', 'match_id', 'tournament_id', 'chain_id',
            'possession_index', 'team_id', 'player_id', 'possession_team_id', 'event_index', 'prev_event_team', 'season_id'])
    data_frames = [df1, df2, df3, df4, df5]
    new_df = pd.concat(data_frames, axis = 'columns')

    new_df['outcome'] = True
    new_df['chain_type'] = 'open_play'
    new_df['datetime'] = pd.Timestamp('2022-11-30 12:00:00.000')
    new_df['possession_xG'], new_df['chain_xG'] = 0.0, 0.0
    new_df = feature_creation(new_df)

    #display(show_arrows(new_df.iloc[:20]))

    #display(new_df)
    df = pd.concat([df, new_df], ignore_index=True, sort=False)
    return df


def split_data(df, x='start_x', y='start_y'):

    # Sort the datframe by x and y columns:
    df = df.sort_values(by=[x, y])

    # create and empty list to store the new data_frames:
    df_list = []

    # loop over the x and y values:
    for a in np.arange(0, 100, 20):
        for b in np.arange(0, 100, 20):
            # select rows where the coordinates are within the range:
            df_slice = df[(df[x] > a) & (df[x] <= a + 20) &
                          (df[y] > b) & (df[y] <= b + 20)]
            # add the dataframe to the list:
            df_list.append(df_slice)

    # find the smallest dataframe in the list:
    min_rows = min([len(df) for df in df_list])
    max_rows = max([len(df) for df in df_list])
    print(min_rows, max_rows)

    # loop over the list of dataframes and sample each to the size of min_rows
    for i, df in enumerate(df_list):
        df_list[i] = df.sample(min_rows)

    # combine all smaller dataframes into one
    df = pd.concat(df_list, ignore_index=True)

    return df


def __create_logistic_model(df, target_label, model_variables):

    # Create Train/Test Data
    X = df.copy()

    # X = X.fillna(0)

    Y = X.loc[:, X.columns == target_label]
    X = sm.add_constant(X)

    # Split Data into Test & Training Sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

    # from sklearn.linear_model import LogisticRegression
    # clf = LogisticRegression(random_state=0).fit(X_train, Y_train)

    # Create Logistic Model
    # Fit Model
    Model = Logit(Y_train, X_train[model_variables])
    xG_Model = Model.fit()

    return xG_Model, X_test, Y_test


def __create_linear_model(df, target_label, model_variables):

    # Create Train/Test Data
    X = df.copy()

    # Transform to integers
    bool_columns = [x for x, y in X.dtypes.items() if y == bool]
    X[bool_columns] = X[bool_columns].astype(int)

    X = X[X[target_label] > 0]

    Y = X.loc[:, X.columns == target_label]

    # # Split Data into Test & Training Sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

    # Fit Model
    Model = sm.OLS(Y_train.values, X_train[model_variables])
    FI_xG_Model_new = Model.fit()

    return FI_xG_Model_new, X_test, Y_test


def train_pass_possession_value_model(df_, log_columns, lin_columns,
                                      target_label_log='chain_shot',
                                      target_label_lin='chain_xG',
                                      out_folder='/Users/axelkjell/github/Soccer/opta/models/'):


    # Filter data

    #df_ = df_[df_['type_id']]

    #if df['end_x'] == float(101) or df['end_x'] == float(-1) or df['end_y'] == float(101) or df['end_y'] == float(-1):
    #    df['chain_xG'] = 0

    # Only Open play passes
    df_ = df_[df_['chain_type'] == 'open_play']

    # Only passes from attacking possessions
    df_ = df_[df_['possession_team_id'] == df_['team_id']]


    # Only successful passes
    df_ = df_[(df_['outcome'])]


    # Transform data
    df = df_

    # Feature creation
    df = feature_creation(df)

    df['const'] = 0.5

    # Feature Calculation
    df = __add_features(df, log_columns+lin_columns)

    # Target Label
    df[target_label_log] = df[target_label_log].astype(bool)


    ### LOGISTIC MODEL
    # Train Model
    pass_log_model, X_test, Y_test = __create_logistic_model(df, target_label_log, log_columns)

    # Summary of Model
    print('Summary of xG Model:')
    print(pass_log_model.summary2())

    # Predict Probability of Goal
    Y_Test_Pred = pass_log_model.predict(X_test[log_columns]).values

    AUC = roc_auc_score(Y_test, Y_Test_Pred)
    print('Pass Model Logistic AUC:', AUC)

    # # Predict Probability of Goal
    # Y_Test_Pred = xgblogistic.predict(X_test[PASS_LOG_MODEL_COLUMNS])
    # AUC = roc_auc_score(Y_test, Y_Test_Pred)
    # print('XGBOOST AUC:', AUC)

    # # Save Model
    out_log_model_name = f"{out_folder}/EPV_Log_Model.sav"

    pass_log_model.save(out_log_model_name, remove_data=True)

    ### LINEAR MODEL
    # Create Linear Model

    pass_lin_model, X_test, Y_test = __create_linear_model(df, target_label_lin, lin_columns)

    # Summary of Model
    print('Summary of FI Model:')
    print(pass_lin_model.summary2())
    Y_Test_Pred = pass_lin_model.predict(X_test[lin_columns]).values
    RMSE = sqrt(mean_squared_error(Y_test, Y_Test_Pred))
    print('RMSE', RMSE)

    # Y_Test_Pred = xgblinear.predict(X_test[PASS_LIN_MODEL_COLUMNS])
    # RMSE = sqrt(mean_squared_error(Y_test, Y_Test_Pred))
    # print('RMSE XGBoost', RMSE)
    out_lin_model_name = f"{out_folder}/EPV_Lin_Model.sav"

    pass_lin_model.save(out_lin_model_name, remove_data=True)


# 12 - Premier League
# 20 - Seria A
# 21 - La Liga
# 3 - Bundesliga
# 13 - Allsvenskan
# 18 - Danish
# 19 - Norway
# 96 - 2022, 89 - 2021, 86 - 2020
if __name__ == '__main__':

    DATA_FOLDER = '/Users/axelkjell/github/Soccer/opta/data'

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

    df = pd.read_parquet(f"{DATA_FOLDER}/possessions_xg.parquet")
    df = feature_creation(df)
    df = add_pass_data(df)

    train_pass_possession_value_model(df,PASS_LOG_MODEL_COLUMNS, PASS_LIN_MODEL_COLUMNS,
                                        target_label_log = 'chain_shot',
                                        target_label_lin = 'chain_xG')
