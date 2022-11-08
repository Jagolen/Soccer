import pandas as pd

from opta.epv.processor import distance_to_goal, feature_creation
import pickle

# import numexpr
from sklearn.model_selection import train_test_split
from math import sqrt
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
                                      target_label_lin='chain_xG', out_folder='../models/'):


    # Filter data

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

    df_['const'] = 1

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
    out_lin_model_name = f"{out_folder}//EPV_Lin_Model.sav"

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

    DATA_FOLDER = 'D:/Opta Data/'

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

    df = pd.read_parquet(f"{DATA_FOLDER}/processed/pass_xg.parquet")

    train_pass_possession_value_model(df,PASS_LOG_MODEL_COLUMNS, PASS_LIN_MODEL_COLUMNS,
                                        target_label_log = 'chain_shot',
                                        target_label_lin = 'chain_xG'
                                      )

