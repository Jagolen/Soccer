import pandas as pd
import matplotlib.pyplot as plt
import pickle
from epv.processor import feature_creation
import sklearn.metrics as skm

df = pd.read_parquet("opta/data/pass_xg.parquet")
lin_model = pickle.load(open("opta/models/EPV_Lin_Model.sav", "rb"))
log_model = pickle.load(open("opta/models/EPV_Log_Model.sav", "rb"))

# Only Open play passes
df = df[df['chain_type'] == 'open_play']

# Only passes from attacking possessions
df = df[df['possession_team_id'] == df['team_id']]

# Only successful passes
df = df[(df['outcome'])]

df = feature_creation(df)

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
df['const'] = 1
for var in (PASS_LOG_MODEL_COLUMNS+PASS_LIN_MODEL_COLUMNS):
    if var not in df:
        df[var] = df.eval(var)

X = df.copy()
Ylog = X.loc[:, X.columns == 'chain_shot']
Ylin = X.loc[:, X.columns == 'chain_xG']
Xlog = X[PASS_LOG_MODEL_COLUMNS]
Xlin = X[PASS_LIN_MODEL_COLUMNS]
Ylog_pred = log_model.predict(Xlog).values
Ylin_pred = lin_model.predict(Xlin).values

skm.RocCurveDisplay.from_predictions(Ylog, Ylog_pred)
skm.v_measure_score
plt.show()

