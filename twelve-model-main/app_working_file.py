
import pickle
import seaborn as sb
import sys
import os
from io import BytesIO
import streamlit as st
import streamlit_pandas as sp
import sklearn.metrics as skm
from matplotlib.colors import ListedColormap
from mplsoccer import Pitch
from st_row_buttons import st_row_buttons
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

import epv.ml_builder as bld
from epv.processor import create_base_dataset, feature_creation
from epv.twelve_xg_model_old import xT_pass, get_EPV_at_location
from settings import ROOT_DIR


# @st.experimental_memo
def get_pass_model():
	return pickle.load(open(f"{ROOT_DIR}/models/EPV_Log_Model.sav", 'rb')), \
		   pickle.load(open(f"{ROOT_DIR}/models/EPV_Lin_Model.sav", 'rb'))


@st.experimental_memo
def load_dataset():

	return pd.read_parquet(f"{ROOT_DIR}/data/possessions_xg.parquet")


def get_img_bytes(fig, custom=False, format='png', dpi=200):
	tmpfile = BytesIO()

	if custom:
		fig.savefig(tmpfile, format=format, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches='tight',
					pad_inches=0.35)
	else:
		fig.savefig(tmpfile, format=format, dpi=dpi, facecolor=fig.get_facecolor(), transparent=False,
					frameon=False)  # , transparent=False, bbox_inches='tight', pad_inches=0.35)

	tmpfile.seek(0)

	return tmpfile


st.set_page_config(page_title='Twelve Analytics Page',
				   #  page_icon='data/img/light-logo-small.png',
				   layout="wide"

				   )

# def app():
with st.spinner("Loading"):
	selected_sub_page = st_row_buttons(["Visualize Model", "Train a model", "Test a model", "Pass Assesment"])

	if selected_sub_page == 'Visualize Model':

		test_old_model = st.selectbox("Select Passing Model", ["Trained model", "xT Positional", 'xT Action Based'])
		type_of_model = st.selectbox("Select type of model", ["loglin", "log"])

		#Load data
		df_dataset = load_dataset()


		#Create list of available models

		if type_of_model == "loglin":
			model_names_lin = []
			for files in os.listdir(f"{ROOT_DIR}/models/lin_models"):
				if files.endswith(".sav"):
					model_names_lin.append(files)

		model_names_log = []
		for files in os.listdir(f"{ROOT_DIR}/models/log_models"):
			if files.endswith(".sav"):
				model_names_log.append(files)

		# Load Models
		load_log, load_lin = st.columns(2)
		with load_log:
			selected_log_model = st.selectbox("Select Logistic Model", model_names_log)
		
		with load_lin:
			if type_of_model == "loglin":
				selected_lin_model = st.selectbox("Select Linear Model", model_names_lin)
			else:
				st.write("")

		#Load log & lin models
		model_pass_log = pickle.load(open(f"{ROOT_DIR}/models/log_models/{selected_log_model}", 'rb'))
		if type_of_model == "loglin":
			model_pass_lin = pickle.load(open(f"{ROOT_DIR}/models/lin_models/{selected_lin_model}", 'rb'))

		if test_old_model == "Trained model":
			#List over features for both models
			if type_of_model == "loglin":
				features_log_lin = model_pass_log.model.exog_names + model_pass_lin.model.exog_names
			else:
				features_log_lin = model_pass_log.model.exog_names
			features_log_lin = [f.split("*") for f in features_log_lin]
			features_log_lin = [x for y in features_log_lin for x in y]
			features_log_lin = list(set(features_log_lin))
			features_log_lin = [x for x in features_log_lin]
			boolean_features = [[x] for x, y in df_dataset.dtypes.items() if y == bool and x in features_log_lin]

			#Setting features True or False
			columns = st.columns(len(boolean_features)+1)
			for i, feature in enumerate(boolean_features):
				temp_value = columns[i].checkbox(feature[0])
				if temp_value:
					feature.append(True)
				else:
					feature.append(False)
		else:
			columns = st.columns(6)
			assist = columns[0].checkbox('Assist')
			cross = columns[1].checkbox('Cross')
			cutback = columns[2].checkbox('Pull-back')
			switch = columns[3].checkbox('Switch')
			through_pass = columns[4].checkbox('Through Pass')

		columns = st.columns(3)
		start_x = columns[0].slider('Start x', 0, 100, 50)
		start_y = columns[1].slider('Start y', 0, 100, 50)




		# Merged
		def create_dataset_start(start_x, start_y, one_dim=True, simple=False):
			df = create_base_dataset(start_x, start_y, one_dim)
			if simple:
				return df

			return df


		starting_point = True
		row_x, row_y, starting_point = 'end_x', 'end_y', True

		# weighed = None
		if test_old_model == 'xT Action Based':
			df = create_dataset_start(start_x, start_y, not starting_point, simple=True)
			df['prob'] = [xT_pass(start_x, start_y, x2, y2, cross=cross, throughBall=through_pass, pullBack=False,
								  chanceCreated=False, flickOn=False) for x2, y2 in df[['end_x', 'end_y']].values]
			df['prob'] *= 0.3

		if test_old_model == "xT Positional":
			df = create_dataset_start(start_x, start_y, not starting_point, simple=True)
			df['prob'] = [get_EPV_at_location((x2, y2)) - get_EPV_at_location((start_x, start_y)) for x2, y2 in
						  df[['end_x', 'end_y']].values]

		else:

			# weighed = st.checkbox("Weighted", True, help="multiplies final probability by 0.3")

			df = create_dataset_start(start_x, start_y, not starting_point, simple=False)

			for f in boolean_features:
				df[f[0]] = f[1]

			#df[['assist', 'cross', 'pull-back', 'switch',
			#	'through_pass']] = assist, cross, cutback, switch, through_pass
			df[['time_from_chain_start', 'time_difference']] = 0, 0

			if type_of_model == "loglin":
				df = bld.__add_features(df, model_pass_log.model.exog_names + model_pass_lin.model.exog_names)

				df['prob_log'] = model_pass_log.predict(df[model_pass_log.model.exog_names])
				df['prob_lin'] = model_pass_lin.predict(df[model_pass_lin.model.exog_names])

				df['prob'] = df['prob_log'] * df['prob_lin']
			else:
				df = bld.__add_features(df, model_pass_log.model.exog_names)

				df['prob_log'] = model_pass_log.predict(df[model_pass_log.model.exog_names])

				df['prob'] = df['prob_log']


			df = df.fillna(0)

		# Heatmap
		category_colors = plt.get_cmap('Greens')(np.linspace(0.10, 0.80, 50))
		newcmp = ListedColormap(category_colors)

		i = 0
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
			ax.set_title(f"{test_old_model}")

			columns[i].pyplot(fig, dpi=100, transparent=False, bbox_inches=None)
			columns[i].download_button('Download file', get_img_bytes(fig), f"{test_old_model}.png")
		else:
			if type_of_model == "loglin":
				probabilities = ['prob_log', 'prob_lin', 'prob']
				columns = st.columns(3)
			else:
				probabilities = ['prob']
				columns = st.columns(1)
			for i, row in enumerate(probabilities):
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
				# if weighed and row == 'prob':
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

		# Only Open play passes
		df_dataset = df_dataset[df_dataset['chain_type'] == 'open_play']

		# Only passes from attacking possessions
		df_dataset = df_dataset[df_dataset['possession_team_id'] == df_dataset['team_id']]

		# Only successful passes
		df_dataset = df_dataset[(df_dataset['outcome'])]


		df_dataset = df_dataset[df_dataset['start_x'] > 0]
		df_dataset = df_dataset[df_dataset['start_y'] > 0]
		df_dataset = df_dataset[df_dataset['end_x'] > 0]
		df_dataset = df_dataset[df_dataset['end_y'] > 0]

		df_dataset = feature_creation(df_dataset)
		df_dataset['const'] = 1

		if type_of_model == "loglin":
			df_dataset = bld.__add_features(df_dataset, model_pass_log.model.exog_names + model_pass_lin.model.exog_names)
			df_dataset['prob_log'] = model_pass_log.predict(df_dataset[model_pass_log.model.exog_names])
			df_dataset['prob_lin'] = model_pass_lin.predict(df_dataset[model_pass_lin.model.exog_names])
			df_dataset['prob'] = df_dataset['prob_log'] * df_dataset['prob_lin']
			df_dataset['prob'] = df_dataset['prob'].astype(float)
		else:
			df_dataset = bld.__add_features(df_dataset, model_pass_log.model.exog_names)
			df_dataset['prob'] = model_pass_log.predict(df_dataset[model_pass_log.model.exog_names])
			df_dataset['prob'] = df_dataset['prob'].astype(float)
		
		how_many_points_save = st.text_input("Choose how many datapoints to save. A number under 10 000 is recommended")
		
		if how_many_points_save and how_many_points_save != 0:
			how_many_points_save = int(how_many_points_save)
			df_dataset = df_dataset.nlargest(how_many_points_save, 'prob')
			min_value=df_dataset["prob"].min()
			max_value=df_dataset["prob"].max()
			
			probability_slider = st.slider("Probability", 
				help="Probability that a pass leads to a chot which then leads to a goal. Limit this for fewer but (ideally) better passes",
				min_value=float(min_value), 
				max_value=float(max_value),
				step=0.01,
				value=float((max_value+min_value)/2))

			df_dataset = df_dataset[df_dataset['prob'] > probability_slider]
			max_value = df_dataset["prob"].max()

			fig_columns = st.columns(len(boolean_features)+2)

			with fig_columns[0]:
				pitch = Pitch(line_color='black',pitch_type='opta', line_zorder = 2)
				fig, ax = pitch.grid(axis=False)
				for i, row in df_dataset.iterrows():
					value = row["prob"]
					#adjust the line width so that the more passes, the wider the line
					line_width = (value / max_value)
					#get angle
					if (row.end_x - row.start_x) != 0:
						angle = np.arctan((row.end_y - row.start_y)/(row.end_x - row.start_x))*180/np.pi
					else:
						angle = np.arctan((row.end_y - row.start_y)/0.000001)*180/np.pi

					#plot lines on the pitch
					if row.prob != max_value:
						pitch.arrows(row.start_x, row.start_y, row.end_x, row.end_y,
											alpha=0.6, width=line_width, zorder=2, color="blue", ax = ax["pitch"])
					else:
						pitch.arrows(row.start_x, row.start_y, row.end_x, row.end_y,
											alpha=1, width=line_width*2, zorder=2, color="red", ax = ax["pitch"])        
					#annotate max text
						ax["pitch"].text((row.start_x+row.end_x-8)/2, (row.start_y+row.end_y-4)/2, str(value)[:5], fontweight = "bold", color = "purple", zorder = 4, fontsize = 16, rotation = int(angle))
				ax['title'].text(0.5, 0.5, 'All Data', ha='center', va='center', fontsize=30)
				plt.axis("off")
				st.pyplot(fig)

			for i, att in enumerate(boolean_features):
				with fig_columns[i+1]:
					temp_df = df_dataset.copy()
					temp_df = temp_df[temp_df[att[0]] == True]
					max_value = temp_df["prob"].max()
					pitch = Pitch(line_color='black',pitch_type='opta', line_zorder = 2)
					fig, ax = pitch.grid(axis=False)
					for i, row in temp_df.iterrows():
						value = row["prob"]
						#adjust the line width so that better prob gives a wider line
						line_width = (value / max_value)
						
						#get angle
						if (row.end_x - row.start_x) != 0:
							angle = np.arctan((row.end_y - row.start_y)/(row.end_x - row.start_x))*180/np.pi
						else:
							angle = np.arctan((row.end_y - row.start_y)/0.000001)*180/np.pi

						#plot lines on the pitch
						if row.prob != max_value:
							pitch.arrows(row.start_x, row.start_y, row.end_x, row.end_y,
												alpha=0.6, width=line_width, zorder=2, color="blue", ax = ax["pitch"])
						else:
							pitch.arrows(row.start_x, row.start_y, row.end_x, row.end_y,
												alpha=1, width=line_width*2, zorder=2, color="red", ax = ax["pitch"])        
						#annotate max text
							ax["pitch"].text((row.start_x+row.end_x-8)/2, (row.start_y+row.end_y-4)/2, str(value)[:5], fontweight = "bold", color = "purple", zorder = 4, fontsize = 16, rotation = int(angle))
					ax['title'].text(0.5, 0.5, att[0], ha='center', va='center', fontsize=30)
					plt.axis("off")
					st.pyplot(fig)
		else:
			st.write("No data to visualize!")

	if selected_sub_page == "Train a model":

		# Session state is used to save the attributes whenever a change is made on the page bu the user, otherwise the list would be reset
		if 'attributes' not in st.session_state:
			st.session_state.attributes = []
		if "active_attributes" not in st.session_state:
			st.session_state.active_attributes = []
		if "train_the_model" not in st.session_state:
			st.session_state.train_the_model = False

		st.header("Model information")
		#Output name for model
		model_type = st.selectbox("Select Model Type", ["Linear Regression Model", "Logistic Regression Model"])
		model_name = "model.sav"
		model_name_input = st.text_input("Model Name")
		if model_name_input:
			model_name = model_name_input + ".sav"
			filtered_settings_name = model_name_input + "_filter_settings.txt"

		
		# Load Dataset
		df_train = load_dataset()

		df_train['chain_xG > 0.5'] = df_train['chain_xG'] > 0.05

		# Get all columns
		all_columns = [x for x, y in df_train.dtypes.items()]

		# Gives a clear box at the start
		all_columns.insert(0, "")

		#Adds features used but not in the input dataframe
		all_columns.extend(["const", "directness", "distance_start", "distance_end"])

		# Get all boolean columns
		bool_columns = [x for x, y in df_train.dtypes.items() if y == bool]

		# Get all number columns, some columns like team_id,player_id are ignored as well
		float_columns = [x for x, y in df_train.dtypes.items() if y != bool and 'id' not in x and 'type' not in x]

		# Get all string columns, some columns like team_id,player_id are ignored as well
		categorical_columns = [x for x, y in df_train.dtypes.items() if 'type' in x or 'id' in x]

		# Extends the data with custom data, hopefully will generate a better model
		extend_the_data = st.checkbox("Extend pass data?", 
			help="This will add 100 000 new rows with type id 1 and with random start and end values \
				that goes outside the pitch, hopefully will improve the model around the edges of the field")
		
		if extend_the_data:
			df_train = bld.add_pass_data(df_train)
		
		split_data = st.checkbox("Split data?", help="This will split the field into sections and draw as \
			many samples from each section as the smallest section. This means each section will have the same\
				amount of data points")

		if split_data:
			df_train = bld.split_data(df_train)

		st.header("Filter data by attributes")
		st.subheader("Filter True and False attributes")
		true_attr_col, false_attr_col = st.columns(2)

		with true_attr_col:
			# Select which attributes should be true
			selected_att = st.multiselect("Only keep data when these attributes are true", bool_columns)
			for sa in selected_att:
				df_train = df_train[df_train[sa] == True]

		with false_attr_col:
			# Select which attributes should be False
			selected_att_f = st.multiselect('Only keep data when these attributes are false', bool_columns, [])
			for sa in selected_att_f:
				df_train = df_train[df_train[sa] == False]

		st.subheader("Filter float attributes")
		# Select which attributes should be greated then zero, here can be added slider for each attribute
		selected_att_to_limit_float = st.multiselect('Limit these attributes', float_columns, [])

		#Creates sliders for each attribute
		if selected_att_to_limit_float:
			selected_att_to_limit_with_limit_values = [[x, np.nan, np.nan] for x in selected_att_to_limit_float]
			for att in selected_att_to_limit_with_limit_values:
				#Min and Max values from data is used as min and max value from sliders
				if df_train.empty:
					attribute_max_value = 0
					attribute_min_value = 0
				else:
					attribute_max_value = float(df_train[att[0]].max())
					attribute_min_value = float(df_train[att[0]].min())
				attributes_to_limit, attributes_to_limit_slider = st.columns(2)
				with attributes_to_limit:
					st.write(att[0])

				with attributes_to_limit_slider:
					# Sets the corresponding elements in the selected_att_to_limit_with_limit_values list to the correct elements
					# NaN is used if the value is not limited, i.e. the element representing 'greater than' will be NaN if only 'less than' is used
					limiting_factor = st.radio(att[0], index=0, horizontal=True, options=("Less Than", "Greater Than", "Intervall"), label_visibility="collapsed")
					if attribute_min_value == attribute_max_value:
						st.write("No data left in Data Frame!")
					else:
						if limiting_factor == "Less Than":
							limiting_factor_value = st.slider(att[0], value=0.0, min_value=attribute_min_value, max_value=attribute_max_value, label_visibility="collapsed")
							att[1] = limiting_factor_value

						if limiting_factor == "Greater Than":
							limiting_factor_value = st.slider(att[0], value=0.0, min_value=attribute_min_value, max_value=attribute_max_value, label_visibility="collapsed")
							att[2] = limiting_factor_value

						if limiting_factor == "Intervall":
							limiting_factor_value = st.slider(att[0], value=(attribute_min_value, attribute_max_value), min_value=attribute_min_value, max_value=attribute_max_value, label_visibility="collapsed")
							att[1] = limiting_factor_value[1]
							att[2] = limiting_factor_value[0]
				
				# Limits the data based on the selected limits
				if not np.isnan(att[1]):
					df_train = df_train[df_train[att[0]] < att[1]]
				if not np.isnan(att[2]):
					df_train = df_train[df_train[att[0]] > att[2]]

		st.subheader("Filter categorical attributes")
		# Select which attribute to limit by catogery
		selected_att_to_limit_cat = st.multiselect('Limit categories', categorical_columns, [])
		if selected_att_to_limit_cat:
			for attlim in selected_att_to_limit_cat:

				# Get list of possible categories to pass to multiselect
				attribute_values = pd.unique(df_train[attlim])
				attributes_to_limit_2, limiting_factor_categories = st.columns(2)
				with attributes_to_limit_2:
					st.write(attlim)
				with limiting_factor_categories:

					#This is used to display event names instead of just their id
					if attlim == "type_id" or attlim == "prev_event_type" or attlim == "chain_start_type_id":
						#Creating a lookup table for type id consisting of two dictionaries so the values can easily be converted back and forth
						type_id_lookup_table = {
							1: 'Pass',
							2: 'Offside Pass',
							3: 'Take On',
							4: 'Foul',
							5: 'Out',
							6: 'Corner Awarded',
							7: 'Tackle',
							8: 'Interception',
							10: 'Save Goalkeeper',
							11: 'Claim Goalkeeper',
							12: 'Clearance',
							13: 'Miss',
							14: 'Post',
							15: 'Attempt Saved',
							16: 'Goal',
							17: 'Card Bookings',
							18: 'Player off',
							19: 'Player on',
							20: 'Player retired',
							21: 'Player returns',
							22: 'Player becomes goalkeeper',
							23: 'Goalkeeper becomes player',
							24: 'Condition change',
							25: 'Official change',
							27: 'Start delay',
							28: 'End delay',
							30: 'End',
							32: 'Start',
							34: 'Team set up',
							35: 'Player changed position',
							36: 'Player changed Jersey',
							37: 'Collection End',
							38: 'Temp_Goal',
							39: 'Temp_Attempt',
							40: 'Formation change',
							41: 'Punch',
							42: 'Good Skill',
							43: 'Deleted event',
							44: 'Aerial',
							45: 'Challenge',
							47: 'Rescinded card',
							49: 'Ball recovery',
							50: 'Dispossessed',
							51: 'Error',
							52: 'Keeper pick-up',
							53: 'Cross not claimed',
							54: 'Smother',
							55: 'Offside provoked',
							56: 'Shield ball opp',
							57: 'Foul throw-in',
							58: 'Penalty faced',
							59: 'Keeper Sweeper',
							60: 'Chance missed',
							61: 'Ball touch',
							63: 'Temp_Save',
							64: 'Resume',
							65: 'Contentious referee decision',
							74: 'Blocked Pass',
						}

						# The id number is converted to its corresponding string to be shown in the selectbox	
						type_id_lookup_table_reverse = dict(zip(type_id_lookup_table.values(), type_id_lookup_table.keys()))
						attribute_values = [type_id_lookup_table[x] for x in attribute_values if x in type_id_lookup_table_reverse.values()]
						chosen_attribute_values = st.multiselect(attlim, attribute_values, label_visibility="collapsed")

						# Convert the string back to the id value which is used for filtering
						chosen_attribute_values = [type_id_lookup_table_reverse[x] for x in chosen_attribute_values]

					elif attlim == "tournament_id":
						#Lookup table for league
						tournament_id_lookup_table = {
							"Allsvenskan": 13,
							"Bundesliga": 3,
							"Serie A": 20,
							"Premier League": 12,
							"La Liga": 21,
							"Norway": 19,
							"Denmark": 18,
						}

						# The id number is converted to its corresponding string to be shown in the selectbox	
						tournament_id_lookup_table_reverse = dict(zip(tournament_id_lookup_table.values(), tournament_id_lookup_table.keys()))
						attribute_values = [tournament_id_lookup_table_reverse[x] for x in attribute_values if x in tournament_id_lookup_table.values()]
						chosen_attribute_values = st.multiselect(attlim, attribute_values, label_visibility="collapsed")

						# Convert the string back to the id value which is used for filtering
						chosen_attribute_values = [tournament_id_lookup_table[x] for x in chosen_attribute_values]

					else:
						# Simply using the numerical id	
						chosen_attribute_values = st.multiselect(attlim, attribute_values, label_visibility="collapsed")

				
				#Remove the values not chosen from the chosen attribute
				if chosen_attribute_values:
					df_train = df_train[df_train[attlim].isin(chosen_attribute_values)]

		st.subheader("Limit so that two features are equal")
		limit_features_based_on_equality = st.multiselect("Limit these features so that only data when they are equal to another feature is saved", all_columns)
		if limit_features_based_on_equality:
			for limeq in limit_features_based_on_equality:
				eq_limit1, eq_limit2 = st.columns(2)
				with eq_limit1:
					st.write(limeq)
				with eq_limit2:
					limit_when_equal = st.selectbox(limeq, all_columns, label_visibility="collapsed")
				
				#Limit features
				if limit_when_equal:
					df_train = df_train[df_train[limeq] == df_train[limit_when_equal]]

		# Show dataset descriptions
		with st.expander("Dataset",False):
			st.write(df_train.shape)
			st.dataframe(df_train.describe())

		#Target attribute for the model
		st.header("Target attribute and desired attributes")
		if model_type == "Linear Regression Model":
			target_attribute = st.selectbox("Select target attribute for the model", float_columns)
		
		if model_type == "Logistic Regression Model":
			target_attribute = st.selectbox("Select target attribute for the model", bool_columns)

		#Let the user select attributes
		col111, col222 = st.columns([1, 1])
		with col111:
			add_attribute = st.selectbox("Input Desired Attributes for the model", all_columns)
		with col222:
			combine = st.button("Combine",help="Combines selected attributes, requires 2 or more attributes to be selected")
			delete = st.button("Delete", help="Deletes selected attributes")
			delete_all = st.button("Delete All", help="Deletes all attributes")
			square = st.button("Square", help="Creates the square of the attribute, i.e. the attribute multiplied with itself, as a new attribute. Requires that only one attribute is selected")
			default_features_log = st.button("Default Features (Logistic Model)", help="Creates the default attributes for the Logistic Model")
			default_features_lin = st.button("Default Features (Linear Model)", help="Creates the default attributes for the Linear Model")

		# Adds the chosen attributes to the attributes list
		if add_attribute and add_attribute not in st.session_state.attributes:
			st.session_state.attributes.append(add_attribute)
		if st.session_state.attributes:
			st.write("**Use the checkboxes to select attributes to use with the buttons on the right**")
			for i in st.session_state.attributes:
				temp = st.checkbox(i)
				if temp and i not in st.session_state.active_attributes:
					st.session_state.active_attributes.append(i)
				if not temp and i in st.session_state.active_attributes:
					st.session_state.active_attributes.remove(i)
			st.write(f"active attributes: {st.session_state.active_attributes}")

			# Handles combining attributes, squaring attributes, deleting attributes and so on
			if delete:
				for i in st.session_state.active_attributes:
					st.session_state.attributes.remove(i)
				st.session_state.active_attributes = []
				st.experimental_rerun()
			if combine:
				if len(st.session_state.active_attributes) > 1:
					new_feature = st.session_state.active_attributes[0]
					for i in range(1, len(st.session_state.active_attributes)):
						new_feature = new_feature + "*" + st.session_state.active_attributes[i]
					st.session_state.attributes.append(new_feature)
					st.experimental_rerun()
			if square and len(st.session_state.active_attributes) == 1:
				new_feature = st.session_state.active_attributes[0] + "*" + st.session_state.active_attributes[0]
				st.session_state.attributes.append(new_feature)
				st.experimental_rerun()

			if delete_all:
				st.session_state.active_features = []
				st.session_state.attributes = []
				st.experimental_rerun()

		if default_features_lin:
			st.session_state.attributes = [
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
			st.session_state.active_features = []
			st.experimental_rerun()

		if default_features_log:
			st.session_state.attributes = [
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
			st.session_state.active_features = []
			st.experimental_rerun()

		# Feature creation
		df_train = feature_creation(df_train)
		df_train["const"] = 1

		# Evaluate combinations of features
		df_train = bld.__add_features(df_train, st.session_state.attributes)

		#Create new list and include target attribute
		chosen_attributes_final = [x for x in st.session_state.attributes]

		if target_attribute:
			chosen_attributes_final.append(target_attribute)

		# Use only selected features and target attribute
		df_train = df_train[chosen_attributes_final]

		#Replaces NaN values with zeroes
		df_train.fillna(0)

		#Creating Correlation Matrix if any feature is chosen
		if st.session_state.attributes:
		# Create Correlation Matrix of selected features
			st.write("**Correlation Matrix**")
			show_correlation_matrix = st.checkbox("Show Correlation Matrix")
			if show_correlation_matrix:
				correlation_matrix = df_train.corr()

				#The correlation matrix is symmetrical, so a mask is created that will hide the upper triangle
				triangle_mask = np.zeros_like(correlation_matrix, dtype=bool)
				triangle_mask[np.triu_indices_from(triangle_mask)] = True

				#Create the Correlation Matrix using a heatmap
				fig, ax = plt.subplots()
				c_matrix_heatmap = sb.heatmap(correlation_matrix,
				xticklabels=True, # Makes all the x-labels visible
				yticklabels=True, # Makes all the y-labels visible
				square = True, # Creates a square heatmap (the boxes will be squares rather than rectangles)
				cmap = "coolwarm", # Blue-to-Red colormap
				mask = triangle_mask, 
				linewidths = 0.5, # Separates the heatmap boxes slightly
				vmin = -1,
				vmax = 1,
				annot=True, # Shows the values inside the boxes
				annot_kws= {"size": 60/len(correlation_matrix.columns)}) # Scales the text inside the heatmap boxes based on how many attributes are chosen

				plt.rc("xtick", labelsize=6)
				plt.rc("ytick", labelsize=6)
				plt.title("Correlation Matrix")
				st.pyplot(fig)

		# This will activate the actual training
		st.header("Training")
		train = st.button("Train the Model")
		if train:
			st.session_state.train_the_model = True
		
		if st.session_state.train_the_model:
			if target_attribute:
				if model_type == "Linear Regression Model":
					lin_model, x_test, y_test = bld.__create_linear_model(df_train, target_attribute, st.session_state.attributes)
					st.write("Model has been trained")
					show_train_stats = st.checkbox("Show statistics?")
					if show_train_stats:
						st.write(lin_model.summary2())
					save_an_output_file = st.button("Save model to disk?")
					if save_an_output_file:
						output_name = f"{ROOT_DIR}/models/lin_models/{model_name}"
						lin_model.save(output_name, remove_data=True)
						st.write(f"Model has been saved at {output_name}")
						st.session_state.train_the_model = False
					
					


				if model_type == "Logistic Regression Model":
					#Target label
					df_train[target_attribute] = df_train[target_attribute].astype(bool)
					log_model, x_test, y_test = bld.__create_logistic_model(df_train, target_attribute, st.session_state.attributes)
					show_train_stats = st.checkbox("Show statistics?")
					if show_train_stats:
						st.write(log_model.summary2())
					save_an_output_file = st.button("Save model to disk?")
					if save_an_output_file:
						output_name = f"{ROOT_DIR}/models/log_models/{model_name}"
						log_model.save(output_name, remove_data=True)
						st.write(f"Model has been saved at {output_name}")
						st.session_state.train_the_model = False
			else:
				st.write("Error: No target attribute")
	
	if selected_sub_page == "Test a model":

		how_many_models_test = st.selectbox("Test one model or compare two models?", ["Test a model", "Compare two models"])

		#Create list of available models
		model_names_lin = []
		model_names_log = []
		for files in os.listdir(f"{ROOT_DIR}/models/lin_models"):
			if files.endswith(".sav"):
				model_names_lin.append(files)

		for files in os.listdir(f"{ROOT_DIR}/models/log_models"):
			if files.endswith(".sav"):
				model_names_log.append(files)

		if how_many_models_test == "Test a model":
			load_log_model, load_lin_model = st.columns(2)

			with load_log_model:
				selected_log_model = st.selectbox("Select Logistic Model", model_names_log)
			
			with load_lin_model:
				selected_lin_model = st.selectbox("Select Linear Model", model_names_lin)

			#Load log & lin models
			log_model = pickle.load(open(f"{ROOT_DIR}/models/log_models/{selected_log_model}", 'rb'))
			lin_model = pickle.load(open(f"{ROOT_DIR}/models/lin_models/{selected_lin_model}", 'rb'))

			st.header("Statistics")

			st.subheader("Logistic Model")
			st.write("The model as a whole")

			stat_df_log_main = pd.DataFrame()
			stat_df_log_main["Metric"] = ["Model Type", "Output Variable", "Pseudo R^2", "Log-Likelihood", "LLR P-value", "Akaike Information Criterion"]
			stat_df_log_main["Value"] = [log_model.model.__class__.__name__, log_model.model.endog_names, log_model.prsquared, log_model.llf, log_model.llr_pvalue, log_model.aic]
			st.table(stat_df_log_main)

			st.write("Coefficients")

			stat_df_log_coef = pd.DataFrame()
			#stat_df_log_coef["Coefficient"] = log_model.exog_names
			stat_df_log_coef["Coefficient value"] = log_model.params
			stat_df_log_coef["Std Error"] = log_model.bse
			stat_df_log_coef["z-value"] = log_model.tvalues
			stat_df_log_coef["P-value (P>|z|)"] = log_model.pvalues
			stat_df_log_coef["Confidence Interval Lower Limit"] = log_model.conf_int()[0]
			stat_df_log_coef["Confidence Interval Upper Limit"] = log_model.conf_int()[1]
			st.table(stat_df_log_coef)

			st.subheader("Linear Model")
			st.write("The model as a whole")
			st.write(dir(lin_model))

			stat_df_lin_main = pd.DataFrame()
			stat_df_lin_main["Metric"] = ["Model Type", "Output Variable", "R^2", "Log-Likelihood", "F-statistic", "P-value (F-statistic)", "Akaike Information Criterion"]
			stat_df_lin_main["Value"] = [lin_model.model.__class__.__name__, lin_model.model.endog_names, lin_model.rsquared, lin_model.llf, lin_model.fvalue, lin_model.f_pvalue, lin_model.aic]
			st.table(stat_df_lin_main)

			st.write("Coefficients")

			stat_df_lin_coef = pd.DataFrame()
			stat_df_lin_coef["Coefficient value"] = lin_model.params
			stat_df_lin_coef["Std Error"] = lin_model.bse
			stat_df_lin_coef["t-value"] = lin_model.tvalues
			stat_df_lin_coef["P-value (P>|t|)"] = lin_model.pvalues
			stat_df_lin_coef["Confidence Interval Lower Limit"] = lin_model.conf_int()[0]
			stat_df_lin_coef["Confidence Interval Upper Limit"] = lin_model.conf_int()[1]
			st.table(stat_df_lin_coef)

		if how_many_models_test == "Compare two models":
			load_log_model, load_lin_model = st.columns(2)
			with load_log_model:
				selected_log_model = st.selectbox("Select First Logistic Model", model_names_log)
				selected_log_model2 = st.selectbox("Select Second Logistic Model", model_names_log)		
			with load_lin_model:
				selected_lin_model = st.selectbox("Select First Linear Model", model_names_lin)
				selected_lin_model2 = st.selectbox("Select Second Linear Model", model_names_lin)

			#Load log & lin models
			log_model = pickle.load(open(f"{ROOT_DIR}/models/log_models/{selected_log_model}", 'rb'))
			lin_model = pickle.load(open(f"{ROOT_DIR}/models/lin_models/{selected_lin_model}", 'rb'))
			log_model2 = pickle.load(open(f"{ROOT_DIR}/models/log_models/{selected_log_model2}", 'rb'))
			lin_model2 = pickle.load(open(f"{ROOT_DIR}/models/lin_models/{selected_lin_model2}", 'rb'))

			st.header("Statistics")

			st.subheader("Logistic Model")
			log_stats1, log_stats2 = st.columns(2)
			with log_stats1:
				st.write("**Model 1**")
				st.write("The model as a whole")
				stat_df_log_main = pd.DataFrame()
				stat_df_log_main["Metric"] = ["Model Type", "Output Variable", "Pseudo R^2", "Log-Likelihood", "LLR P-value", "Akaike Information Criterion"]
				stat_df_log_main["Value"] = [log_model.model.__class__.__name__, log_model.model.endog_names, log_model.prsquared, log_model.llf, log_model.llr_pvalue, log_model.aic]
				st.table(stat_df_log_main)

				st.write("Coefficients")
				stat_df_log_coef = pd.DataFrame()
				#stat_df_log_coef["Coefficient"] = log_model.exog_names
				stat_df_log_coef["Coefficient value"] = log_model.params
				stat_df_log_coef["Std Error"] = log_model.bse
				stat_df_log_coef["z-value"] = log_model.tvalues
				stat_df_log_coef["P-value (P>|z|)"] = log_model.pvalues
				stat_df_log_coef["Confidence Interval Lower Limit"] = log_model.conf_int()[0]
				stat_df_log_coef["Confidence Interval Upper Limit"] = log_model.conf_int()[1]
				st.table(stat_df_log_coef)
			
			with log_stats2:
				st.write("**Model 2**")
				st.write("The model as a whole")
				stat_df_log_main2 = pd.DataFrame()
				stat_df_log_main2["Metric"] = ["Model Type", "Output Variable", "Pseudo R^2", "Log-Likelihood", "LLR P-value", "Akaike Information Criterion"]
				stat_df_log_main2["Value"] = [log_model2.model.__class__.__name__, log_model2.model.endog_names, log_model2.prsquared, log_model2.llf, log_model2.llr_pvalue, log_model2.aic]
				st.table(stat_df_log_main2)

				st.write("Coefficients")
				stat_df_log_coef2 = pd.DataFrame()
				#stat_df_log_coef["Coefficient"] = log_model.exog_names
				stat_df_log_coef2["Coefficient value"] = log_model2.params
				stat_df_log_coef2["Std Error"] = log_model2.bse
				stat_df_log_coef2["z-value"] = log_model2.tvalues
				stat_df_log_coef2["P-value (P>|z|)"] = log_model2.pvalues
				stat_df_log_coef2["Confidence Interval Lower Limit"] = log_model2.conf_int()[0]
				stat_df_log_coef2["Confidence Interval Upper Limit"] = log_model2.conf_int()[1]
				st.table(stat_df_log_coef2)				

			st.subheader("Linear Model")

			lin_stats, lin_stats2 = st.columns(2)
			with lin_stats:
				stat_df_lin_main = pd.DataFrame()
				stat_df_lin_main["Metric"] = ["Model Type", "Output Variable", "R^2", "Log-Likelihood", "F-statistic", "P-value (F-statistic)", "Akaike Information Criterion"]
				stat_df_lin_main["Value"] = [lin_model.model.__class__.__name__, lin_model.model.endog_names, lin_model.rsquared, lin_model.llf, lin_model.fvalue, lin_model.f_pvalue, lin_model.aic]
				st.table(stat_df_lin_main)

				st.write("Coefficients")

				stat_df_lin_coef = pd.DataFrame()
				stat_df_lin_coef["Coefficient value"] = lin_model.params
				stat_df_lin_coef["Std Error"] = lin_model.bse
				stat_df_lin_coef["t-value"] = lin_model.tvalues
				stat_df_lin_coef["P-value (P>|t|)"] = lin_model.pvalues
				stat_df_lin_coef["Confidence Interval Lower Limit"] = lin_model.conf_int()[0]
				stat_df_lin_coef["Confidence Interval Upper Limit"] = lin_model.conf_int()[1]
				st.table(stat_df_lin_coef)
			
			with lin_stats2:
				stat_df_lin_main2 = pd.DataFrame()
				stat_df_lin_main2["Metric"] = ["Model Type", "Output Variable", "R^2", "Log-Likelihood", "F-statistic", "P-value (F-statistic)", "Akaike Information Criterion"]
				stat_df_lin_main2["Value"] = [lin_model2.model.__class__.__name__, lin_model2.model.endog_names, lin_model2.rsquared, lin_model2.llf, lin_model2.fvalue, lin_model2.f_pvalue, lin_model2.aic]
				st.table(stat_df_lin_main2)

				st.write("Coefficients")

				stat_df_lin_coef2 = pd.DataFrame()
				stat_df_lin_coef2["Coefficient value"] = lin_model2.params
				stat_df_lin_coef2["Std Error"] = lin_model2.bse
				stat_df_lin_coef2["t-value"] = lin_model2.tvalues
				stat_df_lin_coef2["P-value (P>|t|)"] = lin_model2.pvalues
				stat_df_lin_coef2["Confidence Interval Lower Limit"] = lin_model2.conf_int()[0]
				stat_df_lin_coef2["Confidence Interval Upper Limit"] = lin_model2.conf_int()[1]
				st.table(stat_df_lin_coef2)





		# Load Models
		#model_pass_log, model_pass_lin, = get_pass_model()

		# For ROC/SCORE
		dfr = pd.read_parquet(f"{ROOT_DIR}/data/possessions_xg.parquet")

		# Only Open play passes
		dfr = dfr[dfr['chain_type'] == 'open_play']

		# Only passes from attacking possessions
		dfr = dfr[dfr['possession_team_id'] == dfr['team_id']]

		# Only successful passes
		dfr = dfr[(dfr['outcome'])]

		#User can choose team or use all data
		dataset = st.selectbox("Select Dataset for stats", ["Use All Data", "Choose League"])

		if dataset == "Choose League":
			allsvenska = st.checkbox('Allsvenskan', value=True)
			bundes = st.checkbox('Bundesliga', value=True)
			seriea = st.checkbox('Serie A', value=True)
			pleague = st.checkbox('Premier League', value=True)
			laliga = st.checkbox('La Liga', value=True)
			norway = st.checkbox('Norway', value=True)
			denmark = st.checkbox('Denmark', value=True)

			#Adding chosen teams based on their id
			tournaments = []
			if allsvenska:
				tournaments.append(13)
			if bundes:
				tournaments.append(3)
			if seriea:
				tournaments.append(20)
			if pleague:
				tournaments.append(12)
			if laliga:
				tournaments.append(21)
			if norway:
				tournaments.append(19)
			if denmark:
				tournaments.append(18)

			#Backup is used if the resulting dataframe becomes empty
			backup_dataframe = dfr.copy()

			#Chose only the chosen series from the data frame
			dfr = dfr[dfr['tournament_id'].isin(tournaments)]
			if dfr.empty:
				dfr = backup_dataframe
				st.write("No data for this selection, using whole dataset instead")

		# Handles
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

		# Evaluate combinations of features
		dfr = bld.__add_features(dfr, PASS_LOG_MODEL_COLUMNS + PASS_LIN_MODEL_COLUMNS)
		
		#Prediction
		X = dfr.copy()
		Ylog = X.loc[:, X.columns == 'chain_shot']
		Ylin = X.loc[:, X.columns == 'chain_xG']
		Xlog = X[PASS_LOG_MODEL_COLUMNS]
		Xlin = X[PASS_LIN_MODEL_COLUMNS]
		Ylog_pred = log_model.predict(Xlog).values
		Ylin_pred = lin_model.predict(Xlin).values

		if how_many_models_test == "Test a model":

			#ROC Curve, score and RMSE
			fpr, tpr, weight = skm.roc_curve(Ylog, Ylog_pred)
			auc = skm.auc(fpr, tpr)
			Ylog_pred_bin = [round(elements) for elements in Ylog_pred]
			score = skm.accuracy_score(Ylog, Ylog_pred_bin)
			rmse = skm.mean_squared_error(Ylin, Ylin_pred)
			rmse = np.sqrt(rmse)
			
			#Plot result, columns scales the result down in the streamlit window
			roc1, roc2, roc3 = st.columns([2, 5, 2])
			with roc1:
				st.write("")
			
			with roc2:
				fig = plt.figure()
				plt.plot(fpr, tpr, color="orange", label="ROC Curve")
				plt.plot([0, 1], [0, 1], color="blue", label="Random Guess", linestyle="--")
				plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
				plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
				plt.grid(True)
				plt.legend()
				plt.xlabel("False Positive Rate")
				plt.ylabel("True Positive Rate")
				plt.title("ROC curve")
				st.pyplot(fig)
			
			with roc3:
				st.write("")

			#Write out the stats with 3 decimals
			st.write(f"AUC : {auc:.3f}")
			st.write(f"Score : {(score * 100):.3f}%")
			st.write(f"RMSE : {rmse:.3f}")

		if how_many_models_test == "Compare two models":

			Ylog_pred2 = log_model2.predict(Xlog).values
			Ylin_pred2 = lin_model2.predict(Xlin).values

			#ROC Curve, score and RMSE
			fpr, tpr, weight = skm.roc_curve(Ylog, Ylog_pred)
			fpr2, tpr2, weight2 = skm.roc_curve(Ylog, Ylog_pred2)
			auc = skm.auc(fpr, tpr)
			auc2 = skm.auc(fpr2, tpr2)
			Ylog_pred_bin = [round(elements) for elements in Ylog_pred]
			Ylog_pred_bin2 = [round(elements) for elements in Ylog_pred2]
			score = skm.accuracy_score(Ylog, Ylog_pred_bin)
			score2 = skm.accuracy_score(Ylog, Ylog_pred_bin2)
			rmse = skm.mean_squared_error(Ylin, Ylin_pred)
			rmse = np.sqrt(rmse)
			rmse2 = skm.mean_squared_error(Ylin, Ylin_pred2)
			rmse2 = np.sqrt(rmse2)
			
			#Plot result, columns scales the result down in the streamlit window
			roc1, roc2 = st.columns(2)
			with roc1:
				fig = plt.figure()
				plt.plot(fpr, tpr, color="orange", label="ROC Curve")
				plt.plot([0, 1], [0, 1], color="blue", label="Random Guess", linestyle="--")
				plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
				plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
				plt.grid(True)
				plt.legend()
				plt.xlabel("False Positive Rate")
				plt.ylabel("True Positive Rate")
				plt.title("ROC curve (Model 1)")
				st.pyplot(fig)
			
			with roc2:
				fig = plt.figure()
				plt.plot(fpr2, tpr2, color="orange", label="ROC Curve")
				plt.plot([0, 1], [0, 1], color="blue", label="Random Guess", linestyle="--")
				plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
				plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
				plt.grid(True)
				plt.legend()
				plt.xlabel("False Positive Rate")
				plt.ylabel("True Positive Rate")
				plt.title("ROC curve (Model 2)")
				st.pyplot(fig)

			#Write out the stats with 3 decimals
			st.write(f"AUC) : {auc:.3f} (Model 1), {auc2:.3f} (Model 2)")
			st.write(f"Score : {(score * 100):.3f}% (Model 1), {(score2 * 100):.3f}% (Model 2)")
			st.write(f"RMSE : {rmse:.3f} (Model 1), {rmse2:.3f} (Model 2)")

	if selected_sub_page == "Pass Assesment":
		type_of_model = st.selectbox("Select type of model", ["loglin", "log"])

		#Load data
		df_gp = load_dataset()


		#Create list of available models

		if type_of_model == "loglin":
			model_names_lin = []
			for files in os.listdir(f"{ROOT_DIR}/models/lin_models"):
				if files.endswith(".sav"):
					model_names_lin.append(files)

		model_names_log = []
		for files in os.listdir(f"{ROOT_DIR}/models/log_models"):
			if files.endswith(".sav"):
				model_names_log.append(files)

		# Load Models
		load_log, load_lin = st.columns(2)
		with load_log:
			selected_log_model = st.selectbox("Select Logistic Model", model_names_log)
		
		with load_lin:
			if type_of_model == "loglin":
				selected_lin_model = st.selectbox("Select Linear Model", model_names_lin)
			else:
				st.write("")

		#Load log & lin models
		model_pass_log = pickle.load(open(f"{ROOT_DIR}/models/log_models/{selected_log_model}", 'rb'))
		if type_of_model == "loglin":
			model_pass_lin = pickle.load(open(f"{ROOT_DIR}/models/lin_models/{selected_lin_model}", 'rb'))

		# Only Open play passes
		df_gp = df_gp[df_gp['chain_type'] == 'open_play']

		# Only passes from attacking possessions
		df_gp = df_gp[df_gp['possession_team_id'] == df_gp['team_id']]

		# Only successful passes
		df_gp = df_gp[(df_gp['outcome'])]

		what_pass_data = st.selectbox("Look at individual matches or look at best passes in a season?", ["Individual Matches", "Season"])

		if what_pass_data == "Individual Matches":
			#Choose League
			#Lookup table for league
			tournament_id_lookup_table = {
				"Allsvenskan": 13,
				"Bundesliga": 3,
				"Serie A": 20,
				"Premier League": 12,
				"La Liga": 21,
				"Norway": 19,
				"Denmark": 18,
			}

			leagues = tournament_id_lookup_table.keys()

			# The id number is converted to its corresponding string to be shown in the selectbox	
			chosen_league = st.selectbox("Choose a league", leagues, label_visibility="collapsed")

			# Convert the string back to the id value which is used for filtering
			chosen_league = tournament_id_lookup_table[chosen_league]

			df_gp = df_gp[df_gp["tournament_id"] == chosen_league]

			match_values = pd.unique(df_gp["match_id"])
			chosen_match = st.selectbox("Choose Match", match_values)

			df_gp = df_gp[df_gp["match_id"] == chosen_match]

			what_to_visualize = st.selectbox("Show both teams or only one team?", ["Both", "One Team"])

			if what_to_visualize == "One Team":
				teams_playing = pd.unique(df_gp["team_id"])
				selected_team = st.selectbox("Show which team?", teams_playing)
				df_gp = df_gp[df_gp["team_id"] == selected_team]
		
		if what_pass_data == "Season":
			#Lookup table for season
			season_id_lookup_table = {
				2020: 86,
				2021: 89,
				2022: 96,
			}			
			seasons = season_id_lookup_table.keys()
			# The id number is converted to its corresponding string to be shown in the selectbox	
			chosen_season = st.selectbox("Choose a season", seasons, label_visibility="collapsed")

			# Convert the string back to the id value which is used for filtering
			chosen_season = season_id_lookup_table[chosen_season]

			df_gp = df_gp[df_gp["season_id"] == chosen_season]

		df_gp = df_gp[df_gp['start_x'] > 0]
		df_gp = df_gp[df_gp['start_y'] > 0]
		df_gp = df_gp[df_gp['end_x'] > 0]
		df_gp = df_gp[df_gp['end_y'] > 0]

		df_gp = feature_creation(df_gp)
		df_gp['const'] = 1

		if type_of_model == "loglin":
			df_gp = bld.__add_features(df_gp, model_pass_log.model.exog_names + model_pass_lin.model.exog_names)
			df_gp['prob_log'] = model_pass_log.predict(df_gp[model_pass_log.model.exog_names])
			df_gp['prob_lin'] = model_pass_lin.predict(df_gp[model_pass_lin.model.exog_names])
			df_gp['prob'] = df_gp['prob_log'] * df_gp['prob_lin']
			df_gp['prob'] = df_gp['prob'].astype(float)
		else:
			df_gp = bld.__add_features(df_gp, model_pass_log.model.exog_names)
			df_gp['prob'] = model_pass_log.predict(df_gp[model_pass_log.model.exog_names])
			df_gp['prob'] = df_gp['prob'].astype(float)
		
		df_gp = df_gp.sort_values(by= ['prob'], ascending=False)
		df_gp['rank'] = range(1, 1+len(df_gp))
		max_value = df_gp["prob"].max()

		#Len of DF
		len_of_df = len(df_gp)

		if 'top_pass' not in st.session_state:
			st.session_state.top_pass = 10
		if st.session_state.top_pass <= len_of_df:
			df_gp = df_gp[st.session_state.top_pass-10:st.session_state.top_pass]
		else:
			df_gp = df_gp[st.session_state.top_pass-10:len_of_df]
		
		st.write(f"**Showing the best {st.session_state.top_pass} passes**")

		#Alt A
		selected_features_log = st.multiselect("Select features for the Logicstic Regression Model (Order Matters)", model_pass_log.model.exog_names)
		if type_of_model == "loglin":
			selected_features_lin = st.multiselect("Select features for the Linear Regression Model (Order Matters)", model_pass_lin.model.exog_names)
		if (type_of_model == "log" and len(selected_features_log) == 5) or (type_of_model == "loglin" and len(selected_features_log) == 5 and len(selected_features_lin) == 5):
			stats_df = pd.DataFrame()
			stats_df["rank"] = df_gp["rank"]
			stats_df["prob"] = df_gp["prob"]
			log_res = []
			for i in range(len(selected_features_log)):
				temp = pd.DataFrame()
				active_feats = []
				for j in range(i+1):
					active_feats.append(selected_features_log[j])
				not_selected_feat = [x for x in model_pass_log.model.exog_names if x not in active_feats]
				for feat in active_feats:
					temp[feat] = df_gp[feat]
				for feat in not_selected_feat:
					temp[feat] = 0
				pred_single = model_pass_log.predict(temp)
				if i == 0:
					stats_df[f"{selected_features_log[i]} (log)"] = pred_single
				else:
					stats_df[f"{selected_features_log[i]} (log)"] = pred_single - stats_df[f"{selected_features_log[i-1]} (log)"]

			if type_of_model == "loglin":
				lin_res = []
				for i in range(len(selected_features_lin)):
					temp = pd.DataFrame()
					active_feats = []
					for j in range(i+1):
						active_feats.append(selected_features_lin[j])
					not_selected_feat = [x for x in model_pass_lin.model.exog_names if x not in active_feats]
					for feat in active_feats:
						temp[feat] = df_gp[feat]
					for feat in not_selected_feat:
						temp[feat] = 0
					pred_single = model_pass_lin.predict(temp)
					if i == 0:
						stats_df[f"{selected_features_lin[i]} (lin)"] = pred_single
					else:
						stats_df[f"{selected_features_lin[i]} (lin)"] = pred_single - stats_df[f"{selected_features_lin[i-1]} (lin)"]


			pitch = Pitch(line_color='black',pitch_type='opta', line_zorder = 2)
			fig, ax = pitch.grid(axis=False)
			for i, row in df_gp.iterrows():
				value = row["rank"]
				#adjust the line width so that the more passes, the wider the line
				line_width = 3
				#get angle
				if (row.end_x - row.start_x) != 0:
					angle = np.arctan((row.end_y - row.start_y)/(row.end_x - row.start_x))*180/np.pi
				else:
					angle = np.arctan((row.end_y - row.start_y)/0.000001)*180/np.pi

				#plot lines on the pitch
				pitch.arrows(row.start_x, row.start_y, row.end_x, row.end_y,
									alpha=0.6, width=line_width, zorder=2, color=get_cmap('Greens')(row.prob/max_value), ax = ax["pitch"])
				#annotate max text
				ax["pitch"].text((row.start_x+row.end_x-8)/2, (row.start_y+row.end_y-4)/2, str(value)[:5], fontweight = "bold", color = "purple", zorder = 4, fontsize = 16, rotation = int(angle))
			ax['title'].text(0.5, 0.5, 'All Data', ha='center', va='center', fontsize=30)
			plt.axis("off")
			st.pyplot(fig)
			next_ranks = []
			prev_ranks = []
			if st.session_state.top_pass > 10:
				prev_ranks = st.button("Previous 10?")
			if st.session_state.top_pass < len_of_df:
				next_ranks = st.button("Next 10?")
			if next_ranks:
				st.session_state.top_pass += 10
				st.experimental_rerun()
			if prev_ranks:
				st.session_state.top_pass -= 10
				st.experimental_rerun()
			top_ten = st.button("Top 10")
			if top_ten:
				st.session_state.top_pass = 10
				st.experimental_rerun()
			
			#Show the table
			st.table(stats_df)
		
		else:
			st.write("Select five features!")
