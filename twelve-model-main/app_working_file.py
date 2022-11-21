
import pickle
import seaborn as sb
import sys
from io import BytesIO
import streamlit as st
import sklearn.metrics as skm
from matplotlib.colors import ListedColormap
from mplsoccer import Pitch
from st_row_buttons import st_row_buttons
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
	selected_sub_page = st_row_buttons(["Visualize Model", "Train a model", "Test a model"])

	if selected_sub_page == 'Visualize Model':

		test_old_model = st.selectbox("Select Passing Model", ["Trained model", "xT Positional", 'xT Action Based'])

		#Setting features True or False
		columns = st.columns(6)
		assist = columns[0].checkbox('Assist')
		cross = columns[1].checkbox('Cross')
		cutback = columns[2].checkbox('Cutback')
		switch = columns[3].checkbox('Switch')
		through_pass = columns[4].checkbox('Through Pass')

		columns = st.columns(3)
		start_x = columns[0].slider('Start x', 0, 100, 50)
		start_y = columns[1].slider('Start y', 0, 100, 50)

		# Load Models
		model_pass_log, model_pass_lin, = get_pass_model()


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
			df[['assist', 'cross', 'pull-back', 'switch',
				'through_pass']] = assist, cross, cutback, switch, through_pass
			df[['time_from_chain_start', 'time_difference']] = 0, 0

			df = bld.__add_features(df, model_pass_log.model.exog_names + model_pass_lin.model.exog_names)

			df['prob_log'] = model_pass_log.predict(df[model_pass_log.model.exog_names])
			df['prob_lin'] = model_pass_lin.predict(df[model_pass_lin.model.exog_names])

			df['prob'] = df['prob_log'] * df['prob_lin']

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
	if selected_sub_page == "Train a model":

		# Session state is used to save the attributes whenever a change is made on the page bu the user, otherwise the list would be reset
		if 'attributes' not in st.session_state:
			st.session_state.attributes = []
		if "active_attributes" not in st.session_state:
			st.session_state.active_attributes = []

		#Output name for model
		model_type = st.selectbox("Select Model Type", ["Linear Regression Model", "Logistic Regression Model"])
		model_name = "model.sav"
		model_name_input = st.text_input("Model Name")
		if model_name_input:
			model_name = model_name_input + ".sav"

		st.write("**Filter data by attributes**")
		# Load Dataset
		df_train = load_dataset()

		# Get all columns
		all_columns = [x for x, y in df_train.dtypes.items() if 'id' not in x]

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

		st.write("Filter True and False attributes")
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

		st.write("Filter float attributes")
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

		st.write("Filter categorical attributes")
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
				

		# Show dataset descriptions
		with st.expander("Dataset",False):
			st.write(df_train.shape)
			st.dataframe(df_train.describe())

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

		# Use only selected features
		df_train = df_train[st.session_state.attributes]

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
		train = st.button("Train the Model")
		if train:
			if model_type == "Linear Regression Model":
				pass

			if model_type == "Logistic Regression Model":
				pass
	
	if selected_sub_page == "Test a model":
		# Load Models
		model_pass_log, model_pass_lin, = get_pass_model()

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
		Ylog_pred = model_pass_log.predict(Xlog).values
		Ylin_pred = model_pass_lin.predict(Xlin).values

		#ROC Curve, score and RMSE
		fpr, tpr, weight = skm.roc_curve(Ylog, Ylog_pred)
		auc = skm.auc(fpr, tpr)
		Ylog_pred_bin = [round(elements) for elements in Ylog_pred]
		score = skm.accuracy_score(Ylog, Ylog_pred_bin)
		rmse = skm.mean_squared_error(Ylin, Ylin_pred)
		rmse = np.sqrt(rmse)
		
		#Plot result, columns scales the result down
		roc1, roc2, roc3 = st.columns([2, 5, 2])
		with roc1:
			st.write("")
		
		with roc2:
			fig = plt.figure()
			plt.plot(fpr, tpr, color="orange", label="ROC Curve")
			plt.plot([0, 1], [0, 1], color="blue", label="Random Guess", linestyle="--")
			plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
			plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
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