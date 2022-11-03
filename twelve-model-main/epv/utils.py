OPTA_MAPPING_EVENTS = {
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


DATA_COLUMNS = ['id', 'match_id', 'tournament_id', 'chain_id', 'possession_index', 'team_id', 'player_id',
                 'possession_team_id', 'event_index', 'time_difference', 'time_from_chain_start', 'start_x', 'start_y',
                 'end_x', 'end_y', 'end_y_adj', 'start_y_adj', 'outcome', 'datetime', 'chain_xG', 'chain_goal',
                 'chain_shot', 'chain_start_type_id', 'chain_type', 'prev_event_type', 'prev_event_team', 'cross',
                 'head_pass', 'through_pass', 'freekick_pass', 'corner_pass', 'throw-in', 'chipped', 'lay-off',
                 'launch', 'flick-on', 'pull-back', 'switch', 'pass_length', 'pass_length_test', 'pass_angle',
                 'assist', '2nd_assist', 'in-swing', 'out-swing', 'straight', 'overhit_cross', 'driven_cross',
                 'floated_cross', 'pass_length_diff']