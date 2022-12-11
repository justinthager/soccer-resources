credit = """
   /$$$$$$$$/$$$$$$$ /$$              /$$$$$$  /$$$$$$  /$$$$$$  /$$$$$$ /$$$$$$$$/$$$$$$$
  | $$_____| $$__  $| $$             /$$__  $$/$$__  $$/$$__  $$/$$__  $| $$_____| $$__  $$
  | $$     | $$  \ $| $$            | $$  \__| $$  \ $| $$  \__| $$  \__| $$     | $$  \ $$
  | $$$$$  | $$$$$$$| $$            |  $$$$$$| $$  | $| $$     | $$     | $$$$$  | $$$$$$$/
  | $$__/  | $$____/| $$             \____  $| $$  | $| $$     | $$     | $$__/  | $$__  $$
  | $$     | $$     | $$             /$$  \ $| $$  | $| $$    $| $$    $| $$     | $$  \ $$
  | $$$$$$$| $$     | $$$$$$$$      |  $$$$$$|  $$$$$$|  $$$$$$|  $$$$$$| $$$$$$$| $$  | $$
  |________|__/     |________/       \______/ \______/ \______/ \______/|________|__/  |__/

## Authors: Claire Gillaspy, Justin Hager, and Salem Wear ##
"""
"""
# This project was started in May 2021 as a part of Furman
#   University's Summer Undergraduate Math Research program.
#   The code held in this script is an adaptation and
#   improvement of a process that was previously contained in
#   MATLAB code and .xlsx files. The data are now stored in a
#   SQL Server and the code is executed in Python.
#
# python3 "/Users/justinhager/Furman University/Soccer Repository/soccer/Python Code/soccer.py"
# python3 "/Users/justinhager/Furman University/Soccer Repository/soccer/Python Code/all_soccer.py"
"""

############################################################
# OPTIONS
############################################################
##############################
# CODE OPTIONS
##############################

add_goal_multiplier = True
add_save_multiplier = False
scale_by_appearances = True
season_specific_metrics = False
weighting_system = 'skill'  # 'default', 'skill', or 'position'
goal_baseline = 3
save_baseline = 3

##############################
# EXPORT OPTIONS
##############################
export_game_data = True
export_goal_data = True
export_pass_network = True
export_directed_graphs = False

##############################
# OTHER OPTIONS
##############################
install_packages = False
pass_protected = False
host = 'localhost'
port = '5432'
db = 'postgres'


############################################################
# Main Method
############################################################
def main():
    banner('Running the main method...')
    data_pre_process()
    data_import()
    init_result_tables()
    execution()
    centrality()
    dynamic_duos()
    data_post_process()
    data_export()


############################################################
# Import the Required Packages
############################################################
import os
import sys

if install_packages:
    print('Installing missing Python packages...')
    packages = ['pandas', 'sqlalchemy', 'numpy', 'alive_progress', 'math', 'getpass_asterisk', 'psycopg2-binary',
                'openpyxl', 'scipy', 'fuzzywuzzy', 'psycopg2', 'time', 'datetime', 'python-Levenshtein', 'getkey',
                'networkx', 'matplotlib']
    try:
        os.system('export PATH=$PATH:/Applications/Postgres.app/Contents/Versions/latest/bin')
        os.system('pip3 install --upgrade pip')
        for i in packages:
            os.system(f'python3 -m pip install {i}')
    except Exception as code_error:
        print('Could not install Python packages')
        sys.exit()

import pandas as pd
from sqlalchemy import create_engine
import numpy
from numpy.linalg import eig
from alive_progress import alive_bar
import time
from os import system, name
import getpass
from getpass_asterisk.getpass_asterisk import getpass_asterisk
from scipy import stats
import math
import networkx as nx
import matplotlib.pyplot as plt
from fuzzywuzzy import process

# from sklearn import svm
pd.options.mode.chained_assignment = None
ln = numpy.log


############################################################
# Define Connections and Global Variables
############################################################
def banner(text, size=60, offset=0):
    text = str(text)
    y = round(((size - 4) - len(text)) / 2)
    if len(text) % 2 == 0:
        z = 0
    else:
        z = 1
    print('\n' + ' ' * offset + '#' * size + '\n' + ' ' * offset + '##' + ' ' * y + text + ' ' * (
            y - z) + '##' + '\n' + ' ' * offset + '#' * size + '\n ')


def print_error(message, error=''):
    x = input('\n## ' + str(message) + ' ##\n')
    if x == 'exit':
        print('\n## Exiting script... ##\n')
        time.sleep(1)
        sys.exit()
    elif x == 'error':
        print(error)
        x = input()
        if x == 'exit':
            print('\n## Exiting script... ##\n')
            time.sleep(1)
            sys.exit()
        else:
            title()
    else:
        title()


def fuzzy_merge(df_1, df_2, key1, key2, threshold=90, limit=2):
    s = df_2[key2].tolist()
    m = df_1[key1].apply(lambda x: process.extract(x, s, limit=limit))
    df_1['matches'] = m
    m2 = df_1['matches'].apply(lambda x: ', '.join([i[0] for i in x if i[1] >= threshold]))
    df_1['matches'] = m2
    return df_1


def login():
    banner('Login to Postgres SQL Server')
    global engine, server
    if pass_protected:
        username = getpass.getuser()
        try:
            password = getpass_asterisk(
                f'Attempting to connect to local SQL server on \'{username}\'\nPlease enter server password: ')
        except Exception as code_error:
            print_error('Incorrect password', code_error)
            title()
            login()
        server = f'postgresql://{username}:{password}@{host}:{port}/{db}'
        server_nopass = f'postgresql://{username}:' + '*' * len(password) + f'@{host}:{port}/{db}'
        try:
            engine = create_engine(server)
            engine.connect()
            print(f'Connecting to local SQL sever on:\n    \'{server_nopass}\'...')
            main()
        except Exception as code_error:
            print_error('Incorrect password', error=code_error)
            title()
            login()

    else:
        # connection override
        server = 'postgresql://justinhager:Tyler2001@localhost:5432/postgres'
        server_nopass = f'postgresql://justinthager:' + '*' * 9 + '@localhost:5432/postgres'
        try:
            engine = create_engine(server)
            engine.connect()
            print(f'\nConnecting to local SQL sever on\n    ## {server_nopass} ##')
        except Exception as code_error:
            print_error('Invalid server credentials', error=code_error)
        main()


def postgres_connect():
    try:
        global engine
        engine = create_engine(server)
        engine.connect()
    except Exception as code_error:
        print_error('Could not connect to server', error=code_error)


############################################################
# Data Pre-Processing
############################################################
def data_pre_process():
    banner('Selecting and Pre-processing game data')
    global season_data, start, skill_based_weights, position_based_weights, table_name

    if weighting_system == 'default':
        skill_based_weights = False
        position_based_weights = False
    elif weighting_system == 'skill':
        skill_based_weights = True
        position_based_weights = False
    elif weighting_system == 'position':
        skill_based_weights = False
        position_based_weights = True
    else:
        print_error('Invalid Weighting System')

    postgres_connect()
    select_filenames = """SELECT x.table_name FROM information_schema.tables as x 
        WHERE x.table_schema = 'data' and x.table_name not like 'player_information';"""
    table_names = list(pd.read_sql(select_filenames, engine)['table_name'])
    check_player_information = """SELECT x.table_name FROM information_schema.tables as x 
        WHERE x.table_schema = 'data' and x.table_name like 'player_information';"""
    check_player_information_list = list(pd.read_sql(check_player_information, engine)['table_name'])
    if len(check_player_information_list) == 0:
        print_error('Plese import the \'player_information\' table')

    start = time.time()
    run_all_data = False
    all_data = table_names

    try:
        run_all_data = bool(sys.argv[2])
    except:
        pass

    if run_all_data:
        x = int(sys.argv[1])
    else:
        table_names_string = '\n## select a dataset from below: ##\n'
        num = 1
        for name in table_names:
            table_names_string += '   # ' + str(num) + ' - ' + name + '\n'
            num += 1
        table_names_string += '\n## use game data: '
        x = int(input(table_names_string)) - 1

    try:
        season_data = all_data[x]
    except Exception as code_error:
        print_error('Game Data selection out of range', code_error)


############################################################
# Data Imports
############################################################
def data_import():
    banner('Importing \'' + season_data + '\' data')

    global game_data, goals, players, teams, web_data, player_appearances, radius_sd, theta_sd, player_information
    print('Declaring SQL commands...')

    select_players = """select x.player_name,
           x.player_id,
           x.team_name,
           x.team_id,
           row_number() over (order by x.team_name asc, x.player_name asc) as player_index
    from postgres.data.""" + season_data + """ as x
    where x.player_id is not null and x.player_name not like ' ' and x.owngoal_ind not like 'Y'
    group by x.player_name, x.player_id, x.team_name, x.team_id
    order by x.team_name asc, x.player_name asc"""

    select_players_complete = select_players + ';'

    select_teams = """select x.team_name,
           x.team_id,
           row_number() over () as team_index,
           min(y.player_index) as player_id_min,
           max(y.player_index) as player_id_max,
           max(y.player_index)-min(y.player_index)+1 as team_size
    from postgres.data.""" + season_data + """ as x
    left join (""" + select_players + """) as y on x.team_id = y.team_id
    where x.player_id is not null and x.player_name not like ' ' and x.owngoal_ind not like 'Y'
    group by x.team_name, x.team_id
    order by x.team_name asc;"""

    select_game_data = """select x.*,
           y.player_index as player_index
           from (select x.game_date, x.event_id, x.event_desc, x.play_typ, x.play_outcome_typ,
               x.time_txt, x.owngoal_ind, x.team_id, x.team_name, x.player_id, x.player_name,
               x.x_from, x.y_from,
               case when (x.play_typ = 'Goal' and x.owngoal_ind = 'N') or (x.play_typ = 'PenaltyGoal') then 100
                    when (x.play_typ = 'Goal' and x.owngoal_ind = 'Y') then 0 else x.x_to end as x_to,
               case when (x.play_typ = 'Goal' and x.owngoal_ind = 'N') or (x.play_typ = 'PenaltyGoal') then 50
                    when (x.play_typ = 'Goal' and x.owngoal_ind = 'Y') then 50 else x.y_to end as y_to,
               row_number() over () as play_index
               from postgres.data.""" + season_data + """ as x
               order by x.game_date asc, x.event_id, x.time_txt asc) as x
           left join (""" + select_players + """) as y on x.player_id = y.player_id and x.team_id = y.team_id
           where x.player_id is not null and x.player_name not like ' ' and x.owngoal_ind not like 'Y'
           order by x.play_index asc;"""

    # select_web_data = """select x.* from postgres.sandbox.web_data as x;"""

    select_appearances = """select x.player_name, x.team_name, sum(x.appearances) as appearances
from (select x.player_name, x.team_name, count(distinct x.event_id) as appearances
      from postgres.data.""" + season_data + """ as x
      where x.player_name is not null and x.owngoal_ind not like 'Y'
        and x.player_name not like ' '
      group by x.player_name, x.team_name, x.event_id) as x
group by x.player_name, x.team_name;"""

    select_team_appearances = """select x.player_name, string_agg(x.team_name, ', ') as count
from (select x.player_name, x.team_name, sum(x.appearances) as appearances
from (select x.player_name, x.team_name, count(distinct x.event_id) as appearances
from postgres.data.""" + season_data + """ as x
where x.player_name is not null and x.owngoal_ind not like 'Y'
      and x.player_name not like ' '
group by x.player_name, x.team_name, x.event_id) as x
group by x.player_name, x.team_name) as x
group by x.player_name;"""

    select_goal_data = """select x.player_name, x.team_name, x.play_typ, x.x_from, x.y_from,
                    sqrt((50 - x.y_from)^2 + (99.99 - x.x_from)^2) as radius,
                    atan((x.y_from-50) / (99.99 - x.x_from)) as theta
            from postgres.data.""" + season_data * season_specific_metrics + 'combined_data' * (
                1 - season_specific_metrics) + """ as x
            where x.play_typ like 'Goal' and x.owngoal_ind like 'N';"""

    if season_data == 'epl_man_city_watford':
        season_data2 = 'epl_2019_2020'
    else:
        season_data2 = season_data

    select_player_information = """select x.* from (select x.*, x.player as player_name, x.team as team_name,
            x.annual_gross_base_salary_gbp as salary_gbp,
            case when x.league like 'Premier League' then concat('epl_', x.season)
                 when x.league like 'Ligue 1' then concat('french_ligue_', x.season)
                 when x.league like 'La Liga' then concat('la_liga_', x.season)
                 when x.league like 'Bundesliga' then concat('german_bundesliga_', x.season)
                 when x.league like 'Serie A' then concat('italian_serie_', x.season)
                 else concat(x.league, '_', x.season)
            end as league_year
            from postgres.data.player_information as x) as x
            where league_year like '""" + season_data2 + """'
            and x.player is not null;"""

    postgres_connect()

    # importing the tables we need from the SQL statements below
    try:
        print('Importing Game Data...')
        game_data = pd.read_sql(select_game_data, engine)
        print('Importing Goal Data...')
        goals = pd.read_sql(select_goal_data, engine)
        print('Importing Player Data...')
        players = pd.read_sql(select_players_complete, engine)
        player_information = pd.read_sql(select_player_information, engine)
        print('Importing Team Data...')
        teams = pd.read_sql(select_teams, engine)
        print('Importing Appearance Data...')
        player_appearances = pd.read_sql(select_appearances, engine)
        team_appearances = pd.read_sql(select_team_appearances, engine)
        # print('Importing EPL Web Data...')
        # web_data = pd.read_sql(select_web_data, engine)
    except Exception as code_error:
        print_error('Data could not be imported', error=code_error)
        title()

    print('Joining and Merging Data Tables...')
    players = players.merge(player_appearances, on=['player_name', 'team_name'], how='left')
    players = players.rename(columns={'count': 'appearances'})
    players = players.merge(team_appearances, on=['player_name'], how='left')
    players = players.rename(columns={'count': 'team_appearances'})
    game_data = game_data.merge(teams[['team_id', 'team_index']], on='team_id', how='left')

    players = fuzzy_merge(players, player_information, 'player_name', 'player_name', limit=1)
    players = players.rename(columns={'matches': 'player_name_matches'})
    players = fuzzy_merge(players, player_information, 'team_name', 'team_name', limit=1)
    players = players.rename(columns={'matches': 'team_name_matches'})

    player_information = player_information.drop_duplicates(subset=['player_name', 'season', 'team_name'])
    player_information = player_information.rename(
        columns={'player_name': 'player_name_matches', 'team_name': 'team_name_matches'})
    players = players.merge(player_information, on=['player_name_matches', 'team_name_matches'], how='left')

    players = players.drop(
        columns=['player_name_matches', 'season', 'league', 'league_year', 'team_name_matches',
                 'weekly_gross_base_salary_gbp', 'adj_current_gross_base_salary_gbp', 'estimated_gross_total_gbp',
                 'current_contract_status', 'current_contract_expiration', 'current_contract_length',
                 'outfielder_goalkeeper', 'annual_gross_base_salary_gbp', 'player', 'team'], errors='ignore')

    print('Calculating Goal radius & theta Standard Error...')
    radius = list(goals['radius'])
    radius_mean = numpy.mean(goals['radius'])
    radius.extend(list(goals['radius'] * -1))
    radius_sd = numpy.std(radius)

    theta_mean = numpy.mean(goals['theta'])
    theta_sd = numpy.std(goals['theta'])

    print(f'\n## {season_data} goal stats:'
          f'\n    # radius \u03BC: {round(radius_mean, 4)}'
          f'\n    # theta  \u03BC: {round(radius_sd, 4)}'
          f'\n    # radius \u03C3: {round(theta_mean, 4)}'
          f'\n    # theta  \u03C3: {round(theta_sd, 4)}')


############################################################
# Create all the tables
############################################################
def init_result_tables():
    banner('Initializing the Results Tables...')
    global pass_matrix, offense_score_results, recorded_goals, own_goals, own_goals_count, offense_plays, \
        offense_possessions, defense_possessions, defense_plays, defense_score_results, goal_tracker, goal_columns, \
        failed_passes, completed_passes
    n = len(players)
    m = len(teams)
    numpy.seterr(divide='ignore')

    # creating an empty matrix to store all the pass data
    pass_matrix = numpy.zeros((n, n))
    failed_passes = numpy.zeros(n)
    completed_passes = numpy.zeros(n)

    # table to store goals and offense score
    offense_score_results = numpy.zeros(n)
    offense_plays = numpy.zeros(n)
    defense_score_results = numpy.zeros(n)
    defense_plays = numpy.zeros(n)
    recorded_goals = numpy.zeros(n)
    own_goals = numpy.zeros(n)

    offense_possessions = numpy.zeros(m)
    defense_possessions = numpy.zeros(m)

    game_data['home_team_score'] = 0
    game_data['away_team_score'] = 0
    game_data['score_diff'] = 0
    game_data['seconds'] = 0
    game_data['off_loc_score'] = 0
    game_data['def_loc_score'] = 0
    game_data['tot_loc_score'] = 0

    # creating empty data frame to score goal data goal_tracker = pd.DataFrame(columns = ['player_name', 'player_id',
    # 'team_name', 'team_id', 'play_typ', 'time', 'loc_score', 'gm'])
    goal_columns = ['event_desc', 'player_name', 'team_name', 'play_typ', 'time', 'before_goal',
                    'after_goal', 'score_diff', 'goal_bonus', 'loc_score', 'total_bonus']
    goal_tracker = pd.DataFrame(
        columns=goal_columns)


############################################################
# Defining our important Math functions
############################################################


def score(index):
    try:
        if game_data.event_id[index] == game_data.event_id[index - 1]:
            game_data.home_team_score[index] = game_data.home_team_score[index - 1]
            game_data.away_team_score[index] = game_data.away_team_score[index - 1]

        if game_data.play_typ[index] == 'Goal' and game_data.owngoal_ind[index] == 'N' or \
                game_data.play_typ[index] == 'PenaltyGoal':
            if game_data.team_name[index] == game_data.event_desc[index].split(' @ ')[1]:
                game_data.home_team_score[index] += 1
            elif game_data.team_name[index] == game_data.event_desc[index].split(' @ ')[0]:
                game_data.away_team_score[index] += 1

        elif game_data.play_typ[index] == 'Goal' and game_data.owngoal_ind[index] == 'Y':
            if game_data.team_name[index] == game_data.event_desc[index].split(' @ ')[0]:
                game_data.home_team_score[index] += 1
            elif game_data.team_name[index] == game_data.event_desc[index].split(' @ ')[1]:
                game_data.away_team_score[index] += 1

        game_data.score_diff[index] = game_data.home_team_score[index] - game_data.away_team_score[index]
    except Exception as code_error:
        print_error('Game Score Error on line ' + str(index), code_error)


def game_time(index):
    try:
        hundreds = 0
        if len(game_data.time_txt[index]) > 6:
            hundreds, minutes, seconds = game_data.time_txt[index].split(':')
        else:
            minutes, seconds = game_data.time_txt[index].split(':')
        return int(minutes) * 60 + int(hundreds) * 60 + int(seconds)
    except Exception as code_error:
        print_error('Game Time Error on line ' + str(index), code_error)


def game_time_itt(index):
    hundreds = 0
    if len(game_data.time_txt[index]) > 6:
        hundreds, minutes, seconds = game_data.time_txt[index].split(':')
    else:
        minutes, seconds = game_data.time_txt[index].split(':')
    return int(minutes) * 60 + int(hundreds) * 60 + int(seconds)


def k(x, a, b, c, d):  # scales inputs from [a,b] to outputs on [c,d]
    return (x * (d - c) + c * b - a * d) / (b - a)


def f(x, t, n1, n2, n3, n4):  # calculates the multiplier at a given x
    if 0 <= x < (t / 2):
        return k(x, 0, (t / 2), n1, n2)
    elif (t / 2) <= x <= t:
        return k(x, (t / 2), t, n2, n3)
    elif t < x:
        return n4


def g(x, t0, t, n1, n2, n3, n4):  # calculates the goal multiplier
    # s1 is the difference in score
    s1 = 2
    # s0 is the fraction of the total given at s1
    s0 = 1.25

    return f(t0, t, n1, n2, n3, n4) * math.exp(-((x ** 2) * ln(s1)) / (s0 ** 2)) + 1


def goal_multiplier(index):
    t = 90 * 60
    t0 = game_time(index)
    x = game_data.score_diff[index - 1]

    n1 = 1  # n1: GM at 0 minutes
    n2 = 3  # n2: GM at 45 minutes
    n3 = 8  # n3: GM at 90 minutes
    n4 = 9  # n4: GM at 90+ minutes

    return g(x, t0, t, n1, n2, n3, n4)


def location_score(ri, rf, ti, tf, rsd, tsd):
    norm_dist_ri = math.exp((-1 / 2) * (ri / rsd) ** 2)
    norm_dist_rf = math.exp((-1 / 2) * (rf / rsd) ** 2)
    norm_dist_tf = math.exp((-1 / 2) * (tf / tsd) ** 2)
    norm_dist_ti = math.exp((-1 / 2) * (ti / tsd) ** 2)
    loc_score = (norm_dist_rf * norm_dist_tf) - (norm_dist_ri * norm_dist_ti)

    return loc_score


def initial_location_score(ri, ti, rsd, tsd):
    norm_dist_ri = math.exp((-1 / 2) * (ri / rsd) ** 2)
    norm_dist_ti = math.exp((-1 / 2) * (ti / tsd) ** 2)
    initial_loc_score = norm_dist_ri * norm_dist_ti

    return initial_loc_score


def offense_score(index):
    goal_tracker_list = [''] * len(goal_columns)
    try:
        if game_data.play_typ[index] == 'Pass':
            if game_data.play_outcome_typ[index] == 'Success':
                if game_data.play_typ[index - 1] == 'Goal' or game_data.play_typ[index - 1] == 'Miss' or \
                        game_data.play_typ[index - 1] == 'Out' or game_data.play_typ[index - 1] == 'Pickup' or \
                        game_data.play_typ[index - 1] == 'Save' or game_data.play_typ[index - 1] == 'Smother' or \
                        game_data.play_typ[index - 1] == 'SubstitutionIn' or \
                        game_data.play_typ[index - 1] == 'SubstitutionOut' or \
                        game_data.play_typ[index - 1] == 'OffsideProvoked':
                    initial_radius = math.sqrt(
                        (50 - game_data.y_from[index]) ** 2 + (99.99 - game_data.x_from[index - 1]) ** 2)
                    initial_theta = math.atan((game_data.y_from[index] - 50) / (99.99 - game_data.x_from[index]))
                    final_radius = math.sqrt((50 - game_data.y_to[index]) ** 2 + (99.99 - game_data.x_to[index]) ** 2)
                    final_theta = math.atan((game_data.y_to[index] - 50) / (99.99 - game_data.x_to[index]))

                    pass_loc_score = location_score(initial_radius, final_radius, initial_theta, final_theta, radius_sd,
                                                    theta_sd)
                    dribble_loc_score = 0

                else:
                    dribble_initial_radius = math.sqrt(
                        (50 - game_data.y_to[index]) ** 2 + (99.99 - game_data.x_to[index - 1]) ** 2)
                    dribble_initial_theta = math.atan(
                        (game_data.y_to[index - 1] - 50) / (99.99 - game_data.x_to[index - 1]))
                    dribble_final_radius = math.sqrt(
                        (50 - game_data.y_from[index]) ** 2 + (99.99 - game_data.x_from[index]) ** 2)
                    dribble_final_theta = math.atan((game_data.y_from[index] - 50) / (99.99 - game_data.x_from[index]))

                    pass_initial_radius = math.sqrt(
                        (50 - game_data.y_from[index]) ** 2 + (99.99 - game_data.x_from[index]) ** 2)
                    pass_initial_theta = math.atan((game_data.y_from[index] - 50) / (99.99 - game_data.x_from[index]))
                    pass_final_radius = math.sqrt(
                        (50 - game_data.y_to[index]) ** 2 + (99.99 - game_data.x_to[index]) ** 2)
                    pass_final_theta = math.atan((game_data.y_to[index] - 50) / (99.99 - game_data.x_to[index]))
                    pass_loc_score = location_score(pass_initial_radius, pass_final_radius, pass_initial_theta,
                                                    pass_final_theta, radius_sd, theta_sd)
                    dribble_loc_score = location_score(dribble_initial_radius, dribble_final_radius,
                                                       dribble_initial_theta, dribble_final_theta, radius_sd, theta_sd)

            else:
                if game_data.play_typ[index - 1] == 'Goal' or game_data.play_typ[index - 1] == 'Miss' or \
                        game_data.play_typ[index - 1] == 'Out' or game_data.play_typ[index - 1] == 'Pickup' or \
                        game_data.play_typ[index - 1] == 'Save' or game_data.play_typ[index - 1] == 'Smother' or \
                        game_data.play_typ[index - 1] == 'SubstitutionIn' or \
                        game_data.play_typ[index - 1] == 'SubstitutionOut' or \
                        game_data.play_typ[index - 1] == 'OffsideProvoked':
                    pass_loc_score = 0
                    dribble_loc_score = 0
                else:
                    initial_radius = math.sqrt(
                        (50 - game_data.y_to[index - 1]) ** 2 + (99.99 - game_data.x_to[index - 1]) ** 2)
                    initial_theta = math.atan((game_data.y_to[index - 1] - 50) / (99.99 - game_data.x_to[index - 1]))
                    final_radius = math.sqrt((50 - game_data.y_to[index]) ** 2 + (99.99 - game_data.x_to[index]) ** 2)
                    final_theta = math.atan((game_data.y_to[index] - 50) / (99.99 - game_data.x_to[index]))

                    dribble_loc_score = location_score(initial_radius, final_radius, initial_theta, final_theta,
                                                       radius_sd, theta_sd)
                    pass_loc_score = 0

        elif (game_data.play_typ[index] == 'Goal' and game_data.owngoal_ind[index] == 'N') or \
                game_data.play_typ[index] == 'PenaltyGoal':
            dribble_initial_radius = math.sqrt(
                (50 - game_data.y_to[index - 1]) ** 2 + (99.99 - game_data.x_to[index - 1]) ** 2)
            dribble_initial_theta = math.atan((game_data.y_to[index - 1] - 50) / (99.99 - game_data.x_to[index - 1]))
            # dribble_final_radius = math.sqrt((50 - game_data.y_from[index]) ** 2 + (99.99 - game_data.x_from[
            # index]) ** 2) dribble_final_theta = math.atan((game_data.y_from[index] - 50) / (99.99 -
            # game_data.x_from[index]))
            dribble_final_radius = 0
            dribble_final_theta = 0

            pass_initial_radius = math.sqrt(
                (50 - game_data.y_from[index]) ** 2 + (99.99 - game_data.x_from[index]) ** 2)
            pass_initial_theta = math.atan((game_data.y_from[index] - 50) / (99.99 - game_data.x_from[index]))
            # pass_final_radius = math.sqrt((50 - game_data.y_to[index]) ** 2 + (99.99 - game_data.x_to[index]) ** 2)
            # pass_final_theta = math.atan((game_data.y_to[index] - 50) / (99.99 - game_data.x_to[index]))
            pass_final_radius = 0
            pass_final_theta = 0

            pass_loc_score = location_score(pass_initial_radius, pass_final_radius, pass_initial_theta,
                                            pass_final_theta, radius_sd, theta_sd)
            dribble_loc_score = location_score(dribble_initial_radius, dribble_final_radius, dribble_initial_theta,
                                               dribble_final_theta, radius_sd, theta_sd)

            recorded_goals[game_data.player_index[index] - 1] += 1

            offense_score_results[game_data.player_index[index] - 1] += goal_multiplier(
                index) * add_goal_multiplier + goal_baseline

            score_format = '{} to {}'
            goal_tracker_list[0] = game_data.event_desc[index]
            goal_tracker_list[1] = game_data.player_name[index]
            goal_tracker_list[2] = game_data.team_name[index]
            goal_tracker_list[3] = game_data.play_typ[index]
            goal_tracker_list[4] = round(game_time(index) / 60, 2)
            goal_tracker_list[5] = score_format.format(game_data.away_team_score[index - 1],
                                                       game_data.home_team_score[index - 1])
            goal_tracker_list[6] = score_format.format(game_data.away_team_score[index],
                                                       game_data.home_team_score[index])
            goal_tracker_list[7] = game_data.score_diff[index - 1]
            goal_tracker_list[8] = str(round(goal_multiplier(index), 2) * add_goal_multiplier)
            goal_tracker_list[9] = str(round(3 + pass_loc_score + dribble_loc_score, 3))
            goal_tracker_list[10] = str(round(3 + pass_loc_score + dribble_loc_score + goal_multiplier(index), 3))
            goal_tracker.loc[len(goal_tracker)] = goal_tracker_list

        else:
            pass_loc_score = 0
            dribble_loc_score = 0

        if game_data.owngoal_ind[index] == 'N':
            if pass_loc_score > 0:
                offense_score_results[game_data.player_index[index] - 1] += pass_loc_score
                offense_plays[game_data.player_index[index] - 1] += 1
                game_data.off_loc_score[index] += pass_loc_score

            if dribble_loc_score > 0:
                offense_score_results[game_data.player_index[index] - 1] += dribble_loc_score
                offense_plays[game_data.player_index[index] - 1] += 1
                game_data.off_loc_score[index] += dribble_loc_score

            if game_data.play_typ[index] == 'PenaltyGoal':
                offense_score_results[game_data.player_index[index] - 1] += 0.75
                game_data.off_loc_score[index] += 0.75

            if game_data.play_typ[index] == 'AttemptSaved' and game_data.play_typ[index + 2] == 'ShotFaced':
                offense_score_results[game_data.player_index[index] - 1] -= 0.25
                game_data.off_loc_score[index] += -0.25

        elif game_data.owngoal_ind[index] == 'Y':
            own_goals[game_data.player_index[index] - 1] += 1

    except Exception as code_error:
        print(index)
        print_error('Office Score Error on line ' + str(index), code_error)


def defense_score(index):
    try:
        if not (game_data.play_typ[index] == 'ShotFaced' and game_data.play_outcome_typ[index] == 'Success'):
            neg_x_from = 99.99 - game_data.x_from[index]
            neg_y_from = 99.99 - game_data.y_from[index]

            neg_initial_radius = math.sqrt((50 - neg_y_from) ** 2 + (99.99 - neg_x_from) ** 2)
            neg_initial_theta = math.atan((neg_y_from - 50) / (99.99 - neg_x_from))
            neg_loc_score = initial_location_score(neg_initial_radius, neg_initial_theta, radius_sd, theta_sd)

            pos_initial_radius = math.sqrt((50 - game_data.y_from[index]) ** 2 + (99.99 - game_data.x_from[index]) ** 2)
            pos_initial_theta = math.atan((game_data.y_from[index] - 50) / (99.99 - game_data.x_from[index]))
            pos_loc_score = initial_location_score(pos_initial_radius, pos_initial_theta, radius_sd, theta_sd)

            new_loc_score = max(pos_loc_score + neg_loc_score, 0.01)
            defense_score_results[game_data.player_index[index] - 1] += new_loc_score
            defense_plays[game_data.player_index[index] - 1] += 1
            game_data.def_loc_score[index] += new_loc_score

        if game_data.play_typ[index] == 'Save':
            if game_data.play_typ[index + 1] == 'ShotFaced':
                defense_score_results[game_data.player_index[index] - 1] += 0.75
                game_data.def_loc_score[index] += 0.75

                # defense_score_results[game_data.player_index[index] - 1] += goal_multiplier(index) * add_save_multiplier + save_baseline
                # game_data.def_loc_score[index] += goal_multiplier(index) * add_save_multiplier + save_baseline

            else:
                defense_score_results[game_data.player_index[index] - 1] += 0.5
                game_data.def_loc_score[index] += 0.5

                # defense_score_results[game_data.player_index[index] - 1] += goal_multiplier(index)/2 * add_save_multiplier + save_baseline
                # game_data.def_loc_score[index] += goal_multiplier(index)/2 * add_save_multiplier + save_baseline


        elif game_data.play_typ[index] == 'ShotFaced' and game_data.play_typ[index - 1] == 'PenaltyGoal':
            defense_score_results[game_data.player_index[index] - 1] -= 0.25
            game_data.def_loc_score[index] -= 0.25

            # defense_score_results[game_data.player_index[index] - 1] -= goal_multiplier(index)/3 * add_save_multiplier + save_baseline
            # game_data.def_loc_score[index] -= goal_multiplier(index)/3 * add_save_multiplier + save_baseline
    except Exception as code_error:
        print_error('Defense Score Error on line ' + str(index), code_error)


def control_score(index):
    try:
        if game_data.play_typ[index] == 'Pass' and game_data.play_outcome_typ[index] == 'Success' and \
                game_data.team_id[index] == game_data.team_id[index + 1]:
            pass_matrix[game_data.player_index[index] - 1][game_data.player_index[index + 1] - 1] += 1
            completed_passes[game_data.player_index[index] - 1] += 1
        elif game_data.play_typ[index] == 'Pass' and game_data.play_outcome_typ[index] == 'Fail':
            failed_passes[game_data.player_index[index] - 1] += 1
    except Exception as code_error:
        print_error('Centrality Error on line ' + str(index), code_error)


def possession(index):
    try:
        if (game_data.play_typ[index] == 'Pass' or game_data.play_outcome_typ[index] == 'Goal') and \
                game_data.team_id[index - 1] != game_data.team_id[index]:
            offense_possessions[game_data.team_index[index] - 2] += 1
            if game_data.event_id[index - 1] == game_data.event_id[index]:
                defense_possessions[game_data.team_index[index] - 1] += 1
    except Exception as code_error:
        print_error('Possession Error on line ' + str(index), code_error)


############################################################
# Execute the play by play calculations
############################################################
def execution():
    n = len(game_data) - 1
    m = "{:,}".format(n - 1)
    banner('Iterating through ' + m + ' lines of game data... ')
    with alive_bar(n - 1, force_tty=True) as bar:
        for i in range(1, n):
            try:
                try:
                    game_data.seconds[i] = game_time_itt(index=i)
                except:
                    continue
                score(index=i)
                offense_score(index=i)
                defense_score(index=i)
                control_score(index=i)
                possession(index=i)
                time.sleep(0.000001)
                bar()
            except Exception as code_error:
                print_error('Iteration Error on line ' + str(i), code_error)


############################################################
# Calculation Eigenvector Centrality from Adjacency Matrix
############################################################
def centrality():
    banner('Calculating Centrality...')
    try:
        global norm_eigenvectors_list, norm_eigenvectors_list_z, team_pass_matrix, team_pass_matrix_nonnorm, norm_eigenvectors, eigenvalues, eigenvectors
        team_pass_matrix = numpy.zeros((len(teams), max(teams.team_size), max(teams.team_size)))
        eigenvalues = numpy.zeros((max(teams.team_size), len(teams)), dtype='complex_')
        eigenvectors = numpy.zeros((max(teams.team_size), len(teams)), dtype='complex_')
        norm_eigenvectors = numpy.zeros((max(teams.team_size), len(teams)), dtype='complex_')

        for i in range(len(teams)):
            lower = teams.player_id_min[i] - 1
            upper = teams.player_id_max[i]
            diff = upper - lower
            team_pass_matrix[i, 0:diff, 0:diff] = pass_matrix[lower:upper, lower:upper]

            for j in range(max(teams.team_size)):
                team_pass_matrix[i, j, j] = 0
                if j != 0:
                    team_pass_matrix[i, j - 1, j] += 1
                if team_pass_matrix[i, :, j].sum(axis=0) == 0:
                    team_pass_matrix[i, j, j] = 1
                    team_pass_matrix_nonnorm[i, j, j] = 1
                else:
                    team_pass_matrix[i, :, j] = team_pass_matrix[i, :, j] / team_pass_matrix[i, :, j].sum(axis=0)

            # calculating the eigenvectors
            eigenvalues[:, i] = eig(team_pass_matrix[i])[0].real
            eigenvectors[:, i] = eig(team_pass_matrix[i])[1][:, 0].real

        for i in range(len(eigenvectors[0])):
            norm_eigenvectors[:, i] = abs(100 * eigenvectors[:, i] / eigenvectors[:, i].sum(axis=0))
    except Exception as code_error:
        print_error('Error Calculating Centrality', code_error)

    try:
        norm_eigenvectors_list = numpy.zeros(len(players))
        norm_eigenvectors_list_z = numpy.zeros(len(players))
        for i in range(len(teams)):
            norm_eigenvectors_list[teams.player_id_min[i] - 1:teams.player_id_max[i]] = norm_eigenvectors[
                                                                                        0:teams.team_size[i], i].real
            norm_eigenvectors_list_z[teams.player_id_min[i] - 1:teams.player_id_max[i]] = stats.zscore(
                norm_eigenvectors[0:teams.team_size[i], i].real)
    except Exception as code_error:
        print_error('Error Creating Normalized Eigenvectors', code_error)


def dynamic_duos():
    banner('Finding Dynamic Duos')
    global duos, updated_pass_matrix, completed_pass_percentage, total_passes

    updated_pass_matrix = pass_matrix
    completed_pass_percentage = numpy.zeros(len(updated_pass_matrix[0]))
    total_passes = numpy.zeros(len(updated_pass_matrix[0]))
    for i in range(len(updated_pass_matrix[0])):
        updated_pass_matrix[i][i] = 0
        if failed_passes[i] + completed_passes[i] != 0:
            completed_pass_percentage[i] = completed_passes[i] / (failed_passes[i] + completed_passes[i]) * 100
        else:
            completed_pass_percentage[i] = 0
        total_passes[i] = failed_passes[i] + completed_passes[i]

    n = numpy.count_nonzero(updated_pass_matrix)

    rows = updated_pass_matrix.shape[0]
    max_indices = [(e // rows, e - (e // rows * rows)) for e in updated_pass_matrix.flatten().argsort()[::-1][:n]]
    max_indices.sort()
    # for element in max_indices:
    #     print(element)

    duos_columns = ['player1', 'player1_index', 'player2', 'player2_index', 'num_passes']
    duos = pd.DataFrame(columns=duos_columns)
    duos_list = [''] * len(duos_columns)

    with alive_bar(len(max_indices), force_tty=True) as bar:
        for i in range(len(max_indices)):
            duos_list[0] = players['player_name'][max_indices[i][0]]
            duos_list[1] = max_indices[i][0]
            duos_list[2] = players['player_name'][max_indices[i][1]]
            duos_list[3] = max_indices[i][1]
            duos_list[4] = int(updated_pass_matrix[max_indices[i][0]][max_indices[i][1]])
            duos.loc[len(duos)] = duos_list
            bar()

    duos = duos.sort_values(by='num_passes', ascending=False)


############################################################
# Data Post-Processing
############################################################
def data_post_process():
    banner('Post-processing analysis results...')
    global soccer_output
    pd.options.mode.chained_assignment = None

    # putting all the data into the same file
    soccer_output = pd.DataFrame(players)
    soccer_output = soccer_output.drop(columns=['player_id', 'team_id'])
    soccer_output['goals'] = numpy.nan_to_num(recorded_goals, nan=0)
    soccer_output['num_pass'] = total_passes
    soccer_output['pass_comp'] = completed_pass_percentage
    soccer_output['z_pass_comp'] = stats.zscore(numpy.nan_to_num(completed_pass_percentage, nan=0))
    game_data['tot_loc_score'] = game_data['off_loc_score'] + game_data['def_loc_score']

    # adding offense score
    soccer_output['offense'] = numpy.nan_to_num(offense_score_results, nan=0)
    soccer_output['defense'] = numpy.nan_to_num(defense_score_results, nan=0)
    soccer_output['centrality'] = numpy.nan_to_num(norm_eigenvectors_list, nan=0)

    if scale_by_appearances:
        j = 0
        for i in range(len(soccer_output)):
            if i != len(soccer_output) - 1:
                if soccer_output['team_name'][i] != soccer_output['team_name'][i + 1]:
                    j += 1
            soccer_output['offense'][i] /= (
                    ((soccer_output['appearances'][i] + 36) * (19 / 37)) / 38)  # * offense_possesions[j]
            soccer_output['defense'][i] /= (
                    ((soccer_output['appearances'][i] + 36) * (19 / 37)) / 38)  # * defense_possesions[j]

    soccer_output['z_offense'] = stats.zscore(numpy.nan_to_num(offense_score_results, nan=0))
    soccer_output['z_defense'] = stats.zscore(numpy.nan_to_num(defense_score_results, nan=0))
    soccer_output['z_centrality'] = numpy.nan_to_num(norm_eigenvectors_list_z, nan=0)

    # adding total score
    soccer_output['total'] = 0
    for i in range(len(soccer_output)):
        soccer_output['team_appearances'][i] = soccer_output['team_appearances'][i].replace(
            soccer_output['team_name'][i], '')
        soccer_output['team_appearances'][i] = soccer_output['team_appearances'][i].replace(', ', '')
        if soccer_output['team_appearances'][i] == '':
            soccer_output['team_appearances'][i] = 'None'

        if skill_based_weights:
            scores = [soccer_output['z_offense'][i], soccer_output['z_defense'][i], soccer_output['z_centrality'][i]]
            max_index = scores.index(max(scores))
            total_weights = [0.5, 0.5, 0.5]
            total_weights[max_index] += 0.5
            soccer_output['total'][i] = total_weights[0] * soccer_output['z_offense'][i] + total_weights[1] * \
                                        soccer_output['z_defense'][i] + total_weights[2] * \
                                        soccer_output['z_centrality'][i]
        elif position_based_weights:
            print('still under development')
            if soccer_output['position'][i] == 'Forward':
                total_weights = [1, 0.5, 0.5]
            elif soccer_output['position'][i] == 'Midfielder':
                total_weights = [0.5, 1, 0.5]
            elif soccer_output['position'][i] == 'Defender' or soccer_output['position'][i] == 'Goalkeeper':
                total_weights = [0.5, 0.5, 1]
            else:
                total_weights = [1, 0.5, 0.5]

            soccer_output['total'][i] = total_weights[0] * soccer_output['z_offense'][i] + total_weights[1] * \
                                        soccer_output['z_defense'][i] + total_weights[2] * \
                                        soccer_output['z_centrality'][i]
        else:
            soccer_output['total'][i] = soccer_output['z_offense'][i] + 0.5 * soccer_output['z_defense'][i] + 0.5 * \
                                        soccer_output['z_centrality'][i]

    soccer_output['z_total'] = stats.zscore(soccer_output['total'])
    soccer_output = soccer_output.sort_values(by='z_total', ascending=False)

    soccer_output['rank'] = numpy.arange(soccer_output.shape[0]) + 1


############################################################
# Data Export
############################################################
def data_export():
    banner('Exporting analysis results...')

    current_path = os.path.dirname(os.path.realpath(__file__))
    current_path = current_path.rsplit('/', 1)[0]
    back_path = current_path.rsplit('/', 1)[0]
    results_path = current_path + '/Analysis Results/'
    game_data_path = back_path + '/Processed Game Data/'
    goal_tracker_path = current_path + '/Goal Data/'
    pass_network_path = current_path + '/Pass Network/'

    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if not os.path.exists(game_data_path):
        os.makedirs(game_data_path)
    if not os.path.exists(goal_tracker_path):
        os.makedirs(goal_tracker_path)
    if not os.path.exists(pass_network_path):
        os.makedirs(pass_network_path)

    if add_goal_multiplier and add_save_multiplier:
        mult_ext = '_GSM'
    elif add_goal_multiplier and not add_save_multiplier:
        mult_ext = '_GM'
    elif not add_goal_multiplier and add_save_multiplier:
        mult_ext = '_SM'

    extension = '' + '_AP' * scale_by_appearances + mult_ext + '_SS' * season_specific_metrics + '_SBW' * skill_based_weights + '_PBW' * position_based_weights + '.csv'

    # creating the directed graph pictures
    if export_directed_graphs:
        def nudge(pos, x_shift, y_shift):
            return {n: (x + x_shift, y + y_shift) for n, (x, y) in pos.items()}

        with alive_bar(len(teams), force_tty=True) as bar:
            for i in range(len(teams)):
                lower = teams.player_id_min[i] - 1
                upper = teams.player_id_max[i]
                updated_team_pass_matrix = updated_pass_matrix[lower:upper, lower:upper]
                compiled_network = nx.from_numpy_matrix(updated_team_pass_matrix, create_using=nx.DiGraph)

                # layout = nx.spring_layout(compiled_network, k=0.3)
                layout = nx.circular_layout(compiled_network)
                pos_nodes = nudge(layout, 0, 0.1)
                labels = dict(enumerate((players[players.team_appearances == teams.team_name[i]].player_name)))

                widths = nx.get_edge_attributes(compiled_network, 'weight')
                nx.draw_networkx_nodes(compiled_network, layout,
                                       nodelist=compiled_network.nodes(),
                                       node_size=20,
                                       node_color='black',
                                       alpha=1)
                nx.draw_networkx_edges(compiled_network, layout,
                                       edgelist=widths.keys(),
                                       width=list(widths.values()),
                                       arrows=False,
                                       edge_color='darkred',
                                       alpha=0.5)
                nx.draw_networkx_edges(compiled_network, layout,
                                       edge_color='black',
                                       alpha=1)
                nx.draw_networkx_labels(compiled_network, pos_nodes, labels, font_size=11, font_color="black")

                plt.tight_layout()
                plt.axis("off")
                plt.savefig(pass_network_path + season_data + '_' + teams.team_name[i] + '_pass_network.png',
                            format='PNG',
                            dpi=1000, bbox_inches='tight')
                plt.clf()
                bar()

    try:
        results_file = results_path + season_data + '_output' + extension
        soccer_output.to_csv(results_file, index=False, encoding='utf-8-sig')

        if export_game_data:
            game_data_file = game_data_path + season_data + '_game_data' + extension
            game_data.to_csv(game_data_file, index=False, encoding='utf-8-sig')

        if export_goal_data:
            goal_tracker_file = goal_tracker_path + season_data + '_goal_tracker' + extension
            goal_tracker.sort_values(by=['event_desc', 'time'], ascending=True).to_csv(goal_tracker_file, index=False,
                                                                                       encoding='utf-8-sig')

        if export_pass_network:
            pass_network_file = pass_network_path + season_data + '_pass_network.csv'
            duos_file = pass_network_path + season_data + '_dynamic_duos.csv'
            pd.DataFrame(updated_pass_matrix).to_csv(pass_network_file)
            duos.to_csv(duos_file, index=False, encoding='utf-8-sig')

        print('Exporting analysis results to \"' + results_file + '\"')
    except Exception as code_error:
        print_error('Invalid Filepath', code_error)

    end = time.time()
    total = end - start
    minutes = int(total / 60)
    seconds = str(int(total - minutes * 60))
    if len(seconds) == 1:
        seconds = '0' + seconds
    print('\n\n## Total Time: ' + str(minutes) + ':' + seconds + ' ##\n\n')


############################################################
# Finally, run the whole thing!!
############################################################
def title():
    def clear():
        if name == 'nt':
            _ = system('cls')
        else:
            _ = system('clear')

    clear()
    banner('Executing \'soccer.py\' Soccer Analytics')
    print(credit)


title()
login()
