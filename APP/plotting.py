import pandas as pd
import numpy as np

import json
import os


import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

import plotly.express as px
from plotly.io import show
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from timple.timedelta import strftimedelta
import datetime
import fastf1
import fastf1.plotting
from fastf1.core import Laps
from fastf1.ergast import Ergast

import statistics as st
from time import sleep

import warnings
warnings.filterwarnings("ignore")


###################################################################################
###################################################################################

## TEMPORADA

###################################################################################

#HEAD TO HEAD COMPARISONS

#Obtain the results for a given year

def get_season_results(year):
    ergast = Ergast()
    races = ergast.get_race_schedule(year)  
    results = []
    sprint_results = []

    # For each race in the season
    for rnd, race in races['raceName'].items():

        # Get race results
        temp = ergast.get_race_results(season=year, round=rnd + 1)
        result = pd.DataFrame(temp.content[0])
        
        # If there is a sprint, get the results as well
        sprint = ergast.get_sprint_results(season=year, round=rnd + 1)
        if sprint.content and sprint.description['round'][0] == rnd + 1:
            sprint_result = pd.DataFrame(sprint.content[0])
            sprint_result['raceName'] = race
            sprint_results.append(sprint_result)

        result['raceName'] = race
        results.append(result)

    # Concatenate all results
    results = pd.concat(results, ignore_index=True)
    sprint_results = pd.concat(sprint_results, ignore_index=True) if sprint_results else pd.DataFrame()
    os.makedirs(rf'APP/data/bueno/{year}/HtH', exist_ok=True)
    results.to_csv(rf'APP/data/bueno/{year}/HtH/{year}_results.csv', index=False)
    sprint_results.to_csv(rf'APP/data/bueno/{year}/HtH/{year}_sprint_results.csv', index=False)

#Obtain the qualifying results for a given year
def get_season_q_results(year):
    ergast = Ergast()
    races = ergast.get_race_schedule(year)  
    q_results = []

    # For each race in the season
    for rnd, race in races['raceName'].items():

        # Get results
        temp = ergast.get_qualifying_results(season=year, round=rnd + 1)
        q_result = pd.DataFrame(temp.content[0])
        q_result['raceName'] = race
        q_results.append(q_result)

    # Concatenate all results
    q_results = pd.concat(q_results, ignore_index=True)
    q_results = q_results.copy()

    q_results['Q1'] = pd.to_timedelta(q_results['Q1'])
    q_results['Q2'] = pd.to_timedelta(q_results['Q2'])
    q_results['Q3'] = pd.to_timedelta(q_results['Q3'])

    q_results['Q1 (s)'] = q_results['Q1'].dt.total_seconds().round(3)
    q_results['Q2 (s)'] = q_results['Q2'].dt.total_seconds().round(3)
    q_results['Q3 (s)'] = q_results['Q3'].dt.total_seconds().round(3)
    
    os.makedirs(rf'APP/data/bueno/{year}/HtH', exist_ok=True)
    q_results.to_csv(rf'APP/data/bueno/{year}/HtH/{year}_q_results.csv', index=False)


#Filter the results to get only the drivers indicated

def results_pair(results, drivers_to_comp):
    results_to_comp = results[results['driverCode'].isin(drivers_to_comp)]

    for index, row in results_to_comp.iterrows():
        if row['grid'] == 0:
            results_to_comp.at[index, 'grid'] = 20

    return results_to_comp

def results_pair_sprint(sprint_results, drivers_to_comp):
    sprint_results_to_comp = sprint_results[sprint_results['driverCode'].isin(drivers_to_comp)]

    for index, row in sprint_results_to_comp.iterrows():
        if row['grid'] == 0:
            sprint_results_to_comp.at[index, 'grid'] = 20

    return sprint_results_to_comp


## Functions for comparisons Head to Head
# Compare total points for each driver
def compare_points(drivers_to_comp, results_to_comp, sprint_results):
    points_comp = {}
    for driver in drivers_to_comp:
        # Sum points for each driver including sprint results
        race_points = results_to_comp[results_to_comp['driverCode'] == driver]['points'].sum()
        sprint_points = sprint_results[sprint_results['driverCode'] == driver]['points'].sum()
        points_comp[driver] = race_points + sprint_points
    return points_comp



# Compare final positions for each race
def compare_final_positions(drivers_to_comp, results_to_comp):
    final_positions = pd.DataFrame()
    for race in results_to_comp['raceName'].unique():
        race_positions = {}
        for driver in drivers_to_comp:
            # Get final position for each driver in each race
            final_pos = results_to_comp['position'][(results_to_comp['raceName'] == race) & (results_to_comp['driverCode'] == driver)].values[0]
            race_positions[driver] = final_pos
        final_positions[race] = pd.Series(race_positions)
    final_positions.index = drivers_to_comp
    final_positions = final_positions.T
    return final_positions

# Compare final position changes between drivers
def compare_final_position_comp(drivers_to_comp, final_positions):
    final_position_comp = {driver: 0 for driver in drivers_to_comp}
    for i in final_positions.diff(axis=1)[drivers_to_comp[1]]:
        # Compare position changes between drivers
        if i > 0:
            final_position_comp[drivers_to_comp[0]] += 1
        elif i < 0:
            final_position_comp[drivers_to_comp[1]] += 1
    return final_position_comp



# Compare positions gained during races
def compare_positions_gained(drivers_to_comp, results_to_comp):
    positions_gained = pd.DataFrame()
    for race in results_to_comp['raceName'].unique():
        race_positions = {}
        for driver in drivers_to_comp:
            # Calculate positions gained by each driver
            grid_pos = results_to_comp['grid'][(results_to_comp['raceName'] == race) & (results_to_comp['driverCode'] == driver)].values[0]
            final_pos = results_to_comp['position'][(results_to_comp['raceName'] == race) & (results_to_comp['driverCode'] == driver)].values[0]
            race_positions[driver] = grid_pos - final_pos
        positions_gained[race] = pd.Series(race_positions)
    positions_gained.index = drivers_to_comp
    positions_gained = positions_gained.T
    return positions_gained

# Compare positions gained over the year
def compare_year_positions_gained(drivers_to_comp, positions_gained):
    positions_gained_comp = {}

    for driver in drivers_to_comp:
        # Calculate total positions gained for each driver
        positions_gained_comp[driver] = positions_gained[driver].sum()
    return positions_gained_comp



# Get qualifying positions for each race
def get_quali_positions(drivers_to_comp, quali_results):
    quali_positions = pd.DataFrame()
    for race in quali_results['raceName'].unique():
        race_positions = {}
        for driver in drivers_to_comp:
            # Get qualifying position for each driver in each race
            final_pos = quali_results['position'][(quali_results['raceName'] == race) & (quali_results['driverCode'] == driver)].values[0]
            race_positions[driver] = final_pos
        quali_positions[race] = pd.Series(race_positions)
    quali_positions.index = drivers_to_comp
    quali_positions = quali_positions.T
    return quali_positions

# Compare qualifyig position changes between drivers
def compare_quali_position(drivers_to_comp, quali_positions):
    quali_position_comp = {driver: 0 for driver in drivers_to_comp}
    for i in quali_positions.diff(axis=1)[drivers_to_comp[1]]:
        # Compare qualifying position changes between drivers
        if i > 0:
            quali_position_comp[drivers_to_comp[0]] += 1
        elif i < 0:
            quali_position_comp[drivers_to_comp[1]] += 1
    return quali_position_comp

# Get qualifying times for each race
def get_quali_times(drivers_to_comp, quali_results):
    quali_times = pd.DataFrame(index=quali_results['raceName'].unique(), columns=drivers_to_comp)
    quali_results_to_compare = quali_results[quali_results['driverCode'].isin(drivers_to_comp)]

    for _, row in quali_results_to_compare.iterrows():
        if pd.isna(row['Q3 (s)']):
            if pd.isna(row['Q2 (s)']):
                quali_times.loc[row['raceName'], row['driverCode']] = row['Q1 (s)']
            else:
                quali_times.loc[row['raceName'], row['driverCode']] = row['Q2 (s)']
        else:
            quali_times.loc[row['raceName'], row['driverCode']] = row['Q3 (s)']

    return quali_times

# Get qualifying times for each race
def compare_quali_times(drivers_to_comp, quali_times):
    quali_diff = {}

    quali_diff[drivers_to_comp[0]] = -(quali_times.diff(axis=1).loc[:, drivers_to_comp[1]].mean().round(3))
    quali_diff[drivers_to_comp[1]] = quali_times.diff(axis=1).loc[:, drivers_to_comp[1]].mean().round(3)

    return quali_diff


# Compare number of DNFs for each driver
def compare_dnfs(drivers_to_comp, results_to_comp):
    n_DNF = {driver: 0 for driver in drivers_to_comp}
    for driver in drivers_to_comp:
        for _, row in results_to_comp.iterrows():
            # Count DNFs for each driver
            if row['driverCode'] == driver and row['status'] != 'Finished':
                if row['status'][0] != '+':
                    n_DNF[driver] += 1
    return n_DNF

# Compare number of wins and podiums for each driver
def compare_wins_and_podiums(drivers_to_comp, results_to_comp):
    n_wins = {driver: 0 for driver in drivers_to_comp}
    n_podiums = {driver: 0 for driver in drivers_to_comp}
    for _, row in results_to_comp.iterrows():
        # Count wins and podiums for each driver
        if row['position'] == 1:
            n_wins[row['driverCode']] += 1
        if row['position'] <= 3:
            n_podiums[row['driverCode']] += 1
    return n_wins, n_podiums

# Compare number of poles for each driver
def compare_poles(drivers_to_comp, quali_positions):
    n_poles = {driver: 0 for driver in drivers_to_comp}
    for driver in quali_positions.columns:
        n_poles[driver] = quali_positions[driver].value_counts().get(1, 0)
    return n_poles




#Recap all HtH functions
def compare_results_pair(year, drivers_to_comp):
    results = pd.read_csv(rf'APP/data/bueno/{year}/HtH/{year}_results.csv')
    sprint_results = pd.read_csv(rf'APP/data/bueno/{year}/HtH/{year}_sprint_results.csv')
    q_results = pd.read_csv(rf'APP/data/bueno/{year}/HtH/{year}_q_results.csv')

    results_to_comp = results_pair(results, drivers_to_comp)
    sprint_results_to_comp = results_pair_sprint(sprint_results, drivers_to_comp)

    points_comp = compare_points(drivers_to_comp, results_to_comp, sprint_results_to_comp)

    final_positions = compare_final_positions(drivers_to_comp, results_to_comp)
    final_position_comp = compare_final_position_comp(drivers_to_comp, final_positions)
    
    positions_gained = compare_positions_gained(drivers_to_comp, results_to_comp)
    year_positions_gained = compare_year_positions_gained(drivers_to_comp, positions_gained)
    
    quali_positions = get_quali_positions(drivers_to_comp, q_results)
    quali_position_comp = compare_quali_position(drivers_to_comp, quali_positions)
    quali_times = get_quali_times(drivers_to_comp, q_results)
    quali_diff = compare_quali_times(drivers_to_comp, quali_times)

    dnfs = compare_dnfs(drivers_to_comp, results_to_comp)
    wins, podiums = compare_wins_and_podiums(drivers_to_comp, results_to_comp)
    poles = compare_poles(drivers_to_comp, quali_positions)
        
    return {
        'points_comp': points_comp,
        'final_positions': final_positions,
        'final_position_comp': final_position_comp,
        'positions_gained': positions_gained,
        'year_positions_gained': year_positions_gained,
        'quali_positions': quali_positions,
        'quali_position_comp': quali_position_comp,
        'quali_times': quali_times,
        'quali_diff': quali_diff,
        'dnfs': dnfs,
        'wins': wins,
        'podiums': podiums,
        'poles': poles
        }



# Functions to plot the comparisons Head to Head

def plot_comparisons(year, comparisons):
    
    with open(rf'APP/data/bueno/{year}/Ritmos/Drivers/driver_info_{year}.json', 'r') as f:
        driver_info = json.load(f)
    driver_palette = driver_info['driver_palette']

    colors = [driver_palette[driver] for driver in comparisons['points_comp'].keys()]

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 25), facecolor='#f4f4f4')
    fig.suptitle('Comparisons', fontsize=30)

    bar_width = 0.4

    # Points Comparison
    axes[0, 0].bar(comparisons['points_comp'].keys(), comparisons['points_comp'].values(), color = colors, width=bar_width)
    axes[0, 0].set_title('Points Comparison', fontsize=14)
    axes[0, 0].set_ylabel('Points')
    for i, v in enumerate(comparisons['points_comp'].values()):
        offset = v * 0.01 if v > 0 else v*0.09
        axes[0, 0].text(i, v + offset, str(v), color='black', ha='center', fontsize=12)

    max_y = max(comparisons['points_comp'].values())* 1.15
    if min(comparisons['points_comp'].values()) < 0:
        min_y = min(comparisons['points_comp'].values())* 1.20
    else:
        min_y = 0

    axes[0,0].set_ylim([min_y, max_y])

    # Final Position Head-to-head
    axes[0, 1].bar(comparisons['final_position_comp'].keys(), comparisons['final_position_comp'].values(), color = colors, width=bar_width)
    axes[0, 1].set_title('Final Position Head-to-head', fontsize=14)
    axes[0, 1].set_ylabel('Wins')
    for i, v in enumerate(comparisons['final_position_comp'].values()):
        offset = v * 0.01 if v > 0 else v*0.09
        axes[0, 1].text(i, v + offset, str(v), color='black', ha='center', fontsize=12)

    max_y = max(comparisons['final_position_comp'].values())* 1.15
    if min(comparisons['final_position_comp'].values()) < 0:
        min_y = min(comparisons['final_position_comp'].values())* 1.20
    else:
        min_y = 0

    axes[0, 1].set_ylim([min_y, max_y])

    # Year Positions Gained(+)/Lost(-)
    axes[0, 2].bar(comparisons['year_positions_gained'].keys(), comparisons['year_positions_gained'].values(), color = colors, width=bar_width)
    axes[0, 2].set_title('Year Positions Gained(+)/Lost(-)', fontsize=14)
    axes[0, 2].set_ylabel('Positions Gained')
    for i, v in enumerate(comparisons['year_positions_gained'].values()):
        offset = v * 0.01 if v > 0 else v*0.28
        axes[0, 2].text(i, v + offset, str(v), color='black', ha='center', fontsize=12)

    max_y = max(comparisons['year_positions_gained'].values())* 1.15
    if min(comparisons['year_positions_gained'].values()) < 0:
        min_y = min(comparisons['year_positions_gained'].values())* 1.30
    else:
        min_y = 0

    axes[0, 2].set_ylim([min_y, max_y])

    # Qualifying Position Head-to-head
    axes[1, 0].bar(comparisons['quali_position_comp'].keys(), comparisons['quali_position_comp'].values(), color = colors, width=bar_width)
    axes[1, 0].set_title('Qualifying Position Head-to-head', fontsize=14)
    axes[1, 0].set_ylabel('Wins')
    for i, v in enumerate(comparisons['quali_position_comp'].values()):
        offset = v * 0.01 if v > 0 else v*0.09
        axes[1, 0].text(i, v + offset, str(v), color='black', ha='center', fontsize=12)

    max_y = max(comparisons['quali_position_comp'].values())* 1.15
    if min(comparisons['quali_position_comp'].values()) < 0:
        min_y = min(comparisons['quali_position_comp'].values())* 1.20
    else:
        min_y = 0

    axes[1, 0].set_ylim([min_y, max_y])

    # Qualifying Times Difference
    axes[1, 1].bar(comparisons['quali_diff'].keys(), comparisons['quali_diff'].values(), color = colors, width=bar_width)
    axes[1, 1].set_title('Qualifying Times Difference', fontsize=14)
    axes[1, 1].set_ylabel('Time Difference (s)')
    for i, v in enumerate(comparisons['quali_diff'].values()):
        offset = v * 0.01 if v > 0 else v*0.15
        axes[1, 1].text(i, v + offset, str(v), color='black', ha='center', fontsize=12)

    max_y = max(comparisons['quali_diff'].values())* 1.15
    if min(comparisons['quali_diff'].values()) < 0:
        min_y = min(comparisons['quali_diff'].values())* 1.20
    else:
        min_y = 0

    axes[1, 1].set_ylim([min_y, max_y])

    # DNFs Comparison
    axes[1, 2].bar(comparisons['dnfs'].keys(), comparisons['dnfs'].values(), color = colors, width=bar_width)
    axes[1, 2].set_title('DNFs Comparison', fontsize=14)
    axes[1, 2].set_ylabel('DNFs')
    for i, v in enumerate(comparisons['dnfs'].values()):
        offset = v * 0.01 if v > 0 else v*0.09
        axes[1, 2].text(i, v + offset, str(v), color='black', ha='center', fontsize=12)

    max_y = max(comparisons['dnfs'].values())* 1.15
    if min(comparisons['dnfs'].values()) < 0:
        min_y = min(comparisons['dnfs'].values())* 1.20
    else:
        min_y = 0

    axes[1, 2].set_ylim([min_y, max_y])

    # Wins Comparison
    axes[2, 0].bar(comparisons['wins'].keys(), comparisons['wins'].values(), color = colors, width=bar_width)
    axes[2, 0].set_title('Wins Comparison', fontsize=14)
    axes[2, 0].set_ylabel('Wins')
    for i, v in enumerate(comparisons['wins'].values()):
        offset = v * 0.01 if v > 0 else v*0.09
        axes[2, 0].text(i, v + offset, str(v), color='black', ha='center', fontsize=12)

    max_y = max(comparisons['wins'].values())* 1.15
    if min(comparisons['wins'].values()) < 0:
        min_y = min(comparisons['wins'].values())* 1.20
    else:
        min_y = 0

    axes[2, 0].set_ylim([min_y, max_y])

    # Podiums Comparison
    axes[2, 1].bar(comparisons['podiums'].keys(), comparisons['podiums'].values(), color = colors, width=bar_width)
    axes[2, 1].set_title('Podiums Comparison', fontsize=14)
    axes[2, 1].set_ylabel('Podiums')
    for i, v in enumerate(comparisons['podiums'].values()):
        offset = v * 0.01 if v > 0 else v*0.09
        axes[2, 1].text(i, v + offset, str(v), color='black', ha='center', fontsize=12)

    max_y = max(comparisons['podiums'].values())* 1.15
    if min(comparisons['podiums'].values()) < 0:
        min_y = min(comparisons['podiums'].values())* 1.20
    else:
        min_y = 0

    axes[2, 1].set_ylim([min_y, max_y])

    # Poles Comparison
    axes[2, 2].bar(comparisons['poles'].keys(), comparisons['poles'].values(), color=colors, width=bar_width)
    axes[2, 2].set_title('Poles Comparison', fontsize=14)
    axes[2, 2].set_ylabel('Poles')
    for i, v in enumerate(comparisons['poles'].values()):
        offset = v * 0.01 if v > 0 else v * 0.15
        axes[2, 2].text(i, v + offset, str(v), color='black', ha='center', fontsize=12)

    max_y = max(comparisons['poles'].values())* 1.15
    if min(comparisons['poles'].values()) < 0:
        min_y = min(comparisons['poles'].values())* 1.20
    else:
        min_y = 0

    axes[2, 2].set_ylim([min_y, max_y])


    fig.patch.set_alpha(0)  
    plt.gca().patch.set_alpha(0)
    fig.set_size_inches(15, 10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show()
    return fig


###################################################################################

# PACE FUNCTIONS

#Plot the pace of the drivers in each race

def plot_year_pace_driver(year):
    df_ritmos = pd.read_csv(rf'APP/data/bueno/{year}/Ritmos/Drivers/df_ritmos_{year}.csv', index_col=0)
    
    with open(rf'APP/data/bueno/{year}/Ritmos/Drivers/driver_info_{year}.json', 'r') as f:
        driver_info = json.load(f)
    driver_palette = driver_info['driver_palette']
    driver_line = driver_info['driver_line']

    driver_styles = {driver: {'color': driver_palette[driver], 'line': driver_line[driver]} for driver in driver_palette}
    
    # Ensure all columns are of the same type
    df_ritmos = df_ritmos.apply(pd.to_numeric, errors='coerce')
    fig = px.line(df_ritmos, x=df_ritmos.index, y=df_ritmos.columns, line_shape='linear',
                  labels={'value': 'Time Difference (seconds)', 'index': 'Circuits'}, 
                  title=f'Time Difference Progression Compared to Average Season {year}',
                  markers=True)

    fig.update_layout(xaxis_title='Circuits', yaxis_title='Time Difference (seconds)', 
                      legend_title='Driver', xaxis=dict(tickangle=-60), template='plotly_white')

    fig.update_yaxes(autorange='reversed')
    for driver, style in driver_styles.items():
        fig.update_traces(selector=dict(name=driver), line=dict(color=style['color'], dash=style['line'])) 
        fig.update_layout(width=1200, height=800)
    
    return fig


#Plot the pace of the teams in each race

def plot_year_pace_team(year):
    df_ritmos = pd.read_csv(rf'/data/bueno/{year}/Ritmos/Teams/df_ritmos_{year}.csv', index_col=0)
        
    with open(rf'APP/data/bueno/{year}/Ritmos/Teams/team_info_{year}.json', 'r') as f:
        team_info = json.load(f)
    team_palette = team_info['team_palette']

    team_styles = {driver: {'color': team_palette[driver]} for driver in team_palette}

    df_ritmos = df_ritmos.apply(pd.to_numeric, errors='coerce')
    fig = px.line(df_ritmos, x=df_ritmos.index, y=df_ritmos.columns, line_shape='linear',
                  labels={'value': 'Time Difference (seconds)', 'index': 'Circuits'}, 
                  title=f'Time Difference Progression Compared to Average Season {year}',
                  markers=True)

    fig.update_layout(xaxis_title='Circuits', yaxis_title='Time Difference (seconds)', 
                      legend_title='Team', xaxis=dict(tickangle=-60), template='plotly_white')

    fig.update_yaxes(autorange='reversed')
    for driver, style in team_styles.items():
        fig.update_traces(selector=dict(name=driver), line=dict(color=style['color'])) 
        fig.update_layout(width=1200, height=800)

    # Show the plot
    return fig




###################################################################################
###################################################################################

#GP FUNCTIONS

#Plot qualifying delta times for a given event
def plot_qualifying_times(year, event):
    delta_times = pd.read_csv(rf'APP/data/bueno/{year}/qualifying_times/{event}_qualifying_times.csv', index_col=0)

    for _, row in delta_times.iterrows():
        if row.isnull().any():
            delta_times.drop(index=row.name, inplace=True)

    with open(rf'APP/data/bueno/{year}/qualifying_times/{event}_complementary_info.json', 'r') as f:
        complementary_info = json.load(f)

    pole_lap = complementary_info['pole_lap']
    team_colors = complementary_info['driver_colors']
    
    fig, ax = plt.subplots(figsize=(16, 6.9))

    ax.barh(delta_times.index, delta_times['LapTimeDelta'],
            color=[team_colors[driver] for driver in delta_times['Driver']], edgecolor='black', linewidth=0.5)
    ax.set_yticks(delta_times.index)
    ax.set_yticklabels(delta_times['Driver'], color='black')
    # ax.tick_params(axis='x', colors='black')
    # show fastest at the top
    ax.invert_yaxis()

    # draw vertical lines behind the bars
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, which='major', linestyle='--', color='black', zorder=-1000)

    lap_time_string = strftimedelta(pd.to_timedelta(pole_lap['LapTime'], unit='s'), '%m:%s.%ms')

    plt.suptitle(f"{event} {year} Qualifying/n"
                 f"Fastest Lap: {lap_time_string} ({pole_lap['Driver']})",
                 fontsize=22, color='black')
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.yaxis.grid(False)
    
    
    ax.ticklabel_format(useOffset=False, style='plain', axis='x')
    
    # Format yticks as MM:SS.ms
    def format_func(value, tick_number):
        mins, secs = divmod(value, 60)
        return f'{secs:05.3f}'

    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    
    ax.set_xlabel('Time difference (s)', color='black')
    ax.patch.set_alpha(0.0)
    plt.gca().patch.set_alpha(0)
    plt.tight_layout()
    return fig

#Plot position changes during the race
def plot_position_changes(year, event):
    df_position = pd.read_csv(rf'APP/data/bueno/{year}/pos_changes_race/df_position_{year}_{event}.csv', index_col=0)
    with open(rf'APP/data/bueno/{year}/pos_changes_race/driver_style_{year}_{event}.json', 'r') as f:
        driver_style = json.load(f)

    total_drivers = len(df_position)
    total_laps = len(df_position.columns)

    fig, ax = plt.subplots(figsize=(18.0, 6.9))
    fig.patch.set_facecolor('#f3f3f3')
    ax.set_facecolor('#f3f3f3')
    
    for driver in df_position.index:
        ax.plot(df_position.columns, df_position.loc[driver], label=driver, linewidth=3, **driver_style[driver])
            
    # Set plot limits and labels
    ax.set_ylim([20.5, 0.5])
    ax.set_yticks(range(1, total_drivers+1))
    ax.set_xlim([0, total_laps + 1])
    ax.set_xlabel('Lap', color='black')
    ax.set_ylabel('Position', color='black')
    ax.set_xticks([1] + list(range(5, total_laps + 1, 5)))
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')

    # Create a secondary y-axis
    ax2 = ax.twinx()
    ax2.set_ylim([20.5, 0.5])
    ax2.set_yticks(range(1, total_drivers+1))
    ax2.tick_params(axis='y', colors='black', direction='in', pad=-20)

    ax.legend(bbox_to_anchor=(1.0, 1.02))
    # Order the elements in the layer using the last column of the DataFrame as indicator of the order
    order = df_position.iloc[:, -1].sort_values().index
    handles, labels = ax.get_legend_handles_labels()
    ordered_handles = [handles[labels.index(driver)] for driver in order]
    ordered_labels = [labels[labels.index(driver)] for driver in order]
    ax.legend(ordered_handles, ordered_labels, bbox_to_anchor=(1.0, 1.02))
    plt.tight_layout()
    ax.set_title(f'{year} {event} - Position Changes During the Race', fontsize=30, color='black')
    
    # Hide grid
    ax.grid(False)
    return fig


#Plot the relative distances of the drivers to the leader in each lap
def plot_relative_distances(year, event):
    distances_to_first = pd.read_csv(rf'APP/data/bueno/{year}/relative_distances/{event}_relative_distances.csv', index_col=0)

    with open(rf'APP/data/bueno/{year}/relative_distances/{event}_styles.json', 'r') as f:
        drivers_style = json.load(f)
    fig = go.Figure()
    fig.update_layout(width=1200, height=800)
    for driver in distances_to_first.columns:
        fig.add_trace(go.Scatter(
            x=distances_to_first.index,
            y=distances_to_first[driver],
            mode='lines',
            name=driver,
            line=dict(color=drivers_style[driver]['color'], dash='dash' if drivers_style[driver]['linestyle'] == 'dashed' else 'solid'), 
            visible='legendonly'
        ))

        fig.update_layout(
            title={
            'text': f'{year} {event} - Distance to First During the Race',
            'x': 0.5,
            'xanchor': 'center'
            },
            xaxis_title='Lap',
            yaxis_title='Distance to First (s)',
            legend_title='Driver',
            yaxis=dict(autorange='reversed'),
            template='plotly_white'
        )

    return fig


#Plot the pitstop strategies of all drivers for a given event
def plot_pitstop_estrategy(year, event):
    stints = pd.read_csv(rf'APP/data/bueno/{year}/pitstop_strategies/{event}_pitstop_strategies.csv')
    drivers = pd.read_csv(rf'APP/data/bueno/{year}/pitstop_strategies/{event}_positions.csv')['Driver']
    with open(rf'APP/data/bueno/Settings/compound_colors.json', 'r') as f:
        compound_colors = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 10))

    # fig.patch.set_facecolor('#f4f4f4')
    # ax.set_facecolor('#f4f4f4')

    for driver in drivers:
        driver_stints = stints.loc[stints["Driver"] == driver]

        previous_stint_end = 0
        for _, row in driver_stints.iterrows():
            plt.barh(
                y=driver,
                width=row["StintLength"],
                left=previous_stint_end,
                color=compound_colors[row["Compound"]],
                edgecolor="black",
                fill=True
            )

            previous_stint_end += row["StintLength"]

    plt.title(f"{year} {event} Strategies", color='black')
    # Create custom legend
    legend_elements = [plt.Line2D([0], [0], color=color, lw=4, label=compound) for compound, color in compound_colors.items()]
    ax.legend(handles=legend_elements, title="Compound", bbox_to_anchor=(1.05, 1), loc='upper left')
    # Change the background color of the legend
    legend = ax.get_legend()
    legend.get_frame().set_facecolor('#a7a7a7')

    # Change the color of the legend text
    for text in legend.get_texts():
        text.set_color('black')

    # Set frame thickness and color
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('black')

    plt.xlabel("Lap Number", color='black')
    plt.ylabel("Driver", color='black')
    plt.grid(False)
    ax.invert_yaxis()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')

    # plt.tight_layout()
    fig.patch.set_alpha(0)  
    plt.gca().patch.set_alpha(0)
    return fig


#Plot telemetry data for the qualifying lap
def plot_overlap_telemetries(year, event):
    # Load telemetries from json
    with open(rf'APP/data/bueno/{year}/telemetries/{event}_telemetries.json', 'r') as f:
        telemetries = json.load(f)

    # Convert telemetries back to DataFrame
    telemetries = {driver: pd.DataFrame(data) for driver, data in telemetries.items()}

    # Load styles from json
    with open(rf'APP/data/bueno/{year}/telemetries/{event}_styles.json', 'r') as f:
        drivers_style = json.load(f)

    with open(rf'APP/data/bueno/{year}/telemetries/{event}_laptimes.json', 'r') as f:
        laptimes = json.load(f)


    # drivers_style = {driver: {'color': style['color'], 'linestyle': 'dash' if style['linestyle'] == 'dashed' else style['linestyle']} for driver, style in drivers_style.items()}

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('Speed', 'Throttle', 'Brake'))

    for driver, telemetry in telemetries.items():
        style = drivers_style.get(driver, {})
        color = style.get('color', 'black')  # Color por defecto: negro
        dash_style = 'dash' if style.get('linestyle') == 'dashed' else 'solid'
        telemetry = telemetries[driver]
        
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],  
                mode='lines', name=str(str(driver) + ' (' + laptimes[driver] + ')'),  
                line=dict(color=color, dash=dash_style),
                legendgroup=driver, visible='legendonly' 
            )
        )
        fig.add_trace(go.Scatter(
            x=telemetry['Distance'], y=telemetry['Speed'],
              mode='lines', name=f"{driver} Speed", 
              line=dict(color=color, dash=dash_style), 
              legendgroup=driver, showlegend=False, visible='legendonly'), row=1, col=1)
        
        
        fig.add_trace(go.Scatter(
            x=telemetry['Distance'], y=telemetry['Throttle'], 
            mode='lines', name=f"{driver} Throttle", 
              line=dict(color=color, dash=dash_style), 
              legendgroup=driver, showlegend=False, visible='legendonly'), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=telemetry['Distance'], y=telemetry['Brake'], 
            mode='lines', name=f"{driver} Brake", 
              line=dict(color=color, dash=dash_style), 
              legendgroup=driver, showlegend=False , visible='legendonly'), row=3, col=1)

    fig.update_layout(height=1500, width=1200, title_text=f'Qualifying Lap Telemetry Comparison - {event} {year}', 
                      title_x=0.3, showlegend=True, legend_title='Driver', template='plotly_white')
    fig.update_xaxes(title_text='Distance (m)')
    fig.update_yaxes(title_text='Speed (km/h)', row=1, col=1)
    fig.update_yaxes(title_text='Throttle (%)', row=2, col=1)
    fig.update_yaxes(title_text='Brake (%)', row=3, col=1)

    return fig


#Plot the laptimes of the drivers in a given event
def plot_laptimes_race(year, event):
    with open(rf'APP/data/bueno/{year}/laptimes/{event}_laptimes.json', 'r') as f:
        all_lap_times = json.load(f)

    all_lap_times = {driver: pd.DataFrame(data) for driver, data in all_lap_times.items()}

    with open(rf'APP/data/bueno/{year}/laptimes/{event}_styles.json', 'r') as f:
        drivers_style = json.load(f)

    fig = px.line(width=1200, height=800) 

    for driver, lap_times in all_lap_times.items():
        style = drivers_style.get(driver, {})
        color = style.get('color', 'black')
        dash_style = 'dash' if style.get('linestyle') == 'dashed' else 'solid'
        
        fig.add_scatter(x=lap_times['LapNumber'], y=lap_times['LapTime'], mode='lines+markers', name=driver,
                        line=dict(color=color, dash=dash_style), visible='legendonly')

    fig.update_layout(title={'text': f"Lap Times Comparison for {event} - {year}",
                             'x': 0.5,
                             'xanchor': 'center'},
                      xaxis_title="Lap Number",
                      yaxis_title="Lap Time",
                      template='plotly_white', legend_title='Driver')
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(autorange='reversed')
    return fig

        
###################################################################################
###################################################################################

## CIRCUIT FUNCTIONS

#Plot the length of the circuits
def plot_length_circuit(circuits_info_df, event):
    # Order the circuits by length in descending order
    circuits_info_df_sorted = circuits_info_df.sort_values(by='Length (km)', ascending=True)

    # Set the colors for the bars
    colors = ['red' if circuit == event else 'skyblue' for circuit in circuits_info_df_sorted['EventName']]

    # Create a horizontal bar plot
    fig = plt.figure(figsize=(11, 8))
    bars = plt.barh(circuits_info_df_sorted['EventName'], circuits_info_df_sorted['Length (km)'], color=colors)
    plt.xlabel('Length (km)')
    plt.ylabel('Circuit')
    plt.title('Circuit Lengths')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f' {str(int((bar.get_width())*1000))}', 
                 va='center', ha='left', color='black', fontsize=10)
    plt.gca().set_axisbelow(True)
    fig.patch.set_alpha(0)  
    plt.gca().patch.set_alpha(0)
    return fig


#Plot the mean speed of the circuits
def plot_mean_speed_circuit(circuits_info_df, event):
    # Order the circuits by mean speed in descending order
    circuits_info_df_sorted = circuits_info_df.sort_values(by='Mean Speed (km/h)', ascending=True)

    # Set the colors for the bars
    colors = ['red' if circuit == event else 'skyblue' for circuit in circuits_info_df_sorted['EventName']]

    # Create a horizontal bar plot
    fig = plt.figure(figsize=(11, 8))
    bars = plt.barh(circuits_info_df_sorted['EventName'], circuits_info_df_sorted['Mean Speed (km/h)'], color=colors)
    plt.xlabel('Mean Speed (km/h)')
    plt.ylabel('Circuit')
    plt.title('Circuit Mean Speeds')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f' {bar.get_width().round(2)}', 
                 va='center', ha='left', color='black', fontsize=10)
    plt.gca().set_axisbelow(True)
    fig.patch.set_alpha(0)  
    plt.gca().patch.set_alpha(0)  
    # plt.show()
    return fig
    
# Plot the number of laps of the circuits
def plot_number_of_laps_circuit(circuits_info_df, event):
    # Order the circuits by number of laps in descending order
    circuits_info_df_sorted = circuits_info_df.sort_values(by='Laps', ascending=True)

    # Set the colors for the bars
    colors = ['red' if circuit == event else 'skyblue' for circuit in circuits_info_df_sorted['EventName']]

    # Create a horizontal bar plot
    fig = plt.figure(figsize=(11, 8))
    bars = plt.barh(circuits_info_df_sorted['EventName'], circuits_info_df_sorted['Laps'], color=colors)
    plt.xlabel('Number of Laps')
    plt.ylabel('Circuit')
    plt.title('Number of Laps')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f' {bar.get_width().round(2)}', 
                 va='center', ha='left', color='black', fontsize=10)
    plt.gca().set_axisbelow(True)

    fig.patch.set_alpha(0)  
    plt.gca().patch.set_alpha(0)  

    return fig

# Plot the number of turns of the circuits
def plot_number_of_turns_circuit(circuits_info_df, event):
    # Order the circuits by number of turns in descending order
    circuits_info_df_sorted = circuits_info_df.sort_values(by='Turns', ascending=True)

    # Set the colors for the bars
    colors = ['red' if circuit == event else 'skyblue' for circuit in circuits_info_df_sorted['EventName']]

    # Create a horizontal bar plot
    fig = plt.figure(figsize=(11, 8))
    bars = plt.barh(circuits_info_df_sorted['EventName'], circuits_info_df_sorted['Turns'], color=colors)
    plt.xlabel('Number of Turns')
    plt.ylabel('Circuit')
    plt.title('Number of Turns')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f' {bar.get_width().round(2)}', 
                 va='center', ha='left', color='black', fontsize=10)
    plt.gca().set_axisbelow(True)

    fig.patch.set_alpha(0)  
    plt.gca().patch.set_alpha(0)
    return fig

# Plot the turns/km ratio of the circuits
def plot_turns_per_km_circuit(circuits_info_df, event):
    # Order the circuits by turns/km ratio in descending order
    circuits_info_df_sorted = circuits_info_df.sort_values(by='Turns/km', ascending=True)

    # Set the colors for the bars
    colors = ['red' if circuit == event else 'skyblue' for circuit in circuits_info_df_sorted['EventName']]

    # Create a horizontal bar plot
    fig = plt.figure(figsize=(11, 8))
    bars = plt.barh(circuits_info_df_sorted['EventName'], circuits_info_df_sorted['Turns/km'], color=colors)
    plt.xlabel('Turns/km')
    plt.ylabel('Circuit')
    plt.title('Turns per km')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f' {bar.get_width().round(2)}', 
                 va='center', ha='left', color='black', fontsize=10)
    plt.gca().set_axisbelow(True)

    fig.patch.set_alpha(0)  
    plt.gca().patch.set_alpha(0)
    return fig



###################################################################################



