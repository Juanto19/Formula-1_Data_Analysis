import pandas as pd
import numpy as np

import json

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
from plotly.io import show

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
    results = pd.read_csv(rf'.\data\bueno\{year}\HtH\{year}_results.csv')
    sprint_results = pd.read_csv(rf'.\data\bueno\{year}\HtH\{year}_sprint_results.csv')
    q_results = pd.read_csv(rf'.\data\bueno\{year}\HtH\{year}_q_results.csv')

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
    
    with open(rf'.\data\bueno\{year}\Ritmos\Drivers\driver_info_{year}.json', 'r') as f:
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



    fig.set_size_inches(15, 10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show()
    return fig


###################################################################################

# PACE FUNCTIONS

#Calculate the pace of the drivers in each race

def data_year_pace_driver(year):
    schedule = fastf1.get_event_schedule(year)
    races = schedule[schedule['EventName'].notna()]

    mean_diff_list = []
    team_drivers = {}
    driver_palette = {}
    driver_number = {}
    driver_line = {}

    for i in range(1, len(races)):
        race_name = races.loc[i, 'EventName']
        
        # Load race data
        race = fastf1.get_session(year, race_name, 'R')
        race.load()
        laps = race.laps.pick_quicklaps(1.15)

        # Transform lap time to seconds
        transformed_laps = laps.copy()
        transformed_laps.loc[:, "LapTime (s)"] = laps["LapTime"].dt.total_seconds()

        # Calculate mean lap time
        drivers_time = transformed_laps[['LapNumber', 'Driver', 'LapTime (s)']]
        mean_laps = transformed_laps[["LapNumber", "LapTime (s)"]].groupby("LapNumber").mean()
        mean_laps.rename(columns={'LapTime (s)': 'MeanLapTime'}, inplace=True)

        drivers_difference = pd.merge(drivers_time, mean_laps, on='LapNumber', how="inner")
        drivers_difference['Difference'] = drivers_difference['LapTime (s)'] - drivers_difference['MeanLapTime']

        # Create a dictionary to map each driver to their team
        drivers = pd.DataFrame(data=transformed_laps[['Driver', 'Team']].groupby(['Driver'], as_index=False, sort=False).max())
        for _, row in drivers.iterrows():
            driver = row['Driver']
            team = row['Team']
            if team in team_drivers:
                if driver not in team_drivers[team]:
                    team_drivers[team].append(driver)
            else:
                team_drivers[team] = [driver]

        # Group by driver and calculate mean difference
        mean_diff_driver = drivers_difference[["Driver", "Difference"]].groupby("Driver").mean()["Difference"].sort_values()
        mean_diff_driver = pd.Series(data=mean_diff_driver, name=race_name)
        mean_diff_list.append(mean_diff_driver)

        for driver in list(mean_diff_driver.index):
            if driver not in driver_palette.keys():
                driver_palette[driver] = fastf1.plotting.get_driver_color(driver, race)
        
    for team in team_drivers.keys():
        n = 0
        for driver in team_drivers[team]:
            driver_number[driver] = n
            n += 1

    mean_diff_df = pd.concat(mean_diff_list, axis=1)

    # Assign line style
    for driver in driver_number.keys():
        if driver_number[driver] == 0:
            driver_line[driver] = 'solid'
        elif driver_number[driver] == 1:
            driver_line[driver] = 'dash'
        elif driver_number[driver] == 2:
            driver_line[driver] = 'dash'
        elif driver_number[driver] >= 3:
            driver_line[driver] = 'dashdot'
        
    # Transpose dataframe
    df_ritmos = mean_diff_df.T  
    df_ritmos.to_csv(rf'.\data\bueno\{year}\Ritmos\Drivers\df_ritmos_{year}.csv')
    data = {
        "driver_palette": driver_palette,
        "driver_line": driver_line
    }

    with open(rf'.\data\bueno\{year}\Ritmos\Drivers\driver_info_{year}.json', 'w') as f:
        json.dump(data, f)

#Plot the pace of the drivers in each race


def plot_year_pace_driver(year):
    df_ritmos = pd.read_csv(rf'.\data\bueno\{year}\Ritmos\Drivers\df_ritmos_{year}.csv', index_col=0)
    
    with open(rf'.\data\bueno\{year}\Ritmos\Drivers\driver_info_{year}.json', 'r') as f:
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


#Calculate the pace of the teams in each race

def data_year_pace_team(year):
    schedule = fastf1.get_event_schedule(year)
    races = schedule[schedule['EventName'].notna()]    

    mean_diff_list = []
    team_palette = {}

    for i in range(1, len(races)):
        race_name = races.loc[i, 'EventName']
        
        # Load race data
        race = fastf1.get_session(year, race_name, 'R')
        race.load()
        laps = race.laps.pick_quicklaps(1.15)

        # Transform lap time to seconds
        transformed_laps = laps.copy()
        transformed_laps.loc[:, "LapTime (s)"] = laps["LapTime"].dt.total_seconds()

        teams_time =  transformed_laps[['LapNumber', 'Team', 'LapTime (s)']].groupby(['Team', 'LapNumber']).mean().reset_index()
        mean_laps = transformed_laps[["LapNumber", "LapTime (s)"]].groupby("LapNumber").mean()
        mean_laps.rename(columns={'LapTime (s)': 'MeanLapTime'}, inplace=True)

        teams_difference = pd.merge(teams_time, mean_laps, on='LapNumber', how="inner")
        teams_difference['Difference'] = teams_difference['LapTime (s)'] - teams_difference['MeanLapTime']

        mean_diff_team = teams_difference[["Team", "Difference"]].groupby("Team").mean()["Difference"].sort_values()
        mean_diff_team = pd.Series(data=mean_diff_team, name=race_name)
        mean_diff_list.append(mean_diff_team)

        for team in list(mean_diff_team.index):
            if team not in team_palette.keys():
                team_palette[team] = fastf1.plotting.get_team_color(team, race)


        mean_diff_df = pd.concat(mean_diff_list, axis=1)
        # Transpose dataframe
        df_ritmos = mean_diff_df.T  
        df_ritmos.to_csv(rf'.\data\bueno\{year}\Ritmos\Teams\df_ritmos_{year}.csv')
        data = {
            "team_palette": team_palette
        }

        with open(rf'.\data\bueno\{year}\Ritmos\Teams\team_info_{year}.json', 'w') as f:
            json.dump(data, f)

#Plot the pace of the teams in each race

def plot_year_pace_team(year):
    df_ritmos = pd.read_csv(rf'.\data\bueno\{year}\Ritmos\Teams\df_ritmos_{year}.csv', index_col=0)
        
    with open(rf'.\data\bueno\{year}\Ritmos\Teams\team_info_{year}.json', 'r') as f:
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

# HEATMAP FUNCTIONS






























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
    # plt.show()
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

    # plt.show()
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

    # plt.show()
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

    # plt.show()
    return fig



###################################################################################



