# data_loading.py  (top of file)

# ------------------------------------------------------------
# Get the global ADAPTER from streamlit_app *if* that module
# has already been imported; otherwise build our own on the fly.
# ------------------------------------------------------------
import importlib
import tomli as tomllib, pathlib
from data_adapter import get_adapter

try:
    # If streamlit_app is already loaded (normal Streamlit run),
    # re-use its singleton so we share the in-memory CSV cache.
    streamlit_app = importlib.import_module("streamlit_app")
    ADAPTER = streamlit_app.ADAPTER
except ModuleNotFoundError:
    # CLI / pytest run: build a fresh adapter based on config.toml
    cfg = tomllib.loads(pathlib.Path("config.toml").read_text())
    ADAPTER = get_adapter(cfg["app"]["data_source"])



import pandas as pd
import numpy as np

import json
import os
import sys

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

from pathlib import Path
# from streamlit_app import ADAPTER   

import warnings
warnings.filterwarnings("ignore")




###################################################################################
###################################################################################

## TEMPORADA

###################################################################################

#HEAD TO HEAD COMPARISONS

#Obtain the results for a given year
'''old get_season_results function
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
    
    os.makedirs(rf'.\APP\data\bueno\{year}\HtH', exist_ok=True)

    results.to_csv(rf'.\APP\data\bueno\{year}\HtH\{year}_results.csv', index=False)
    sprint_results.to_csv(rf'.\APP\data\bueno\{year}\HtH\{year}_sprint_results.csv', index=False)
'''


def get_season_results(year: int) -> pd.DataFrame:  #New get_season_results function
    """
    Return a tidy DataFrame with every classified result (and points)
    for the requested championship season, **regardless of back-end**.

    The signature and the column order match the old Ergast helper so
    no downstream code has to change.
    """
    # 1) one-liner fetch from the active adapter
    df = ADAPTER.season_results(year)

    # 2) normalise column names so callers see the same headers
    rename_map = {
        "driverRef": "driverCode",       # Kaggle → Ergast naming
        "constructorRef": "constructorName",
        "name": "raceName",              # from races.csv
        "round": "round",                # already correct
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # 3) reorder for perfect backward compatibility
    wanted = [
        "raceName", "round",
        "driverCode", "constructorName",
        "grid", "position", "status",
        "points", "milliseconds", "fastestLapTime"
    ]
    df = df[[c for c in wanted if c in df.columns]]

    # 4) ↩️ keep the old “write to CSV” side-effect so plotting.py still works
    out_dir = Path(f"APP/data/bueno/{year}/HtH")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{year}_results.csv", index=False)

    return df

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
    
    os.makedirs(rf'.\APP\data\bueno\{year}\HtH', exist_ok=True)
    q_results.to_csv(rf'.\APP\data\bueno\{year}\HtH\{year}_q_results.csv', index=False)

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

    os.makedirs(rf'.\APP\data\bueno\{year}\Ritmos', exist_ok=True)
    os.makedirs(rf'.\APP\data\bueno\{year}\Ritmos\Drivers', exist_ok=True)

    # Transpose dataframe
    df_ritmos = mean_diff_df.T  
    df_ritmos.to_csv(rf'.\APP\data\bueno\{year}\Ritmos\Drivers\df_ritmos_{year}.csv')
    data = {
        "driver_palette": driver_palette,
        "driver_line": driver_line
    }

    with open(rf'.\APP\data\bueno\{year}\Ritmos\Drivers\driver_info_{year}.json', 'w') as f:
        json.dump(data, f)


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
        
        
        os.makedirs(rf'.\APP\data\bueno\{year}\Ritmos', exist_ok=True)
        os.makedirs(rf'.\APP\data\bueno\{year}\Ritmos\Teams', exist_ok=True)

        df_ritmos.to_csv(rf'.\APP\data\bueno\{year}\Ritmos\Teams\df_ritmos_{year}.csv')
        data = {
            "team_palette": team_palette
        }

        with open(rf'.\APP\data\bueno\{year}\Ritmos\Teams\team_info_{year}.json', 'w') as f:
            json.dump(data, f)


###################################################################################

# HEATMAP FUNCTIONS

#Heatmap with points per race of each driver (sns)
def season_points_heatmap(year):
    plt.style.use('tableau-colorblind10')
    ergast = Ergast()
    races = ergast.get_race_schedule(year)  # Races in year 2022
    results = []

    # For each race in the season
    for rnd, race in races['raceName'].items():

        temp = ergast.get_race_results(season=year, round=rnd + 1)
        temp = temp.content[0]

        # If there is a sprint, get the results as well
        sprint = ergast.get_sprint_results(season=year, round=rnd + 1)
        if sprint.content and sprint.description['round'][0] == rnd + 1:
            temp = pd.merge(temp, sprint.content[0], on='driverCode', how='left')
            # Add sprint points and race points to get the total
            temp['points'] = temp['points_x'] + temp['points_y']
            temp.drop(columns=['points_x', 'points_y'], inplace=True)

        # Add round no. and grand prix name
        temp['round'] = rnd + 1
        temp['race'] = race.removesuffix(' Grand Prix')
        temp = temp[['round', 'race', 'driverCode', 'points']]  # Keep useful cols.
        results.append(temp)

    # Append all races into a single dataframe
    results = pd.concat(results)
    races = results['race'].drop_duplicates()

    results = results.pivot(index='driverCode', columns='round', values='points')

    # Rank the drivers by their total points
    results['Total'] = results.sum(axis=1)
    results = results.sort_values(by='Total', ascending=False)

    # Use race name, instead of round no., as column names
    results.columns = list(races) + ['Total']

    #Set cmap boundaries
    zmin = results.iloc[:, :-1].min().min()
    zmax = results.iloc[:, :-1].max().max()

    # Define the colors
    colors = ['#ffffff', '#ffbb92']

    # Create the colormap
    cmap = LinearSegmentedColormap.from_list('custom_gradient', colors)

    # Create a heatmap using seaborn
    plt.figure(figsize=(12, 8))
    sns.heatmap(results, annot=True, fmt=".0f", cmap=cmap, cbar=False, linewidths=.5, vmin=zmin, vmax=zmax)

    # Set the title and labels
    plt.title(f'{year} Driver Points by Race', fontsize=16)
    plt.xlabel('Race', fontsize=12)
    plt.ylabel('Driver', fontsize=12)

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Adjust layout to avoid clipping
    plt.tight_layout()

    plt.gcf().set_facecolor('#ffffff')

    #Save the plot
    plt.savefig(rf'.\APP\images\points_heatmaps\{year}_drivers_points_heatmap.png')
    # Show the plot
    plt.show()


def season_points_heatmap_by_team(year):
    plt.style.use('seaborn-v0_8-bright')
    ergast = Ergast()
    races = ergast.get_race_schedule(year)
    results = []

    # For each race in the season
    for rnd, race in races['raceName'].items():
        temp = ergast.get_race_results(season=year, round=rnd + 1)
        temp = temp.content[0]

        # If there is a sprint, get the results as well
        sprint = ergast.get_sprint_results(season=year, round=rnd + 1)
        if sprint.content and sprint.description['round'][0] == rnd + 1:
            temp = pd.merge(temp, sprint.content[0], on='constructorName', how='left')
            # Add sprint points and race points to get the total
            temp['points'] = temp['points_y'] + temp['points_y']
            temp.drop(columns=['points_x', 'points_y'], inplace=True)

        # Add round no. and grand prix name
        temp['round'] = rnd + 1
        temp['race'] = race.removesuffix(' Grand Prix')
        temp = temp[['round', 'race', 'constructorName', 'points']]  # Keep useful cols.
        results.append(temp)

    results = pd.concat(results)
    races = results['race'].drop_duplicates()

    results.reset_index(drop=True, inplace=True)
    
    results_team = results.groupby(['constructorName', 'race'])['points'].sum().unstack().fillna(0)


    # Rank the drivers by their total points
    results_team['Total'] = results_team.sum(axis=1)
    results_team = results_team.sort_values(by='Total', ascending=False)

    # Use race name, instead of round no., as column names
    results_team.columns = list(races) + ['Total']

    #Set cmap boundaries
    zmin = results_team.iloc[:, :-1].min().min()
    zmax = results_team.iloc[:, :-1].max().max()

    # Define the colors
    colors = ['#ffffff', '#ffbb92']

    # Create the colormap
    cmap = LinearSegmentedColormap.from_list('custom_gradient', colors)

    # Create a heatmap using seaborn
    plt.figure(figsize=(12, 8))
    sns.heatmap(results_team, annot=True, fmt=".0f", cmap=cmap, cbar=False, linewidths=.5, vmin=zmin, vmax=zmax)

    # Set the title and labels
    plt.title(f'{year} Team Points by Race', fontsize=16)
    plt.xlabel('Race', fontsize=12)
    plt.ylabel('Team', fontsize=12)

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Adjust layout to avoid clipping
    plt.tight_layout()
    plt.gcf().set_facecolor('#ffffff')

    plt.savefig(f'.\APP\images\points_heatmaps\{year}_teams_points_heatmap.png')

    # Show the plot
    plt.show()



###################################################################################
###################################################################################

#GP FUNCTIONS

#Get result data for the GP analysis
def data_results_info(year, event):
    all_results = pd.read_csv(rf'.\APP\data\bueno\{year}\HtH\{year}_results.csv')
    results = all_results[['driverCode', 'position', 'totalRaceTime', 'status', 'points']][all_results['raceName'] == event].reset_index(drop=True)

    results = results.rename(columns={'driverCode': 'Driver', 'position':'Position', 
                                      'totalRaceTime':'Time', 'status':'Status', 'points':'Points'})

    results['Position'] = results['Position'].astype(int)
    results['Points'] = results['Points'].astype(int)
    results['Time'] = results['Time'].apply(lambda x: x if pd.isnull(x) else str(x).split(' ')[-1])
    max_time = results['Time'][0]


    for index, row in results.iterrows():
        if pd.isna(row['Time']) or row['Time'] == max_time:
            continue
        else:
            results.at[index, 'Time'] = '+' + str(row['Time'])


    os.makedirs(rf'.\APP\data\bueno\{year}\results_info', exist_ok=True)
    results.to_csv(rf'.\APP\data\bueno\{year}\results_info\{event}_results.csv', index=False)


#Calculate qualifying delta times for a given event
def data_qualifying_times(year, event):
    session = fastf1.get_session(year, event, 'Q')
    session.load()

    drivers = pd.unique(session.laps['Driver'])

    list_fastest_laps = list()
    for drv in drivers:
        drvs_fastest_lap = session.laps.pick_driver(drv).pick_fastest()
        list_fastest_laps.append(drvs_fastest_lap)
    fastest_laps = Laps(list_fastest_laps) \
        .sort_values(by='LapTime') \
        .reset_index(drop=True)

    pole_lap = fastest_laps.pick_fastest()
    delta_times = fastest_laps[['LapTime', 'Driver', 'Team']].copy()
    delta_times['LapTimeDelta'] = delta_times['LapTime'] - pole_lap['LapTime']
    delta_times['LapTimeDelta'] = delta_times['LapTimeDelta'].dt.total_seconds()

    driver_colors = {}
    for _, lap in delta_times.iterlaps():
        try:
            color = fastf1.plotting.get_team_color(lap['Team'], session=session)
        except:
            continue
        driver_colors[lap['Driver']] = color

    pole_lap_info = {
        'Driver': pole_lap['Driver'],
        'LapTime': pole_lap['LapTime'].total_seconds()
    }

    complementary_info = {
        'driver_colors': driver_colors,
        'pole_lap': pole_lap_info
    }

    os.makedirs(rf'.\APP\data\bueno\{year}\qualifying_times', exist_ok=True)

    delta_times.to_csv(rf'.\APP\data\bueno\{year}\qualifying_times\{event}_qualifying_times.csv')
    with open(rf'.\APP\data\bueno\{year}\qualifying_times\{event}_complementary_info.json', 'w') as f:
        json.dump(complementary_info, f)


#Calculate position changes during the race
def data_position_changes(year, event):
    race = fastf1.get_session(year, event, 'R')
    race.load(telemetry=False, weather=False)
    # event_name = race.event['EventName']

    drivers_style = {}
    all_laps = []
    for drv in race.drivers:
        drv_laps = race.laps.pick_driver(drv)
        final_positions = race.results['Position'].to_dict()
        drv_laps['Position'] = drv_laps['Position'].apply(lambda x: final_positions[drv] if pd.isna(x) else x)
        all_laps.append(drv_laps[['LapNumber', 'Position', 'Driver']])
        abb = drv_laps['Driver'].unique()
        if len(abb) < 1:
            continue
        else:
            abb = abb[0]
        try:
            style = fastf1.plotting.get_driver_style(identifier=abb, style=['color', 'linestyle'], session=race)
        except:
            continue
        drivers_style[drv] = style


    final_positions = {race.get_driver(driver)['Abbreviation']: pos for driver, pos in final_positions.items()}
    drivers_style = {race.get_driver(driver)['Abbreviation']: pos for driver, pos in drivers_style.items()}
    all_laps_df = pd.concat(all_laps)
    all_laps_df = all_laps_df.pivot(index='Driver', columns='LapNumber', values='Position')

    for driver, row in all_laps_df.iterrows():
        for lap in row.index:
            if pd.isna(row[lap]):
                all_laps_df.at[driver, lap] = final_positions[driver]

    os.makedirs(rf'.\APP\data\bueno\{year}\pos_changes_race', exist_ok=True)
    all_laps_df.to_csv(rf'.\APP\data\bueno\{year}\pos_changes_race\df_position_{year}_{event}.csv')
    with open(rf'.\APP\data\bueno\{year}\pos_changes_race\driver_style_{year}_{event}.json', 'w') as f:
        json.dump(drivers_style, f)

#Calculate the relative distances of the drivers to the leader in each lap
def data_relative_distances(year, event):
    race = fastf1.get_session(year, event, 'R')
    race.load()

    # Preparar datos
    laps = race.laps
    drivers = race.drivers
    event_name = race.event['EventName']
    # Crear un diccionario para almacenar el tiempo de inicio del primer piloto de cada vuelta
    first_driver_start_times = {}

    # Iterar sobre cada vuelta
    for lap in laps['LapNumber'].unique():
        # Filtrar las vueltas del piloto y seleccionar la primera posición
        first_driver_lap = laps[(laps['LapNumber'] == lap) & (laps['Position'] == 1)]

        if not first_driver_lap.empty:
            # Obtener el tiempo de inicio del primer piloto
            start_time = pd.Timedelta(first_driver_lap['Time'].values[0]).total_seconds()
            first_driver = first_driver_lap['DriverNumber'].values[0]
            
            first_driver_start_times[lap] = [start_time, first_driver]

    # # Crear un DataFrame para almacenar las distancias de cada piloto al primero en cada vuelta
    distances_to_first = pd.DataFrame(index=laps['LapNumber'].unique(), columns=drivers)
    
    # Iterar sobre cada vuelta y cada piloto
    for lap in first_driver_start_times.keys():
        for driver in drivers:
            try: 
                # Filtrar las vueltas del piloto y seleccionar la vuelta correspondiente
                driver_lap = laps[(laps['LapNumber'] == lap) & (laps['DriverNumber'] == driver)]
            except:
                continue
            if not driver_lap.empty:
                # Obtener el tiempo de inicio del piloto
                driver_start_time = pd.Timedelta(driver_lap['Time'].values[0]).total_seconds()
                # Calcular la distancia al primer piloto en segundos
                distance_to_first = driver_start_time - first_driver_start_times[lap][0]
                distances_to_first.loc[lap, driver] = distance_to_first

    # Convertir el DataFrame a tipo float
    distances_to_first.astype(float)
    # Change the column names from driverNumber to Driver (3 letter abbreviation)
    driver_abbr = laps[['DriverNumber', 'Driver']].drop_duplicates().set_index('DriverNumber')['Driver'].to_dict()
    distances_to_first.rename(columns=driver_abbr, inplace=True)

    drivers_style = {}
    for drv in distances_to_first.columns:
        try:
            style = fastf1.plotting.get_driver_style(identifier=drv, style=['color', 'linestyle'], session=race)
            drivers_style[drv] = style
        except:
            continue
    os.makedirs(rf'.\APP\data\bueno\{year}\relative_distances', exist_ok=True)

    distances_to_first.to_csv(rf'.\APP\data\bueno\{year}\relative_distances\{event}_relative_distances.csv', index=True)
    with open(rf'.\APP\data\bueno\{year}\relative_distances\{event}_styles.json', 'w') as f:
        json.dump(drivers_style, f)

#Calculate the pitstop strategies of all drivers for a given event
def data_pitstop_estrategy(year, event):
    race = fastf1.get_session(year, event, 'R')
    race.load()
    event_name = race.event['EventName']

    laps = race.laps
    drivers = race.drivers
    drivers = [race.get_driver(driver)["Abbreviation"] for driver in drivers]
    drivers_df = pd.DataFrame(drivers, columns=['Driver'])

    stints = laps[["Driver", "Stint", "Compound", "LapNumber"]]
    stints = stints.groupby(["Driver", "Stint", "Compound"])
    stints = stints.count().reset_index()

    stints = stints.rename(columns={"LapNumber": "StintLength"})

    compound_colors = {}
    for compound in stints["Compound"].unique():
        compound_color = fastf1.plotting.get_compound_color(compound, session=race)
        compound_colors[compound] = compound_color

    os.makedirs(rf'.\APP\data\bueno\{year}\pitstop_strategies', exist_ok=True)

    stints.to_csv(rf'.\APP\data\bueno\{year}\pitstop_strategies\{event_name}_pitstop_strategies.csv')
    drivers_df.to_csv(rf'.\APP\data\bueno\{year}\pitstop_strategies\{event_name}_positions.csv', index=False)
    file_path = rf'.\APP\data\bueno\{year}\pitstop_strategies\compound_colors.json'
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump(compound_colors, f)


#Calculate the telemetry data for the qualifying lap
def data_overlap_telemetries(year, event):
    session = fastf1.get_session(year, event, 'Q')
    session.load()

    drivers = session.laps.Driver.unique()
    drivers_style = {}

    telemetries = {}
    laptimes = {}

    for driver in drivers:
        try: 
            lap = session.laps.pick_driver(driver).pick_fastest()
            laptime = lap['LapTime']
            laptime = pd.to_timedelta(laptime, unit='s')
            minutes = int(laptime.total_seconds() // 60)
            seconds = int(laptime.total_seconds() % 60)
            milliseconds = int(laptime.microseconds // 1000)

            # Format as Minutes:Seconds.Milliseconds
            readable_format = f"{minutes:02}:{seconds:02}.{milliseconds:03}"

            #obtain the Abbreaviation of the driver from the driver number
            

            laptimes[driver] = readable_format
            telemetry = lap.get_car_data().add_distance()
            style = fastf1.plotting.get_driver_style(identifier=driver, style=['color', 'linestyle'], session=session)
            drivers_style[driver] = style

            telemetries[driver] = telemetry[['Distance', 'Speed', 'Throttle', 'Brake']].copy()
        
        except:
            continue

    telemetries_path = rf'.\APP\data\bueno\{year}\telemetries\{event}_telemetries.json'
    styles_path = rf'.\APP\data\bueno\{year}\telemetries\{event}_styles.json'
    laps_path = rf'.\APP\data\bueno\{year}\telemetries\{event}_laptimes.json'

    # Ensure the directory exists
    os.makedirs(os.path.dirname(telemetries_path), exist_ok=True)

    # Convert telemetries to a serializable format
    telemetries_serializable = {driver: telemetry.to_dict(orient='list') for driver, telemetry in telemetries.items()}

    # Save telemetries to json
    with open(telemetries_path, 'w') as f:
        json.dump(telemetries_serializable, f)

    # Save styles to json
    with open(styles_path, 'w') as f:
        json.dump(drivers_style, f)

    with open(laps_path, 'w') as f:
        json.dump(laptimes, f)


#Calculate the laptimes of the drivers in a given event
def data_laptimes_race(year, event):
    race = fastf1.get_session(year, event, 'R')
    race.load()
    drivers = race.laps.Driver.unique()

    drivers_style = {}
    all_lap_times = {}
    for driver in drivers:
        laps = race.laps.pick_driver(driver).pick_quicklaps(2).reset_index()
        laps['LapTime']=laps['LapTime'].dt.total_seconds()
        lap_times = laps[['LapNumber', 'LapTime']]
        all_lap_times[driver] = lap_times
        style = fastf1.plotting.get_driver_style(identifier=driver, style=['color', 'linestyle'], session=race)
        drivers_style[driver] = style

    lap_times_path = rf'.\APP\data\bueno\{year}\laptimes\{event}_laptimes.json'
    styles_path = rf'.\APP\data\bueno\{year}\laptimes\{event}_styles.json'

    # Convert telemetries to a serializable format
    all_lap_times_serializable = {driver: lap_times.to_dict(orient='list') for driver, lap_times in all_lap_times.items()}

    # Ensure the directory exists
    os.makedirs(os.path.dirname(lap_times_path), exist_ok=True)

    # Save telemetries to json
    with open(lap_times_path, 'w') as f:
        json.dump(all_lap_times_serializable, f)

    # Save styles to json
    with open(styles_path, 'w') as f:
        json.dump(drivers_style, f)



def load_all_data(year):
    get_season_results(year)
    get_season_q_results(year)
    data_year_pace_driver(year)
    data_year_pace_team(year)

    season_points_heatmap(year)
    season_points_heatmap_by_team(year)
    
    all_results = pd.read_csv(rf'.\APP\data\bueno\{year}\HtH\{year}_results.csv')
    events = all_results['raceName'].unique()

    for event in events:
        data_results_info(year, event)
        data_qualifying_times(year, event)
        data_position_changes(year, event)
        data_relative_distances(year, event)
        data_pitstop_estrategy(year, event)
        data_overlap_telemetries(year, event)
        data_laptimes_race(year, event)


if __name__ == '__main__':
    year = int(sys.argv[1])
    os.makedirs(rf'.\APP\data\bueno\{year}', exist_ok=True)
    load_all_data(year)