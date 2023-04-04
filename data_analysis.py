"""
Description: This program does the majority of the data analysis. It takes every season from 2012 - 2022 and finds the
players that were in the top 50 in scoring for each and every season. It compares data related to Points, Field Goal
Attempts, Minutes Played, and Games played and charts this data. It creates charts for averages and totals of the
above mentioned categories for both individual seasons and all seasons that were analyzed. Heat Maps for correlation
of the above categories are created for each player individually as well as for all players. It also creates many of the
important dataframes used in the analysis and predictions throughout the project.
"""

# import the below
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import warnings
import seaborn as sn

# ignore warnings
warnings.filterwarnings('ignore')


""" pts_vs_mp takes input parameters season_df, top_df, and season. From the input parameters, the function outputs a
chart with the minutes played and points scored for the specified season, for each player that is in the top 50 in each
and every season from 2012 - 2022. """


def pts_vs_mp(season_df, top_df, season):
    # for name in list_top_50
    for name in list_of_top_50:
        # row = player name in season_df
        row = (season_df[(season_df['Player'] == name)])
        # append row to top_df
        top_df = top_df._append(row, ignore_index=True)
    # define x = top_df MP, y = top_df PTS, text = top_df Player, text_mp = top_df MP
    x = top_df["MP"]
    y = top_df["PTS"]
    text_player = top_df["Player"]
    text_mp = top_df["MP"]
    # set figure size and label each point with player name and minutes played
    figure(figsize=(15, 15), dpi=80)
    plt.scatter(x, y)
    for i in range(len(x)):
        plt.annotate(text_player[i] + '   ', (x[i], y[i]), horizontalalignment='right')
    for i in range(len(x)):
        plt.annotate(' ' + str(text_mp[i]) + ' MP', (x[i], y[i]), horizontalalignment='left')
    # set x and y label for plot, set title and save figure
    plt.xlabel('Minutes Played (MP)')
    plt.ylabel('Points Scored (PTS)')
    plt.title(season + ' Regular Season Points Scored vs. Minutes Played')
    plt.savefig('Statistics/All Players/Points VS Minutes/' + str(season) + ' PTS VS MP.pdf')
    # clear plot
    plt.clf()


""" pts_vs_fga takes input parameters season_df, top_df, and season. From the input parameters, the function outputs a
chart with the field goals attempted and points scored for the specified season, for each player that is in the top 50 
in each and every season from 2012 - 2022. """


def pts_vs_fga(season_df, top_df, season):
    # for name in list_top_50
    for name in list_of_top_50:
        # row = player name in season_df
        row = (season_df[(season_df['Player'] == name)])
        # append row to top_df
        top_df = top_df._append(row, ignore_index=True)
    # define x = top_df FGA, y = top_df PTS, text = top_df Player, text_mp = top_df FGA
    x = top_df["FGA"]
    y = top_df["PTS"]
    text_player = top_df["Player"]
    text_fga = top_df["FGA"]
    # set figure size and label each point with player name and field goal attempts
    figure(figsize=(15, 15), dpi=80)
    plt.scatter(x, y)
    for i in range(len(x)):
        plt.annotate(text_player[i] + '   ', (x[i], y[i]), horizontalalignment='right')
    for i in range(len(x)):
        plt.annotate(' ' + str(text_fga[i]) + ' FGA', (x[i], y[i]), horizontalalignment='left')
    # set x and y label for plot, set title and save figure
    plt.xlabel('Field Goal Attempts (FGA)')
    plt.ylabel('Points Scored (PTS)')
    plt.title(season + ' Regular Season Points Scored vs Field Goals Attempted')
    plt.savefig('Statistics/All Players/Points VS FGA/' + str(season) + ' PTS VS FGA.pdf')
    # clear plot
    plt.clf()


""" pts_vs_g takes input parameters season_df, top_df, and season. From the input parameters, the function outputs a
chart with the games played and points scored for the specified season, for each player that is in the top 50 in each 
and every season from 2012 - 2022. """


def pts_vs_g(season_df, top_df, season):
    # for name in list_top_50
    for name in list_of_top_50:
        # row = player name in season_df
        row = (season_df[(season_df['Player'] == name)])
        # append row to top_df
        top_df = top_df._append(row, ignore_index=True)
    # define x = top_df G, y = top_df PTS, text = top_df Player, text_mp = top_df G
    x = top_df["G"]
    y = top_df["PTS"]
    text_player = top_df["Player"]
    text_g = top_df["G"]
    # set figure size and label each point with player name and games played
    figure(figsize=(15, 15), dpi=80)
    plt.scatter(x, y)
    for i in range(len(x)):
        plt.annotate(text_player[i] + '   ', (x[i], y[i]), horizontalalignment='right')
    for i in range(len(x)):
        plt.annotate(' ' + str(text_g[i]) + ' GP', (x[i], y[i]), horizontalalignment='left')
    # set x and y label for plot, set title and save figure
    plt.xlabel('Games Played (GP)')
    plt.ylabel('Points Scored (PTS)')
    plt.title(season + ' Regular Season Points Scored vs Games Played')
    plt.savefig('Statistics/All Players/Points VS Games/' + str(season) + ' PTS VS GP.pdf')
    # clear plot
    plt.clf()


# read csvs into dataframes
df_2012_2013 = pd.read_csv("2012_2013.csv")
df_2013_2014 = pd.read_csv("2013_2014.csv")
df_2014_2015 = pd.read_csv("2014_2015.csv")
df_2015_2016 = pd.read_csv("2015_2016.csv")
df_2016_2017 = pd.read_csv("2016_2017.csv")
df_2017_2018 = pd.read_csv("2017_2018.csv")
df_2018_2019 = pd.read_csv("2018_2019.csv")
df_2019_2020 = pd.read_csv("2019_2020.csv")
df_2020_2021 = pd.read_csv("2020_2021.csv")
df_2021_2022 = pd.read_csv("2021_2022.csv")

# add column with season to each dataframe
df_2012_2013["Season"] = '2012-2013'
df_2013_2014["Season"] = '2013-2014'
df_2014_2015["Season"] = '2014-2015'
df_2015_2016["Season"] = '2015-2016'
df_2016_2017["Season"] = '2016-2017'
df_2017_2018["Season"] = '2017-2018'
df_2018_2019["Season"] = '2018-2019'
df_2019_2020["Season"] = '2019-2020'
df_2020_2021["Season"] = '2020-2021'
df_2021_2022["Season"] = '2021-2022'

# drop all but top 50
df_2012_2013.drop(df_2012_2013.loc[50:499].index, inplace=True)
df_2013_2014.drop(df_2013_2014.loc[50:499].index, inplace=True)
df_2014_2015.drop(df_2014_2015.loc[50:499].index, inplace=True)
df_2015_2016.drop(df_2015_2016.loc[50:499].index, inplace=True)
df_2016_2017.drop(df_2016_2017.loc[50:499].index, inplace=True)
df_2017_2018.drop(df_2017_2018.loc[50:499].index, inplace=True)
df_2018_2019.drop(df_2018_2019.loc[50:499].index, inplace=True)
df_2019_2020.drop(df_2019_2020.loc[50:499].index, inplace=True)
df_2020_2021.drop(df_2020_2021.loc[50:499].index, inplace=True)
df_2021_2022.drop(df_2021_2022.loc[50:499].index, inplace=True)

# remove duplicates by merging to see what players have been on list for all 8 years
df_all_players = pd.merge(df_2012_2013, df_2013_2014, on=["Player"], how='left', indicator=True)
df_all_players = (df_all_players.loc[df_all_players._merge == 'both', ['Player']])
df_all_players = pd.merge(df_all_players, df_2014_2015, on=["Player"], how='left', indicator=True)
df_all_players = (df_all_players.loc[df_all_players._merge == 'both', ['Player']])
df_all_players = pd.merge(df_all_players, df_2015_2016, on=["Player"], how='left', indicator=True)
df_all_players = (df_all_players.loc[df_all_players._merge == 'both', ['Player']])
df_all_players = pd.merge(df_all_players, df_2016_2017, on=["Player"], how='left', indicator=True)
df_all_players = (df_all_players.loc[df_all_players._merge == 'both', ['Player']])
df_all_players = pd.merge(df_all_players, df_2017_2018, on=["Player"], how='left', indicator=True)
df_all_players = (df_all_players.loc[df_all_players._merge == 'both', ['Player']])
df_all_players = pd.merge(df_all_players, df_2018_2019, on=["Player"], how='left', indicator=True)
df_all_players = (df_all_players.loc[df_all_players._merge == 'both', ['Player']])
df_all_players = pd.merge(df_all_players, df_2019_2020, on=["Player"], how='left', indicator=True)
df_all_players = (df_all_players.loc[df_all_players._merge == 'both', ['Player']])
df_all_players = pd.merge(df_all_players, df_2020_2021, on=["Player"], how='left', indicator=True)
df_all_players = (df_all_players.loc[df_all_players._merge == 'both', ['Player']])
df_all_players = pd.merge(df_all_players, df_2021_2022, on=["Player"], how='left', indicator=True)
df_all_players = (df_all_players.loc[df_all_players._merge == 'both', ['Player']])
df_all_players = pd.merge(df_all_players, df_2021_2022, on=["Player"], how='left', indicator=True)
df_all_players = (df_all_players.loc[df_all_players._merge == 'both', ['Player']])

# list_of_top_50 all players added to list
list_of_top_50 = df_all_players["Player"].tolist()

# create dataframes for each season and all seasons
top_df_2012_2013 = pd.DataFrame(columns=['Rk', 'Player', 'Pos', 'Age', 'Tm', 'G','GS', 'MP', 'FG', 'FGA', 'FG%', '3P',
                                         '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB',
                                         'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Season'])
top_df_2013_2014 = pd.DataFrame(columns=['Rk', 'Player', 'Pos', 'Age', 'Tm', 'G','GS', 'MP', 'FG', 'FGA', 'FG%', '3P',
                                         '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB',
                                         'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Season'])
top_df_2014_2015 = pd.DataFrame(columns=['Rk', 'Player', 'Pos', 'Age', 'Tm', 'G','GS', 'MP', 'FG', 'FGA', 'FG%', '3P',
                                         '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB',
                                         'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Season'])
top_df_2015_2016 = pd.DataFrame(columns=['Rk', 'Player', 'Pos', 'Age', 'Tm', 'G','GS', 'MP', 'FG', 'FGA', 'FG%', '3P',
                                         '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB',
                                         'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Season'])
top_df_2016_2017 = pd.DataFrame(columns=['Rk', 'Player', 'Pos', 'Age', 'Tm', 'G','GS', 'MP', 'FG', 'FGA', 'FG%', '3P',
                                         '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB',
                                         'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Season'])
top_df_2017_2018 = pd.DataFrame(columns=['Rk', 'Player', 'Pos', 'Age', 'Tm', 'G','GS', 'MP', 'FG', 'FGA', 'FG%', '3P',
                                         '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB',
                                         'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Season'])
top_df_2018_2019 = pd.DataFrame(columns=['Rk', 'Player', 'Pos', 'Age', 'Tm', 'G','GS', 'MP', 'FG', 'FGA', 'FG%', '3P',
                                         '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB',
                                         'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Season'])
top_df_2019_2020 = pd.DataFrame(columns=['Rk', 'Player', 'Pos', 'Age', 'Tm', 'G','GS', 'MP', 'FG', 'FGA', 'FG%', '3P',
                                         '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB',
                                         'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Season'])
top_df_2020_2021 = pd.DataFrame(columns=['Rk', 'Player', 'Pos', 'Age', 'Tm', 'G','GS', 'MP', 'FG', 'FGA', 'FG%', '3P',
                                         '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB',
                                         'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Season'])
top_df_2021_2022 = pd.DataFrame(columns=['Rk', 'Player', 'Pos', 'Age', 'Tm', 'G','GS', 'MP', 'FG', 'FGA', 'FG%', '3P',
                                         '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB',
                                         'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Season'])
top_df_all_seasons = pd.DataFrame(columns=['Rk', 'Player', 'Pos', 'Age', 'Tm', 'G','GS', 'MP', 'FG', 'FGA', 'FG%', '3P',
                                           '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB',
                                           'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Season'])
player_df = pd.DataFrame(columns=['Rk', 'Player', 'Pos', 'Age', 'Tm', 'G','GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA',
                                  '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST',
                                  'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Season'])
player_df_all = pd.DataFrame(columns=['Rk', 'Player', 'Pos', 'Age', 'Tm', 'G','GS', 'MP', 'FG', 'FGA', 'FG%', '3P',
                                      '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB',
                                      'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Season'])
player_df_sum = pd.DataFrame(columns=['Player', 'PTS', 'G', 'MP', 'FGA'])

# append each players season in top 100 list to the top_df_all_seasons dataframe
for name in list_of_top_50:
    row = (df_2012_2013[(df_2012_2013['Player'] == name)])
    top_df_all_seasons = top_df_all_seasons._append(row, ignore_index=True)
    row = (df_2013_2014[(df_2013_2014['Player'] == name)])
    top_df_all_seasons = top_df_all_seasons._append(row, ignore_index=True)
    row = (df_2014_2015[(df_2014_2015['Player'] == name)])
    top_df_all_seasons = top_df_all_seasons._append(row, ignore_index=True)
    row = (df_2015_2016[(df_2015_2016['Player'] == name)])
    top_df_all_seasons = top_df_all_seasons._append(row, ignore_index=True)
    row = (df_2016_2017[(df_2016_2017['Player'] == name)])
    top_df_all_seasons = top_df_all_seasons._append(row, ignore_index=True)
    row = (df_2017_2018[(df_2017_2018['Player'] == name)])
    top_df_all_seasons = top_df_all_seasons._append(row, ignore_index=True)
    row = (df_2018_2019[(df_2018_2019['Player'] == name)])
    top_df_all_seasons = top_df_all_seasons._append(row, ignore_index=True)
    row = (df_2019_2020[(df_2019_2020['Player'] == name)])
    top_df_all_seasons = top_df_all_seasons._append(row, ignore_index=True)
    row = (df_2020_2021[(df_2020_2021['Player'] == name)])
    top_df_all_seasons = top_df_all_seasons._append(row, ignore_index=True)
    row = (df_2021_2022[(df_2021_2022['Player'] == name)])
    top_df_all_seasons = top_df_all_seasons._append(row, ignore_index=True)

# set index to 0
index = 0
# for each name in the list of top 50 ie players in top 50 in every season 2012 - 2022
for name in list_of_top_50:
    # Points Scored vs Season All Seasons
    # define row = top_df_all_seasons column player == name
    row = (top_df_all_seasons[(top_df_all_seasons['Player'] == name)])
    # player_df append row
    player_df = player_df._append(row, ignore_index=True)
    # set x to player df season, y player df season, text player df points
    x = player_df["Season"]
    y = player_df["PTS"]
    text = player_df["PTS"]
    # set figure size and plot x, y and annotate text on plots
    figure(figsize=(15, 15), dpi=80)
    plt.plot(x, y)
    for i in range(len(x)):
        plt.annotate(str(text[i]) + ' PTS', (x[i], y[i],))
    # set x label, y label, title and save as pdf
    plt.xlabel('Season')
    plt.ylabel('Points Scored (PTS)')
    plt.title(name + ':  Total Points Scored Per Season')
    plt.savefig('Statistics/Individual Players/Points VS Season/' + str(name) + ' 2012 - 2022 Total Points Per '
                                                                       'Season.pdf')
    # clear plot
    plt.clf()

    # Games Played vs Season All Seasons
    # set x to player df season, y player df games, text player df games
    x = player_df["Season"]
    y = player_df["G"]
    text = player_df["G"]
    # set figure size and plot x, y and annotate text on plots
    figure(figsize=(15, 15), dpi=80)
    plt.plot(x, y)
    for i in range(len(x)):
        plt.annotate(str(text[i]) + ' GP', (x[i], y[i],))
    # set x label, y label, title and save as pdf
    plt.xlabel('Season')
    plt.ylabel('Games Played (GP)')
    plt.title(name + ':  Total Games Played Per Season')
    plt.savefig('Statistics/Individual Players/Games VS Season/' + str(name) + ' 2012 - 2022 Total Games Played Per '
                                                                                'Season.pdf')
    # clear plot
    plt.clf()

    # Minutes Played vs Season All Seasons
    # set x to player df season, y player df minutes played, text player df minutes played
    x = player_df["Season"]
    y = player_df["MP"]
    text = player_df["MP"]
    # set figure size and plot x, y and annotate text on plots
    figure(figsize=(15, 15), dpi=80)
    plt.plot(x, y)
    for i in range(len(x)):
        plt.annotate(str(text[i]) + ' MP', (x[i], y[i],))
    # set x label, y label, title and save as pdf
    plt.xlabel('Season')
    plt.ylabel('Minutes Played (MP)')
    plt.title(name + ':  Total Minutes Played Per Season')
    plt.savefig('Statistics/Individual Players/Minutes VS Season/' + str(name) + ' 2012 - 2022 Total Minutes Played Per'
                                                                                 ' Season.pdf')
    # clear plot
    plt.clf()

    # Field Goal Attempts vs Season All Seasons
    # set x to player df season, y player df field goal attempts, text player df field goal attempts
    x = player_df["Season"]
    y = player_df["FGA"]
    text = player_df["FGA"]
    # set figure size and plot x, y and annotate text on plots
    figure(figsize=(15, 15), dpi=80)
    plt.plot(x, y)
    for i in range(len(x)):
        plt.annotate(str(text[i]) + ' FGA', (x[i], y[i],))
    # set x label, y label, title and save as pdf
    plt.xlabel('Season')
    plt.ylabel('Field Goal Attempts (FGA)')
    plt.title(name + ':  Total Field Goal Attempts Per Season')
    plt.savefig('Statistics/Individual Players/FGA VS Season/' + str(name) + ' 2012 - 2022 Total Field Goal Attempts '
                                                                             'Per Season.pdf')
    # clear plot
    plt.clf()

    # append row to player_df_all
    player_df_all = player_df_all._append(row)
    # player_cum = player_df_all deep copy
    player_sum = player_df_all[['Player', 'PTS', 'G', 'MP', 'FGA']].copy(deep=True)
    # player_df_pts_g_mp_fg = player sum deep copy
    player_df_pts_g_mp_fga = player_sum.copy(deep=True)
    # sum points, games, minutes, fga from player_sum df
    total_points = player_sum['PTS'].sum()
    total_games = player_sum['G'].sum()
    total_minutes = player_sum['MP'].sum()
    total_fga = player_sum['FGA'].sum()
    # populate player_df_sum with sums at index to have sum for each player g, mp, fga, pts
    player_df_sum.at[index, 'Player'] = name
    player_df_sum.at[index, 'G'] = total_games
    player_df_sum.at[index, 'MP'] = total_minutes
    player_df_sum.at[index, 'FGA'] = total_fga
    player_df_sum.at[index, 'PTS'] = total_points

    # LeBron James heatmap
    if index == 0:
        # create james_df from deep copy of player_df_pts_g_mp_fga
        james_df = player_df_pts_g_mp_fga.copy(deep=True)
        # set figure size
        figure(figsize=(15, 15), dpi=80)
        # define james_df_hm as deep copy of james_df
        james_df_hm = james_df.copy(deep=True)
        # drop player column from heat map df
        james_df_hm = james_df_hm.drop('Player', axis=1)
        # make values as type int
        james_df_hm = james_df_hm.astype(int)
        # define correlation
        corr = james_df_hm.corr()
        # create heatmap from correlation
        sn.heatmap(corr, annot=True)
        # set title, save figure and clear plot
        plt.title("LeBron James Correlation Heat Map")
        plt.subplots_adjust(bottom=0.3)
        plt.savefig('Heat Maps/Individual Players/LeBron James Correlation Heat Map.pdf')
        plt.clf()
        pass
    # James Harden heat map
    elif index == 1:
        # create harden_df from deep copy of player_df_pts_g_mp_fga
        harden_df = player_df_pts_g_mp_fga.copy(deep=True)
        # set figure size
        figure(figsize=(15, 15), dpi=80)
        # define harden_df_hm as deep copy of harden_df
        harden_df_hm = harden_df.copy(deep=True)
        # drop player column from heat map df
        harden_df_hm = harden_df_hm.drop('Player', axis=1)
        # make values as type int
        harden_df_hm = harden_df_hm.astype(int)
        # define correlation
        corr = harden_df_hm.corr()
        # create heatmap from correlation
        sn.heatmap(corr, annot=True)
        # set title, save figure and clear plot
        plt.title("James Harden Correlation Heat Map")
        plt.subplots_adjust(bottom=0.3)
        plt.savefig('Heat Maps/Individual Players/James Harden Correlation Heat Map.pdf')
        plt.clf()
        pass
    # DeMar Derozan heat map
    elif index == 2:
        # create derozan_df from deep copy of player_df_pts_g_mp_fga
        derozan_df = player_df_pts_g_mp_fga.copy(deep=True)
        # set figure size
        figure(figsize=(15, 15), dpi=80)
        # define derozan_df_hm as deep copy of derozan_df
        derozan_df_hm = derozan_df.copy(deep=True)
        # drop player column from heat map df
        derozan_df_hm = derozan_df_hm.drop('Player', axis=1)
        # make values as type int
        derozan_df_hm = derozan_df_hm.astype(int)
        # define correlation
        corr = derozan_df_hm.corr()
        # create heatmap from correlation
        sn.heatmap(corr, annot=True)
        # set title, save figure and clear plot
        plt.title("DeMar Derozan Correlation Heat Map")
        plt.subplots_adjust(bottom=0.3)
        plt.savefig('Heat Maps/Individual Players/DeMar Derozan Correlation Heat Map.pdf')
        plt.clf()
        pass

    # clears rows from dataframe
    player_df = player_df[0:0]
    player_sum = player_sum[0:0]
    player_df_all = player_df_all[0:0]
    index = index + 1

# frames = each player heat map dataframe
frames = [james_df_hm, harden_df_hm, derozan_df_hm]
# create all_player_heat_df from concat frames
all_player_heat_df = pd.concat(frames)

# set figure size
figure(figsize=(15, 15), dpi=80)
# set values as type int in all_player_heat_df
all_player_heat_df = all_player_heat_df.astype(int)
# define correlation of all players heat map
corr = all_player_heat_df.corr()
# create heatmap from correlation
sn.heatmap(corr, annot=True)
# set title, save figure, and clear plot
plt.title("All Players Correlation Heat Map")
plt.subplots_adjust(bottom=0.3)
plt.savefig('Heat Maps/All Players/All Players Correlation Heat Map.pdf')
plt.clf()

# plot total player points
# set x = player, y = points, text = points
x = player_df_sum['Player']
y = player_df_sum['PTS']
text = player_df_sum['PTS']
# set figure size, scatter plot annotate with text
figure(figsize=(15, 15), dpi=80)
plt.scatter(x, y)
for i in range(len(x)):
    plt.annotate(str(text[i]) + ' PTS', (x[i], y[i],))
# set x, y labels, title chart, save figure and clear plot
plt.xlabel('Player')
plt.ylabel('Points Scored (PTS)')
plt.title('Total Points Per Player 2012 - 2022')
plt.savefig('Statistics/All Players/Player Totals/Total Points Per Player 2012 - 2022.pdf')
plt.clf()

# plot total player games
# set x = player, y = games, text = games
x = player_df_sum['Player']
y = player_df_sum['G']
text = player_df_sum['G']
# set figure size, scatter plot annotate with text
figure(figsize=(15, 15), dpi=80)
plt.scatter(x, y)
for i in range(len(x)):
    plt.annotate(str(text[i]) + ' GP', (x[i], y[i],))
# set x, y labels, title chart, save figure and clear plot
plt.xlabel('Player')
plt.ylabel('Games Played (GP)')
plt.title('Total Games Played Per Player 2012 - 2022')
plt.savefig('Statistics/All Players/Player Totals/Total Games Played Per Player 2012 - 2022.pdf')
plt.clf()

# plot total player minutes
# set x = player, y = minutes played, text = minutes played
x = player_df_sum['Player']
y = player_df_sum['MP']
text = player_df_sum['MP']
# set figure size, scatter plot annotate with text
figure(figsize=(15, 15), dpi=80)
plt.scatter(x, y)
for i in range(len(x)):
    plt.annotate(str(text[i]) + ' MP', (x[i], y[i],))
# set x, y labels, title chart, save figure and clear plot
plt.xlabel('Player')
plt.ylabel('Minutes Played (MP)')
plt.title('Total Minutes Played Per Player 2012 - 2022')
plt.savefig('Statistics/All Players/Player Totals/Total Minutes Played Per Player 2012 - 2022.pdf')
plt.clf()

# plot total player field goal attempts
# set x = player, y = minutes field goal attempts, text = field goal attempts
x = player_df_sum['Player']
y = player_df_sum['FGA']
text = player_df_sum['FGA']
# set figure size, scatter plot annotate with text
figure(figsize=(15, 15), dpi=80)
plt.scatter(x, y)
for i in range(len(x)):
    plt.annotate(str(text[i]) + ' FGA', (x[i], y[i],))
# set x, y labels, title chart, save figure and clear plot
plt.xlabel('Player')
plt.ylabel('Field Goal Attempts (FGA)')
plt.title('Total Field Goal Attempts Per Player 2012 - 2022')
plt.savefig('Statistics/All Players/Player Totals/Total Field Goal Attempts Per Player 2012 - 2022.pdf')
plt.clf()

# create averages of sum total / 10 for 10 seasons
# player df sum avg = deep copy player df sum
player_df_sum_avg = player_df_sum.copy(deep=True)
# divide g, mp, fga, pts by 10 ie total seasons played
player_df_sum_avg = player_df_sum_avg[['G', 'MP', 'FGA', 'PTS']].div(10, axis=0)
# add player column as = to player_df_sum player column
player_df_sum_avg['Player'] = player_df_sum['Player']
# first column = player df sum avg pop Player column
first_column = player_df_sum_avg.pop('Player')
# insert first column at index 0 so it is the first column
player_df_sum_avg.insert(0, 'Player', first_column)


# average total points per player per season
# set x = player df sum avg player, y = pts, text = pts
x = player_df_sum_avg['Player']
y = player_df_sum_avg['PTS']
text = player_df_sum_avg['PTS']
# set figure size and annotate plots with text
figure(figsize=(15, 15), dpi=80)
plt.scatter(x, y)
for i in range(len(x)):
    plt.annotate(str(text[i]) + ' PTS', (x[i], y[i],))
# set x, y, title, save figure and clear plot
plt.xlabel('Player')
plt.ylabel('Average Points Scored (PTS)')
plt.title('Average Total Points Per Player Per Season 2012 - 2022')
plt.savefig('Statistics/All Players/Player Averages/Average Total Points Per Player Per Season 2012 - 2022.pdf')
plt.clf()

# average total games per player per season
# set x = player df sum avg player, y = games, text = games
x = player_df_sum_avg['Player']
y = player_df_sum_avg['G']
text = player_df_sum_avg['G']
# set figure size and annotate plots with text
figure(figsize=(15, 15), dpi=80)
plt.scatter(x, y)
for i in range(len(x)):
    plt.annotate(str(text[i]) + ' GP', (x[i], y[i],))
# set x, y, title, save figure and clear plot
plt.xlabel('Player')
plt.ylabel('Average Games Played (GP)')
plt.title('Average Total Games Played Per Player Per Season 2012 - 2022')
plt.savefig('Statistics/All Players/Player Averages/Average Total Games Played Per Player Per Season 2012 - 2022.pdf')
plt.clf()

# average total field goal attempts per player per season
# set x = player df sum avg player, y = fga, text = fga
x = player_df_sum_avg['Player']
y = player_df_sum_avg['FGA']
text = player_df_sum_avg['FGA']
# set figure size and annotate plots with text
figure(figsize=(15, 15), dpi=80)
plt.scatter(x, y)
for i in range(len(x)):
    plt.annotate(str(text[i]) + ' FGA', (x[i], y[i],))
# set x, y, title, save figure and clear plot
plt.xlabel('Player')
plt.ylabel('Average Field Goal Attempts (FGA)')
plt.title('Average Total Games Played Per Player Per Season 2012 - 2022')
plt.savefig('Statistics/All Players/Player Averages/Average Total Field Goal Attempts Per Player Per Season '
            '2012 - 2022.pdf')
plt.clf()

# average totalminutes played per player per season
# set x = player df sum avg player, y = mp, text = mp
x = player_df_sum_avg['Player']
y = player_df_sum_avg['MP']
text = player_df_sum_avg['MP']
# set figure size and annotate plots with text
figure(figsize=(15, 15), dpi=80)
plt.scatter(x, y)
for i in range(len(x)):
    plt.annotate(str(text[i]) + ' MP', (x[i], y[i],))
# set x, y, title, save figure and clear plot
plt.xlabel('Player')
plt.ylabel('Average Minutes Played (MP)')
plt.title('Average Total Minutes Played Per Player Per Season 2012 - 2022')
plt.savefig('Statistics/All Players/Player Averages/Average Total Minutes Played Per Player Per Season 2012 - 2022.pdf')
plt.clf()

# call method pts_vs_mp - each player minutes vs points for season in parameter
pts_vs_mp(df_2012_2013, top_df_2012_2013, '2012 - 2013')
pts_vs_mp(df_2013_2014, top_df_2013_2014, '2013 - 2014')
pts_vs_mp(df_2014_2015, top_df_2014_2015, '2014 - 2015')
pts_vs_mp(df_2015_2016, top_df_2015_2016, '2015 - 2016')
pts_vs_mp(df_2016_2017, top_df_2016_2017, '2016 - 2017')
pts_vs_mp(df_2017_2018, top_df_2017_2018, '2017 - 2018')
pts_vs_mp(df_2018_2019, top_df_2018_2019, '2018 - 2019')
pts_vs_mp(df_2019_2020, top_df_2019_2020, '2019 - 2020')
pts_vs_mp(df_2020_2021, top_df_2020_2021, '2020 - 2021')
pts_vs_mp(df_2021_2022,top_df_2021_2022, '2021 - 2022')

# call method pts_vs_fga - each player field goal attempts vs points for season in parameter
pts_vs_fga(df_2012_2013, top_df_2012_2013, '2012 - 2013')
pts_vs_fga(df_2013_2014, top_df_2013_2014, '2013 - 2014')
pts_vs_fga(df_2014_2015, top_df_2014_2015, '2014 - 2015')
pts_vs_fga(df_2015_2016, top_df_2015_2016, '2015 - 2016')
pts_vs_fga(df_2016_2017, top_df_2016_2017, '2016 - 2017')
pts_vs_fga(df_2017_2018, top_df_2017_2018, '2017 - 2018')
pts_vs_fga(df_2018_2019, top_df_2018_2019, '2018 - 2019')
pts_vs_fga(df_2019_2020, top_df_2019_2020, '2019 - 2020')
pts_vs_fga(df_2020_2021, top_df_2020_2021, '2020 - 2021')
pts_vs_fga(df_2021_2022,top_df_2021_2022, '2021 - 2022')

# call method pts_vs_g - each player games vs points for season in parameter
pts_vs_g(df_2012_2013, top_df_2012_2013, '2012 - 2013')
pts_vs_g(df_2013_2014, top_df_2013_2014, '2013 - 2014')
pts_vs_g(df_2014_2015, top_df_2014_2015, '2014 - 2015')
pts_vs_g(df_2015_2016, top_df_2015_2016, '2015 - 2016')
pts_vs_g(df_2016_2017, top_df_2016_2017, '2016 - 2017')
pts_vs_g(df_2017_2018, top_df_2017_2018, '2017 - 2018')
pts_vs_g(df_2018_2019, top_df_2018_2019, '2018 - 2019')
pts_vs_g(df_2019_2020, top_df_2019_2020, '2019 - 2020')
pts_vs_g(df_2020_2021, top_df_2020_2021, '2020 - 2021')
pts_vs_g(df_2021_2022,top_df_2021_2022, '2021 - 2022')
