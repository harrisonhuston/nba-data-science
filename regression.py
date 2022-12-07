"""
Description: This program runs both linear and quadratic regression to predict points based off of average games,
minutes played, and field goal attempts. In addition, it predicts points based off of last season's games, minutes
played, and field goal attempts for each player. Lastly, it predicts points based off of a 5% increase and decrease in
both average and last seasons' games, minutes, and field goal attempts. All model results, regression line vs actual
plots for each model and player are saved as a pdf chart. All model predictions are saved as a pdf chart as well. The
system will print all results and for each model prediction, the RMSE and R-Squared for that prediction.
"""

# import the below
from data_analysis import harden_df, james_df, derozan_df
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from data_analysis import player_df_sum_avg
import warnings
# ignore warnings
warnings.filterwarnings('ignore')


""" pts_g_regression takes in input parameters of each player's dataframe, their average games, the chart name, the
degree of the polynomial, the player's name, the player's average points, and the player's games played last season.
The method trains the model for regression to predict the points based off of games. It produces both predictions and
model charts (chart of linear regression line vs plots)."""


def pts_g_regression(dataframe, avg_dataframe, chart_name, degree, player_name, avg_points, last_season_g,
                     last_season_pts):
    # define x and y based off parameter dataframe as type int
    X = np.array(dataframe.iloc[:, 2].values).astype(int)
    y = np.array(dataframe.iloc[:, 1].values).astype(int)
    # train_test_split with test_size = .3, random state=0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
    # define weights fit x and y train, and the degree parameter
    weights = np.polyfit(X_train, y_train, degree)
    # define model np.poly1d for weights
    model = np.poly1d(weights)
    # y_pred = model for X_test
    y_pred = model(X_test)
    # future_prediction = the model for avg_dataframe from input parameter
    future_prediction = model(avg_dataframe)
    # last_season_future_prediction = model for last season games input parameter
    last_season_future_prediction = model(last_season_g)
    # define future increase and decrease for 5% for both avg_dataframe and last_season_g
    future_increase = avg_dataframe * 1.05
    future_decrease = avg_dataframe * .95
    last_season_increase = last_season_g * 1.05
    last_season_decrease = last_season_g * .95
    # predict with same model for increase and decrease for both average games and last season's games
    future_prediction_increase = model(future_increase)
    future_prediction_decrease = model(future_decrease)
    last_season_prediction_increase = model(last_season_increase)
    last_season_prediction_decrease = model(last_season_decrease)
    # print string plus input parameter chart name + degrees
    print(str(chart_name) + " : " + str(degree) + ' degree(s)')
    # calculate rmse and print
    rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
    print("RMSE = " + str('{:,}'.format(rmse)))
    # calculate r_squared and print
    r_squared = (r2_score(y_test, y_pred))
    print("R-Squared = " + str(r_squared))
    # print string plus avg_points ie player's average points per season
    print("Average Points Per Season = " + str(avg_points))
    # print strings with player name and all predictions for avg games
    print('\n*** Based off of this model, if ' + str(player_name) + ' plays his average games of ' + str(avg_dataframe)
          + ' next season, he will score ' + str(future_prediction) + ' total points in the season. ***\n')
    print('*** Based off of this model, if ' + str(player_name) + ' plays 5% less games than his average games of '
          + str(avg_dataframe) + ' (' + str(future_decrease) + ' games)' + ' next season, he will score ' +
          str(future_prediction_decrease) + ' total points in the season. ***\n')
    print('*** Based off of this model, if ' + str(player_name) + ' plays 5% more games than his average games of '
          + str(avg_dataframe) + ' (' + str(future_increase) + ' games)' + ' next season, he will score ' +
          str(future_prediction_increase) + ' total points in the season. ***\n')

    # print strings with player name and all predictions for last season's games
    print('\n*** Based off of this model, if ' + str(player_name) + ' plays the same amount of games as last season, '
          'which is '+ str(last_season_g) + ' games, next season, he will score '
          + str(last_season_future_prediction) + ' total points in the season. ***\n')
    print('*** Based off of this model, if ' + str(player_name) + ' plays 5% less games than his ' + str(last_season_g)
          + ' games last season, (' + str(last_season_decrease) + ' games), next season, he will score ' +
          str(last_season_prediction_decrease) + ' total points in the season. ***\n')
    print('*** Based off of this model, if ' + str(player_name) + ' plays 5% more games than his ' + str(last_season_g)
          + ' games last season (' + str(last_season_increase) + ' games) next season, he will score ' +
          str(last_season_prediction_increase) + ' total points in the season. ***\n')

    # set figure size define x and y
    figure(figsize=(15, 15), dpi=80)
    y = ['5% Decrease\nLast Season GP', '5% Increase\nLast Season GP', 'Model Prediction With\nLast Season GP',
         'Points\nLast Season', '5% Decrease\nAverage GP', '5% Increase\nAverage GP',
         'Model Prediction With\nAverage GP', 'Average Points\nPer Season']
    x = [float(round(last_season_prediction_decrease, 2)), float(round(last_season_prediction_increase, 2)),
         float(round(last_season_future_prediction, 2)), int(last_season_pts),
         float(round(future_prediction_decrease, 2)), float(round(future_prediction_increase, 2)),
         float(round(future_prediction, 2)), float(round(avg_points, 2))]
    # plt bar chart horizontal and annotate
    plt.barh(y, x)
    for i in range(len(x)):
        plt.annotate(x[i], (x[i], y[i]), horizontalalignment='right')
    # set y and x label, title, save figure and clear
    plt.ylabel('')
    plt.xlabel(str(player_name) + ' Model Prediction\nAverage Games Played (GP) Per Season: ' +
               str(float(round(avg_dataframe, 3))) + '\nGames Played (GP) Last Season: ' + str(last_season_g) +
               '\nRMSE: ' + str(float(round(rmse, 4))) + ' | R-Squared: ' + str(float(round(r_squared, 4))))
    plt.title(str(degree) + ' : Degree(s) : ' + str(chart_name) + ' Predictions (Rounded)')
    plt.savefig('Regression/Regression Predictions/Points VS Games/' + str(degree) + ' degree(s) ' + str(player_name) +
                ' Points VS Games.pdf')
    plt.clf()

    # define chart size sns scatter
    figure(figsize=(15, 15), dpi=80)
    sns.regplot(x=y_test, y=y_pred, ci=None, scatter_kws={"color": "red"})
    # set title, x, y, and save figure
    plt.title(str(degree) + ' : Degree(s) : ' + str(chart_name))
    plt.xlabel('Points Actual (Scatter)')
    plt.ylabel('Points Predicted (Line)')
    plt.savefig('Regression/Regression Models/Points VS Games/' + str(degree) + ' degree(s) ' + str(player_name) +
                ' Points VS Games.pdf')
    # clear plot
    plt.clf()


""" pts_mp_regression takes in input parameters of each player's dataframe, their average minutes, the chart name, the
degree of the polynomial, the player's name, the player's average points, and the player's minutes played last 
season. The method trains the model for regression to predict the points based off of minutes. It produces both
predictions and model charts (chart of linear regression line vs plots)."""


def pts_mp_regression(dataframe, avg_dataframe, chart_name, degree, player_name, avg_points, last_season_mp,
                      last_season_pts):
    # define x and y based off parameter dataframe as type int
    X = np.array(dataframe.iloc[:, 3].values).astype(int)
    y = np.array(dataframe.iloc[:, 1].values).astype(int)
    # train_test_split with test_size = .3, random state=0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
    # define weights fit x and y train, and the degree parameter
    weights = np.polyfit(X_train, y_train, degree)
    # define model np.poly1d for weights
    model = np.poly1d(weights)
    # y_pred = model for X_test
    y_pred = model(X_test)
    # future_prediction = the model for avg_dataframe from input parameter
    future_prediction = model(avg_dataframe)
    # last_season_future_prediction = model for last season minutes played input parameter
    last_season_future_prediction = model(last_season_mp)
    # define future increase and decrease for 5% for both avg_dataframe and last_season_mp
    future_increase = avg_dataframe * 1.05
    future_decrease = avg_dataframe * .95
    last_season_increase = last_season_mp * 1.05
    last_season_decrease = last_season_mp * .95
    # predict with same model for increase and decrease for both average games and last season's minutes
    future_prediction_increase = model(future_increase)
    future_prediction_decrease = model(future_decrease)
    last_season_prediction_increase = model(last_season_increase)
    last_season_prediction_decrease = model(last_season_decrease)
    # print string plus input parameter chart name + degrees
    print(str(chart_name) + " : " + str(degree) + ' degree(s)')
    # calculate rmse and print
    rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
    print("RMSE = " + str('{:,}'.format(rmse)))
    # calculate r_squared and print
    r_squared = (r2_score(y_test, y_pred))
    print("R-Squared = " + str(r_squared))
    # print string plus avg_points ie player's average points per season
    print("Average Points Per Season = " + str(avg_points))
    # print strings with player name and all predictions for avg minutes
    print('\n*** Based off of this model, if ' + str(player_name) + ' plays his average minutes of ' +
          str(avg_dataframe) + ' next season, he will score ' + str(future_prediction) +
          ' total points in the season. ***\n')
    print('*** Based off of this model, if ' + str(player_name) + ' plays 5% less minutes than his average games of '
          + str(avg_dataframe) + ' (' + str(future_decrease) + ' minutes)' + ' next season, he will score ' +
          str(future_prediction_decrease) + ' total points in the season. ***\n')
    print('*** Based off of this model, if ' + str(player_name) + ' plays 5% more minutes than his average games of '
          + str(avg_dataframe) + ' (' + str(future_increase) + ' minutes)' + ' next season, he will score ' +
          str(future_prediction_increase) + ' total points in the season. ***\n')

    # print strings with player name and all predictions for last season's games
    print('\n*** Based off of this model, if ' + str(player_name) +
          ' plays the same amount of minutes as last season, which is ' + str(last_season_mp) +
          ' minutes, next season, he will score ' + str(last_season_future_prediction) +
          ' total points in the season. ***\n')
    print('*** Based off of this model, if ' + str(player_name) + ' plays 10% less minutes than his ' +
          str(last_season_mp) + ' minutes last season, (' + str(last_season_decrease) +
          ' minutes), next season, he will score ' + str(last_season_prediction_decrease) +
          ' total points in the season. ***\n')
    print('*** Based off of this model, if ' + str(player_name) + ' plays 10% more minutes than his ' +
          str(last_season_mp) + ' minutes last season (' + str(last_season_increase) +
          ' minutes) next season, he will score ' + str(last_season_prediction_increase) +
          ' total points in the season. ***\n')

    # set figure size define x and y
    figure(figsize=(15, 15), dpi=80)
    y = ['5% Decrease\nLast Season MP', '5% Increase\nLast Season MP', 'Model Prediction With\nLast Season MP',
         'Points\nLast Season', '5% Decrease\nAverage MP', '5% Increase\nAverage MP',
         'Model Prediction With\nAverage MP', 'Average Points\nPer Season']
    x = [float(round(last_season_prediction_decrease, 2)), float(round(last_season_prediction_increase, 2)),
         float(round(last_season_future_prediction, 2)), int(last_season_pts),
         float(round(future_prediction_decrease, 2)), float(round(future_prediction_increase, 2)),
         float(round(future_prediction, 2)), float(round(avg_points, 2))]
    # plt bar chart horizontal and annotate
    plt.barh(y, x)
    for i in range(len(x)):
        plt.annotate(x[i], (x[i], y[i]), horizontalalignment='right')
    # set y and x label, title, save figure and clear
    plt.ylabel('')
    plt.xlabel(str(player_name) + ' Model Prediction\nAverage Minutes Played (MP) Per Season: ' +
               str(float(round(avg_dataframe, 3))) + '\nMinutes Played (MP) Last Season: ' + str(last_season_mp) +
               '\nRMSE: ' + str(float(round(rmse, 4))) + ' | R-Squared: ' + str(float(round(r_squared, 4))))
    plt.title(str(degree) + ' : Degree(s) : ' + str(chart_name) + ' Predictions (Rounded)')
    plt.savefig('Regression/Regression Predictions/Points VS Minutes/' + str(degree) + ' degree(s) ' + str(player_name)
                + ' Points VS Minutes.pdf')
    plt.clf()

    # define chart size sns scatter
    figure(figsize=(15, 15), dpi=80)
    sns.regplot(x=y_test, y=y_pred, ci=None, scatter_kws={"color": "red"})
    # set title, x, y, and save figure
    plt.title(str(degree) + ' : Degree(s) : ' + str(chart_name))
    plt.xlabel('Points Actual (Scatter)')
    plt.ylabel('Points Predicted (Line)')
    plt.savefig('Regression/Regression Models/Points VS Minutes/' + str(degree) + ' degree(s) ' + str(player_name) +
                ' Points VS Minutes.pdf')
    # clear plot
    plt.clf()


""" pts_fga_regression takes in input parameters of each player's dataframe, their average field goal attempts, the 
chart name, the degree of the polynomial, the player's name, the player's average points, and the player's field goal
attempts last season.The method trains the model for regression to predict the points based off of field goal attempts. 
It produces both predictions and model charts (chart of linear regression line vs plots)."""


def pts_fga_regression(dataframe, avg_dataframe, chart_name, degree, player_name, avg_points, last_season_fga,
                       last_season_pts):
    # define x and y based off parameter dataframe as type int
    X = np.array(dataframe.iloc[:, 4].values).astype(int)
    y = np.array(dataframe.iloc[:, 1].values).astype(int)
    # train_test_split with test_size = .3, random state=0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
    # define weights fit x and y train, and the degree parameter
    weights = np.polyfit(X_train, y_train, degree)
    # define model np.poly1d for weights
    model = np.poly1d(weights)
    # y_pred = model for X_test
    y_pred = model(X_test)
    # future_prediction = the model for avg_dataframe from input parameter
    future_prediction = model(avg_dataframe)
    # last_season_future_prediction = model for last season field goal attempts input parameter
    last_season_future_prediction = model(last_season_fga)
    # define future increase and decrease for 5% for both avg_dataframe and last_season_fga
    future_increase = avg_dataframe * 1.05
    future_decrease = avg_dataframe * .95
    last_season_increase = last_season_fga * 1.05
    last_season_decrease = last_season_fga * .95
    # predict with same model for increase and decrease for both average games and last season's fga
    future_prediction_increase = model(future_increase)
    future_prediction_decrease = model(future_decrease)
    last_season_prediction_increase = model(last_season_increase)
    last_season_prediction_decrease = model(last_season_decrease)
    # print string plus input parameter chart name + degrees
    print(str(chart_name) + " : " + str(degree) + ' degree(s)')
    # calculate rmse and print
    rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
    print("RMSE = " + str('{:,}'.format(rmse)))
    # calculate r_squared and print
    r_squared = (r2_score(y_test, y_pred))
    print("R-Squared = " + str(r_squared))
    # print string plus avg_points ie player's average points per season
    print("Average Points Per Season = " + str(avg_points))
    # print strings with player name and all predictions for avg minutes
    print('\n*** Based off of this model, if ' + str(player_name) + ' takes his average field goal attempts of '
          + str(avg_dataframe) + ' next season, he will score ' + str(future_prediction) +
          ' total points in the season. ***\n')
    print('*** Based off of this model, if ' + str(player_name) +
          ' takes 5% less field goal attempts than his average games of ' + str(avg_dataframe) + ' (' +
          str(future_decrease) + ' field goal attempts)' + ' next season, he will score ' +
          str(future_prediction_decrease) + ' total points in the season. ***\n')
    print('*** Based off of this model, if ' + str(player_name) +
          ' takes 5% more field goal attempts than his average games of '
          + str(avg_dataframe) + ' (' + str(future_increase) + ' field goal attempts)' + ' next season, he will score '
          + str(future_prediction_increase) + ' total points in the season. ***\n')

    # print strings with player name and all predictions for last season's games
    print('\n*** Based off of this model, if ' + str(player_name) +
          ' takes the same amount of field goal attempts as last season, which is '
          + str(last_season_fga) + ' field goal attempts, next season, he will score '
          + str(last_season_future_prediction) + ' total points in the season. ***\n')
    print('*** Based off of this model, if ' + str(player_name) + ' takes 5% less field goal attempts than his ' +
          str(last_season_fga) + ' field goal attempts last season, (' + str(last_season_decrease)
          + ' field goal attempts), next season, he will score ' + str(last_season_prediction_decrease) +
          ' total points in the season. ***\n')
    print('*** Based off of this model, if ' + str(player_name) + ' takes 5% more field goal attempts than his ' +
          str(last_season_fga) + ' field goal attempts last season (' + str(last_season_increase)
          + ' field goal attempts) next season, he will score ' + str(last_season_prediction_increase) +
          ' total points in the season. ***\n')

    # set figure size define x and y
    figure(figsize=(15, 15), dpi=80)
    y = ['5% Decrease\nLast Season FGA', '5% Increase\nLast Season FGA',
         'Model Prediction With\nLast Season FGA', 'Points\nLast Season', '5% Decrease\nAverage FGA',
         '5% Increase\nAverage FGA', 'Model Prediction With\nAverage FGA',
         'Average Points\nPer Season']
    x = [float(round(last_season_prediction_decrease, 2)), float(round(last_season_prediction_increase, 2)),
         float(round(last_season_future_prediction, 2)), int(last_season_pts),
         float(round(future_prediction_decrease, 2)), float(round(future_prediction_increase, 2)),
         float(round(future_prediction, 2)), float(round(avg_points, 2))]
    # plt bar chart horizontal and annotate
    plt.barh(y, x)
    for i in range(len(x)):
        plt.annotate(x[i], (x[i], y[i]), horizontalalignment='right')
    # set y and x label, title, save figure and clear
    plt.ylabel('')
    plt.xlabel(str(player_name) + ' Model Prediction\nAverage Field Goal Attempts (FGA) Per Season: ' +
               str(float(round(avg_dataframe, 3))) + '\nField Goal Attempts (FGA) Last Season: ' + str(last_season_fga)
               + '\nRMSE: ' + str(float(round(rmse, 4))) + ' | R-Squared: ' + str(float(round(r_squared, 4))))
    plt.title(str(degree) + ' : Degree(s) : ' + str(chart_name) + ' Predictions (Rounded)')
    plt.savefig(
        'Regression/Regression Predictions/Points VS FGA/' + str(degree) + ' degree(s) ' + str(player_name) +
        ' Points VS FGA.pdf')
    plt.clf()

    # define chart size sns scatter
    figure(figsize=(15, 15), dpi=80)
    sns.regplot(x=y_test, y=y_pred, ci=None, scatter_kws={"color": "red"})
    # set title, x, y, and save figure
    plt.title(str(degree) + ' : Degree(s) : ' + str(chart_name))
    plt.xlabel('Points Actual (Scatter)')
    plt.ylabel('Points Predicted (Line)')
    plt.savefig('Regression/Regression Models/Points VS FGA/' + str(degree) + ' degree(s) ' + str(player_name) +
                ' Points VS FGA.pdf')
    # clear plot
    plt.clf()

# print string
print('*****\nAll below regression data is non-rounded for complete data purposes. Individual regression prediction'
      ' charts are rounded in the following manner: \n- Games, Minutes Played, and Field Goal Attempts : 3 decimal '
      'places\n- Points Scored : 2 decimal places\n- RSME and R-Squared : 4 decimal places\n*****\n')

# call method pts_g_regression (LeBron James with linear and quadratic regression)
pts_g_regression(james_df, player_df_sum_avg.iloc[0, 1], 'LeBron James Points vs Games', 1, 'LeBron James',
                 player_df_sum_avg.iloc[0, 4], james_df.iloc[9, 2], james_df.iloc[9, 1])
pts_g_regression(james_df, player_df_sum_avg.iloc[0, 1], 'LeBron James Points vs Games', 2, 'LeBron James',
                 player_df_sum_avg.iloc[0, 4], james_df.iloc[9, 2], james_df.iloc[9, 1])

# call method pt_g_regression (James Harden with linear and quadratic regression)
pts_g_regression(harden_df, player_df_sum_avg.iloc[1, 1], 'James Harden Points vs Games', 1, 'James Harden',
                 player_df_sum_avg.iloc[1, 4], harden_df.iloc[9, 2], harden_df.iloc[9, 1])
pts_g_regression(harden_df, player_df_sum_avg.iloc[1, 1], 'James Harden Points vs Games', 2, 'James Harden',
                 player_df_sum_avg.iloc[1, 4], harden_df.iloc[9, 2], harden_df.iloc[9, 1])

# call method pt_g_regression (DeMar Derozan with linear and quadratic regression)
pts_g_regression(derozan_df, player_df_sum_avg.iloc[2, 1], 'DeMar Derozan Points vs Games', 1, 'DeMar Derozan',
                 player_df_sum_avg.iloc[2, 4], derozan_df.iloc[9, 2], derozan_df.iloc[9, 1])
pts_g_regression(derozan_df, player_df_sum_avg.iloc[2, 1], 'DeMar Derozan Points vs Games', 2, 'DeMar Derozan',
                 player_df_sum_avg.iloc[2,4], derozan_df.iloc[9, 2], derozan_df.iloc[9, 1])

# call method pts_mp_regression (LeBron James with linear and quadratic regression)
pts_mp_regression(james_df, player_df_sum_avg.iloc[0, 2], 'LeBron James Points vs Minutes', 1, 'LeBron James',
                  player_df_sum_avg.iloc[0, 4], james_df.iloc[9, 3], james_df.iloc[9, 1])
pts_mp_regression(james_df, player_df_sum_avg.iloc[2, 2], 'LeBron James Points vs Minutes', 2, 'LeBron James',
                   player_df_sum_avg.iloc[0, 4], james_df.iloc[9, 3], james_df.iloc[9, 1])

# call method pts_mp_regression (James Harden with linear and quadratic regression)
pts_mp_regression(harden_df, player_df_sum_avg.iloc[1, 2], 'James Harden Points vs Minutes', 1, 'James Harden',
                   player_df_sum_avg.iloc[1, 4], harden_df.iloc[9, 3], harden_df.iloc[9, 1])
pts_mp_regression(harden_df, player_df_sum_avg.iloc[1, 2], 'James Harden Points vs Minutes', 2, 'James Harden',
                  player_df_sum_avg.iloc[1, 4], harden_df.iloc[9, 3], harden_df.iloc[9, 1])

# call method pts_mp_regression (DeMar Derozan with linear and quadratic regression)
pts_mp_regression(derozan_df, player_df_sum_avg.iloc[2, 2], 'DeMar Derozan Points vs Minutes', 1, 'DeMar Derozan',
                  player_df_sum_avg.iloc[2, 4], derozan_df.iloc[9, 3], derozan_df.iloc[9, 1])
pts_mp_regression(derozan_df, player_df_sum_avg.iloc[2, 2], 'DeMar Derozan Points vs Minutes', 2, 'DeMar Derozan',
                  player_df_sum_avg.iloc[2, 4], derozan_df.iloc[9, 3], derozan_df.iloc[9, 1])

# call method pts_fga_regression (LeBron James with linear and quadratic regression)
pts_fga_regression(james_df, player_df_sum_avg.iloc[0, 3], 'LeBron James Points vs Field Goal Attempts', 1,
                   'LeBron James', player_df_sum_avg.iloc[0, 4], james_df.iloc[9, 4], james_df.iloc[9, 1])
pts_fga_regression(james_df, player_df_sum_avg.iloc[0, 3], 'LeBron James Points vs Field Goal Attempts', 2,
                  'LeBron James', player_df_sum_avg.iloc[0, 4], james_df.iloc[9, 4], james_df.iloc[9, 1])

# call method pts_mp_regression (James Harden with linear and quadratic regression)
pts_fga_regression(harden_df, player_df_sum_avg.iloc[1, 3], 'James Harden Points vs Field Goal Attempts', 1,
                   'James Harden', player_df_sum_avg.iloc[1, 4], harden_df.iloc[9, 4], harden_df.iloc[9, 1])
pts_fga_regression(harden_df, player_df_sum_avg.iloc[1, 3], 'James Harden Points vs Field Goal Attempts', 2,
                   'James Harden', player_df_sum_avg.iloc[1, 4], harden_df.iloc[9, 4], harden_df.iloc[9, 1])

# call method pts_mp_regression (DeMar Derozan with linear and quadratic regression)
pts_fga_regression(derozan_df, player_df_sum_avg.iloc[2, 3], 'DeMar Derozan Points vs Field Goal Attempts', 1,
                   'DeMar Derozan', player_df_sum_avg.iloc[2, 4], derozan_df.iloc[9, 4], derozan_df.iloc[9, 1])
pts_fga_regression(derozan_df, player_df_sum_avg.iloc[2, 3], 'DeMar Derozan Points vs Field Goal Attempts', 2,
                   'DeMar Derozan', player_df_sum_avg.iloc[2, 4], derozan_df.iloc[9, 4], derozan_df.iloc[9, 1])