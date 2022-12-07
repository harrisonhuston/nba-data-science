"""
Description: This program runs random forest and decision tree classifiers using the player's average minutes, games,
and field goal attempts. It runs random forest at depths of 1-2 and subtrees of 1-5 and uses the conglomerate of these
to predict the random forest classifier outcome. Predictions are based on 100 point increments and final
prediction outcomes will be an output of a range of 100 points, while individual point prediction outcomes will be
either above, below, or undecided for next season. It computes the overall accuracy for each prediction as well as all
predictions. Additionally, for each random forest prediction it will calculate the true positives, false positives,
true negatives, and false negatives for each point range, as well as the true positive and true negative rates where
applicable. Random forest predictions for each point level are also saved as PDFs displaying the predicted label and
its associated depth and number of subtrees for each point level.
"""

# import the below
from data_analysis import player_df_sum_avg
from data_analysis import harden_df
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import numpy as np
import pandas as pd

# deep copy derozan_df
harden_random_forest = harden_df.copy(deep=True)

# add labels for below or above each of the below 1200-2200 in 100 increments >= points 1, else 0
harden_random_forest['AboveBelow1200'] = np.where(harden_random_forest['PTS'] >= 1200, 1, 0)
harden_random_forest['AboveBelow1300'] = np.where(harden_random_forest['PTS'] >= 1300, 1, 0)
harden_random_forest['AboveBelow1400'] = np.where(harden_random_forest['PTS'] >= 1400, 1, 0)
harden_random_forest['AboveBelow1500'] = np.where(harden_random_forest['PTS'] >= 1500, 1, 0)
harden_random_forest['AboveBelow1600'] = np.where(harden_random_forest['PTS'] >= 1600, 1, 0)
harden_random_forest['AboveBelow1700'] = np.where(harden_random_forest['PTS'] >= 1700, 1, 0)
harden_random_forest['AboveBelow1800'] = np.where(harden_random_forest['PTS'] >= 1800, 1, 0)
harden_random_forest['AboveBelow1900'] = np.where(harden_random_forest['PTS'] >= 1900, 1, 0)
harden_random_forest['AboveBelow2000'] = np.where(harden_random_forest['PTS'] >= 2000, 1, 0)
harden_random_forest['AboveBelow2100'] = np.where(harden_random_forest['PTS'] >= 2100, 1, 0)
harden_random_forest['AboveBelow2200'] = np.where(harden_random_forest['PTS'] >= 2200, 1, 0)

# create below lists
above_list = []
below_list = []
decision_tree_above = []
decision_tree_below = []
undecided_list = []
confidence_list = []
decision_tree_accuracy_list = []
y_pred_list = []
y_test_list = []

# print strings plus player information from averages
print('*****\nAll below prediction accuracies and sum predictions are rounded to 4 decimal places for readability '
      'purposes. \nAll True Positive and True Negative Rates are non-rounded for complete date purposes.\n*****\n')
print('*** Predictions for the Decision Tree classifier are based off of a single label. ***\n'
      '*** Predictions for the Random Forest classifier are based off of a conglomerate of labels for all depths and '
      'subtrees. ***\n')
print('*** Player & Classifier Details ***')
print('James Harden : Decision Tree | Random Forest with Depths of 1-2 and Subtrees of 1-5.')
print("Assuming average games (" + str(player_df_sum_avg.iloc[0, 1]) +
      "), minutes (" + str(player_df_sum_avg.iloc[0, 2]) +
      "), and field goal attempts (" + str(player_df_sum_avg.iloc[0, 3]) + ")\n")

# set points to 1200 and counts to zero
points = 1200
true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0
# input data equals minutes, fga, games from df
input_data = harden_random_forest[['MP', 'FGA', 'G']]
# define x as input data values and y as label values for above below points
X = input_data.values
y = harden_random_forest['AboveBelow1200'].values
# split data 50/50 test train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# define decision tree and fit X_train and y_train to model
decision_tree = tree.DecisionTreeClassifier(criterion='entropy')
decision_tree.fit(X_train, y_train)
# define y_pred as decision tree predict X_test
y_pred = decision_tree.predict(X_test)
# compute accuracy of prediction vs y test and multiply by 100 for model
decision_tree_accuracy = accuracy_score(y_test, y_pred)
decision_tree_accuracy = decision_tree_accuracy * 100
# define avg_instance from player averages
avg_instance = np.asmatrix([player_df_sum_avg.iloc[1, 1], player_df_sum_avg.iloc[1, 2],
                                        player_df_sum_avg.iloc[1, 3]])
# print string and points
print('Decision Tree : ' + str(points) + ' Points')
# define decision tree label to equal prediction for average instance
decision_tree_label = decision_tree.predict(avg_instance)
# if decision tree label less than 1, ie negative label < points total, print strings and accuracy
if decision_tree_label < 1:
    print('James Harden will score below ' + str(points) + ' points next season.')
    print('Accuracy of decision tree = ' + str((float(round(decision_tree_accuracy, 4)))) + '%')
    # append points to decision_tree_below
    decision_tree_below.append(points)
# else decision tree label not less than 1, ie positive label > points total, print strings and accuracy
else:
    print('James Harden will score above ' + str(points) + ' points next season.')
    print('Accuracy of decision tree = ' + str((float(round(decision_tree_accuracy, 4)))) + '%')
    # append points to decision_tree_above
    decision_tree_above.append(points)
# append decision to decision_tree_accuracy list
decision_tree_accuracy_list.append(decision_tree_accuracy)
# print new line
print('\n')

# define subtree and depth, create empty lists
subtrees = 1
depth = 1
label_list = []
depth_list = []
subtrees_list = []
accuracy_list = []
# while loop nested to iterate through depths 1-2 and subtrees 1-5
while depth <= 2:
    while subtrees <= 5:
        # random forest classifier with subtrees and depth from iterations
        random_forest = RandomForestClassifier(n_estimators=subtrees, max_depth=depth, criterion='entropy')
        # fit random_forest X_train, y_train, define pred y to predict X_test
        random_forest.fit(X_train, y_train)
        y_pred = random_forest.predict(X_test)
        # define accuracy and append accuracy to list, append y_pred and y_test to respective lists
        accuracy = (accuracy_score(y_test, y_pred))
        accuracy_list.append(accuracy)
        y_pred_list.append(y_pred)
        y_test_list.append(y_test)

        # define average instance for player averages
        avg_instance = np.asmatrix([player_df_sum_avg.iloc[1, 1], player_df_sum_avg.iloc[1, 2],
                                        player_df_sum_avg.iloc[1, 3]])
        # define label as random_forest predict avg instance
        label = random_forest.predict(avg_instance)
        # append label, subtrees, depth to respective lists
        label_list.append(label)
        subtrees_list.append(subtrees)
        depth_list.append(depth)
        # increment subtrees
        subtrees = subtrees + 1
    # increment depth set subtrees back to 1
    depth = depth + 1
    subtrees = 1

# create player tree dataframe with subtrees, depth and labels from their lists
harden_tree_df = pd.DataFrame(columns=['Subtrees', 'Depth', 'Label'])
harden_tree_df['Subtrees'] = subtrees_list
harden_tree_df['Depth'] = depth_list
harden_tree_df['Label'] = label_list
# define x, y and text, annotate text with label column
x = harden_tree_df['Depth']
y = harden_tree_df['Subtrees']
text = harden_tree_df['Label']

figure(figsize=(15, 15), dpi=80)
plt.scatter(x, y)
for i in range(len(x)):
    plt.annotate(text[i], (x[i], y[i]),)

# plot x and y labels set title, save figure
plt.xlabel('Depth\nAnnotation = Label \nLabel : 1 = Above Total Points | 0 = Below Total Points')
plt.ylabel('Subtrees')
plt.title('James Harden Random Forrest Classifier Scatter Plot ' + str(points) + ' Points')
plt.savefig('Random Forest/James Harden/James Harden Random Forrest Classifier Scatter Plot '
            + str(points) + ' Points.pdf')

# define sum_label as sum of label list
sum_label = sum(label_list)
# print string and points
print('Random Forest : ' + str(points) + ' Points')
# if sum_label > length of label list/2 ie sum of all random forests depths/subtrees
if sum_label > len(label_list) / 2:
    # print string for above plus points
    print('James Harden will score above ' + str(points) + ' points next season.')
    # append points to above list
    above_list.append(points)
# elif sum_label < length of label list/2 ie sum of all random forests depths/subtrees
elif sum_label < len(label_list) / 2:
    # print string for below plus points
    print('James Harden will score below ' + str(points) + ' points next season.')
    # append points to below list
    below_list.append(points)
# else sum_label = length of label list/2 ie it is a split undecided if above or below
else:
    # print string undecided plus points
    print('It is undecided if James Harden will score above or below ' + str(points) + ' points next season.')
    # append to below list as undecided is conservatively not a positive result
    below_list.append(points)
# define sum_accuracy = sum of accuracy list
sum_accuracy = sum(accuracy_list)
# confidence = sum_accuracy / length of accuracy list * 100
confidence = (sum_accuracy / len(accuracy_list)) * 100
# append confidence to confidence list
confidence_list.append(confidence)
# print string plus sum accuracy for all depths and subtrees rounded
print('Sum accuracy of all subtrees and depths for prediction = ' + str(float(round(confidence, 4))) + '%')

# create list of arrays for y_pred and y_test
y_pred_list = [arr.tolist() for arr in y_pred_list]
y_test_list = [arr.tolist() for arr in y_test_list]

# iterate items in sublist of lists to get each element
y_pred_list = [item for sublist in y_pred_list for item in sublist]
y_test_list = [item for sublist in y_test_list for item in sublist]


# set counter to 0, while i < length of y_pred list
i = 0
while i < len(y_pred_list):
    # calculate true positive, true negative, false positive, false negative through counts comparing each list at i
    if y_pred_list[i] == y_test_list[i] and y_pred_list[i] == 1:
        true_positive = true_positive + 1
    elif y_pred_list[i] == y_test_list[i] and y_pred_list[i] == 0:
        true_negative = true_negative + 1
    elif y_pred_list[i] != y_test_list[i] and y_pred_list[i] == 1:
        false_positive = false_positive + 1
    else:
        false_negative = false_negative + 1
    # increment counter for while loop
    i = i + 1

# print string with amount of predictions and tp, fp, tn, fn
print('Of ' + str(len(y_pred_list)) + ' predictions from all subtrees and depths : True Positives = '
      + str(true_positive) + ' | False Positives = ' + str(false_positive) + ' | True Negatives = '
      + str(true_negative) + ' | False Negatives = ' + str(false_negative))
# if elif else, to calculate tnr and tpr, if applicable, if not N/A
if true_negative > 0:
    tnr = true_negative / (true_negative + false_positive)
elif false_positive > 0:
    tnr = 0
else:
    tnr = 'N/A'
if true_positive > 0:
    tpr = true_positive / (true_positive + false_negative)
elif false_negative > 0:
    tpr = 0
else:
    tpr = 'N/A'

# print TPR and TNR string plus results and blank line
print('True Positive Rate = ' + str(tpr))
print('True Negative Rate = ' + str(tnr))
print('\n')
# clear lists
y_pred_list.clear()
y_test_list.clear()
accuracy_list.clear()

# set points to 1300 and counts to zero
points = 1300
true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0
# input data equals minutes, fga, games from df
input_data = harden_random_forest[['MP', 'FGA', 'G']]
# define x as input data values and y as label values for above below points
X = input_data.values
y = harden_random_forest['AboveBelow1300'].values
# split data 50/50 test train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# define decision tree and fit X_train and y_train to model
decision_tree = tree.DecisionTreeClassifier(criterion='entropy')
decision_tree.fit(X_train, y_train)
# define y_pred as decision tree predict X_test
y_pred = decision_tree.predict(X_test)
# compute accuracy of prediction vs y test and multiply by 100 for model
decision_tree_accuracy = accuracy_score(y_test, y_pred)
decision_tree_accuracy = decision_tree_accuracy * 100
# define avg_instance from player averages
avg_instance = np.asmatrix([player_df_sum_avg.iloc[1, 1], player_df_sum_avg.iloc[1, 2],
                                        player_df_sum_avg.iloc[1, 3]])
# print string and points
print('Decision Tree : ' + str(points) + ' Points')
# define decision tree label to equal prediction for average instance
decision_tree_label = decision_tree.predict(avg_instance)
# if decision tree label less than 1, ie negative label < points total, print strings and accuracy
if decision_tree_label < 1:
    print('James Harden will score below ' + str(points) + ' points next season.')
    print('Accuracy of decision tree = ' + str((float(round(decision_tree_accuracy, 4)))) + '%')
    # append points to decision_tree_below
    decision_tree_below.append(points)
# else decision tree label not less than 1, ie positive label > points total, print strings and accuracy
else:
    print('James Harden will score above ' + str(points) + ' points next season.')
    print('Accuracy of decision tree = ' + str((float(round(decision_tree_accuracy, 4)))) + '%')
    # append points to decision_tree_above
    decision_tree_above.append(points)
# append decision to decision_tree_accuracy list
decision_tree_accuracy_list.append(decision_tree_accuracy)
# print new line
print('\n')

# define subtree and depth, create empty lists
subtrees = 1
depth = 1
label_list = []
depth_list = []
subtrees_list = []
accuracy_list = []
# while loop nested to iterate through depths 1-2 and subtrees 1-5
while depth <= 2:
    while subtrees <= 5:
        # random forest classifier with subtrees and depth from iterations
        random_forest = RandomForestClassifier(n_estimators=subtrees, max_depth=depth, criterion='entropy')
        # fit random_forest X_train, y_train, define pred y to predict X_test
        random_forest.fit(X_train, y_train)
        y_pred = random_forest.predict(X_test)
        # define accuracy and append accuracy to list, append y_pred and y_test to respective lists
        accuracy = (accuracy_score(y_test, y_pred))
        accuracy_list.append(accuracy)
        y_pred_list.append(y_pred)
        y_test_list.append(y_test)

        # define average instance for player averages
        avg_instance = np.asmatrix([player_df_sum_avg.iloc[1, 1], player_df_sum_avg.iloc[1, 2],
                                        player_df_sum_avg.iloc[1, 3]])
        # define label as random_forest predict avg instance
        label = random_forest.predict(avg_instance)
        # append label, subtrees, depth to respective lists
        label_list.append(label)
        subtrees_list.append(subtrees)
        depth_list.append(depth)
        # increment subtrees
        subtrees = subtrees + 1
    # increment depth set subtrees back to 1
    depth = depth + 1
    subtrees = 1

# create player tree dataframe with subtrees, depth and labels from their lists
harden_tree_df = pd.DataFrame(columns=['Subtrees', 'Depth', 'Label'])
harden_tree_df['Subtrees'] = subtrees_list
harden_tree_df['Depth'] = depth_list
harden_tree_df['Label'] = label_list
# define x, y and text, annotate text with label column
x = harden_tree_df['Depth']
y = harden_tree_df['Subtrees']
text = harden_tree_df['Label']

figure(figsize=(15, 15), dpi=80)
plt.scatter(x, y)
for i in range(len(x)):
    plt.annotate(text[i], (x[i], y[i]),)

# plot x and y labels set title, save figure
plt.xlabel('Depth\nAnnotation = Label \nLabel : 1 = Above Total Points | 0 = Below Total Points')
plt.ylabel('Subtrees')
plt.title('James Harden Random Forrest Classifier Scatter Plot ' + str(points) + ' Points')
plt.savefig('Random Forest/James Harden/James Harden Random Forrest Classifier Scatter Plot '
            + str(points) + ' Points.pdf')

# define sum_label as sum of label list
sum_label = sum(label_list)
# print string and points
print('Random Forest : ' + str(points) + ' Points')
# if sum_label > length of label list/2 ie sum of all random forests depths/subtrees
if sum_label > len(label_list) / 2:
    # print string for above plus points
    print('James Harden will score above ' + str(points) + ' points next season.')
    # append points to above list
    above_list.append(points)
# elif sum_label < length of label list/2 ie sum of all random forests depths/subtrees
elif sum_label < len(label_list) / 2:
    # print string for below plus points
    print('James Harden will score below ' + str(points) + ' points next season.')
    # append points to below list
    below_list.append(points)
# else sum_label = length of label list/2 ie it is a split undecided if above or below
else:
    # print string undecided plus points
    print('It is undecided if James Harden will score above or below ' + str(points) + ' points next season.')
    # append to below list as undecided is conservatively not a positive result
    below_list.append(points)
# define sum_accuracy = sum of accuracy list
sum_accuracy = sum(accuracy_list)
# confidence = sum_accuracy / length of accuracy list * 100
confidence = (sum_accuracy / len(accuracy_list)) * 100
# append confidence to confidence list
confidence_list.append(confidence)
# print string plus sum accuracy for all depths and subtrees rounded
print('Sum accuracy of all subtrees and depths for prediction = ' + str(float(round(confidence, 4))) + '%')

# create list of arrays for y_pred and y_test
y_pred_list = [arr.tolist() for arr in y_pred_list]
y_test_list = [arr.tolist() for arr in y_test_list]

# iterate items in sublist of lists to get each element
y_pred_list = [item for sublist in y_pred_list for item in sublist]
y_test_list = [item for sublist in y_test_list for item in sublist]


# set counter to 0, while i < length of y_pred list
i = 0
while i < len(y_pred_list):
    # calculate true positive, true negative, false positive, false negative through counts comparing each list at i
    if y_pred_list[i] == y_test_list[i] and y_pred_list[i] == 1:
        true_positive = true_positive + 1
    elif y_pred_list[i] == y_test_list[i] and y_pred_list[i] == 0:
        true_negative = true_negative + 1
    elif y_pred_list[i] != y_test_list[i] and y_pred_list[i] == 1:
        false_positive = false_positive + 1
    else:
        false_negative = false_negative + 1
    # increment counter for while loop
    i = i + 1

# print string with amount of predictions and tp, fp, tn, fn
print('Of ' + str(len(y_pred_list)) + ' predictions from all subtrees and depths : True Positives = '
      + str(true_positive) + ' | False Positives = ' + str(false_positive) + ' | True Negatives = '
      + str(true_negative) + ' | False Negatives = ' + str(false_negative))
# if elif else, to calculate tnr and tpr, if applicable, if not N/A
if true_negative > 0:
    tnr = true_negative / (true_negative + false_positive)
elif false_positive > 0:
    tnr = 0
else:
    tnr = 'N/A'
if true_positive > 0:
    tpr = true_positive / (true_positive + false_negative)
elif false_negative > 0:
    tpr = 0
else:
    tpr = 'N/A'

# print TPR and TNR string plus results and blank line
print('True Positive Rate = ' + str(tpr))
print('True Negative Rate = ' + str(tnr))
print('\n')
# clear lists
y_pred_list.clear()
y_test_list.clear()
accuracy_list.clear()

# set points to 1400 and counts to zero
points = 1400
true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0
# input data equals minutes, fga, games from df
input_data = harden_random_forest[['MP', 'FGA', 'G']]
# define x as input data values and y as label values for above below points
X = input_data.values
y = harden_random_forest['AboveBelow1400'].values
# split data 50/50 test train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# define decision tree and fit X_train and y_train to model
decision_tree = tree.DecisionTreeClassifier(criterion='entropy')
decision_tree.fit(X_train, y_train)
# define y_pred as decision tree predict X_test
y_pred = decision_tree.predict(X_test)
# compute accuracy of prediction vs y test and multiply by 100 for model
decision_tree_accuracy = accuracy_score(y_test, y_pred)
decision_tree_accuracy = decision_tree_accuracy * 100
# define avg_instance from player averages
avg_instance = np.asmatrix([player_df_sum_avg.iloc[1, 1], player_df_sum_avg.iloc[1, 2],
                                        player_df_sum_avg.iloc[1, 3]])
# print string and points
print('Decision Tree : ' + str(points) + ' Points')
# define decision tree label to equal prediction for average instance
decision_tree_label = decision_tree.predict(avg_instance)
# if decision tree label less than 1, ie negative label < points total, print strings and accuracy
if decision_tree_label < 1:
    print('James Harden will score below ' + str(points) + ' points next season.')
    print('Accuracy of decision tree = ' + str((float(round(decision_tree_accuracy, 4)))) + '%')
    # append points to decision_tree_below
    decision_tree_below.append(points)
# else decision tree label not less than 1, ie positive label > points total, print strings and accuracy
else:
    print('James Harden will score above ' + str(points) + ' points next season.')
    print('Accuracy of decision tree = ' + str((float(round(decision_tree_accuracy, 4)))) + '%')
    # append points to decision_tree_above
    decision_tree_above.append(points)
# append decision to decision_tree_accuracy list
decision_tree_accuracy_list.append(decision_tree_accuracy)
# print new line
print('\n')

# define subtree and depth, create empty lists
subtrees = 1
depth = 1
label_list = []
depth_list = []
subtrees_list = []
accuracy_list = []
# while loop nested to iterate through depths 1-2 and subtrees 1-5
while depth <= 2:
    while subtrees <= 5:
        # random forest classifier with subtrees and depth from iterations
        random_forest = RandomForestClassifier(n_estimators=subtrees, max_depth=depth, criterion='entropy')
        # fit random_forest X_train, y_train, define pred y to predict X_test
        random_forest.fit(X_train, y_train)
        y_pred = random_forest.predict(X_test)
        # define accuracy and append accuracy to list, append y_pred and y_test to respective lists
        accuracy = (accuracy_score(y_test, y_pred))
        accuracy_list.append(accuracy)
        y_pred_list.append(y_pred)
        y_test_list.append(y_test)

        # define average instance for player averages
        avg_instance = np.asmatrix([player_df_sum_avg.iloc[1, 1], player_df_sum_avg.iloc[1, 2],
                                        player_df_sum_avg.iloc[1, 3]])
        # define label as random_forest predict avg instance
        label = random_forest.predict(avg_instance)
        # append label, subtrees, depth to respective lists
        label_list.append(label)
        subtrees_list.append(subtrees)
        depth_list.append(depth)
        # increment subtrees
        subtrees = subtrees + 1
    # increment depth set subtrees back to 1
    depth = depth + 1
    subtrees = 1

# create player tree dataframe with subtrees, depth and labels from their lists
harden_tree_df = pd.DataFrame(columns=['Subtrees', 'Depth', 'Label'])
harden_tree_df['Subtrees'] = subtrees_list
harden_tree_df['Depth'] = depth_list
harden_tree_df['Label'] = label_list
# define x, y and text, annotate text with label column
x = harden_tree_df['Depth']
y = harden_tree_df['Subtrees']
text = harden_tree_df['Label']

figure(figsize=(15, 15), dpi=80)
plt.scatter(x, y)
for i in range(len(x)):
    plt.annotate(text[i], (x[i], y[i]),)

# plot x and y labels set title, save figure
plt.xlabel('Depth\nAnnotation = Label \nLabel : 1 = Above Total Points | 0 = Below Total Points')
plt.ylabel('Subtrees')
plt.title('James Harden Random Forrest Classifier Scatter Plot ' + str(points) + ' Points')
plt.savefig('Random Forest/James Harden/James Harden Random Forrest Classifier Scatter Plot '
            + str(points) + ' Points.pdf')

# define sum_label as sum of label list
sum_label = sum(label_list)
# print string and points
print('Random Forest : ' + str(points) + ' Points')
# if sum_label > length of label list/2 ie sum of all random forests depths/subtrees
if sum_label > len(label_list) / 2:
    # print string for above plus points
    print('James Harden will score above ' + str(points) + ' points next season.')
    # append points to above list
    above_list.append(points)
# elif sum_label < length of label list/2 ie sum of all random forests depths/subtrees
elif sum_label < len(label_list) / 2:
    # print string for below plus points
    print('James Harden will score below ' + str(points) + ' points next season.')
    # append points to below list
    below_list.append(points)
# else sum_label = length of label list/2 ie it is a split undecided if above or below
else:
    # print string undecided plus points
    print('It is undecided if James Harden will score above or below ' + str(points) + ' points next season.')
    # append to below list as undecided is conservatively not a positive result
    below_list.append(points)
# define sum_accuracy = sum of accuracy list
sum_accuracy = sum(accuracy_list)
# confidence = sum_accuracy / length of accuracy list * 100
confidence = (sum_accuracy / len(accuracy_list)) * 100
# append confidence to confidence list
confidence_list.append(confidence)
# print string plus sum accuracy for all depths and subtrees rounded
print('Sum accuracy of all subtrees and depths for prediction = ' + str(float(round(confidence, 4))) + '%')

# create list of arrays for y_pred and y_test
y_pred_list = [arr.tolist() for arr in y_pred_list]
y_test_list = [arr.tolist() for arr in y_test_list]

# iterate items in sublist of lists to get each element
y_pred_list = [item for sublist in y_pred_list for item in sublist]
y_test_list = [item for sublist in y_test_list for item in sublist]


# set counter to 0, while i < length of y_pred list
i = 0
while i < len(y_pred_list):
    # calculate true positive, true negative, false positive, false negative through counts comparing each list at i
    if y_pred_list[i] == y_test_list[i] and y_pred_list[i] == 1:
        true_positive = true_positive + 1
    elif y_pred_list[i] == y_test_list[i] and y_pred_list[i] == 0:
        true_negative = true_negative + 1
    elif y_pred_list[i] != y_test_list[i] and y_pred_list[i] == 1:
        false_positive = false_positive + 1
    else:
        false_negative = false_negative + 1
    # increment counter for while loop
    i = i + 1

# print string with amount of predictions and tp, fp, tn, fn
print('Of ' + str(len(y_pred_list)) + ' predictions from all subtrees and depths : True Positives = '
      + str(true_positive) + ' | False Positives = ' + str(false_positive) + ' | True Negatives = '
      + str(true_negative) + ' | False Negatives = ' + str(false_negative))
# if elif else, to calculate tnr and tpr, if applicable, if not N/A
if true_negative > 0:
    tnr = true_negative / (true_negative + false_positive)
elif false_positive > 0:
    tnr = 0
else:
    tnr = 'N/A'
if true_positive > 0:
    tpr = true_positive / (true_positive + false_negative)
elif false_negative > 0:
    tpr = 0
else:
    tpr = 'N/A'

# print TPR and TNR string plus results and blank line
print('True Positive Rate = ' + str(tpr))
print('True Negative Rate = ' + str(tnr))
print('\n')
# clear lists
y_pred_list.clear()
y_test_list.clear()
accuracy_list.clear()

# set points to 1500 and counts to zero
points = 1500
true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0
# input data equals minutes, fga, games from df
input_data = harden_random_forest[['MP', 'FGA', 'G']]
# define x as input data values and y as label values for above below points
X = input_data.values
y = harden_random_forest['AboveBelow1500'].values
# split data 50/50 test train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# define decision tree and fit X_train and y_train to model
decision_tree = tree.DecisionTreeClassifier(criterion='entropy')
decision_tree.fit(X_train, y_train)
# define y_pred as decision tree predict X_test
y_pred = decision_tree.predict(X_test)
# compute accuracy of prediction vs y test and multiply by 100 for model
decision_tree_accuracy = accuracy_score(y_test, y_pred)
decision_tree_accuracy = decision_tree_accuracy * 100
# define avg_instance from player averages
avg_instance = np.asmatrix([player_df_sum_avg.iloc[1, 1], player_df_sum_avg.iloc[1, 2],
                                        player_df_sum_avg.iloc[1, 3]])
# print string and points
print('Decision Tree : ' + str(points) + ' Points')
# define decision tree label to equal prediction for average instance
decision_tree_label = decision_tree.predict(avg_instance)
# if decision tree label less than 1, ie negative label < points total, print strings and accuracy
if decision_tree_label < 1:
    print('James Harden will score below ' + str(points) + ' points next season.')
    print('Accuracy of decision tree = ' + str((float(round(decision_tree_accuracy, 4)))) + '%')
    # append points to decision_tree_below
    decision_tree_below.append(points)
# else decision tree label not less than 1, ie positive label > points total, print strings and accuracy
else:
    print('James Harden will score above ' + str(points) + ' points next season.')
    print('Accuracy of decision tree = ' + str((float(round(decision_tree_accuracy, 4)))) + '%')
    # append points to decision_tree_above
    decision_tree_above.append(points)
# append decision to decision_tree_accuracy list
decision_tree_accuracy_list.append(decision_tree_accuracy)
# print new line
print('\n')

# define subtree and depth, create empty lists
subtrees = 1
depth = 1
label_list = []
depth_list = []
subtrees_list = []
accuracy_list = []
# while loop nested to iterate through depths 1-2 and subtrees 1-5
while depth <= 2:
    while subtrees <= 5:
        # random forest classifier with subtrees and depth from iterations
        random_forest = RandomForestClassifier(n_estimators=subtrees, max_depth=depth, criterion='entropy')
        # fit random_forest X_train, y_train, define pred y to predict X_test
        random_forest.fit(X_train, y_train)
        y_pred = random_forest.predict(X_test)
        # define accuracy and append accuracy to list, append y_pred and y_test to respective lists
        accuracy = (accuracy_score(y_test, y_pred))
        accuracy_list.append(accuracy)
        y_pred_list.append(y_pred)
        y_test_list.append(y_test)

        # define average instance for player averages
        avg_instance = np.asmatrix([player_df_sum_avg.iloc[1, 1], player_df_sum_avg.iloc[1, 2],
                                        player_df_sum_avg.iloc[1, 3]])
        # define label as random_forest predict avg instance
        label = random_forest.predict(avg_instance)
        # append label, subtrees, depth to respective lists
        label_list.append(label)
        subtrees_list.append(subtrees)
        depth_list.append(depth)
        # increment subtrees
        subtrees = subtrees + 1
    # increment depth set subtrees back to 1
    depth = depth + 1
    subtrees = 1

# create player tree dataframe with subtrees, depth and labels from their lists
harden_tree_df = pd.DataFrame(columns=['Subtrees', 'Depth', 'Label'])
harden_tree_df['Subtrees'] = subtrees_list
harden_tree_df['Depth'] = depth_list
harden_tree_df['Label'] = label_list
# define x, y and text, annotate text with label column
x = harden_tree_df['Depth']
y = harden_tree_df['Subtrees']
text = harden_tree_df['Label']

figure(figsize=(15, 15), dpi=80)
plt.scatter(x, y)
for i in range(len(x)):
    plt.annotate(text[i], (x[i], y[i]),)

# plot x and y labels set title, save figure
plt.xlabel('Depth\nAnnotation = Label \nLabel : 1 = Above Total Points | 0 = Below Total Points')
plt.ylabel('Subtrees')
plt.title('James Harden Random Forrest Classifier Scatter Plot ' + str(points) + ' Points')
plt.savefig('Random Forest/James Harden/James Harden Random Forrest Classifier Scatter Plot '
            + str(points) + ' Points.pdf')

# define sum_label as sum of label list
sum_label = sum(label_list)
# print string and points
print('Random Forest : ' + str(points) + ' Points')
# if sum_label > length of label list/2 ie sum of all random forests depths/subtrees
if sum_label > len(label_list) / 2:
    # print string for above plus points
    print('James Harden will score above ' + str(points) + ' points next season.')
    # append points to above list
    above_list.append(points)
# elif sum_label < length of label list/2 ie sum of all random forests depths/subtrees
elif sum_label < len(label_list) / 2:
    # print string for below plus points
    print('James Harden will score below ' + str(points) + ' points next season.')
    # append points to below list
    below_list.append(points)
# else sum_label = length of label list/2 ie it is a split undecided if above or below
else:
    # print string undecided plus points
    print('It is undecided if James Harden will score above or below ' + str(points) + ' points next season.')
    # append to below list as undecided is conservatively not a positive result
    below_list.append(points)
# define sum_accuracy = sum of accuracy list
sum_accuracy = sum(accuracy_list)
# confidence = sum_accuracy / length of accuracy list * 100
confidence = (sum_accuracy / len(accuracy_list)) * 100
# append confidence to confidence list
confidence_list.append(confidence)
# print string plus sum accuracy for all depths and subtrees rounded
print('Sum accuracy of all subtrees and depths for prediction = ' + str(float(round(confidence, 4))) + '%')

# create list of arrays for y_pred and y_test
y_pred_list = [arr.tolist() for arr in y_pred_list]
y_test_list = [arr.tolist() for arr in y_test_list]

# iterate items in sublist of lists to get each element
y_pred_list = [item for sublist in y_pred_list for item in sublist]
y_test_list = [item for sublist in y_test_list for item in sublist]


# set counter to 0, while i < length of y_pred list
i = 0
while i < len(y_pred_list):
    # calculate true positive, true negative, false positive, false negative through counts comparing each list at i
    if y_pred_list[i] == y_test_list[i] and y_pred_list[i] == 1:
        true_positive = true_positive + 1
    elif y_pred_list[i] == y_test_list[i] and y_pred_list[i] == 0:
        true_negative = true_negative + 1
    elif y_pred_list[i] != y_test_list[i] and y_pred_list[i] == 1:
        false_positive = false_positive + 1
    else:
        false_negative = false_negative + 1
    # increment counter for while loop
    i = i + 1

# print string with amount of predictions and tp, fp, tn, fn
print('Of ' + str(len(y_pred_list)) + ' predictions from all subtrees and depths : True Positives = '
      + str(true_positive) + ' | False Positives = ' + str(false_positive) + ' | True Negatives = '
      + str(true_negative) + ' | False Negatives = ' + str(false_negative))
# if elif else, to calculate tnr and tpr, if applicable, if not N/A
if true_negative > 0:
    tnr = true_negative / (true_negative + false_positive)
elif false_positive > 0:
    tnr = 0
else:
    tnr = 'N/A'
if true_positive > 0:
    tpr = true_positive / (true_positive + false_negative)
elif false_negative > 0:
    tpr = 0
else:
    tpr = 'N/A'

# print TPR and TNR string plus results and blank line
print('True Positive Rate = ' + str(tpr))
print('True Negative Rate = ' + str(tnr))
print('\n')
# clear lists
y_pred_list.clear()
y_test_list.clear()
accuracy_list.clear()

# set points to 1600 and counts to zero
points = 1600
true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0
# input data equals minutes, fga, games from df
input_data = harden_random_forest[['MP', 'FGA', 'G']]
# define x as input data values and y as label values for above below points
X = input_data.values
y = harden_random_forest['AboveBelow1600'].values
# split data 50/50 test train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# define decision tree and fit X_train and y_train to model
decision_tree = tree.DecisionTreeClassifier(criterion='entropy')
decision_tree.fit(X_train, y_train)
# define y_pred as decision tree predict X_test
y_pred = decision_tree.predict(X_test)
# compute accuracy of prediction vs y test and multiply by 100 for model
decision_tree_accuracy = accuracy_score(y_test, y_pred)
decision_tree_accuracy = decision_tree_accuracy * 100
# define avg_instance from player averages
avg_instance = np.asmatrix([player_df_sum_avg.iloc[1, 1], player_df_sum_avg.iloc[1, 2],
                                        player_df_sum_avg.iloc[1, 3]])
# print string and points
print('Decision Tree : ' + str(points) + ' Points')
# define decision tree label to equal prediction for average instance
decision_tree_label = decision_tree.predict(avg_instance)
# if decision tree label less than 1, ie negative label < points total, print strings and accuracy
if decision_tree_label < 1:
    print('James Harden will score below ' + str(points) + ' points next season.')
    print('Accuracy of decision tree = ' + str((float(round(decision_tree_accuracy, 4)))) + '%')
    # append points to decision_tree_below
    decision_tree_below.append(points)
# else decision tree label not less than 1, ie positive label > points total, print strings and accuracy
else:
    print('James Harden will score above ' + str(points) + ' points next season.')
    print('Accuracy of decision tree = ' + str((float(round(decision_tree_accuracy, 4)))) + '%')
    # append points to decision_tree_above
    decision_tree_above.append(points)
# append decision to decision_tree_accuracy list
decision_tree_accuracy_list.append(decision_tree_accuracy)
# print new line
print('\n')

# define subtree and depth, create empty lists
subtrees = 1
depth = 1
label_list = []
depth_list = []
subtrees_list = []
accuracy_list = []
# while loop nested to iterate through depths 1-2 and subtrees 1-5
while depth <= 2:
    while subtrees <= 5:
        # random forest classifier with subtrees and depth from iterations
        random_forest = RandomForestClassifier(n_estimators=subtrees, max_depth=depth, criterion='entropy')
        # fit random_forest X_train, y_train, define pred y to predict X_test
        random_forest.fit(X_train, y_train)
        y_pred = random_forest.predict(X_test)
        # define accuracy and append accuracy to list, append y_pred and y_test to respective lists
        accuracy = (accuracy_score(y_test, y_pred))
        accuracy_list.append(accuracy)
        y_pred_list.append(y_pred)
        y_test_list.append(y_test)

        # define average instance for player averages
        avg_instance = np.asmatrix([player_df_sum_avg.iloc[1, 1], player_df_sum_avg.iloc[1, 2],
                                        player_df_sum_avg.iloc[1, 3]])
        # define label as random_forest predict avg instance
        label = random_forest.predict(avg_instance)
        # append label, subtrees, depth to respective lists
        label_list.append(label)
        subtrees_list.append(subtrees)
        depth_list.append(depth)
        # increment subtrees
        subtrees = subtrees + 1
    # increment depth set subtrees back to 1
    depth = depth + 1
    subtrees = 1

# create player tree dataframe with subtrees, depth and labels from their lists
harden_tree_df = pd.DataFrame(columns=['Subtrees', 'Depth', 'Label'])
harden_tree_df['Subtrees'] = subtrees_list
harden_tree_df['Depth'] = depth_list
harden_tree_df['Label'] = label_list
# define x, y and text, annotate text with label column
x = harden_tree_df['Depth']
y = harden_tree_df['Subtrees']
text = harden_tree_df['Label']

figure(figsize=(15, 15), dpi=80)
plt.scatter(x, y)
for i in range(len(x)):
    plt.annotate(text[i], (x[i], y[i]),)

# plot x and y labels set title, save figure
plt.xlabel('Depth\nAnnotation = Label \nLabel : 1 = Above Total Points | 0 = Below Total Points')
plt.ylabel('Subtrees')
plt.title('James Harden Random Forrest Classifier Scatter Plot ' + str(points) + ' Points')
plt.savefig('Random Forest/James Harden/James Harden Random Forrest Classifier Scatter Plot '
            + str(points) + ' Points.pdf')

# define sum_label as sum of label list
sum_label = sum(label_list)
# print string and points
print('Random Forest : ' + str(points) + ' Points')
# if sum_label > length of label list/2 ie sum of all random forests depths/subtrees
if sum_label > len(label_list) / 2:
    # print string for above plus points
    print('James Harden will score above ' + str(points) + ' points next season.')
    # append points to above list
    above_list.append(points)
# elif sum_label < length of label list/2 ie sum of all random forests depths/subtrees
elif sum_label < len(label_list) / 2:
    # print string for below plus points
    print('James Harden will score below ' + str(points) + ' points next season.')
    # append points to below list
    below_list.append(points)
# else sum_label = length of label list/2 ie it is a split undecided if above or below
else:
    # print string undecided plus points
    print('It is undecided if James Harden will score above or below ' + str(points) + ' points next season.')
    # append to below list as undecided is conservatively not a positive result
    below_list.append(points)
# define sum_accuracy = sum of accuracy list
sum_accuracy = sum(accuracy_list)
# confidence = sum_accuracy / length of accuracy list * 100
confidence = (sum_accuracy / len(accuracy_list)) * 100
# append confidence to confidence list
confidence_list.append(confidence)
# print string plus sum accuracy for all depths and subtrees rounded
print('Sum accuracy of all subtrees and depths for prediction = ' + str(float(round(confidence, 4))) + '%')

# create list of arrays for y_pred and y_test
y_pred_list = [arr.tolist() for arr in y_pred_list]
y_test_list = [arr.tolist() for arr in y_test_list]

# iterate items in sublist of lists to get each element
y_pred_list = [item for sublist in y_pred_list for item in sublist]
y_test_list = [item for sublist in y_test_list for item in sublist]


# set counter to 0, while i < length of y_pred list
i = 0
while i < len(y_pred_list):
    # calculate true positive, true negative, false positive, false negative through counts comparing each list at i
    if y_pred_list[i] == y_test_list[i] and y_pred_list[i] == 1:
        true_positive = true_positive + 1
    elif y_pred_list[i] == y_test_list[i] and y_pred_list[i] == 0:
        true_negative = true_negative + 1
    elif y_pred_list[i] != y_test_list[i] and y_pred_list[i] == 1:
        false_positive = false_positive + 1
    else:
        false_negative = false_negative + 1
    # increment counter for while loop
    i = i + 1

# print string with amount of predictions and tp, fp, tn, fn
print('Of ' + str(len(y_pred_list)) + ' predictions from all subtrees and depths : True Positives = '
      + str(true_positive) + ' | False Positives = ' + str(false_positive) + ' | True Negatives = '
      + str(true_negative) + ' | False Negatives = ' + str(false_negative))
# if elif else, to calculate tnr and tpr, if applicable, if not N/A
if true_negative > 0:
    tnr = true_negative / (true_negative + false_positive)
elif false_positive > 0:
    tnr = 0
else:
    tnr = 'N/A'
if true_positive > 0:
    tpr = true_positive / (true_positive + false_negative)
elif false_negative > 0:
    tpr = 0
else:
    tpr = 'N/A'

# print TPR and TNR string plus results and blank line
print('True Positive Rate = ' + str(tpr))
print('True Negative Rate = ' + str(tnr))
print('\n')
# clear lists
y_pred_list.clear()
y_test_list.clear()
accuracy_list.clear()

# set points to 1700 and counts to zero
points = 1700
true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0
# input data equals minutes, fga, games from df
input_data = harden_random_forest[['MP', 'FGA', 'G']]
# define x as input data values and y as label values for above below points
X = input_data.values
y = harden_random_forest['AboveBelow1700'].values
# split data 50/50 test train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# define decision tree and fit X_train and y_train to model
decision_tree = tree.DecisionTreeClassifier(criterion='entropy')
decision_tree.fit(X_train, y_train)
# define y_pred as decision tree predict X_test
y_pred = decision_tree.predict(X_test)
# compute accuracy of prediction vs y test and multiply by 100 for model
decision_tree_accuracy = accuracy_score(y_test, y_pred)
decision_tree_accuracy = decision_tree_accuracy * 100
# define avg_instance from player averages
avg_instance = np.asmatrix([player_df_sum_avg.iloc[1, 1], player_df_sum_avg.iloc[1, 2],
                                        player_df_sum_avg.iloc[1, 3]])
# print string and points
print('Decision Tree : ' + str(points) + ' Points')
# define decision tree label to equal prediction for average instance
decision_tree_label = decision_tree.predict(avg_instance)
# if decision tree label less than 1, ie negative label < points total, print strings and accuracy
if decision_tree_label < 1:
    print('James Harden will score below ' + str(points) + ' points next season.')
    print('Accuracy of decision tree = ' + str((float(round(decision_tree_accuracy, 4)))) + '%')
    # append points to decision_tree_below
    decision_tree_below.append(points)
# else decision tree label not less than 1, ie positive label > points total, print strings and accuracy
else:
    print('James Harden will score above ' + str(points) + ' points next season.')
    print('Accuracy of decision tree = ' + str((float(round(decision_tree_accuracy, 4)))) + '%')
    # append points to decision_tree_above
    decision_tree_above.append(points)
# append decision to decision_tree_accuracy list
decision_tree_accuracy_list.append(decision_tree_accuracy)
# print new line
print('\n')

# define subtree and depth, create empty lists
subtrees = 1
depth = 1
label_list = []
depth_list = []
subtrees_list = []
accuracy_list = []
# while loop nested to iterate through depths 1-2 and subtrees 1-5
while depth <= 2:
    while subtrees <= 5:
        # random forest classifier with subtrees and depth from iterations
        random_forest = RandomForestClassifier(n_estimators=subtrees, max_depth=depth, criterion='entropy')
        # fit random_forest X_train, y_train, define pred y to predict X_test
        random_forest.fit(X_train, y_train)
        y_pred = random_forest.predict(X_test)
        # define accuracy and append accuracy to list, append y_pred and y_test to respective lists
        accuracy = (accuracy_score(y_test, y_pred))
        accuracy_list.append(accuracy)
        y_pred_list.append(y_pred)
        y_test_list.append(y_test)

        # define average instance for player averages
        avg_instance = np.asmatrix([player_df_sum_avg.iloc[1, 1], player_df_sum_avg.iloc[1, 2],
                                        player_df_sum_avg.iloc[1, 3]])
        # define label as random_forest predict avg instance
        label = random_forest.predict(avg_instance)
        # append label, subtrees, depth to respective lists
        label_list.append(label)
        subtrees_list.append(subtrees)
        depth_list.append(depth)
        # increment subtrees
        subtrees = subtrees + 1
    # increment depth set subtrees back to 1
    depth = depth + 1
    subtrees = 1

# create player tree dataframe with subtrees, depth and labels from their lists
harden_tree_df = pd.DataFrame(columns=['Subtrees', 'Depth', 'Label'])
harden_tree_df['Subtrees'] = subtrees_list
harden_tree_df['Depth'] = depth_list
harden_tree_df['Label'] = label_list
# define x, y and text, annotate text with label column
x = harden_tree_df['Depth']
y = harden_tree_df['Subtrees']
text = harden_tree_df['Label']

figure(figsize=(15, 15), dpi=80)
plt.scatter(x, y)
for i in range(len(x)):
    plt.annotate(text[i], (x[i], y[i]),)

# plot x and y labels set title, save figure
plt.xlabel('Depth\nAnnotation = Label \nLabel : 1 = Above Total Points | 0 = Below Total Points')
plt.ylabel('Subtrees')
plt.title('James Harden Random Forrest Classifier Scatter Plot ' + str(points) + ' Points')
plt.savefig('Random Forest/James Harden/James Harden Random Forrest Classifier Scatter Plot '
            + str(points) + ' Points.pdf')

# define sum_label as sum of label list
sum_label = sum(label_list)
# print string and points
print('Random Forest : ' + str(points) + ' Points')
# if sum_label > length of label list/2 ie sum of all random forests depths/subtrees
if sum_label > len(label_list) / 2:
    # print string for above plus points
    print('James Harden will score above ' + str(points) + ' points next season.')
    # append points to above list
    above_list.append(points)
# elif sum_label < length of label list/2 ie sum of all random forests depths/subtrees
elif sum_label < len(label_list) / 2:
    # print string for below plus points
    print('James Harden will score below ' + str(points) + ' points next season.')
    # append points to below list
    below_list.append(points)
# else sum_label = length of label list/2 ie it is a split undecided if above or below
else:
    # print string undecided plus points
    print('It is undecided if James Harden will score above or below ' + str(points) + ' points next season.')
    # append to below list as undecided is conservatively not a positive result
    below_list.append(points)
# define sum_accuracy = sum of accuracy list
sum_accuracy = sum(accuracy_list)
# confidence = sum_accuracy / length of accuracy list * 100
confidence = (sum_accuracy / len(accuracy_list)) * 100
# append confidence to confidence list
confidence_list.append(confidence)
# print string plus sum accuracy for all depths and subtrees rounded
print('Sum accuracy of all subtrees and depths for prediction = ' + str(float(round(confidence, 4))) + '%')

# create list of arrays for y_pred and y_test
y_pred_list = [arr.tolist() for arr in y_pred_list]
y_test_list = [arr.tolist() for arr in y_test_list]

# iterate items in sublist of lists to get each element
y_pred_list = [item for sublist in y_pred_list for item in sublist]
y_test_list = [item for sublist in y_test_list for item in sublist]


# set counter to 0, while i < length of y_pred list
i = 0
while i < len(y_pred_list):
    # calculate true positive, true negative, false positive, false negative through counts comparing each list at i
    if y_pred_list[i] == y_test_list[i] and y_pred_list[i] == 1:
        true_positive = true_positive + 1
    elif y_pred_list[i] == y_test_list[i] and y_pred_list[i] == 0:
        true_negative = true_negative + 1
    elif y_pred_list[i] != y_test_list[i] and y_pred_list[i] == 1:
        false_positive = false_positive + 1
    else:
        false_negative = false_negative + 1
    # increment counter for while loop
    i = i + 1

# print string with amount of predictions and tp, fp, tn, fn
print('Of ' + str(len(y_pred_list)) + ' predictions from all subtrees and depths : True Positives = '
      + str(true_positive) + ' | False Positives = ' + str(false_positive) + ' | True Negatives = '
      + str(true_negative) + ' | False Negatives = ' + str(false_negative))
# if elif else, to calculate tnr and tpr, if applicable, if not N/A
if true_negative > 0:
    tnr = true_negative / (true_negative + false_positive)
elif false_positive > 0:
    tnr = 0
else:
    tnr = 'N/A'
if true_positive > 0:
    tpr = true_positive / (true_positive + false_negative)
elif false_negative > 0:
    tpr = 0
else:
    tpr = 'N/A'

# print TPR and TNR string plus results and blank line
print('True Positive Rate = ' + str(tpr))
print('True Negative Rate = ' + str(tnr))
print('\n')
# clear lists
y_pred_list.clear()
y_test_list.clear()
accuracy_list.clear()

# set points to 1800 and counts to zero
points = 1800
true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0
# input data equals minutes, fga, games from df
input_data = harden_random_forest[['MP', 'FGA', 'G']]
# define x as input data values and y as label values for above below points
X = input_data.values
y = harden_random_forest['AboveBelow1800'].values
# split data 50/50 test train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# define decision tree and fit X_train and y_train to model
decision_tree = tree.DecisionTreeClassifier(criterion='entropy')
decision_tree.fit(X_train, y_train)
# define y_pred as decision tree predict X_test
y_pred = decision_tree.predict(X_test)
# compute accuracy of prediction vs y test and multiply by 100 for model
decision_tree_accuracy = accuracy_score(y_test, y_pred)
decision_tree_accuracy = decision_tree_accuracy * 100
# define avg_instance from player averages
avg_instance = np.asmatrix([player_df_sum_avg.iloc[1, 1], player_df_sum_avg.iloc[1, 2],
                                        player_df_sum_avg.iloc[1, 3]])
# print string and points
print('Decision Tree : ' + str(points) + ' Points')
# define decision tree label to equal prediction for average instance
decision_tree_label = decision_tree.predict(avg_instance)
# if decision tree label less than 1, ie negative label < points total, print strings and accuracy
if decision_tree_label < 1:
    print('James Harden will score below ' + str(points) + ' points next season.')
    print('Accuracy of decision tree = ' + str((float(round(decision_tree_accuracy, 4)))) + '%')
    # append points to decision_tree_below
    decision_tree_below.append(points)
# else decision tree label not less than 1, ie positive label > points total, print strings and accuracy
else:
    print('James Harden will score above ' + str(points) + ' points next season.')
    print('Accuracy of decision tree = ' + str((float(round(decision_tree_accuracy, 4)))) + '%')
    # append points to decision_tree_above
    decision_tree_above.append(points)
# append decision to decision_tree_accuracy list
decision_tree_accuracy_list.append(decision_tree_accuracy)
# print new line
print('\n')

# define subtree and depth, create empty lists
subtrees = 1
depth = 1
label_list = []
depth_list = []
subtrees_list = []
accuracy_list = []
# while loop nested to iterate through depths 1-2 and subtrees 1-5
while depth <= 2:
    while subtrees <= 5:
        # random forest classifier with subtrees and depth from iterations
        random_forest = RandomForestClassifier(n_estimators=subtrees, max_depth=depth, criterion='entropy')
        # fit random_forest X_train, y_train, define pred y to predict X_test
        random_forest.fit(X_train, y_train)
        y_pred = random_forest.predict(X_test)
        # define accuracy and append accuracy to list, append y_pred and y_test to respective lists
        accuracy = (accuracy_score(y_test, y_pred))
        accuracy_list.append(accuracy)
        y_pred_list.append(y_pred)
        y_test_list.append(y_test)

        # define average instance for player averages
        avg_instance = np.asmatrix([player_df_sum_avg.iloc[1, 1], player_df_sum_avg.iloc[1, 2],
                                        player_df_sum_avg.iloc[1, 3]])
        # define label as random_forest predict avg instance
        label = random_forest.predict(avg_instance)
        # append label, subtrees, depth to respective lists
        label_list.append(label)
        subtrees_list.append(subtrees)
        depth_list.append(depth)
        # increment subtrees
        subtrees = subtrees + 1
    # increment depth set subtrees back to 1
    depth = depth + 1
    subtrees = 1

# create player tree dataframe with subtrees, depth and labels from their lists
harden_tree_df = pd.DataFrame(columns=['Subtrees', 'Depth', 'Label'])
harden_tree_df['Subtrees'] = subtrees_list
harden_tree_df['Depth'] = depth_list
harden_tree_df['Label'] = label_list
# define x, y and text, annotate text with label column
x = harden_tree_df['Depth']
y = harden_tree_df['Subtrees']
text = harden_tree_df['Label']

figure(figsize=(15, 15), dpi=80)
plt.scatter(x, y)
for i in range(len(x)):
    plt.annotate(text[i], (x[i], y[i]),)

# plot x and y labels set title, save figure
plt.xlabel('Depth\nAnnotation = Label \nLabel : 1 = Above Total Points | 0 = Below Total Points')
plt.ylabel('Subtrees')
plt.title('James Harden Random Forrest Classifier Scatter Plot ' + str(points) + ' Points')
plt.savefig('Random Forest/James Harden/James Harden Random Forrest Classifier Scatter Plot '
            + str(points) + ' Points.pdf')

# define sum_label as sum of label list
sum_label = sum(label_list)
# print string and points
print('Random Forest : ' + str(points) + ' Points')
# if sum_label > length of label list/2 ie sum of all random forests depths/subtrees
if sum_label > len(label_list) / 2:
    # print string for above plus points
    print('James Harden will score above ' + str(points) + ' points next season.')
    # append points to above list
    above_list.append(points)
# elif sum_label < length of label list/2 ie sum of all random forests depths/subtrees
elif sum_label < len(label_list) / 2:
    # print string for below plus points
    print('James Harden will score below ' + str(points) + ' points next season.')
    # append points to below list
    below_list.append(points)
# else sum_label = length of label list/2 ie it is a split undecided if above or below
else:
    # print string undecided plus points
    print('It is undecided if James Harden will score above or below ' + str(points) + ' points next season.')
    # append to below list as undecided is conservatively not a positive result
    below_list.append(points)
# define sum_accuracy = sum of accuracy list
sum_accuracy = sum(accuracy_list)
# confidence = sum_accuracy / length of accuracy list * 100
confidence = (sum_accuracy / len(accuracy_list)) * 100
# append confidence to confidence list
confidence_list.append(confidence)
# print string plus sum accuracy for all depths and subtrees rounded
print('Sum accuracy of all subtrees and depths for prediction = ' + str(float(round(confidence, 4))) + '%')

# create list of arrays for y_pred and y_test
y_pred_list = [arr.tolist() for arr in y_pred_list]
y_test_list = [arr.tolist() for arr in y_test_list]

# iterate items in sublist of lists to get each element
y_pred_list = [item for sublist in y_pred_list for item in sublist]
y_test_list = [item for sublist in y_test_list for item in sublist]


# set counter to 0, while i < length of y_pred list
i = 0
while i < len(y_pred_list):
    # calculate true positive, true negative, false positive, false negative through counts comparing each list at i
    if y_pred_list[i] == y_test_list[i] and y_pred_list[i] == 1:
        true_positive = true_positive + 1
    elif y_pred_list[i] == y_test_list[i] and y_pred_list[i] == 0:
        true_negative = true_negative + 1
    elif y_pred_list[i] != y_test_list[i] and y_pred_list[i] == 1:
        false_positive = false_positive + 1
    else:
        false_negative = false_negative + 1
    # increment counter for while loop
    i = i + 1

# print string with amount of predictions and tp, fp, tn, fn
print('Of ' + str(len(y_pred_list)) + ' predictions from all subtrees and depths : True Positives = '
      + str(true_positive) + ' | False Positives = ' + str(false_positive) + ' | True Negatives = '
      + str(true_negative) + ' | False Negatives = ' + str(false_negative))
# if elif else, to calculate tnr and tpr, if applicable, if not N/A
if true_negative > 0:
    tnr = true_negative / (true_negative + false_positive)
elif false_positive > 0:
    tnr = 0
else:
    tnr = 'N/A'
if true_positive > 0:
    tpr = true_positive / (true_positive + false_negative)
elif false_negative > 0:
    tpr = 0
else:
    tpr = 'N/A'

# print TPR and TNR string plus results and blank line
print('True Positive Rate = ' + str(tpr))
print('True Negative Rate = ' + str(tnr))
print('\n')
# clear lists
y_pred_list.clear()
y_test_list.clear()
accuracy_list.clear()

# set points to 1900 and counts to zero
points = 1900
true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0
# input data equals minutes, fga, games from df
input_data = harden_random_forest[['MP', 'FGA', 'G']]
# define x as input data values and y as label values for above below points
X = input_data.values
y = harden_random_forest['AboveBelow1900'].values
# split data 50/50 test train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# define decision tree and fit X_train and y_train to model
decision_tree = tree.DecisionTreeClassifier(criterion='entropy')
decision_tree.fit(X_train, y_train)
# define y_pred as decision tree predict X_test
y_pred = decision_tree.predict(X_test)
# compute accuracy of prediction vs y test and multiply by 100 for model
decision_tree_accuracy = accuracy_score(y_test, y_pred)
decision_tree_accuracy = decision_tree_accuracy * 100
# define avg_instance from player averages
avg_instance = np.asmatrix([player_df_sum_avg.iloc[1, 1], player_df_sum_avg.iloc[1, 2],
                                        player_df_sum_avg.iloc[1, 3]])
# print string and points
print('Decision Tree : ' + str(points) + ' Points')
# define decision tree label to equal prediction for average instance
decision_tree_label = decision_tree.predict(avg_instance)
# if decision tree label less than 1, ie negative label < points total, print strings and accuracy
if decision_tree_label < 1:
    print('James Harden will score below ' + str(points) + ' points next season.')
    print('Accuracy of decision tree = ' + str((float(round(decision_tree_accuracy, 4)))) + '%')
    # append points to decision_tree_below
    decision_tree_below.append(points)
# else decision tree label not less than 1, ie positive label > points total, print strings and accuracy
else:
    print('James Harden will score above ' + str(points) + ' points next season.')
    print('Accuracy of decision tree = ' + str((float(round(decision_tree_accuracy, 4)))) + '%')
    # append points to decision_tree_above
    decision_tree_above.append(points)
# append decision to decision_tree_accuracy list
decision_tree_accuracy_list.append(decision_tree_accuracy)
# print new line
print('\n')

# define subtree and depth, create empty lists
subtrees = 1
depth = 1
label_list = []
depth_list = []
subtrees_list = []
accuracy_list = []
# while loop nested to iterate through depths 1-2 and subtrees 1-5
while depth <= 2:
    while subtrees <= 5:
        # random forest classifier with subtrees and depth from iterations
        random_forest = RandomForestClassifier(n_estimators=subtrees, max_depth=depth, criterion='entropy')
        # fit random_forest X_train, y_train, define pred y to predict X_test
        random_forest.fit(X_train, y_train)
        y_pred = random_forest.predict(X_test)
        # define accuracy and append accuracy to list, append y_pred and y_test to respective lists
        accuracy = (accuracy_score(y_test, y_pred))
        accuracy_list.append(accuracy)
        y_pred_list.append(y_pred)
        y_test_list.append(y_test)

        # define average instance for player averages
        avg_instance = np.asmatrix([player_df_sum_avg.iloc[1, 1], player_df_sum_avg.iloc[1, 2],
                                        player_df_sum_avg.iloc[1, 3]])
        # define label as random_forest predict avg instance
        label = random_forest.predict(avg_instance)
        # append label, subtrees, depth to respective lists
        label_list.append(label)
        subtrees_list.append(subtrees)
        depth_list.append(depth)
        # increment subtrees
        subtrees = subtrees + 1
    # increment depth set subtrees back to 1
    depth = depth + 1
    subtrees = 1

# create player tree dataframe with subtrees, depth and labels from their lists
harden_tree_df = pd.DataFrame(columns=['Subtrees', 'Depth', 'Label'])
harden_tree_df['Subtrees'] = subtrees_list
harden_tree_df['Depth'] = depth_list
harden_tree_df['Label'] = label_list
# define x, y and text, annotate text with label column
x = harden_tree_df['Depth']
y = harden_tree_df['Subtrees']
text = harden_tree_df['Label']

figure(figsize=(15, 15), dpi=80)
plt.scatter(x, y)
for i in range(len(x)):
    plt.annotate(text[i], (x[i], y[i]),)

# plot x and y labels set title, save figure
plt.xlabel('Depth\nAnnotation = Label \nLabel : 1 = Above Total Points | 0 = Below Total Points')
plt.ylabel('Subtrees')
plt.title('James Harden Random Forrest Classifier Scatter Plot ' + str(points) + ' Points')
plt.savefig('Random Forest/James Harden/James Harden Random Forrest Classifier Scatter Plot '
            + str(points) + ' Points.pdf')

# define sum_label as sum of label list
sum_label = sum(label_list)
# print string and points
print('Random Forest : ' + str(points) + ' Points')
# if sum_label > length of label list/2 ie sum of all random forests depths/subtrees
if sum_label > len(label_list) / 2:
    # print string for above plus points
    print('James Harden will score above ' + str(points) + ' points next season.')
    # append points to above list
    above_list.append(points)
# elif sum_label < length of label list/2 ie sum of all random forests depths/subtrees
elif sum_label < len(label_list) / 2:
    # print string for below plus points
    print('James Harden will score below ' + str(points) + ' points next season.')
    # append points to below list
    below_list.append(points)
# else sum_label = length of label list/2 ie it is a split undecided if above or below
else:
    # print string undecided plus points
    print('It is undecided if James Harden will score above or below ' + str(points) + ' points next season.')
    # append to below list as undecided is conservatively not a positive result
    below_list.append(points)
# define sum_accuracy = sum of accuracy list
sum_accuracy = sum(accuracy_list)
# confidence = sum_accuracy / length of accuracy list * 100
confidence = (sum_accuracy / len(accuracy_list)) * 100
# append confidence to confidence list
confidence_list.append(confidence)
# print string plus sum accuracy for all depths and subtrees rounded
print('Sum accuracy of all subtrees and depths for prediction = ' + str(float(round(confidence, 4))) + '%')

# create list of arrays for y_pred and y_test
y_pred_list = [arr.tolist() for arr in y_pred_list]
y_test_list = [arr.tolist() for arr in y_test_list]

# iterate items in sublist of lists to get each element
y_pred_list = [item for sublist in y_pred_list for item in sublist]
y_test_list = [item for sublist in y_test_list for item in sublist]


# set counter to 0, while i < length of y_pred list
i = 0
while i < len(y_pred_list):
    # calculate true positive, true negative, false positive, false negative through counts comparing each list at i
    if y_pred_list[i] == y_test_list[i] and y_pred_list[i] == 1:
        true_positive = true_positive + 1
    elif y_pred_list[i] == y_test_list[i] and y_pred_list[i] == 0:
        true_negative = true_negative + 1
    elif y_pred_list[i] != y_test_list[i] and y_pred_list[i] == 1:
        false_positive = false_positive + 1
    else:
        false_negative = false_negative + 1
    # increment counter for while loop
    i = i + 1

# print string with amount of predictions and tp, fp, tn, fn
print('Of ' + str(len(y_pred_list)) + ' predictions from all subtrees and depths : True Positives = '
      + str(true_positive) + ' | False Positives = ' + str(false_positive) + ' | True Negatives = '
      + str(true_negative) + ' | False Negatives = ' + str(false_negative))
# if elif else, to calculate tnr and tpr, if applicable, if not N/A
if true_negative > 0:
    tnr = true_negative / (true_negative + false_positive)
elif false_positive > 0:
    tnr = 0
else:
    tnr = 'N/A'
if true_positive > 0:
    tpr = true_positive / (true_positive + false_negative)
elif false_negative > 0:
    tpr = 0
else:
    tpr = 'N/A'

# print TPR and TNR string plus results and blank line
print('True Positive Rate = ' + str(tpr))
print('True Negative Rate = ' + str(tnr))
print('\n')
# clear lists
y_pred_list.clear()
y_test_list.clear()
accuracy_list.clear()

# set points to 2000 and counts to zero
points = 2000
true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0
# input data equals minutes, fga, games from df
input_data = harden_random_forest[['MP', 'FGA', 'G']]
# define x as input data values and y as label values for above below points
X = input_data.values
y = harden_random_forest['AboveBelow2000'].values
# split data 50/50 test train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# define decision tree and fit X_train and y_train to model
decision_tree = tree.DecisionTreeClassifier(criterion='entropy')
decision_tree.fit(X_train, y_train)
# define y_pred as decision tree predict X_test
y_pred = decision_tree.predict(X_test)
# compute accuracy of prediction vs y test and multiply by 100 for model
decision_tree_accuracy = accuracy_score(y_test, y_pred)
decision_tree_accuracy = decision_tree_accuracy * 100
# define avg_instance from player averages
avg_instance = np.asmatrix([player_df_sum_avg.iloc[1, 1], player_df_sum_avg.iloc[1, 2],
                                        player_df_sum_avg.iloc[1, 3]])
# print string and points
print('Decision Tree : ' + str(points) + ' Points')
# define decision tree label to equal prediction for average instance
decision_tree_label = decision_tree.predict(avg_instance)
# if decision tree label less than 1, ie negative label < points total, print strings and accuracy
if decision_tree_label < 1:
    print('James Harden will score below ' + str(points) + ' points next season.')
    print('Accuracy of decision tree = ' + str((float(round(decision_tree_accuracy, 4)))) + '%')
    # append points to decision_tree_below
    decision_tree_below.append(points)
# else decision tree label not less than 1, ie positive label > points total, print strings and accuracy
else:
    print('James Harden will score above ' + str(points) + ' points next season.')
    print('Accuracy of decision tree = ' + str((float(round(decision_tree_accuracy, 4)))) + '%')
    # append points to decision_tree_above
    decision_tree_above.append(points)
# append decision to decision_tree_accuracy list
decision_tree_accuracy_list.append(decision_tree_accuracy)
# print new line
print('\n')

# define subtree and depth, create empty lists
subtrees = 1
depth = 1
label_list = []
depth_list = []
subtrees_list = []
accuracy_list = []
# while loop nested to iterate through depths 1-2 and subtrees 1-5
while depth <= 2:
    while subtrees <= 5:
        # random forest classifier with subtrees and depth from iterations
        random_forest = RandomForestClassifier(n_estimators=subtrees, max_depth=depth, criterion='entropy')
        # fit random_forest X_train, y_train, define pred y to predict X_test
        random_forest.fit(X_train, y_train)
        y_pred = random_forest.predict(X_test)
        # define accuracy and append accuracy to list, append y_pred and y_test to respective lists
        accuracy = (accuracy_score(y_test, y_pred))
        accuracy_list.append(accuracy)
        y_pred_list.append(y_pred)
        y_test_list.append(y_test)

        # define average instance for player averages
        avg_instance = np.asmatrix([player_df_sum_avg.iloc[1, 1], player_df_sum_avg.iloc[1, 2],
                                        player_df_sum_avg.iloc[1, 3]])
        # define label as random_forest predict avg instance
        label = random_forest.predict(avg_instance)
        # append label, subtrees, depth to respective lists
        label_list.append(label)
        subtrees_list.append(subtrees)
        depth_list.append(depth)
        # increment subtrees
        subtrees = subtrees + 1
    # increment depth set subtrees back to 1
    depth = depth + 1
    subtrees = 1

# create player tree dataframe with subtrees, depth and labels from their lists
harden_tree_df = pd.DataFrame(columns=['Subtrees', 'Depth', 'Label'])
harden_tree_df['Subtrees'] = subtrees_list
harden_tree_df['Depth'] = depth_list
harden_tree_df['Label'] = label_list
# define x, y and text, annotate text with label column
x = harden_tree_df['Depth']
y = harden_tree_df['Subtrees']
text = harden_tree_df['Label']

figure(figsize=(15, 15), dpi=80)
plt.scatter(x, y)
for i in range(len(x)):
    plt.annotate(text[i], (x[i], y[i]),)

# plot x and y labels set title, save figure
plt.xlabel('Depth\nAnnotation = Label \nLabel : 1 = Above Total Points | 0 = Below Total Points')
plt.ylabel('Subtrees')
plt.title('James Harden Random Forrest Classifier Scatter Plot ' + str(points) + ' Points')
plt.savefig('Random Forest/James Harden/James Harden Random Forrest Classifier Scatter Plot '
            + str(points) + ' Points.pdf')

# define sum_label as sum of label list
sum_label = sum(label_list)
# print string and points
print('Random Forest : ' + str(points) + ' Points')
# if sum_label > length of label list/2 ie sum of all random forests depths/subtrees
if sum_label > len(label_list) / 2:
    # print string for above plus points
    print('James Harden will score above ' + str(points) + ' points next season.')
    # append points to above list
    above_list.append(points)
# elif sum_label < length of label list/2 ie sum of all random forests depths/subtrees
elif sum_label < len(label_list) / 2:
    # print string for below plus points
    print('James Harden will score below ' + str(points) + ' points next season.')
    # append points to below list
    below_list.append(points)
# else sum_label = length of label list/2 ie it is a split undecided if above or below
else:
    # print string undecided plus points
    print('It is undecided if James Harden will score above or below ' + str(points) + ' points next season.')
    # append to below list as undecided is conservatively not a positive result
    below_list.append(points)
# define sum_accuracy = sum of accuracy list
sum_accuracy = sum(accuracy_list)
# confidence = sum_accuracy / length of accuracy list * 100
confidence = (sum_accuracy / len(accuracy_list)) * 100
# append confidence to confidence list
confidence_list.append(confidence)
# print string plus sum accuracy for all depths and subtrees rounded
print('Sum accuracy of all subtrees and depths for prediction = ' + str(float(round(confidence, 4))) + '%')

# create list of arrays for y_pred and y_test
y_pred_list = [arr.tolist() for arr in y_pred_list]
y_test_list = [arr.tolist() for arr in y_test_list]

# iterate items in sublist of lists to get each element
y_pred_list = [item for sublist in y_pred_list for item in sublist]
y_test_list = [item for sublist in y_test_list for item in sublist]


# set counter to 0, while i < length of y_pred list
i = 0
while i < len(y_pred_list):
    # calculate true positive, true negative, false positive, false negative through counts comparing each list at i
    if y_pred_list[i] == y_test_list[i] and y_pred_list[i] == 1:
        true_positive = true_positive + 1
    elif y_pred_list[i] == y_test_list[i] and y_pred_list[i] == 0:
        true_negative = true_negative + 1
    elif y_pred_list[i] != y_test_list[i] and y_pred_list[i] == 1:
        false_positive = false_positive + 1
    else:
        false_negative = false_negative + 1
    # increment counter for while loop
    i = i + 1

# print string with amount of predictions and tp, fp, tn, fn
print('Of ' + str(len(y_pred_list)) + ' predictions from all subtrees and depths : True Positives = '
      + str(true_positive) + ' | False Positives = ' + str(false_positive) + ' | True Negatives = '
      + str(true_negative) + ' | False Negatives = ' + str(false_negative))
# if elif else, to calculate tnr and tpr, if applicable, if not N/A
if true_negative > 0:
    tnr = true_negative / (true_negative + false_positive)
elif false_positive > 0:
    tnr = 0
else:
    tnr = 'N/A'
if true_positive > 0:
    tpr = true_positive / (true_positive + false_negative)
elif false_negative > 0:
    tpr = 0
else:
    tpr = 'N/A'

# print TPR and TNR string plus results and blank line
print('True Positive Rate = ' + str(tpr))
print('True Negative Rate = ' + str(tnr))
print('\n')
# clear lists
y_pred_list.clear()
y_test_list.clear()
accuracy_list.clear()

# set points to 2100 and counts to zero
points = 2100
true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0
# input data equals minutes, fga, games from df
input_data = harden_random_forest[['MP', 'FGA', 'G']]
# define x as input data values and y as label values for above below points
X = input_data.values
y = harden_random_forest['AboveBelow2100'].values
# split data 50/50 test train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# define decision tree and fit X_train and y_train to model
decision_tree = tree.DecisionTreeClassifier(criterion='entropy')
decision_tree.fit(X_train, y_train)
# define y_pred as decision tree predict X_test
y_pred = decision_tree.predict(X_test)
# compute accuracy of prediction vs y test and multiply by 100 for model
decision_tree_accuracy = accuracy_score(y_test, y_pred)
decision_tree_accuracy = decision_tree_accuracy * 100
# define avg_instance from player averages
avg_instance = np.asmatrix([player_df_sum_avg.iloc[1, 1], player_df_sum_avg.iloc[1, 2],
                                        player_df_sum_avg.iloc[1, 3]])
# print string and points
print('Decision Tree : ' + str(points) + ' Points')
# define decision tree label to equal prediction for average instance
decision_tree_label = decision_tree.predict(avg_instance)
# if decision tree label less than 1, ie negative label < points total, print strings and accuracy
if decision_tree_label < 1:
    print('James Harden will score below ' + str(points) + ' points next season.')
    print('Accuracy of decision tree = ' + str((float(round(decision_tree_accuracy, 4)))) + '%')
    # append points to decision_tree_below
    decision_tree_below.append(points)
# else decision tree label not less than 1, ie positive label > points total, print strings and accuracy
else:
    print('James Harden will score above ' + str(points) + ' points next season.')
    print('Accuracy of decision tree = ' + str((float(round(decision_tree_accuracy, 4)))) + '%')
    # append points to decision_tree_above
    decision_tree_above.append(points)
# append decision to decision_tree_accuracy list
decision_tree_accuracy_list.append(decision_tree_accuracy)
# print new line
print('\n')

# define subtree and depth, create empty lists
subtrees = 1
depth = 1
label_list = []
depth_list = []
subtrees_list = []
accuracy_list = []
# while loop nested to iterate through depths 1-2 and subtrees 1-5
while depth <= 2:
    while subtrees <= 5:
        # random forest classifier with subtrees and depth from iterations
        random_forest = RandomForestClassifier(n_estimators=subtrees, max_depth=depth, criterion='entropy')
        # fit random_forest X_train, y_train, define pred y to predict X_test
        random_forest.fit(X_train, y_train)
        y_pred = random_forest.predict(X_test)
        # define accuracy and append accuracy to list, append y_pred and y_test to respective lists
        accuracy = (accuracy_score(y_test, y_pred))
        accuracy_list.append(accuracy)
        y_pred_list.append(y_pred)
        y_test_list.append(y_test)

        # define average instance for player averages
        avg_instance = np.asmatrix([player_df_sum_avg.iloc[1, 1], player_df_sum_avg.iloc[1, 2],
                                        player_df_sum_avg.iloc[1, 3]])
        # define label as random_forest predict avg instance
        label = random_forest.predict(avg_instance)
        # append label, subtrees, depth to respective lists
        label_list.append(label)
        subtrees_list.append(subtrees)
        depth_list.append(depth)
        # increment subtrees
        subtrees = subtrees + 1
    # increment depth set subtrees back to 1
    depth = depth + 1
    subtrees = 1

# create player tree dataframe with subtrees, depth and labels from their lists
harden_tree_df = pd.DataFrame(columns=['Subtrees', 'Depth', 'Label'])
harden_tree_df['Subtrees'] = subtrees_list
harden_tree_df['Depth'] = depth_list
harden_tree_df['Label'] = label_list
# define x, y and text, annotate text with label column
x = harden_tree_df['Depth']
y = harden_tree_df['Subtrees']
text = harden_tree_df['Label']

figure(figsize=(15, 15), dpi=80)
plt.scatter(x, y)
for i in range(len(x)):
    plt.annotate(text[i], (x[i], y[i]),)

# plot x and y labels set title, save figure
plt.xlabel('Depth\nAnnotation = Label \nLabel : 1 = Above Total Points | 0 = Below Total Points')
plt.ylabel('Subtrees')
plt.title('James Harden Random Forrest Classifier Scatter Plot ' + str(points) + ' Points')
plt.savefig('Random Forest/James Harden/James Harden Random Forrest Classifier Scatter Plot '
            + str(points) + ' Points.pdf')

# define sum_label as sum of label list
sum_label = sum(label_list)
# print string and points
print('Random Forest : ' + str(points) + ' Points')
# if sum_label > length of label list/2 ie sum of all random forests depths/subtrees
if sum_label > len(label_list) / 2:
    # print string for above plus points
    print('James Harden will score above ' + str(points) + ' points next season.')
    # append points to above list
    above_list.append(points)
# elif sum_label < length of label list/2 ie sum of all random forests depths/subtrees
elif sum_label < len(label_list) / 2:
    # print string for below plus points
    print('James Harden will score below ' + str(points) + ' points next season.')
    # append points to below list
    below_list.append(points)
# else sum_label = length of label list/2 ie it is a split undecided if above or below
else:
    # print string undecided plus points
    print('It is undecided if James Harden will score above or below ' + str(points) + ' points next season.')
    # append to below list as undecided is conservatively not a positive result
    below_list.append(points)
# define sum_accuracy = sum of accuracy list
sum_accuracy = sum(accuracy_list)
# confidence = sum_accuracy / length of accuracy list * 100
confidence = (sum_accuracy / len(accuracy_list)) * 100
# append confidence to confidence list
confidence_list.append(confidence)
# print string plus sum accuracy for all depths and subtrees rounded
print('Sum accuracy of all subtrees and depths for prediction = ' + str(float(round(confidence, 4))) + '%')

# create list of arrays for y_pred and y_test
y_pred_list = [arr.tolist() for arr in y_pred_list]
y_test_list = [arr.tolist() for arr in y_test_list]

# iterate items in sublist of lists to get each element
y_pred_list = [item for sublist in y_pred_list for item in sublist]
y_test_list = [item for sublist in y_test_list for item in sublist]


# set counter to 0, while i < length of y_pred list
i = 0
while i < len(y_pred_list):
    # calculate true positive, true negative, false positive, false negative through counts comparing each list at i
    if y_pred_list[i] == y_test_list[i] and y_pred_list[i] == 1:
        true_positive = true_positive + 1
    elif y_pred_list[i] == y_test_list[i] and y_pred_list[i] == 0:
        true_negative = true_negative + 1
    elif y_pred_list[i] != y_test_list[i] and y_pred_list[i] == 1:
        false_positive = false_positive + 1
    else:
        false_negative = false_negative + 1
    # increment counter for while loop
    i = i + 1

# print string with amount of predictions and tp, fp, tn, fn
print('Of ' + str(len(y_pred_list)) + ' predictions from all subtrees and depths : True Positives = '
      + str(true_positive) + ' | False Positives = ' + str(false_positive) + ' | True Negatives = '
      + str(true_negative) + ' | False Negatives = ' + str(false_negative))
# if elif else, to calculate tnr and tpr, if applicable, if not N/A
if true_negative > 0:
    tnr = true_negative / (true_negative + false_positive)
elif false_positive > 0:
    tnr = 0
else:
    tnr = 'N/A'
if true_positive > 0:
    tpr = true_positive / (true_positive + false_negative)
elif false_negative > 0:
    tpr = 0
else:
    tpr = 'N/A'

# print TPR and TNR string plus results and blank line
print('True Positive Rate = ' + str(tpr))
print('True Negative Rate = ' + str(tnr))
print('\n')
# clear lists
y_pred_list.clear()
y_test_list.clear()
accuracy_list.clear()

# set points to 2200 and counts to zero
points = 2200
true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0
# input data equals minutes, fga, games from df
input_data = harden_random_forest[['MP', 'FGA', 'G']]
# define x as input data values and y as label values for above below points
X = input_data.values
y = harden_random_forest['AboveBelow2200'].values
# split data 50/50 test train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# define decision tree and fit X_train and y_train to model
decision_tree = tree.DecisionTreeClassifier(criterion='entropy')
decision_tree.fit(X_train, y_train)
# define y_pred as decision tree predict X_test
y_pred = decision_tree.predict(X_test)
# compute accuracy of prediction vs y test and multiply by 100 for model
decision_tree_accuracy = accuracy_score(y_test, y_pred)
decision_tree_accuracy = decision_tree_accuracy * 100
# define avg_instance from player averages
avg_instance = np.asmatrix([player_df_sum_avg.iloc[1, 1], player_df_sum_avg.iloc[1, 2],
                                        player_df_sum_avg.iloc[1, 3]])
# print string and points
print('Decision Tree : ' + str(points) + ' Points')
# define decision tree label to equal prediction for average instance
decision_tree_label = decision_tree.predict(avg_instance)
# if decision tree label less than 1, ie negative label < points total, print strings and accuracy
if decision_tree_label < 1:
    print('James Harden will score below ' + str(points) + ' points next season.')
    print('Accuracy of decision tree = ' + str((float(round(decision_tree_accuracy, 4)))) + '%')
    # append points to decision_tree_below
    decision_tree_below.append(points)
# else decision tree label not less than 1, ie positive label > points total, print strings and accuracy
else:
    print('James Harden will score above ' + str(points) + ' points next season.')
    print('Accuracy of decision tree = ' + str((float(round(decision_tree_accuracy, 4)))) + '%')
    # append points to decision_tree_above
    decision_tree_above.append(points)
# append decision to decision_tree_accuracy list
decision_tree_accuracy_list.append(decision_tree_accuracy)
# print new line
print('\n')

# define subtree and depth, create empty lists
subtrees = 1
depth = 1
label_list = []
depth_list = []
subtrees_list = []
accuracy_list = []
# while loop nested to iterate through depths 1-2 and subtrees 1-5
while depth <= 2:
    while subtrees <= 5:
        # random forest classifier with subtrees and depth from iterations
        random_forest = RandomForestClassifier(n_estimators=subtrees, max_depth=depth, criterion='entropy')
        # fit random_forest X_train, y_train, define pred y to predict X_test
        random_forest.fit(X_train, y_train)
        y_pred = random_forest.predict(X_test)
        # define accuracy and append accuracy to list, append y_pred and y_test to respective lists
        accuracy = (accuracy_score(y_test, y_pred))
        accuracy_list.append(accuracy)
        y_pred_list.append(y_pred)
        y_test_list.append(y_test)

        # define average instance for player averages
        avg_instance = np.asmatrix([player_df_sum_avg.iloc[1, 1], player_df_sum_avg.iloc[1, 2],
                                        player_df_sum_avg.iloc[1, 3]])
        # define label as random_forest predict avg instance
        label = random_forest.predict(avg_instance)
        # append label, subtrees, depth to respective lists
        label_list.append(label)
        subtrees_list.append(subtrees)
        depth_list.append(depth)
        # increment subtrees
        subtrees = subtrees + 1
    # increment depth set subtrees back to 1
    depth = depth + 1
    subtrees = 1

# create player tree dataframe with subtrees, depth and labels from their lists
harden_tree_df = pd.DataFrame(columns=['Subtrees', 'Depth', 'Label'])
harden_tree_df['Subtrees'] = subtrees_list
harden_tree_df['Depth'] = depth_list
harden_tree_df['Label'] = label_list
# define x, y and text, annotate text with label column
x = harden_tree_df['Depth']
y = harden_tree_df['Subtrees']
text = harden_tree_df['Label']

figure(figsize=(15, 15), dpi=80)
plt.scatter(x, y)
for i in range(len(x)):
    plt.annotate(text[i], (x[i], y[i]),)

# plot x and y labels set title, save figure
plt.xlabel('Depth\nAnnotation = Label \nLabel : 1 = Above Total Points | 0 = Below Total Points')
plt.ylabel('Subtrees')
plt.title('James Harden Random Forrest Classifier Scatter Plot ' + str(points) + ' Points')
plt.savefig('Random Forest/James Harden/James Harden Random Forrest Classifier Scatter Plot '
            + str(points) + ' Points.pdf')

# define sum_label as sum of label list
sum_label = sum(label_list)
# print string and points
print('Random Forest : ' + str(points) + ' Points')
# if sum_label > length of label list/2 ie sum of all random forests depths/subtrees
if sum_label > len(label_list) / 2:
    # print string for above plus points
    print('James Harden will score above ' + str(points) + ' points next season.')
    # append points to above list
    above_list.append(points)
# elif sum_label < length of label list/2 ie sum of all random forests depths/subtrees
elif sum_label < len(label_list) / 2:
    # print string for below plus points
    print('James Harden will score below ' + str(points) + ' points next season.')
    # append points to below list
    below_list.append(points)
# else sum_label = length of label list/2 ie it is a split undecided if above or below
else:
    # print string undecided plus points
    print('It is undecided if James Harden will score above or below ' + str(points) + ' points next season.')
    # append to below list as undecided is conservatively not a positive result
    below_list.append(points)
# define sum_accuracy = sum of accuracy list
sum_accuracy = sum(accuracy_list)
# confidence = sum_accuracy / length of accuracy list * 100
confidence = (sum_accuracy / len(accuracy_list)) * 100
# append confidence to confidence list
confidence_list.append(confidence)
# print string plus sum accuracy for all depths and subtrees rounded
print('Sum accuracy of all subtrees and depths for prediction = ' + str(float(round(confidence, 4))) + '%')

# create list of arrays for y_pred and y_test
y_pred_list = [arr.tolist() for arr in y_pred_list]
y_test_list = [arr.tolist() for arr in y_test_list]

# iterate items in sublist of lists to get each element
y_pred_list = [item for sublist in y_pred_list for item in sublist]
y_test_list = [item for sublist in y_test_list for item in sublist]


# set counter to 0, while i < length of y_pred list
i = 0
while i < len(y_pred_list):
    # calculate true positive, true negative, false positive, false negative through counts comparing each list at i
    if y_pred_list[i] == y_test_list[i] and y_pred_list[i] == 1:
        true_positive = true_positive + 1
    elif y_pred_list[i] == y_test_list[i] and y_pred_list[i] == 0:
        true_negative = true_negative + 1
    elif y_pred_list[i] != y_test_list[i] and y_pred_list[i] == 1:
        false_positive = false_positive + 1
    else:
        false_negative = false_negative + 1
    # increment counter for while loop
    i = i + 1

# print string with amount of predictions and tp, fp, tn, fn
print('Of ' + str(len(y_pred_list)) + ' predictions from all subtrees and depths : True Positives = '
      + str(true_positive) + ' | False Positives = ' + str(false_positive) + ' | True Negatives = '
      + str(true_negative) + ' | False Negatives = ' + str(false_negative))
# if elif else, to calculate tnr and tpr, if applicable, if not N/A
if true_negative > 0:
    tnr = true_negative / (true_negative + false_positive)
elif false_positive > 0:
    tnr = 0
else:
    tnr = 'N/A'
if true_positive > 0:
    tpr = true_positive / (true_positive + false_negative)
elif false_negative > 0:
    tpr = 0
else:
    tpr = 'N/A'

# print TPR and TNR string plus results and blank line
print('True Positive Rate = ' + str(tpr))
print('True Negative Rate = ' + str(tnr))
print('\n')
# clear lists
y_pred_list.clear()
y_test_list.clear()
accuracy_list.clear()

# if length of decision_tree_below greater than 0 ie has lower limit
if len(decision_tree_below) > 0:
    # if decision tree at index 0 == 1200, decision upper = 1200, decision lower = 0
    if decision_tree_below[0] == 1200:
        decision_tree_upper = 1200
        decision_tree_lower = 0
# else for element in list decision tree above check if it is greater than element in below, if so remove
    else:
        for e in list(decision_tree_above):
            if e > decision_tree_below[0]:
                decision_tree_above.remove(e)
        # upper and lower limits = element at index 0 in below and -1 in above
        decision_tree_upper = decision_tree_below[0]
        decision_tree_lower = decision_tree_above[-1]
# no below predicted last above is new low, string above is new high ie all predictions above
else:
    decision_tree_upper = 'above'
    decision_tree_lower = 2200

# if length of decision_tree_below greater than 0 ie has lower limit
if len(below_list) > 0:
    # if below list at index 0 = 1200 ie random forest - upper =1200 lower = 0
    if below_list[0] == 1200:
        upper_level = 1200
        lower_level = 0
    # else for element in list above list check if it is greater than element in below, if so remove
    else:
        for e in list(above_list):
            if e > below_list[0]:
                above_list.remove(e)
        # upper level and lower level = element at index 0 in below and -1 in above
        upper_level = below_list[0]
        lower_level = above_list[-1]
# no below predicted last above is new low, string above is new high ie all predictions above
else:
    upper_level = 'above'
    lower_level = 2200

# decision confidence = sum of decision tree accuracy list / its length (decision tree total accuracy)
decision_confidence = sum(decision_tree_accuracy_list) / len(decision_tree_accuracy_list)

# sum_confidence = sum of confidence list / its length (random forest total accuracy)
sum_confidence = sum(confidence_list) / len(confidence_list)


# print strings, pper and lower scoring predictions, and accuracy for decision tree and random forest classifiers
print("\n*** Final Predictions ***\n")
print('Decision Tree :')
print('Based off of the Decision Tree predictions and average games, minutes, and field goal attempts.\n'
      'A conservative prediction is that James Harden will score between '
      + str(decision_tree_lower) + ' and ' + str(decision_tree_upper) +
      ' points next season given he meets these averages.')
print('Sum accuracy of all Decision Tree predictions = ' + str(float(round(decision_confidence, 4))) + '%')
print('\nRandom Forest :')
print('Based off of the conglomerate Random Forest predictions and average games, minutes, and field goal attempts.\n'
      'A conservative prediction is that James Harden will score between '
      + str(lower_level) + ' and ' + str(upper_level) + ' points next season given he meets these averages.')
print('Sum accuracy of all Random Forest predictions = ' + str(float(round(sum_confidence, 4))) + '%')