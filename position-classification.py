import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import numpy as np

####################
# Data Prep
####################

#Import data
data = pd.read_csv("./nba22-23.csv")

#Drop Duplicates
data = data.drop_duplicates(subset = 'Rk', keep = "first").reset_index(drop=True)

#Remove redundant or irrelavnt features
data = data.drop(["Rk", "Player", "Age", "Tm", "GS", "FG", "FG%", "FG%", "eFG%", "3P", "3P%", "2P", "2P%", "FT", 'FT%', "TRB", "PTS", "G", "MP", "PF", "TOV", "Player-additional"], axis = 1)


scaler = MinMaxScaler()
data.iloc[:,1:] = scaler.fit_transform(data.iloc[:,1:])

#Keep players from main 5 positions
positions = ['SG', 'C', 'SF', 'PF', 'PG']
data = data[data['Pos'].isin(positions)]

#Split data into training and testing sets
y = data['Pos']
x = data.drop("Pos", axis = 1)
trainX, testX, trainY, testY = train_test_split(x, y, train_size=0.75, stratify = y, random_state = 0)

####################
# Decision Tree
####################
print("---------------------\nDecision Tree Results:\n---------------------\n")

#apply decision tree model, use grid search to determine hyperparamters
dt = DecisionTreeClassifier(random_state = 0)
dt_param_grid = {
    'max_depth': [4, 5, 6, 7, 8],
    'min_samples_split': [3, 8, 10],
    'min_samples_leaf': [1, 2, 3]
}
dt_grid_search = GridSearchCV(dt, dt_param_grid, cv = 5, scoring = 'accuracy')
dt_grid_search.fit(trainX, trainY)
best_dt_model = dt_grid_search.best_estimator_

#predict classes using logistic regression model for taining & testing data
dt_y_pred_train = best_dt_model.predict(trainX)
dt_y_pred_test = best_dt_model.predict(testX)

#print training & testing accuracy
print("Training Accuracy:")
print(accuracy_score(trainY, dt_y_pred_train))
print("Testing Accuracy:")
print(accuracy_score(testY, dt_y_pred_test))

#print confusion matrix
print("Confusion matrix:")
print(pd.crosstab(testY, dt_y_pred_test, rownames = ['True'], colnames = ['Predicted'], margins = True))

#use 10 fold cross validation and find average accuracy for all folds
dt_scores = cross_val_score(best_dt_model, x, y, cv = 10)
print("Cross-validation scores:")
print(dt_scores)
print("Average cross-validation score:")
print(dt_scores.mean())

####################
# Decision Tree Feature Importance
####################

print("\n---------------------\nRefined Decision Tree Results:\n---------------------\n")


feature_importances = pd.DataFrame({'Feature': x.columns, 'Importance': best_dt_model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

print("Feature importances:")
print(feature_importances)

#Drop less influential features
data = data.drop(["FGA", "FTA", "2PA", "BLK"], axis = 1)

y = data['Pos']
x = data.drop("Pos", axis = 1)
trainX, testX, trainY, testY = train_test_split(x, y, train_size=0.75, stratify = y, random_state = 0)

#over sample the data to ensure all classes have same number of observations
over_sampler = RandomOverSampler(random_state = 0)
trainX_resampled, trainY_resampled= over_sampler.fit_resample(trainX, trainY)

#apply decision tree model, use grid search to determine hyperparamters
dt_refined = DecisionTreeClassifier(random_state = 0)
dt_refined_param_grid = {
    'max_depth': [4, 5, 6, 7, 8],
    'min_samples_split': [3, 8, 10],
    'min_samples_leaf': [1, 2, 3]
}
dt_refined_grid_search = GridSearchCV(dt_refined, dt_refined_param_grid, cv = 5, scoring = 'accuracy')
dt_refined_grid_search.fit(trainX_resampled, trainY_resampled)

best_dt_refined_model = dt_refined_grid_search.best_estimator_

#predict classes using logistic regression model for taining & testing data
dt_refined_y_pred_train = best_dt_refined_model.predict(trainX)
dt_refined_y_pred_test = best_dt_refined_model.predict(testX)

#print training & testing accuracy
print("Training Accuracy:")
print(accuracy_score(trainY, dt_refined_y_pred_train))
print("Testing Accuracy:")
print(accuracy_score(testY, dt_refined_y_pred_test))

#print confusion matrix
print("Confusion matrix:")
print(pd.crosstab(testY, dt_refined_y_pred_test, rownames = ['True'], colnames = ['Predicted'], margins = True))

#use 10 fold cross validation and find average accuracy for all folds
dt_refined_scores = cross_val_score(best_dt_refined_model, x, y, cv = 10)
print("Cross-validation scores:")
print(dt_refined_scores)
print("Average cross-validation score:")
print(dt_refined_scores.mean())

####################
# K Nearest Neighbors
####################

print("\n---------------------\nK Nearest Neighbors Results:\n---------------------\n")

#apply knn model, use grid search to determine hyperparamters
knn = KNeighborsClassifier()
knn_param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # 1 for Manhattan distance (L1), 2 for Euclidean distance (L2)
}

knn_grid_search = GridSearchCV(knn, knn_param_grid, cv=5, scoring='accuracy')
knn_grid_search.fit(trainX_resampled, trainY_resampled)

best_knn_model = knn_grid_search.best_estimator_

knn_y_pred_train = best_knn_model.predict(trainX)
knn_y_pred_test = best_knn_model.predict(testX)

#print training & testing accuracy
print("Training Accuracy:")
print(accuracy_score(trainY, knn_y_pred_train))
print("Testing Accuracy:")
print(accuracy_score(testY, knn_y_pred_test))

#print confusion matrix
print("Confusion matrix:")
print(pd.crosstab(testY, knn_y_pred_test, rownames = ['True'], colnames = ['Predicted'], margins = True))

#use 10 fold cross validation and find average accuracy for all folds
knn_scores = cross_val_score(best_knn_model, x, y, cv = 10)
print("Cross-validation scores:")
print(knn_scores)
print("Average cross-validation score:")
print(knn_scores.mean())

####################
# Support Vector Machine
####################

print("\n---------------------\nSupport Vector Machine Results:\n---------------------\n")

#apply svm model, use grid search to determine hyperparamters
svm = SVC()
svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto', 0.1, 1]
}

svm_grid_search = GridSearchCV(svm, svm_param_grid, cv=5, scoring='accuracy')
svm_grid_search.fit(trainX_resampled, trainY_resampled)

best_svm_model = svm_grid_search.best_estimator_

svm_y_pred_train = best_svm_model.predict(trainX)
svm_y_pred_test = best_svm_model.predict(testX)

#print training & testing accuracy
print("Training Accuracy:")
print(accuracy_score(trainY, svm_y_pred_train))
print("Testing Accuracy:")
print(accuracy_score(testY, svm_y_pred_test))

#print confusion matrix
print("Confusion matrix:")
print(pd.crosstab(testY, svm_y_pred_test, rownames = ['True'], colnames = ['Predicted'], margins = True))

#use 10 fold cross validation and find average accuracy for all folds
svm_scores = cross_val_score(best_svm_model, x, y, cv = 10)
print("Cross-validation scores:")
print(svm_scores)
print("Average cross-validation score:")
print(svm_scores.mean())

####################
# Naive Bayes
####################

print("\n---------------------\nNaive Bayes Results:\n---------------------\n")

#apply naive bayes model, no hyperparameters to tune
nb = GaussianNB()
nb.fit(trainX_resampled, trainY_resampled)

nb_y_pred_train = nb.predict(trainX)
nb_y_pred_test = nb.predict(testX)

#print training & testing accuracy
print("Training Accuracy:")
print(accuracy_score(trainY, nb_y_pred_train))
print("Testing Accuracy:")
print(accuracy_score(testY, nb_y_pred_test))

#print confusion matrix
print("Confusion matrix:")
print(pd.crosstab(testY, nb_y_pred_test, rownames = ['True'], colnames = ['Predicted'], margins = True))

#use 10 fold cross validation and find average accuracy for all folds
nb_scores = cross_val_score(nb, x, y, cv = 10)
print("Cross-validation scores:")
print(nb_scores)
print("Average cross-validation score:")
print(nb_scores.mean())

####################
# Logistic Regression
####################

print("\n---------------------\nLogistic Regression Results:\n---------------------\n")

#apply logistic regression model, use grid search to determine hyperparamters
lr = LogisticRegression(max_iter=1000)  # Adjust max_iter based on your dataset
lr_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100]
}

lr_grid_search = GridSearchCV(lr, lr_param_grid, cv=5, scoring='accuracy')
lr_grid_search.fit(trainX_resampled, trainY_resampled)

best_lr_model = lr_grid_search.best_estimator_

lr_y_pred_train = best_lr_model.predict(trainX)
lr_y_pred_test = best_lr_model.predict(testX)

#print training & testing accuracy
print("Training Accuracy:")
print(accuracy_score(trainY, lr_y_pred_train))
print("Testing Accuracy:")
print(accuracy_score(testY, lr_y_pred_test))

#print confusion matrix
print("Confusion matrix:")
print(pd.crosstab(testY, lr_y_pred_test, rownames = ['True'], colnames = ['Predicted'], margins = True))

#use 10 fold cross validation and find average accuracy for all folds
lr_scores = cross_val_score(best_lr_model, x, y, cv = 10)
print("Cross-validation scores:")
print(lr_scores)
print("Average cross-validation score:")
print(lr_scores.mean())

####################
# Model Comparison
####################

#Models and their corresponding results
models = ['Decision Tree', 'Refined Decision Tree', 'KNN', 'SVM', 'Naive Bayes', 'Logistic Regression']
training_accuracies = [accuracy_score(trainY, dt_y_pred_train),
                       accuracy_score(trainY, dt_refined_y_pred_train),
                       accuracy_score(trainY, knn_y_pred_train),
                       accuracy_score(trainY, svm_y_pred_train),
                       accuracy_score(trainY, nb_y_pred_train),
                       accuracy_score(trainY, lr_y_pred_train)]

testing_accuracies = [accuracy_score(testY, dt_y_pred_test),
                      accuracy_score(testY, dt_refined_y_pred_test),
                      accuracy_score(testY, knn_y_pred_test),
                      accuracy_score(testY, svm_y_pred_test),
                      accuracy_score(testY, nb_y_pred_test),
                      accuracy_score(testY, lr_y_pred_test)]

cv_accuracies = [dt_scores.mean(), dt_refined_scores.mean(), knn_scores.mean(),
                 svm_scores.mean(), nb_scores.mean(), lr_scores.mean()]

#Plotting the bar graph
bar_width = 0.25
index = np.arange(len(models))

fig, ax = plt.subplots(figsize=(10, 6))

bar1 = ax.bar(index - bar_width, training_accuracies, bar_width, label='Training Accuracy')
bar2 = ax.bar(index, testing_accuracies, bar_width, label='Testing Accuracy')
bar3 = ax.bar(index + bar_width, cv_accuracies, bar_width, label='Avg. CV Accuracy')

ax.set_xlabel('Model Type')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Model Comparison: Training, Testing, and Average CV Accuracy')
ax.set_xticks(index)
ax.set_xticklabels(models)
ax.legend()
plt.grid(axis = 'y')
plt.tight_layout()
plt.show()