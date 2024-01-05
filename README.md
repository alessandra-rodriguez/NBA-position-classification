# NBA Position Classification

## Description:
This project focuses on classifying NBA players into their positions (Shooting Guard, Center, Small Forward, Power Forward, Point Guard) using their individual per games statistics from the 2022-2023 season (https://www.basketball-reference.com/leagues/NBA_2023_per_game.html). It involves data preprocessing, feature selection, and the application of various classification algorithms such as Decision Trees, K Nearest Neighbors, Support Vector Machine, Naive Bayes, and Logistic Regression. Models are compared using training accuracy, testing accuracy, and average 10 fold cross validation accuracy.

## Features:
* Data cleaning and preprocessing to prepare the dataset.
* Feature scaling using Min-Max scaling.
* Grid search for hyperparameter tuning.
* Cross-validation to evaluate model performance.
* Evaluation of feature importance.
* Visualizing model performance.

## Usage
1. Clone this repo locally
2. Install and update relevant libraries
3. Execute the script

## Observations
* K Nearest Neighbors achieved perfect training accuracy but performed moderately on testing data, suggesting potential overfitting.
* Logistic Regression showed relatively balanced results between training and testing accuracy and achieved the highest testing accuracy.
* Offensive rebounds, assists, 3-point attempts, defensive rebounds, and steals were most influential in determining a player's style and impact
