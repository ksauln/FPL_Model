# FPL Analytics
This repo contains models and analytics for the Fantasy Premier League (FPL) game using data from the reposity here (https://github.com/vaastav/Fantasy-Premier-League)

The FPL_Predictions_NeuralNet.ipynb, FPL_Predictions_RandomForest.ipynb and FPL_Predictions_XGBoost.ipynb Juypter notebooks contain all the steps to run the data for that specific model and then optimize team selection by using linear programming to choose players based on their predicted performance and budget constraints. 

fplPredictionsMain.ipynb is similar but is the main program to initialize the data load, clean, prep, training of different model types, and then optimized team selection. Currently one can run either the XGBoost or Random Forest code block. 
The workbook calls on the following python files:
  - config
  - data_loader
  - data_processor
  - XGBoost_trainer
  - randomForest_trainer
  - fplPredictions
  - team_optimizer

Detailed desciptions of each file are contained within. 
