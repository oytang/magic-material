import numpy as np 
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import optuna
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate, plot_slice
import optuna.integration.lightgbm as lgb
from lightgbm import early_stopping
from lightgbm import log_evaluation

import json
# def prepare_data():

# 	df_1 = pd.read_csv('data/data_fe_2.csv')
# 	X, Y = df_1.iloc[:,:-6], df_1.iloc[:, -6:]
	
# 	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=4)
# 	cv = KFold(n_splits=5, random_state=1, shuffle=True)
# 	return X_train, X_test, Y_train, Y_test, cv


def objective_lgb(X, y, eval): 

	train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.3)
	dtrain = lgb.Dataset(train_x, label=train_y)
	dval = lgb.Dataset(val_x, label=val_y)

	params = {
        "objective": "regression_l1",
        "metric": "mae",
        "verbosity": -1,
        "boosting_type": "gbdt",
    }
	model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        callbacks=[early_stopping(100), log_evaluation(100)],
    )

	prediction = model.predict(val_x, num_iteration=model.best_iteration)
	mae = mean_absolute_error(val_y, prediction)

	best_params = model.params
	print('Best params:', best_params)
	print('MAE = {}'.format(mae))
	return best_params


def objective_xgb(trial, X, y, eval): 

	params = {
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-8, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.008,0.01,0.012,0.014,0.016,0.018,0.02]),
        'n_estimators': 10000,
        'max_depth': trial.suggest_int("max_depth", 1, 10),
        #'random_state': trial.suggest_categorical('random_state', [2022]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300)
    }
	model = XGBRegressor(**params)

	mae = cv_train_model(model, X, y, cv, eval)
	return mae


def objective_rf(X, y): 

	model = RandomForestRegressor()
	param_distributions = {
        "max_depth": optuna.distributions.IntUniformDistribution(3, 20),
        "min_samples_leaf": optuna.distributions.IntUniformDistribution (1, 300),
        "max_features": optuna.distributions.CategoricalDistribution([0.4,0.5,0.6,0.7,0.8,1.0])
    
    }
	optuna_search = optuna.integration.OptunaSearchCV(
       model, param_distributions, n_trials=100, timeout=600, verbose=2
    )
	optuna_search.fit(X, y)

	best_trial = optuna_search.study_.best_trial
	best_params = best_trial.params
	print('Best params:', best_params)

	return best_params


def cv_train_test_model(model, X_train, y_train, X_test, y_test, cv, scoring):

    results = cross_validate(estimator=model,X=X_train,y=y_train,cv=cv, scoring=scoring,return_train_score=True)
    avg_train = np.abs(results['train_score'].mean())
    avg_val = np.abs(results['test_score'].mean())
    print('Train set: ', np.abs(results['train_score']), '  Avg: ', avg_train)
    print('Validation set: ', np.abs(results['test_score']), '  Avg: ', avg_val)
    #df_results = np.abs(pd.DataFrame(results).iloc[:, -2:])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(predictions, y_test)
    print("Test set: ", mae)
    return pd.DataFrame(data = [[avg_train, avg_val, mae]], columns = ['train','val','test'])
    #return np.abs(results['test_score'].mean())


def cv_train_model(model, X_train, y_train, cv, scoring):
    results = cross_validate(estimator=model,X=X_train,y=y_train,cv=cv, scoring=scoring,return_train_score=True)
    print('Train set: ', np.abs(results['train_score']), '  Avg: ', np.abs(results['train_score'].mean()))
    print('Validation set: ', np.abs(results['test_score']), '  Avg: ', np.abs(results['test_score'].mean()))
    #df_results = np.abs(pd.DataFrame(results).iloc[:, -2:])
    return np.abs(results['test_score'].mean())


 # def feature_importance():
 # 	pass



if __name__ =="__main__":



	scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2', 'neg_mean_squared_log_error' ]

	df_results_all_model_list = []

	all_models = {}

	# Build lgb, xgb and rf models for each property 
	for i in range(6):


		df_1 = pd.read_csv('data/data_fillna_property'+str(i+1)+'.csv')
		X, Y = df_1.iloc[:,:-1], df_1.iloc[:, -1]
	
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=4)
		cv = KFold(n_splits=5, random_state=1, shuffle=True)

		#lgb
		best_params_lgb = objective_lgb(X_train, Y_train, scoring[0])
		best_model_lgb = LGBMRegressor(**best_params_lgb)
		df_results_lgb = cv_train_test_model(best_model_lgb, X_train, Y_train, X_test, Y_test, cv, scoring[0])
		df_results_lgb = df_results_lgb.add_suffix('_lgb').set_axis([Y_train.name], axis=0)
		#df_results_lgb.to_csv(data/str(i)+'_lgb_results.csv', index=False)


		#xgb
		pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
		study_xgb = optuna.create_study(pruner=pruner, direction="minimize")
		study_xgb.optimize(lambda trial: objective_xgb(trial, X_train, Y_train, scoring[0]), n_trials=2)
		print('Number of finished trials: {}'.format(len(study_xgb.trials)))
		best_trial = study_xgb.best_trial
		best_params_xgb = best_trial.params
		best_model_xgb = XGBRegressor(**best_params_xgb)  
		df_results_xgb = cv_train_test_model(best_model_xgb, X_train, Y_train, X_test, Y_test, cv, scoring[0])
		df_results_xgb = df_results_xgb.add_suffix('_xgb').set_axis([Y_train.name], axis=0)
		#df_results_xgb.to_csv(data/str(i)+'_xgb_results.csv', index=False)

		#rf
		best_params_rf = objective_rf(X_train, Y_train)
		best_model_rf = RandomForestRegressor(**best_params_rf)
		df_results_rf = cv_train_test_model(best_model_rf, X_train, Y_train, X_test, Y_test, cv, scoring[0])
		df_results_rf = df_results_rf.add_suffix('_rf').set_axis([Y_train.name], axis=0)
		#df_results_rf.to_csv(data/str(i)+'_rf_results.csv', index=False)

		df_results_all_model = pd.concat([df_results_lgb, df_results_xgb, df_results_rf], axis=1)

		df_results_all_model_list.append(df_results_all_model)

		all_models["property"+str(i+1)] = [best_model_lgb, best_model_xgb, best_model_rf]


	df_results_all_property = pd.concat(df_results_all_model_list, axis=0)
	df_results_all_property.to_csv('data/df_results_fill.csv', index=False)



