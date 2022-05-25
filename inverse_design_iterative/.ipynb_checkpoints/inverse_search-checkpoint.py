import optuna
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from scipy.special import softmax # used for composition constraint
from sklearn.preprocessing import StandardScaler


class inverse_search():

    def __init__(self, model, target, trial_num, sampler="TPE", optim="MSE", use_soft=False, pow_num=2):
        self.study = None
        self.MODEL = model
        self.TARGET = target
        self.SAMPLER = sampler
        self.SOFT = use_soft
        self.NUM = trial_num
        self.OPTIM = optim
        if self.SOFT:
            raise "the pow_num useless"
        else:
            self.POW = pow_num

    def softmax(self, X):
        X_exp = np.exp(X)
        partition = X_exp.sum(1, keepdim=True)
        return X_exp / partition

    def pow_trans(self, X, pow: int):
        X_pow = np.power(X, pow)
        partition = X_pow.sum(1)
        return X_pow / partition

    def objective(self, trial):
        ele1, ele2, ele3, ele4, ele5, ele6, ele7, ele8, ele9, ele10, ele11, ele12 = \
            trial.suggest_float("element1", 0, 1), trial.suggest_float("element2", 0, 1), \
            trial.suggest_float("element3", 0, 1), trial.suggest_float("element4", 0, 1), \
            trial.suggest_float("element5", 0, 1), trial.suggest_float("element6", 0, 1), \
            trial.suggest_float("element7", 0, 1), trial.suggest_float("element8", 0, 1), \
            trial.suggest_float("element9", 0, 1), trial.suggest_float("element10", 0, 1), \
            trial.suggest_float("element11", 0, 1), trial.suggest_float("element12", 0, 1),
        x = np.array([ele1, ele2, ele3, ele4, ele5, ele6, ele7, ele8, ele9, ele10, ele11, ele12]).reshape(-1, 12)
        if self.SOFT:
            x = self.softmax(x)
        else:
            x = self.pow_trans(x, self.POW)
        y = self.MODEL.predict(x)

        if self.OPTIM == "MSE":
            tar = -mean_squared_error(np.array(y).reshape(6, ), self.TARGET)
        elif self.OPTIM == "MAE":
            tar = -mean_absolute_error(np.array(y).reshape(6, ), self.TARGET)
        elif self.OPTIM == "R2":
            tar = r2_score(np.array(y).reshape(6, ), self.TARGET)
        else:
            raise "wrong optimizer"

        # trial.report(R2)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return tar

    def run_search(self):
        if self.SAMPLER == "TPE":
            self.study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
        elif self.SAMPLER == "CMA":
            self.study = optuna.create_study(direction="maximize", sampler=optuna.samplers.CmaEsSampler())
        elif self.SAMPLER == "RANDOM":
            self.study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler())
        else:
            raise "sampler out of exception"

        self.study.optimize(self.objective, n_trials=self.NUM, show_progress_bar=True)
        pruned_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        trial = self.study.best_trial

        print("Study statistics: ")
        print("  Number of finished trials: ", len(self.study.trials))
        print("  Number of complete trials: ", len(complete_trials))
        print("  Best trial {}: ".format(self.OPTIM), trial.value if self.OPTIM == "R2" else -trial.value)
        k = []
        v = []

        for key, value in trial.params.items():
            k.append(key)
            v.append(value)
        v = np.array([v])
        if self.SOFT:
            v = self.softmax(v).squeeze().tolist()
        else:
            v = self.pow_trans(v, self.POW).squeeze().tolist()
        result = pd.DataFrame(v, k)
        print(result)
        return self.study, result

# Bounded region of parameter space
pbounds = {f'element{num:02d}': (0, 1) for num in range(1,13)}
def dict2array(next_point_dict):
    return np.array([[next_point_dict[f'element{num:02d}'] for num in range(1,13)]])


def inverse_BO(X_train, Y_train, Y_given, scaler):
    # Use a GPR as the regressor to predict the property given a probing composition
    gpr = GaussianProcessRegressor(alpha=1, normalize_y=True)
    gpr.fit(X_train, Y_train)

    # objective function for BO
    def diff_property(next_point_array, Y_given, regressor, scaler=scaler):

        """
        Given a probing composition vector (next_point)
        and a given property vector we want to approach (Y_given)

        Return the negative MSE as the objective function

        By maximizing this function in BO, the model would be able to find a composition
        whose property is close to the given one
        """

        Y_pred = regressor.predict(softmax(next_point_array))
        return - mean_squared_error(scaler.inverse_transform(Y_pred), Y_given)

    # BO optimizer
    optimizer = BayesianOptimization(f=None, pbounds=pbounds)

    # register all X-Y pairs in training set:
    # for i in range(len(X_train)):

    #     optimizer.register(
    #         params={f'element{num:02d}':X_train[i][num-1] for num in range(1,13)},
    #         target=-mean_squared_error(Y_given, Y_train[i].reshape(1, -1))
    #     )

    # Expected Improvement acquisition function
    utility = UtilityFunction(kind='ei', xi=0.01, kappa=None)  # kappa is not needed for EI

    max_iter = 500
    target_hist = []
    steps_no_improvement = 0
    for _ in range(max_iter):

        # BO model suggest best next point to probe
        next_point_dict = optimizer.suggest(utility)
        # convert dict to array
        next_point_array = dict2array(next_point_dict)

        # evalute property difference on next point
        target = diff_property(next_point_array, Y_given, gpr)
        target_hist.append(target)

        # register new points into optimizer
        optimizer.register(params=next_point_dict, target=target)

        # print(target, softmax(next_point_array))

        # early stop (patience)
        steps_no_improvement += 1
        if target > optimizer.max['target']:
            steps_no_improvement = 0
        if steps_no_improvement > 50:
            break

    # print(optimizer.max['target'], dict2array(optimizer.max['params']))

    return optimizer.max['target'], dict2array(optimizer.max['params'])