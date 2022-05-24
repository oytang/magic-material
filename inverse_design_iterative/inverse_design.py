import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
import numpy as np
import pandas as pd

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



if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    toy_alloy_data = pd.read_csv("../toy_alloy_data.csv")
    X_name = [f'element {i}' for i in range(1, 13)]
    Y_name = [f'property {i}' for i in range(1, 7)]
    X = np.array(toy_alloy_data[X_name].values)
    Y = np.array(toy_alloy_data[Y_name].values)
    RFR = RandomForestRegressor()
    rfr = RFR.fit(pd.DataFrame(X), pd.DataFrame(Y))

    MODEL = rfr
    TARGET = [-0.05795637837567597
        , 0.6312238714021171
        , 0.759084924046567
        , 0.39885104208581484
        , -0.003599773609153955
        , 1.3429499392913864
              ]

    ex = inverse_search(MODEL, TARGET, 10)
    study, re = ex.run_search()


