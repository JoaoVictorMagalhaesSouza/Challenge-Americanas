from catboost import CatBoostClassifier as ctb
from matplotlib.pyplot import step
from sklearn import metrics
import optuna

class OptimizeCatboost():
    def __init__(self, X_train, x_test, y_train, y_test):
        self.X_train = X_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test


    def objective(self,trial):
        param = {
            #Define params for Catboost
            'max_depth': trial.suggest_int('depth', 4, 10, step=1),
            'n_estimators': trial.suggest_int('iterations', 100, 500, step=50),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5),
            'loss_function': trial.suggest_categorical('loss_function', ['Logloss','CrossEntropy']),
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 0.5, 5),

        }
        regressor = ctb(**param)
        regressor.fit(self.X_train, self.y_train)
        y_pred = regressor.predict(self.x_test)
        auc = metrics.roc_auc_score(y_pred, self.y_test)
        return auc
    
    def optimize(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=100)
        return study.best_params
