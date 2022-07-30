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
            'depth': trial.suggest_int('depth', 3, 7, step=1),
            'iterations': trial.suggest_int('iterations', 50, 200, step=10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
            'loss_function': trial.suggest_categorical('loss_function', ['Logloss','CrossEntropy']),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',5, 30),
            #'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 0.5, 5),
            #scale_pos_weight

        }
        #Colocar o KFold aqui
        regressor = ctb(**param)
        regressor.fit(self.X_train, self.y_train)
        y_pred = regressor.predict(self.x_test)
        auc = metrics.roc_auc_score(y_pred, self.y_test)
        return auc
    
    def optimize(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=100)
        return study.best_params
