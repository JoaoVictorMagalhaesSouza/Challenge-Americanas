from catboost import CatBoostClassifier as ctb
from matplotlib.pyplot import step
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import optuna
import pandas as pd

class OptimizeModel():
    def __init__(self, X_train, x_test, y_train, y_test, folded_data):
        self.X_train = X_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.folded_data = folded_data


    def objective(self,trial):
        # param = {
        #     #Define params for Catboost
        #     'depth': trial.suggest_int('depth', 3, 7, step=1),
        #     'iterations': trial.suggest_int('iterations', 50, 200, step=10),
        #     'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        #     'loss_function': trial.suggest_categorical('loss_function', ['Logloss','CrossEntropy']),
        #     'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',5, 30),
        #     #'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 0.5, 5),
        #     #scale_pos_weight

        # }
        param = {
            'n_estimators': trial.suggest_int('n_estimators',50,500, step=50),
            'criterion': trial.suggest_categorical('criterion',['entropy','gini']),
            'max_depth': trial.suggest_int('max_depth',3,12,step=1),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf',2,12,step=1),
            'random_state':42,            
        }
        #Colocar o KFold aqui
        forest = RandomForestClassifier(**param)
        scores = []
        for fold in self.folded_data.keys():
            print(fold)
            forest.fit(self.folded_data[fold]['X_train'],self.folded_data[fold]['y_train'])
            #y_pred = forest.predict(self.folded_data[fold]['X_test'])
            score = forest.score(self.folded_data[fold]['X_test'], self.folded_data[fold]['y_test'])
            scores.append(score)
        scores = pd.Series(scores)

        return scores.mean()
    
    def optimize(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=100)
        return study.best_params
