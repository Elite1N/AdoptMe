import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, cohen_kappa_score


class TabularModel:
    def __init__(self):
        self.model = None
        self.best_params = None
        
    def preprocess(self, df):
        df = df.copy()
        df["name_length"] = df['Name'].str.len().fillna(0)
    
        # Description length
        df['description_length'] = df['Description'].str.len().fillna(0)
    
        # Drop unused columns
        cols_to_drop = ['Name', 'PetID', 'RescuerID', 'Description']
        df.drop(cols_to_drop, axis=1, inplace=True)
        
        return df

    
    def train(self, X_train, y_train):
        
        parameters = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9, 12],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
        }
        
        rscv = RandomizedSearchCV(
            estimator=xgb.XGBClassifier(objective="multi:softprob", # softmax only return the winning class, this will return a vector we want
                                        eval_metric = "mlogloss"), # validation metric for early stoppage in training
            param_distributions=parameters,
            scoring=make_scorer(cohen_kappa_score, weights='quadratic'), # Decider for best model from rscv
            n_iter=50, # Try n random combinations
            cv=5,
            verbose=1,
            random_state=42,
            n_jobs=-1 # Parallel processing
        )

        rscv.fit(X_train, y_train)
        self.model = rscv.best_estimator_
        self.best_params = rscv.best_params_
        print (f"Training Complete. Best Params: {self.best_params}")
        
        
    def get_probs(self, df):
        if self.model is None:
            raise Exception("Model not trained yet!")
        
        processed_df = self.preprocess(df)
        return self.model.predict_proba(processed_df) # This gives array of prediction (5 labels) as np.array