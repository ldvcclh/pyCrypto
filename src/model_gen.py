import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, recall_score, f1_score, \
    classification_report, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#from lightgbm import LGBMClassifier

class ModelGenerator:
    """
    A class to generate and evaluate machine learning models for time-series data.

    Parameters:
    -----------
    filename (str): The name of the file containing the data with targets.
    print_info (bool, optional): Whether to print information during the process. Defaults to False.

    Methods:
    --------
    create_train_test_split(test_size=0.3, shuffle=False):
        Splits the data into train and test sets.
    train_logistic_regression():
        Trains a logistic regression model.
    train_random_forest(n_estimators=100, random_state=42):
        Trains a random forest model.
    evaluate_model(model, X_test, y_test):
        Evaluates the performance of the given model.
    apply_pca(n_components=20):
        Applies PCA to the train and test sets.
    save_model(model, filename):
        Saves the model to a file using joblib.
    save_data(data, filename):
        Saves the data (e.g., train/test sets) to a file using joblib.
    """

    def __init__(self, filename, datapath, print_info=False):
        self.filename = filename
        self.datapath = datapath
        self.print_info = print_info
        self.df = pd.read_parquet(self.filename, engine='pyarrow')

    def log(self, message):
        """Helper function to handle optional printing."""
        if self.print_info:
            print(message)

    def create_train_test_split(self, test_size=0.3, shuffle=False):
        """
        Splits the data into train and test sets.

        Parameters:
        -----------
        test_size (float): The proportion of the dataset to include in the test split. Defaults to 0.3.
        shuffle (bool): Whether or not to shuffle the data before splitting. Defaults to False.
        """
        X = self.df.drop(columns=['pct_change_close', 'target'])
        y = self.df['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size,
                                                                                shuffle=shuffle)
        self.log("Train/test split completed.")

    def train_logistic_regression(self, pca = False):
        """Trains a logistic regression model."""
        model = LogisticRegression()
        if pca:
            model.fit(self.X_train_pca,self.y_train)
        else:
            model.fit(self.X_train, self.y_train)
        self.log("Logistic regression model trained.")
        return model

    def train_random_forest(self, n_estimators=100, random_state=42, pca = False):
        """Trains a random forest classifier."""
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        if pca:
            rf.fit(self.X_train_pca, self.y_train)
        else:
            rf.fit(self.X_train, self.y_train)
        self.log(f"Random Forest model trained with {n_estimators} trees.")
        return rf

    def train_xgboost(self, params=None):
        """Trains an XGBoost classifier."""
        if params is None:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'use_label_encoder': False,
                'n_estimators': 100,
                'random_state': 42
            }
        xgboost_model = GradientBoostingClassifier(**params)
        xgboost_model.fit(self.X_train, self.y_train)
        self.log("XGBoost model trained.")
        return xgboost_model

    #def train_lightgbm(self, params=None):
    #    """Trains a LightGBM classifier."""
    #    if params is None:
    #        params = {
    #            'objective': 'binary',
    #            'metric': 'binary_logloss',
    #            'n_estimators': 100,
    #            'random_state': 42
    #        }
    #    lightgbm_model = LGBMClassifier(**params)
    #    lightgbm_model.fit(self.X_train, self.y_train)
    #    self.log("LightGBM model trained.")
    #    return lightgbm_model

    def apply_pca(self, n_components=20):
        """Applies PCA to reduce dimensionality of the data."""
        pca = PCA(n_components=n_components)
        self.X_train_pca = pca.fit_transform(self.X_train)
        self.X_test_pca = pca.transform(self.X_test)
        self.log(f"PCA applied with {n_components} components.")
        self.log(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        return pca

    def evaluate_model(self, model, X_test, y_test):
        """Evaluates the performance of the model."""
        # ROC AUC
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        l2_error = mean_squared_error(y_test, y_pred)

        # Print results
        self.log(f"ROC AUC: {roc_auc}")
        self.log(f"Accuracy: {accuracy}")
        self.log(f"Precision: {precision}")
        self.log(f"Recall: {recall}")
        self.log(f"F1-Score: {f1}")
        self.log(f"L2 error: {l2_error}")
        self.log(f"Report:\n{classification_report(y_test, y_pred)}")

    def save_model(self, model, filename):
        """Saves the model to a file."""
        joblib.dump(model, filename)
        self.log(f"Model saved to {filename}")

    def save_data(self, data, filename):
        """Saves data (e.g., train/test sets) to a file."""
        joblib.dump(data, filename)
        self.log(f"Data saved to {filename}")
