import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    BaggingRegressor,
    BaggingClassifier,
    VotingRegressor,
    VotingClassifier,
    AdaBoostRegressor,
    AdaBoostClassifier,
    ExtraTreesRegressor,
    ExtraTreesClassifier,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)


def get_regression_models():
    models = {
        'Random Forest (Bagging)': RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        'Bagging (Decision Tree)': BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=10),
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        'Extra Trees (Bagging)': ExtraTreesRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbosity=0
        ),
        'AdaBoost': AdaBoostRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
    }
    return models


def get_classification_models():
    models = {
        'Random Forest (Bagging)': RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        'Bagging (Decision Tree)': BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=10),
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        'Extra Trees (Bagging)': ExtraTreesClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbosity=0,
            eval_metric='mlogloss'
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
    }
    return models


def build_voting_regressor(X_train, y_train):
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)),
    ]
    voting_model = VotingRegressor(estimators=estimators, n_jobs=-1)
    voting_model.fit(X_train, y_train)
    return voting_model


def build_voting_classifier(X_train, y_train):
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0, eval_metric='mlogloss')),
    ]
    voting_model = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
    voting_model.fit(X_train, y_train)
    return voting_model


def train_regression_models(X, y, test_size=0.2, selected_models=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    all_models = get_regression_models()
    if selected_models:
        all_models = {k: v for k, v in all_models.items() if k in selected_models}

    results = {}
    trained_models = {}

    for name, model in all_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)

        results[name] = {
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4),
            'R2': round(r2, 4),
            'CV_Mean_R2': round(cv_scores.mean(), 4),
            'CV_Std_R2': round(cv_scores.std(), 4),
            'type': 'Bagging' if 'Bagging' in name or 'Forest' in name or 'Extra' in name else 'Boosting',
        }
        trained_models[name] = model

    voting_model = build_voting_regressor(X_train, y_train)
    y_pred_voting = voting_model.predict(X_test)
    mse_v = mean_squared_error(y_test, y_pred_voting)
    cv_voting = cross_val_score(voting_model, X, y, cv=5, scoring='r2', n_jobs=-1)

    results['Voting Ensemble'] = {
        'MSE': round(mse_v, 4),
        'RMSE': round(np.sqrt(mse_v), 4),
        'MAE': round(mean_absolute_error(y_test, y_pred_voting), 4),
        'R2': round(r2_score(y_test, y_pred_voting), 4),
        'CV_Mean_R2': round(cv_voting.mean(), 4),
        'CV_Std_R2': round(cv_voting.std(), 4),
        'type': 'Voting',
    }
    trained_models['Voting Ensemble'] = voting_model

    best_model_name = max(results, key=lambda k: results[k]['R2'])
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'best_regression_model.pkl'))
    joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, 'regression_features.pkl'))

    return results, trained_models, best_model_name, {
        'X_test': X_test, 'y_test': y_test,
        'y_pred': trained_models[best_model_name].predict(X_test)
    }


def train_classification_models(X, y, test_size=0.2, selected_models=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    all_models = get_classification_models()
    if selected_models:
        all_models = {k: v for k, v in all_models.items() if k in selected_models}

    results = {}
    trained_models = {}

    for name, model in all_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)

        results[name] = {
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1_Score': round(f1, 4),
            'CV_Mean_Accuracy': round(cv_scores.mean(), 4),
            'CV_Std_Accuracy': round(cv_scores.std(), 4),
            'type': 'Bagging' if 'Bagging' in name or 'Forest' in name or 'Extra' in name else 'Boosting',
        }
        trained_models[name] = model

    voting_model = build_voting_classifier(X_train, y_train)
    y_pred_voting = voting_model.predict(X_test)
    acc_v = accuracy_score(y_test, y_pred_voting)
    cv_voting = cross_val_score(voting_model, X, y, cv=5, scoring='accuracy', n_jobs=-1)

    results['Voting Ensemble'] = {
        'Accuracy': round(acc_v, 4),
        'Precision': round(precision_score(y_test, y_pred_voting, average='weighted', zero_division=0), 4),
        'Recall': round(recall_score(y_test, y_pred_voting, average='weighted', zero_division=0), 4),
        'F1_Score': round(f1_score(y_test, y_pred_voting, average='weighted', zero_division=0), 4),
        'CV_Mean_Accuracy': round(cv_voting.mean(), 4),
        'CV_Std_Accuracy': round(cv_voting.std(), 4),
        'type': 'Voting',
    }
    trained_models['Voting Ensemble'] = voting_model

    best_model_name = max(results, key=lambda k: results[k]['Accuracy'])
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'best_classification_model.pkl'))
    joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, 'classification_features.pkl'))

    cm = confusion_matrix(y_test, best_model.predict(X_test))
    report = classification_report(y_test, best_model.predict(X_test), output_dict=True)

    return results, trained_models, best_model_name, {
        'X_test': X_test, 'y_test': y_test,
        'y_pred': best_model.predict(X_test),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
    }


def train_timeseries_models(X, y, test_size=0.2):
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    models = {
        'Random Forest (Bagging)': RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbosity=0
        ),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        results[name] = {
            'MSE': round(mse, 4),
            'RMSE': round(np.sqrt(mse), 4),
            'MAE': round(mean_absolute_error(y_test, y_pred), 4),
            'R2': round(r2_score(y_test, y_pred), 4),
            'type': 'Bagging' if 'Forest' in name else 'Boosting',
        }
        trained_models[name] = model

    voting = VotingRegressor(
        estimators=[
            ('rf', RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)),
            ('xgb', XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)),
        ],
        n_jobs=-1
    )
    voting.fit(X_train, y_train)
    y_pred_v = voting.predict(X_test)
    mse_v = mean_squared_error(y_test, y_pred_v)

    results['Voting Ensemble'] = {
        'MSE': round(mse_v, 4),
        'RMSE': round(np.sqrt(mse_v), 4),
        'MAE': round(mean_absolute_error(y_test, y_pred_v), 4),
        'R2': round(r2_score(y_test, y_pred_v), 4),
        'type': 'Voting',
    }
    trained_models['Voting Ensemble'] = voting

    best_model_name = max(results, key=lambda k: results[k]['R2'])
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'best_timeseries_model.pkl'))
    joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, 'timeseries_features.pkl'))

    return results, trained_models, best_model_name, {
        'X_test': X_test, 'y_test': y_test.values,
        'y_pred': best_model.predict(X_test),
    }


def predict_demand(input_data, task_type='regression'):
    if task_type == 'regression':
        model_path = os.path.join(MODEL_DIR, 'best_regression_model.pkl')
        features_path = os.path.join(MODEL_DIR, 'regression_features.pkl')
    elif task_type == 'classification':
        model_path = os.path.join(MODEL_DIR, 'best_classification_model.pkl')
        features_path = os.path.join(MODEL_DIR, 'classification_features.pkl')
    else:
        model_path = os.path.join(MODEL_DIR, 'best_timeseries_model.pkl')
        features_path = os.path.join(MODEL_DIR, 'timeseries_features.pkl')

    if not os.path.exists(model_path):
        return None, "No trained model found. Please train a model first."

    model = joblib.load(model_path)
    features = joblib.load(features_path)

    import pandas as pd
    input_df = pd.DataFrame([input_data])

    for f in features:
        if f not in input_df.columns:
            input_df[f] = 0

    input_df = input_df[features]
    prediction = model.predict(input_df)

    return prediction[0], None
