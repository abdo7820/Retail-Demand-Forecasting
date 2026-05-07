import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session

from models.preprocessing import (
    load_data, get_data_info, handle_missing_values, remove_duplicates,
    encode_categorical, scale_features, feature_engineering,
    prepare_regression_data, prepare_classification_data, prepare_timeseries_data,
)
from models.ensemble_models import (
    train_regression_models, train_classification_models,
    train_timeseries_models, predict_demand,
)
from models.visualization import (
    plot_histogram, plot_boxplot, plot_bar_chart, plot_scatter,
    plot_correlation_heatmap, plot_time_series, plot_category_demand,
    plot_demand_by_region, plot_price_vs_demand, plot_discount_impact,
    plot_model_comparison, plot_feature_importance, plot_actual_vs_predicted,
    plot_residuals, plot_confusion_matrix,
)

app = Flask(__name__)
app.secret_key = 'retail_demand_forecasting_secret_key_2024'

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'retail_demand.csv')

store = {
    'df_raw': None,
    'df_processed': None,
    'df_engineered': None,
    'preprocessing_steps': [],
    'training_results': None,
    'trained_models': None,
    'best_model_name': None,
    'test_data': None,
    'task_type': None,
}


def get_df():
    if store['df_engineered'] is not None:
        return store['df_engineered']
    if store['df_processed'] is not None:
        return store['df_processed']
    if store['df_raw'] is not None:
        return store['df_raw']
    store['df_raw'] = load_data(DATA_PATH)
    return store['df_raw']


@app.route('/')
def index():
    df = get_df()
    info = get_data_info(df)
    return render_template('index.html', info=info, active='home')


@app.route('/data')
def data_exploration():
    df = get_df()
    info = get_data_info(df)
    page = request.args.get('page', 1, type=int)
    per_page = 25
    total_pages = (len(df) + per_page - 1) // per_page
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    end = start + per_page
    data_records = df.iloc[start:end].to_dict('records')
    columns = df.columns.tolist()
    return render_template(
        'data.html', info=info, data=data_records, columns=columns,
        page=page, total_pages=total_pages, active='data',
    )


@app.route('/preprocessing', methods=['GET', 'POST'])
def preprocessing():
    df = get_df()

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'encode':
            method = request.form.get('encode_method', 'label')
            columns = request.form.getlist('encode_columns')
            if columns:
                result_df, _ = encode_categorical(df, columns=columns, method=method)
                store['df_processed'] = result_df
                store['preprocessing_steps'].append(f'Encode ({method}): {", ".join(columns)}')
                flash(f'Columns encoded using {method} encoding.', 'success')
            else:
                flash('Please select columns to encode.', 'warning')

        elif action == 'scale':
            method = request.form.get('scale_method', 'standard')
            columns = request.form.getlist('scale_columns')
            if columns:
                result_df, _ = scale_features(df, columns=columns, method=method)
                store['df_processed'] = result_df
                store['preprocessing_steps'].append(f'Scale ({method}): {", ".join(columns)}')
                flash(f'Columns scaled using {method} scaler.', 'success')
            else:
                flash('Please select columns to scale.', 'warning')

        elif action == 'feature_engineering':
            store['df_engineered'] = feature_engineering(df)
            store['preprocessing_steps'].append('Feature Engineering')
            flash('Feature engineering applied successfully.', 'success')

        elif action == 'reset':
            store['df_raw'] = load_data(DATA_PATH)
            store['df_processed'] = None
            store['df_engineered'] = None
            store['preprocessing_steps'] = []
            flash('Data reset to original.', 'info')

        return redirect(url_for('preprocessing'))

    info = get_data_info(df)
    return render_template(
        'preprocessing.html', info=info,
        steps=store['preprocessing_steps'], active='preprocessing',
    )


@app.route('/visualization', methods=['GET', 'POST'])
def visualization():
    df = get_df()
    columns = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    plot_data = None
    plot_type = None

    if request.method == 'POST':
        plot_type = request.form.get('plot_type')

        if plot_type == 'histogram':
            col = request.form.get('column')
            if col:
                plot_data = plot_histogram(df, col)

        elif plot_type == 'boxplot':
            col = request.form.get('column')
            if col:
                plot_data = plot_boxplot(df, col)

        elif plot_type == 'bar':
            col = request.form.get('column')
            if col:
                plot_data = plot_bar_chart(df, col)

        elif plot_type == 'scatter':
            col_x = request.form.get('col_x')
            col_y = request.form.get('col_y')
            color_col = request.form.get('color_col') or None
            if col_x and col_y:
                plot_data = plot_scatter(df, col_x, col_y, color_col)

        elif plot_type == 'correlation':
            plot_data = plot_correlation_heatmap(df)

        elif plot_type == 'timeseries':
            plot_data = plot_time_series(df)

        elif plot_type == 'category_demand':
            plot_data = plot_category_demand(df)

        elif plot_type == 'region_demand':
            plot_data = plot_demand_by_region(df)

        elif plot_type == 'price_demand':
            plot_data = plot_price_vs_demand(df)

        elif plot_type == 'discount_impact':
            plot_data = plot_discount_impact(df)

    return render_template(
        'visualization.html',
        columns=columns, numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        plot_data=json.dumps(plot_data) if plot_data else None,
        plot_type=plot_type, active='visualization',
    )


@app.route('/train', methods=['GET', 'POST'])
def train():
    df = get_df()

    if 'demand_level' not in df.columns:
        df = feature_engineering(df)
        store['df_engineered'] = df

    if request.method == 'POST':
        task_type = request.form.get('task_type', 'regression')
        test_size = float(request.form.get('test_size', 0.2))
        store['task_type'] = task_type

        try:
            if task_type == 'regression':
                X, y = prepare_regression_data(df, target='demand')
                results, models, best_name, test_data = train_regression_models(X, y, test_size)

            elif task_type == 'classification':
                X, y, label_encoder = prepare_classification_data(df, target='demand_level')
                results, models, best_name, test_data = train_classification_models(X, y, test_size)
                if label_encoder:
                    test_data['label_encoder'] = label_encoder

            elif task_type == 'timeseries':
                X, y, dates = prepare_timeseries_data(df, target='demand')
                results, models, best_name, test_data = train_timeseries_models(X, y, test_size)
                test_data['dates'] = dates

            store['training_results'] = results
            store['trained_models'] = models
            store['best_model_name'] = best_name
            store['test_data'] = test_data

            flash(f'Models trained successfully! Best model: {best_name}', 'success')
            return redirect(url_for('results'))

        except Exception as e:
            import traceback
            traceback.print_exc()
            flash(f'Training error: {str(e)}', 'danger')
            return redirect(url_for('train'))

    return render_template('train.html', active='train')


@app.route('/results')
def results():
    if store['training_results'] is None:
        flash('No training results available. Please train models first.', 'warning')
        return redirect(url_for('train'))

    results = store['training_results']
    best_name = store['best_model_name']
    task_type = store['task_type']
    test_data = store['test_data']

    comparison_plot = plot_model_comparison(results, task_type)

    best_model = store['trained_models'].get(best_name)
    if best_model and test_data:
        features = list(test_data['X_test'].columns) if hasattr(test_data['X_test'], 'columns') else []
        importance_plot = plot_feature_importance(best_model, features) if features else None
    else:
        importance_plot = None

    actual_vs_pred_plot = None
    residual_plot = None
    cm_plot = None

    if test_data:
        y_test = test_data['y_test']
        y_pred = test_data['y_pred']

        if isinstance(y_test, pd.Series):
            y_test_list = y_test.values.tolist()
        elif isinstance(y_test, np.ndarray):
            y_test_list = y_test.tolist()
        else:
            y_test_list = list(y_test)

        if isinstance(y_pred, np.ndarray):
            y_pred_list = y_pred.tolist()
        else:
            y_pred_list = list(y_pred)

        if task_type in ['regression', 'timeseries']:
            actual_vs_pred_plot = plot_actual_vs_predicted(y_test_list, y_pred_list)
            residual_plot = plot_residuals(y_test_list, y_pred_list)
        elif task_type == 'classification':
            cm = test_data.get('confusion_matrix')
            if cm:
                le = test_data.get('label_encoder')
                labels = list(le.classes_) if le else ['Low', 'Medium', 'High']
                cm_plot = plot_confusion_matrix(cm, labels)

    return render_template(
        'results.html',
        results=results, best_name=best_name, task_type=task_type,
        comparison_plot=json.dumps(comparison_plot) if comparison_plot else None,
        importance_plot=json.dumps(importance_plot) if importance_plot else None,
        actual_vs_pred_plot=json.dumps(actual_vs_pred_plot) if actual_vs_pred_plot else None,
        residual_plot=json.dumps(residual_plot) if residual_plot else None,
        cm_plot=json.dumps(cm_plot) if cm_plot else None,
        active='results',
    )


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    df = get_df()
    prediction_result = None
    error = None

    if request.method == 'POST':
        task_type = store.get('task_type', 'regression')
        try:
            input_data = {}
            for key in request.form:
                if key != 'task_type':
                    try:
                        input_data[key] = float(request.form[key])
                    except ValueError:
                        input_data[key] = request.form[key]

            pred, err = predict_demand(input_data, task_type)
            if err:
                error = err
            else:
                if task_type == 'classification':
                    label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
                    prediction_result = label_map.get(int(pred), str(pred))
                else:
                    prediction_result = round(float(pred), 2)

        except Exception as e:
            error = str(e)

    categories = df['category'].unique().tolist() if 'category' in df.columns else []
    products = df['product'].unique().tolist() if 'product' in df.columns else []
    regions = df['region'].unique().tolist() if 'region' in df.columns else []

    return render_template(
        'predict.html',
        prediction=prediction_result, error=error,
        task_type=store.get('task_type', 'regression'),
        categories=categories, products=products, regions=regions,
        active='predict',
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
