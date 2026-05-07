import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import pandas as pd
import numpy as np
import json


def fig_to_json(fig):
    return json.loads(pio.to_json(fig))


def plot_histogram(df, column):
    fig = px.histogram(
        df, x=column, nbins=40,
        title=f'Distribution of {column}',
        color_discrete_sequence=['#4361ee'],
        template='plotly_white',
    )
    fig.update_layout(
        xaxis_title=column, yaxis_title='Count',
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig_to_json(fig)


def plot_boxplot(df, column):
    fig = px.box(
        df, y=column,
        title=f'Box Plot of {column}',
        color_discrete_sequence=['#7209b7'],
        template='plotly_white',
    )
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig_to_json(fig)


def plot_bar_chart(df, column):
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, 'count']
    fig = px.bar(
        counts, x=column, y='count',
        title=f'Value Counts of {column}',
        color='count',
        color_continuous_scale='viridis',
        template='plotly_white',
    )
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig_to_json(fig)


def plot_scatter(df, col_x, col_y, color_col=None):
    fig = px.scatter(
        df, x=col_x, y=col_y, color=color_col,
        title=f'{col_x} vs {col_y}',
        template='plotly_white',
        opacity=0.6,
    )
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig_to_json(fig)


def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr().round(2)

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale='RdBu_r',
        zmin=-1, zmax=1,
        text=corr.values,
        texttemplate='%{text}',
        textfont={"size": 9},
    ))
    fig.update_layout(
        title='Correlation Heatmap',
        template='plotly_white',
        margin=dict(l=40, r=40, t=60, b=40),
        height=600,
    )
    return fig_to_json(fig)


def plot_time_series(df, date_col='date', value_col='demand'):
    ts = df.groupby(date_col)[value_col].mean().reset_index()
    fig = px.line(
        ts, x=date_col, y=value_col,
        title=f'{value_col} Over Time',
        template='plotly_white',
    )
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig_to_json(fig)


def plot_category_demand(df, cat_col='category', value_col='demand'):
    agg = df.groupby(cat_col)[value_col].agg(['mean', 'sum', 'count']).reset_index()
    agg.columns = [cat_col, 'Mean Demand', 'Total Demand', 'Count']

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Mean Demand', 'Total Demand'))
    fig.add_trace(
        go.Bar(x=agg[cat_col], y=agg['Mean Demand'], name='Mean', marker_color='#4361ee'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=agg[cat_col], y=agg['Total Demand'], name='Total', marker_color='#f72585'),
        row=1, col=2
    )
    fig.update_layout(
        title=f'Demand by {cat_col}', template='plotly_white',
        margin=dict(l=40, r=40, t=80, b=40), showlegend=False,
    )
    return fig_to_json(fig)


def plot_demand_by_region(df):
    agg = df.groupby('region')['demand'].mean().reset_index()
    fig = px.bar(
        agg, x='region', y='demand',
        title='Average Demand by Region',
        color='demand',
        color_continuous_scale='plasma',
        template='plotly_white',
    )
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig_to_json(fig)


def plot_price_vs_demand(df):
    fig = px.scatter(
        df, x='final_price', y='demand',
        color='category', size='discount_pct',
        title='Price vs Demand by Category',
        template='plotly_white',
        opacity=0.6,
    )
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig_to_json(fig)


def plot_discount_impact(df):
    df_copy = df.copy()
    df_copy['discount_bin'] = pd.cut(
        df_copy['discount_pct'],
        bins=[-.01, 0, 0.05, 0.1, 0.15, 0.2, 1.0],
        labels=['No Discount', '0-5%', '5-10%', '10-15%', '15-20%', '20%+']
    )
    agg = df_copy.groupby('discount_bin', observed=False)['demand'].mean().reset_index()
    fig = px.bar(
        agg, x='discount_bin', y='demand',
        title='Impact of Discount on Average Demand',
        color='demand',
        color_continuous_scale='viridis',
        template='plotly_white',
    )
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig_to_json(fig)


def plot_model_comparison(results, task_type='regression'):
    names = list(results.keys())
    if task_type == 'regression' or task_type == 'timeseries':
        metric = 'R2'
        title = 'Model Comparison (R2 Score)'
    else:
        metric = 'Accuracy'
        title = 'Model Comparison (Accuracy)'

    values = [results[n][metric] for n in names]
    types = [results[n].get('type', 'Other') for n in names]

    color_map = {'Bagging': '#4361ee', 'Boosting': '#f72585', 'Voting': '#4cc9f0'}
    colors = [color_map.get(t, '#888') for t in types]

    fig = go.Figure(data=[
        go.Bar(x=names, y=values, marker_color=colors, text=[f'{v:.4f}' for v in values], textposition='outside')
    ])
    fig.update_layout(
        title=title, template='plotly_white',
        yaxis_title=metric,
        margin=dict(l=40, r=40, t=60, b=100),
    )
    return fig_to_json(fig)


def plot_feature_importance(model, feature_names, top_n=15):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return None

    indices = np.argsort(importances)[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig = go.Figure(data=[
        go.Bar(x=top_importances, y=top_features, orientation='h', marker_color='#4361ee')
    ])
    fig.update_layout(
        title=f'Top {top_n} Feature Importances',
        template='plotly_white',
        xaxis_title='Importance',
        margin=dict(l=120, r=40, t=60, b=40),
        height=500,
    )
    return fig_to_json(fig)


def plot_actual_vs_predicted(y_test, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(y_test))), y=y_test,
        mode='lines', name='Actual', line=dict(color='#4361ee'),
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(y_pred))), y=y_pred,
        mode='lines', name='Predicted', line=dict(color='#f72585', dash='dash'),
    ))
    fig.update_layout(
        title='Actual vs Predicted',
        template='plotly_white',
        xaxis_title='Sample Index',
        yaxis_title='Value',
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig_to_json(fig)


def plot_residuals(y_test, y_pred):
    residuals = np.array(y_test) - np.array(y_pred)
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Residual Distribution', 'Residuals vs Predicted'))

    fig.add_trace(
        go.Histogram(x=residuals, nbinsx=40, marker_color='#4361ee', name='Residuals'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=y_pred, y=residuals, mode='markers', marker=dict(color='#f72585', opacity=0.5), name='Residuals'),
        row=1, col=2
    )
    fig.add_hline(y=0, line_dash='dash', line_color='gray', row=1, col=2)

    fig.update_layout(
        title='Residual Analysis',
        template='plotly_white',
        margin=dict(l=40, r=40, t=80, b=40),
        showlegend=False,
    )
    return fig_to_json(fig)


def plot_confusion_matrix(cm, labels=None):
    if labels is None:
        labels = [f'Class {i}' for i in range(len(cm))]

    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 14},
    ))
    fig.update_layout(
        title='Confusion Matrix',
        template='plotly_white',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig_to_json(fig)
