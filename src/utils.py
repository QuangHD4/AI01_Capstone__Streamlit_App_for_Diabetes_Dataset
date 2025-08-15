from typing import Sequence, Literal
from colorsys import hsv_to_rgb

import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde, t

def column_multicheck_dropdown_with_aggregations(data: pd.DataFrame) -> Sequence[str]:
    if 'selected_cols' not in st.session_state:
        st.session_state['selected_cols'] = {column : True for column in data.columns}
            
    selected = data.columns.drop([key for key in st.session_state['selected_cols'] if not st.session_state['selected_cols'][key]])

    # render the column selector and update session state for dynamic aggregation
    def toggle_col(name):
        st.session_state['selected_cols'][name] = bool(1 - st.session_state['selected_cols'][name])

    with st.popover(f'*Showing {sum(st.session_state['selected_cols'].values())} / {len(st.session_state['selected_cols'])} columns*'):
        for column in data.columns:
            st.session_state['selected_cols'][column] = st.checkbox(column, value = True, on_change = toggle_col, args = [column])
    
    return selected

def charts_for_cols(data:pd.DataFrame, cols:Sequence[str]|None, type:Sequence[Literal['hist', 'box', '']], container_height:int = 800, single_chart_height:int =300):
    with st.container(border = True, height=container_height, gap=None):
        for pos, column in enumerate(cols):
            if pos != 0:
                st.divider()

            st.markdown(f'##### &ensp; **{column}**')

            for i in range(0, len(type), 2):
                chart_l, chart_r = st.columns(2, gap = 'large')
                with chart_l:
                    _draw_chart(data, column, type[2*i], nbins=20)
                if 2*i + 1 >= len(type):
                    break
                with chart_r:
                    _draw_chart(data, column, type[2*i + 1])

def _draw_chart(data:pd.DataFrame, column:str, type:str, nbins:int = 20, chart_height:int = 300):
    match type:
        case 'hist':
            fig = px.histogram(data[column], column, nbins=nbins)
        case 'box':
            fig = px.box(data[column])
        case _:
            return
    fig.update_layout({'height':chart_height})
    st.plotly_chart(fig)

def plot_cv_dist_with_ci(scores:pd.DataFrame, xaxis_title="Accuracy", bins=20):
    fig = go.Figure()

    n_column = len(scores.columns)
    for i, column in enumerate(scores.columns):
        #make color
        r, g, b = hsv_to_rgb(i/n_column + 1/n_column*.25, .7, 1)
        r, g, b = int(r*255), int(g*255), int(b*255)
        color = f'#{r:2x}{g:2x}{b:2x}'

        col_scores = np.array(scores[column]).astype(float)
        n = len(col_scores)
        mean = col_scores.mean()
        std = col_scores.std()
        se = std / np.sqrt(n)
        ci_half = t.ppf(0.975, df=n-1) * se
        ci_low, ci_high = mean - ci_half, mean + ci_half

        # KDE for smooth density curve
        kde = gaussian_kde(col_scores)
        x_min, x_max = col_scores.min(), col_scores.max()
        pad = 0.02 * (x_max - x_min if x_max > x_min else 1.0)
        xs = np.linspace(x_min - pad, x_max + pad, 400)
        ys = kde(xs)

        # Histogram (density normalized)
        fig.add_trace(go.Histogram(
            x=col_scores, nbinsx=bins, histnorm='probability density',
            name=column, marker_color = color, opacity=0.55
        ))

        # KDE line
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name=column, line={'color':color}))

        # 95% CI shaded region for the mean
        fig.add_vrect(x0=ci_low, x1=ci_high, line_width=0, fillcolor=color, opacity=0.3)

        # Mean line
        fig.add_vline(x=mean, line_dash="dash", line_width=2, fillcolor = color)

    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title='Density',
        bargap=0.05,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig
