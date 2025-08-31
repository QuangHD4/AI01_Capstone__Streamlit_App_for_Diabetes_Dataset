from typing import Sequence, Literal
from colorsys import hsv_to_rgb

import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

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

def plot_charts(data:pd.DataFrame, cols:Sequence[str]|None, type:Sequence[Literal['hist', 'box', '']], container_height:int = 800, single_chart_height:int =300):
    with st.container(border = True, height=container_height, gap=None):
        for pos, column in enumerate(cols):
            if pos != 0:
                st.divider()

            st.markdown(f'##### &ensp; **{column}**')

            for i in range(0, len(type), 2):
                chart_l, chart_r = st.columns(2, gap = 'large')
                with chart_l:
                    plot_chart(data, column, type[2*i], nbins=20)
                if 2*i + 1 >= len(type):
                    break
                with chart_r:
                    plot_chart(data, column, type[2*i + 1])

def plot_chart(data:pd.DataFrame, columns:Sequence[str], type:str, nbins:int = 20, chart_height:int = 300):
    match type:
        case 'hist':
            fig = px.histogram(data[columns], x=columns, nbins=nbins)
        case 'box':
            fig = px.box(data[columns])
        case _:
            return
    fig.update_layout({'height':chart_height})
    st.plotly_chart(fig)

def multicol_hist_with_kde(data:pd.DataFrame|pd.Series, nbins=20, xaxis_title:str = None):
    fig = go.Figure()

    for i, column in enumerate(data.columns):
        #make rainbow
        r, g, b = hsv_to_rgb(i/len(data.columns) + 1/len(data.columns)*.25, .7, 1)
        r, g, b = int(r*255), int(g*255), int(b*255)
        color = f'#{r:2x}{g:2x}{b:2x}'

        col_df = np.array(data[column]).astype(float)

        kde = gaussian_kde(col_df)
        x_min, x_max = col_df.min(), col_df.max()
        pad = 0.02 * (x_max - x_min if x_max > x_min else 1.0)
        xs = np.linspace(x_min - pad, x_max + pad, 400)
        ys = kde(xs)

        fig.add_trace(go.Histogram(
            x=col_df, nbinsx=nbins, histnorm='probability density',
            name=column, marker_color = color, opacity=0.55,
            legendgroup=column
        ))

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines', line={'color':color}, 
            legendgroup=column, showlegend=False
        ))

    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title='Density',
        bargap=0.05,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig)
    return fig

def pairwise_scatter_plots_varying_radius(data:pd.DataFrame, ):
    '''
    For simple pairwise scatter plot, use px.scatter instead
    '''
    for row_no, y_model in enumerate(selected_models):
        for col_no, x_model in enumerate(selected_models):
            if row_no == col_no:
                continue
            counts = kfoldcv_scores.groupby([x_model, y_model]).size().reset_index(name='size')
            pair_kfoldcv_score_fig.add_trace(
                go.Scatter(
                    x=counts[x_model], y=counts[y_model],
                    mode='markers', 
                    marker=dict(
                        size=counts['size']*6/len(selected_models),
                        sizemode='area',
                    ),
                ),
                row=row_no+1, col=col_no+1,
            )
            pair_kfoldcv_score_fig.add_shape(
                type="line",
                x0=min(kfoldcv_scores[y_model].min(), kfoldcv_scores[x_model].min()),
                y0=min(kfoldcv_scores[y_model].min(), kfoldcv_scores[x_model].min()),
                x1=max(kfoldcv_scores[y_model].max(), kfoldcv_scores[x_model].max()),
                y1=max(kfoldcv_scores[y_model].max(), kfoldcv_scores[x_model].max()),
                line=dict(
                    color="yellow",
                    width=1.5,
                    # dash="dash"
                ),
                row=row_no+1, col=col_no+1,
            )
    pair_kfoldcv_score_fig.update_layout(height=800, showlegend=False)
    st.plotly_chart(pair_kfoldcv_score_fig)
