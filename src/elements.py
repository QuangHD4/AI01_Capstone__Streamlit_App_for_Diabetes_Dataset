import re, io
from typing import Sequence, Literal, List, Dict, Any, Tuple
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

def df_info_table(data:pd.DataFrame):
    # Capture df.info() output
    buffer = io.StringIO()
    data.info(buf=buffer)
    info_str = buffer.getvalue()

    # Parse the info_str into a structured DataFrame
    lines = info_str.splitlines()

    # Find the lines that contain column info
    col_lines = []
    for line in lines:
        # Matches lines like: ' 0   foo   object  ...'
        if re.match(r'\s*\d+\s+', line):
            col_lines.append(line.strip())

    # Split each line into parts
    parsed = []
    for line in col_lines:
        parts = re.split(r'\s{2,}', line)  # split on 2+ spaces
        parsed.append(parts)

    # Create DataFrame from parsed info
    info_df = pd.DataFrame(parsed, columns=['Index', 'Column', 'Non-Null Count', 'Dtype']).set_index('Index')

    # Display in Streamlit
    st.dataframe(info_df)

def build_sequence(options:List[str], follow_up_config: Dict[str,Dict[str,Any]], item_caption:str='') -> Tuple[List[str], Dict[int, str]]:
    if "transforms" not in st.session_state:
        st.session_state.transforms = []
    if "follow_ups" not in st.session_state:
        st.session_state.follow_ups = {}

    total_boxes = len(st.session_state.transforms) + 1

    for box_index in range(total_boxes):
        key = f"select_{box_index}"
        default_index = None
        if box_index < len(st.session_state.transforms):
            default_index = options.index(st.session_state.transforms[box_index])
        selected = st.selectbox(
            f"{item_caption} #{box_index + 1}",
            options,
            index=default_index,
            key=key,
            label_visibility='collapsed'
        )

        # new selection
        if box_index == len(st.session_state.transforms):
            if selected:
                st.session_state.transforms.append(selected)
                st.rerun()
        else:
            st.session_state.transforms[box_index] = selected

        # Render follow-up widget if applicable
        if selected in follow_up_config:
            config = follow_up_config[selected]
            follow_key = f"follow_{box_index}"
            if config["type"] == "slider":
                value = st.slider(
                    config["label"],
                    min_value=config["min"],
                    max_value=config["max"],
                    step=config["step"],
                    key=follow_key,
                    value=config.get('default'),
                    label_visibility=config.get('label_visibility', 'visible')
                )
            elif config["type"] == "number_input":
                value = st.number_input(config["label"], key=follow_key)
            else:
                value = None

            st.session_state.follow_ups[box_index] = value
        else:
            st.session_state.follow_ups[box_index] = None

    return st.session_state.transforms, st.session_state.follow_ups

def clear_selections():
    st.session_state.transforms = []
    st.session_state.follow_ups = {}
    st.rerun()