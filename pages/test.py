import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# Sample options to choose from
options = ["Apple", "Banana", "Cherry", "Date", "Elderberry", "Fig"]

follow_up_config = {
    "Elderberry": {
        "type": "slider",
        "label": "Rate your love for Elderberry",
        "min": 0,
        "max": 10,
        "step": 1
    },
    "Fig": {
        "type": "text_input",
        "label": "Why do you like Fig?"
    }
}

# Function to render select boxes dynamically
def render_select_boxes():
    if "transforms" not in st.session_state:
        st.session_state.transforms = []
    if "follow_ups" not in st.session_state:
        st.session_state.follow_ups = {}

    # Remove any "-- None --" transforms
    st.session_state.transforms = [s for s in st.session_state.transforms if s != "-- None --"]

    total_boxes = len(st.session_state.transforms) + 1
    rows = (total_boxes + 3) // 4

    box_index = 0
    new_transforms = []
    follow_ups = {}

    for row in range(rows):
        cols = st.columns(4)
        for col in cols:
            if box_index >= total_boxes:
                break

            key = f"select_{box_index}"
            default_index = None
            if box_index < len(st.session_state.transforms):
                try:
                    default_index = options.index(st.session_state.transforms[box_index])
                except ValueError:
                    default_index = None

            selected = col.selectbox(
                f"Item #{box_index + 1}",
                options,
                index=default_index,
                key=key
            )

            if selected == "-- None --":
                # Skip adding this selection
                pass
            else:
                new_transforms.append(selected)

                # Render follow-up widget if applicable
                if selected in follow_up_config:
                    config = follow_up_config[selected]
                    follow_key = f"follow_{box_index}"
                    if config["type"] == "slider":
                        value = col.slider(
                            config["label"],
                            min_value=config["min"],
                            max_value=config["max"],
                            step=config["step"],
                            key=follow_key
                        )
                    elif config["type"] == "text_input":
                        value = col.text_input(config["label"], key=follow_key)
                    else:
                        value = None
                    follow_ups[box_index] = value

            box_index += 1

    # Update session state
    st.session_state.transforms = new_transforms

    return new_transforms, follow_ups

# Render the select boxes
seq, fu = render_select_boxes()
st.write(seq)
st.write(fu)

st.divider()

