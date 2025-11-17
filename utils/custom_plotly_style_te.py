# Custom TE template for Plotly (Version 1.2)
# Please do not use this template for any other purposes/projects without permission :(

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import pandas as pd

te_style_template = {
    "layout": {
        # == FONTS ==
        "font": {
            "family": "Econ Sans Condensed, Roboto Condensed, Arial Narrow, sans-serif",
            "size": 14,
            "color": "#38545f",
        },
        "title": {
            "font": {"size": 20},
            "x": 0.01,  # 1% from the left edge
            "y": 0.96,  # 4% from the top edge
            "xanchor": "left",  # Anchor the title's LEFT side to the x coordinate
            "yanchor": "top",  # Anchor the title's TOP side to the y coordinate
            "xref": "container",  # Use the entire figure canvas for x
            "yref": "container",  # Use the entire figure canvas for y
        },
        "legend": {
            "font": {"size": 14},
            "bgcolor": "rgba(0,0,0,0)",
            "orientation": "h",  # Horizontal is best for a top-right legend
            "x": 0.95,  # <--- 95% from the left edge (far right)
            "y": 0.95,  # <--- Align with the title's y
            "xanchor": "right",  # <--- Anchor the legend's RIGHT side to the x coordinate
            "yanchor": "top",  # <--- Anchor the legend's TOP side to the y coordinate
            "xref": "container",
            "yref": "container",
        },
        # == COLORS ==
        "paper_bgcolor": "#f7fafa",
        "plot_bgcolor": "#f7fafa",
        "colorway": [
            "#0771A4",
            "#24C0D2",
            "#E7B142",
            "#963D4D",
            "#369D8E",
            "#AAB94B",
            "#AA8B96",
        ],
        "colorscale": {
            "sequential": [
                [0.0, "#86d6f7"],
                [0.5, "#388ac3"],
                [1.0, "#006092"],
            ],
            "diverging": [
                [0.0, "#c63b48"],
                [0.5, "#EFF5F5"],
                [1.0, "#0c75ab"],
            ],
        },
        # == AXIS DEFINITIONS ==
        "xaxis": {
            "title": {"font": {"size": 13}},
            "tickfont": {"size": 14},
            "gridcolor": "#D9D9D9",
            "linecolor": "#D9D9D9",
            "zerolinecolor": "#D9D9D9",
            "showgrid": False,
            "showline": True,
            "zeroline": True,
            "ticks": "outside",
        },
        "yaxis": {
            "title": {"font": {"size": 13}},
            "tickfont": {"size": 14},
            "gridcolor": "#D9D9D9",
            "linecolor": "#D9D9D9",
            "zerolinecolor": "#D9D9D9",
            "showgrid": True,
            "showline": False,
            "zeroline": True,
        },
        # == INTERACTIVITY ==
        "hovermode": "x unified",
        "hoverlabel": {
            "bgcolor": "#FFFFFF",
            "bordercolor": "#D9D9D9",
            "font": {
                "family": "Econ Sans Condensed, Roboto Condensed, Arial Narrow, sans-serif",
                "size": 13,
                "color": "#38545f",
            },
        },
        # == FIGURE DIMENSIONS ==
        "height": 600,
        # == MARGINS ==
        "margin": {
            "t": 80,
            "l": 60,
            "b": 60,
            "r": 40,
        },
    },
    # == DATA DEFAULTS FOR TRACE TYPES ==
    "data": {
        "bar": [{"marker": {"line": {"width": 0}}}],
        "scatter": [{"line": {"width": 2.5}, "marker": {"size": 8, "opacity": 0.7}}],
        "pie": [{"hole": 0.4}],
    },
}
