# Copyright (c) 2025 Vahid Aslanzadeh
# All rights reserved.
#
# This software is proprietary and confidential. No part of this code
# may be copied, modified, distributed, or used in any form without
# explicit written permission from the author.
#
# Author: Vahid Aslanzadeh
# Contact: vaslanzadeh@gmail.com

import os
import dash
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from plotly.colors import get_colorscale


# ========= Load CSVs ===========
data_directory = "parkin_app/data/"
data_files = {
    file: pd.read_csv(os.path.join(data_directory, file))
    for file in os.listdir(data_directory)
    if file.endswith(".csv")
}

# ========= Function to center zero (white) ===========
def generate_centered_colorscale(zmin, zmax, base_colorscale='RdBu_r'):
    colors = get_colorscale(base_colorscale)
    zero_position = (0 - zmin) / (zmax - zmin)

    new_colorscale = []
    for loc, color in colors:
        if loc <= 0.5:
            new_loc = loc * zero_position / 0.5
        else:
            new_loc = zero_position + ((loc - 0.5) * (1 - zero_position) / 0.5)
        new_colorscale.append([new_loc, color])
    return new_colorscale

# ========= App Setup ===========
app = dash.Dash(__name__)
app.title = "VEMViewer"

# ========= Layout ===========
app.layout = html.Div([
    html.H1("Variant effects score distribution and heatmap"),

    html.Label("Select a variant effect score dataset (csv format):"),
    dcc.Dropdown(
        id="file-dropdown",
        options=[{"label": file, "value": file} for file in data_files.keys()],
        multi=False
    ),

    html.Br(),
    
    # Search mutations section (kept before distribution plot)
    html.Label("Search mutations (e.g., S65A, C431A):"),
    dcc.Input(id='mutation-input', type='text', debounce=True, style={'width': '150px'}),
    html.Div(id='mutation-output', style={'marginTop': '10px', 'fontWeight': 'bold'}),

    html.Br(),
    dcc.Graph(id="score-distribution"),
    
    # Min/Max controls moved after distribution plot
    html.Div([
        html.Div("Heatmap Colorbar Range", style={'marginBottom': '10px', 'fontWeight': 'normal'}),
        html.Div([
            html.Label("Min Score:", style={'marginRight': '10px'}),
            dcc.Input(id='zmin-input', type='number', value=-0.8, step=0.1, style={'marginRight': '20px'}),

            html.Label("Max Score:", style={'marginRight': '10px'}),
            dcc.Input(id='zmax-input', type='number', value=0.5, step=0.1),
        ], style={'marginTop': '10px', 'marginBottom': '20px'}),
    ]),

    dcc.Graph(id="heatmap"),

    # Footer
    html.Div(
        "© Vahid Aslanzadeh – All rights reserved",
        style={
            "textAlign": "center",
            "marginTop": "50px",
            "marginBottom": "20px",
            "fontSize": "12px",
            "color": "#888"
        }
    )
])
# ========= Heatmap Callback ===========
@app.callback(
    Output("heatmap", "figure"),
    Input("file-dropdown", "value"),
    Input("zmin-input", "value"),
    Input("zmax-input", "value")
)
def update_heatmap(file, zmin, zmax):
    # Guard clause for missing input
    if file is None or zmin is None or zmax is None:
        return go.Figure()

    # Make sure values are float
    try:
        zmin = float(zmin)
        zmax = float(zmax)
    except (TypeError, ValueError):
        return go.Figure()

    df = data_files[file]
    z_data = df.loc[:, "*":"Y"].values.T
    x_data = df["position"]
    y_labels = df.loc[:, "*":"Y"].columns

    hover_text = [
        [
            f"Mutation: {wt}{pos}{variant}<br>Mutation Score: {score:.2f}<br>Median Score: {median:.2f}"
            for pos, score, wt, median in zip(
                x_data,
                row,
                df["wt_aa"],
                df["median_score"]
            )
        ]
        for variant, row in zip(y_labels, z_data)
    ]

    colorscale = generate_centered_colorscale(zmin, zmax)

    heatmap_fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_data,
        y=y_labels,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(
            title=dict(
                text="variant score (log2)",
                side="right",
                font=dict(size=12)
            ),
            tickvals=[zmin, 0, zmax]
        ),
        hoverongaps=False,
        text=hover_text,
        hovertemplate="%{text}"
    ))

    heatmap_fig.update_layout(
        title=file,
        xaxis=dict(title="Position", showgrid=False),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(y_labels))),
            ticktext=y_labels,
            automargin=True,
            fixedrange=True,
            showgrid=False
        ),
        plot_bgcolor="grey",
    )

    return heatmap_fig
# ========= Mutation Lookup Callback ===========
@app.callback(
    Output("mutation-output", "children"),
    Input("mutation-input", "value"),
    State("file-dropdown", "value")
)
def lookup_mutation(mutation_str, file):
    if not mutation_str or file is None:
        return ""

    df = data_files[file]
    results = []

    try:
        mutations = [m.strip().upper() for m in mutation_str.split(",") if m.strip()]
        for m in mutations:
            if len(m) < 3:
                results.append(f"Invalid format: '{m}'")
                continue

            wt = m[0]
            variant = m[-1]
            try:
                pos = int(m[1:-1])
            except ValueError:
                results.append(f"Invalid position in: '{m}'")
                continue

            match = df[df["position"] == pos]

            if match.empty:
                results.append(f"{m}: Position {pos} not found.")
                continue

            row = match.iloc[0]
            actual_wt = row["wt_aa"]
            if wt != actual_wt:
                results.append(f"{m}: WT amino acid mismatch, expected {actual_wt}")
                continue

            if variant not in df.columns:
                results.append(f"{m}: Variant {variant} not found.")
                continue

            score = row[variant]
            median = row["median_score"]
            results.append(f"{m} → Variant Score: {score:.2f}, Position {m[:-1]} Median Score: {median:.2f}")

        return html.Ul([html.Li(r) for r in results])

    except Exception as e:
        return f"Error: {str(e)}"
@app.callback(
    Output("score-distribution", "figure"),
    Input("file-dropdown", "value"),
    Input("mutation-input", "value")
)
def update_distribution(file, mutation_str):
    if not file:
        return go.Figure()

    df = data_files[file]

    # Collect all variant scores and stop codon scores separately
    variant_cols = df.loc[:, "*":"Y"].columns
    all_scores = df[variant_cols].values.flatten()
    stop_scores = df["*"].values.flatten()

    # Clean: Remove NaNs
    all_scores = all_scores[~pd.isnull(all_scores)]
    stop_scores = stop_scores[~pd.isnull(stop_scores)]

    # Prepare figure
    fig = go.Figure()

    # Add histogram of all scores
    fig.add_trace(go.Histogram(
        x=all_scores,
        nbinsx=100,
        name='All Variant Scores',
        marker=dict(color='lightblue'),
        opacity=0.75
    ))

    # Add histogram of stop codon scores (overlaid)
    fig.add_trace(go.Histogram(
        x=stop_scores,
        nbinsx=100,
        name='Stop Codon Scores',
        marker=dict(color='red'),
        opacity=0.5
    ))

    # Overlay lines for mutation(s)

    # Overlay lines for mutation(s)
    if mutation_str:
        mutations = [m.strip().upper() for m in mutation_str.split(",") if m.strip()]
    
        for i, m in enumerate(mutations):
            if len(m) < 3:
                continue

            wt = m[0]
            variant = m[-1]
            try:
                pos = int(m[1:-1])
            except:
                continue

            match = df[df["position"] == pos]
            if match.empty:
                continue

            row = match.iloc[0]
            if wt != row["wt_aa"] or variant not in df.columns:
                continue

            score = row[variant]
        
            # Calculate y position as percentage from top (90%, 80%, 70%, etc.)
            y_position_percent = 1 - (i * 0.1)  #100%, 90%, 80%, 70%, etc.
        
            fig.add_vline(
                x=score,
                line=dict(color="green", width=2, dash="dash"),
                annotation_text=m,
                annotation_position="top right",
                annotation_y=y_position_percent,
                annotation_yref="paper"  # Use paper coordinates (0-1)
        )

    # Update layout with x-axis tick settings
    fig.update_layout(
        title="Distribution of Variant Scores (Red = Stop Codons)",
        xaxis_title="Variant Score",
        yaxis_title="Count",
        bargap=0.05,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=300,
        xaxis=dict(
            tickmode='linear',
            tick0=-1.0,  # Starting point (adjust based on your data range)
            dtick=0.1,   # Interval between ticks
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGrey'
        ),
        plot_bgcolor='white',  # Set plot background to white
        paper_bgcolor='white', # Set surrounding area background to white
        
    )
    
    # Add vertical line at 0 in black
    fig.add_vline(
        x=0, 
        line_width=1, 
        line_color="black"
    )

    return fig
# ========= Run ===========s
if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", 8051)), host="0.0.0.0")
