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
data_directory = "data/"
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

# ========= Function to center white at 1 for VAMPseq ===========
def generate_vampseq_colorscale(zmin=0, zmax=2, base_colorscale='RdBu_r'):
    colors = get_colorscale(base_colorscale)
    middle_position = (1 - zmin) / (zmax - zmin)  # Center at 1 instead of 0

    new_colorscale = []
    for loc, color in colors:
        if loc <= 0.5:
            new_loc = loc * middle_position / 0.5
        else:
            new_loc = middle_position + ((loc - 0.5) * (1 - middle_position) / 0.5)
        new_colorscale.append([new_loc, color])
    return new_colorscale

# ========= App Setup ===========
app = dash.Dash(__name__)
app.title = "VEMViewer"

# ========= Layout ===========
app.layout = html.Div([
    html.H1("Variant effects score distribution and heatmap"),
    
    html.Div([
        html.Label("Select datasets (tick to enable):", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
        dcc.Checklist(
            id="dataset-checklist",
            options=[{"label": file, "value": file} for file in data_files.keys()],
            value=[],  # Start with none selected
            inline=False,
            labelStyle={'display': 'block', 'margin': '5px 0'}
        ),
    ], style={
        'border': '1px solid #ddd',
        'padding': '15px',
        'marginBottom': '20px',
        'maxHeight': '150px',
        'overflowY': 'auto'
    }),

    html.Br(),
    
    # Search mutations section (kept before distribution plot)
    html.Label("Search mutations (e.g., S65A, C431A):"),
    dcc.Input(id='mutation-input', type='text', debounce=True, style={'width': '150px'}),
    html.Div(id='mutation-output', style={'marginTop': '10px', 'fontWeight': 'bold'}),

    html.Br(),
    
    # First dataset distribution plot
    html.Div([
        html.H3("Dataset 1 Distribution", id="dist-title-1", style={'textAlign': 'center'}),
        dcc.Graph(id="score-distribution-1"),
    ], id="dist-container-1", style={'display': 'none'}),
    
    # Second dataset distribution plot
    html.Div([
        html.H3("Dataset 2 Distribution", id="dist-title-2", style={'textAlign': 'center'}),
        dcc.Graph(id="score-distribution-2"),
    ], id="dist-container-2", style={'display': 'none'}),
    
    # Third dataset distribution plot
    html.Div([
        html.H3("Dataset 3 Distribution", id="dist-title-3", style={'textAlign': 'center'}),
        dcc.Graph(id="score-distribution-3"),
    ], id="dist-container-3", style={'display': 'none'}),

    # Min/Max controls for heatmaps
    html.Div([
        html.Div("Heatmap Colorbar Range (applies to non-VAMPseq datasets)", style={'marginBottom': '10px', 'fontWeight': 'normal'}),
        html.Div([
            html.Label("Min Score:", style={'marginRight': '10px'}),
            dcc.Input(id='zmin-input', type='number', value=-0.8, step=0.1, style={'marginRight': '20px'}),

            html.Label("Max Score:", style={'marginRight': '10px'}),
            dcc.Input(id='zmax-input', type='number', value=0.5, step=0.1),
        ], style={'marginTop': '10px', 'marginBottom': '20px'}),
    ]),

    # First dataset heatmap
    html.Div([
        html.H3("Dataset 1 Heatmap", id="heatmap-title-1", style={'textAlign': 'center'}),
        dcc.Graph(id="heatmap-1"),
    ], id="heatmap-container-1", style={'display': 'none'}),
    
    # Second dataset heatmap
    html.Div([
        html.H3("Dataset 2 Heatmap", id="heatmap-title-2", style={'textAlign': 'center'}),
        dcc.Graph(id="heatmap-2"),
    ], id="heatmap-container-2", style={'display': 'none'}),
    
    # Third dataset heatmap
    html.Div([
        html.H3("Dataset 3 Heatmap", id="heatmap-title-3", style={'textAlign': 'center'}),
        dcc.Graph(id="heatmap-3"),
    ], id="heatmap-container-3", style={'display': 'none'}),

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

# ========= Heatmap Callback for Dataset 1 ===========
@app.callback(
    Output("heatmap-1", "figure"),
    Output("heatmap-container-1", "style"),
    Output("heatmap-title-1", "children"),
    Input("dataset-checklist", "value"),
    Input("zmin-input", "value"),
    Input("zmax-input", "value")
)
def update_heatmap_1(selected_files, zmin, zmax):
    # Get first selected file
    file = selected_files[0] if selected_files and len(selected_files) > 0 else None
    
    # If no file selected, hide container
    if file is None:
        return go.Figure(), {'display': 'none'}, "Dataset 1 Heatmap"
    
    # Check if it's a VAMPseq dataset
    is_vampseq = file.upper().startswith('VAMPSEQ')
    
    # For VAMPseq, use fixed range 0-2 with white at 1
    if is_vampseq:
        zmin_val = 0
        zmax_val = 2
        colorscale = generate_vampseq_colorscale(zmin_val, zmax_val)
        tickvals = [zmin_val, 1, zmax_val]  # Center tick at 1 for WT
    else:
        # Use user-provided values for non-VAMPseq
        if zmin is None or zmax is None:
            return go.Figure(), {'display': 'none'}, f"Heatmap: {file}"
        
        try:
            zmin_val = float(zmin)
            zmax_val = float(zmax)
        except (TypeError, ValueError):
            return go.Figure(), {'display': 'none'}, f"Heatmap: {file}"
        
        colorscale = generate_centered_colorscale(zmin_val, zmax_val)
        tickvals = [zmin_val, 0, zmax_val]

    df = data_files[file]
    z_data = df.loc[:, "*":"median"].values.T
    x_data = df["position"]
    y_labels = df.loc[:, "*":"median"].columns

    hover_text = [
        [
            f"Mutation: {wt}{pos}{variant}<br>Mutation Score: {score:.2f}<br>Median Score: {median:.2f}"
            for pos, score, wt, median in zip(
                x_data,
                row,
                df["wt_aa"],
                df["median"]
            )
        ]
        for variant, row in zip(y_labels, z_data)
    ]

    heatmap_fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_data,
        y=y_labels,
        colorscale=colorscale,
        zmin=zmin_val,
        zmax=zmax_val,
        colorbar=dict(
            title=dict(
                text="variant score (log2)",
                side="right",
                font=dict(size=12)
            ),
            tickvals=tickvals
        ),
        hoverongaps=False,
        text=hover_text,
        hovertemplate="%{text}"
    ))

    title_suffix = "" if is_vampseq else ""
    heatmap_fig.update_layout(
        title=f"{file}{title_suffix}",
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

    return heatmap_fig, {'display': 'block'}, f"Heatmap: {file}{title_suffix}"

# ========= Heatmap Callback for Dataset 2 ===========
@app.callback(
    Output("heatmap-2", "figure"),
    Output("heatmap-container-2", "style"),
    Output("heatmap-title-2", "children"),
    Input("dataset-checklist", "value"),
    Input("zmin-input", "value"),
    Input("zmax-input", "value")
)
def update_heatmap_2(selected_files, zmin, zmax):
    # Get second selected file
    file = selected_files[1] if selected_files and len(selected_files) > 1 else None
    
    # If no file selected, hide container
    if file is None:
        return go.Figure(), {'display': 'none'}, "Dataset 2 Heatmap"
    
    # Check if it's a VAMPseq dataset
    is_vampseq = file.upper().startswith('VAMPSEQ')
    
    # For VAMPseq, use fixed range 0-2 with white at 1
    if is_vampseq:
        zmin_val = 0
        zmax_val = 2
        colorscale = generate_vampseq_colorscale(zmin_val, zmax_val)
        tickvals = [zmin_val, 1, zmax_val]  # Center tick at 1 for WT
    else:
        # Use user-provided values for non-VAMPseq
        if zmin is None or zmax is None:
            return go.Figure(), {'display': 'none'}, f"Heatmap: {file}"
        
        try:
            zmin_val = float(zmin)
            zmax_val = float(zmax)
        except (TypeError, ValueError):
            return go.Figure(), {'display': 'none'}, f"Heatmap: {file}"
        
        colorscale = generate_centered_colorscale(zmin_val, zmax_val)
        tickvals = [zmin_val, 0, zmax_val]

    df = data_files[file]
    z_data = df.loc[:, "*":"median"].values.T
    x_data = df["position"]
    y_labels = df.loc[:, "*":"median"].columns

    hover_text = [
        [
            f"Mutation: {wt}{pos}{variant}<br>Mutation Score: {score:.2f}<br>Median Score: {median:.2f}"
            for pos, score, wt, median in zip(
                x_data,
                row,
                df["wt_aa"],
                df["median"]
            )
        ]
        for variant, row in zip(y_labels, z_data)
    ]

    heatmap_fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_data,
        y=y_labels,
        colorscale=colorscale,
        zmin=zmin_val,
        zmax=zmax_val,
        colorbar=dict(
            title=dict(
                text="variant score (log2)",
                side="right",
                font=dict(size=12)
            ),
            tickvals=tickvals
        ),
        hoverongaps=False,
        text=hover_text,
        hovertemplate="%{text}"
    ))

    title_suffix = "" if is_vampseq else ""
    heatmap_fig.update_layout(
        title=f"{file}{title_suffix}",
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

    return heatmap_fig, {'display': 'block'}, f"Heatmap: {file}{title_suffix}"

# ========= Heatmap Callback for Dataset 3 ===========
@app.callback(
    Output("heatmap-3", "figure"),
    Output("heatmap-container-3", "style"),
    Output("heatmap-title-3", "children"),
    Input("dataset-checklist", "value"),
    Input("zmin-input", "value"),
    Input("zmax-input", "value")
)
def update_heatmap_3(selected_files, zmin, zmax):
    # Get third selected file
    file = selected_files[2] if selected_files and len(selected_files) > 2 else None
    
    # If no file selected, hide container
    if file is None:
        return go.Figure(), {'display': 'none'}, "Dataset 3 Heatmap"
    
    # Check if it's a VAMPseq dataset
    is_vampseq = file.upper().startswith('VAMPSEQ')
    
    # For VAMPseq, use fixed range 0-2 with white at 1
    if is_vampseq:
        zmin_val = 0
        zmax_val = 2
        colorscale = generate_vampseq_colorscale(zmin_val, zmax_val)
        tickvals = [zmin_val, 1, zmax_val]  # Center tick at 1 for WT
    else:
        # Use user-provided values for non-VAMPseq
        if zmin is None or zmax is None:
            return go.Figure(), {'display': 'none'}, f"Heatmap: {file}"
        
        try:
            zmin_val = float(zmin)
            zmax_val = float(zmax)
        except (TypeError, ValueError):
            return go.Figure(), {'display': 'none'}, f"Heatmap: {file}"
        
        colorscale = generate_centered_colorscale(zmin_val, zmax_val)
        tickvals = [zmin_val, 0, zmax_val]

    df = data_files[file]
    z_data = df.loc[:, "*":"median"].values.T
    x_data = df["position"]
    y_labels = df.loc[:, "*":"median"].columns

    hover_text = [
        [
            f"Mutation: {wt}{pos}{variant}<br>Mutation Score: {score:.2f}<br>Median Score: {median:.2f}"
            for pos, score, wt, median in zip(
                x_data,
                row,
                df["wt_aa"],
                df["median"]
            )
        ]
        for variant, row in zip(y_labels, z_data)
    ]

    heatmap_fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_data,
        y=y_labels,
        colorscale=colorscale,
        zmin=zmin_val,
        zmax=zmax_val,
        colorbar=dict(
            title=dict(
                text="variant score (log2)",
                side="right",
                font=dict(size=12)
            ),
            tickvals=tickvals
        ),
        hoverongaps=False,
        text=hover_text,
        hovertemplate="%{text}"
    ))

    title_suffix = "" if is_vampseq else ""
    heatmap_fig.update_layout(
        title=f"{file}{title_suffix}",
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

    return heatmap_fig, {'display': 'block'}, f"Heatmap: {file}{title_suffix}"

# ========= Mutation Lookup Callback ===========
@app.callback(
    Output("mutation-output", "children"),
    Input("mutation-input", "value"),
    Input("dataset-checklist", "value")
)
def lookup_mutation(mutation_str, selected_files):
    if not mutation_str or not selected_files:
        return ""

    results = []
    
    # Process first dataset
    if len(selected_files) > 0:
        file1 = selected_files[0]
        df1 = data_files[file1]
        results.append(html.Div(f"Dataset 1: {file1}", style={'fontWeight': 'bold', 'marginTop': '10px'}))
        results.append(process_mutations_for_dataset(mutation_str, df1, file1))
    
    # Process second dataset
    if len(selected_files) > 1:
        file2 = selected_files[1]
        df2 = data_files[file2]
        results.append(html.Div(f"Dataset 2: {file2}", style={'fontWeight': 'bold', 'marginTop': '10px'}))
        results.append(process_mutations_for_dataset(mutation_str, df2, file2))
    
    # Process third dataset
    if len(selected_files) > 2:
        file3 = selected_files[2]
        df3 = data_files[file3]
        results.append(html.Div(f"Dataset 3: {file3}", style={'fontWeight': 'bold', 'marginTop': '10px'}))
        results.append(process_mutations_for_dataset(mutation_str, df3, file3))

    return html.Div(results)

def process_mutations_for_dataset(mutation_str, df, filename):
    mutation_results = []
    
    try:
        mutations = [m.strip().upper() for m in mutation_str.split(",") if m.strip()]
        for m in mutations:
            if len(m) < 3:
                mutation_results.append(f"Invalid format: '{m}'")
                continue

            wt = m[0]
            variant = m[-1]
            try:
                pos = int(m[1:-1])
            except ValueError:
                mutation_results.append(f"Invalid position in: '{m}'")
                continue

            match = df[df["position"] == pos]

            if match.empty:
                mutation_results.append(f"{m}: Position {pos} not found.")
                continue

            row = match.iloc[0]
            actual_wt = row["wt_aa"]
            if wt != actual_wt:
                mutation_results.append(f"{m}: WT amino acid mismatch, expected {actual_wt}")
                continue

            if variant not in df.columns:
                mutation_results.append(f"{m}: Variant {variant} not found.")
                continue

            score = row[variant]
            median = row["median"]
            mutation_results.append(f"{m} → Variant Score: {score:.2f}, Position {m[:-1]} Median Score: {median:.2f}")

        return html.Ul([html.Li(r) for r in mutation_results])

    except Exception as e:
        return f"Error in {filename}: {str(e)}"

# ========= Distribution Callback for Dataset 1 ===========
@app.callback(
    Output("score-distribution-1", "figure"),
    Output("dist-container-1", "style"),
    Output("dist-title-1", "children"),
    Input("dataset-checklist", "value"),
    Input("mutation-input", "value")
)
def update_distribution_1(selected_files, mutation_str):
    # Get first selected file
    file = selected_files[0] if selected_files and len(selected_files) > 0 else None
    
    if not file:
        return go.Figure(), {'display': 'none'}, "Dataset 1 Distribution"
    
    df = data_files[file]
    
    # Check if it's a VAMPseq dataset
    is_vampseq = file.upper().startswith('VAMPSEQ')

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
    title_suffix = " (VAMPseq)" if is_vampseq else ""
    fig.update_layout(
        title=f"Distribution: {file}{title_suffix}",
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
    
    # Add vertical line: at 1 for VAMPseq, at 0 for others
    if is_vampseq:
        fig.add_vline(
            x=1, 
            line_width=1, 
            line_color="black",
            annotation_text="WT",
            annotation_position="top left"
        )
    else:
        fig.add_vline(
            x=0, 
            line_width=1, 
            line_color="black",
            annotation_text="WT",
            annotation_position="top left"
        )

    return fig, {'display': 'block'}, f"Dataset 1: {file}{title_suffix}"

# ========= Distribution Callback for Dataset 2 ===========
@app.callback(
    Output("score-distribution-2", "figure"),
    Output("dist-container-2", "style"),
    Output("dist-title-2", "children"),
    Input("dataset-checklist", "value"),
    Input("mutation-input", "value")
)
def update_distribution_2(selected_files, mutation_str):
    # Get second selected file
    file = selected_files[1] if selected_files and len(selected_files) > 1 else None
    
    if not file:
        return go.Figure(), {'display': 'none'}, "Dataset 2 Distribution"
    
    df = data_files[file]
    
    # Check if it's a VAMPseq dataset
    is_vampseq = file.upper().startswith('VAMPSEQ')

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
    title_suffix = " (VAMPseq)" if is_vampseq else ""
    fig.update_layout(
        title=f"Distribution: {file}{title_suffix}",
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
    
    # Add vertical line: at 1 for VAMPseq, at 0 for others
    if is_vampseq:
        fig.add_vline(
            x=1, 
            line_width=1, 
            line_color="black",
            annotation_text="WT",
            annotation_position="top left"
        )
    else:
        fig.add_vline(
            x=0, 
            line_width=1, 
            line_color="black",
            annotation_text="WT",
            annotation_position="top left"
        )

    return fig, {'display': 'block'}, f"Dataset 2: {file}{title_suffix}"

# ========= Distribution Callback for Dataset 3 ===========
@app.callback(
    Output("score-distribution-3", "figure"),
    Output("dist-container-3", "style"),
    Output("dist-title-3", "children"),
    Input("dataset-checklist", "value"),
    Input("mutation-input", "value")
)
def update_distribution_3(selected_files, mutation_str):
    # Get third selected file
    file = selected_files[2] if selected_files and len(selected_files) > 2 else None
    
    if not file:
        return go.Figure(), {'display': 'none'}, "Dataset 3 Distribution"
    
    df = data_files[file]
    
    # Check if it's a VAMPseq dataset
    is_vampseq = file.upper().startswith('VAMPSEQ')

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
    title_suffix = " (VAMPseq)" if is_vampseq else ""
    fig.update_layout(
        title=f"Distribution: {file}{title_suffix}",
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
    
    # Add vertical line: at 1 for VAMPseq, at 0 for others
    if is_vampseq:
        fig.add_vline(
            x=1, 
            line_width=1, 
            line_color="black",
            annotation_text="WT",
            annotation_position="top left"
        )
    else:
        fig.add_vline(
            x=0, 
            line_width=1, 
            line_color="black",
            annotation_text="WT",
            annotation_position="top left"
        )

    return fig, {'display': 'block'}, f"Dataset 3: {file}{title_suffix}"


# ========= Run ===========
if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", 8051)), host="0.0.0.0")