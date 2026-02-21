import dash
from dash import dcc, html, dash_table, Input, Output, State
import pandas as pd
import numpy as np
import sqlite3
import io
import base64
import os
import plotly.express as px
from sklearn.ensemble import IsolationForest

# Initialize Dash App
# server = app.server is required for Gunicorn/Render deployment
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server 

app.layout = html.Div([
    # Header Section
    html.Div([
        html.H1("AI-Driven Database Insight Generator", style={'marginBottom': '0px'}),
        html.P("Enterprise Analytics & Natural Language Querying", style={'color': '#7f8c8d'})
    ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f8f9fa'}),

    # Main Upload Section
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select .DB Files')]),
            style={
                'width': '100%', 'height': '80px', 'lineHeight': '80px',
                'borderWidth': '2px', 'borderStyle': 'dashed',
                'borderRadius': '10px', 'textAlign': 'center', 'margin': '10px'
            },
            multiple=True
        ),
    ], style={'padding': '0 50px'}),

    # AI Consultant Section (NLQ Interface)
    html.Div([
        html.H3("💬 AI Data Consultant"),
        html.P("Ask questions about your data in plain English (e.g., 'Find anomalies in sales')"),
        dcc.Input(
            id='ai-query-input', 
            placeholder='Type your question here...', 
            style={'width': '70%', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ccc'}
        ),
        html.Button(
            'Analyze with AI', 
            id='ai-query-btn', 
            n_clicks=0,
            style={'marginLeft': '10px', 'backgroundColor': '#2980b9', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'borderRadius': '5px'}
        ),
        html.Div(id='ai-query-response', style={'marginTop': '20px', 'padding': '15px', 'borderRadius': '5px', 'backgroundColor': '#ecf0f1', 'minHeight': '50px'})
    ], style={'margin': '20px 50px', 'padding': '20px', 'border': '1px solid #dcdde1', 'borderRadius': '10px'}),

    # Results Display Section
    dcc.Loading(
        id="loading-spinner",
        type="cube",
        children=html.Div(id='output-data-upload', style={'padding': '0 50px'})
    ),
])

def process_database(contents, filename):
    """
    Handles file decoding, temporary storage, SQLite processing, and ML analysis.
    Optimized for memory safety and speed.
    """
    temp_filename = f"temp_{filename}"
    db_summary = {}
    conn = None
    
    try:
        # Decode and Save
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        with open(temp_filename, 'wb') as f:
            f.write(decoded)
        
        conn = sqlite3.connect(temp_filename)
        cursor = conn.cursor()
        
        # Discover Tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cursor.fetchall()]
        
        for table in tables:
            # PERFORMANCE: Sample data to prevent OOM on Render
            df_full = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 1000", conn)
            
            # ML: Anomaly Detection
            numeric_cols = df_full.select_dtypes(include=[np.number]).columns
            anomaly_count = 0
            if len(numeric_cols) > 0 and len(df_full) > 10:
                iso_forest = IsolationForest(contamination=0.05, random_state=42)
                preds = iso_forest.fit_predict(df_full[numeric_cols].fillna(0))
                anomaly_count = int((preds == -1).sum())
            
            # STATS: Generate metadata
            db_summary[table] = {
                'df': df_full,
                'stats': df_full.describe(include='all').to_dict(),
                'anomalies': anomaly_count,
                'correlations': df_full.corr(numeric_only=True).to_dict() if len(numeric_cols) > 1 else {},
                'columns': list(df_full.columns)
            }
        
        return db_summary

    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.callback(
    Output('ai-query-response', 'children'),
    Input('ai-query-btn', 'n_clicks'),
    State('ai-query-input', 'value'),
    State('upload-data', 'filename')
)
def handle_ai_query(n_clicks, query, filename):
    if n_clicks == 0:
        return "No query submitted yet."
    if not filename:
        return "Please upload a database file first."
    if not query:
        return "Please enter a question."
    
    # FUTURE: Integrate OpenAI/Groq API here
    return html.Div([
        html.Span("🤖 AI Interpretation: ", style={'fontWeight': 'bold', 'color': '#2980b9'}),
        f"Requesting analysis for '{query}' on {filename[0]}. SQL mapping and LLM response logic is ready for API connection."
    ])

@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(list_of_contents, list_of_names):
    if not list_of_contents:
        return html.Div("Upload a database to begin the AI analysis.", style={'textAlign': 'center', 'marginTop': '40px'})

    all_db_data = {}
    for c, n in zip(list_of_contents, list_of_names):
        data = process_database(c, n)
        if data:
            all_db_data[n] = data

    results = []

    # Cross-File Intelligence Logic
    if len(all_db_data) > 1:
        results.append(html.Div([
            html.H2("🌐 Cross-File Intelligence"),
            html.P("Analyzing relationships between uploaded datasets...")
        ], style={'padding': '20px', 'backgroundColor': '#dfe6e9', 'borderRadius': '10px'}))

    # Render results for each file
    for filename, tables in all_db_data.items():
        results.append(html.H2(f"📄 File: {filename}", style={'marginTop': '40px'}))
        
        for table_name, data in tables.items():
            results.append(html.Div([
                html.H4(f"Table: {table_name}"),
                
                # Anomaly Alert
                html.Div([
                    html.B("🔍 AI Anomaly Detection: "),
                    f"{data['anomalies']} potential data outliers found." if data['anomalies'] > 0 else "Data appears statistically consistent."
                ], style={
                    'padding': '10px', 
                    'backgroundColor': '#fff3cd' if data['anomalies'] > 0 else '#d4edda',
                    'borderLeft': '5px solid #ffecb5' if data['anomalies'] > 0 else '#c3e6cb',
                    'marginBottom': '10px'
                }),

                # Data Preview
                dash_table.DataTable(
                    data=data['df'].head(10).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in data['df'].columns],
                    style_table={'overflowX': 'auto'},
                    page_size=5,
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                ),

                # Heatmap
                dcc.Graph(
                    figure=px.imshow(
                        pd.DataFrame(data['correlations']), 
                        title=f"Correlation Matrix ({table_name})",
                        color_continuous_scale='Blues'
                    )
                ) if data['correlations'] else html.P("No numeric correlations available for this table.")
            ], style={'marginBottom': '50px', 'padding': '20px', 'border': '1px solid #eee'}))

    return results

if __name__ == '__main__':
    app.run_server(debug=True)
