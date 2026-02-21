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
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server # Required for Render (Gunicorn)

app.layout = html.Div([
    html.H1("AI-Driven Database Insight Generator (v2.0)", style={'textAlign': 'center'}),
    html.P("Optimized for Performance & Scalability", style={'textAlign': 'center', 'color': '#666'}),
    
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select .DB Files')]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
        },
        multiple=True
    ),
    
    # Loading Spinner to improve UX during processing
    dcc.Loading(
        id="loading-1",
        type="default",
        children=html.Div(id='output-data-upload')
    ),
])

def process_database(contents, filename):
    temp_filename = f"temp_{filename}"
    db_summary = {}
    conn = None
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # 1. Atomic Write: Save file to disk
        with open(temp_filename, 'wb') as f:
            f.write(decoded)
        
        conn = sqlite3.connect(temp_filename)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cursor.fetchall()]
        
        for table in tables:
            # OPTIMIZATION: Limit initial read to 1000 rows for the preview to prevent OOM
            df_preview = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 1000", conn)
            
            # 2. AI Anomaly Detection (Isolation Forest)
            numeric_cols = df_preview.select_dtypes(include=[np.number]).columns
            anomalies = 0
            if len(numeric_cols) > 0 and len(df_preview) > 10:
                # Contamination set to 5% for standard sensitivity
                model = IsolationForest(contamination=0.05, random_state=42)
                preds = model.fit_predict(df_preview[numeric_cols].fillna(0))
                anomalies = int((preds == -1).sum())
            
            # 3. Statistical Profiling (using the preview sample for speed)
            stats = df_preview.describe(include='all').to_dict()
            corr = df_preview.corr(numeric_only=True).to_dict() if len(numeric_cols) > 1 else {}

            db_summary[table] = {
                'df': df_preview,
                'stats': stats,
                'anomalies': anomalies,
                'correlations': corr,
                'columns': list(df_preview.columns)
            }
        
        return db_summary

    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None
    
    finally:
        # 4. CRITICAL: Cleanup to prevent Render disk overflow
        if conn:
            conn.close()
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))
def update_output(list_of_contents, list_of_names):
    if not list_of_contents:
        return html.Div("No files uploaded yet.", style={'textAlign': 'center', 'padding': '20px'})

    all_db_data = {}
    for c, n in zip(list_of_contents, list_of_names):
        data = process_database(c, n)
        if data:
            all_db_data[n] = data

    if not all_db_data:
        return html.Div("Error: Could not process the uploaded files. Ensure they are valid .db files.", style={'color': 'red'})

    children = []

    # File-Level Render Logic
    for filename, tables in all_db_data.items():
        children.append(html.Hr())
        children.append(html.H3(f"📁 Analysis: {filename}", style={'color': '#2c3e50'}))
        
        for table_name, data in tables.items():
            children.append(html.H4(f"Table: {table_name}", style={'marginLeft': '20px'}))
            
            # AI Insight Badge
            if data['anomalies'] > 0:
                children.append(html.Div([
                    html.B("🔍 AI Insight: "),
                    f"Detected {data['anomalies']} statistical anomalies in this dataset sample."
                ], style={'backgroundColor': '#fff3cd', 'padding': '10px', 'borderRadius': '5px', 'borderLeft': '5px solid #ffecb5'}))

            # Data Table Preview
            children.append(dash_table.DataTable(
                data=data['df'].to_dict('records'),
                columns=[{"name": i, "id": i} for i in data['df'].columns],
                style_table={'overflowX': 'auto', 'marginTop': '10px'},
                page_size=10
            ))
            
            # Visualization
            if data['correlations']:
                fig = px.imshow(pd.DataFrame(data['correlations']), 
                               title=f"Correlation Matrix: {table_name}",
                               color_continuous_scale='RdBu_r')
                children.append(dcc.Graph(figure=fig))

    # Cross-File Intelligence Logic
    if len(all_db_data) > 1:
        children.append(html.Hr())
        children.append(html.H2("🌐 Cross-File Intelligence", style={'color': '#2980b9'}))
        
        shared_keys = []
        filenames = list(all_db_data.keys())
        for i in range(len(filenames)):
            for j in range(i + 1, len(filenames)):
                f1, f2 = filenames[i], filenames[j]
                cols1 = set([col for t in all_db_data[f1].values() for col in t['columns']])
                cols2 = set([col for t in all_db_data[f2].values() for col in t['columns']])
                shared = cols1.intersection(cols2)
                if shared:
                    shared_keys.append(f"{f1} ⟷ {f2}: Common columns {list(shared)}")

        if shared_keys:
            children.append(html.Ul([html.Li(sk) for sk in shared_keys]))
        else:
            children.append(html.P("No common relational keys found across databases."))

    return children

if __name__ == '__main__':
    app.run_server(debug=True)
