import dash
from dash import dcc, html, dash_table, Input, Output, State
import pandas as pd
import numpy as np
import sqlite3
import io
import base64
import plotly.express as px
from sklearn.ensemble import IsolationForest

# Initialize Dash App
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server # Required for Render deployment

app.layout = html.Div([
    html.H1("AI-Driven Database Insight Generator", style={'textAlign': 'center'}),
    html.P("Upload one or multiple .db (SQLite) files to generate deep insights.", style={'textAlign': 'center'}),
    
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
    
    html.Div(id='output-data-upload'),
])

def process_database(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    # Save temp file to read via sqlite3
    temp_filename = f"temp_{filename}"
    with open(temp_filename, 'wb') as f:
        f.write(decoded)
    
    conn = sqlite3.connect(temp_filename)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]
    
    db_summary = {}
    for table in tables:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        
        # 1. Basic Stats
        stats = df.describe(include='all').to_dict()
        
        # 2. Anomaly Detection (using Isolation Forest for numeric columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        anomalies = 0
        if len(numeric_cols) > 0 and len(df) > 5:
            model = IsolationForest(contamination=0.05, random_state=42)
            preds = model.fit_predict(df[numeric_cols].fillna(0))
            anomalies = (preds == -1).sum()
        
        # 3. Correlation Matrix
        corr = df.corr(numeric_only=True).to_dict() if len(numeric_cols) > 1 else {}

        db_summary[table] = {
            'df': df,
            'stats': stats,
            'anomalies': anomalies,
            'correlations': corr,
            'columns': list(df.columns)
        }
    
    conn.close()
    return db_summary

@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))
def update_output(list_of_contents, list_of_names):
    if list_of_contents is None:
        return None

    all_db_data = {}
    for c, n in zip(list_of_contents, list_of_names):
        all_db_data[n] = process_database(c, n)

    children = []

    # Individual File Insights
    for filename, tables in all_db_data.items():
        children.append(html.Hr())
        children.append(html.H3(f"File: {filename}"))
        
        for table_name, data in tables.items():
            children.append(html.H4(f"Table: {table_name}"))
            
            # Data Quality Alert
            if data['anomalies'] > 0:
                children.append(html.Div(
                    f"⚠️ Insight: Detected {data['anomalies']} potential anomalies in numeric distributions.",
                    style={'color': 'red', 'fontWeight': 'bold'}
                ))

            # Statistics Table
            children.append(dash_table.DataTable(
                data=data['df'].head(5).to_dict('records'),
                columns=[{"name": i, "id": i} for i in data['df'].columns],
                style_table={'overflowX': 'auto'},
                title="Data Preview"
            ))
            
            # Correlation Heatmap (if applicable)
            if data['correlations']:
                fig = px.imshow(pd.DataFrame(data['correlations']), title=f"Correlation Matrix: {table_name}")
                children.append(dcc.Graph(figure=fig))

    # Cross-File Relationship Analysis
    if len(all_db_data) > 1:
        children.append(html.Hr())
        children.append(html.H2("Cross-File Intelligence"))
        
        # Find Common Columns (Potential Joins)
        common_cols = {}
        filenames = list(all_db_data.keys())
        for i in range(len(filenames)):
            for j in range(i + 1, len(filenames)):
                f1, f2 = filenames[i], filenames[j]
                cols1 = set([col for t in all_db_data[f1].values() for col in t['columns']])
                cols2 = set([col for t in all_db_data[f2].values() for col in t['columns']])
                shared = cols1.intersection(cols2)
                if shared:
                    common_cols[f"{f1} ↔ {f2}"] = list(shared)

        if common_cols:
            children.append(html.H5("Potential Relational Joins Identified:"))
            for pair, cols in common_cols.items():
                children.append(html.P(f"{pair}: Shared keys {cols}"))
        else:
            children.append(html.P("No common column names found across files."))

    return children

if __name__ == '__main__':
    app.run_server(debug=True)
