import dash
from dash import dcc, html, dash_table, Input, Output, State
import pandas as pd
import numpy as np
import sqlite3
import base64
import os
import plotly.express as px
from sklearn.ensemble import IsolationForest
from openai import OpenAI

# Initialize Dash App
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server 

app.layout = html.Div([
    # Header Section
    html.Div([
        html.H1("DBLens: AI-Driven Insight Generator", style={'marginBottom': '0px', 'color': '#2c3e50'}),
        html.P("Automated Intelligence & Natural Language Querying", style={'color': '#7f8c8d'})
    ], style={'textAlign': 'center', 'padding': '30px', 'backgroundColor': '#f8f9fa', 'borderBottom': '1px solid #eee'}),

    # Upload Section
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select .DB Files')]),
            style={
                'width': '100%', 'height': '80px', 'lineHeight': '80px',
                'borderWidth': '2px', 'borderStyle': 'dashed',
                'borderRadius': '10px', 'textAlign': 'center', 'margin': '20px 0'
            },
            multiple=True
        ),
    ], style={'padding': '0 10%'}),

    # NLQ Consultant Interface
    html.Div([
        html.H3("💬 AI Data Consultant", style={'marginTop': '0'}),
        html.P("Ask complex questions like 'Which products have the highest correlation with profit?' or 'List anomalies'."),
        dcc.Input(
            id='ai-query-input', 
            placeholder='Ask your data a question...', 
            style={'width': '75%', 'padding': '12px', 'borderRadius': '5px', 'border': '1px solid #ccc'}
        ),
        html.Button(
            'Ask AI', 
            id='ai-query-btn', 
            n_clicks=0,
            style={'marginLeft': '15px', 'backgroundColor': '#2980b9', 'color': 'white', 'border': 'none', 'padding': '12px 25px', 'borderRadius': '5px', 'cursor': 'pointer'}
        ),
        dcc.Loading(
            id="loading-ai",
            children=[html.Div(id='ai-query-response', style={'marginTop': '20px', 'padding': '20px', 'borderRadius': '8px', 'backgroundColor': '#ffffff', 'border': '1px solid #e1e4e8', 'minHeight': '50px'})]
        )
    ], style={'margin': '20px 10%', 'padding': '25px', 'backgroundColor': '#f1f2f6', 'borderRadius': '12px'}),

    # Main Dashboard Area
    dcc.Loading(
        id="loading-main",
        type="default",
        children=html.Div(id='output-data-upload', style={'padding': '0 10%'})
    ),
])

# --- HELPER FUNCTIONS ---

def get_schema_context(conn):
    """Extracts SQL schema to give the AI context without sending actual data."""
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
    return "\n".join([row[0] for row in cursor.fetchall() if row[0]])

def process_database(contents, filename):
    temp_filename = f"temp_{filename}"
    db_summary = {}
    conn = None
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        with open(temp_filename, 'wb') as f:
            f.write(decoded)
        
        conn = sqlite3.connect(temp_filename)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cursor.fetchall()]
        
        for table in tables:
            df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 1000", conn)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # Anomaly Detection
            anomalies = 0
            if len(numeric_cols) > 0 and len(df) > 10:
                model = IsolationForest(contamination=0.05, random_state=42)
                preds = model.fit_predict(df[numeric_cols].fillna(0))
                anomalies = int((preds == -1).sum())
            
            db_summary[table] = {
                'df': df,
                'stats': df.describe(include='all').to_dict(),
                'anomalies': anomalies,
                'correlations': df.corr(numeric_only=True).to_dict() if len(numeric_cols) > 1 else {},
                'columns': list(df.columns)
            }
        return db_summary
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        if conn: conn.close()
        # Note: We keep the file temporarily for the AI query session, then wipe on next upload

# --- CALLBACKS ---

@app.callback(
    Output('ai-query-response', 'children'),
    Input('ai-query-btn', 'n_clicks'),
    State('ai-query-input', 'value'),
    State('upload-data', 'filename')
)
def handle_ai_query(n_clicks, query, filenames):
    if n_clicks == 0 or not query or not filenames:
        return "The AI Consultant is ready. Upload a file and ask a question."

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return html.Div("⚠️ Configuration Error: OPENAI_API_KEY not set in Render Environment.", style={'color': 'red'})

    temp_path = f"temp_{filenames[0]}"
    if not os.path.exists(temp_path):
        return "File session expired. Please re-upload your database."

    try:
        client = OpenAI(api_key=api_key)
        conn = sqlite3.connect(temp_path)
        schema = get_schema_context(conn)

        prompt = f"""
        Database Schema:
        {schema}
        
        Task: Convert the user's question into a valid SQLite query. 
        Question: {query}
        Return ONLY the raw SQL. No markdown formatting, no explanation.
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a SQL expert generator."},
                      {"role": "user", "content": prompt}]
        )

        sql_query = response.choices[0].message.content.strip().replace('```sql', '').replace('```', '')
        result_df = pd.read_sql_query(sql_query, conn)
        conn.close()

        return html.Div([
            html.P([html.B("Generated Query: "), html.Code(sql_query)], style={'fontSize': '0.8em', 'color': '#636e72'}),
            html.H5("AI Query Result:"),
            dash_table.DataTable(
                data=result_df.head(10).to_dict('records'),
                columns=[{"name": i, "id": i} for i in result_df.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'}
            )
        ])
    except Exception as e:
        return html.Div(f"❌ Analysis Error: {str(e)}", style={'color': '#d63031'})

@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_dashboard(list_of_contents, list_of_names):
    if not list_of_contents:
        return html.Div("Awaiting database upload...", style={'textAlign': 'center', 'marginTop': '50px', 'color': '#bdc3c7'})

    all_data = {}
    for c, n in zip(list_of_contents, list_of_names):
        res = process_database(c, n)
        if res: all_data[n] = res

    output = []
    for fname, tables in all_data.items():
        output.append(html.H2(f"📊 Dataset: {fname}", style={'borderBottom': '2px solid #2980b9', 'paddingBottom': '10px'}))
        for tname, data in tables.items():
            output.append(html.Div([
                html.H4(f"Table: {tname}"),
                html.Div([
                    html.B("🤖 Status: "),
                    f"Found {data['anomalies']} anomalies." if data['anomalies'] > 0 else "Data looks healthy."
                ], style={'padding': '10px', 'backgroundColor': '#f8f9fa', 'marginBottom': '10px', 'borderRadius': '5px'}),
                
                dash_table.DataTable(
                    data=data['df'].head(5).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in data['df'].columns],
                    style_table={'overflowX': 'auto'}
                ),
                
                dcc.Graph(figure=px.imshow(pd.DataFrame(data['correlations']), title=f"Correlation Matrix: {tname}")) if data['correlations'] else None
            ], style={'marginBottom': '40px', 'padding': '20px', 'border': '1px solid #eee', 'borderRadius': '8px'}))

    return output

if __name__ == '__main__':
    app.run_server(debug=True)
