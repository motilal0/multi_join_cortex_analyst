from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import snowflake.connector
import streamlit as st

from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re


DATABASE = "CORTEX_ANALYST_DEMO_1"
SCHEMA = "REVENUE_TIMESERIES"
STAGE = "RAW_DATA"
FILE = "multi_join_timeseries.yaml"
WAREHOUSE = "cortex_analyst_wh"

# Reading the entire content of the file
with open("snowflake_password.txt", 'r') as file:
    ps = file.read()

# replace values below with your Snowflake connection information
# HOST = "<host>"
ACCOUNT = "tz80493.us-east-2.aws"
USER = "motilal"
PASSWORD = ps
ROLE = "SYSADMIN"

if 'CONN' not in st.session_state or st.session_state.CONN is None:
    st.session_state.CONN = snowflake.connector.connect(
        user=USER,
        password=PASSWORD,
        account=ACCOUNT,
        # host=HOST,
        port=443,
        warehouse=WAREHOUSE,
        role=ROLE,
    )


# Forecasting

# Fetch data from Snowflake
def fetch_data():
    query = """
    SELECT date, revenue
    FROM CORTEX_ANALYST_DEMO_1.REVENUE_TIMESERIES.DAILY_REVENUE_NEW
    ORDER BY date ASC;
    """
    data = pd.read_sql(query, st.session_state.CONN)
    # st.write("Original DataFrame Columns:", data.columns)
    return data

# Forecast future sales using Prophet
def forecast_sales(data, periods=30):
    data.columns = data.columns.str.lower()
    data = data.rename(columns={"date": "ds", "revenue": "y"})
    data['ds'] = pd.to_datetime(data['ds'])
    data['y'] = pd.to_numeric(data['y'], errors='coerce')
    data = data.dropna()
    
    model = Prophet()
    model.fit(data)
    
    last_date = data['ds'].max()
    future = model.make_future_dataframe(periods=periods, freq='D', include_history=False)
    forecast = model.predict(future)
    
    # Limit forecast to only the requested period
    forecast = forecast[forecast['ds'] > last_date]
    forecast = forecast.head(periods)  # Show only the next 'periods' rows
    
    return data, forecast, model

# Extract forecast period from user input
def extract_forecast_period(prompt: str) -> int:
    match = re.search(r'next\s+(\d+)\s+days?', prompt.lower())
    if match:
        return int(match.group(1))
    return 180  # Default to 180 days if not specified

# Plot the forecast using Plotly
def plot_forecast(data, forecast, title="Projected Sales Forecast"):
    fig = make_subplots(rows=1, cols=1)
    
    # Actual data
    fig.add_trace(
        go.Scatter(
            x=data['ds'], y=data['y'],
            mode='lines+markers', name='Actual Revenue',
            marker=dict(color='blue')
        )
    )

    # Forecasted data
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'], y=forecast['yhat'],
            mode='lines', name='Forecasted Revenue',
            line=dict(color='green', dash='dash')
        )
    )

    # Confidence intervals
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'], y=forecast['yhat_upper'],
            fill=None, mode='lines', name='Upper Confidence',
            line=dict(color='lightgreen', dash='dot')
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'], y=forecast['yhat_lower'],
            fill='tonexty', mode='lines', name='Lower Confidence',
            line=dict(color='lightgreen', dash='dot')
        )
    )

    # Layout settings
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Revenue",
        legend_title="Legend",
        template="plotly_white"
    )

    return fig

# Streamlit interface
# st.title("Sales Projection Dashboard")
data = fetch_data()
# st.write("Data Preview:", data.head())  # Verify data

# Get both the renamed data and forecast
data_renamed, forecast, model = forecast_sales(data)

# st.subheader("Forecasted Data")
# st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30))  # Show last 30 forecasted days

# st.subheader("Forecast Plot")
fig = plot_forecast(data_renamed, forecast)  # Pass renamed data
# st.plotly_chart(fig, use_container_width=True)




def send_message(prompt: str) -> Dict[str, Any]:
    """Calls the REST API and returns the response."""
    request_body = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "semantic_model_file": f"@{DATABASE}.{SCHEMA}.{STAGE}/{FILE}",
    }
    resp = requests.post(
        url=f"https://{ACCOUNT}.snowflakecomputing.com/api/v2/cortex/analyst/message",
        json=request_body,
        headers={
            "Authorization": f'Snowflake Token="{st.session_state.CONN.rest.token}"',
            "Content-Type": "application/json",
        },
    )
    request_id = resp.headers.get("X-Snowflake-Request-Id")
    if resp.status_code < 400:
        return {**resp.json(), "request_id": request_id}  # type: ignore[arg-type]
    else:
        raise Exception(
            f"Failed request (id: {request_id}) with status {resp.status_code}: {resp.text}"
        )

# Adjust process_message function for period-specific forecast
def process_message(prompt: str) -> None:
    st.session_state.messages.append(
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    )
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            if any(keyword in prompt.lower() for keyword in ["forecast", "predict", "future","projected"]):
                try:
                    periods = extract_forecast_period(prompt)
                    data = fetch_data()
                    data_renamed, forecast, model = forecast_sales(data, periods=periods)
                    
                    # Set dynamic title
                    title = f"Projected Sales for Next {periods} Days"
                    
                    # Display forecast table and plot
                    st.subheader("Forecasted Data")
                    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

                    st.subheader("Forecast Plot")
                    fig = plot_forecast(data_renamed, forecast, title=title)
                    st.plotly_chart(fig, use_container_width=True)

                    content = [{"type": "text", "text": f"Here is the sales forecast for the next {periods} days."}]
                    request_id = "forecast_generated"
                except Exception as e:
                    content = [{"type": "text", "text": f"Error generating forecast: {str(e)}"}]
                    request_id = "error"
            else:
                response = send_message(prompt=prompt)
                request_id = response.get("request_id", "unknown_request")
                content = response["message"]["content"]
                display_content(content=content, request_id=request_id)
            
    st.session_state.messages.append(
        {"role": "assistant", "content": content, "request_id": request_id}
    )

def display_content(
    content: List[Dict[str, str]],
    request_id: Optional[str] = None,
    message_index: Optional[int] = None,
) -> None:
    """Displays a content item for a message."""
    message_index = message_index or len(st.session_state.messages)
    if request_id:
        with st.expander("Request ID", expanded=False):
            st.markdown(request_id)
    for item in content:
        if item["type"] == "text":
            st.markdown(item["text"])
        elif item["type"] == "suggestions":
            with st.expander("Suggestions", expanded=True):
                for suggestion_index, suggestion in enumerate(item["suggestions"]):
                    if st.button(suggestion, key=f"{message_index}_{suggestion_index}"):
                        st.session_state.active_suggestion = suggestion
        elif item["type"] == "sql":
            with st.expander("SQL Query", expanded=False):
                st.code(item["statement"], language="sql")
            with st.expander("Results", expanded=True):
                with st.spinner("Running SQL..."):
                    df = pd.read_sql(item["statement"], st.session_state.CONN)
                    if len(df.index) > 1:
                        data_tab, line_tab, bar_tab = st.tabs(
                            ["Data", "Line Chart", "Bar Chart"]
                        )
                        data_tab.dataframe(df)
                        if len(df.columns) > 1:
                            df = df.set_index(df.columns[0])
                        with line_tab:
                            st.line_chart(df)
                        with bar_tab:
                            st.bar_chart(df)
                    else:
                        st.dataframe(df)
st.markdown("<h1 style='font-size:36px;'>Cortex Analyst: Multi-Join & Prediction</h1>", unsafe_allow_html=True)
# st.title("Cortex Analyst: Multi-Join & Prediction")
st.markdown(f"Semantic Model: `{FILE}`")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.suggestions = []
    st.session_state.active_suggestion = None

for message_index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        display_content(
            content=message["content"],
            request_id=message.get("request_id"),
            message_index=message_index,
        )

if user_input := st.chat_input("What is your question?"):
    process_message(prompt=user_input)

if st.session_state.active_suggestion:
    process_message(prompt=st.session_state.active_suggestion)
    st.session_state.active_suggestion = None

