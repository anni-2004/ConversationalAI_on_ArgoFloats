#dashboard.py
import streamlit as st

# Configure Streamlit page - MUST BE FIRST
st.set_page_config(
    page_title="FloatChat: AI-Powered ARGO Data Chatbot",
    page_icon="🌊",
    layout="wide",
)

import pandas as pd
import plotly.express as px
from datetime import datetime
from agentic_rag.agent_executor import AgenticSQLQueryExecutor
import uuid
import logging

# Initialize RAG System with Agentic Loop
@st.cache_resource
def get_rag_executor():
    return AgenticSQLQueryExecutor()

rag_executor = get_rag_executor()

# --- Custom CSS for a modern, vibrant look ---
st.markdown(
    """
    <style>
    /* Main body and block container styling */
    .reportview-container {
        background-color: #f5f5f5;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }

    /* Card styling for all containers */
    .st-emotion-cache-1c7y2qn, .st-emotion-cache-18ni7ap {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
    }

    /* Button styling */
    .stButton>button {
        border: none;
        background: linear-gradient(45deg, #007bff, #0056b3);
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #0056b3, #004085);
        transform: translateY(-2px);
    }
    
    /* Text styling */
    .st-emotion-cache-10o9y34 {
        color: #333333;
    }
    h1 {
        color: #1a237e;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
    }
    h2, h3, h4, h5, h6 {
        color: #3f51b5;
        font-weight: 600;
    }

    /* Metric card styling */
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-card h4 {
        color: #6a1b9a;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .metric-card h2 {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.2rem;
    }
    .metric-card p {
        color: #7f8c8d;
        font-size: 0.9rem;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar ---
with st.sidebar:
    st.header("FloatChat Settings")
    st.markdown("Use this chatbot to query the ARGO float database.")
    # You can add more settings here later if needed

# Initialize chat history and RAG executor
if "messages" not in st.session_state:
    st.session_state.messages = []
if "executor" not in st.session_state:
    st.session_state.executor = get_rag_executor()

# Helper function for visualization
def clean_for_plot(plot_df, y_col):
    """Removes rows with NaN values for a given column."""
    if y_col in plot_df.columns:
        return plot_df.dropna(subset=[y_col])
    return pd.DataFrame()

def create_visualizations(df):
    """Dynamically creates visualizations based on available columns."""
    
    st.markdown("### 📊 Visualizations")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🗺️ Geospatial Map")
        if 'lat' in df.columns and 'lon' in df.columns and not df.empty:
            st.map(df, latitude='lat', longitude='lon', use_container_width=True)
        else:
            st.info("No data available to create a map.")

    with col2:
        st.subheader("🌡️ Depth-Temperature Profile")
        temp_df = clean_for_plot(df, 'temperature')
        if 'depth' in df.columns and not temp_df.empty:
            temp_fig = px.line(
                temp_df,
                x='temperature',
                y='depth',
                title="Temperature vs. Depth",
                labels={"temperature": "Temperature (°C)", "depth": "Depth (m)"},
                color_discrete_sequence=['#ff6b6b']
            )
            temp_fig.update_yaxes(autorange="reversed")
            st.plotly_chart(temp_fig, use_container_width=True)
        else:
            st.info("No data available for temperature visualization.")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("💧 Depth-Salinity Profile")
        sal_df = clean_for_plot(df, 'salinity')
        if 'depth' in df.columns and not sal_df.empty:
            sal_fig = px.line(
                sal_df,
                x='salinity',
                y='depth',
                title="Salinity vs. Depth",
                labels={"salinity": "Salinity (PSU)", "depth": "Depth (m)"},
                color_discrete_sequence=['#54a0ff']
            )
            sal_fig.update_yaxes(autorange="reversed")
            st.plotly_chart(sal_fig, use_container_width=True)
        else:
            st.info("No data available for salinity visualization.")

    with col4:
        st.subheader("📈 Time Series Data")
        if 'time' in df.columns and ('temperature' in df.columns or 'salinity' in df.columns) and not df.empty:
            fig = px.line(df, x='time', y=['temperature', 'salinity'], title='Time Series Data')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for time series visualization.")

# App Title and Introduction
st.title("🌊 FloatChat: AI-Powered ARGO Data Chatbot")
st.markdown("Hello there! I'm your AI assistant for exploring ARGO ocean data. How can I help you today?")
st.markdown("---")

# --- Display dashboard-like metrics (dynamic with fallback) ---
st.markdown("### 📊 ARGO Data Metrics")
col1, col2, col3, col4 = st.columns(4)

try:
    metrics = st.session_state.executor.db_manager.get_metrics()
    
    if metrics["success"]:
        active_floats = metrics.get("active_floats", "N/A")
        ocean_profiles = metrics.get("ocean_profiles", "N/A")
        data_points = metrics.get("data_points", "N/A")
        
        with col1:
            st.markdown(f'<div class="metric-card"><h4>Active Floats</h4><h2>{active_floats}</h2><p>Real-time</p></div>', unsafe_allow_html=True)
    
        with col2:
            st.markdown(f'<div class="metric-card"><h4>Ocean Profiles</h4><h2>{ocean_profiles}</h2><p>Real-time</p></div>', unsafe_allow_html=True)
    
        with col3:
            st.markdown(f'<div class="metric-card"><h4>Data Points</h4><h2>{data_points}</h2><p>Real-time</p></div>', unsafe_allow_html=True)
    
        with col4:
            st.markdown(f'<div class="metric-card"><h4>Last Update</h4><h2>{datetime.now().strftime("%H:%M:%S")}</h2><p>Real-time</p></div>', unsafe_allow_html=True)
    else:
        st.warning("Failed to fetch real-time metrics. Displaying placeholders.")
        with col1:
            st.markdown(f'<div class="metric-card"><h4>Active Floats</h4><h2>156</h2><p>+12%</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h4>Ocean Profiles</h4><h2>1,247</h2><p>+8%</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h4>Data Points</h4><h2>89,432</h2><p>+24%</p></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><h4>Last Update</h4><h2>2 hours ago</h2><p>Real-time</p></div>', unsafe_allow_html=True)
except Exception as e:
    st.error(f"An error occurred while fetching metrics: {e}")
    with col1:
        st.markdown(f'<div class="metric-card"><h4>Active Floats</h4><h2>156</h2><p>+12%</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h4>Ocean Profiles</h4><h2>1,247</h2><p>+8%</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h4>Data Points</h4><h2>89,432</h2><p>+24%</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><h4>Last Update</h4><h2>2 hours ago</h2><p>Real-time</p></div>', unsafe_allow_html=True)

st.markdown("---")

# Main chat UI
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "data" in message and message["data"] is not None:
            df = pd.DataFrame(message["data"])
            with st.expander("Show Raw Data"):
                st.dataframe(df, use_container_width=True)
            create_visualizations(df)
        if "query" in message:
            with st.expander("Generated SQL Query"):
                st.code(message["query"], language="sql")

# React to user input
if prompt := st.chat_input("Ask a question about ARGO data..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Executing query..."):
            response = st.session_state.executor.query_with_rag(prompt)
            
        if response["success"]:
            st.markdown(response["enhanced_response"])
            message = {
                "role": "assistant",
                "content": response["enhanced_response"],
                "data": response["data"],
                "query": response["generated_query"]
            }
            st.session_state.messages.append(message)
            
            if response["data"] is not None and len(response["data"]) > 0:
                df = pd.DataFrame(response["data"])
                with st.expander("Show Raw Data"):
                    st.dataframe(df, use_container_width=True)
                    st.download_button(
                        label="Download CSV",
                        data=df.to_csv(index=False).encode('utf-8'),
                        file_name=f'argo_data_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv',
                        mime='text/csv'
                    )
                create_visualizations(df)
            else:
                st.info("The query was successful, but no data was found to visualize.")
                create_visualizations(pd.DataFrame())
        else:
            st.markdown(response["enhanced_response"])
            message = {
                "role": "assistant",
                "content": response["enhanced_response"],
                "data": [],
                "query": response.get("generated_query", "No query generated.")
            }
            st.session_state.messages.append(message)
            create_visualizations(pd.DataFrame())