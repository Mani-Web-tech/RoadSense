"""
RoadSense — AI-Powered Road Safety and Collision Warning Application
Main Entry Point
"""
import streamlit as st
from config import AppConfig
from ui.dashboard import render_dashboard
from utils.logging_config import logger

st.set_page_config(
    page_title="RoadSense — AI Collision Warning System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Initializes application configuration and renders the RoadSense dashboard."""
    try:
        config = AppConfig()
        render_dashboard(config)
    except Exception as e:
        logger.error(f"Top-level application error: {e}", exc_info=True)
        st.error(f"Fatal Application Error: {e}")


if __name__ == "__main__":
    main()
