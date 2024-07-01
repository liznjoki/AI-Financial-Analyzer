import streamlit as st
from app_pages import Welcome, Financial_Dashboard

# Create a sidebar for navigation
st.sidebar.title("AI Financial Analyzer")
page = st.sidebar.selectbox("Choose a page", ["Upload File", "Financial Dashboard"])

# Load the selected page
if page == "Upload File":
    Welcome.load_page()
elif page == "Financial Dashboard":
    Financial_Dashboard.load_page()
