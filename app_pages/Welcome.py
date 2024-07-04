import streamlit as st
import pandas as pd
import base64



def load_page():
    left_column, right_column = st.columns(2)
    with right_column:
        file_path = r"./data/smartphone.png"
    with open(file_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()

    # Inject CSS to center the image
    st.sidebar.markdown(
            """
            <style>
            .centered-image {
                display: flex;
                justify-content: center;
                align-items: center;
            }
            </style>
            """, unsafe_allow_html=True
        )

    # Display the image in the sidebar
    st.sidebar.markdown(
        f'<div class="centered-image"><img src="data:image/png;base64,{encoded_image}" width="300"></div>',
            unsafe_allow_html=True
        )
    st.header("Welcome to the Bank Transaction Analyzer:bulb:", divider="rainbow")
    st.write("""I'm here to help you easily review and analyze your bank transactions.
                 \n My purpose is to assist you in gaining insights into your spending habits by identifying your top
                 spenders. Whether you're looking to manage your budget better or simply curious about where your money
                 goes, I'm here to make the process quick and simple. Just provide me with your transaction data,
                 and I'll help you understand the key details you need to take control of your finances.
                 \n Let's get started on your journey to smarter spending!"""
                 )

    uploaded_file = st.file_uploader("Please upload your bank statement with Date, Description, Payment and Deposit Columns. Don't worry we will not save any information 
                        beyond this session", type= "csv")

    if uploaded_file is not None:
        with st.spinner("Processing data......"):
            file_details = {"File Name": uploaded_file.name}
            df = pd.read_csv(uploaded_file)
            df['Payments'] = df['Payments'].fillna(0)
            df['Deposits'] = df['Deposits'].fillna(0)
            st.session_state.df = df

            st.write("----------------------------------------------------------------------------")

            st.write("File uploaded Successfully! Here is a summary of the transactions uploaded.")
            st.write(df)

            st.write("----------------------------------------------------------------------------")

            st.write("Have a look at your personalized insight and dashboards in your Financial Dashboard tab.")





