from pathlib import Path
import streamlit as st
import sys
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

ROOT_DIR = Path(__file__).resolve().parent.parents
PARENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PARENT_DIR))
import re
import os
from src.TransactionCategorizer import TransactionCategorize

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
def load_page():
    st.header("Personal Financial Dashboard", divider="rainbow")
    st.write("Stay ahead of your finances with comprehensive, personalized insights and dashboards.")
    #Check if file has been uploaded
    if "df" in st.session_state:
        df = st.session_state.df
        #Rename the description column to be standard
        second_column_name = df.columns.values[1]
        df.rename(columns={second_column_name: "Description"}, inplace=True)
        #Categorize the transactions

        with st.spinner("Processing data......"):
            categorize = TransactionCategorize()

            # Display top expense categories in a table
            categorize.category_column(df, "Description")

            categorized_data = categorize.top_expenses_df(df, "Payments", "Category")

            key_results = categorize.financial_analysis(df)

            # Display average yearly figures
            st.subheader("Yearly Figures")
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Average Annual Income", value=key_results['Average Annual Income'])
            col2.metric(label="Average Annual Expense", value=key_results['Average Annual Expense'])
            col3.metric(label="Savings Rate", value=key_results['Annual Savings Rate'])

            # Display top expense categories in a table
            st.subheader("Top Expense Categories")
            st.write(categorized_data)


            st.write("----------------------------------------------------------------------------------")
            api_key = os.getenv("GROQ_API_KEY")
            llm = ChatGroq(
                model= "mixtral-8x7b-32768",
                api_key = groq_api_key,
                temperature = 0.3
            )
            human_prompt = f"Give me a recommendation of how to reduce my expenditure using my top expenses category: {categorized_data}"
            system_prompt = f"""You are a helpful and expert finance planner.
                            Based on the following analysis: {categorized_data}, make a summary of the financial status
                             of the user and suggest tips on savings. Highlight categories where the user can potentially
                             reduce expenses. Tailor these suggestions to fit the user’s lifestyle and financial objectives.
                             Use a friendly tone."""
            prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])
            chain = prompt | llm
            st.write(chain.invoke({"categorized_data}": categorized_data}).content)

    else:
        st.warning("No file uploaded. Please go to the 'Upload File' page to upload a file.")




