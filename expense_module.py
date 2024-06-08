import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path
import json
from datetime import datetime, timedelta
import re
import os

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_NAME = "mgrella/autonlp-bank-transaction-classification-5521155"
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)


#This Class Categorizes Transactions for fixed and variable expenses:
class TransactionCategorize:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.categories = self.model.config.id2label.values()
        self.data_path=os.path.join(ROOT_DIR,'data')

    #Read Fixed Expense CSVs and pre_process column name to transaction_description to be able to merge with variable expense
    @staticmethod
    def fixed_expenses(file_name):
        df1 = pd.read_csv(file_name)

        # Convert 'expense_type' to transaction_descrption
        df1.rename(columns={'expense_type': 'transaction_description'}, inplace=True)  # rename column

        return df1

    #Read the Variable Expense CSV
    @staticmethod
    def variable_expenses(file_name):
        df2 = pd.read_csv(file_name)
        return df2

    # Merges two CSV files and keeps only the specified columns.
    @staticmethod
    def merge_data(table1, table2, specified_columns, user_id):
        #keep only specified columns
        table1 = table1[specified_columns]
        table2 = table2[specified_columns]
        # Combine data (assuming user_id is the common key)
        combined_data = pd.concat([table1, table2], join="inner", ignore_index=True)
        filtered_data = combined_data[combined_data["user_id"] == user_id]
        return filtered_data

    # Categorize Transactions using the Pre-trained Model
    def categorize_transaction(self, text):
        if "rent" in text:
            return "HOUSING"
        else:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
            category_idx = probabilities.index(max(probabilities))
            full_category = list(self.categories)[category_idx]
            match = re.search(r'\.(.*?)_',full_category)  #pattern searches for a dot (.) followed by any sequence of characters ending with an underscore (_).
            if match:
                final_category = match.group(1)  # holds the actual category name without the prefix "Category."
            return final_category

    def transaction_category_df(self, data, text_column):
        data["category"] = data[text_column].apply(self.categorize_transaction)
        return data

    # Identify the top 5 categories
    @staticmethod
    def top_expenses_df(data, amount_column, category_column):
        filtered_df = data.groupby(category_column)[amount_column].sum().reset_index()
        return filtered_df[[category_column, amount_column]].sort_values(ascending=False, by=amount_column,
        ignore_index=True).head(5)

    def get_top_expenses(self, user_id):
        specified_columns = ["user_id", "transaction_description", "amount"]
        output_path= os.path.join(self.data_path,'output',f'{str(user_id)}_model_outputs.json')
        #put logic to check last updated date and trigger the model
        file_path_fixed_expense = os.path.join(self.data_path, "fixed_expenses.csv")
        file_path_variable_expense = os.path.join(self.data_path, "variable_expenses.csv")
        #Merge both fixed and variable expenses
        result_df = TransactionCategorize.merge_data(table1=TransactionCategorize.variable_expenses(file_path_variable_expense),
                                                     table2=TransactionCategorize.fixed_expenses(file_path_fixed_expense),
                                                     specified_columns=specified_columns,
                                                     user_id=user_id)

        #Categorize the combined data and create a column that categorizes each expense
        final_df = self.transaction_category_df(data=result_df,text_column="transaction_description")
        #Calculate the top expenses
        top_expenses = TransactionCategorize.top_expenses_df(final_df, 'amount', 'category')
        top_expenses= top_expenses.to_json(orient='values')
        current_datetime = datetime.now()
        with open(output_path, 'r') as file:
            data = json.load(file)
        new_value = {
            'last_updated':data["top_spending_categories"]['last_updated'],
            "top_spendings":top_expenses,
            "recommendations":data['top_spending_categories']['recommendations']
        }
        
        data["top_spending_categories"] = new_value
        with open(output_path, 'w') as file:
                json.dump(data, file, indent=4)

        return top_expenses
    def is_data_changed(self,model_ran_date,timestamp_column='last_updated',):
        file_path = os.path.join(self.data_path, "fixed_expenses.csv")
        df = pd.read_csv(file_path)
        # Convert the timestamp column to datetime objects
        df[timestamp_column] = pd.to_datetime(df[timestamp_column], format="%m-%d-%Y %H:%M:%S")

        # Calculate the threshold date
        # threshold_date = datetime.now() - timedelta(days=days_threshold)
        model_ran = datetime.strptime(model_ran_date, "%m-%d-%Y %H:%M:%S")

        # Debug output
        print(f"Threshold date: {model_ran}")

        # Check if any timestamp is within the threshold
        recent_updates = df[timestamp_column] > model_ran

        # Debug output
        print(f"Timestamps within threshold: {df[recent_updates][timestamp_column]}")

        return recent_updates.any()
        


if __name__ == "__main__":

    user_id_to_view = 374576  # Will change according to the current user logged in
    expenses = TransactionCategorize()
    file_path = os.path.join(expenses.data_path, "fixed_expenses.csv")
    output_path = os.path.join(expenses.data_path,'output','374576_model_outputs.json')
    with open(output_path, 'r') as file:
            data = json.load(file)
    model_run_date = data['top_spending_categories']['last_updated']
    df = pd.read_csv(file_path)
    if expenses.is_data_changed(model_run_date ):
        top_expenses=expenses.get_top_expenses(user_id_to_view)
        print(top_expenses)
    

