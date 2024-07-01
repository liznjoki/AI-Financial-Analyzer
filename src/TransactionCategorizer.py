import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import os
from pathlib import Path
from transformers import pipeline

ROOT_DIR = Path(__file__).resolve().parent.parent

model_path = os.path.join(ROOT_DIR, "model", "bank_text_classification")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


class TransactionCategorize():
    def __init__(self):
        pass

    @staticmethod
    def bank_classification(text):
        nlp = pipeline("text-classification", model=model, tokenizer=tokenizer)
        if "Fees" in text or "Mobile Banking Trnsfr" in text:
            return "transaction fees"
        category = nlp(text)[0]
        return category["label"]

    def category_column(self, data, text_column):
        data["Category"] = data[text_column].apply(self.bank_classification)
        return data

        # Identify the top 5 categories

    @staticmethod
    def top_expenses_df(data, amount_column, category_column):
        data[amount_column] = data[amount_column].abs()
        filtered_df = data.groupby(category_column)[amount_column].sum().reset_index()
        return filtered_df[[category_column, amount_column]].sort_values(ascending=False, by=amount_column,
                                                                         ignore_index=True).head(5)

    @staticmethod
    def financial_analysis(df):
        key_figures = {}

        #Calculate yearly/monthly total income and total expenses
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        df["YearMonth"] = df["Date"].dt.to_period("M")
        df["Year"] = df["Date"].dt.year
        yearly_income = df.groupby("Year")["Deposits"].sum().mean()
        yearly_expenses = df.groupby("Year")["Payments"].sum().mean()

        #Determine saving rate
        savings = yearly_income - yearly_expenses
        savings_rate = (savings/yearly_income) * 100 if savings > 0 else 0

        key_figures["Average Annual Income"] = f"Kes {yearly_income:,.0f}"
        key_figures["Average Annual Expense"] = f"Kes {yearly_expenses:,.0f}"
        key_figures["Annual Savings Rate"] = f"Kes {savings_rate:,.0f}%"
        key_figures["Average Annual Income"] = f"Kes {yearly_income:,.0f}"
        return key_figures

