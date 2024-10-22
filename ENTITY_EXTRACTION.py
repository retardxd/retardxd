# -*- coding: utf-8 -*-
"""HAIML EXP 5 FINAL .ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lkTcdvKFotXvB_s7cBcudyAb6bEry8Rs
"""

import pandas as pd
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the CSV file into a DataFrame
admissions_df = pd.read_csv('dataset.csv')

# Print the columns to verify their names
print("Columns in DataFrame:", admissions_df.columns)

# Inspect the first few rows of the DataFrame
print(admissions_df.head())

# Function to perform NER
def perform_ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# First Scenario: Using a 'yourcolumn_name' column for NER
# First Scenario Start
def scenario_yourcolumn_name_ner():
    text_column = 'yourcolumn_name'  # Adjust this if you want to use a different column
    # Extract text data from the 'yourcolumn_name' column
    text_data = admissions_df[text_column].dropna().astype(str).tolist()

    # Process the first few entries for demonstration
    for text in text_data[:5]:  # Limiting to first 5 for demonstration
        entities = perform_ner(text)
        print(f"Text: {text}\nEntities: {entities}\n")
# First Scenario End

# Second Scenario: Creating a custom report from relevant columns for NER


# Uncomment one of the following lines to toggle between scenarios:
scenario_yourcolumn_name_ner()  # Use this for NER based on the 'yourcolumn_name' column


# ----------------------------------------------------------------------------------------------------------------------------
# Second Scenario : IF ALL DATA IS NUMERIC THEN USE THIS ELSE DELETE Start
# def scenario_report_ner():
#     def create_report(row):
#         return (f"Patient ID: {row['subject_id']}, "
#                 f"Hemoglobin: {row['Hemoglobin']}, "
#                 f"Eosinophils: {row['Eosinophils']}, "
#                 f"Lymphocytes: {row['Lymphocytes']}, "
#                 f"Monocytes: {row['Monocytes']}, "
#                 f"Basophils: {row['Basophils']}, "
#                 f"Hematocrit: {row['Hematocrit']}, "
#                 f"Gender: {row['gender']}, "
#                 f"Date of Birth: {row['dob']}, "
#                 f"Date of Death: {row['dod']}, "
#                 f"Short Title: {row['short_title']}, "
#                 f"Long Title: {row['long_title']}.")

#     # Apply the report creation function to each row
#     admissions_df['report'] = admissions_df.apply(create_report, axis=1)

#     # Extract text data from the newly created 'report' column
#     text_data = admissions_df['report'].dropna().astype(str).tolist()

#     # Process the first few entries for demonstration
#     for text in text_data[:5]:  # Limiting to the first 5 for demonstration
#         entities = perform_ner(text)
#         print(f"Text: {text}\nEntities: {entities}\n")

#         #scenario_report_ner()  # Use this for NER based on the custom report
# # Second Scenario End

