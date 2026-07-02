# Car Dealership Lead Conversion Analytics

## Project Overview

This project predicts which car dealership customer leads are most likely to convert into vehicle purchases using CRM-style sales and enquiry data.

The aim is to build a practical lead scoring model that helps dealership sales teams prioritise high-intent customers, improve follow-up efficiency and understand the key factors that influence conversion.

## Business Problem

Car dealerships receive leads from multiple sources such as websites, walk-ins, phone enquiries, referral campaigns, social media and online car marketplaces.

However, not every lead has the same likelihood of converting into a sale. Some customers are ready to buy, while others are still browsing. Without a structured lead scoring process, sales teams may spend too much time on low-intent leads while missing high-potential customers.

This project answers the question:

**Which customer leads are most likely to convert into vehicle purchases, and what factors influence conversion?**

## Objectives

- Generate a realistic synthetic dealership CRM dataset
- Explore lead conversion patterns across customer, enquiry and sales activity data
- Build a classification model to predict conversion likelihood
- Score leads into high, medium and low priority groups
- Translate model results into practical sales and marketing recommendations

## Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Jupyter Notebook

## Project Structure

```text
car-dealership-lead-conversion/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_generate_dataset.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_lead_scoring.ipynb
│
├── src/
├── visuals/
├── README.md
└── requirements.txt
```
