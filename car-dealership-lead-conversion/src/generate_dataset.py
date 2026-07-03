import pandas as pd
import numpy as np
import os

np.random.seed(42)

# Number of customer leads
n_leads = 1500

# Create lead IDs
lead_ids = [f"L{str(i).zfill(4)}" for i in range(1, n_leads + 1)]

# Generate synthetic CRM-style dealership data
data = pd.DataFrame({
    "lead_id": lead_ids,
    "lead_source": np.random.choice(
        ["Website", "Walk-in", "Phone", "Carsales", "Referral", "Social Media"],
        size=n_leads,
        p=[0.25, 0.18, 0.16, 0.20, 0.11, 0.10]
    ),
    "customer_age_group": np.random.choice(
        ["18-24", "25-34", "35-44", "45-54", "55+"],
        size=n_leads,
        p=[0.12, 0.30, 0.27, 0.20, 0.11]
    ),
    "customer_location": np.random.choice(
        ["Local", "Nearby Suburb", "Far Suburb"],
        size=n_leads,
        p=[0.45, 0.35, 0.20]
    ),
    "vehicle_type": np.random.choice(
        ["SUV", "Sedan", "Hatchback", "Ute", "EV"],
        size=n_leads,
        p=[0.35, 0.20, 0.18, 0.17, 0.10]
    ),
    "new_or_used": np.random.choice(
        ["New", "Used", "Demo"],
        size=n_leads,
        p=[0.45, 0.40, 0.15]
    ),
    "budget_range": np.random.choice(
        ["Low", "Medium", "High"],
        size=n_leads,
        p=[0.35, 0.45, 0.20]
    ),
    "finance_enquiry": np.random.choice([0, 1], size=n_leads, p=[0.55, 0.45]),
    "trade_in": np.random.choice([0, 1], size=n_leads, p=[0.62, 0.38]),
    "test_drive_completed": np.random.choice([0, 1], size=n_leads, p=[0.58, 0.42]),
    "quote_provided": np.random.choice([0, 1], size=n_leads, p=[0.50, 0.50]),
    "previous_customer": np.random.choice([0, 1], size=n_leads, p=[0.72, 0.28]),
    "response_time_hours": np.random.gamma(shape=2.2, scale=5.0, size=n_leads).round(1),
    "follow_up_count": np.random.poisson(lam=3, size=n_leads),
    "days_since_enquiry": np.random.randint(1, 45, size=n_leads)
})

# Create a realistic conversion probability
conversion_score = (
    -2.2
    + 1.4 * data["test_drive_completed"]
    + 0.8 * data["finance_enquiry"]
    + 0.6 * data["trade_in"]
    + 0.9 * data["quote_provided"]
    + 0.5 * data["previous_customer"]
    + 0.12 * data["follow_up_count"]
    - 0.035 * data["response_time_hours"]
    - 0.025 * data["days_since_enquiry"]
)

# Add source-based effects
conversion_score += data["lead_source"].map({
    "Referral": 0.7,
    "Walk-in": 0.5,
    "Phone": 0.2,
    "Website": 0.0,
    "Carsales": -0.1,
    "Social Media": -0.3
})

# Add budget effects
conversion_score += data["budget_range"].map({
    "High": 0.25,
    "Medium": 0.15,
    "Low": -0.15
})

# Convert score into probability
conversion_probability = 1 / (1 + np.exp(-conversion_score))

# Generate final conversion outcome
data["converted"] = np.random.binomial(1, conversion_probability)

# Save dataset
output_path = "car-dealership-lead-conversion/data/raw/dealership_leads.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

data.to_csv(output_path, index=False)

print("Dataset created successfully.")
print(f"Rows: {data.shape[0]}")
print(f"Columns: {data.shape[1]}")
print(f"Conversion rate: {data['converted'].mean():.2%}")
print(f"Saved to: {output_path}")
