import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# --- CONFIGURATION ---
NUM_PROPERTIES = 1000
NUM_USERS = 500
NUM_INTERACTIONS = 10000

# --- 1. GENERATE PROPERTIES (ITEM TOWER DATA) ---
print("Generating Properties...")
property_types = ['Apartment', 'Villa', 'Townhouse', 'Penthouse']
locations = ['Downtown Dubai', 'Marina', 'Palm Jumeirah', 'Business Bay', 'JVC'] # UAE context as per JD

properties = pd.DataFrame({
    'property_id': range(1, NUM_PROPERTIES + 1),
    'type': np.random.choice(property_types, NUM_PROPERTIES),
    'location': np.random.choice(locations, NUM_PROPERTIES),
    'bedrooms': np.random.randint(1, 6, NUM_PROPERTIES),
    'price': np.random.randint(500000, 5000000, NUM_PROPERTIES),
    'area_sqft': np.random.randint(400, 5000, NUM_PROPERTIES),
    'listing_freshness': np.random.choice(['New', 'Old'], NUM_PROPERTIES, p=[0.2, 0.8]) # JD: Freshness
})

# Add some "investor appeal" logic (High ROI potential for cheaper units in popular areas)
properties['roi_potential'] = properties.apply(lambda x: 'High' if x['price'] < 1500000 and x['location'] in ['JVC', 'Business Bay'] else 'Medium', axis=1)

# --- 2. GENERATE USERS (USER TOWER DATA) ---
print("Generating Users...")
user_personas = ['Investor', 'End-User', 'Luxury-Buyer'] # JD: Dynamic Segmentation

users = pd.DataFrame({
    'user_id': range(1, NUM_USERS + 1),
    'persona': np.random.choice(user_personas, NUM_USERS, p=[0.2, 0.7, 0.1]),
    'geo_location': np.random.choice(['UAE', 'International'], NUM_USERS, p=[0.6, 0.4]) # JD: Local vs International
})

# --- 3. GENERATE INTERACTIONS (THE CLICKSTREAM) ---
print("Generating Clickstream Data...")
interactions = []

for _ in range(NUM_INTERACTIONS):
    user = users.sample(1).iloc[0]
    
    # LOGIC: Bias clicks based on persona to make the pattern "learnable"
    if user['persona'] == 'Luxury-Buyer':
        # They prefer Villas and Penthouses > 3M
        preferred_props = properties[properties['price'] > 3000000]
    elif user['persona'] == 'Investor':
        # They prefer High ROI properties
        preferred_props = properties[properties['roi_potential'] == 'High']
    else:
        # End-users are random but price sensitive
        preferred_props = properties[properties['price'] < 2000000]
    
    if preferred_props.empty:
        selected_prop = properties.sample(1).iloc[0]
    else:
        selected_prop = preferred_props.sample(1).iloc[0]

    # JD: Intent Signals (Rent vs Buy behavior can be inferred from event types)
    event_type = np.random.choice(['view', 'view', 'view', 'shortlist', 'contact_agent'], p=[0.6, 0.2, 0.1, 0.05, 0.05])
    
    # Generate timestamp (Sequence modeling requires time)
    random_days = np.random.randint(0, 30)
    timestamp = datetime.now() - timedelta(days=random_days)
    
    interactions.append({
        'user_id': user['user_id'],
        'property_id': selected_prop['property_id'],
        'timestamp': timestamp,
        'event_type': event_type,
        'location_viewed': selected_prop['location'], # Helpful for debugging
        'price_viewed': selected_prop['price']
    })

interactions_df = pd.DataFrame(interactions)
interactions_df = interactions_df.sort_values(by=['user_id', 'timestamp']) # Sort for sequence processing

# --- SAVE DATA ---
properties.to_csv('properties.csv', index=False)
users.to_csv('users.csv', index=False)
interactions_df.to_csv('interactions.csv', index=False)

print("✅ Data Generation Complete!")
print(f"Properties: {properties.shape}")
print(f"Users: {users.shape}")
print(f"Interactions: {interactions_df.shape}")