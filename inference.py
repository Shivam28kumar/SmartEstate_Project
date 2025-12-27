import tensorflow as tf
import pandas as pd
import numpy as np

# 1. Load the Saved Model
print("Loading the exported model...")
loaded_model = tf.saved_model.load("exported_model")

# 2. Load Properties Data (For display)
properties = pd.read_csv('properties.csv')

def get_property_details(prop_id):
    # Filter by ID
    prop = properties[properties['property_id'] == int(prop_id)]
    if prop.empty:
        return f"Property ID {prop_id} (Details not found in CSV)"
    prop = prop.iloc[0]
    return f"{prop['type']} in {prop['location']} (Price: ${prop['price']:,})"

# 3. Make a Recommendation
user_id = "42"
print(f"\n🔮 Generating Recommendations for User {user_id}...")

# [FIX] Pass a Dictionary, not a list!
# The model expects inputs["user_id"]
test_input = {
    "user_id": np.array([user_id]) 
}

# The loaded model is now callable because we warmed it up
scores, titles = loaded_model(test_input)

# 4. Print Results
print(f"Top 3 Recommendations:")
for i in range(3):
    rec_prop_id = titles[0][i].numpy().decode('utf-8')
    score = scores[0][i].numpy()
    details = get_property_details(rec_prop_id)
    print(f"Rank {i+1}: {details} | (Match Score: {score:.4f})")