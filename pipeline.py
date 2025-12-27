import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 1. Load the Data
print("Loading data...")
interactions = pd.read_csv('interactions.csv')
properties = pd.read_csv('properties.csv')
users = pd.read_csv('users.csv')

# 2. Data Pre-processing
# We treat IDs as strings for Embedding Layers (categorical features)
interactions['user_id'] = interactions['user_id'].astype(str)
interactions['property_id'] = interactions['property_id'].astype(str)

properties['property_id'] = properties['property_id'].astype(str)
users['user_id'] = users['user_id'].astype(str)

# Convert timestamp to a numeric value (Unix timestamp) for potential sequence modeling later
interactions['timestamp'] = pd.to_datetime(interactions['timestamp'])
interactions['timestamp_unix'] = interactions['timestamp'].astype(np.int64) // 10**9

# 3. Create Vocabularies (Unique Lists)
# This maps every ID to an integer index internally.
unique_user_ids = users['user_id'].unique()
unique_property_ids = properties['property_id'].unique()
unique_property_titles = properties['location'].unique() # Using location as a feature

print(f"Unique Users: {len(unique_user_ids)}")
print(f"Unique Properties: {len(unique_property_ids)}")

# 4. Split into Train and Test
# We split by time to simulate a real production scenario (Train on past, Test on future)
# JD Reference: "Experimentation & Model Validation" 
train_df, test_df = train_test_split(interactions, test_size=0.2, random_state=42)

# 5. Create tf.data.Datasets
# This converts the Pandas DataFrame into a TensorFlow dictionary dataset
def df_to_dataset(dataframe, shuffle=True, batch_size=128):
    dataframe = dataframe.copy()
    labels = dataframe.pop('property_id') # In retrieval, the label IS the property_id
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

# Create the specific datasets for our Two-Tower model
# We map the inputs needed for each tower.
train_ds = tf.data.Dataset.from_tensor_slices({
    "user_id": train_df['user_id'].values,
    "property_id": train_df['property_id'].values,
    # We add context features here later
})

test_ds = tf.data.Dataset.from_tensor_slices({
    "user_id": test_df['user_id'].values,
    "property_id": test_df['property_id'].values,
})

# Candidate Dataset (All possible properties to recommend)
properties_ds = tf.data.Dataset.from_tensor_slices(properties['property_id'].values)

# Optimization: Cache and Prefetch for speed (High Performance Engineering)
train_ds = train_ds.shuffle(100_000).batch(128).cache().prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(64).cache().prefetch(tf.data.AUTOTUNE)
properties_ds = properties_ds.batch(32).cache().prefetch(tf.data.AUTOTUNE)

print("✅ Pipeline Created!")
print(f"Train batches: {len(train_ds)}")
print(f"Test batches: {len(test_ds)}")

# --- KEEP THESE VARIABLES FOR NEXT STEP ---
# In a notebook, these persist. If using a script, you'd export them.