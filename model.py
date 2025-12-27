import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from sklearn.model_selection import train_test_split

# --- 1. RE-LOAD DATA ---
print("Loading Data...")
interactions = pd.read_csv('interactions.csv')
properties = pd.read_csv('properties.csv')
users = pd.read_csv('users.csv')

interactions['user_id'] = interactions['user_id'].astype(str)
interactions['property_id'] = interactions['property_id'].astype(str)
properties['property_id'] = properties['property_id'].astype(str)
users['user_id'] = users['user_id'].astype(str)

unique_user_ids = users['user_id'].unique()
unique_property_ids = properties['property_id'].unique()

train_df, test_df = train_test_split(interactions, test_size=0.2, random_state=42)

train_ds = tf.data.Dataset.from_tensor_slices(dict(train_df))
test_ds = tf.data.Dataset.from_tensor_slices(dict(test_df))
properties_ds = tf.data.Dataset.from_tensor_slices(dict(properties))

train_ds = train_ds.map(lambda x: {"user_id": x["user_id"], "property_id": x["property_id"]})
test_ds = test_ds.map(lambda x: {"user_id": x["user_id"], "property_id": x["property_id"]})
properties_ds = properties_ds.map(lambda x: x["property_id"])

train_ds = train_ds.shuffle(100_000).batch(128).cache()
test_ds = test_ds.batch(64).cache()
properties_ds = properties_ds.batch(32).cache()

# --- 2. DEFINE MODEL ---
embedding_dimension = 32

class UserModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])
    def call(self, inputs):
        # CRITICAL: The model expects a dictionary inputs["user_id"]
        return self.user_embedding(inputs["user_id"])

class PropertyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.property_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_property_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_property_ids) + 1, embedding_dimension)
        ])
    def call(self, inputs):
        return self.property_embedding(inputs)

class SmartEstateModel(tfrs.Model):
    def __init__(self):
        super().__init__()
        self.query_model = UserModel()
        self.candidate_model = PropertyModel()
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=properties_ds.map(self.candidate_model)
            )
        )
    def compute_loss(self, features, training=False):
        query_embeddings = self.query_model(features)
        candidate_embeddings = self.candidate_model(features["property_id"])
        return self.task(query_embeddings, candidate_embeddings)

# --- 3. TRAIN ---
print("Training...")
model = SmartEstateModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
model.fit(train_ds, epochs=5)

# --- 4. INDEXING & SAVING ---
index = tfrs.layers.factorized_top_k.BruteForce(model.query_model)
index.index_from_dataset(
  tf.data.Dataset.zip((properties_ds, properties_ds.map(model.candidate_model)))
)

# [FIX] WARM UP THE MODEL (The Dry Run)
# We must call the index once so TensorFlow knows the input shape (Dictionary)
print("\nWarming up the index...")
_ = index({"user_id": np.array(["1"])}) 

print("Saving model...")
tf.saved_model.save(index, "exported_model")
print("\n✅ Model Saved with Signatures!")