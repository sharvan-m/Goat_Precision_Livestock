import os
import pandas as pd
from goat_weight_system.dataset_manager import DatasetManager
from goat_weight_system.regression_models import ModelBenchmarkEngine

print("1. Generating veterinary population dataset...")
dm = DatasetManager()
df = dm.load_data()
print("   Dataset loaded. Total rows:", len(df), "Goats:", df["goat_id"].nunique())
print("   Cohort distribution:")
print(df["cohort"].value_counts())
print("\n   Weight summary (kg):")
print(df["actual_weight_kg"].describe().round(2))

print("\n2. Splitting data leak-free by goat_id...")
train_df, val_df, test_df = dm.get_group_splits()
print(f"   Train goats: {train_df['goat_id'].nunique()} ({len(train_df)} rows)")
print(f"   Val goats:   {val_df['goat_id'].nunique()} ({len(val_df)} rows)")
print(f"   Test goats:  {test_df['goat_id'].nunique()} ({len(test_df)} rows)")

print("\n3. Training & Benchmarking 5 Regression Models...")
benchmarker = ModelBenchmarkEngine()
results = benchmarker.benchmark(train_df, val_df, test_df)

print("\n=== Benchmark Results ===")
print(results.to_string(index=False))

print(f"\nWinning best model: {benchmarker.best_model_name} saved to models/best_weight_model.pkl")
