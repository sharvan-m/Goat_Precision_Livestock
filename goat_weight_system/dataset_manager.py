"""
Dataset management and leakage-free splitting module.
Handles loading, validation, synthetic data generation, and Group-based (goat_id) train/val/test splits.
"""

from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

from goat_weight_system.config import DATASET_CSV_PATH


class DatasetManager:
    """Manages tabular dataset loading, feature preprocessing, and leak-free group splitting."""

    FEATURE_COLUMNS = [
        "body_length",
        "height",
        "girth",
        "body_width",
        "mask_area",
        "solidity",
        "bbox_aspect_ratio",
        "major_axis_px",
        "minor_axis_px",
        "eccentricity",
    ]

    TARGET_COLUMN = "actual_weight_kg"
    GROUP_COLUMN = "goat_id"

    def __init__(self, csv_path: Optional[str] = None):
        self.csv_path = csv_path if csv_path else str(DATASET_CSV_PATH)
        self.df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Load dataset from CSV file."""
        if not pd.io.common.file_exists(self.csv_path):
            raise FileNotFoundError(f"Dataset CSV file not found at: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        # Validate required columns
        required_cols = [self.TARGET_COLUMN, self.GROUP_COLUMN] + self.FEATURE_COLUMNS
        missing_cols = [c for c in required_cols if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Dataset missing required columns: {missing_cols}")

        return self.df

    def get_group_splits(
        self,
        test_size: float = 0.2,
        val_size: float = 0.15,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into Train, Validation, and Test sets based on goat_id groups.
        Guarantees no images of the same goat ID appear across different splits!

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.df is None:
            self.load_data()

        df = self.df.copy()

        # Step 1: Split into Train+Val and Test using GroupShuffleSplit on goat_id
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_val_idx, test_idx = next(gss_test.split(df, groups=df[self.GROUP_COLUMN]))

        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        # Step 2: Split Train+Val into Train and Validation using GroupShuffleSplit
        val_ratio_adjusted = val_size / (1.0 - test_size)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio_adjusted, random_state=random_state)
        train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df[self.GROUP_COLUMN]))

        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

        # Verify no overlap between goat IDs across splits
        train_ids = set(train_df[self.GROUP_COLUMN].unique())
        val_ids = set(val_df[self.GROUP_COLUMN].unique())
        test_ids = set(test_df[self.GROUP_COLUMN].unique())

        assert len(train_ids.intersection(val_ids)) == 0, "Leakage detected between Train and Val sets!"
        assert len(train_ids.intersection(test_ids)) == 0, "Leakage detected between Train and Test sets!"
        assert len(val_ids.intersection(test_ids)) == 0, "Leakage detected between Val and Test sets!"

        return train_df, val_df, test_df

    @staticmethod
    def generate_synthetic_dataset(
        num_goats: int = 50,
        samples_per_goat: int = 6,
        output_csv_path: Optional[str] = None,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Generates realistic synthetic dataset based on veterinary allometric caprine weight formulas:
        Weight (kg) = k * (Body Length)^1.0 * (Heart Girth)^2.0 + noise.

        Allows out-of-the-box benchmarking and training before field data is populated.
        """
        np.random.seed(random_state)
        records = []

        # Generate realistic physical characteristics per goat
        for g_id in range(1, num_goats + 1):
            goat_str_id = f"GOAT_{g_id:03d}"

            # True physical traits for goat g_id
            true_length = np.random.uniform(55.0, 95.0)   # cm (body length)
            true_girth = np.random.uniform(50.0, 90.0)    # cm (chest girth)
            true_height = true_length * np.random.uniform(0.70, 0.85)  # cm (withers height)
            true_width = true_girth * 0.35

            # Allometric weight formula: W (kg) = 0.000085 * Length * (Girth^2) + noise
            base_weight = 0.000082 * true_length * (true_girth ** 2.0)
            actual_weight = float(np.clip(base_weight + np.random.normal(0, 1.2), 15.0, 85.0))

            # Multiple image frames per goat with measurement variation (angle/pose noise)
            for img_num in range(1, samples_per_goat + 1):
                image_id = f"img_{goat_str_id}_f{img_num:02d}.jpg"

                # Visual measurement fluctuations from pose / distance
                scale = np.random.uniform(0.95, 1.05)
                length_meas = true_length * scale + np.random.normal(0, 0.5)
                girth_meas = true_girth * scale + np.random.normal(0, 0.6)
                height_meas = true_height * scale + np.random.normal(0, 0.5)
                width_meas = true_width * scale + np.random.normal(0, 0.3)

                mask_area = length_meas * height_meas * np.random.uniform(0.58, 0.65)
                solidity = float(np.clip(np.random.uniform(0.72, 0.92), 0.70, 0.98))
                aspect_ratio = length_meas / max(height_meas, 1.0)

                major_axis = length_meas * 10.0
                minor_axis = height_meas * 10.0
                eccentricity = np.sqrt(1.0 - (minor_axis / major_axis) ** 2)

                records.append({
                    "image_id": image_id,
                    "goat_id": goat_str_id,
                    "body_length": round(length_meas, 2),
                    "height": round(height_meas, 2),
                    "girth": round(girth_meas, 2),
                    "body_width": round(width_meas, 2),
                    "mask_area": round(mask_area, 2),
                    "solidity": round(solidity, 4),
                    "bbox_aspect_ratio": round(aspect_ratio, 3),
                    "major_axis_px": round(major_axis, 2),
                    "minor_axis_px": round(minor_axis, 2),
                    "eccentricity": round(eccentricity, 4),
                    "pixel_scale_cm": 1.0,
                    "actual_weight_kg": round(actual_weight, 2),
                })

        df = pd.DataFrame(records)

        if output_csv_path:
            df.to_csv(output_csv_path, index=False)

        return df
