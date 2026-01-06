import pandas as pd
import numpy as np
from scipy.stats import zscore

# 1. Load the raw IMU data
df = pd.read_csv("/Users/shrishtisrivastava/Desktop/Latest_ml_HFE/Consolidated_IMU_data_sheet.csv")

# 2. Clean column names (remove spaces and newlines)
df.columns = df.columns.str.replace('\n','').str.strip()

# 3. List of sensor columns
sensor_cols = ['ACC X (G)','ACC Y (G)','ACC Z (G)',
               'GYRO X (deg/s)','GYRO Y (deg/s)','GYRO Z (deg/s)']

# 4. Normalize sensor data using Z-score
df[sensor_cols] = df[sensor_cols].apply(zscore)

# 5. Split data into sliding windows
window_size = 100  # number of samples per window
step_size = 50     # overlap between windows

segments = []
movement_labels = []
condition_labels = []

for start in range(0, len(df) - window_size + 1, step_size):
    end = start + window_size
    window = df[sensor_cols].iloc[start:end].values
    
    # Extract features: mean, std, max, min for each sensor
    feat = []
    feat.extend(window.mean(axis=0))
    feat.extend(window.std(axis=0))
    feat.extend(window.max(axis=0))
    feat.extend(window.min(axis=0))
    
    segments.append(feat)
    
    # Get the most common movement and condition in this window
    movement_labels.append(df['Movement'].iloc[start:end].mode()[0])
    condition_labels.append(df['Condition'].iloc[start:end].mode()[0])

# Convert to numpy arrays
segments = np.array(segments)
movement_labels = np.array(movement_labels)
condition_labels = np.array(condition_labels)

# 6. Make column names for features
feature_names = []
for sensor in sensor_cols:
    feature_names.extend([f"{sensor}_mean", f"{sensor}_std", f"{sensor}_max", f"{sensor}_min"])

# 7. Create a new DataFrame with features and labels
df_features = pd.DataFrame(segments, columns=feature_names)
df_features['Movement'] = movement_labels
df_features['Condition'] = condition_labels

# 8. Save features as CSV for ML
df_features.to_csv("imu_features.csv", index=False)

print("CSV file 'imu_features.csv' created successfully!")
