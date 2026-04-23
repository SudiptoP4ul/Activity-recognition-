This project implements a robust machine learning pipeline for classifying human activities based on accelerometer, gyroscope, and barometer data. It addresses common real-world challenges in sensor data collection, such as hardware synchronization and varying recording lengths.

🚀 Key Features
Multimodal Sensor Fusion: Combines linear acceleration, angular velocity, and atmospheric pressure to improve classification accuracy.

Robust Synchronization: Employs an "Accelerometer-as-Master" strategy to align sensors with different start times and sampling rates.

Intelligent Feature Engineering: Utilizes custom features like Pressure Variance and Mean Gyro Magnitude to distinguish between physically similar activities (e.g., Elevator vs. Stairs).

Sliding Window Processing: Implements a 100-sample window with 50% overlap for capturing temporal patterns.

📊 Experimental Rationale (TO DO 7)
To solve the "between-class" confusion inherent in vertical movement, the following features were engineered:

Mean Gyroscope Magnitude: Captures the "angular energy" of the body. It effectively distinguishes Running (high rotation/arm swing) from Elevator rides (near-zero rotation).

Pressure Variance (Jitter): Captures the texture of altitude change. While both elevators and stairs change altitude, the Stairs activity introduces rhythmic human "bounce" (high variance), whereas an Elevator is a smooth mechanical lift (low variance).

Frequency Domain (FFT): Identifies the dominant cadence of the user's stride to separate walking speeds from running.

🛠️ Installation & Usage
Prerequisites

Python 3.x

Dependencies: pandas, numpy, matplotlib, scikit-learn, statsmodels

Data Structure

Place your sensor files in the following directory structure:
A3_Data/
└── [net_id]/
    ├── [activity]_accel.txt
    ├── [activity]_gyro.txt
    └── [activity]_pressure.txt

Running the Pipeline

The core logic for data processing and feature extraction can be executed via the main script:
import os
import pandas as pd
import numpy as np
from collections import Counter

# Define activity mapping
activity_indices = {
    'Stationary': 0, 'Walking-flat-surface': 1, 'Walking-up-stairs': 2,
    'Walking-down-stairs': 3, 'Running': 4, 'Elevator-up': 5, 'Elevator-down': 6
}

# The backbone: Robust sensor synchronization logic
def compute_raw_data_robust(path, activity):
    # Load accelerometer as the master clock
    accel_raw = pd.read_csv(get_sensor_file(path, activity, 'accel'), names=['time', 'x', 'y', 'z'])
    t_min, t_max = accel_raw['time'].min(), accel_raw['time'].max()
    
    # Auto-detect units (ms vs seconds)
    step = 20 if (t_max - t_min) > 1000 else 0.02
    timestamps = np.arange(t_min + 500, t_max - 500, step) 
    
    # Interpolate all sensors onto common timestamps
    # ... (full code available in a3.ipynb)

📈 Results & Analysis
The pipeline generates three primary visualizations:

Mean Gyro Magnitude: Visualizes rotational intensity across activities.

Pressure Variance: Visualizes the "jitter" signature of physical climbing vs. mechanical lifting.

Activity Timeline: A ground-truth staircase plot showing successful processing of all input files.

📝 Evaluation
The model is evaluated using:

5-Fold Stratified Cross-Validation: Ensures the model generalizes across different segments of the data.

Between-Class Confusion Matrix: Used to identify and mitigate confusion between similar activities like "Walking-up-stairs" and "Elevator-up".
