import os
import sys
import torch
import pandas as pd
import logging

# Ensure root directory is on the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.train_image_features import ImageFeaturesMLP

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def demonstrate_inference(csv_131: str, csv_193: str, model_path: str):
    logger.info("--- End-to-End Inference Demonstration ---")
    
    # 1. Load the Trained Model
    logger.info("1. Loading the trained classification model...")
    model = ImageFeaturesMLP(input_size=2560, num_classes=5)
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}. Please train the model first.")
        return
        
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode
    logger.info("   Model loaded successfully.")
    
    # 2. Load a Single Sample from the Datasets
    logger.info("2. Fetching a sample from the datasets...")
    df_131 = pd.read_csv(csv_131)
    df_193 = pd.read_csv(csv_193)
    
    # We will pick the first flare sequence to test
    sample_flare_id = df_131['flare_id'].iloc[0]
    sample_time = df_131['timestamp'].iloc[0]
    
    row_131 = df_131[(df_131['flare_id'] == sample_flare_id) & (df_131['timestamp'] == sample_time)]
    
    # Convert timestamp to roughly match 193 if needed, as we did in training
    time_rounded = pd.to_datetime(sample_time).round('10min')
    df_193['time_rounded'] = pd.to_datetime(df_193['timestamp']).dt.round('10min')
    row_193 = df_193[(df_193['flare_id'] == sample_flare_id) & (df_193['time_rounded'] == time_rounded)]
    
    if row_131.empty or row_193.empty:
        logger.error("Could not find a matching sample across both datasets.")
        return
        
    # 3. Extract the Features
    logger.info(f"   Found matching sample for Flare ID: {sample_flare_id}")
    logger.info("3. Extracting 2560-dimensional features...")
    
    f131_cols = [f'f{i}' for i in range(1280)]
    f193_cols = [f'f{i}' for i in range(1280)]
    
    features_131 = row_131.iloc[0][f131_cols].values.astype(float)
    features_193 = row_193.iloc[0][f193_cols].values.astype(float)
    
    import numpy as np
    combined_features = np.concatenate([features_131, features_193])
    
    # Convert to PyTorch tensor and add a batch dimension (1, 2560)
    input_tensor = torch.tensor(combined_features, dtype=torch.float32).unsqueeze(0)
    
    # 4. Run Inference (Prediction)
    logger.info("4. Running the model prediction...")
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
    # 5. Output the Results
    logger.info("=========================================")
    logger.info(f"Input Flare ID  : {sample_flare_id}")
    logger.info(f"Input Timestamp : {sample_time}")
    logger.info(f"Class Probabilities: {probabilities[0].numpy().round(3)}")
    logger.info(f"Predicted Class : {predicted_class} (Dummy Category)")
    logger.info("=========================================")
    logger.info("Demonstration complete!")

if __name__ == "__main__":
    csv_131 = "data/features/spectral/image_features_131.csv"
    csv_193 = "data/features/spectral/image_features_193.csv"
    model_path = "models/image_features/mlp_model.pt"
    
    demonstrate_inference(csv_131, csv_193, model_path)
