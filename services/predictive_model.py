from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import geopandas as gpd
import numpy as np

class LandUsePredictiveModel:
    def __init__(self, training_data_path: str):
        """
        Initialize predictive model with training data
        
        :param training_data_path: Path to training dataset
        """
        self.training_data = gpd.read_file(training_data_path)
    
    def prepare_training_data(self):
        """
        Prepare features and target variables for model training
        """
        # Select relevant features for prediction
        features = self.training_data[['elevation', 'slope', 'distance_to_road', 'soil_type']]
        target = self.training_data['land_use_type']
        
        return train_test_split(features, target, test_size=0.2, random_state=42)
    
    def train_model(self):
        """
        Train random forest classifier for land use prediction
        
        :return: Trained model
        """
        X_train, X_test, y_train, y_test = self.prepare_training_data()
        
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        
        return model