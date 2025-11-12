# -*- coding: utf-8 -*-
"""Protein Stability Prediction - Local Version"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr, rankdata

# Try to import TensorFlow (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available, using XGBoost only")
    TF_AVAILABLE = False

# Try to import XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    print("XGBoost not available, please install it")
    XGB_AVAILABLE = False

# Bio imports
try:
    from biopandas.pdb import PandasPdb
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley
    BIO_AVAILABLE = True
except ImportError:
    print("BioPython not available, structural features will be limited")
    BIO_AVAILABLE = False

# Visualization (optional)
try:
    import py3Dmol
    VIS_AVAILABLE = True
except ImportError:
    print("py3Dmol not available, visualization disabled")
    VIS_AVAILABLE = False

class ProteinStabilityPredictor:
    def __init__(self, data_dir="."):
        self.data_dir = data_dir
        self.WT = "VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQRVNEKTKNSNRDRSGANKGPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGTNAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKPQQTKGTWFQITKFTGAAGPYCKALGSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK"
        self.amino_acid_groups = {
            'hydrophobic': ["A", "I", "L", "M", "F", "W", "Y", "V"],
            'negative': ["D", "E"],
            'positive': ["R", "H", "K"],
            'special': ["C", "G", "P"],
            'polar': ["S", "T", "N", "Q"]
        }
        
    def load_data(self):
        """Load and prepare the training and test data"""
        try:
            self.train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
            self.test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
            self.submission = pd.read_csv(os.path.join(self.data_dir, 'sample_submission.csv'))
            print(f"Training data shape: {self.train_df.shape}")
            print(f"Test data shape: {self.test_df.shape}")
        except FileNotFoundError as e:
            print(f"Error loading data files: {e}")
            print("Please make sure train.csv, test.csv, and sample_submission.csv are in the data directory")
            raise
        
        # Load updates if available
        try:
            train_updates = pd.read_csv(os.path.join(self.data_dir, 'train_updates_20220929.csv'))
            print("Train updates loaded")
        except FileNotFoundError:
            print("No train updates found")
            
        return self.train_df, self.test_df
    
    def generate_synthetic_features(self, df):
        """Generate synthetic structural features when real PDB files are not available"""
        df_synth = df.copy()
        
        # Generate synthetic B-factor based on sequence properties
        df_synth['b_factor'] = df_synth['protein_sequence'].apply(
            lambda x: np.mean([ord(c) for c in x]) / 100.0  # Simple synthetic feature
        )
        
        # Generate synthetic SASA based on sequence length and composition
        df_synth['SASA'] = df_synth['seq_len'] * 2.5 + df_synth['protein_sequence'].apply(
            lambda x: x.count('A') * 0.1  # Hydrophobic residues affect SASA
        )
        
        return df_synth
    
    def preprocess_sequence_features(self, df):
        """Extract features from protein sequences"""
        df_processed = df.copy()
        
        # Basic sequence features
        df_processed['letter_count'] = df_processed['protein_sequence'].apply(lambda x: len(set(x)))
        df_processed['seq_len'] = df_processed['protein_sequence'].apply(len)
        
        # Amino acid composition (percentages)
        all_amino_acids = list(set(self.WT))
        for aa in all_amino_acids:
            df_processed[aa] = df_processed.apply(
                lambda row: row['protein_sequence'].count(aa) / row['seq_len'], axis=1
            )
        
        # Amino acid group composition
        for group_name, group_aa in self.amino_acid_groups.items():
            df_processed[group_name] = df_processed[group_aa].sum(axis=1)
        
        return df_processed
    
    def add_structural_features(self, df, is_train=True):
        """Add structural features like B-factor and SASA"""
        if not BIO_AVAILABLE:
            print("BioPython not available, using synthetic structural features")
            return self.generate_synthetic_features(df)
            
        df_structural = df.copy()
        
        # Initialize columns
        df_structural['b_factor'] = np.nan
        df_structural['SASA'] = np.nan
        
        pdb_parser = PDBParser(QUIET=1)
        sr = ShrakeRupley()
        
        pdb_files_found = 0
        for idx, row in df_structural.iterrows():
            try:
                pdb_file = None
                if is_train and 'CIF' in row and pd.notna(row['CIF']):
                    pdb_file = f"{row['CIF']}-model_v3.pdb"
                elif not is_train:
                    mutation_info = self.get_mutation_info(row['protein_sequence'])
                    if mutation_info:
                        pdb_file = f"{mutation_info}_unrelaxed_rank_1_model_3.pdb"
                
                if pdb_file and os.path.exists(os.path.join(self.data_dir, pdb_file)):
                    # Calculate B-factor
                    atom_df = PandasPdb().read_pdb(os.path.join(self.data_dir, pdb_file)).df['ATOM']
                    df_structural.loc[idx, 'b_factor'] = atom_df.groupby('residue_number')['b_factor'].mean().mean()
                    
                    # Calculate SASA
                    structure = pdb_parser.get_structure(pdb_file, os.path.join(self.data_dir, pdb_file))
                    sr.compute(structure, level="S")
                    df_structural.loc[idx, 'SASA'] = structure.sasa
                    pdb_files_found += 1
                else:
                    # Use wildtype as fallback
                    wt_pdb = "wildtype_structure_prediction_af2.pdb"
                    if os.path.exists(os.path.join(self.data_dir, wt_pdb)):
                        atom_df = PandasPdb().read_pdb(os.path.join(self.data_dir, wt_pdb)).df['ATOM']
                        df_structural.loc[idx, 'b_factor'] = atom_df['b_factor'].mean()
                        
                        structure = pdb_parser.get_structure(wt_pdb, os.path.join(self.data_dir, wt_pdb))
                        sr.compute(structure, level="S")
                        df_structural.loc[idx, 'SASA'] = structure.sasa
                    
            except Exception as e:
                if idx < 3:  # Only print first few errors
                    print(f"Error processing row {idx}: {e}")
                continue
        
        print(f"Processed PDB files for {pdb_files_found} sequences")
        
        # Fill missing values with synthetic features
        if df_structural['b_factor'].isna().any():
            synth_features = self.generate_synthetic_features(df_structural[df_structural['b_factor'].isna()])
            df_structural.loc[df_structural['b_factor'].isna(), 'b_factor'] = synth_features['b_factor']
            df_structural.loc[df_structural['SASA'].isna(), 'SASA'] = synth_features['SASA']
        
        return df_structural
    
    def get_mutation_info(self, sequence):
        """Extract mutation information from sequence"""
        if len(sequence) != len(self.WT):
            return None
            
        for i, (wt_aa, mut_aa) in enumerate(zip(self.WT, sequence)):
            if wt_aa != mut_aa:
                return f"{wt_aa}{i+1}{mut_aa}"
        return None
    
    def prepare_features(self, train_df, test_df, use_structural_features=True):
        """Prepare features for model training"""
        print("Preprocessing sequence features...")
        X_train = self.preprocess_sequence_features(train_df)
        X_test = self.preprocess_sequence_features(test_df)
        
        if use_structural_features:
            print("Adding structural features...")
            X_train = self.add_structural_features(X_train, is_train=True)
            X_test = self.add_structural_features(X_test, is_train=False)
        else:
            # Add basic synthetic features
            X_train = self.generate_synthetic_features(X_train)
            X_test = self.generate_synthetic_features(X_test)
        
        # Columns to drop
        cols_to_drop = ['protein_sequence', 'seq_id', 'data_source', 'pH']
        cols_to_drop = [col for col in cols_to_drop if col in X_train.columns]
        
        X_train_clean = X_train.drop(columns=cols_to_drop, errors='ignore')
        X_test_clean = X_test.drop(columns=cols_to_drop, errors='ignore')
        
        # Align columns
        common_cols = X_train_clean.columns.intersection(X_test_clean.columns)
        X_train_clean = X_train_clean[common_cols]
        X_test_clean = X_test_clean[common_cols]
        
        print(f"Final training features: {X_train_clean.shape}")
        print(f"Final test features: {X_test_clean.shape}")
        
        # Target variable
        y_train = X_train['tm'] if 'tm' in X_train.columns else None
        
        return X_train_clean, X_test_clean, y_train
    
    def train_xgboost_model(self, X_train, y_train):
        """Train XGBoost model"""
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is not available. Please install it with: pip install xgboost")
        
        # Split data
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {X_tr.shape}, Validation set: {X_val.shape}")
        
        # Train model
        model = xgb.XGBRegressor(
            n_estimators=100,  # Reduced for faster training
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        print("Training XGBoost model...")
        model.fit(X_tr, y_tr)
        
        # Validate
        val_pred = model.predict(X_val)
        spearman_corr = spearmanr(y_val, val_pred)[0]
        mae = mean_absolute_error(y_val, val_pred)
        
        print(f"Validation MAE: {mae:.4f}")
        print(f"Validation Spearman Correlation: {spearman_corr:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))
        
        return model
    
    def train_neural_network(self, X_train, y_train):
        """Train neural network model (optional)"""
        if not TF_AVAILABLE:
            print("TensorFlow not available, skipping neural network")
            return None
            
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Build model
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        print("Training neural network...")
        history = model.fit(
            X_train_scaled, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        return model, history, scaler
    
    def create_submission(self, model, X_test, output_file='submission.csv'):
        """Create submission file"""
        predictions = model.predict(X_test)
        
        submission_df = self.submission.copy()
        submission_df['tm'] = predictions
        
        submission_df.to_csv(os.path.join(self.data_dir, output_file), index=False)
        print(f"Submission file saved as {output_file}")
        print(f"Predictions range: {predictions.min():.2f} to {predictions.max():.2f}")
        
        return submission_df
    
    def visualize_structure(self, pdb_file="wildtype_structure_prediction_af2.pdb"):
        """Visualize protein structure"""
        if not VIS_AVAILABLE:
            print("py3Dmol not available for visualization")
            return None
            
        pdb_path = os.path.join(self.data_dir, pdb_file)
        if not os.path.exists(pdb_path):
            print(f"PDB file {pdb_file} not found in {self.data_dir}")
            return None
            
        try:
            with open(pdb_path) as f:
                protein_structure = "".join([x for x in f])
            
            view = py3Dmol.view(width=800, height=600)
            view.addModelsAsFrames(protein_structure)
            style = {'cartoon': {'color': 'spectrum'}, 'stick': {}}
            view.setStyle({'model': -1}, style)
            view.zoom(0.12)
            return view
        except Exception as e:
            print(f"Error visualizing structure: {e}")
            return None

    def exploratory_data_analysis(self, df):
        """Perform basic exploratory data analysis"""
        if 'tm' not in df.columns:
            print("No target variable 'tm' for EDA")
            return
            
        print("\n=== Exploratory Data Analysis ===")
        print(f"Target variable (tm) statistics:")
        print(df['tm'].describe())
        
        # Plot target distribution
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        df['tm'].hist(bins=30)
        plt.title('Distribution of Tm values')
        plt.xlabel('Tm')
        plt.ylabel('Frequency')
        
        if 'pH' in df.columns:
            plt.subplot(1, 2, 2)
            plt.scatter(df['pH'], df['tm'], alpha=0.5)
            plt.title('Tm vs pH')
            plt.xlabel('pH')
            plt.ylabel('Tm')
        
        plt.tight_layout()
        plt.show()

def check_installation():
    """Check if required packages are installed"""
    packages = {
        'numpy': np,
        'pandas': pd,
        'scikit-learn': None,
        'scipy': None,
        'xgboost': xgb if 'xgb' in globals() else None,
        'biopandas': PandasPdb if 'PandasPdb' in globals() else None,
        'biopython': PDBParser if 'PDBParser' in globals() else None,
    }
    
    print("Checking package availability:")
    for pkg, obj in packages.items():
        try:
            if obj is not None:
                print(f"✓ {pkg}")
            else:
                __import__(pkg.replace('-', '_'))
                print(f"✓ {pkg}")
        except ImportError:
            print(f"✗ {pkg}")

def main():
    """Main execution function"""
    print("Protein Stability Prediction - Local Version")
    
    # Check installations
    check_installation()
    
    # Initialize predictor
    data_directory = input("Enter path to data directory (or press Enter for current directory): ").strip()
    if not data_directory:
        data_directory = "."
    
    predictor = ProteinStabilityPredictor(data_dir=data_directory)
    
    try:
        # Load data
        train_df, test_df = predictor.load_data()
        
        # Perform EDA
        predictor.exploratory_data_analysis(train_df)
        
        # Ask user for feature options
        use_structural = input("Use structural features? (y/n, default n): ").strip().lower() == 'y'
        
        # Prepare features
        X_train, X_test, y_train = predictor.prepare_features(
            train_df, test_df, 
            use_structural_features=use_structural
        )
        
        # Train model
        if XGB_AVAILABLE:
            model = predictor.train_xgboost_model(X_train, y_train)
            
            # Create submission
            submission = predictor.create_submission(model, X_test)
            
            # Show sample predictions
            print("\nSample predictions:")
            print(submission.head(10))
        else:
            print("XGBoost not available. Please install it to run the model.")
        
        # Optional: Visualize structure
        if VIS_AVAILABLE:
            visualize = input("Visualize protein structure? (y/n, default n): ").strip().lower() == 'y'
            if visualize:
                view = predictor.visualize_structure()
                if view:
                    view.show()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the data files are in the correct directory")
        return None, None
    
    return model, submission

if __name__ == "__main__":
    model, submission = main()