# Script path: functions/cartgen_ir.py
# This script implements CARTGen-IR, a rarity-weighted CART synthesizer for generating synthetic data for imbalanced regression tasks.

## load dependency - third party
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from synthpop import MissingDataHandler, DataProcessor, CARTMethod
from synthpop.metrics import MetricsReport
from sklearn.neighbors import KernelDensity
from denseweight import DenseWeight


## load dependency - internal
from functions.relevance_function import phi
from functions.relevance_function_ctrl_pts import phi_ctrl_pts


## RarityWeightedCARTSynthesizer class
class RarityWeightedCARTSynthesizer:
    
    # Initializes the synthesizer with a DataFrame, target column, and random state.
    # The DataFrame is reset to ensure a clean index, and the target column is specified for rarity calculations.
    def __init__(self, df, target_column, random_state=4040):
        self.df = df.reset_index(drop=True)
        self.target_column = target_column
        self.random_state = random_state


    ## Computes global rarity scores based on the specified density method.
    # The method supports three density estimation techniques: 'kde_baseline', 'denseweight', and 'relevance'.
    # Each method calculates rarity scores for the target variable and normalizes them.
    # The computed rarity scores are added to the DataFrame as a new column 'global_rarity'.
    def _compute_global_rarity(self, density_method='kde_baseline', bandwidth=0.05, alpha=1.5):
        """
        Returns global rarity scores for the target baraible based on the specified density method.
        
        Parameters:
        - density_method: str, one of 'kde_baseline', 'denseweight', or 'relevance'
        - bandwidth: float, bandwidth for KDE (if using 'kde_baseline')
        - alpha: float, exponent for rarity calculation (default is 1.5)

        Returns:
        - None, but modifies self.df to include a new column 'global_rarity' with rarity scores.
        Raises:
        - ValueError if an unknown density_method is provided.
        - Warning if the relevance method produces null rarity.
        """
        target_values = self.df[self.target_column].values.reshape(-1, 1)
        
        if density_method == 'kde_baseline':
            
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
            kde.fit(target_values)
            density = np.exp(kde.score_samples(target_values))
            
            rarity = 1 / (density + 1e-5) ** alpha
            rarity = rarity / rarity.sum()
            
        elif density_method == 'denseweight': # DenseWeight method, proposed by Steininger, M., Kobs, K., Davidson, P., Krause, A., Hotho, A.: Density-based weighting for imbalanced regression. Machine Learning 110(8), 2187–2211 (2021)
            
            dw = DenseWeight(alpha)
            weights = dw.fit(target_values)
            
            rarity = weights
            rarity = rarity / rarity.sum()

            # Plotting the DenseWeight values
            plt.figure(figsize=(10, 6))
            plt.scatter(target_values, weights, c=weights, cmap='viridis', alpha=0.8)

            plt.colorbar(label="Density-based weight")
            plt.xlabel(f"{self.target_column}")
            plt.ylabel("Weight")
            plt.title("DenseWeight Values")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
        elif density_method == 'relevance': # Relevance method, proposed by Ribeiro, R.P., Moniz, N.: Imbalanced regression and extreme value prediction. Machine Learning 109(9), 1803–1835 (2020)
            
            phi_params = phi_ctrl_pts(y = self.df[self.target_column])
            relevance_values = phi(self.df[self.target_column], phi_params)
            
            rarity = np.array(relevance_values) ** alpha
            rarity = rarity / rarity.sum()
            
            # Null Rarity Verification
            if np.sum(rarity) <= 0 or np.isnan(np.sum(rarity)):
                print("Relevance method presents a uniform distribution, setting rarity to uniform distribution.")
                rarity = np.ones(len(np.array(relevance_values)))
                rarity = rarity / rarity.sum()

            # Plotting the relevance function values
            plt.figure(figsize=(10, 6))
            plt.scatter(target_values, relevance_values, c=relevance_values, cmap='viridis', alpha=0.8)

            plt.colorbar(label="Density-based weight")
            plt.xlabel(f"{self.target_column}")
            plt.ylabel("Relevance Values")
            plt.title("Relevance Function")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
        else:
            raise ValueError(f"Unknown density_method: {density_method}")
        
        self.df = self.df.copy()
        self.df['global_rarity'] = rarity


    # Generates synthetic data using a rarity-weighted resampling approach followed by CART synthesis.
    def generate_synthetic_data(self,
                                sampling_proportion = 'balance',
                                density_method = 'kde_baseline',
                                bandwidth=1.0,
                                alpha=1.0,
                                noise=True):
        """
        Generates synthetic data using a rarity-weighted resampling approach followed by CART synthesis.

        Parameters:
        - sampling_proportion: str, either 'balance' or 'extreme' to determine resampling size
        - density_method: str, one of 'kde_baseline', 'denseweight', or 'relevance' for rarity calculation
            - bandwidth: float, bandwidth for KDE (if using 'kde_baseline')
            - alpha: float, exponent for rarity calculation (default is 1.5)
        - noise: bool, whether to add Gaussian noise to numeric variables in the resampled DataFrame

        Returns:
        - augmented_df: pd.DataFrame, the combined DataFrame of original and synthetic data
        - synthetic_df: pd.DataFrame, the generated synthetic data

        Raises:
        - ValueError if an unknown density_method is provided
        - AttributeError if resampled_df is not found when adding noise
        """
        
        # ----- Step 0: Compute rarity scores -----

        self._compute_global_rarity(density_method=density_method, bandwidth=bandwidth, alpha=alpha)
        

        # ----- Step 1: Rarity-weighted resampling -----
        
        if sampling_proportion == 'balance':
            n_samples_total = int(self.df.shape[0] * 0.5)
            resample_size = int(n_samples_total / 5)
            
        elif sampling_proportion == 'extreme':
            n_samples_total = int(self.df.shape[0] * 0.75)
            resample_size = int(n_samples_total / 5)
    
        if resample_size is None:
            resample_size = len(self.df)
    
        resampled_df = self.df.sample(
            n=resample_size,
            replace=True,
            weights=self.df['global_rarity'],
            random_state=self.random_state
        )
    
        # Count how many times each original row was sampled
        self.df['resample_count'] = 0
        resample_indices = resampled_df.index.value_counts()
        self.df.loc[resample_indices.index, 'resample_count'] = resample_indices.values
        
        # Keep a copy for plotting/analysis
        self.resampled_df = resampled_df.reset_index(drop=True)
        
        if noise:
            
        # Optional: Add noise to numeric variables in the resampled DataFrame
            self.resampled_df = self.resampled_df.copy()  # Ensure no view issues
    
            self.add_noise_to_resampled(
                jitter_scale=0.01,
                only_duplicates=True,
                exclude_target=True
            )
            
            # Update local resampled_df used in pipeline
            resampled_df = self.resampled_df.copy()

        print(f"Resampled {len(resampled_df)} rows based on rarity weighting.")
        
        # ----- Step 2: Prepare data for CART -----
        # Drop metadata columns BEFORE fitting the model
        resampled_df_for_model = resampled_df.drop(columns=['global_rarity', 'resample_count'], errors='ignore')

        # CRITICAL: reset index to ensure unique indices
        resampled_df_for_model = resampled_df_for_model.reset_index(drop=True)
        
        # Handle missing data
        md_handler = MissingDataHandler()
        metadata = md_handler.get_column_dtypes(resampled_df_for_model)
        missingness_dict = md_handler.detect_missingness(resampled_df_for_model)
        real_df = md_handler.apply_imputation(resampled_df_for_model, missingness_dict)
        
        self.metadata = metadata
        self.real_df = real_df
        
        # Process the data for CART
        # This includes encoding categorical variables and scaling numeric variables
        # The DataProcessor will handle the metadata and ensure proper preprocessing
        processor = DataProcessor(metadata)
        processed_data = processor.preprocess(real_df)
    
        if processed_data.isnull().any().any():
            raise ValueError("Missing values found in processed_data — preprocessing failed!")
        
        # ----- Step 3: Fit CART -----
        cart = CARTMethod(metadata, smoothing=True, proper=False, minibucket=5, random_state=self.random_state)
        cart.fit(processed_data)
        
        # ----- Step 4: Generate synthetic data -----
        synthetic_processed = cart.sample(n_samples_total)
        synthetic_df = processor.postprocess(synthetic_processed)
    
        for col in ['global_rarity', 'resample_count']:
            if col in synthetic_df.columns:
                synthetic_df = synthetic_df.drop(columns=col)
    
        synthetic_df['resample_count'] = np.nan
        synthetic_df['global_rarity'] = np.nan
        synthetic_df['origin'] = 'synthetic'
    
        original_df = self.df.copy()
        original_df['origin'] = 'real'
    
        self.synthetic_df = synthetic_df
        self.augmented_df = pd.concat([original_df, synthetic_df], ignore_index=True)
    
        return self.augmented_df, synthetic_df
    
    
    # Adds Gaussian noise to numeric columns in the resampled dataset.
    def add_noise_to_resampled(self, jitter_scale=0.01, only_duplicates=True, exclude_target=True):
        """
        Adds Gaussian noise to numeric columns in the resampled dataset.
        This helps diversify repeated rare samples during oversampling.
    
        Parameters:
        - jitter_scale: float, standard deviation of the Gaussian noise
        - only_duplicates: bool, if True, only add noise to duplicated rows
        - exclude_target: bool, if True, do not apply noise to the target column

        Raises:
        - AttributeError if resampled_df is not found
        - ValueError if no numeric columns are available for noise addition
        - Warning if no numeric columns are found in the resampled DataFrame

        Returns:
        - None, but modifies self.resampled_df by adding noise to numeric columns.
        - Prints a message indicating how many rows were altered.
        """
        # Ensure resampled_df exists
        if not hasattr(self, 'resampled_df'):
            raise AttributeError("resampled_df not found. Run generate_synthetic_data() first.")
    
        # Select numeric columns
        numeric_cols = self.resampled_df.select_dtypes(include=[np.number]).columns.tolist()
    
        # Optionally exclude target variable
        if exclude_target and self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)
    
        if not numeric_cols:
            print("No numeric columns available for noise addition.")
            return
    
        if only_duplicates:
            # Identify duplicate rows (considering all columns)
            duplicate_mask = self.resampled_df.duplicated(keep=False)
            subset = self.resampled_df.loc[duplicate_mask, numeric_cols]
            noise = np.random.normal(loc=0, scale=jitter_scale, size=subset.shape)
            self.resampled_df.loc[duplicate_mask, numeric_cols] += noise
            print(f"Added noise to {duplicate_mask.sum()} duplicated resampled rows.")
        else:
            # Apply noise to all numeric columns
            noise = np.random.normal(loc=0, scale=jitter_scale, size=self.resampled_df[numeric_cols].shape)
            self.resampled_df[numeric_cols] += noise
            print(f"✅ Added noise to all {len(self.resampled_df)} resampled rows.")


    # Plots the distribution of many variables in the original, resampled and synthetic datasets.
    def plot_distributions(self, show_rarity=False, compare_resampled=True):
        """
        Plots the distribution of the target variable in both the original and synthetic datasets.
        Parameters:
        - show_rarity: bool, if True, plots the global rarity scores against the target variable
        - compare_resampled: bool, if True, compares the original and resampled target distributions
        Returns:
        - None, but displays plots comparing the distributions.
        Raises:
        - ValueError if the target column is not found in the DataFrame
        - Warning if synthetic_df is empty or not generated
        """

        if self.synthetic_df is None or self.synthetic_df.empty:
            print("No synthetic data to plot.")
            return

        # Plot the distribution of the target variable in the original and synthetic datasets
        plt.figure(figsize=(10, 6))
        sns.kdeplot(self.df[self.target_column], label='Real Data', fill=True, alpha=0.5)
        sns.kdeplot(self.synthetic_df[self.target_column], label='Synthetic Data', fill=True, alpha=0.5)
        plt.title('Target Variable Distribution: Real vs Synthetic Data')
        plt.xlabel(self.target_column)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot the density of the target variable in the original and resampled datasets
        if compare_resampled and hasattr(self, 'resampled_df'):
            plt.figure(figsize=(10, 6))
            sns.kdeplot(self.df[self.target_column], label='Original Data', fill=True, alpha=0.5)
            sns.kdeplot(self.resampled_df[self.target_column], label='Resampled Data', fill=True, alpha=0.5)
            plt.title('Original vs Resampled Target Distribution')
            plt.xlabel(self.target_column)
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True)
            plt.show()

        # Plot the resample count against the target variable
        if 'resample_count' in self.df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.df[self.target_column], self.df['resample_count'], alpha=0.5)
            plt.title('Resample Count vs Target Value')
            plt.xlabel(self.target_column)
            plt.ylabel('Resample Count')
            plt.grid(True)
            plt.show()

        # Plot the global rarity scores against the target variable
        if show_rarity:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.df[self.target_column], self.df['global_rarity'], alpha=0.5, label='Global Rarity')
            plt.title('Global Rarity Scores by Target Value')
            plt.xlabel(self.target_column)
            plt.ylabel('Global Rarity')
            plt.legend()
            plt.grid(True)
            plt.show()


    # Plots the distribution of the original and synthetic datasets.
    def plot_synthetic_vs_original(self):
        """
        Plots the distribution of the target variable in both the original and synthetic datasets.
        This function uses seaborn to create a KDE plot and a histogram for visual comparison.
        Returns:
        - None, but displays plots comparing the distributions.
        Raises:
        - ValueError if the target column is not found in the DataFrame
        - Warning if synthetic_df is empty or not generated
        """

        if self.synthetic_df is None or self.synthetic_df.empty:
            print("No synthetic data to plot.")
            return

        if self.augmented_df is None or self.augmented_df.empty:
            print("No augmented data to plot.")
            return

        plt.figure(figsize=(10, 6))
        sns.kdeplot(
            data=self.augmented_df, 
            x=self.target_column, 
            hue="origin", 
            fill=True, 
            common_norm=False,
            alpha=0.5
        )
        plt.title('Target Variable Distribution: Original vs Synthetic Data')
        plt.xlabel(self.target_column)
        plt.ylabel('Density')
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=self.augmented_df, 
            x=self.target_column, 
            hue="origin", 
            stat="density", 
            common_norm=False, 
            multiple="layer", 
            alpha=0.5
        )
        plt.title('Histogram: Original vs Synthetic Data')
        plt.xlabel(self.target_column)
        plt.ylabel('Density')
        plt.grid(True)
        plt.show()
    
    # Plots the distribution of the target variable in both the original and augmented datasets.
    def plot_augmented_vs_original(self):
        """
        Plots the distribution of the target variable in both the original and augmented datasets.
        This function uses seaborn to create a KDE plot and a histogram for visual comparison.
        Returns:
        - None, but displays plots comparing the distributions.
        Raises:
        - ValueError if the target column is not found in the DataFrame
        - Warning if synthetic_df is empty or not generated
        """

        if self.synthetic_df is None or self.synthetic_df.empty:
            print("No synthetic data to plot.")
            return

        if self.augmented_df is None or self.augmented_df.empty:
            print("No augmented data to plot.")
            return
        
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot the histogram for "Original" data
        ax2 = ax1.twinx()  # Create a second y-axis
        ax2.hist(self.df[self.target_column], bins=30, alpha=0.3, color='gray', label="Original", density=True)
        ax2.set_ylabel("Histogram Frequency of Original Distribution")

        # Plot KDE plots for other datasets
        sns.kdeplot(self.df[self.target_column], label="Original", ax=ax1)
        sns.kdeplot(self.augmented_df[self.target_column], label="Augmented", ax=ax1)

        # Labels
        ax1.set_xlabel(f"{self.target_column}")
        ax1.set_ylabel("Density")
        ax1.legend()

        # Show the plot
        plt.show()

    
    # Generates a metrics report comparing the real and synthetic datasets.
    def generate_metrics(self):
        """
        Generates a metrics report comparing the real and synthetic datasets.
        This function uses the MetricsReport class to compute various metrics such as
        distribution similarity, feature importance, and more.
        Returns:
        - report: MetricsReport object containing the generated metrics
        Raises:
        - ValueError if the real_df or synthetic_df is not set
        - Warning if synthetic_df is empty or not generated
        """

        if self.synthetic_df is None or self.synthetic_df.empty:
            print("No synthetic data to evaluate.")
            return None
        
        report = MetricsReport(self.real_df, self.synthetic_df, self.metadata)
        
        return report.generate_report()
    