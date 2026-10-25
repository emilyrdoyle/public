#%% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
import warnings
import seaborn as sns
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import math
from itertools import product
from scipy.optimize import differential_evolution

warnings.filterwarnings('ignore')

#%% Data Loading

files = [
    'ExpDat_20260119_Fe_cycling_porous_500C.csv',
    'ExpDat_20260120_Fe_cycling_porous_700C.csv',
    'ExpDat_20260121_Fe_cycling_porous_900C.csv',
    'ExpDat_20260121_Fe_cycling_spherical_500C.csv',
    'ExpDat_20260126_Fe_cycling_spherical_700C.csv',
    'ExpDat_20260129_Fe_cycling_spherical_900C.csv'
]

fe_masses = [45.9, 41.9, 45.4, 43.1, 66.0, 74.4] 
types = ['Porous', 'Porous', 'Porous', 'Spherical', 'Spherical', 'Spherical']
temps = [500, 700, 900, 500, 700, 900]
max_oxy_capacity = 0.4297  

# files = [
#     'ExpDat_20260121_Fe_cycling_spherical_500C.csv',
#     'ExpDat_20260126_Fe_cycling_spherical_700C.csv',
#     'ExpDat_20260129_Fe_cycling_spherical_900C.csv'
# ]

# fe_masses = [43.1, 66.0, 74.4] 
# types = [ 'Spherical', 'Spherical', 'Spherical']
# temps = [ 500, 700, 900]
# max_oxy_capacity = 0.4297  

# files = [
#     'ExpDat_20260119_Fe_cycling_porous_500C.csv',
#     'ExpDat_20260120_Fe_cycling_porous_700C.csv',
#     'ExpDat_20260121_Fe_cycling_porous_900C.csv',
# ]

# fe_masses = [45.9, 41.9, 45.4] 
# types = ['Porous', 'Porous', 'Porous']
# temps = [500, 700, 900]
# max_oxy_capacity = 0.4297  


#%% DATA EXTRACTION FUNCTION

def extract_cycles(time, alpha, col4, col5):
    ox_active = (col4 == 10).astype(int)
    ox_starts = np.where(np.diff(ox_active, prepend=0) == 1)[0]
    ox_ends = np.where(np.diff(ox_active, append=0) == -1)[0]

    red_active = (col5 == 10).astype(int)
    red_starts = np.where(np.diff(red_active, prepend=0) == 1)[0]
    red_ends = np.where(np.diff(red_active, append=0) == -1)[0]

    cycles = []
    for i in range(min(len(ox_starts), len(red_starts))):
        o_s, o_e = ox_starts[i], ox_ends[i]
        r_s, r_e = red_starts[i], red_ends[i]
        
        # Tau_ox
        a_s_ox, a_p = alpha[o_s], alpha[o_e]
        t_ox_slice = time[o_s : o_e + 1]
        target_ox = a_s_ox + 0.632 * (a_p - a_s_ox)
        idx_ox = np.argmax(alpha[o_s : o_e + 1] >= target_ox)
        tau_ox = -(t_ox_slice[idx_ox] - t_ox_slice[0]) if (t_ox_slice[idx_ox] - t_ox_slice[0]) > 0 else -1e-6
        
        # TIME NORMALIZATION (Phase duration mapped to 0-1)
        t_ox_norm = (t_ox_slice - t_ox_slice[0]) / (t_ox_slice[-1] - t_ox_slice[0] + 1e-9)
        
        # Tau_red
        a_s_red, a_e_red = alpha[r_s], alpha[r_e]
        t_red_slice = time[r_s : r_e + 1]
        target_red = a_s_red - 0.632 * (a_s_red - a_e_red)
        idx_red = np.argmax(alpha[r_s : r_e + 1] <= target_red)
        tau_red = t_red_slice[idx_red] - t_red_slice[0]
        if tau_red <= 0: tau_red = 1e-6
        
        t_red_norm = (t_red_slice - t_red_slice[0]) / (t_red_slice[-1] - t_red_slice[0] + 1e-9)
        
        cycles.append({
            'cycle': i + 1,  
            'tau_ox': tau_ox,
            'tau_red': tau_red,
            'peak_alpha': a_p,
            't_start': time[o_s],
            't_peak': time[o_e],
            't_end': time[r_e],
            'raw_time': time[o_s : r_e + 1],
            'raw_alpha': alpha[o_s : r_e + 1],
            'ox_duration': time[o_e] - time[o_s],
            'red_duration': time[r_e] - time[r_s],
            'norm_time_ox': t_ox_norm,
            'norm_time_red': t_red_norm,
        })
    return cycles

# PROCESS THE FILES
all_cycles = []
for i, (file, fe_mass, typ, temp) in enumerate(zip(files, fe_masses, types, temps)):
    try:
        df = pd.read_csv(file, skiprows=71, header=None, encoding='latin1')
        time, mass, c4, c5 = df.iloc[:, 1].values, df.iloc[:, 2].values, df.iloc[:, 3].values, df.iloc[:, 4].values
        time -= time[0]
        alpha = np.clip((mass - mass[0]) / (fe_mass * max_oxy_capacity), 0, 1)
        cycs = extract_cycles(time, alpha, c4, c5)
        for c in cycs:
            c.update({'exp_id': f"{typ}_{temp}", 'porosity': typ, 'temperature': temp})
            all_cycles.append(c)
    except FileNotFoundError: pass

df_cycles = pd.DataFrame(all_cycles)

#%% RF Model TRAINING

df_cycles['is_spherical'] = (df_cycles['porosity'] == 'Spherical').astype(int)
df_cycles['is_porous'] = (df_cycles['porosity'] == 'Porous').astype(int)
df_cycles['spherical_cycle'] = df_cycles['is_spherical'] * df_cycles['cycle']
df_cycles['porous_cycle'] = df_cycles['is_porous'] * df_cycles['cycle']

df_cycles = df_cycles.sort_values(['exp_id', 'cycle'])

lag_cols = ['tau_ox', 'tau_red', 'peak_alpha']
for col in lag_cols:
    df_cycles[f'{col}_prev'] = df_cycles.groupby('exp_id')[col].shift(1)
    df_cycles[f'{col}_prev'] = df_cycles[f'{col}_prev'].fillna(df_cycles[col])

df_cycles['porous_tau_ox_prev'] = df_cycles['is_porous'] * df_cycles['tau_ox_prev']
df_cycles['porous_tau_red_prev'] = df_cycles['is_porous'] * df_cycles['tau_red_prev']
df_cycles['spherical_tau_ox_prev'] = df_cycles['is_spherical'] * df_cycles['tau_ox_prev']
df_cycles['spherical_tau_red_prev'] = df_cycles['is_spherical'] * df_cycles['tau_red_prev']
df_cycles['spherical_peak_alpha_prev'] = df_cycles['is_spherical'] * df_cycles['peak_alpha_prev']
df_cycles['porous_peak_alpha_prev'] = df_cycles['is_porous'] * df_cycles['peak_alpha_prev']

features_with_lag = [
    'porosity', 'temperature', 'porous_cycle', 'spherical_cycle',
    'porous_tau_ox_prev', 'porous_tau_red_prev', 'spherical_tau_ox_prev', 
    'spherical_tau_red_prev', 'porous_peak_alpha_prev', 'spherical_peak_alpha_prev'
]

targets = ['tau_ox', 'tau_red', 'peak_alpha']

X = df_cycles[features_with_lag]
y = df_cycles[targets]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

preprocessor_rf = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['porosity']),
    ('num', MinMaxScaler(), [
        'temperature', 'porous_cycle', 'spherical_cycle', 
        'porous_tau_ox_prev', 'porous_tau_red_prev', 'spherical_tau_ox_prev',
        'spherical_tau_red_prev', 'porous_peak_alpha_prev', 'spherical_peak_alpha_prev'
    ]) 
])

pipeline_rf = Pipeline([
    ('prep', preprocessor_rf),
    ('rf', RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42))
])

pipeline_rf.fit(X_train, y_train)

df_cycles[[f'pred_{t}' for t in targets]] = pipeline_rf.predict(X)

#%% RF Plotting

def plot_triple_correlation(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    targets_info = [
        ('tau_ox', 'pred_tau_ox', r'$\tau_{ox}$ (Kinetics)'),
        ('tau_red', 'pred_tau_red', r'$\tau_{red}$ (Kinetics)'),
        ('peak_alpha', 'pred_peak_alpha', r'Peak Conversion ($\alpha$)')
    ]
    for ax, (true_col, pred_col, title) in zip(axes, targets_info):
        r2 = r2_score(df[true_col], df[pred_col])
        rmse = np.sqrt(mean_squared_error(df[true_col], df[pred_col]))
        ax.scatter(df[true_col], df[pred_col], color='crimson', alpha=0.4, edgecolors='none', s=25)
        lims = [df[true_col].min(), df[true_col].max()]
        ax.plot(lims, lims, 'k--', lw=2, alpha = 0.3)
        ax.set_title(f'{title}\n$R^2$: {r2:.4f} | RMSE: {rmse:.4f}')
        ax.set_xlabel('Experimental'); ax.set_ylabel('Predicted'); ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout(); plt.show()

def plot_importance(model, X_val, y_val):
    results = permutation_importance(pipeline_rf, X_val, y_val, n_repeats=10, random_state=42)
    sorted_idx = results.importances_mean.argsort()
    
    plt.figure(figsize=(10, 6))
    plt.barh(X_val.columns[sorted_idx], results.importances_mean[sorted_idx], color='crimson')
    plt.xlabel("Permutation Importance (Drop in R^2)")
    plt.title("Feature Importance: Recursive Model")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    

def plot_reconstruction(df, exp_target, n_cycles=5):
    exp_data = df[(df['exp_id'] == exp_target) & (df['cycle'] <= n_cycles)].sort_values('cycle')
    plt.figure(figsize=(12, 5))
    plt.grid(True, alpha=0.3)
    t_off, last_a_p = 0, 0
    for _, row in exp_data.iterrows():
        t_raw, a_raw = row['raw_time'], row['raw_alpha']
        r_start = row['t_end'] - row['red_duration']
        ox_m, red_m = t_raw <= row['t_peak'], t_raw >= r_start
        t_ox_s = t_raw[ox_m] - row['t_start'] + t_off
        t_red_s = t_raw[red_m] - r_start + t_ox_s[-1]
        plt.plot(np.concatenate([t_ox_s, t_red_s]), np.concatenate([a_raw[ox_m], a_raw[red_m]]), 'k', alpha=0.3, lw=2)
        tau_o, tau_r, pk = min(row['pred_tau_ox'], -1e-6), max(row['pred_tau_red'], 1e-6), row['pred_peak_alpha']
        t_ox_array = t_raw[ox_m] - row['t_start']
        
        t_ox_max = t_ox_array.max()
        # Updated to normalize based on max alpha reached (april 6)
        max_exp_ox = max(1 - np.exp(t_ox_max / tau_o), 1e-6) # Prevent division by zero
        a_ox_p = last_a_p + (pk - last_a_p) * ((1 - np.exp(t_ox_array / tau_o)) / max_exp_ox)
        a_red_p = pk * np.exp(-(t_raw[red_m] - r_start) / tau_r)
        
        plt.plot(np.concatenate([t_ox_s, t_red_s]), np.concatenate([a_ox_p, a_red_p]), color = 'crimson',linestyle='--', lw=2)
        t_off, last_a_p = t_red_s[-1], a_red_p[-1]
        
    plt.title(f'Reconstruction: {exp_target}')
    plt.ylabel(r'Conversion ($\alpha$)')
    plt.grid(True)
    plt.show()
    
def plot_correlation_heatmap(df):
    inputs = [
        'temperature', 'porous_cycle', 'spherical_cycle', 
        'tau_ox_prev', 'tau_red_prev', 'peak_alpha_prev'
    ]
    outputs = ['tau_ox', 'tau_red', 'peak_alpha']
    
    # Ensure all columns exist
    all_cols = inputs + outputs
    valid_cols = [c for c in all_cols if c in df.columns]
    
    # Calculate the full correlation matrix
    full_corr = df[valid_cols].corr()
    
    # Slice the matrix to show only Inputs (rows) vs Outputs (columns)
    # This is the "Input-Output Correlation" view
    io_corr = full_corr.loc[inputs, outputs]

    plt.figure(figsize=(10, 8))
    sns.heatmap(full_corr, 
                annot=True, 
                fmt=".2f", 
                cmap='RdBu_r', 
                center=0, 
                linewidths=1,
                square=False) 
    
    #plt.title("Recursive Model: Input-Output Feature Correlations", fontsize=14)
   # plt.ylabel("Input Features (Including Lagged States)")
    #plt.xlabel("Model Targets (Current Cycle)")
    plt.tight_layout()
    plt.show()
    
# Execution
if not df_cycles.empty:
    plot_triple_correlation(df_cycles)
    plot_importance(pipeline_rf, X_test, y_test)
    plot_reconstruction(df_cycles, 'Spherical_900', n_cycles=5)
    plot_correlation_heatmap(df_cycles)
    
#%% LR Model TRAINING

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor_lr = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), ['porosity']),
    ('num', MinMaxScaler(), ['temperature', 'porous_cycle', 'spherical_cycle', 
                             'porous_tau_ox_prev', 'porous_tau_red_prev','spherical_tau_ox_prev',
                             'spherical_tau_red_prev', 'porous_peak_alpha_prev','spherical_peak_alpha_prev']) 
])
pipeline_lr = Pipeline([
    ('prep', preprocessor_lr),
    ('model', LinearRegression())
])

pipeline_lr.fit(X_train, y_train)

lr_preds = pipeline_lr.predict(X)
for i, t in enumerate(targets):
    df_cycles[f'pred_lr_{t}'] = lr_preds[:, i]

# Calculate metrics on test set specifically
test_preds = pipeline_lr.predict(X_test_lr)
mse = mean_squared_error(y_test_lr, test_preds)
r2 = r2_score(y_test_lr, test_preds)

print(f"Linear Regression MSE: {mse:.6f}")
print(f"Linear Regression R2: {r2:.4f}")

#%% LR Plotting

def plot_lr_correlation(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    targets_info = [
        ('tau_ox', 'pred_lr_tau_ox', r'$\tau_{ox}$ (Kinetics)'),
        ('tau_red', 'pred_lr_tau_red', r'$\tau_{red}$ (Kinetics)'),
        ('peak_alpha', 'pred_lr_peak_alpha', r'Peak Conversion ($\alpha$)')
    ]
    for ax, (true_col, pred_col, title) in zip(axes, targets_info):
        r2 = r2_score(df[true_col], df[pred_col])
        ax.scatter(df[true_col], df[pred_col], color='blue', alpha=0.4, s=25)
        lims = [df[true_col].min(), df[true_col].max()]
        ax.plot(lims, lims, 'k--', lw=2, alpha = 0.3)
        ax.set_title(f'LR: {title}\n$R^2$: {r2:.4f}')
        ax.set_xlabel('Experimental'); ax.set_ylabel('Predicted'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

def plot_lr_reconstruction(df, exp_target, n_cycles=5):
    exp_data = df[(df['exp_id'] == exp_target) & (df['cycle'] <= n_cycles)].sort_values('cycle')
    plt.figure(figsize=(12, 5))
    plt.grid(True, alpha=0.3)
    t_off, last_a_p = 0, 0
    for _, row in exp_data.iterrows():
        t_raw, a_raw = row['raw_time'], row['raw_alpha']
        r_start = row['t_end'] - row['red_duration']
        ox_m, red_m = t_raw <= row['t_peak'], t_raw >= r_start
        t_ox_s = t_raw[ox_m] - row['t_start'] + t_off
        t_red_s = t_raw[red_m] - r_start + t_ox_s[-1]
        
        plt.plot(np.concatenate([t_ox_s, t_red_s]), np.concatenate([a_raw[ox_m], a_raw[red_m]]), color='black', alpha=0.2, lw=3)
        
        # Predictions using LR columns
        tau_o, tau_r, pk = min(row['pred_lr_tau_ox'], -1e-6), max(row['pred_lr_tau_red'], 1e-6), row['pred_lr_peak_alpha']
        
        # updated the same as RF plotting function
        t_ox_array = t_raw[ox_m] - row['t_start']
        t_ox_max = t_ox_array.max()
        max_exp_ox = max(1 - np.exp(t_ox_max / tau_o), 1e-6) 
        a_ox_p = last_a_p + (pk - last_a_p) * ((1 - np.exp(t_ox_array / tau_o)) / max_exp_ox)
        a_red_p = pk * np.exp(-(t_raw[red_m] - r_start) / tau_r)
        
        plt.plot(np.concatenate([t_ox_s, t_red_s]), np.concatenate([a_ox_p, a_red_p]), 'b--', lw=1.5)
        t_off, last_a_p = t_red_s[-1], a_red_p[-1]
        
    plt.title(f'LR Reconstruction: {exp_target}')
    plt.ylabel(r'Conversion ($\alpha$)')
    plt.grid(True)
    plt.show()
    
    
def plot_lr_coefficients(pipeline):
    preprocessor = pipeline.named_steps['prep']
    lr_model = pipeline.named_steps['model']
    
    all_feature_names = preprocessor.get_feature_names_out()
    cleaned_names = []
    for name in all_feature_names:
        clean_name = name.split('__')[-1]
        if 'porosity' in clean_name:
            clean_name = 'porosity'
        cleaned_names.append(clean_name)
    
    targets = ['tau_ox', 'tau_red', 'peak_alpha']
    coef_df = pd.DataFrame(
        lr_model.coef_, 
        columns=cleaned_names, 
        index=targets
    ).T # Transpose so features are on the Y-axis
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(coef_df, 
                annot=True, 
                fmt=".3f", 
                cmap='RdBu_r', 
                center=0, 
                linewidths=1,
                cbar_kws={'label': 'Coefficient Weight'})
    
    plt.tight_layout()
    plt.show()


plot_lr_correlation(df_cycles)
plot_lr_reconstruction(df_cycles, 'Spherical_900', n_cycles=5)
plot_lr_coefficients(pipeline_lr)
    























#%% SVM Model training

preprocessor_svm = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['porosity']),
    ('num', MinMaxScaler(), [
        'temperature', 'porous_cycle', 'spherical_cycle', 
        'porous_tau_ox_prev', 'porous_tau_red_prev', 'spherical_tau_ox_prev',
        'spherical_tau_red_prev', 'porous_peak_alpha_prev', 'spherical_peak_alpha_prev'
    ]) 
])

pipeline_svm = Pipeline([
    ('prep', preprocessor_svm),
    ('svm', MultiOutputRegressor(SVR(kernel='rbf', C=10, epsilon=0.01, gamma='scale')))
])

pipeline_svm.fit(X_train, y_train)

svm_preds = pipeline_svm.predict(X)
for i, t in enumerate(targets):
    df_cycles[f'pred_svm_{t}'] = svm_preds[:, i]

svm_test_preds = pipeline_svm.predict(X_test)
print(f"SVM Test R2: {r2_score(y_test, svm_test_preds):.4f}")
print(f"SVM Test MSE: {mean_squared_error(y_test, svm_test_preds):.6f}")


#%% SVM Plotting


def plot_svm_correlation(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    targets_info = [
        ('tau_ox', 'pred_svm_tau_ox', r'$\tau_{ox}$ (Kinetics)'),
        ('tau_red', 'pred_svm_tau_red', r'$\tau_{red}$ (Kinetics)'),
        ('peak_alpha', 'pred_svm_peak_alpha', r'Peak Conversion ($\alpha$)')
    ]
    for ax, (true_col, pred_col, title) in zip(axes, targets_info):
        r2 = r2_score(df[true_col], df[pred_col])
        ax.scatter(df[true_col], df[pred_col], color='forestgreen', alpha=0.4, s=25)
        lims = [df[true_col].min(), df[true_col].max()]
        ax.plot(lims, lims, 'k--', lw=2, alpha = 0.3)
        ax.set_title(f'SVM: {title}\n$R^2$: {r2:.4f}')
        ax.set_xlabel('Experimental'); ax.set_ylabel('Predicted'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()
    
    
def plot_svm_reconstruction(df, exp_target, n_cycles=5):
    exp_data = df[(df['exp_id'] == exp_target) & (df['cycle'] <= n_cycles)].sort_values('cycle')
    
    plt.figure(figsize=(12, 5))
    plt.grid(True, alpha=0.3)
    
    t_off, last_a_p = 0, 0
    
    for _, row in exp_data.iterrows():
        t_raw, a_raw = row['raw_time'], row['raw_alpha']
        r_start = row['t_end'] - row['red_duration']
        ox_m = t_raw <= row['t_peak']
        red_m = t_raw >= r_start
        t_ox_s = t_raw[ox_m] - row['t_start'] + t_off
        t_red_s = t_raw[red_m] - r_start + t_ox_s[-1]
        plt.plot(np.concatenate([t_ox_s, t_red_s]), 
                 np.concatenate([a_raw[ox_m], a_raw[red_m]]), 
                 color='black', alpha=0.2, lw=3, label='Experimental' if row['cycle']==1 else "")
        
        # Get SVM specific columns
        tau_o = min(row['pred_svm_tau_ox'], -1e-6)
        tau_r = max(row['pred_svm_tau_red'], 1e-6)
        pk = row['pred_svm_peak_alpha']
        
        # normalization fix
        t_ox_array = t_raw[ox_m] - row['t_start']
        t_ox_max = t_ox_array.max()
        max_exp_ox = max(1 - np.exp(t_ox_max / tau_o), 1e-6) 
        a_ox_p = last_a_p + (pk - last_a_p) * ((1 - np.exp(t_ox_array / tau_o)) / max_exp_ox)
        a_red_p = pk * np.exp(-(t_raw[red_m] - r_start) / tau_r)
        
        # Plot
        plt.plot(np.concatenate([t_ox_s, t_red_s]), 
                 np.concatenate([a_ox_p, a_red_p]), 
                 color='forestgreen', linestyle='--', lw=1.5, label='SVM Prediction' if row['cycle']==1 else "")
        
        # Update offsets for the next cycle
        t_off, last_a_p = t_red_s[-1], a_red_p[-1]
        
    plt.title(f'SVM Reconstruction: {exp_target}')
    plt.xlabel('Cumulative Time (s)')
    plt.ylabel(r'Conversion ($\alpha$)')
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_svm_correlation(df_cycles)
plot_svm_reconstruction(df_cycles, 'Spherical_900', n_cycles=5)



def plot_relationships(df):
    # Define the features we want to investigate
    features = [
        'temperature', 'cycle', 'tau_ox_prev', 
        'tau_red_prev', 'peak_alpha_prev', 'ox_duration'
    ]
    
    # Calculate grid size (rows x cols)
    num_features = len(features)
    cols = 3
    rows = math.ceil(num_features / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()
    
    # Plot each feature against tau_ox
    for i, feature in enumerate(features):
        ax = axes[i]
        
        # Scatter plot colored by porosity
        for por in df['porosity'].unique():
            subset = df[df['porosity'] == por]
            ax.scatter(subset[feature], subset['peak_alpha'], 
                       alpha=0.6, edgecolors='w', label=por)
        
        ax.set_title(f'peak_alpha vs {feature}', fontsize=14, fontweight='bold')
        ax.set_ylabel(r'$peak_{alpha}$')
        ax.set_xlabel(feature)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Only add legend to the first plot to keep it clean
        if i == 0:
            ax.legend()

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle(r'Exploratory Analysis: $\tau_{ox}$ Sensitivity to Input Features', 
                 fontsize=20, y=1.02)
    plt.tight_layout()
    plt.show()

# Execution
plot_relationships(df_cycles)

#%% Non monotomic temp program prediction

def simulate_non_monotonic_program(model, df_ref, porosity='Porous', temp_sequence=[500, 700, 900, 700, 500]):
    results = []
    avg_ox_dur = df_ref['ox_duration'].median()
    avg_red_dur = df_ref['red_duration'].median()
    
    # Safer initialization: use the global median if specific temp median isn't found
    start_temp = temp_sequence[0]
    ref_subset = df_ref[df_ref['temperature'] == start_temp]
    
    if ref_subset.empty:
        prev_tau_ox = df_ref['tau_ox'].median()
        prev_tau_red = df_ref['tau_red'].median()
        prev_peak = df_ref['peak_alpha'].median()
    else:
        prev_tau_ox = ref_subset['tau_ox'].median()
        prev_tau_red = ref_subset['tau_red'].median()
        prev_peak = 0

    for i, temp in enumerate(temp_sequence):
        cycle_num = i + 1
        is_p = 1 if porosity == 'Porous' else 0
        is_s = 1 if porosity == 'Spherical' else 0
        
        input_data = pd.DataFrame([{
            'porosity': porosity,
            'temperature': temp,
            'porous_cycle': is_p * cycle_num,
            'spherical_cycle': is_s * cycle_num,
            'porous_tau_ox_prev': is_p * prev_tau_ox,
            'porous_tau_red_prev': is_p * prev_tau_red,
            'spherical_tau_ox_prev': is_s * prev_tau_ox,
            'spherical_tau_red_prev': is_s * prev_tau_red,
            'porous_peak_alpha_prev': is_p * prev_peak,
            'spherical_peak_alpha_prev': is_s * prev_peak
        }])
        
        preds = model.predict(input_data)[0]
        tau_ox, tau_red, peak = preds[0], preds[1], preds[2]
        
        # Physical constraints to prevent math errors in np.exp()
        tau_ox = min(tau_ox, -1e-6)
        tau_red = max(tau_red, 1e-6)
        
        results.append({
            'cycle': cycle_num, 'temp': temp,
            'tau_ox': tau_ox, 'tau_red': tau_red, 'peak_alpha': peak
        })
        
        prev_tau_ox, prev_tau_red, prev_peak = tau_ox, tau_red, peak
        
    return pd.DataFrame(results), avg_ox_dur, avg_red_dur

def plot_non_monotonic_reconstruction(sim_df, avg_ox_dur, avg_red_dur):
    full_t, full_a = [], []
    curr_t, last_a = 0, 0
    cycle_duration = avg_ox_dur + avg_red_dur
    
    for _, row in sim_df.iterrows():
        t_ox = np.linspace(0, avg_ox_dur, 100)

        max_exp = 1 - np.exp(avg_ox_dur / row['tau_ox'])
        if abs(max_exp) < 1e-9: max_exp = 1e-9 
        
        # Normalized exponential growth (april 6)
        a_ox = last_a + (row['peak_alpha'] - last_a) * ((1 - np.exp(t_ox / row['tau_ox'])) / max_exp)
        
        full_t.extend(t_ox + curr_t)
        full_a.extend(a_ox)
        curr_t += avg_ox_dur
        t_red = np.linspace(0, avg_red_dur, 100)
        a_red = row['peak_alpha'] * np.exp(-t_red / row['tau_red'])
        
        full_t.extend(t_red + curr_t)
        full_a.extend(a_red)
        curr_t += avg_red_dur
        last_a = a_red[-1]

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Plot conversion 
    lns1 = ax1.plot(full_t, full_a, color='crimson', lw=2, label=r'Predicted $\alpha$')
    ax1.set_xlabel('Total Time (s)')
    ax1.set_ylabel(r'Conversion ($\alpha$)', color='crimson')
    ax1.tick_params(axis='y', labelcolor='crimson')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    t_temp = np.arange(len(sim_df)) * cycle_duration + (cycle_duration / 2)
    lns2 = ax2.step(t_temp, sim_df['temp'], where='mid', 
                    color='grey', linestyle='--', alpha=0.3, lw=2, label='Temp Program')
    
    ax2.set_ylabel('Temperature (degC)', color='grey')
    ax2.tick_params(axis='y', labelcolor='grey')
    lns = lns1 + [lns2[0]]
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left')

    plt.title('Non-Monotonic Temperature Simulation')
    plt.tight_layout()
    plt.show()
temps = [500, 700, 700, 500, 500]
sim_data, ox_d, red_d = simulate_non_monotonic_program(pipeline_rf, df_cycles,  # CAN SELECT HERE THE MODEL W THE PIPELINE
                                                        porosity='Spherical', 
                                                        temp_sequence=temps)
plot_non_monotonic_reconstruction(sim_data, ox_d, red_d)

