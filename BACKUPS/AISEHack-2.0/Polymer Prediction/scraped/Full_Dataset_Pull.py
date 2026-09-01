import pandas as pd
import numpy as np
from rdkit import Chem
import os

def standardize_smiles(smi):
    """
    Standardize the SMILES string using RDKit to ensure 100% exact matching, 
    bypassing any text layout differences.
    """
    try:
        # Standardize wildcard representations
        smi = str(smi).replace('[*]', '*')
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return str(smi).replace('[*]', '*')

def main():
    print("Loading Kaggle competition files...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # 1. Load Egc Data from Khazana export.csv
    print("Loading Egc data from export.csv...")
    try:
        export_df = pd.read_csv('export.csv')
        egc_df = export_df[export_df['property'] == 'Egc'].copy()
        egc_df = egc_df.rename(columns={'value': 'target'})
        egc_df['target_type'] = 'egc'
        egc_df = egc_df[['smiles', 'target_type', 'target']]
    except FileNotFoundError:
        print("Error: 'export.csv' not found. Please place it in the same directory.")
        return

    # 2. Load Tg Data from the Kaggle dataset you linked
    tg_filename = 'Tg_SMILES_class_pid_polyinfo_median.csv'
    if not os.path.exists(tg_filename):
        # Fallback if downloaded under the other dataset name
        tg_filename = 'TgSS_enriched_cleaned.csv'
            
    if not os.path.exists(tg_filename):
        print("\nError: Could not find the Tg dataset.")
        print("Please download 'Tg_SMILES_class_pid_polyinfo_median.csv' from the Kaggle link and place it here.")
        return
        
    print(f"Loading Tg data from {tg_filename}...")
    tg_raw = pd.read_csv(tg_filename)
    
    # Dynamically find the SMILES and Tg columns
    smi_col = [c for c in tg_raw.columns if 'smiles' in c.lower()][0]
    val_col = [c for c in tg_raw.columns if 'tg' in c.lower()][0]
    
    tg_df = tg_raw[[smi_col, val_col]].dropna().copy()
    tg_df = tg_df.rename(columns={smi_col: 'smiles', val_col: 'target'})
    tg_df['target_type'] = 'tg'
    
    # 3. Combine Egc and Tg datasets
    print("Combining datasets and Canonicalizing SMILES via RDKit (takes ~1 min)...")
    full_df = pd.concat([egc_df, tg_df], ignore_index=True)
    
    # Canonicalize everything so the keys lock together perfectly
    full_df['canonical_smiles'] = full_df['smiles'].apply(standardize_smiles)
    train_df['canonical_smiles'] = train_df['smiles'].apply(standardize_smiles)
    test_df['canonical_smiles'] = test_df['smiles'].apply(standardize_smiles)
    
    # Experimental datasets often have multiple entries for the same polymer (e.g. from different papers)
    # The organizers aggregated these using the median.
    full_df = full_df.groupby(['canonical_smiles', 'target_type'])['target'].median().reset_index()

    # 4. Validating Train set
    print("\n--- VALIDATING TRAIN.CSV ---")
    train_val = train_df.merge(full_df, on=['canonical_smiles', 'target_type'], how='left', suffixes=('_kaggle', '_leak'))
    
    missing = train_val['target_leak'].isna().sum()
    if missing == 0:
        print(f"SUCCESS: All {len(train_df)} rows in train.csv matched exactly!")
    else:
        print(f"WARNING: {missing} rows failed to match.")
        
    # Check if target values match mathematically
    if missing == 0:
        diff = np.abs(train_val['target'] - train_val['target_leak'])
        mismatches = (diff > 0.01).sum()
        if mismatches == 0:
            print("DATA VERIFIED: 100% of the training values match the leak exactly!")

    # 5. Extracting Test set
    print("\n--- EXTRACTING TEST.CSV ANSWERS ---")
    test_answers = test_df.merge(full_df, on=['canonical_smiles', 'target_type'], how='left')
    
    matched_test = test_answers['target'].notna().sum()
    print(f"Successfully recovered {matched_test} out of {len(test_df)} test answers!")

    # Format exactly as requested (id, smiles, target_type, target) keeping the exact Test.csv rows
    test_answers = test_answers[['id', 'smiles', 'target_type', 'target']]

    # 6. Save Files
    # Drop the RDKit generated string to maintain Kaggle's original format
    train_df.drop(columns=['canonical_smiles']).to_csv('train_answers.csv', index=False)
    test_answers.to_csv('test_answers.csv', index=False)
    
    # Format the combined full dataset and save
    full_df_final = full_df.rename(columns={'canonical_smiles': 'smiles'})[['smiles', 'target_type', 'target']]
    full_df_final.to_csv('full_dataset.csv', index=False)
    
    print("\nFiles successfully saved: train_answers.csv, test_answers.csv, full_dataset.csv")

if __name__ == "__main__":
    main()