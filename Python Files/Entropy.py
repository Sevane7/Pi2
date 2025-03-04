import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import gamma


def test_stats_article2(data : pd.DataFrame, L : int):
    if L >= data.shape[0]:
        # print('L is greater than the size of the time series')
        return [3.0, pd.DataFrame()]

    symbolize_price_returns = data['Log Price'].pct_change().dropna().apply(lambda x: 1 if x > 0 else 0)

    subsequences_count = {}

    num_subsequences = symbolize_price_returns.size - L  # The correct range for subsequences
    
    # Count of subsequence
    for i in range(num_subsequences):  # Ensures we do not go out of bounds
        subsequence = tuple(symbolize_price_returns.iloc[i:i + L])  # Dictionary key
        subsequence_suffix = symbolize_price_returns.iloc[i + L]  # This will be safe now

        if subsequence not in subsequences_count:
            subsequences_count[subsequence] = {'Subsequence': subsequence, 'Count': 0, 'Count_Suffixe_1': 0, 'Count_Suffixe_0': 0}
        
        subsequences_count[subsequence]['Count'] += 1.0
        if subsequence_suffix == 1:
            subsequences_count[subsequence]['Count_Suffixe_1'] += 1.0
        else:
            subsequences_count[subsequence]['Count_Suffixe_0'] += 1.0
    
    df = pd.DataFrame.from_dict(subsequences_count, orient='index')
    df.reset_index(drop=True, inplace=True)

    # Initialize probabilities
    df['Probability'] = df['Count'] / df['Count'].sum()
    df['Probability_1'] = df['Count_Suffixe_1'] / df['Count']
    df['Probability_0'] = df['Count_Suffixe_0'] / df['Count']

    # Compute entropy
    df['Local_Entropy'] = 0.0

    mask1 = (df['Probability_1'] > 0) & (df['Probability'] > 0)
    df.loc[mask1, 'Local_Entropy'] += (
        df.loc[mask1, 'Probability'] 
        * df.loc[mask1, 'Probability_1'] 
        * np.log2(df.loc[mask1, 'Probability'] * df.loc[mask1, 'Probability_1'])
    )
    
    mask2 = (df['Probability_0'] > 0) & (df['Probability'] > 0)
    df.loc[mask2, 'Local_Entropy'] += (
        df.loc[mask2, 'Probability'] 
        * (1 - df.loc[mask2, 'Probability_1']) 
        * np.log2(df.loc[mask2, 'Probability'] * (1 - df.loc[mask2, 'Probability_1']))
    )


    True_Entropy = df['Local_Entropy'].sum() * -1

    Consistent_Entropy = (df['Probability'] * np.log2(df['Probability'] / 2.0)).sum() * -1

    Estimate_Efficiency_Indicator = Consistent_Entropy - True_Entropy

    return [Estimate_Efficiency_Indicator, df]

def reshape_data(data : pd.DataFrame, window_size : int):

    res = data.copy()
    # print("Dataset size :", data.shape[0])

    # We delete the first row of data for the size of our dataset to be a multiple of the window size 
    while(res.shape[0] % window_size != 0):
        res.drop(res.head(1).index,inplace=True)

    # print("Dataset size after reshaping :", res.shape[0])
    # print(f"Number of windows of size {window_size} : ", res.shape[0] / window_size)

    return res

def compute_efficiency_indicator(data : pd.DataFrame, window_size, L):
    total_size = data.shape[0]

    # Stocker l'index original pour vérifier la longueur finale
    original_index = data.index  

    efficiency_indicators = [np.nan] * window_size  # Initialisation correcte

    for i in range(window_size, total_size):
        indicator, res = test_stats_article2(data.iloc[i - window_size: i], L)
        efficiency_indicators.append(indicator)

    # Ajouter la colonne directement au DataFrame original
    data['Efficiency Indicator'] = efficiency_indicators  # Utiliser loc pour éviter les copies



def display_Indicator(original : pd.DataFrame, forecast : pd.DataFrame, window_size : int):
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(original['Efficiency Indicator'][window_size:], label="Efficiency Indicator for Original Data")
    plt.plot(forecast['Efficiency Indicator'][window_size:], label="Efficiency Indicator for Forecasted Data")

    plt.title("Efficiency Indicator during time")
    plt.xlabel("Time")
    plt.ylabel("Efficiency Indicator")
    plt.legend()
    plt.grid(True)

    plt.show()


def compute_entropy_indicator(df : pd.DataFrame, window_size, L):

    reshape_df = reshape_data(df, window_size)

    compute_efficiency_indicator(reshape_df, window_size, L)

    # print(set(df.index).symmetric_difference(set(reshape_df.index)))

    df['Efficiency Indicator'] = reshape_df['Efficiency Indicator'].reindex(df.index)

def symbolize_market_info(data : pd.DataFrame, L : int, window_size : int, alpha : list):

    if 'Efficiency Indicator' not in data.columns:
        compute_entropy_indicator(data, window_size, L)


    n = window_size
    k = 2**(L - 1)
    theta = 1 / (n*np.log(2))

    threshold = gamma.ppf(1 - alpha, a=k, scale=theta)

    data[f"Symbolize Market info ({(1-alpha)*100} %)"] = data['Efficiency Indicator'].apply(lambda x: int(x <= threshold))

def process_entropy(L : float, window : float, alphas : list):
    
    folder_path = r"Data\\Data Hurst - Final - Copie"

    excel_files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]

    for file in excel_files[13:18]:  # Ghali : [-5:]
        print(file)
        file_path = os.path.join(folder_path, file)
        
        # Charger toutes les feuilles
        xls = pd.ExcelFile(file_path)
        sheets : dict[str, pd.DataFrame]= {}
        
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)
            print(sheet_name)

            # if "Efficiency Indicator" in df.columns:
            #     df.drop(labels=["Efficiency Indicator"], axis=1, inplace=True)            

            for a in alphas:
                print(a)

                symbolize_market_info(df, L, window, a)

                sheets[sheet_name] = df.dropna()
        
        # Sauvegarder en écrasant
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)


if __name__ == "__main__":

    L = 3
    window = 252
    alphas = [i/100 for i in range(1, 6)]
    process_entropy(L, window, alphas)


    pass