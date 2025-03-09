import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_vol(returns : pd.Series) :
    return returns.std() * np.sqrt(252)

def calculate_sharpe(returns : pd.Series, risk_free_rate=0):

    rolling_mean = returns.sum()
    rolling_std = calculate_vol(returns)
    sharpe_ratio = (rolling_mean - risk_free_rate) / rolling_std
    return sharpe_ratio

def calculate_Cummul_Return(returns : pd.Series):
    res = (1 + returns).cumprod()
    return pd.Series(res)

def calculate_maxDrawdown(returns : pd.Series):

    cummprod = calculate_Cummul_Return(returns)
    peak = cummprod.cummax()
    drawdown = (cummprod - peak) / peak
    return drawdown.min()


        

def process_metrics(folder_path : str):

    excel_files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]

    for file in excel_files:
        print(file)
        file_path = os.path.join(folder_path, file)
        
        # Charger toutes les feuilles
        xls = pd.ExcelFile(file_path)
        sheets : dict[str, pd.DataFrame]= {}
        
        for sheet_name in xls.sheet_names:
            print(sheet_name)
                  
            df = xls.parse(sheet_name)

            # Ici ajouter les methodes sortino, VaR
            df["Price"] = np.exp(df["Log Price"])
            df["Return"] = df["Price"].diff() / df["Price"]
            df["Log Return"] = df["Log Price"].diff()
            df["Volatility"] = calculate_vol(df["Log Return"], 252)
            df["MaxDrawdown"] = calculate_maxDrawdown(df["Price"], 252)
            df["Sharpe"] = calculate_sharpe(df,252)

            sheets[sheet_name] = df.dropna()
        
        # Sauvegarder en écrasant
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)


def metrics_strategy(df : pd.DataFrame):

    df["Price"] = np.exp(df["Log Price"])
    df["Return"] = df["Price"].diff() / df["Price"]
    df["Log Return"] = df["Log Price"].diff()
    df["Max Drawdown"] = df["Return"].rolling(252).apply(calculate_maxDrawdown, raw=True)
    df["Sharpe Ratio"] = df["Return"].rolling(252).apply(calculate_sharpe, raw=True)
    df["Sign Forecast"] = np.sign(df["Forecast"] - df["Log Price"])
    df["Return Strategy"] = df["Sign Forecast"] * df["Return"]
    df["Max Drawdown Strategy"] = df["Return Strategy"].rolling(252).apply(calculate_maxDrawdown, raw=True)
    df["Sharpe Ratio Strategy"] = df["Return Strategy"].rolling(252).apply(calculate_sharpe, raw=True)


def plot_compare_metrics(df : pd.DataFrame, asset : str, name_metrics = "Log Price"):

    title = f"{asset} {name_metrics} VS "
    comparative = "Forecast" if name_metrics == "Log Price" else f"{name_metrics} Strategy"
    title += comparative

    plt.figure(figsize=(12,8))

    if name_metrics == "Cumprod":
        cumprod = calculate_Cummul_Return(df["Return"])
        cumprod_stragey = calculate_Cummul_Return(df["Return Strategy"])
        plt.plot(cumprod.index, cumprod, label=name_metrics, color="blue")
        plt.plot(cumprod_stragey.index, cumprod_stragey, label=name_metrics + "Strategy", color="red")
    
    else:
        plt.plot(df.index, df[name_metrics], label=name_metrics, color="blue")
        plt.plot(df.index, df[comparative], label=comparative, color="red")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(name_metrics)

    plt.legend() 
    plt.grid(True)
    plt.show()


def process_metrics_strategy(dir_path_metrics : str):

    files = [file for file in os.listdir(dir_path_metrics) if file.endswith(".xlsx")]

    for file in files:

        print(file)

        file_path = os.path.join(dir_path_metrics, file)

        df = pd.read_excel(file_path, index_col=0)

        metrics_strategy(df)

        df.to_excel(file_path)





if __name__ == "__main__":

    dir_accurate_files = r"Data\\Accurate Files - Copie"

    for file in os.listdir(dir_accurate_files)[5:6]:
        file_path = os.path.join(dir_accurate_files, file)
        df = pd.read_excel(file_path, index_col=0)
        # plot_compare_metrics(df, file, "Log Price")
        print(df["Sign Forecast"].value_counts())


    # process_metrics_strategy(dir_accurate_files)

