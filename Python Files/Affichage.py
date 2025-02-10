import pandas as pd
import os

metrics_file_path = "Data\\Forecasting\\Metrics\\Daily\\SPY.xlsx"

def sort_forecast(metrics_file_path = metrics_file_path):

    metrics = pd.read_excel(metrics_file_path)

    sorted_metrics = metrics.sort_values(by="MSE")

    print(sorted_metrics[["Forcast", "MSE", "Hit Ratio"]])


def find_forecast(file_name : str) -> dict[str:pd.DataFrame]:

    res = {
        "Orignal" : pd.DataFrame(None),
        "Forecast" : pd.DataFrame(None)
    }
    
    for root, dirs, files in os.walk("Data\Forecasting"):

        for file in files:

            if (file.lower() == file_name.lower()):

                file_path = os.path.join(root, file_name)

                for i in res.keys():
                    res[i] = pd.read_excel(file_path, sheet_name=i)

                break

    return res


def plot_original_VS_forecated(data : dict):

    df_origin = data["Original"]

    df_forecast = data["Forecast"]

sort_forecast()