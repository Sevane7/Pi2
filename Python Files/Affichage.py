import pandas as pd
import matplotlib.pyplot as plt
import os
import re

metrics_file_path = "Data\\Forecasting\\Metrics\\Daily\\SPY.xlsx"

def find_files(metrics_file_path = metrics_file_path) -> list:

    metrics = pd.read_excel(metrics_file_path)

    metrics["Horizon"] = metrics["Forcast"].apply(lambda x : int(x.split("Horizon ")[1]))

    metrics["Frequence"] = metrics["Forcast"].apply(lambda x : re.search(r"freq (.*?) ", x).group(1))

    metrics["Forcast"] = metrics["Forcast"].apply(lambda x : x + ".xlsx")

    sorted_metrics = metrics.sort_values(by="MSE")

    return sorted_metrics[["Hit Ratio", "MSE", "Frequence", "Horizon"]]

    # return sorted_metrics["Forcast"].to_list()[:6]


def find_forecast(file_name : str) -> dict[str:pd.DataFrame]:

    res = {
        "Original" : pd.DataFrame(None),
        "Forecast" : pd.DataFrame(None)
    }
    
    for root, dirs, files in os.walk("Data\Forecasting"):

        for file in files:

            if (file.lower() == file_name.lower()):

                file_path = os.path.join(root, file_name)

                print(file_name)

                for i in res.keys():
                    res[i] = pd.read_excel(file_path, sheet_name=i, index_col=0)

                break

    return res


def remove_duplicates(df1 : pd.Series, df2 : pd.Series) -> None:

    commun_index = df1[df1.isin(df2)].index

    df1.drop(index = commun_index, inplace=True)

    df2.drop(index = commun_index, inplace=True)




def plot_original_VS_forecated(data : dict, filename : str):

    df_origin : pd.DataFrame = data["Original"]["Price"]

    df_forecast : pd.DataFrame = data["Forecast"]["Price"].round(2)

    remove_duplicates(df_forecast, df_origin)

    print(df_origin)
    print(df_forecast)

    plt.scatter(df_origin.index, df_origin, c='red', label='original data', marker='x', s=10, alpha=0.7)
    plt.scatter(df_forecast.index, df_forecast, c='blue', label='mean forecasted_data', marker='s', s=10, alpha=0.7)
    
    plt.title(f"{os.path.splitext(filename)[0]}")
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.ylabel("Price")
    plt.legend()

    plt.grid(True, alpha = 0.7)
    plt.show()


files = find_files()

print(files.tail(10))

# for f in files:
    
#     file_name = f + ".xlsx"

#     d = find_forecast(f)

#     print(file_name)

#     plot_original_VS_forecated(d, file_name)
