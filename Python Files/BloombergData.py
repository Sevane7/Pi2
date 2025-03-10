import pandas as pd
import numpy as np
import os
from Entropy import compute_entropy_indicator

class BloombergData:

    def __init__(self,
                 ticker : str,
                 timeframe : str,
                 start_date = None,
                 end_date = None):
        
        self.start = pd.to_datetime(start_date)
        self.end = pd.to_datetime(end_date)
        
        self.ticker = ticker
        self.timeframe = timeframe
        
        self.path_file = os.path.join(os.getcwd(), rf"Data\\Bloomberg Original Data\\{self.timeframe}\\{self.ticker}.xlsx")


    def load_data(self) -> pd.DataFrame:

        df =  pd.read_excel(self.path_file, index_col="Date")

        df.index = pd.DatetimeIndex(df.reset_index()["Date"])

        df["Log Price"] = np.log(df[self.ticker])

        df = df.drop(columns=self.ticker)

        # df["Log Return"] = np.log(df["Log Price"]/df["Log Price"].shift())

        # compute_entropy_indicator(df, 5, 2)

        #df[f"{ticker} Cumprod"] = (1 + df[f"{ticker} Log Return"]).cumprod() - 1

        return df.dropna()
    
    def bound_data(self, df : pd.DataFrame) -> pd.DataFrame:
        
        if self.start is None or self.start < df.index.min():
            self.start = df.index.min()

        else:
            while self.start not in df.index:
                self.start += pd.Timedelta(days=1)

        if self.end is None or self.end > df.index.max():
            self.end = df.index.max()
        
        else:
            while self.end not in df.index and self.end < df.index.max():
                self.end += pd.Timedelta(days=1)
        
        return df.loc[self.start:self.end]
    
    def compare_forecast_data(self, horizon : int) -> np.ndarray:

        df = self.original_data.loc[self.end : ]["Price"].to_numpy()

        try:
            return df[:horizon]
        
        except IndexError:

            print("Impossible de comparer, données manquantes")
            
            return df 


    def __str__(self):

        start_format, end_format = self.start, self.end
        
        if(self.timeframe == "Daily"):

            start_format = pd.to_datetime(str(self.start)).strftime("%d-%m-%Y")
            end_format = pd.to_datetime(str(self.end)).strftime("%d-%m-%Y")
    
        return f"{self.ticker} - {self.timeframe} - {start_format} To {end_format}"

    def __len__(self):
        return len(self.original_data)
        