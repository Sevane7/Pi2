import pandas as pd
import numpy as np

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

        self.original_data = self.load_data()
        self.bounded_data = self.bound_data()


    def load_data(self) -> pd.DataFrame:

        df =  pd.read_excel(f"Data\{self.timeframe}\{self.ticker}.xlsx", index_col="Date")

        df.index = pd.DatetimeIndex(df.reset_index()["Date"])

        df = df.rename(columns={self.ticker : "Price"})

        df[f"Log Return"] = np.log(df["Price"]/df["Price"].shift())

        #df[f"{ticker} Cumprod"] = (1 + df[f"{ticker} Log Return"]).cumprod() - 1

        return df.dropna()
    
    def bound_data(self) -> pd.DataFrame:
        
        if self.start is None or self.start < self.original_data.index.min():
            self.start = self.original_data.index.min()

        else:
            while self.start not in self.original_data.index:
                self.start += pd.Timedelta(days=1)

        if self.end is None or self.end > self.original_data.index.max():
            self.end = self.original_data.index.max()
        
        else:
            while self.end not in self.original_data.index and self.end < self.original_data.index.max():
                self.end += pd.Timedelta(days=1)
        
        return self.original_data.loc[self.start:self.end]
    
    def compare_forecast_data(self, horizon : int) -> np.ndarray:

        df = self.original_data.loc[self.end : ]["Price"].to_numpy()

        try:
            return df[:horizon]
        
        except IndexError:

            print("Impossible de comparer, données manquantes")
            
            return df 


    def __str__(self):

        start_format = pd.to_datetime(str(self.start)).strftime("%A %d %B")
        end_format = pd.to_datetime(str(self.end)).strftime("%A %d %B")
    
        return f"{self.ticker} From {start_format} To {end_format}."

    def __len__(self):
        return len(self.original_data)
        