from typing import List, Dict, Optional, Union, Tuple, Any
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import torch.nn as nn
import torch
import json
import math
import pandas as pd
import os
import argparse
import yfinance as yf
from datetime import datetime, timedelta
from copy import deepcopy as dc
from plots import *
from arch import arch_model

#---------------------------------------- Preprocessing code ---------------------------------------#
class Preprocessing:
    def __init__(self,
                 args: argparse.Namespace) -> None:
        #Paths.
        self.open_path: str = args.open_path
        self.training_path: str = args.training_path
        self.prediction_path: str = args.prediction_path
        self.save: bool = args.save
        self.jason_path = args.jason_path

        #Special preprocessing.
        self.del_nulls: bool = args.del_nulls
        self.encoding_: bool = args.encoding_
        
        #Scale dataset.
        self.normalized: bool = args.normalize 
        self.normalizer: MinMaxScaler = MinMaxScaler()
        self.standardized: bool = args.standardize
        self.scaler: StandardScaler = StandardScaler()
        
        #Windows and technical and others.
        self.windows: int = args.windows
        self.technical: bool = args.technical
        self.Garch: bool = args.GARCH
        
        #Redimension.
        self.PCA_: bool = args.PCA_
        self.LDA_: bool = args.LDA_
        self.Linear_projection_: bool = args.Linear_projection_

#---------------------------------------- Download and Open Dataset ---------------------------------------#             
    def download_data(self,
                      ticker: str,
                      start: str,
                      end: str,
                      save_dir: str):
        
        df = yf.download(tickers = ticker, start = start, end = end)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        if self.technical:
            dataset_name = f"{ticker} {start} to {end}_tech.csv".replace(":", "-")
        else:
            dataset_name = f"{ticker} {start} to {end}.csv".replace(":", "-")
        
        save_path = os.path.join(save_dir, dataset_name)

        df.to_csv(save_path)
        print(f"\nData saved successfully to:\n{save_path}\n")
        
    def open_csv(self,
                 ticker: str,
                 start: str,
                 end: str,
                 verbose: bool = False) -> pd.DataFrame: 
        try:
            if not os.path.exists(self.open_path):
                print(f"\nError: The directory '{self.open_path}' does not exist.\n")
                return pd.DataFrame()

            # Select this file if `self.technical`
            filename = f'{ticker} {start} to {end}_tech.csv' if self.technical else f'{ticker} {start} to {end}.csv'
            df = pd.read_csv(os.path.join(self.open_path, filename))
            
            if verbose:
                print(f'\n{df.head(10)}\n') 
            
            return df

        except FileNotFoundError:
            print(f"\nError: The file '{filename}' was not found in '{self.open_path}'.\n")
            return pd.DataFrame()
        except pd.errors.EmptyDataError:
            print(f"\nError: The file '{filename}' is empty.\n")
            return pd.DataFrame()
        except pd.errors.ParserError:
            print(f"\nError: The file '{filename}' could not be parsed.\n")
            return pd.DataFrame()

#---------------------------------------- Scale Dataset ---------------------------------------#    
    def normalize(self,
                  df: pd.DataFrame,
                  start: str,#To put dates on the name of the dataset.
                  end: str,
                  ticker: str,
                  start_col: Optional[str] = None,
                  end_col: Optional[Union[str, List[str]]] = None,
                  verbose: bool = False) -> pd.DataFrame:
        
        excluded_cols = None
        # Case 1: Normalize from start_col to end_col
        if start_col is not None and isinstance(end_col, str):
            assert start_col in df.columns and end_col in df.columns, f'\n{start_col} and/or {end_col} not found in the DataFrame\n'
            cols_to_normalize = df.loc[:, start_col:end_col].columns

        # Case 2: Exclude columns from normalization
        elif end_col is not None:
            if isinstance(end_col, str):
                end_col = [end_col]
            missing_cols = [col for col in end_col if col not in df.columns]# Check that all columns in end_col are in the DataFrame
            assert not missing_cols, f'\nThe following columns are not in the DataFrame: {missing_cols}\n'
            excluded_cols = df[end_col].copy()
            cols_to_normalize = df.columns.drop(end_col)

        # Case 3: Normalize the entire DataFrame
        else:
            cols_to_normalize = df.columns

        # Normalize the selected columns
        df_normalized = self.normalizer.fit_transform(df[cols_to_normalize])
        df[cols_to_normalize] = pd.DataFrame(df_normalized, columns=cols_to_normalize, index=df.index)
        
        # Extract the min and max values for each column
        if self.Garch:
            normalization_values = {
                'Adj Close': {
                    "min": float(self.normalizer.data_min_[-1]),  # Última columna normalizada (Adj Close)
                    "max": float(self.normalizer.data_max_[-1])   # Última columna normalizada (Adj Close)
                }
            }
            # Save the min and max values to a JSON file
            with open(os.path.join(self.jason_path, f'{ticker} {start} to {end} GARCH.json'), 'w') as file:
                json.dump(normalization_values, file, indent=4)

        elif self.windows:
            normalization_values = {
                'Adj Close': {
                    "min": float(self.normalizer.data_min_[-1]),  # Última columna normalizada (Adj Close)
                    "max": float(self.normalizer.data_max_[-1])   # Última columna normalizada (Adj Close)
                }
            }
        
            # Save the min and max values to a JSON file
            with open(os.path.join(self.jason_path, f'{ticker} {start} to {end} processed.json'), 'w') as file:
                json.dump(normalization_values, file, indent=4)

        elif self.technical:
            normalization_values = {
            col: {
                "min": float(self.normalizer.data_min_[i]),
                "max": float(self.normalizer.data_max_[i])  
            }for i, col in enumerate(cols_to_normalize)#The for loop at the final.
            }
            # Save the min and max values to a JSON file
            with open(os.path.join(self.jason_path, f'{ticker} {start} to {end} processed tech.json'), 'w') as file:
                json.dump(normalization_values, file, indent=4)

        else:
            normalization_values = {
                'Adj Close': {
                    "min": float(self.normalizer.data_min_[-1]),  # Última columna normalizada (Adj Close)
                    "max": float(self.normalizer.data_max_[-1])   # Última columna normalizada (Adj Close)
                }
            }
        
            # Save the min and max values to a JSON file
            with open(os.path.join(self.jason_path, f'{ticker} {start} to {end} normal.json'), 'w') as file:
                json.dump(normalization_values, file, indent=4)
            
        # Restore the excluded columns after normalization
        if excluded_cols is not None:
            df[end_col] = excluded_cols

        if verbose:
            print(f'\nThis is the Normalized dataset (first 10 rows):\n{df.head(10)}\n')
            print(f'\nThis is the Normalized dataset (last 10 rows):\n{df.tail(10)}\n')
        return df
    
    def standardize(self,
                    df: pd.DataFrame,
                    start_col: Optional[str] = None,
                    end_col: Optional[Union[str, List[str]]] = None,
                    verbose: bool = False) -> pd.DataFrame:
        
        excluded_cols = None 
        
        # Case 1: Standardize from start_col to end_col
        if start_col is not None and isinstance(end_col, str):
            assert start_col in df.columns and end_col in df.columns, f'\n{start_col} and/or {end_col} not found in the DataFrame\n'
            cols_to_standardize = df.iloc[:, start_col:end_col].columns

        # Case 2: Exclude columns from Standardization
        elif end_col is not None and isinstance(end_col, str):
            end_col = [end_col]
            missing_cols = [col for col in end_col if col not in df.columns]
            assert not missing_cols, f'\nThe following columns are not in the DataFrame: {missing_cols}\n' 
            excluded_cols = df[end_col].copy()
            cols_to_standardize = df.columns.drop(end_col)

        else:# Case 3: Standardize the entire DataFrame
            cols_to_standardize = df.columns

        df_standardized=self.scaler.fit_transform(df[cols_to_standardize])
        df[cols_to_standardize] = pd.DataFrame(df_standardized , columns=df.columns , index=df.index)
        
        # Restore the excluded columns after Standardization
        if excluded_cols is not None:
            df[end_col] = excluded_cols

        if verbose:
            print(f'\nThis is the Standardized dataset (first 10 rows):\n{df.head(10)}\n')
            print(f'\nThis is the Standardized dataset (last 10 rows):\n{df.tail(10)}\n')
        
        return df
    
    #Normalize the dataset using a different range in the selected cols depending of the max and min vals of the col.
    def diff_range_norm(self,
                        df: pd.DataFrame,
                        cols_to_norm: str):
        df = df.copy()
        if isinstance(cols_to_norm, str):
            cols_to_norm = [cols_to_norm]

        for col in cols_to_norm:
            df[col] = 2 * ((df[col] - df[col].min()) / (df[col].max() - df[col].min())) - 1
        
        print(f'\nThis is the dataset using ("diff_range_norm"), (first 10 rows):\n{df.head(10)}\n')
        print(f'\nThis is the dataset using ("diff_range_norm"), (last 10 rows):\n{df.tail(10)}\n')

        return df
#---------------------------------------- Process Cols and Rows ---------------------------------------#
    def drop_columns(self,
                     df: pd.DataFrame,
                     columns_name: str,
                     verbose: bool = True):
        
        if isinstance(columns_name, str):
            columns_name = [columns_name]
        
        for col in columns_name:
            df.drop(col,axis=1, inplace=True)
        
        if verbose:
            print(f"""\ndrop_columns func has been applied to the dataset.
                  Column : {columns_name} was removed from the dataset\n""")
        
        return df
    
    def move_col_to_end(self,
                        df: pd.DataFrame,
                        col_name: str = 'Adj Close'):
        
        cols = [col for col in df.columns if col != col_name] + [col_name]  
        return df[cols]
    
    def set_col_index(self,
                      df: pd.DataFrame,
                      column_name: str = 'Date'):
        
        df.set_index([column_name] , inplace=True)

        print('*' * 50)
        print(f"Column : {column_name} was set as the index of the dataset")
        print('*' * 50)
        return df
    
    def how_many(self,
                 df: pd.DataFrame,
                 row_from: int,
                 row_to: int,
                 col_from: int,
                 col_to: int,
                 verbose: bool = False):
        
        if row_to is None:
            row_to = len(df)
        years = (len(df) - row_from) / 261 # 261 is an appr. of one year time without weekdays  

        df = df.iloc[row_from:row_to , col_from:col_to]
        if verbose:
            print(f'\nDataset after "How many Func", {years:.2f} in total, first 10 rows\n')
            print(df.head(10))
            print(f'\nDataset after "How many Func", {years:.2f} in total, last 10 rows\n')
            print(df.tail(10))
        return df
    
    def delete_blank_rows(self,
                          df: pd.DataFrame):
        df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        df = df.dropna()
        print(f"\ndelete_blank_rows func has been applied to the dataset succesfully.\n") 
        return df
    
    def encoding(self,
                 df: pd.DataFrame,
                 column_name: str,
                 start: str,#To put dates on the name of the dataset.
                 end: str,
                 stratify: bool = True,
                 min_elements: int = 10):
        df[column_name] = df[column_name].str.strip()
        if stratify:
            class_counts = df[column_name].value_counts()
            valid_classes = class_counts[class_counts >= min_elements].index
            df = df[df[column_name].isin(valid_classes)]
            
            removed_classes = class_counts[class_counts < min_elements].index
            if not removed_classes.empty:
                print(f"Eliminated classes with less than: {min_elements}\nelements: {list(removed_classes)}")

        encoder = LabelEncoder()
        df[column_name] = encoder.fit_transform(df[column_name])
        label_mapping = {original: encoded for original, encoded in zip(encoder.classes_, range(len(encoder.classes_)))}
        class_counts = df[column_name].value_counts().to_dict()
        label_info = {
            'mapping': label_mapping,
            'class_counts': {encoder.inverse_transform([key])[0]: value for key, value in class_counts.items()}
        }
        with open(os.path.join(self.jason_path, f'NVDA {start} to {end} processed.json'), 'w') as json_file:
            json.dump(label_info, json_file, indent=4)
        
        print(f'Mapping and class counts saved to {self.jason_path}') 
        
        return df
    
    def isolate_cols(self,
                    df: pd.DataFrame,
                    cols_to_isolate: Optional [Union[list[str], str]],
                    verbose: bool = True):
        
        if isinstance(cols_to_isolate, list):
            cols_to_isolate = [cols_to_isolate]

        df_copy = df[[cols_to_isolate]].copy()

        if verbose:
            print(f'\nDataset  first 10 rows with the isolate cols :\n{df_copy.head(10)}\n')
            print(f'\nDataset  last 10 rows with the isolate cols :\n{df_copy.tail(10)}\n')

        return df_copy 
#---------------------------------------- Time series ---------------------------------------#     
    def windows_preprocessing(self,
                              df: pd.DataFrame,
                              windows_size: int,
                              cols_to_windows: Optional[Union[list[str], str]],
                              idx: bool = True,
                              verbose: bool = True):
        
        if isinstance(cols_to_windows, str):
            cols_to_windows = [cols_to_windows]

        index_dates = []
        df_copy = df[cols_to_windows].copy()

        for col in cols_to_windows:
            for i in range(1, windows_size +1):
                df_copy[f'{col}(t-{i})'] = df_copy[col].shift(i)

        df_copy.dropna(inplace=True)#Avoid empty or Nan values in the dataset.
        df_copy = df_copy.iloc[:, ::-1]#Flip columns

        if idx:
            for i in range(len(df_copy)):
                date_of_pred = df.index[i + windows_size]#Calculate the date till the windows size ends.
                index_dates.append(f"{date_of_pred}")
        else:
            index_dates = list(range(len(df_copy)))
        
        df_copy.index = index_dates
        df_copy.index.name = "Date"

        if self.Garch:
            #Drop garch_vol col to avoid giving info about actual Adj close to the model.
            self.drop_columns(df_copy, 'garch_vol')

        if verbose:
            print(f'\nDataset  first 10 rows with windows size :{windows_size}:\n{df_copy.head(10)}\n')
            print(f'\nDataset  last 10 rows with windows size :{windows_size}:\n{df_copy.tail(10)}\n')

        return df_copy

#---------------------------------------- technical analysis ---------------------------------------# 
    def GARCH(self,
              df: pd.DataFrame,
              cols: Optional[Union[list[str], list]],
              verbose: bool = True):
        
        if isinstance(cols, list):
            cols = [cols]

        am = arch_model(df[cols], vol='Garch', p=1, q=1, dist='t')
        res = am.fit(update_freq=5, disp='off')
        
        df['garch_vol'] = res.conditional_volatility
        #Drop the log return col.
        self.drop_columns(df, 'LogR_Adj Close_t-1')

        if verbose:
            print(f'\nDataset  first 10 rows after apply GARCH :\n{df.head(10)}\n')
            print(f'\nDataset  last 10 rows after apply GARCH :\n{df.tail(10)}\n')

        return df

    def get_previous_business_day(self,
                                  start_date: str,
                                  business_days: int = 85) -> str:
        """
        Calculate the date that is a given number of business days 
        (excluding weekends) before the start date.
        """
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        days_counted = 0
        
        while days_counted < business_days:
            current_date -= timedelta(days=1)
            if current_date.weekday() < 5:
                days_counted += 1
        print(f"\n\nThe recommended date for the dataset is: {current_date.strftime('%Y-%m-%d')}\n\n")
        
        return current_date.strftime("%Y-%m-%d")
    
    #Simple Moving Average.
    def SMA(self,
            df: pd.DataFrame,
            cols: str = 'Adj Close',
            periods: int = 20,
            verbose: bool = True):
        
        if isinstance(cols, str):
            cols = [cols]

        for col in cols:
            sma_series = df[col].rolling(window=periods).mean()
            sma_col_name = f'SMA_{col}_{periods}_periods'
            df[sma_col_name] = sma_series
        
        sma_columns = [f'SMA_{col}_{periods}_periods' for col in cols]
        df.dropna(subset=sma_columns, inplace=True)

        if verbose:
            print(f'\nDataset  first 10 rows after applying "SMA" func :\n{df.head(10)}\n')
            print(f'\nDataset  last 10 rows after applying "SMA" func :\n{df.tail(10)}\n')
        
        return df
    
    def EMA(self,
            price_series: str,
            periods: int):
        
        sma = price_series[:periods].mean() #Only the first period
        ema_series = pd.Series([sma]) # is a empty series with only sma then it will be filled
        multiplier = 2 / (periods + 1)

        for price in price_series[periods:]:
            current_ema = ((price - ema_series.iloc[-1]) * multiplier) + ema_series.iloc[-1]
            ema_series = ema_series._append(pd.Series([current_ema]), ignore_index = True)

        ema_series = pd.concat([pd.Series([None] * (periods - 1)), ema_series], ignore_index= True)
        
        ema_series.index = price_series.index
        return ema_series
    
    def bollinger_bands(self,
                        df: pd.DataFrame,
                        cols:str,
                        periods: int,
                        num_std_dev: int):
        
        if isinstance(cols, str):
            cols = [cols]

        for col in cols:
            #Calcultae SMA
            sma_col_name = f'SMA_{col}_{periods}_periods'
            df[sma_col_name] = df[col].rolling(window=periods).mean()
            
            # Calculate Std
            std_col_name = f'STD_{col}_{periods}_periods'
            df[std_col_name] = df[col].rolling(window=periods).std()
            
            #calculate upper band and lower band
            upper_band_col_name = f'Upper_BB_{col}_{periods}_periods'
            lower_band_col_name = f'Lower_BB_{col}_{periods}_periods'
            df[upper_band_col_name] = df[sma_col_name] + (df[std_col_name] * num_std_dev)
            df[lower_band_col_name] = df[sma_col_name] - (df[std_col_name] * num_std_dev)
        
        df.drop([std_col_name], axis=1, inplace=True)
        
        return df
    
    def MFI(self,
            df: pd.DataFrame,
            periods: int):
        
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        mf = tp * df['Volume']

        df['Positive_MF'] = mf.where(tp > tp.shift(1), 0)
        df['Negative_MF'] = mf.where(tp < tp.shift(1), 0)

        df['Sum_Positive_MF'] = df['Positive_MF'].rolling(window=periods).sum()
        df['Sum_Negative_MF'] = df['Negative_MF'].rolling(window=periods).sum()

        df['Money_Flow_Ratio'] = df['Sum_Positive_MF'] / df['Sum_Negative_MF']

        df['MFI'] = 100 - (100 / (1 + df['Money_Flow_Ratio']))
        df.drop(['Positive_MF', 'Negative_MF', 'Sum_Positive_MF', 'Sum_Negative_MF', 'Money_Flow_Ratio'], axis=1, inplace=True)
        
        return df
    
    #Moving Average Convergence Divergence.
    def MACD(self,
             df: pd.DataFrame,
             cols:str,
             short_period: int = 12,
             long_period: int = 26,
             signal_period: int = 9):
        
        if isinstance(cols, str):
            cols = [cols]

        for col in cols:
            ema_short = self.EMA(df[col], short_period)
            ema_long = self.EMA(df[col], long_period)
            df['MACD'] = ema_short - ema_long
            df['Signal Line'] = self.EMA(df['MACD'].dropna(), signal_period)
            df['Histogram'] = df['MACD'] - df['Signal Line']
            df.dropna(subset = ['MACD','Signal Line','Histogram'], inplace=True )
            
        return df
    
    #On-Balance Volume
    def OBV(self,
            df: pd.DataFrame,
            prices:str,
            volunme:str):
        
        obv = 0
        obv_series = []
        prices = df[prices]
        volunme = df[volunme]

        for i in range(1, len(prices)):
            if prices[i] > prices [i - 1]:
                obv += volunme[i]
            elif prices[i] < prices[i - 1]:
                obv -= volunme[i]
            obv_series.append(obv)

        obv_series = pd.Series(obv_series, index=prices.index[1:])
        df['OBV'] = pd.concat([pd.Series([0], index=[prices.index[0]]), obv_series])
        
        return df
    
    def log_return(self,
               df: pd.DataFrame,
               cols:Optional[Union[list[str], str]],
               window_size: int,
               verbose: bool = True):
        
        if isinstance(cols, str):
            cols = [cols]  
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.intersection(cols)
        
        for col in numeric_cols:
            df[f'LogR_{col}_t-{window_size}'] = np.log(df[col] / df[col].shift(window_size))
        
        df.dropna(inplace=True)
        
        if verbose:
            print(f'\nDataset  first 10 rows after applying "log_return" func :\n{df.head(10)}\n')
            print(f'\nDataset  last 10 rows after applying "log_return" func :\n{df.tail(10)}\n')
        
        return df

    #Relative Strength Index (RSI)
    def RSI(self,
            df: pd.DataFrame,
            col: str = 'Adj Close',
            periods: int = 14,
            verbose: bool = True):
       
        delta = df[col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        
        rs = gain / loss
        df[f'RSI_{col}_{periods}_periods'] = 100 - (100 / (1 + rs))
        df.dropna(inplace=True)
        
        if verbose:
            print(f'\nDataset  first 10 rows after applying "RSI" func :\n{df.head(10)}\n')
            print(f'\nDataset  last 10 rows after applying "RSI" func :\n{df.tail(10)}\n')
        
        return df
    
    #Momentum
    def momentum_ratio(self,
                       df: pd.DataFrame,
                       col: str = 'Adj Close',
                       periods: int = 14,
                       verbose: bool = True):
        
        df[f'Momentum_Ratio_{col}_{periods}d'] = (df[col] / df[col].shift(periods)) * 100
        df.dropna(inplace=True)
        
        if verbose:
            print(f'\nDataset  first 10 rows after applying "momentum_ratio" func :\n{df.head(10)}\n')
            print(f'\nDataset  last 10 rows after applying "momentum_ratio" func :\n{df.tail(10)}\n')
        
        return df
    
    #True Range (TR)
    def true_range(self,
                   df: pd.DataFrame,
                   verbose: bool = True):
       
        previous_close = df['Adj Close'].shift(1)
    
        df['True_Range'] = df.apply(
            lambda row: max(
                row['High'] - row['Low'],  # Rango intradía
                abs(row['High'] - previous_close[row.name]),  # Gap al alza
                abs(row['Low'] - previous_close[row.name])    # Gap a la baja
            ), axis=1
        )
        df.reset_index(drop=True, inplace=True)
    
        if verbose:
            print(f'\nDataset  first 10 rows after applying "true_range" func :\n{df.head(10)}\n')
            print(f'\nDataset  last 10 rows after applying "true_range" func :\n{df.tail(10)}\n')
        
        return df
    
    #Calculates the Average True Range (ATR) using the True Range (TR)
    def average_true_range(self,
                           df: pd.DataFrame,
                           periods: int = 14,
                           verbose: bool = True):
       
        df = self.true_range(df)
        
        df['ATR'] = df['True_Range'].rolling(window=periods).mean()
        df.dropna(inplace=True)
     
        for i in range(periods, len(df)):
            previous_atr = df.at[i-1, 'ATR']
            current_tr = df.at[i, 'True_Range']
            df.at[i, 'ATR'] = ((previous_atr * (periods - 1)) + current_tr) / periods
        df.reset_index(drop=True, inplace=True)

        if verbose:
            print(f'\nDataset  first 10 rows after applying "average_true_range" func :\n{df.head(10)}\n')
            print(f'\nDataset  last 10 rows after applying "average_true_range" func :\n{df.tail(10)}\n')
        
        return df

    #Parabolic SAR.
    def parabolic_sar(self,
                      df: pd.DataFrame,
                      af_start: float = 0.02,
                      af_step: float = 0.02,
                      af_max: float = 0.2,
                      verbose: bool = True):
        """
        Calculates the Parabolic SAR for a given DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing 'High' and 'Low' prices.
            af_start (float): Initial Acceleration Factor (default is 0.02).
            af_step (float): Step to increase Acceleration Factor (default is 0.02).
            af_max (float): Maximum value of Acceleration Factor (default is 0.2).
            verbose (bool): If True, prints the dataset before and after applying Parabolic SAR.
            
        Returns:
            pd.DataFrame: DataFrame with a new 'PSAR' column for the Parabolic SAR values.
        """
        # Initial settings
        af = af_start
        ep = df['Low'][0]
        psar = df['High'][0]
        psar_list = [psar]
        uptrend = True

        for i in range(1, len(df)):
            prev_psar = psar

            # Calculate the current PSAR
            if uptrend:
                psar = prev_psar + af * (ep - prev_psar)
            else:
                psar = prev_psar - af * (prev_psar - ep)
            
            # Determine the trend and update EP, AF
            if uptrend:
                if df['Low'][i] < psar:
                    uptrend = False
                    psar = ep  # Set PSAR to EP when trend reverses
                    ep = df['Low'][i]
                    af = af_start
                else:
                    if df['High'][i] > ep:
                        ep = df['High'][i]
                        af = min(af + af_step, af_max)
            else:
                if df['High'][i] > psar:
                    uptrend = True
                    psar = ep
                    ep = df['High'][i]
                    af = af_start
                else:
                    if df['Low'][i] < ep:
                        ep = df['Low'][i]
                        af = min(af + af_step, af_max)

            psar_list.append(psar)
        
        df['PSAR'] = psar_list

        if verbose:
            print(f'\nDataset first 10 rows after applying "Parabolic SAR":\n{df.head(10)}\n')
            print(f'\nDataset last 10 rows after applying "Parabolic SAR":\n{df.tail(10)}\n')

        return df
    
    def commodity_channel_index(self,
                                df: pd.DataFrame,
                                periods: int = 20,
                                verbose: bool = True):
        """
        Calculates the Commodity Channel Index (CCI) for the given DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
            periods (int): Number of periods to calculate the CCI over (default is 20).
            verbose (bool): If True, prints the dataset before and after applying CCI.
            
        Returns:
            pd.DataFrame: DataFrame with a new 'CCI' column for the Commodity Channel Index values.
        """
        # Calculate the Typical Price
        df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
        
        # Calculate the Simple Moving Average of the Typical Price
        df['SMA_TP'] = df['Typical_Price'].rolling(window=periods).mean()
        
        # Calculate the Mean Deviation
        df['Mean_Deviation'] = df['Typical_Price'].rolling(window=periods).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=False)
        
        # Calculate the CCI
        df['CCI'] = (df['Typical_Price'] - df['SMA_TP']) / (0.015 * df['Mean_Deviation'])
        
        # Drop unnecessary columns and NaNs
        df.drop(columns=['Typical_Price', 'SMA_TP', 'Mean_Deviation'], inplace=True)
        df.dropna(inplace=True)
        
        if verbose:
            print(f'\nDataset first 10 rows after applying "CCI":\n{df.head(10)}\n')
            print(f'\nDataset last 10 rows after applying "CCI":\n{df.tail(10)}\n')

        return df




#---------------------------------------- Special Preprocessing ---------------------------------------#    
    def convert_currency(self,
                         dataset: pd.DataFrame,
                         columns_to_convert: str,
                         exchange_rate: float,
                         target_currency: str,
                         verbose: bool = True):
        
        if isinstance(columns_to_convert, str):
            columns_to_convert = [columns_to_convert]

        dataset_copy = dataset.copy()

        for column in columns_to_convert:
            dataset_copy[column] = dataset_copy[column] * exchange_rate
        
        if verbose:
            print(f'\nThis is the original dataset:\n{dataset.head(20)}\n')
            print(f'\nThis is the dataset after changing currency to {target_currency}:\n{dataset_copy.head(20)}\n')
        
        return dataset_copy
#---------------------------------------- Dimension reduction ---------------------------------------#        
    def apply_pca(self,
                  df: pd.DataFrame,
                  target_column: str):
        
        features = df.drop(columns=[target_column])
        target = df[target_column]

        pca = PCA(n_components=1)
        features_reduced = pca.fit_transform(features)
        
        df_reduced = pd.DataFrame(features_reduced, columns=['X_feature'], index=df.index)
        
        df_reduced[target_column] = target
        
        return df_reduced
    
    def lda_dimensionality_reduction(self,
                                     df: pd.DataFrame,
                                     target_column: str):
        
        X = df.drop(columns=[target_column])
        y = df[target_column]

        lda = LDA(n_components=1)  
        X_lda = lda.fit_transform(X, y)

        lda_df = pd.DataFrame(X_lda, columns=['X_feature'], index=df.index)
        lda_df[target_column] = y.values 

        return lda_df
    
    def linear_projection(self,
                          df: pd.DataFrame,
                          target_column: str,
                          d_model: int):
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        input_dim = X.shape[1]
        
        embedding = nn.Linear(input_dim, 1)

        X_tensor = torch.tensor(X.values, dtype=torch.float32)

        X_low_dim = embedding(X_tensor)  
        X_low_dim_scaled = X_low_dim * math.sqrt(d_model)
        X_low_dim_scaled = X_low_dim_scaled.detach().numpy()
        low_dim_df = pd.DataFrame(X_low_dim_scaled, columns=['X_feature'], index=df.index)
        low_dim_df[target_column] = y.values

        return low_dim_df

#---------------------------------------- Visualization ---------------------------------------#        
    def analize_df(self,
                   df: pd.DataFrame,
                   start: str,#To put dates on the name of the dataset.
                   end: str,
                   ticker: str,
                   set_idx: bool = False,
                   currency: bool = False,
                   describe:bool = False,
                   corr_plot:bool = False,
                   plot_corr_scatt:bool = False,
                   plot_box: bool = False) -> pd.DataFrame:
        
        df_numeric = df.round(6)

        if set_idx:
            df_numeric = self.set_col_index(df,'Date')
        
        if self.del_nulls:
            df = self.delete_blank_rows(df)

        if self.encoding_:
            df = self.encoding(df,
                               column_name = 'Y_predict',
                               start = start,
                               end = end)

        if currency:
            df_numeric = self.convert_currency(df_numeric,['Open', 'High', 'Low', 'Close', 'Adj Close'],None,'USD',verbose=True)

        #Scale dataset.
        if self.normalized:
            df_numeric = self.normalize(df_numeric,
                                        start = start,
                                        end = end,
                                        ticker=ticker,
                                        verbose=True)

        if self.standardized:
            df_numeric = self.standardize(df_numeric, verbose=True)
        
        #Redimension.
        if self.PCA_:
            df = self.apply_pca(df, 'Y_predict')

        if self.LDA_:
            df = self.lda_dimensionality_reduction(df, 'Y_predict')
        
        if self.Linear_projection_:
            df = self.linear_projection(df, 'Y_predict', d_model=128)#Change here acording of the model dim.

        corr_matrix=df_numeric.corr()

        if not df_numeric.empty and describe:
            df_numeric.info()
            print(f'\nNumber of Null values per column:\n{df_numeric.isnull().sum()}\n')
            print(f'\nDescription:\n{df_numeric.describe()}\n') #Describe all the columns even those that are not numerical
            print(f'\nDataset after preprocessing first 10 rows:\n{df_numeric.head(50)}\n\nlast 10 rows:\n{df_numeric.tail(50)}\n')
            # print(f'\n{corr_matrix}\n')

        if corr_plot:
            plot_corr_matrix(corr_matrix)

        if plot_corr_scatt:
            plot_corr_scatter(df_numeric)

        if plot_box:
            boxplot(df_numeric,'Close')

        return df_numeric

#---------------------------------------- Run Preprocessing ---------------------------------------#        
    def main(self):
        if self.technical:
            start = self.get_previous_business_day('2018-01-01')
        else:
            start=self.get_previous_business_day('2014-12-18', 14)#Set it to min 7 0r 8 days for GARCH
        end='2024-12-31'
        ticker = 'NVDA'
        if args.download:
            self.download_data(ticker=ticker,
                               start=start,
                               end=end,
                               save_dir='./datasets/download/')
        else:
            df = self.open_csv(ticker=ticker,
                               start = start,
                               end = end,
                               verbose=True)

            if self.technical:
                cols = [col for col in df.columns]
                df = self.log_return(df, cols=cols, window_size=4)
                df = self.RSI(df)
                df = self.momentum_ratio(df)
                df = self.average_true_range(df)#True range func is applied inside.
                df = self.parabolic_sar(df)
                df = self.commodity_channel_index(df)
                df = self.SMA(df)
                df = self.set_col_index(df, 'Date')
                df = self.move_col_to_end(df)
                df = self.analize_df(df,
                                    start = start,
                                    end = end,
                                    ticker=ticker,
                                    describe = True)
                
                if self.save:
                    #Save dataset for training
                    csv_path = os.path.join(self.training_path, f'{ticker} {start} to {end} processed tech.csv')
                    print(f'\nDataset saved in path: {self.training_path}\n')
                    df.to_csv(csv_path, index=True)

                    #Save dataset for prediction
                    csv_path = os.path.join(self.prediction_path, f'{ticker} {start} to {end} processed tech.csv')
                    print(f'\nDataset saved in path: {self.prediction_path}\n')
                    df.to_csv(csv_path, index=True)
                
            elif self.windows:
                # df = self.how_many(df, None, None, None, None, verbose=True)
                df = self.set_col_index(df, 'Date')
                df = self.windows_preprocessing(df,
                                                14,'Adj Close',
                                                idx=True,
                                                verbose=True)
                df = self.move_col_to_end(df, 'Adj Close')
                df = self.analize_df(df,
                                    start = start,
                                    end = end,
                                    ticker=ticker,
                                    describe = True)

                if self.save:
                    #Save dataset for training
                    csv_path = os.path.join(self.training_path, f'{ticker} {start} to {end} processed.csv')
                    print(f'\nDataset saved in path: {self.training_path}\n')
                    df.to_csv(csv_path, index=True)
                   
                    #Save dataset for prediction
                    csv_path = os.path.join(self.prediction_path, f'{ticker} {start} to {end} processed.csv')
                    print(f'\nDataset saved in path: {self.prediction_path}\n')
                    df.to_csv(csv_path, index=True)
            
            elif args.GARCH:
                df = self.set_col_index(df, 'Date')
                df = self.isolate_cols(df, 'Adj Close')
                df = self.log_return(df, cols='Adj Close', window_size=1)
                df = self.GARCH(df, 'LogR_Adj Close_t-1')
                df = self.windows_preprocessing(df, 7, ['Adj Close', 'garch_vol'], idx=True )
                df = self.analize_df(df,
                                    start = start,
                                    end = end,
                                    ticker=ticker,
                                    describe = True)
                
                if self.save:
                    #Save dataset for training
                    csv_path = os.path.join(self.training_path, f'{ticker} {start} to {end} GARCH.csv')
                    print(f'\nDataset saved in path: {self.training_path}\n')
                    df.to_csv(csv_path, index=True)
                   
                    #Save dataset for prediction
                    csv_path = os.path.join(self.prediction_path, f'{ticker} {start} to {end} GARCH.csv')
                    print(f'\nDataset saved in path: {self.prediction_path}\n')
                    df.to_csv(csv_path, index=True)


            else:
                df = self.set_col_index(df, 'Date')
                df = self.move_col_to_end(df, 'Adj Close')
                df = self.analize_df(df,
                                    start = start,
                                    end = end,
                                    ticker=ticker,
                                    describe = True)
                
                if self.save:
                    #Save dataset for training
                    csv_path = os.path.join(self.training_path, f'{ticker} {start} to {end} normal.csv')
                    print(f'\nDataset saved in path: {self.training_path}\n')
                    df.to_csv(csv_path, index=True)
                   
                    #Save dataset for prediction
                    csv_path = os.path.join(self.prediction_path, f'{ticker} {start} to {end} normal.csv')
                    print(f'\nDataset saved in path: {self.prediction_path}\n')
                    df.to_csv(csv_path, index=True)
        
#---------------------------------------- Hyperparameters ---------------------------------------#
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--open_path', type=str, 
    default= './datasets/download/', 
    help='Open dataset path')

    parser.add_argument('--training_path', type=str, 
    default= './datasets/train/', 
    help='Save dataset path')

    parser.add_argument('--prediction_path', type=str, 
    default= './datasets/test/', 
    help='Save dataset path')

    parser.add_argument('--jason_path', type=str, 
    default= './datasets/test/', 
    help='Save json file path')
    
    parser.add_argument('--download', type=bool, default=False, help='Download or not a dataset')
    parser.add_argument('--windows', type=bool, default=True, help='Transform a feature in a time series data')
    parser.add_argument('--technical', type=bool, default=False, help='Perform technical preprocessing the dataset')
    parser.add_argument('--GARCH', type=bool, default=False, help='Perform GARCH preprocessing the dataset')
    parser.add_argument('--save', type=bool, default=True, help='Save or not the dataset after preprocessing')
    parser.add_argument('--normalize', type=bool, default=True, help='Normalize the dataset or not')
    parser.add_argument('--standardize', type=bool, default=False, help='Standardize the dataset or not')
    parser.add_argument('--del_nulls', type=bool, default=False, help='Delete the null values in the dataset')
    parser.add_argument('--encoding_', type=bool, default=False, help='Encode a column and save the jason file')
    parser.add_argument('--PCA_', type=bool, default=False, help='Apply PCA dimencionality reduction in the dataset')
    parser.add_argument('--LDA_', type=bool, default=False, help='Apply LDA dimencionality reduction in the dataset')
    parser.add_argument('--Linear_projection_', type=bool, default=False, help='Apply Linear projection to reduce dim in the dataset')
  
    args = parser.parse_args([])
    return args 

if __name__ == '__main__':
    args  = get_args()
    pre = Preprocessing(args)
    pre.main()





