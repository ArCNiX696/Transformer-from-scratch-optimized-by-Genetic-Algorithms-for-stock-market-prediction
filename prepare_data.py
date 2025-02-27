#My modules
from args import get_args
args = get_args()

#Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

# Third-Party Libraries
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
import os
import pandas as pd
from tkinter import Tk, filedialog
from sklearn.model_selection import train_test_split
import logging
logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO,
    format=f'{"-" * 100}\n%(asctime)s - %(name)s - Porpuse: %(levelname)s - %(message)s\n',
    handlers=[
        logging.StreamHandler() 
    ]
)
logger = logging.getLogger(__name__)

device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#==============================================================================================#
#                                       PREPARE DATASET                                        #
#==============================================================================================#
class PrepareDataset:
    def __init__(self,
                 args: Any) -> None:
        self.stratify = args.stratify
        self.windows = args.windows
        self.debug = args.debug
        
        
    def get_file_path(self):
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select Dataset File",
            filetypes=[("CSV files","*.csv"), ("All files","*.*")]
            )
        
        return file_path
        
    def import_dataset(self,
                       verbose: bool = True):
        file_path = self.get_file_path()
        df = pd.read_csv(file_path)

        if verbose:
            print(f'\nSelected dataset: {os.path.basename(file_path)}\n')
            print(f'\n{df.head(10)}\n') 
        
        return df
    
    def split_data(self,
                   test_size:float):
        
        df = self.import_dataset(verbose=False)
    
        if df is None:
            raise SystemExit(f'\nNO DATASET FILE!:\nExecution stopped because no dataset file was loaded.\n')
       
        else:
            if 'Date' in df.columns:
                df = df.set_index('Date')

            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

            if self.stratify:
                split_type = "Stratified"
                stratify_labels = pd.cut(y, bins=5, labels=False)
                X_train ,X_test ,y_train , y_test = train_test_split(X, 
                                                                     y,
                                                                     test_size=test_size,
                                                                     stratify=stratify_labels,
                                                                     random_state=42)
      
            elif self.windows:
                split_type = "Time series"
                split_index = int((1 - test_size) * len(X))
                X_train, X_test = X[:split_index], X[split_index:]
                y_train, y_test = y[:split_index], y[split_index:]
            
            else:
                split_type = "Random"
                X_train ,X_test ,y_train , y_test = train_test_split(X, 
                                                                     y,
                                                                     test_size=test_size,
                                                                     random_state=42)

            if self.debug == 21:
                logger.debug(f"""\n\nSplit data function:\n\n
                X train shape:{X_train.shape}\nX train:\n{X_train}\n\n
                X test shape:{X_test.shape}\nX test:\n{X_test}\n\n
                y train shape:{y_train.shape}\ny train:\n{y_train}\n\n
                y test shape:{y_test.shape}\ny test:\n{y_test}\n\n
                Split type: {split_type}\n\n{"-" * 100}\n""")
                raise SystemExit(f'\nDEBUG MODE: Execution stopped intentionally.\n')
        
        return X_train ,X_test ,y_train , y_test
        
    def data_redimension(self,
                         test_size: float):
        X_train ,X_test ,y_train , y_test = self.split_data(test_size)
      
        # Reshape for Time Series
        if self.windows:
            reshape_type = "Time series"
            X_train = torch.tensor(X_train, dtype=torch.float).view(-1, X_train.shape[1], 1)#rows,windows size, num features
            X_test = torch.tensor(X_test, dtype=torch.float).view(-1, X_train.shape[1], 1)
        
        else:
            # Reshape for Random or Stratified sampling
            reshape_type = "Stratified" if self.stratify else "Random"
            X_train = torch.tensor(X_train, dtype=torch.float).view(X_train.shape[0], X_train.shape[1])
            X_test = torch.tensor(X_test, dtype=torch.float).view(X_test.shape[0], X_train.shape[1])

        #y reshape is the same in most of the cases.
        y_train = torch.tensor(y_train, dtype=torch.float).view(-1 , 1)#rows, num features
        y_test = torch.tensor(y_test, dtype=torch.float).view(-1 , 1)
        
        if self.debug == 22:
            logger.debug(f"""\n\nData redimension function:\n\n
            X train shape:{X_train.shape}\nX train:\n{X_train}\n\n
            X test shape:{X_test.shape}\nX test:\n{X_test}\n\n
            y train shape:{y_train.shape}\ny train:\n{y_train}\n\n
            y test shape:{y_test.shape}\ny test:\n{y_test}\n\n
            Reshape type: {reshape_type}\n\n{"-" * 100}\n""")
            raise SystemExit(f'\nDEBUG MODE: Execution stopped intentionally.\n')
            
        return X_train ,X_test ,y_train , y_test
    
#==============================================================================================#
#                                       PREPARE DATASET                                        #
#==============================================================================================#
class DatasetLoader(Dataset):
    def __init__(self,
                 X: torch.Tensor,
                 y: torch.Tensor) -> None:
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
    @staticmethod
    def prepare_data(batch_size: int,
                     test_size: float):
        
        data = PrepareDataset(args=args)
        x_train ,x_test ,y_train , y_test = data.data_redimension(test_size)

        if args.windows or args.stratify:
            datasetLoader_type = "Stratified" if args.stratify else "Time Series"
            train_dataset = DatasetLoader(x_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        
        else:
            datasetLoader_type = "Random"
            train_dataset = DatasetLoader(x_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = DatasetLoader(x_test, y_test)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        if args.debug == 23:
            logger.debug(f"""\n\nPrepare Data Function:\n\n
            Dataset Loader Type: {datasetLoader_type}\n
            Batch Size: {batch_size}\n
            Test Size: {test_size}\n
            Training Data Shapes -> X: {x_train.shape}, y: {y_train.shape}\n
            Validation Data Shapes -> X: {x_test.shape}, y: {y_test.shape}\n
            """)
            
            for _, b in enumerate(train_loader):
                x_b, y_b = b[0].to(device), b[1].to(device)
                logger.debug(f'\nBatch -> x_b shape: {x_b.shape}, y_b shape: {y_b.shape}\n\n{"-" * 100}\n')
                break

            raise SystemExit(f'\nDEBUG MODE: Execution stopped intentionally.\n')
        
        return train_loader , val_loader
    
if __name__ == '__main__':
    args = get_args()
    DatasetLoader.prepare_data(args.batch_size, args.test_size)