#My Modules 
import plots as pl

#Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchmetrics import MeanAbsolutePercentageError
from torchmetrics import R2Score

# Third-Party Libraries
from typing import Optional, Any, Callable
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import random
import argparse
import math
import time
import os
from tqdm import tqdm
import gc
import logging

ticker = 'NVDA'
device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"\nUsing CUDA as device: {torch.cuda.get_device_name(device)}\n")
else:
    print("\nUsing CPU\n")

#==============================================================================================#
#                                          DEBUG ARGS                                          #
#==============================================================================================#
def debug_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=int, default=None,
                        help="""To performe some prints in key points of the code to debug,
                        Transformer Model key points:
                        debug = None ---> Nothing
                        debug = 1 ---> Input embedding
                        debug = 2 ---> Positional encoding
                        debug = 3 ---> Multihead attention
                        debug = 4 ---> Layer Normalization
                        debug = 5 ---> Feed Forward
                        debug = 6 ---> Residual Connection
                        debug = 7 ---> Encoder LayerAAPL 2017-12-12 to 2024-12-01 processed
                        debug = 8 ---> Decoder Layer
                        debug = 9 ---> Model Encoder block
                        debug = 10 ---> Model Decoder block
                        debug = 11 ---> Transformer Model
                        debug = 12 ---> Model Training debug
                        debug = 14 ---> Training Gt and Predictions visualization
                        debug = 15 ---> Model Validation debug
                        debug = 16 ---> Validation Gt and Predictions visualization
                        debug = 17 ---> Input, src and tgt data visualization in the Prediction func.
                        
                        Prepare data module:
                        debug = 21 ---> Split Data Func.
                        debug = 22 ---> Data Redimension Func.
                        debug = 23 ---> Prepare Data Func.
                        """)
    
    debug_args = parser.parse_args([])
    return debug_args 

debug = debug_args()

logging.basicConfig(
    level=logging.DEBUG if debug.debug else logging.INFO,
    format=f'{"-" * 100}\n%(asctime)s - %(name)s - Porpuse: %(levelname)s - %(message)s\n',
    handlers=[
        logging.StreamHandler() 
    ]
)
logger = logging.getLogger(__name__)

#==============================================================================================#
#                                       PREPARE DATASET                                        #
#==============================================================================================#
class PrepareDataset:
    def __init__(self,
                 args: Any) -> None:
        self.stratify = args.stratify
        self.windows = args.windows
        self.debug = debug.debug
        
    def import_dataset(self,
                       verbose: bool = True):
        file_path = os.path.join('./datasets/train/',
                                 'NVDA 2014-11-28 to 2024-12-31 processed.csv')
        
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
    def prepare_data(args):
        data = PrepareDataset(args=args)
        x_train ,x_test ,y_train , y_test = data.data_redimension(args.test_size)

        if args.windows or args.stratify:
            datasetLoader_type = "Stratified" if args.stratify else "Time Series"
            train_dataset = DatasetLoader(x_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        
        else:
            datasetLoader_type = "Random"
            train_dataset = DatasetLoader(x_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        val_dataset = DatasetLoader(x_test, y_test)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        if debug.debug == 23:
            logger.debug(f"""\n\nPrepare Data Function:\n\n
            Dataset Loader Type: {datasetLoader_type}\n
            Batch Size: {args.batch_size}\n
            Test Size: {args.test_size}\n
            Training Data Shapes -> X: {x_train.shape}, y: {y_train.shape}\n
            Validation Data Shapes -> X: {x_test.shape}, y: {y_test.shape}\n
            """)
            
            for _, b in enumerate(train_loader):
                x_b, y_b = b[0].to(device), b[1].to(device)
                logger.debug(f'\nBatch -> x_b shape: {x_b.shape}, y_b shape: {y_b.shape}\n\n{"-" * 100}\n')
                break

            raise SystemExit(f'\nDEBUG MODE: Execution stopped intentionally.\n')
        
        return train_loader , val_loader
    
#==============================================================================================#
#                                         MODEL LAYERS                                         #
#==============================================================================================#
class InputEmbeddings(nn.Module):
    def __init__(self,
                 args: Any) -> None:
        super().__init__()
        self.d_model: int = args.d_model
        self.embedding: nn.Linear = nn.Linear(args.input_dim, self.d_model)
    
    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        
        if debug.debug == 1:
            assert isinstance(x, torch.Tensor), f"Expected x to be a Tensor, but got {type(x)}"
            logger.debug(f'\n\nx:\n{x}\n{"-" * 100}\n')
            raise SystemExit(f'\nDEBUG MODE: Execution stopped intentionally.\n')
        
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, 
                 args: Any) -> None:
        super().__init__()
        self.d_model: int = args.d_model 
        self.max_len: int = args.max_len
        self.dropout: nn.Dropout = nn.Dropout(args.dropout)

        position = torch.arange(0, self.max_len , dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0 , self.d_model , 2) * -(torch.log(torch.tensor(10000.0)) / self.d_model))
        
        pe = torch.zeros(self.max_len, self.d_model).to(device)
        pe[: , 0::2] = torch.sin(position * div_term)#Pe even 
        pe[: , 1::2] = torch.cos(position * div_term)#Pe odd
        pe = pe.unsqueeze(0).to(device)
        self.register_buffer('pe',pe)
        
    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        
        if debug.debug == 2:
            assert isinstance(x, torch.Tensor), f'\nExpected x to be a tensor but you got: {type(x)}\n'
            logger.debug(f'\n\nShape of x:\n{x.shape}\n\nShape of self.pe:\n{self.pe.shape}\n{"-" * 100}\n')
            raise SystemExit(f'\nDEBUG MODE: Execution stopped intentionally.\n')
        
        x = x + self.pe[: ,:x.size(1) ,:].to(device) 
        return self.dropout(x)
    
class MultiHeadAtention(nn.Module):
    def __init__(self,
                 args: Any,
                 enc_or_dec:str,
                 layer_idx: int) -> None:
        super().__init__()
        self.plot_attn = args.plot_attn
        self.enc_or_dec: str = enc_or_dec
        self.layer_idx: int = layer_idx
        self.d_model: int = args.d_model
        self.num_heads: int = args.nhead
        self.nlayers: int = args.nlayers
        assert self.d_model % self.num_heads == 0 , f'\nWARNING:\nd_model = {self.d_model} must be divisible for the number of heads which is = {self.num_heads}\n'
        self.d_k: int = self.d_model // self.num_heads

        #Lineal layers for Q, K ,V and the proyection w_o (fully connected layers)
        self.w_q: nn.Linear = nn.Linear(self.d_model, self.d_model).to(device)
        self.w_k: nn.Linear = nn.Linear(self.d_model, self.d_model).to(device)
        self.w_v: nn.Linear = nn.Linear(self.d_model, self.d_model).to(device)
        self.w_o: nn.Linear = nn.Linear(self.d_model, self.d_model).to(device)
        self.dropout: nn.Dropout = nn.Dropout(args.dropout)
        self.scale: float = math.sqrt(self.d_k)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        query, key, value = query.to(device), key.to(device), value.to(device)
        
        if not (query.device == device and key.device == device and value.device == device):
            logger.warning(f'\nquery, key, value Tensors are not on device {device}\n')

        if mask is not None:
            mask = mask.to(device) 

        batch_size = query.size(0)
        # if query.dim() == 2:  #ATTENTION: in case query has the form of (batch_size, d_model) activate the line below.
        #     query = query.unsqueeze(1)
        
        #Connect Q, K, V
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_k(value)
        
        # Divide Q, K, V in multi heads y reorganize the tensor [batch_size, self.num_heads, seq_len(-1 to calculate), self.d_k]
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)#[batch_size, num_heads, seq_len_q, d_k]
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scaled_scores = torch.matmul(q , k.transpose(-2 ,-1)) / self.scale #k.transpose(-2, -1) is [batch_size, num_heads, d_k, seq_len_k] the result will be [batch_size, num_heads, seq_len_q, seq_len_k] 
        
        if mask is not None:                                                   
            scaled_scores = scaled_scores.masked_fill(mask == 0, float('-inf'))#replace the vals of 0 for -inf in the mask.
                                                                               #softmax of -inf = 0 , so the masked positions will not contribute to the attention.
        attn = torch.softmax(scaled_scores, dim=-1)# apply softmax to the las dim seq_len_k
        attn = self.dropout(attn)

        if self.plot_attn:
            pl.plot_attention(attn_weights = attn,
                              keys = k,
                              queries = q,
                              model = 1,
                              enc_or_dec = self.enc_or_dec,
                              layer_idx = self.layer_idx)
            
            if self.enc_or_dec == 'Decoder MUlHead 2' and self.layer_idx == self.nlayers - 1:
                print("All heatmaps have been successfully saved.")
                raise SystemExit("Stopping the code execution after saving all heatmaps.")
       
        # Apply attentionn to V
        x = torch.matmul(attn, v)
        
        # Concatenate the heads and project them(fully connected)
        #hanges from [batch_size, num_heads, seq_len_q, d_v] to [batch_size, seq_len_q, num_heads, d_v]
        #x.contiguous() ensures that the data in memory is contiguous.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)#Then, view(batch_size, -1, self.d_model) reshapes the tensor to [batch_size, seq_len_q, d_model], where d_model = num_heads * d_v.
        x = self.w_o(x) #linear

        if debug.debug == 3:
            logger.debug(f'\nShape of query:\n{query.shape}\nkey:\n{key.shape}\nvalue:\n{value.shape}\n')
            logger.debug(f'\nShape of Q:\n{q.shape}\nK:\n{k.shape}\nV:\n{v.shape}\n')
            logger.debug(f'\nShape after final projection - X:\n{x.shape}\n{"-" * 100}\n')
            raise SystemExit(f'\nDEBUG MODE: Execution stopped intentionally.\n')
        return x
 
class LayerNormalization(nn.Module):
    def __init__(self,
                 args: Any) -> None:
        super().__init__()
        # nn.Parameter, torch interpretates this as a parameter of the model so it will be optimized during the training.
        self.gamma: nn.Parameter = nn.Parameter(torch.ones(args.d_model))#Lr
        self.beta: nn.Parameter = nn.Parameter(torch.zeros(args.d_model))#Bias
        self.eps: float = args.LN_eps

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean(dim=-1 , keepdim=True)
        sigma = x.std(dim=-1 , keepdim=True)
        norm_x = (x - x_mean) / (sigma + self.eps)
        
        if debug.debug == 4:
            logger.debug(f'\n\nx:\n{x}\n\nShape of x:\n{x.shape}\\n')
            logger.debug(f"""\nX shape:\n{x.shape}\nX mean:\n{x_mean}\n
                         \nsigma shape:\n{sigma.shape}\nsigma:\n{sigma}\n
                         \nnorm_x shape:\n{norm_x.shape}\nnorm_x:\n{norm_x}\n\n{"-" * 100}\n""")
            raise SystemExit(f'\nDEBUG MODE: Execution stopped intentionally.\n')
        
        return self.gamma * norm_x + self.beta
    
class FeedForward(nn.Module):
    def __init__(self,
                 args: Any) -> None:
        super().__init__()
        self.d_model: int = args.d_model
        self.d_ff: int = args.d_ff
        self.linear1: nn.Linear = nn.Linear(self.d_model , self.d_ff).to(device)
        self.relu: nn.ReLU = nn.ReLU()
        self.dropout: nn.Dropout = nn.Dropout(args.dropout)
        self.linear2: nn.Linear = nn.Linear(self.d_ff , self.d_model).to(device)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        
        assert isinstance(x, torch.Tensor), f"\nWARNING: Expected x to be a Tensor, but got {type(x)}\n"
        if debug.debug == 5:  # Debugging step before forward pass
            logger.debug(f'\nInput Shape before FeedForward:\n{x.shape}\nx: {x}\n\n')
        
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        if debug.debug == 5:  # Debugging step after forward pass
            logger.debug(f'\nOutput Shape after FeedForward:\n{x.shape}\nx: {x}\n\n{"-" * 100}\n\n')
            raise SystemExit(f'\nDEBUG MODE: Execution stopped intentionally.\n')
        
        return x
    
class ResidualConnection(nn.Module):
    def __init__(self,
                 args: Any) -> None:
        super().__init__()
        self.layer_norm: nn.Module = LayerNormalization(args).to(device)
        self.dropout: nn.Dropout = nn.Dropout(args.dropout)

    def forward(self,
                x: torch.Tensor,
                sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        
        assert callable(sublayer), f"\nExpected 'sublayer' to be callable, but got {type(sublayer)}\n"
        
        if debug.debug == 6: 
            logger.debug(f"""\nx Shape:\n{x.shape}\nx: {x}\n\n
            sublayer type: should be a callable\n{type(sublayer)}\n\n{"-" * 100}\n\n""")
            raise SystemExit(f'\nDEBUG MODE: Execution stopped intentionally.\n')
        
        return x + self.dropout(sublayer(self.layer_norm(x)))#sublayer is the multihead or feeded layer
        
class EncoderLayer(nn.Module):
    def __init__(self,
                 args: Any,
                 layer_idx: int) -> None:
        super().__init__()
        self.multi_head: Callable[[torch.Tensor], torch.Tensor] = MultiHeadAtention(
            args, 'Encoder MUlHead 1', layer_idx).to(device)
        self.feed_forward: Callable[[torch.Tensor], torch.Tensor] = FeedForward(args).to(device)
        self.residual1: Callable[[torch.Tensor], torch.Tensor] = ResidualConnection(args).to(device)
        self.residual2: Callable[[torch.Tensor], torch.Tensor] = ResidualConnection(args).to(device)

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if debug.debug == 7:
            logger.debug(f'\nEncoder layer forward input:\nx Shape:\n{x.shape}\nx:\n{x}\n\n')

        x = x.to(device)
        assert x.device == device, f'\n\nWARNING: x Tensors in the Encoder layer are not on device {device}\n\n'
         
        if mask is not None:
            mask = mask.to(device)
        x = self.residual1(x, lambda x: self.multi_head(x, x, x, mask))
        x = self.residual2(x, self.feed_forward)
        
        if debug.debug == 7:
            logger.debug(f'\nEncoder layer forward output:\nx Shape:\n{x.shape}\nx:\n{x}\n\n{"-" * 100}\n\n')
            raise SystemExit(f'\nDEBUG MODE: Execution stopped intentionally.\n')
        
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self,
                 args: Any,
                 layer_idx: int) -> None:
        super().__init__()
        self.multi_head1: Callable[[torch.Tensor], torch.Tensor] = MultiHeadAtention(
            args,'Decoder MUlHead 1', layer_idx).to(device)
        self.multi_head2: Callable[[torch.Tensor], torch.Tensor] = MultiHeadAtention(
            args,'Decoder MUlHead 2', layer_idx).to(device)
        self.residual1: Callable[[torch.Tensor], torch.Tensor] = ResidualConnection(
            args).to(device)
        self.residual2: Callable[[torch.Tensor], torch.Tensor] = ResidualConnection(args).to(device)
        self.residual3: Callable[[torch.Tensor], torch.Tensor] = ResidualConnection(args).to(device)
        self.feed_forward: Callable[[torch.Tensor], torch.Tensor] = FeedForward(args).to(device)

    def forward(self,
                x: torch.Tensor,
                encoder_out: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if debug.debug == 8:
            logger.debug(f'\nDecoder layer forward input:\nx Shape:\n{x.shape}\nx:\n{x}\n\n')
        
        x = x.to(device)
        assert x.device == device, f'\n\nWARNING: x Tensors in the Decoder layer are not on device {device}\n\n'
        
        encoder_out = encoder_out.to(device)
        assert encoder_out.device == device, f'\n\nWARNING: encoder_out Tensors in the Decoder layer are not on device {device}\n\n'
        
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(device)
        
        if src_mask is not None:
            src_mask = src_mask.to(device)
        
        x = self.residual1(x, lambda x: self.multi_head1(x, x, x, tgt_mask))
        x = self.residual2(x, lambda x: self.multi_head2(x, encoder_out, encoder_out, src_mask))
        x = self.residual3(x, self.feed_forward)
        
        if debug.debug == 8:
            logger.debug(f'\nDecoder layer forward output:\nx Shape:\n{x.shape}\nx:\n{x}\n\n{"-" * 100}\n\n')
            raise SystemExit(f'\nDEBUG MODE: Execution stopped intentionally.\n')
        
        return x
#==============================================================================================#
#                                         MODEL BLOCKS                                         #
#==============================================================================================#
class ModelEncoder(nn.Module):
    def __init__(self,
                 args: Any) -> None:
        super().__init__()
        self.d_model: int = args.d_model
        self.num_layers: int = args.nlayers
        self.input_embedding: Callable[[torch.Tensor], torch.Tensor] = InputEmbeddings(args).to(device)
        self.pos_encoder: Callable[[torch.Tensor], torch.Tensor] = PositionalEncoding(args).to(device)
        self.layers: nn.ModuleList = nn.ModuleList([EncoderLayer(args, layer_idx=i).to(device) for i in range(self.num_layers)])
        self.layer_norm: Callable[[torch.Tensor], torch.Tensor] = LayerNormalization(args).to(device)

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if debug.debug == 9:
            logger.debug(f'\nEncoder block forward src input:\nsrc Shape:\n{src.shape}\nsrc:\n{src}\n\n')

        src = src.to(device)
        assert src.device == device, f'\n\nWARNING: src Tensors in the Encoder block are not on device {device}\n\n'
        
        if src_mask is not None:
            src_mask = src_mask.to(device)
        
        src = self.input_embedding(src)
        src = self.pos_encoder(src)
        
        for layer in self.layers:
            src = layer(src, src_mask)

        src = self.layer_norm(src)
        
        if debug.debug == 9:
            logger.debug(f'\nEncoder block forward src output:\nsrc Shape:\n{src.shape}\nsrc:\n{src}\n\n{"-" * 100}\n\n')
            raise SystemExit(f'\nDEBUG MODE: Execution stopped intentionally.\n')
        
        return src

class ModelDecoder(nn.Module):
    def __init__(self,
                 args: Any) -> None:
        super().__init__()
        self.d_model: int = args.d_model
        self.num_layers: int = args.nlayers
        self.input_embedding: Callable[[torch.Tensor], torch.Tensor] = InputEmbeddings(args).to(device)
        self.pos_encoder: Callable[[torch.Tensor], torch.Tensor] = PositionalEncoding(args).to(device)
        self.layers: nn.ModuleList = nn.ModuleList([DecoderLayer(args, layer_idx=i).to(device) for i in range(self.num_layers)])
        self.layer_norm: Callable[[torch.Tensor], torch.Tensor] = LayerNormalization(args).to(device)
        
    def forward(self,
                tgt: torch.Tensor,
                src: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if debug.debug == 10:
            logger.debug(f'\nDecoder block forward src input:\nsrc Shape:\n{src.shape}\nsrc:\n{src}\n\n')
        
        src = src.to(device)
        assert src.device == device, f'\n\nWARNING: src Tensors in the Decoder block are not on device {device}\n\n'
        
        tgt = tgt.to(device)
        assert tgt.device == device, f'\n\nWARNING: tgt Tensors in the Decoder block are not on device {device}\n\n'

        tgt = self.input_embedding(tgt)

        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(device)
        
        if src_mask is not None:
            src_mask = src_mask.to(device)

        tgt = self.pos_encoder(tgt)
        
        for layer in self.layers:
            tgt = layer(tgt, src, tgt_mask, src_mask)
        tgt = self.layer_norm(tgt)
        
        if debug.debug == 10:
            logger.debug(f'\nDecoder block forward tgt output:\ntgt Shape:\n{tgt.shape}\ntgt:\n{tgt}\n\n{"-" * 100}\n')
            raise SystemExit(f'\nDEBUG MODE: Execution stopped intentionally.\n')
        
        return tgt
#==============================================================================================#
#                                       TRANSFORMER MODEL                                      #
#==============================================================================================#
class TransformerModel(nn.Module):
    def __init__(self,
                 args: Any) -> None:
        super().__init__()
        self.encoder: Callable[[torch.Tensor], torch.Tensor] = ModelEncoder(args).to(device)
        self.decoder: Callable[[torch.Tensor], torch.Tensor] = ModelDecoder(args).to(device)
        self.fc_out: nn.Linear = nn.Linear(args.d_model, args.output_dim).to(device)
        self.apply(self.weights_init)
        
    def weights_init(self,
                     m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def create_look_ahead_mask(self,
                               size: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size), diagonal=1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if debug.debug == 11:
            logger.debug(f'\nTransformer Model src input:\nsrc Shape:\n{src.shape}\nsrc:\n{src}\n\n')

        src = src.to(device)
        assert src.device == device, f'\n\nWARNING: src Tensors in the Transformer Model are not on device {device}\n\n'
        
        tgt = tgt.to(device)
        assert tgt.device == device, f'\n\nWARNING: tgt Tensors in the Transformer Model are not on device {device}\n\n'
        
        if src_mask is not None:
            src_mask = src_mask.to(device)

        tgt_mask = self.create_look_ahead_mask(tgt.size(1)).to(device)
        
        if debug.debug == 11:
            if src_mask is not None:
                logger.debug(f'\nTransformer Model src_mask shape:\n{src_mask.shape}\nsrc_mask:\n{src_mask}\n\n')
            else:
                logger.debug(f'\nTransformer Model src_mask:\n{src_mask}\n\n')
            logger.debug(f'\nTransformer Model tgt_mask shape:\n{tgt_mask.shape}\ntgt_mask:\n{tgt_mask}\n\n')
        
        encoder_output = self.encoder(src, src_mask)
        
        if debug.debug == 11:
            logger.debug(f'\nTransformer Model encoder_output shape:\n{encoder_output.shape}\nTransformer Model encoder_output:\n{encoder_output}\n\n')
          
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        if debug.debug == 11:
            logger.debug(f'\nTransformer Model decoder_output shape:\n{decoder_output.shape}\nTransformer Model decoder_output:\n{decoder_output}\n\n')
          
        output = self.fc_out(decoder_output)
        if debug.debug == 11:
            logger.debug(f'\nTransformer Model Output shape:\n{output.shape}\nTransformer Model Output:\n{output}\n\n{"-" * 100}\n')
            raise SystemExit(f'\nDEBUG MODE: Execution stopped intentionally.\n')
        
        return output
#==============================================================================================#
#                                TRAINING, VALIDATION AND PREDICTION                            #
#==============================================================================================#
class ModelOps: #Model Operations
    def __init__(self,
                 args: Any):
        self.MSE: nn.MSELoss = nn.MSELoss()
        self.MAE: nn.L1Loss = nn.L1Loss()
        self.MAPE = MeanAbsolutePercentageError().to(device)
        self.R2_train: R2Score = R2Score().to(device) 
        self.R2_validation: R2Score = R2Score().to(device)
        self.R2_prediction: R2Score = R2Score().to(device)
        self.R2_simulation: R2Score = R2Score().to(device)
        self.model: callable[[torch.Tensor], torch.Tensor] = TransformerModel(args=args).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.best_loss: float = float('inf')
        self.training_losses: list[float] = []
        self.validation_losses: list[float] = []
        self.args = args
        # self.epochs = args.epochs
        # self.early_stop = args.early_stop
        self.save_best_model: str = args.save_best_model
        self.stats_dir: str = args.stats_dir
        self.early_stop_counter: int = 0
                                              #For Prediction
        # self.predict_data_path: str = args.predict_data_path
        # self.d_model: int = args.d_model

#------------------------------------------------ Training --------------------------------------------------#
    def train(self):
        self.train_loader, self.val_loader = DatasetLoader.prepare_data(args=self.args)
        for self.epoch in range(self.args.epochs):
            
            if self.early_stop_counter >= self.args.early_stop:
                print(f'Early stopping in epoch: {self.epoch + 1}')
                break
            
            self.model.train()
            total_loss = 0.0

            for X_batch, y_batch in tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1}/{self.args.epochs}'):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                src = X_batch[:, :self.args.max_len - 1, :]
                tgt = X_batch[:, self.args.max_len - 1:, :] 
                self.optimizer.zero_grad()
                output = self.model(src, tgt)
                output = output[:, -1, :]
                output = output.view(-1)
                y_batch = y_batch.view(-1)
                
                if debug.debug == 12:
                    src = X_batch.view(-1) 
                    logger.debug(f"""\nsrc.shape: {src.shape}\nsrc:\n{src}\n\n
                                tgt.shape: {y_batch.shape}\ntgt:\n{y_batch}\n\n
                                Output.shape: {output.shape}\nOutput:\n{output}\n\n{"-" * 100}\n\n""")
                    raise SystemExit(f'\nDEBUG MODE: Execution stopped intentionally.\n')
                
                if debug.debug == 14:#Visualize Gt vs Predictions
                    print(f"""\ny_batch.shape: {y_batch.shape}\ny_batch:\n{y_batch}\n
                          Output.shape: {output.shape}\nOutput:\n{output}\n\n""")
                    if self.epoch == 1:
                        raise SystemExit(f'\nDEBUG MODE: Execution stopped intentionally.\n')

                loss = self.MSE(output, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                self.R2_train.update(output , y_batch)

            epoch_loss = total_loss / len(self.train_loader)
            R2_train_score = self.R2_train.compute().item()
            print(f'\nTraining Loss: {epoch_loss:.6f}\nTraining R2 score: {R2_train_score:.4f}\n')
            self.R2_train.reset()
            self.training_losses.append(epoch_loss)
            self.validation()

#------------------------------------------------ Validation --------------------------------------------------#
    def validation(self):
        self.model.eval()
        val_loss = 0
        MAE_loss = 0.0
        MAPE_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in tqdm(self.val_loader, desc=f'Validation Epoch {self.epoch + 1}'):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                src = X_batch[:, :self.args.max_len - 1, :]
                tgt = X_batch[:, self.args.max_len - 1:, :]
                output = self.model(src, tgt)
                output = output[:, :, :]
                output = output.view(-1)
                y_batch = y_batch.view(-1)
                
                if debug.debug == 15:
                    src = X_batch.view(-1) 
                    logger.debug(f"""\nsrc.shape: {src.shape}\nsrc:\n{src}\n\n
                                tgt.shape: {y_batch.shape}\ntgt:\n{y_batch}\n\n
                                Output.shape:\n{output.shape}\nOutput:\n{output}\n\n{"-" * 100}\n\n""")
                    raise SystemExit(f'\nDEBUG MODE: Execution stopped intentionally.\n')
                
                if debug.debug == 16:#Visualize Gt vs Predictions
                    print(f"""\ny_batch.shape: {y_batch.shape}\ny_batch:\n{y_batch}\n
                          Output.shape: {output.shape}\nOutput:\n{output}\n\n""")
                #MAE loss
                mae_loss = self.MAE(output, y_batch)
                MAE_loss += mae_loss.item()
                MAE_loss = MAE_loss / len(self.val_loader)
                #MAPE loss
                mape_loss = self.MAPE(output, y_batch)
                MAPE_loss += mape_loss.item()
                MAPE_loss = MAPE_loss / len(self.val_loader)
                #Validation loss
                loss = self.MSE(output, y_batch)
                val_loss += loss.item()
                #R2
                self.R2_validation.update(output , y_batch)#R2

        avg_loss = val_loss / len(self.val_loader)
        R2_val_score = self.R2_validation.compute().item()
        self.R2_validation.reset()
        self.validation_losses.append(avg_loss)
        print(f'\nvalidation loss Avg : {avg_loss:.4f}\nR2 validation score: {R2_val_score:.4f} \n{"*" * 120}\n')
        
        if avg_loss < self.best_loss and R2_val_score >= 0.5:
            print(f'New best model found in epoch: {self.epoch + 1}')
            self.best_loss = avg_loss
            self.plot_evaluation(f'best val loss {self.best_loss:.4f},R2 {R2_val_score:.4f},epoch {self.epoch + 1}',plot=False)
            file_name = f'{ticker} R2_{R2_val_score:.4f}_loss_{self.best_loss:.4f}_best_model_epoch_{self.epoch + 1}.pth'
            torch.save(self.model.state_dict(), os.path.join(self.save_best_model, file_name))
        
            log_name = f'{ticker} R2_{R2_val_score:.4f}_loss_{self.best_loss:.4f}_best_model_epoch_{self.epoch + 1}.txt'
            with open(os.path.join(self.save_best_model, log_name), 'a') as log_file:
                log_file.write(f'best_model_epoch: {self.epoch + 1}, MSE: {self.best_loss:.4f}, R2: {R2_val_score:.4f},MAE: {MAE_loss},MAPE: {MAPE_loss}, opt: Adam\n')
                log_file.write(f'Hyperparameters: \n')
                log_file.write(f'  - test_size: {self.args.test_size}\n')
                log_file.write(f'  - batch_size: {self.args.batch_size}\n')
                log_file.write(f'  - d_model: {self.args.d_model}\n')
                log_file.write(f'  - dropout: {self.args.dropout}\n')
                log_file.write(f'  - nhead: {self.args.nhead}\n')
                log_file.write(f'  - nlayers: {self.args.nlayers}\n')
                log_file.write(f'  - d_ff: {self.args.d_ff}\n')
                log_file.write(f'  - epochs: {self.args.epochs}\n')
                log_file.write(f'  - learning_rate: {self.args.learning_rate}\n')
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1  
#------------------------------------------------ Plots --------------------------------------------------#
    def plot_evaluation(self,file_name:str,plot=False):
        if not os.path.exists(self.stats_dir):
            os.makedirs(self.stats_dir)

        epochs = range(1, len(self.training_losses) + 1)
        
        plt.figure(figsize=(8,8))
        plt.subplot(2,1,1)
        plt.plot(epochs, self.training_losses , label= 'Training losses',color='blue',marker='o')
        plt.title('Training loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2,1,2) #row , column , number of plot in the figure
        plt.plot(epochs, self.validation_losses , label= 'Validation losses',color='orange',marker='x')
        plt.title('Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(self.stats_dir,f'{file_name}.png')
        plt.savefig(plot_path)
        plt.close()
        gc.collect() 
        
        if plot == True:
            plt.show()
#==============================================================================================#
#                                               MAIN                                           #
#==============================================================================================#
    def main(self):
        self.train()
        return self.best_loss
        
def get_gemga_args():
    parser = argparse.ArgumentParser()
    #GEMGA
    parser.add_argument('--population_size', type=int, default=10, help='Number of individuals in the chromosome of GEMGA')
    parser.add_argument('--chromosome_len', type=int, default=22, help='Number of genes or len of the chromosome of GEMGA')
    parser.add_argument('--generations', type=int, default=5, help='Number of generations of GEMGA algorithms.')
    parser.add_argument('--adaptivity_factor', type=float, default=1.0, help='Probability of adaptivity of GEMGA algorithms.')
    parser.add_argument('--test_mode', type=bool, default=False, help='Use Small values as Hyperparameters just for test the algorithm')
    args = parser.parse_args([])
    return args

"""
Messy algorithms, including GEMGA, use variable-length chromosomes to efficiently 
identify key gene combinations, making them faster than traditional GAs. GEMGA, 
specifically, enhances this process by focussing on the most promising gene expressions,
leading to quicker and more effective optimization.
"""
"""
Calculates the number of possible combinations that can be represented with 'n' bits.
Formula: 2^n, where 'n' is the number of bits.
Example: With 3 bits, you can represent 2^3 = 8 possible combinations.
"""
class GemGa:
    def __init__(self, gemga_args):
        self.test_mode = gemga_args.test_mode
        self.population_size = gemga_args.population_size
        self.chromosome_len = gemga_args.chromosome_len
        self.generations = gemga_args.generations
        self.adaptivity = gemga_args.adaptivity_factor
        self.best_losses_per_generation = []

    def create_binary_individual(self):
        return np.random.randint(0, 2, self.chromosome_len)
 
    def initial_population(self,verbose=False):
        population = np.array([self.create_binary_individual() for _ in range(self.population_size)])
        if verbose:
            np.set_printoptions(threshold=np.inf)
            print(f'\n{"-" * 100}\nInitial population:\n{population}\n{"-" * 100}\n')
        return population
    
    def decode(self, genes, values:np.array):
        bit_srt = ''.join(str(gene) for gene in genes)
        real_val = int(bit_srt, 2)
        return values[real_val]
    
    def hyperparameters(self, individual:np.array, verbose=False):
        if self.test_mode:
            test_size = [0.1, 0.1, 0.1, 0.1]
            batch_size = [128, 128, 128, 256]
            d_model = [16, 16, 16, 16, 16, 16, 16, 16, 16] #3 genes
            dropout = [0.1, 0.2, 0.3, 0.4] 
            nhead = [4, 8, 4, 8]
            nlayers = [4, 8, 4, 8]
            d_ff = [32, 32, 32, 32]
            epochs = [2, 2, 2, 2, 2, 3, 3, 2] #3 genes
            optimizer = ['adam', 'RMSprop', 'adagrad', 'adadelta']
            lr = [0.1, 0.01, 0.01, 0.01]
        else:
            test_size = [0.3, 0.2, 0.1, 0.4]
            batch_size = [32, 16, 32, 8]
            d_model = [32, 32, 512, 64, 128, 256, 512, 1024, 32] #3 genes
            dropout = [0.1, 0.2, 0.3, 0.4] 
            nhead = [4, 8, 16, 32]
            nlayers = [4, 8, 16, 32]
            d_ff = [64, 128, 256, 512]
            epochs = [900, 1000, 1750, 1500, 1850, 1400, 1650, 1600] #3 genes
            optimizer = ['adam', 'RMSprop', 'adagrad', 'adadelta']
            lr = [3e-4, 0.01, 0.001, 0.0001]

        test_size_genes = self.decode(individual[:2], test_size)
        batch_size_genes = self.decode(individual[2:4], batch_size)
        d_model_genes = self.decode(individual[4:7], d_model)
        dropout_genes = self.decode(individual[7:9], dropout)
        nhead_genes = self.decode(individual[9:11], nhead)
        nlayers_genes = self.decode(individual[11:13], nlayers)
        d_ff_genes = self.decode(individual[13:15], d_ff)
        epochs_genes = self.decode(individual[15:18], epochs)
        optimizer_genes = self.decode(individual[18:20], optimizer)
        lr_genes = self.decode(individual[20:], lr)
        
        if verbose:
            print(f'\n{"-" * 100}\ntest_size: {test_size_genes}')
            print(f'batch_size: {batch_size_genes}')
            print(f'd_model: {d_model_genes}')
            print(f'dropout: {dropout_genes}')
            print(f'nhead: {nhead_genes}')
            print(f'nlayers: {nlayers_genes}')
            print(f'd_ff: {d_ff_genes}')
            print(f'epochs: {epochs_genes}')
            print(f'optimizer: {optimizer_genes}')
            print(f'learning_rate: {lr_genes}\n{"-" * 100}\n')

        return test_size_genes, batch_size_genes, d_model_genes, dropout_genes, nhead_genes, nlayers_genes, d_ff_genes, epochs_genes, optimizer_genes, lr_genes
      
    def adaptative_transcription_phase_I_binary(self, population):
        """
        1)perturbe the binary gene by changing its binary value, then generate a random value from 0.0 to 1.0 if that value is lower 
        than the adaptativity factor then change the original gen for the perturbed gene. 
        2)adaptativity factor normally is set as 0.8 to 1.0 and it is reduced gradually in every generation.
        3)if the fitness of the perturbated chomosome is better than the original , then i will become our new chromosome.
        """
        for i in range(len(population)):
            original = population[i] 
            perturbed = np.array([(gene + random.choice([0, 1])) % 2 if random.random() <= self.adaptivity else gene for gene in original])
            if self.objective_function(perturbed) < self.objective_function(original):
                population[i] = perturbed
    
    def transcription_phase_II(self,population):
        for i in range(len(population) - 1):
            for j in range(i + 1, len(population)):
                if self.objective_function(population[i]) < self.objective_function(population[j]):
                    population[i], population[j] = population[j], population[i]

    def selection_and_recombination_binary(self, population, verbose=False):
        population = np.array(sorted(population, key= lambda ind: self.objective_function(ind)))
        new_population = np.array(population[:len(population)//2])
        while len(new_population) < len(population):
            indices = np.random.choice(len(new_population), 2, replace=False)
            parent1, parent2 = new_population[indices[0]], new_population[indices[1]]
            child = np.array([random.choice([p1,p2]) for p1,p2 in zip(parent1, parent2)])
            new_population = np.vstack([new_population, child])
        
        if verbose:
            np.set_printoptions(threshold=np.inf)
            print(f'\n{"-" * 100}\nNew population:\n{new_population}\n{"-" * 100}\n')
       
        return new_population
    
    def objective_function(self, individual):
        test_size, batch_size, d_model, dropout, nhead, nlayers, d_ff, epochs, optimizer, lr = self.hyperparameters(individual, verbose=True)

        def transformer_args():
            trans_parser = argparse.ArgumentParser()
            trans_parser.add_argument('--stratify', type=bool, default=False, help='Stratify the input data')
            trans_parser.add_argument('--windows', type=bool, default=True, help='Transform a feature in a time series data')
            trans_parser.add_argument('--test_size', type=float, default=test_size, help='Portion of data from the input dataset that is gonna be used for test')
            trans_parser.add_argument('--batch_size', type=int, default=batch_size, help='Number of samples per batch to load')
            trans_parser.add_argument('--d_model', type=int , default=d_model, help='Dimensions of the Transformer model' )
            trans_parser.add_argument('--max_len', type=int , default=14, help='Sequence len(Windows size for a time series model)' )
            trans_parser.add_argument('--dropout', type=float , default=dropout, help='Neural networts to be disregarded to prevent overfitting' )
            trans_parser.add_argument('--nhead', type=int, default=nhead, help='Number of attention heads in the Transformer model')
            trans_parser.add_argument('--nlayers', type=int, default=nlayers, help='Number of layers in the Transformer encoder')
            trans_parser.add_argument('--input_dim', type=int, default=1, help='Number of features in the input data')
            trans_parser.add_argument('--output_dim', type=int, default=1, help='Dimension of the output')
            trans_parser.add_argument('--LN_eps', type=float, default=1e-6, help='Epsilon value for Layer Normalization')
            trans_parser.add_argument('--d_ff', type=int, default=d_ff, help='Number of hidden units in the feed-forward layers')
            trans_parser.add_argument('--optimizer', type=str, default=optimizer, help='Optimizer choice for the model parameters optimization')
            trans_parser.add_argument('--epochs', type=int, default=epochs, help='Number of epochs to train the model')
            trans_parser.add_argument('--learning_rate', type=float, default=lr, help='Learning rate for the optimizer')
            trans_parser.add_argument('--plot_attn', type=bool, default=False, help='Plot the self attention of the model')
            trans_parser.add_argument('--early_stop', type=int, default=400, help='Stop the training in case of not improvement')
            
            trans_parser.add_argument('--save_best_model', type=str,
                        default='./Best model weights/Gemga/',
                        help='Path of the dir to save the best model fouded during training')
            
            trans_parser.add_argument('--stats_dir', type=str,
                        default='./Statistics results/Gemga_stats/',
                        help='Path of the dir to save the stats of training and validation')
            
            trans_args = trans_parser.parse_args([])
            return trans_args
        
        trans_args = transformer_args()
        model = ModelOps(trans_args)
        fitness_score = model.main()

        #free memory
        del model
        torch.cuda.empty_cache()
        gc.collect()

        return fitness_score
      
    def gemga_binary(self):
        population = self.initial_population(verbose=True)
        
        for generation in range(self.generations):
            print(f'\n{"-" * 100}\ngeneration: {generation}')
            self.adaptative_transcription_phase_I_binary(population)
            self.transcription_phase_II(population)
            population = self.selection_and_recombination_binary(population)
            self.adaptivity *= 0.99

            # Calculate the best loss per generation to plot.
            best_individual = min(population, key=lambda ind: self.objective_function(ind))
            best_loss = self.objective_function(best_individual)
            self.best_losses_per_generation.append(best_loss)
            self.plot_best_loss_per_generation()
        
        return best_individual
    
    def plot_best_loss_per_generation(self):
        stats_dir = './Statistics results/Gemga_stats/'
        if not os.path.exists(stats_dir):
            os.makedirs(stats_dir)
        
        generations = range(1, len(self.best_losses_per_generation) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(generations, self.best_losses_per_generation, marker='o', linestyle='-', color='b')
        plt.title('Best Loss per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.xticks(generations)
        
        # Save the plot in the statistics directory
        plot_path = os.path.join(stats_dir, 'Best_Loss_per_Generation.png')
        plt.savefig(plot_path)
        plt.close()
        
        print(f'\n{"="*115}\nPlot of best loss per generation saved at: {plot_path}\n{"="*115}\n')
   
if __name__ == '__main__':
    gemga_args = get_gemga_args()
    GGA = GemGa(gemga_args)

    start_time = time.time()
    best_individual = GGA.gemga_binary()
    best_hyperparameters = GGA.hyperparameters(best_individual, verbose=True)
    print(f"Best Hyperparameters: {best_hyperparameters}")
 
    run_time = time.time() - start_time
    print(f"Optimization run time: {run_time:.6f} seconds")

