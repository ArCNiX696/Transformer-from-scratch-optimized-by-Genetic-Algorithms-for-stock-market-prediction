# ** My modules ** #
from prepare_data import DatasetLoader
import plots as pl
from args import get_args
args = get_args()
from utils import inverse_transform

# ** Torch ** #
import torch
import torch.nn as nn
from torchmetrics import R2Score, MeanAbsolutePercentageError

# ** Third-Party Libraries ** #
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchmetrics import R2Score
from tqdm import tqdm
import json
import gc
import logging
logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO,
    format=f'{"-" * 100}\n%(asctime)s - %(name)s - Porpuse: %(levelname)s - %(message)s\n',
    handlers=[
        logging.StreamHandler() 
    ]
)
logger = logging.getLogger(__name__)

# ** Python Standard Libraries ** #
from copy import deepcopy as dc
import os
from datetime import datetime, timedelta
import math

ticker = 'NVDA up2'
device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"\nUsing CUDA device: {torch.cuda.get_device_name(device)}\n")
else:
    print("\nUsing CPU\n")
    
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
        
        if args.debug == 1:
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
        
        if args.debug == 2:
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

        if args.plot_attn:
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

        if args.debug == 3:
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
        
        if args.debug == 4:
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
        if args.debug == 5:  # Debugging step before forward pass
            logger.debug(f'\nInput Shape before FeedForward:\n{x.shape}\nx: {x}\n\n')
        
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        if args.debug == 5:  # Debugging step after forward pass
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
        
        if args.debug == 6: 
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
        
        if args.debug == 7:
            logger.debug(f'\nEncoder layer forward input:\nx Shape:\n{x.shape}\nx:\n{x}\n\n')

        x = x.to(device)
        assert x.device == device, f'\n\nWARNING: x Tensors in the Encoder layer are not on device {device}\n\n'
         
        if mask is not None:
            mask = mask.to(device)
        x = self.residual1(x, lambda x: self.multi_head(x, x, x, mask))
        x = self.residual2(x, self.feed_forward)
        
        if args.debug == 7:
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
        
        if args.debug == 8:
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
        
        if args.debug == 8:
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
        
        if args.debug == 9:
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
        
        if args.debug == 9:
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
        
        if args.debug == 10:
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
        
        if args.debug == 10:
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
        
        if args.debug == 11:
            logger.debug(f'\nTransformer Model src input:\nsrc Shape:\n{src.shape}\nsrc:\n{src}\n\n')

        src = src.to(device)
        assert src.device == device, f'\n\nWARNING: src Tensors in the Transformer Model are not on device {device}\n\n'
        
        tgt = tgt.to(device)
        assert tgt.device == device, f'\n\nWARNING: tgt Tensors in the Transformer Model are not on device {device}\n\n'
        
        if src_mask is not None:
            src_mask = src_mask.to(device)

        tgt_mask = self.create_look_ahead_mask(tgt.size(1)).to(device)
        
        if args.debug == 11:
            if src_mask is not None:
                logger.debug(f'\nTransformer Model src_mask shape:\n{src_mask.shape}\nsrc_mask:\n{src_mask}\n\n')
            else:
                logger.debug(f'\nTransformer Model src_mask:\n{src_mask}\n\n')
            logger.debug(f'\nTransformer Model tgt_mask shape:\n{tgt_mask.shape}\ntgt_mask:\n{tgt_mask}\n\n')
        
        encoder_output = self.encoder(src, src_mask)
        
        if args.debug == 11:
            logger.debug(f'\nTransformer Model encoder_output shape:\n{encoder_output.shape}\nTransformer Model encoder_output:\n{encoder_output}\n\n')
          
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        if args.debug == 11:
            logger.debug(f'\nTransformer Model decoder_output shape:\n{decoder_output.shape}\nTransformer Model decoder_output:\n{decoder_output}\n\n')
          
        output = self.fc_out(decoder_output)
        if args.debug == 11:
            logger.debug(f'\nTransformer Model Output shape:\n{output.shape}\nTransformer Model Output:\n{output}\n\n{"-" * 100}\n')
            raise SystemExit(f'\nDEBUG MODE: Execution stopped intentionally.\n')
        
        return output
#==============================================================================================#
#                                TRAINING, VALIDATION AND PREDICTION                            #
#==============================================================================================#
class ModelOps: #Model Operations
    def __init__(self,
                 args: Any):
        self.MSE = nn.MSELoss()
        self.MAE = nn.L1Loss()
        self.MAPE = MeanAbsolutePercentageError().to(device)
        self.R2_train = R2Score().to(device) 
        self.R2_validation = R2Score().to(device)
        self.R2_prediction = R2Score().to(device)
        self.R2_simulation = R2Score().to(device)
        self.model = TransformerModel(args=args).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.best_loss = float('inf')
        self.training_losses = []
        self.validation_losses = []
        self.save_best_model = args.save_best_model
        self.train_stats_dir = args.train_stats_dir
        self.early_stop_counter = 0
                                              #For Prediction
        self.predict_data_path = args.predict_data_path
        self.test_stats_dir = args.test_stats_dir
        self.d_model = args.d_model

#------------------------------------------------ Training --------------------------------------------------#
    def train(self):
        self.train_loader, self.val_loader = DatasetLoader.prepare_data(args.batch_size, args.test_size)
        for self.epoch in range(args.epochs):
            
            if self.early_stop_counter >= args.early_stop:
                print(f'Early stopping in epoch: {self.epoch + 1}')
                break
            
            self.model.train()
            total_loss = 0.0

            for X_batch, y_batch in tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1}/{args.epochs}'):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                src = X_batch[:, :args.max_len - 1, :]
                tgt = X_batch[:, args.max_len - 1:, :] 
                self.optimizer.zero_grad()
                output = self.model(src, tgt)
                output = output[:, -1, :]
                output = output.view(-1)
                y_batch = y_batch.view(-1)
                
                if args.debug == 12:
                    src = X_batch.view(-1) 
                    logger.debug(f"""\nsrc.shape: {src.shape}\nsrc:\n{src}\n\n
                                tgt.shape: {y_batch.shape}\ntgt:\n{y_batch}\n\n
                                Output.shape: {output.shape}\nOutput:\n{output}\n\n{"-" * 100}\n\n""")
                    raise SystemExit(f'\nDEBUG MODE: Execution stopped intentionally.\n')
                
                if args.debug == 14:#Visualize Gt vs Predictions
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
                src = X_batch[:, :args.max_len - 1, :]
                tgt = X_batch[:, args.max_len - 1:, :]
                output = self.model(src, tgt)
                output = output[:, :, :]
                output = output.view(-1)
                y_batch = y_batch.view(-1)
                
                if args.debug == 15:
                    src = X_batch.view(-1) 
                    logger.debug(f"""\nsrc.shape: {src.shape}\nsrc:\n{src}\n\n
                                tgt.shape: {y_batch.shape}\ntgt:\n{y_batch}\n\n
                                Output.shape:\n{output.shape}\nOutput:\n{output}\n\n{"-" * 100}\n\n""")
                    raise SystemExit(f'\nDEBUG MODE: Execution stopped intentionally.\n')
                
                if args.debug == 16:#Visualize Gt vs Predictions
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
        
        if avg_loss < self.best_loss and R2_val_score >= 0.9:
            print(f'New best model found in epoch: {self.epoch + 1}')
            self.best_loss = avg_loss
            file_name = f'{ticker} R2_{R2_val_score:.4f}_loss_{self.best_loss:.4f}_best_model_epoch_{self.epoch + 1}.pth'
            torch.save(self.model.state_dict(), os.path.join(self.save_best_model, file_name))
        
            log_name = f'{ticker} R2_{R2_val_score:.4f}_loss_{self.best_loss:.4f}_best_model_epoch_{self.epoch + 1}.txt'
            with open(os.path.join(self.save_best_model, log_name), 'a') as log_file:
                log_file.write(f'best_model_epoch: {self.epoch + 1}, MSE: {self.best_loss:.4f}, R2: {R2_val_score:.4f},MAE: {MAE_loss},MAPE: {MAPE_loss}, opt: Adam\n')
                log_file.write(f'Hyperparameters: \n')
                log_file.write(f'  - test_size: {args.test_size}\n')
                log_file.write(f'  - batch_size: {args.batch_size}\n')
                log_file.write(f'  - d_model: {args.d_model}\n')
                log_file.write(f'  - dropout: {args.dropout}\n')
                log_file.write(f'  - nhead: {args.nhead}\n')
                log_file.write(f'  - nlayers: {args.nlayers}\n')
                log_file.write(f'  - d_ff: {args.d_ff}\n')
                log_file.write(f'  - epochs: {args.epochs}\n')
                log_file.write(f'  - learning_rate: {args.learning_rate}\n')
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1  

#------------------------------------------------ Prediction --------------------------------------------------#
    def predict(self,
                predict_data_path: str,
                load_best_model: str = args.load_best_model,
                save: bool = False,
                verbose: bool = False,
                plot: bool = False):
        try:
            self.model.load_state_dict(torch.load(load_best_model, weights_only=True))
        except FileExistsError:
            print(f'\nError T1: Could not find the best model file {load_best_model}\n')
            return
        except Exception as e:
            print(f'\nError T1: Could not load the best model: {e}\n')
            return

        self.model.to(device)
        self.model.eval()

        try:
            input_data = pd.read_csv(predict_data_path,
                                     index_col = 'Date')
        except FileNotFoundError:
            print(f"\nError: The dataframe at path '{input_data}' does not exist.\n")
            return None
        
        #Keep the dates for plotting
        input_data.index = pd.to_datetime(input_data.index, errors='coerce')#errors='coerce replace invalid vals in Nat vals
        assert not input_data.index.isnull().any(), f"\nWarning: Some dates in index 'Date' could not be converted to datetime format and will be dropped.\n"
        dates = input_data.index

        input_data = input_data.select_dtypes(include=[np.number]).values# Ensure all columns are numeric and exclude the date column if it exists
        
        GT = input_data[:,-1].flatten()#GTs are in the last col of input dataset.
        GT_last = GT.copy()[-1].reshape(-1)#So we need the last val in that col to create a new sequence for one day ahead pred.
        GT = torch.tensor(GT, dtype=torch.float32).to(device)

        input_data = input_data[:, :-1]#The input data for prediction are all the cols but not the last GT col.
        #Copy the last input sequence, then add the GT last to create the new seq. for one day ahead pred.
        data_last_seq = input_data.copy()[-1,1:]  
        last_day_input = np.concatenate((data_last_seq, GT_last), axis=0).reshape(1,-1)
        input_data = np.concatenate((input_data, last_day_input), axis=0)

        input_data = input_data.reshape((-1, input_data.shape[1], 1))# Reshape the data to match the training format (batch_size/rows, sequence_length/cols, 1)
        input_data = torch.tensor(input_data, dtype=torch.float32).to(device)
        
        if input_data.shape[-1] != args.input_dim:
            raise ValueError(f"Expected input features size: {args.input_dim}, but got: {input_data.shape[-1]}")

        with torch.no_grad():
            if args.plot_attn:
                src = input_data[:, :, :]
                tgt = input_data[:, :, :]
            else:
                src = input_data[:, :args.max_len - 1, :]
                tgt = input_data[:, args.max_len - 1:, :]
          
            if args.debug == 17:
                logger.debug(f"""\n\nInput_data shape: {input_data.shape}\n\nInput data x:\n{input_data.reshape(-1)}\n\n
                             src.shape: {src.shape}\n\nsrc:\n{src.reshape(-1)}\n\n
                             tgt.shape: {tgt.shape}\n\ntgt:\n{tgt.reshape(-1)}\n{"-" * 100}\n\n""")
                raise SystemExit(f'\nDEBUG MODE: Execution stopped intentionally.\n')
            
            predictions = self.model(src, tgt).view(-1)
            predictions = predictions.clone().detach().to(device)# detach the gradients info for prediction.
           
        #Extract vals of max and min from jason file.
        pred_dataset_name = os.path.splitext(os.path.basename(predict_data_path))[0]#extract the name of the pred dataset to compare.
        json_file_path = os.path.join(args.json_path, f'{pred_dataset_name}.json')
        
        #Verify that there is a json file with the same name as the csv pred dataset name.
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as json_file:
                normalization_data = json.load(json_file)
                Adj_Close_data = list(normalization_data.keys())[0]
                min_ = normalization_data[Adj_Close_data]['min']
                max_ = normalization_data[Adj_Close_data]['max']
                
                print(f""""\n{'=' * 80}\nValues of max and min for inverse transform are:\n
                      Min: {min_}\n
                      Max : {max_}\n\n{'=' * 80}\n\n""")
        else:
            raise SystemExit(f'\nError: Execution stopped intentionally because json file does not exist or its name is not as same as csv pred dataset.\n')
          
        prediction_real_vals = inverse_transform(predictions, min_, max_)
        Gt_real_vals = inverse_transform(GT, min_, max_)

        last_day_pred_norm = predictions[-1]
        last_day_pred_real = prediction_real_vals[-1]

        #Calculate model performance metrics but not for the last prediction, to match GT tensor shape.
        MSE = self.MSE(predictions[:-1], GT)
        MAE = self.MAE(predictions[:-1], GT)
        self.R2_prediction.update(predictions[:-1], GT)
        R2_pre_score = self.R2_prediction.compute().item() 
        
        if verbose: 
            print(f"""\n{"-" * 100}\n
                            GT vs Predictions:\n\n
                GT shape: {GT.shape}\n\n
                GT normalized: {GT}\n\n
                Prediction shape: {predictions.shape}\n\n
                Predictions normalized Tensor: {predictions}\n\n
                GT real values: {Gt_real_vals}\n\n
                Predictions real values: {prediction_real_vals}\n'
                                Date: {dates[-1] + pd.Timedelta(days=1)}\n
                Last day prediction normalized : {last_day_pred_norm}\n
                Last day prediction real value : {last_day_pred_real}\n\n
                These metrics are caluculated based on the normalized values:\n\n 
                MSE Error: {MSE.item():.6f}\n
                MAE Error: {MAE.item():.6f}\n
                R2 score: {R2_pre_score:.6f}\n
                \n{"-" * 100}\n\n""")
                
        if save:
            log_name = f'My Transformer Predictions vs GT model_log.txt'
            with open(os.path.join(self.save_best_model, log_name), "a") as log_file:
                log_file.write(f"""\n{"-" * 100}\n
                                GT vs Predictions:\n\n
                    GT shape: {GT.shape}\n\n
                    GT normalized: {GT}\n\n
                    Prediction shape: {predictions.shape}\n\n
                    Predictions normalized Tensor: {predictions}\n\n
                    GT real values: {Gt_real_vals}\n\n
                    Predictions real values: {prediction_real_vals}\n'
                                Date: {dates[-1] + pd.Timedelta(days=1)}\n
                    Last day prediction normalized : {last_day_pred_norm}\n
                    Last day prediction real value : {last_day_pred_real}\n\n
                    These metrics are calculated based on the normalized values:\n\n
                    MSE Error: {MSE.item():.6f}\n
                    MAE Error: {MAE.item():.6f}\n
                    R2 score: {R2_pre_score:.6f}\n
                    \n{"-" * 100}\n\n""")

        self.R2_prediction.reset()

        if plot:
            Gt_real_vals_cpu = Gt_real_vals.clone().cpu()
            prediction_real_vals_cpu = prediction_real_vals.clone().cpu()
            GT_dic = dict(zip(dates, Gt_real_vals_cpu))
            prediction_real_vals_dic = dict(zip(dates, prediction_real_vals_cpu))

            pl.plot_GT_vs_pred(GT_dic,
                               prediction_real_vals_dic,
                               stock_name='NVDA',
                               model_name='Transformer',
                               save_path = self.test_stats_dir,
                               plot2_label='Transformer predictions')
            
        return GT, Gt_real_vals, predictions, prediction_real_vals, dates, MSE, MAE, R2_pre_score
#==============================================================================================#
#                                               MAIN                                           #
#==============================================================================================#
    def main(self):
        if args.train:
            self.train()
            pl.plot_evaluation(self.training_losses,
                               self.validation_losses,
                               self.train_stats_dir,
                               f'{ticker}training_val_loss_{args.epochs} epochs',
                               model_name='Transformer',
                               plot=True)
        else:
            self.predict(predict_data_path = args.predict_data_path,
                         save = False,
                         verbose = True,
                         plot = True)

if __name__ == '__main__':
    TrainModel = ModelOps(args)
    TrainModel.main()