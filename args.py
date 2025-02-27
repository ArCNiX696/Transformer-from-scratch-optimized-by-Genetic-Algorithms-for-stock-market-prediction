import argparse
import os 
#==============================================================================================#
#                                   TRANSFORMER MODEL ARGS                                     #
#==============================================================================================#
def get_args():
    parser = argparse.ArgumentParser()
    # ** Dataset loader args ** #
    parser.add_argument('--stratify', type=bool, default=False, help='Stratify the input data')
    parser.add_argument('--windows', type=bool, default=True, help='Transform a feature in a time series data')
    parser.add_argument('--test_size', type=float, default=0.4, help='Percentage of data from the dataset that is going to be used for training')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of samples per batch to load')

    # ** Paths for ---> Test dataset, json file , save and load best model, stats. ** #
    parser.add_argument('--predict_data_path', type=str,
                        default='./datasets/test/NVDA 2014-11-28 to 2024-12-31 processed.csv',
                        help='Load the input dataset for predictions')

    parser.add_argument('--json_path', type=str,
                        default='./datasets/test/',
                        help='Load the json file for predictions')

    parser.add_argument('--save_best_model', type=str,
                        default='./best_model_weights/transformer/',
                        help='Path of the dir to save the best model fouded during training')
    
    parser.add_argument('--load_best_model', type=str,
                        default=os.path.join("./best_model_weights/transformer", "NVDA c1 R2_0.9206_loss_0.0007_best_model_epoch_3712.pth"),
                        help='Load the best model for predictions')

    parser.add_argument('--train_stats_dir', type=str,
                        default='./stats/training/',
                        help='Path of the dir to save the stats of training and validation')
    
    parser.add_argument('--test_stats_dir', type=str,
                        default='./stats/test/transformer__GTs_vs_Preds.png',
                        help='Path of the dir to save the stats of training and validation')
    
    # ** Training args * * #
    parser.add_argument('--train', type=bool, default=False, help='Decide between train the model or make predictions')
    parser.add_argument('--early_stop', type=int, default=6000, help='Number of epochs max for early stoping in case the error does not improve')
    parser.add_argument('--plot_attn', type=bool, default=False, help='Plot the the a heapmap to show where the model is paying attention')
    parser.add_argument('--epochs', type=int, default=6000, help='Number of epochs to train the model')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--dropout', type=float , default=0.2, help='Neural networts to be disregarded to prevent overfitting' )

    # ** Model args ** #
    parser.add_argument('--d_model', type=int , default=512, help='Dimensions of the Transformer model' )
    parser.add_argument('--max_len', type=int , default=14, help='Sequence len(Windows size for a time series model)' )
    parser.add_argument('--nhead', type=int, default=16, help='Number of attention heads in the Transformer model')
    parser.add_argument('--nlayers', type=int, default=4, help='Number of layers in the Transformer encoder')
    parser.add_argument('--input_dim', type=int, default=1, help='Number of features in the input data')
    parser.add_argument('--output_dim', type=int, default=1, help='Dimension of the output')
    parser.add_argument('--LN_eps', type=float, default=1e-6, help='Epsilon value for Layer Normalization')
    parser.add_argument('--d_ff', type=int, default=128, help='Number of hidden units in the feed-forward layers')
    
    # ** Debug Options ** #
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
                        debug = 7 ---> Encoder Layer
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
    args = parser.parse_args([])
    return args 

