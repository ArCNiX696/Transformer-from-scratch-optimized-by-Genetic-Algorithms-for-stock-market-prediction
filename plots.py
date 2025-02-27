from typing import List, Dict, Optional, Union, Tuple, Any
import seaborn as sns
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib
matplotlib.use('Agg')
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import numpy as np
from datetime import datetime
import itertools
import gc
import os

#-------------------------Plot Gt vs Ensemble or only one random model predictions. --------------------------------------------------#
def plot_predictions(GT: Union[List[float], np.array, torch.Tensor],
                     prediction: Union[List[float], np.array, torch.Tensor],
                     model_name: str,
                     save_path: Optional[str] = None,
                     xlabel: str ="Days",
                     ylabel: str ="Normalized Price",
                     dates: Optional[Union[List[datetime], pd.DatetimeIndex]] = None) -> None:
    
    if isinstance(GT, torch.Tensor):
        GT = GT.cpu().numpy()

    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()

    plt.figure(figsize=(12, 8))

    if dates is not None:
        # Handle large number of dates by reducing the number of ticks
        interval = max(len(dates) // 10, 1)  # Show at most 10 date labels
        x_ticks = range(0, len(dates), interval)  # Select indices for ticks
        plt.plot(dates, GT, label='Ground Truth', linestyle='-', color='b')
        plt.plot(dates, prediction, label=f'{model_name} Predictions', linestyle='--', color='r')
        plt.xticks(ticks=[dates[i] for i in x_ticks], 
                   labels=[dates[i].strftime('%Y-%m-%d') for i in x_ticks],
                   rotation=45)
    else:
        plt.plot(GT, label='Ground Truth', linestyle='-', color='b')
        plt.plot(prediction, label=f'{model_name} Predictions', linestyle='--', color='r')

    plt.title(f'Ground Truth vs {model_name} Predictions')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path)

    plt.show()

#------------------------------------------ Plot Gt vs Up to 8 Models. --------------------------------------------------#
def plot_GT_vs_models(GT: Union[List[float], np.array, torch.Tensor],
                      model_pred_1: Union[List[float], np.array, torch.Tensor] = None,
                      model_pred_2: Union[List[float], np.array, torch.Tensor] = None,
                      model_pred_3: Union[List[float], np.array, torch.Tensor] = None,
                      model_pred_4: Union[List[float], np.array, torch.Tensor] = None,
                      model_pred_5: Union[List[float], np.array, torch.Tensor] = None,
                      model_pred_6: Union[List[float], np.array, torch.Tensor] = None,
                      model_pred_7: Union[List[float], np.array, torch.Tensor] = None,
                      model_pred_8: Union[List[float], np.array, torch.Tensor] = None,
                      model_name_1: str = None,
                      model_name_2: str = None,
                      model_name_3: str = None,
                      model_name_4: str = None,
                      model_name_5: str = None,
                      model_name_6: str = None,
                      model_name_7: str = None,
                      model_name_8: str = None,
                      save_path: Optional[str] = None,
                      xlabel: str = "Days",
                      ylabel: str = "Normalized Price",
                      dates: Optional[Union[List[datetime], pd.DatetimeIndex]] = None) -> None:
    
    # Convert tensors to numpy if necessary
    if isinstance(GT, torch.Tensor):
        GT = GT.cpu().numpy()
    if isinstance(model_pred_1, torch.Tensor):
        model_pred_1 = model_pred_1.cpu().numpy()
    if isinstance(model_pred_2, torch.Tensor):
        model_pred_2 = model_pred_2.cpu().numpy()
    if isinstance(model_pred_3, torch.Tensor):
        model_pred_3 = model_pred_3.cpu().numpy()
    if isinstance(model_pred_4, torch.Tensor):
        model_pred_4 = model_pred_4.cpu().numpy()
    if isinstance(model_pred_5, torch.Tensor):
        model_pred_5 = model_pred_5.cpu().numpy()
    if isinstance(model_pred_6, torch.Tensor):
        model_pred_6 = model_pred_6.cpu().numpy()
    if isinstance(model_pred_7, torch.Tensor):
        model_pred_7 = model_pred_7.cpu().numpy()
    if isinstance(model_pred_8, torch.Tensor):
        model_pred_8 = model_pred_8.cpu().numpy()
        
    # Create the plot
    plt.figure(figsize=(16, 10))

    # Handle dates if provided
    if dates is not None:
        # Validate that dates match the length of GT
        if len(dates) != len(GT):
            raise SystemExit(f"""\n{'=' * 80}\nError: Execution stopped intentionally 
                             because The length of 'dates': {len(dates)}, must match the length of 'GT': {len(GT)}.\n{'=' * 80}\n""")
        
        # Handle large number of dates by reducing the number of ticks
        interval = max(len(dates) // 10, 1)  # Show at most 10 date labels
        x_ticks = range(0, len(dates), interval)  # Select indices for ticks
        
        plt.plot(dates, GT, label='Ground Truth', color='black')

        if model_pred_1 is not None and len(model_pred_1) > 0:
            plt.plot(dates, model_pred_1, label=f'{model_name_1} Predictions', linestyle='--', color='red')
        if model_pred_2 is not None and len(model_pred_2) > 0:
            plt.plot(dates, model_pred_2, label=f'{model_name_2} Predictions', linestyle='--', color='orange')
        if model_pred_3 is not None and len(model_pred_3) > 0:
            plt.plot(dates, model_pred_3, label=f'{model_name_3} Predictions', linestyle='--', color='brown')
        if model_pred_4 is not None and len(model_pred_4) > 0:
            plt.plot(dates, model_pred_4, label=f'{model_name_4} Predictions', linestyle='--', color='lime')
        if model_pred_5 is not None and len(model_pred_5) > 0:
            plt.plot(dates, model_pred_5, label=f'{model_name_5} Predictions', linestyle='--', color='dimgray')
        if model_pred_6 is not None and len(model_pred_6) > 0:
            plt.plot(dates, model_pred_6, label=f'{model_name_6} Predictions', linestyle='--', color='blue')
        if model_pred_7 is not None and len(model_pred_7) > 0:
            plt.plot(dates, model_pred_7, label=f'{model_name_7} Predictions', linestyle='--', color='royalblue')
        if model_pred_8 is not None and len(model_pred_8) > 0:
            plt.plot(dates, model_pred_8, label=f'{model_name_8} Predictions', linestyle='--', color='magenta')

        # Set X-axis ticks
        plt.xticks(ticks=[dates[i] for i in x_ticks], 
                   labels=[dates[i].strftime('%Y-%m-%d') for i in x_ticks], 
                   rotation=45)
    else:
        # Plot without dates
        plt.plot(GT, label='Ground Truth', color='black')

        if model_pred_1 is not None and len(model_pred_1) > 0:
            plt.plot(model_pred_1, label=f'{model_name_1} Predictions', linestyle='--', color='red')
        if model_pred_2 is not None and len(model_pred_2) > 0:
            plt.plot(model_pred_2, label=f'{model_name_2} Predictions', linestyle='--', color='orange')
        if model_pred_3 is not None and len(model_pred_3) > 0:
            plt.plot(model_pred_3, label=f'{model_name_3} Predictions', linestyle='--', color='brown')
        if model_pred_4 is not None and len(model_pred_4) > 0:
            plt.plot(model_pred_4, label=f'{model_name_4} Predictions', linestyle='--', color='lime')
        if model_pred_5 is not None and len(model_pred_5) > 0:
            plt.plot(model_pred_5, label=f'{model_name_5} Predictions', linestyle='--', color='dimgray')
        if model_pred_6 is not None and len(model_pred_6) > 0:
            plt.plot(model_pred_6, label=f'{model_name_6} Predictions', linestyle='--', color='blue')
        if model_pred_7 is not None and len(model_pred_7) > 0:
            plt.plot(model_pred_7, label=f'{model_name_7} Predictions', linestyle='--', color='royalblue')
        if model_pred_8 is not None and len(model_pred_8) > 0:
            plt.plot(model_pred_8, label=f'{model_name_8} Predictions', linestyle='--', color='magenta')

    # Add labels, title, and legend
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'Ground Truth vs Models Predictions')
    plt.legend()
    plt.grid()

    # Save the plot if a path is provided
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path)

    plt.show()

#------------------------------------------ Plot a Barplot of Models Performance metrics MSE, MAE, R2. --------------------------------------------------#
def metrics_barplot(metrics_df: pd.DataFrame,
                    save_path: Optional[str] = None) -> None:
    
    metrics_df.set_index('Model', inplace=True)
    
    #MSE Barplot.
    metrics_df[['MSE']].plot(kind='bar', figsize=(10, 5), color='green')
    plt.title('MSE Metric Comparison')
    plt.ylabel('Error')
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.xticks(rotation=360)
    plt.show()

    #MAE Barplot.
    metrics_df[['MAE']].plot(kind='bar', figsize=(10, 5), color='blue')
    plt.title('MAE Metric Comparison')
    plt.ylabel('Error')
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.xticks(rotation=360)
    plt.show()

    # R2 Barplot
    metrics_df[['R2']].plot(kind='bar', figsize=(10, 5), color='orange')
    plt.title('R2 Score Comparison')
    plt.ylabel('R2 Score')
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.xticks(rotation=360)

    # Save the plot if a path is provided
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path)

    plt.show()

#------------- Print and Plot the confusion matrix, print a Classification Report and Calculate metrics. -------------------------#
def plot_confusion_matrix(cm: np.ndarray, 
                          classes: List[str],
                          GT_pos_neg: List[int],
                          model_pos_neg: List[int],
                          model_name: str, 
                          cmap: Optional[plt.cm.ScalarMappable] = plt.cm.Blues,
                          save_path: Optional[str] = None) -> None:
    
    #Print the cm for visualization.
    print(f"""\n{'=' * 80}\n
           {model_name} Confusion Matrix:\n\n{cm}\n""")
    
    #Print a Classification Report.
    print(f"\n{model_name} Classification Report:\n")
    print(classification_report(GT_pos_neg, model_pos_neg, labels=[1, -1]))
   
    # Calculate metrics (precision, recall, F1, accuracy) and print them.
    accuracy = np.trace(cm) / float(np.sum(cm))
    precision = cm[0, 0] / float(cm[0, 0] + cm[1, 0])
    recall = cm[0, 0] / float(cm[0, 0] + cm[0, 1])
    f1 = 2 * (precision * recall) / (precision + recall)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"\n\n{'=' * 80}\n")

    #Plot the cm.
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(f'{model_name} Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Pintar los números en la matriz
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Save the plot if a path is provided
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path)

    plt.show()
#------------------------------------------------- Plot AUC-ROC. ---------------------------------------------#
def plot_AUC_ROC(GT_pos_neg: List[int],
                 models_pos_neg: Dict[str, List[float]]):
    
    plt.figure(figsize=(10, 8))

    for model_name, pos_neg in models_pos_neg.items():
        # AUC-ROC calculation for each model
        fpr, tpr, _ = roc_curve(GT_pos_neg, pos_neg, pos_label=1)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve for the model
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

    # Plot baseline
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Baseline')

    # Configure the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) for Multiple Models')
    plt.legend(loc='lower right')
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.show()

#------------------------------------------------- Plot Tran and Validation results. ---------------------------------------------#
def plot_evaluation(training_losses: float,
                    validation_losses: float,
                    graphics_dir: str,
                    file_name: str,
                    model_name: str,
                    plot: bool = False):
    
    if not os.path.exists(graphics_dir):
        os.makedirs(graphics_dir)
    # print(f'training losses:{len(training_losses)}')
    # print(f'validation losses:{len(validation_losses)}')
    # exit()
    epochs = range(1, len(training_losses) + 1)
    best_val_loss = validation_losses[len(validation_losses) - 1]

    plt.figure(figsize=(8,8))

    plt.subplot(2,1,1)
    plt.plot(epochs, training_losses , label= f'{model_name} Training losses',color='green',marker='o')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    plt.subplot(2,1,2) #row , column , number of plot in the figure
    plt.plot(epochs, validation_losses , label= f'{model_name} Validation losses',color='darkred',marker='x')
    plt.title('Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(graphics_dir,f'{file_name}, best val loss {best_val_loss:.6f}.png')
    plt.savefig(plot_path)
    plt.close()
    gc.collect() 
    
    if plot == True:
        plt.show()

#------------------------------------------------- Plot Direction preds. ---------------------------------------------#
def plot_GT_vs_models_directions(cumulative_gt: List[int], 
                                 cumulative_models: Dict[str, List[int]], 
                                 dates: List[str],
                                 save_path: Optional[str] = None) -> None:
    
    """
    Plot cumulative directional changes for GT and multiple models.

    Parameters:
        cumulative_gt: Dictionary containing the cumulative directions of the GT.
        cumulative_models: Dictionary where keys are model names and values are their cumulative directions.
        dates: List of dates corresponding to the data points.
        save_path: Optional path to save the plot as an image file.
    """
    
    plt.figure(figsize=(12, 8))
    max_points = 100  
    step = max(1, len(dates) // max_points)  

    # Filter The GT data.
    filtered_dates = dates[::step]
    filtered_values_gt = cumulative_gt[::step]

    plt.plot(filtered_dates, filtered_values_gt, label='GT Cumulative', color='black', marker='o', markersize=5, linewidth=2)

    # Assign unique colors for each model
    colors = plt.cm.get_cmap('tab10', len(cumulative_models))  # Use a colormap with enough distinct colors
    
    for idx, (model_name, cumulative_directions) in enumerate(cumulative_models.items()):
        filtered_values_model = cumulative_directions[::step]
        plt.plot(filtered_dates,
                 filtered_values_model,label = model_name,
                 color= colors(idx),
                 marker='x',
                 markersize=5,
                 linestyle='--')
    
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative Directions', fontsize=14)
    plt.title('GT Directions vs Predictions Directions')
    plt.legend()
    plt.grid(True)
    plt.xticks(filtered_dates, rotation=45)
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path)
    plt.show()
    
#------------------------------------------------- Pairplot to visualize active experts. ---------------------------------------------#
def plot_active_experts(expert_contribution: List[Dict[str, List[int]]],
                        model_names: List[str],
                        title: str = "Active Experts in MoE by date",
                        save_path: Optional[str] = None) -> None:
    """
    Creates a pairplot-like visualization where the y-axis represents dates, the x-axis represents active models,
    and the points correspond to the models active on each date. The dates are shown in descending order, and
    the x-axis respects the order of the model names provided.

    Parameters:
        expert_contribution (list): List of dictionaries containing active experts per date.
        model_names (list): List of model names corresponding to the indices in `expert_contribution`.
    """
    data = []
    for entry in expert_contribution:
        date = entry['Date']
        active_experts = entry['Active Experts']
        for expert in active_experts:
            model_name = model_names[expert -1]
            data.append({'Date': date, 'Active Expert': model_name})

    df = pd.DataFrame(data)

    # Ensure the dates are sorted in descending order
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date', ascending=False)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    # Ensure the 'Active Model' column respects the order of `model_names`
    df['Active Expert'] = pd.Categorical(df['Active Expert'], categories=model_names, ordered=True)
    
    # Create the pairplot-like visualization using scatterplot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Active Expert', y='Date', hue='Active Expert', style='Active Expert', s=100,
                    palette='tab10')

    # Customize the plot
    plt.title(f'{title}', fontsize=16)
    plt.xlabel('Active Experts', fontsize=12)
    plt.ylabel('Dates', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title='Experts', fontsize=10)
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Save the plot if a path is provided
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path)

    plt.show()

#------------------------------------------------- Not commonly used funcs. ---------------------------------------------#

def plot_GT_vs_IndMod_vs_Trasn(dates,
                               GT_cpu,
                               individual_predictions_cpu,
                               ensemble_predictions_cpu,
                               GT_directions,
                               ensemble_directions):
        
    #Plot GT vs individual transformers vs emsemble model.
        plt.figure(figsize=(16, 10))
        plt.plot(dates, GT_cpu, label='Ground Truth', color='black')
        for idx, predictions in enumerate(individual_predictions_cpu):
            plt.plot(dates, predictions[:-1], label=f'Model {idx+1}')
        plt.plot(dates, ensemble_predictions_cpu, label='Ensemble', linestyle='--', color='red')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Model Predictions vs Ground Truth')
        plt.legend()
        plt.show()

        #Plot GT vs emsemble model.
        plt.figure(figsize=(16, 10))
        plt.plot(dates, GT_cpu, label='Ground Truth', color='black')
        plt.plot(dates, ensemble_predictions_cpu, label='Ensemble', linestyle='--', color='red')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Ensemble Model Predictions vs Ground Truth')
        plt.legend()
        plt.show()

        #Plot directions.
        plt.figure(figsize=(16, 10))
        plt.plot(dates, GT_directions, label='GT Directions', color='blue')
        plt.plot(dates, ensemble_directions, label='Ensemble Directions', color='red', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Direction')
        plt.title('GT vs Ensemble Predicted Directions')
        plt.legend()
        plt.show()

def plot_corr_matrix(corr_matrix):
    plt.figure(figsize=(8,8))
    sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',fmt='.4f') #annot=show corr values ,cmap= colors, fmt=decimals 
    plt.title('Correlation Matrix')
    plt.show()

def boxplot(df, target_variable=str):
    # Create a figure and axes for the plot
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    # Create the boxplot for each column with respect to the target variable
    sns.boxplot(data=df.drop(columns=target_variable), ax=ax)
    sns.swarmplot(data=df.drop(columns=target_variable), color=".25", ax=ax)
    
    # Set the title and axes labels
    ax.set_title('Boxplot of distribution with respect to ' + target_variable)
    ax.set_ylabel('Distributon')
    
    # Rotate x-axis labels for better visualization
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_corr_scatter(df):
    sns.set_theme(style='whitegrid')
    sns.pairplot(df,height=1.6)
    plt.show()

def plot_datetime(df,column=str,column2=str):
    df= df[[column,column2]]

    df[column]= pd.to_datetime(df[column])

    plt.title(f'{column2} with respect to {column}')
    plt.xlabel(column)
    plt.ylabel(column2)
    plt.plot(df[column] , df[column2])

def plot_GT_vs_pred(GT: Dict[str, float],
                    pred: Dict[str, float],
                    stock_name: str,
                    model_name : str,
                    save_path: Optional[str] = None,
                    plot1_label: str = 'Actual Prices',
                    plot2_label: str = 'Predictions',
                    ) -> None:
    
    dates_real = list(GT.keys())
    prices_real = list(GT.values())

    dates_pred = list(pred.keys())
    prices_pred = list(pred.values())
    
    plt.figure(figsize=(15, 7))
    plt.plot(dates_real, prices_real, label=plot1_label, marker='.', color='blue', markersize=5)
    plt.plot(dates_pred, prices_pred, label=plot2_label, marker='.', color='red', markersize=5)
    plt.xticks(rotation=45, ha="right")
    plt.title(f'{stock_name} Stock Prices: Actual vs {model_name} {plot2_label}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    max_labels = 10  
    if len(dates_real) > max_labels:
        step = len(dates_real) // max_labels
        plt.xticks(dates_real[::step], rotation=45, ha="right")
    else:
        plt.xticks(dates_real, rotation=45, ha="right")

    plt.tight_layout()
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path)
    plt.show()
            
def plot_GT_vs_pred_direction(cumulative_gt: Dict[str, int], 
                              cumulative_predictions: Dict[str, int], 
                              save_path: Optional[str] = None) -> None:
    plt.figure(figsize=(15, 7))
    max_points = 100  
    step = max(1, len(cumulative_gt) // max_points)  
    
    filtered_dates_gt = list(cumulative_gt.keys())[::step]
    filtered_values_gt = list(cumulative_gt.values())[::step]
    
    filtered_dates_pred = list(cumulative_predictions.keys())[::step]
    filtered_values_pred = list(cumulative_predictions.values())[::step]
    
    plt.plot(filtered_dates_gt, filtered_values_gt, label='GT Cumulative', marker='o', markersize=5)
    plt.plot(filtered_dates_pred, filtered_values_pred, label='Predictions Cumulative', marker='x', markersize=5)
    
    plt.xlabel('Date')
    plt.ylabel('Cumulative Value')
    plt.title('Cumulative GT vs Cumulative Predictions')
    plt.legend()
    plt.grid(True)
    plt.xticks(filtered_dates_gt, rotation=45)
    plt.tight_layout()
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path)
    plt.show()
    
def plot_attention(attn_weights, keys, queries, model:int, enc_or_dec:str, layer_idx: int):
    key_entries = [f't-{key}' if key != 1 else 'Adj close' for key in range(keys.size(2), 0, -1)]
    query_entries = [f't-{query}' if query != 1 else 'Adj close' for query in range(queries.size(2), 0, -1)]

    num_heads = attn_weights.size(1)

    for head_idx in range(num_heads):
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attn_weights[0, head_idx].cpu().detach().numpy(),
            xticklabels=key_entries,
            yticklabels=query_entries,
            cmap='viridis',
            cbar_kws={'label': 'Attention Score'},
            square=True,  # Asegurar celdas cuadradas
            linewidths=1,  # Líneas de grid más visibles
            linecolor='black'  # Color del grid
        )
        plt.title(f"Attention Weights for Head {head_idx + 1}")
        plt.xlabel("Key Sequence")
        plt.ylabel("Query Sequence")

        plot_path = os.path.join(
            f'./Statistics results/My Trans statistics/Time series models/model {model}/Head Attention/',
            f'{enc_or_dec}_layer_{layer_idx + 1}_head_{head_idx + 1}.png'
        )
        plt.savefig(plot_path)
        plt.close()
        gc.collect()

if __name__ == '__main__':
    pass
