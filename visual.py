import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import pearsonr

# def plot_heatmap(x: np.ndarray, y: np.ndarray, savefig='heatmap.png'):
#     # x, y = x.detach().cpu().numpy(), y.detach().cpu().numpy()
#     corr_matrix = np.corrcoef(x, y)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True,
#                 cbar_kws={"shrink": .8}, fmt=".2f")

#     plt.title('Correlation Heatmap Between X and Y')
#     plt.xlabel('edge weight prediction')
#     plt.ylabel('edge weight ground truth')
#     plt.savefig(savefig)
#     plt.show()
#     plt.close()

def plot_pearson_correlation_scatter_arr(array_x: np.array, array_y: np.array, upper: float = None, savefig = 'pearsons.png', xlabel = 'Prediction', ylabel = 'Ground Truth'): 
    ''' array_x: ground truth, array_y: prediction '''
    assert array_x.shape == array_y.shape, f'{array_x.shape} | {array_y.shape}'
    r2 = pearsonr(array_x, array_y)
    if upper is None: 
        upper = max(np.max(array_x), np.max(array_y))
    plt.figure(figsize=(8, 8))
    plt.xlim(0, upper)
    plt.ylim(0, upper)
    plt.scatter(array_x, array_y, color = 'blue', alpha = 0.5)
    plt.title(f'{r2}')
    x = np.arange(0, upper, upper/100)
    y = x
    plt.plot(x, y, color = 'r', linestyle='dashed')
    plt.ylabel(ylabel) 
    plt.xlabel(xlabel)
    plt.grid()
    plt.savefig(savefig)
    # plt.show()
    plt.close()

def plot_pearson_correlation_scatter_lis(array_x: list, array_y: list, upper: float = None, savefig = 'pearsons.png', xlabel = 'Prediction', ylabel = 'Ground Truth'): 
    ''' array_x: ground truth, array_y: prediction '''
    assert len(array_x) == len(array_y), f'{len(array_x)} | {len(array_y)}'
    r2 = pearsonr(array_x, array_y)
    if upper is None: 
        upper = max(max(array_x), max(array_y))
    plt.figure(figsize=(8, 8))
    plt.xlim(0, upper)
    plt.ylim(0, upper)
    plt.scatter(array_x, array_y, color = 'blue', alpha = 0.5)
    plt.title(f'{r2}')
    x = np.arange(0, upper, upper/100)
    y = x
    plt.plot(x, y, color = 'r', linestyle='dashed')
    plt.ylabel(ylabel) 
    plt.xlabel(xlabel)
    plt.grid()
    plt.savefig(savefig)
    # plt.show()
    plt.close()

def plot_line_chart(array_x: np.array, array_y_upper: np.array, array_y_lower: np.array, savefig = 'demand.png', xlabel = 'Time interval', ylabel = 'Traffic demand (unit: veh/h)'): 
    ''' array x: true, array y: prediction '''
    assert len(array_x) == len(array_y_upper), f'{len(array_x)} | {len(array_y_upper)}'
    # true
    plt.plot(range(len(array_x)), array_x, marker='^', color = 'red', label = 'GT')  
    # prediction
    plt.plot(range(len(array_y_upper)), array_y_upper, marker='o', color = 'skyblue', label = 'upper') 
    plt.plot(range(len(array_y_lower)), array_y_lower, marker='*', color = 'blue', label = 'lower') 
    plt.fill_between(range(len(array_x)), array_y_upper, array_y_lower, color='lightsteelblue', alpha=0.5)

    plt.title("Dynamic OD Demand")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(ticks = range(len(array_x)), labels = [str(i) for i in range(len(array_x))])
    plt.legend()
    plt.grid()
    plt.savefig(savefig)
    # plt.show()
    plt.close()

def plot_histogram(array_x: np.array, bin_n: int, range: tuple, savefig='hist.png'):
    plt.hist(array_x, bins=bin_n, range=range, density=True)
    plt.savefig(savefig)
    plt.close()

def plot_historical_loss(x, y_list, label_list, savefig = 'historical_loss.png'):
    for idx, y in enumerate(y_list): 
        plt.plot(x, y, 'o-', label = label_list[idx]) 
    plt.ylabel('loss')
    plt.xlabel('#epochs')
    # plt.xticks(ticks = [x[i] for i in range(0, len(x))], labels = [str(x[i]) for i in range(0, len(x))])
    plt.savefig(savefig)
    # plt.show()
    plt.close()