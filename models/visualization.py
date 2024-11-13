import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def plot_predictions(group_data, predictions, drug_name, factory_name, font_path, plot_dir=None, show_plot=False):
    font = FontProperties(fname=font_path)
    plt.figure(figsize=(10, 6))
    plt.plot(group_data.index, group_data['减少数量'], label='Actual')
    plt.plot(group_data.index[len(group_data) - len(predictions):], predictions, label='Forecast', linestyle='--')
    plt.xlabel('Date', fontproperties=font)
    plt.ylabel('减少数量', fontproperties=font)
    plt.title(f'SARIMAX Forecast vs Actual - {drug_name} + {factory_name}', fontproperties=font)
    plt.legend(prop=font)

    # Set default plot directory to one level up, in the 'pic' folder
    if plot_dir is None:
        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pic')
    else:
        plot_dir = os.path.abspath(plot_dir)
    
    # Save the plot to the specified directory
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"SARIMAX_Prediction_{drug_name}_{factory_name}.png")
    plt.savefig(plot_path, format='png', dpi=300)
    print(f"Plot saved to {plot_path}")

    # Show the plot if specified
    if show_plot:
        plt.show()

    # Close the plot to free memory
    plt.close()