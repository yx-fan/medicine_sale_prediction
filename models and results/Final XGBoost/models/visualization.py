import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import platform

def get_chinese_font():
    """Get Chinese font path based on the operating system"""
    system = platform.system()
    
    # Try to find Chinese fonts on different systems
    font_paths = []
    
    if system == 'Darwin':  # macOS
        font_paths = [
            '/System/Library/Fonts/STHeiti Light.ttc',
            '/System/Library/Fonts/PingFang.ttc',
            '/Library/Fonts/Arial Unicode.ttf'
        ]
    elif system == 'Linux':
        font_paths = [
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
            '/usr/share/fonts/truetype/arphic/uming.ttc',
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf',
        ]
    elif system == 'Windows':
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',
            'C:/Windows/Fonts/msyh.ttc',
            'C:/Windows/Fonts/simsun.ttc'
        ]
    
    # Try to find an existing font
    for path in font_paths:
        if os.path.exists(path):
            return FontProperties(fname=path)
    
    # Fallback: use matplotlib's default font (may not support Chinese)
    return FontProperties()
    # Alternative: use font name instead of file path
    # return FontProperties(family='DejaVu Sans', size=10)

def plot_predictions(group_data, predictions, drug_name, factory_name, font_path=None, plot_dir=None, show_plot=False):
    # Use the new font detection function instead of direct font_path
    font = get_chinese_font()
    plt.figure(figsize=(10, 6))
    plt.plot(group_data.index, group_data['减少数量'], label='Actual')
    plt.plot(group_data.index[len(group_data) - len(predictions):], predictions, label='Forecast', linestyle='--')
    # Use English labels to avoid font issues, or use font if available
    try:
        plt.xlabel('Date', fontproperties=font)
        plt.ylabel('Consumption', fontproperties=font)  # Changed from Chinese to English
        plt.title(f'XGBoost Forecast vs Actual - {drug_name} + {factory_name}', fontproperties=font)
        plt.legend(prop=font)
    except:
        # Fallback to default font if Chinese font fails
        plt.xlabel('Date')
        plt.ylabel('Consumption')
        plt.title(f'XGBoost Forecast vs Actual - {drug_name} + {factory_name}')
        plt.legend()

    # Replace invalid characters (e.g., "/") in the drug and factory names
    safe_drug_name = drug_name.replace('/', '_')
    safe_factory_name = factory_name.replace('/', '_')

    # Set default plot directory to one level up, in the 'pic' folder
    if plot_dir is None:
        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pic')
    else:
        plot_dir = os.path.abspath(plot_dir)
    
    # Create the plot directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)

    # Save the plot to the specified directory with safe file names
    plot_path = os.path.join(plot_dir, f"SARIMAX_Prediction_{safe_drug_name}_{safe_factory_name}.png")
    plt.savefig(plot_path, format='png', dpi=300)
    print(f"Plot saved to {plot_path}")

    # Show the plot if specified
    if show_plot:
        plt.show()

    # Close the plot to free memory
    plt.close()
