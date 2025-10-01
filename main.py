from utils_ensemble import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_directory', type=str, default='./', help='Directory where csv files are stored')
parser.add_argument('--topk', type=int, default=2, help='Top-k number of classes')
args = parser.parse_args()

r1 = r'D:\COVID-Detection-Gompertz-Function-Ensemble-main\sars-cov-2_csv\vgg11.csv'
r2 = r'D:\COVID-Detection-Gompertz-Function-Ensemble-main\sars-cov-2_csv\wideresnet50-2.csv'
r3 = r'D:\COVID-Detection-Gompertz-Function-Ensemble-main\sars-cov-2_csv\inception.csv'

# Define the paths to CSV files
csv_files = [
    r1,
    r2,
    r3,
]

root = args.data_directory
if not root.endswith('/'):
    root += '/'

try:
    # Load data from CSV files
    data = [getfile(os.path.join(root, file)) for file in csv_files]
    if any(data_item is None for data_item in data):
        raise FileNotFoundError("One or more CSV files not found.")

    # Unpack data
    p1, labels = data[0]
    p2, _ = data[1]
    p3, _ = data[2]

    top = args.topk  # top 'k' classes
    predictions = Gompertz(top, p1, p2, p3)

    # Print shapes for debugging
    print(f"Shapes - predictions: {predictions.shape}, labels: {labels.shape}")

    correct = np.sum(predictions == labels)
    total = labels.shape[0]

    accuracy = correct / total
    print(f"Accuracy = {accuracy:.4f}")

    classes = list(map(str, range(p1.shape[1])))  # Assuming classes are numbered 0, 1, 2, ..., n-1
    metrics(labels, predictions, classes)
    plot_roc(labels, predictions)

except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")



import pickle

try:
    # Open the .pkl file
    with open('model_weights.pkl', 'rb') as f:
        data = pickle.load(f)
    print(data)
except pickle.UnpicklingError as e:
    print(f"Error loading pickle file: {e}")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
