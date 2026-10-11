import torch
import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


from src import config, dataset, utils

def get_predictions_and_labels(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch_idx, data in enumerate(progress_bar):
            inputs, labels = data
            if inputs is None or labels is None:
                print(f"Skipping batch {batch_idx} in evaluation due to data loading error.")
                continue

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)

def calculate_metrics(y_true, y_pred, class_names=None):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true, y_pred, average=config.AVERAGE_METHOD, zero_division=0
    )

    metrics = {
        'accuracy': accuracy,
        f'{config.AVERAGE_METHOD}_precision': precision,
        f'{config.AVERAGE_METHOD}_recall': recall,
        f'{config.AVERAGE_METHOD}_f1_score': f1_score
    }

    clf_report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_str = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    if class_names:
        for class_name in class_names:
            if class_name in clf_report_dict:
                metrics[f'{class_name}_precision'] = clf_report_dict[class_name]['precision']
                metrics[f'{class_name}_recall'] = clf_report_dict[class_name]['recall']
                metrics[f'{class_name}_f1_score'] = clf_report_dict[class_name]['f1-score']

    return metrics, report_str

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "confusion_matrix.png"))
    plt.close()

def run_evaluation(model_path, data_dir, data_type="Test"):
    print(f"\n--- Running Evaluation on {data_type} Data ---")
    print(f"Model path: {model_path}")
    print(f"Data directory: {data_dir}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    class_to_idx, idx_to_class = utils.load_class_mapping()
    if not class_to_idx or not idx_to_class:
        print("Error: Could not load class mapping. Evaluation cannot proceed accurately without it.")
        return

    num_classes = len(class_to_idx)
    class_names = [idx_to_class[i] for i in range(num_classes)]

    print("Loading model...")
    trained_model, _, _, _ = utils.load_model(model_path, num_classes_override=num_classes)
    if trained_model is None:
        print(f"Failed to load model from {model_path}. Evaluation aborted.")
        return
    trained_model.to(config.DEVICE)

    print(f"Loading {data_type} data...")
    eval_loader = dataset.get_test_dataloader(
        test_data_dir=data_dir,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        class_to_idx=class_to_idx
    )

    if eval_loader is None or len(eval_loader.dataset) == 0:
        print(f"The {data_type} dataset is empty. Nothing to evaluate.")
        return

    print(f"Getting predictions from {data_type} set...")
    y_pred, y_true = get_predictions_and_labels(trained_model, eval_loader, config.DEVICE)

    if len(y_true) == 0 or len(y_pred) == 0:
        print(f"No predictions or true labels obtained. Evaluation aborted.")
        return

    print(f"Calculating metrics for {data_type} set...")
    metrics, report_str = calculate_metrics(y_true, y_pred, class_names=class_names)

    metrics_filename = f"{data_type.lower()}_{config.METRICS_FILE_NAME}"
    metrics_save_path = os.path.join(config.RESULTS_DIR, metrics_filename)
    utils.save_metrics(metrics, metrics_save_path)

    report_filename = f"{data_type.lower()}_classification_report.txt"
    report_save_path = os.path.join(config.RESULTS_DIR, report_filename)
    with open(report_save_path, 'w') as f:
        f.write(f"Evaluation Metrics for {data_type} Set\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Data: {data_dir}\n\n")
        f.write(report_str)
    print(f"Classification report saved to {report_save_path}")

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        csv_filename = f"{data_type.lower()}_metrics_summary_{timestamp}.csv"
        csv_save_path = os.path.join(config.RESULTS_DIR, csv_filename)

        overall_metrics_data = {
            'Metric': ['Accuracy', f'{config.AVERAGE_METHOD.capitalize()} Precision', f'{config.AVERAGE_METHOD.capitalize()} Recall', f'{config.AVERAGE_METHOD.capitalize()} F1-Score'],
            'Score': [metrics.get('accuracy', 'N/A'), metrics.get(f'{config.AVERAGE_METHOD}_precision', 'N/A'), metrics.get(f'{config.AVERAGE_METHOD}_recall', 'N/A'), metrics.get(f'{config.AVERAGE_METHOD}_f1_score', 'N/A')]
        }
        df_overall = pd.DataFrame(overall_metrics_data)

        per_class_data = {'Class': [], 'Precision': [], 'Recall': [], 'F1-Score': [], 'Support': []}
        clf_report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        for class_name in class_names:
            if class_name in clf_report_dict:
                per_class_data['Class'].append(class_name)
                per_class_data['Precision'].append(clf_report_dict[class_name]['precision'])
                per_class_data['Recall'].append(clf_report_dict[class_name]['recall'])
                per_class_data['F1-Score'].append(clf_report_dict[class_name]['f1-score'])
                per_class_data['Support'].append(clf_report_dict[class_name]['support'])
        df_per_class = pd.DataFrame(per_class_data)

        with open(csv_save_path, 'w') as f:
            f.write(f"Overall Metrics ({config.AVERAGE_METHOD} average):\n")
        df_overall.to_csv(csv_save_path, mode='a', index=False, float_format='%.4f')
        with open(csv_save_path, 'a') as f:
            f.write("\n\nPer-Class Metrics:\n")
        df_per_class.to_csv(csv_save_path, mode='a', index=False, float_format='%.4f')
        print(f"Metrics table saved to {csv_save_path}")
    except Exception as e:
        print(f"Could not generate or save CSV metrics table: {e}")

    plot_confusion_matrix(y_true, y_pred, class_names)

    print(f"\n--- Evaluation for {data_type} data complete. Results in {config.RESULTS_DIR} ---")

if __name__ == '__main__':
    model_file_path = os.path.join(config.MODEL_SAVE_DIR, config.BEST_MODEL_NAME)
    if os.path.exists(model_file_path) and os.path.exists(config.TEST_DATA_DIR):
        run_evaluation(
            model_path=model_file_path,
            data_dir=config.TEST_DATA_DIR,
            data_type="Test"
        )
    else:
        print("Skipping run_evaluation: Model or test data directory is missing.")
        if not os.path.exists(model_file_path):
            print(f"Missing: {model_file_path}")
        if not os.path.exists(config.TEST_DATA_DIR):
            print(f"Missing: {config.TEST_DATA_DIR}")
