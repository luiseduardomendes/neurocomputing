import os
import re
import pandas as pd

def parse_parameters(file_path):
    """Parses the parameters.txt file and returns a dictionary of parameters."""
    params = {}
    with open(file_path, "r") as f:
        for line in f:
            if ":" in line:
                key, value = line.strip().split(": ")
                params[key] = value if not value.replace('.', '', 1).isdigit() else float(value)
    params.pop("n_splits", None)  # Remove n_splits if it exists
    return params

def parse_results(file_path):
    """Parses the results.txt file and returns a dictionary of results."""
    results = {}
    current_dataset = None
    current_model = None

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("=== Dataset:"):
                current_dataset = line.split(":")[1].strip()
                results[current_dataset] = {}
            elif line.endswith(":") and current_dataset:
                current_model = line[:-1]
                results[current_dataset][current_model] = {}
            elif "Mean Accuracy" in line and current_model:
                mean_acc = float(re.search(r"Mean Accuracy: ([\d.]+)%", line).group(1))
                results[current_dataset][current_model]["mean_accuracy"] = mean_acc
            elif "Standard Deviation" in line and current_model:
                std_dev = float(re.search(r"Standard Deviation: ([\d.]+)%", line).group(1))
                results[current_dataset][current_model]["std_deviation"] = std_dev
    return results

def find_best_results(runs_folder):
    """Finds the best overall, best per dataset, and best per classifier."""
    best_overall = None
    best_per_dataset = {}
    best_per_classifier = {}

    for run_folder in os.listdir(runs_folder):
        run_path = os.path.join(runs_folder, run_folder)
        if not os.path.isdir(run_path):
            continue

        parameters_file = os.path.join(run_path, "parameters.txt")
        results_file = os.path.join(run_path, "results.txt")

        if not os.path.exists(parameters_file) or not os.path.exists(results_file):
            continue

        params = parse_parameters(parameters_file)
        results = parse_results(results_file)

        # Find the best overall
        for dataset, models in results.items():
            for model, metrics in models.items():
                if best_overall is None or metrics["mean_accuracy"] > best_overall[0]:
                    best_overall = (
                        metrics["mean_accuracy"],
                        metrics["std_deviation"],
                        model,
                        dataset,
                        params,
                        run_folder,
                    )

                # Find the best per dataset
                if dataset not in best_per_dataset or metrics["mean_accuracy"] > best_per_dataset[dataset][0]:
                    best_per_dataset[dataset] = (
                        metrics["mean_accuracy"],
                        metrics["std_deviation"],
                        model,
                        params,
                        run_folder,
                    )

                # Find the best per classifier
                if model not in best_per_classifier or metrics["mean_accuracy"] > best_per_classifier[model][0]:
                    best_per_classifier[model] = (
                        metrics["mean_accuracy"],
                        metrics["std_deviation"],
                        params,
                        run_folder,
                    )

    return best_overall, best_per_dataset, best_per_classifier

def display_results(best_overall, best_per_dataset, best_per_classifier):
    """Displays the parameters and performance of the best results using pandas DataFrame."""
    # Best Overall
    print("\n=== Best Overall ===")
    df_overall = pd.DataFrame([{
        "Category": "Best Overall",
        "Model": best_overall[2],
        "Dataset": best_overall[3],
        "Run Folder": best_overall[5],
        "Mean Accuracy (%)": best_overall[0],
        "Standard Deviation (%)": best_overall[1],
        **best_overall[4]  # Unpack parameters
    }])
    print(df_overall)

    # Best Per Dataset
    print("\n=== Best Per Dataset ===")
    dataset_rows = []
    for dataset, result in best_per_dataset.items():
        dataset_rows.append({
            "Category": "Best Per Dataset",
            "Model": result[2],
            "Dataset": dataset,
            "Run Folder": result[4],
            "Mean Accuracy (%)": result[0],
            "Standard Deviation (%)": result[1],
            **result[3]  # Unpack parameters
        })
    df_dataset = pd.DataFrame(dataset_rows)
    print(df_dataset)

    # Best Per Classifier
    print("\n=== Best Per Classifier ===")
    classifier_rows = []
    for model, result in best_per_classifier.items():
        classifier_rows.append({
            "Category": "Best Per Classifier",
            "Model": model,
            "Dataset": "",
            "Run Folder": result[3],
            "Mean Accuracy (%)": result[0],
            "Standard Deviation (%)": result[1],
            **result[2]  # Unpack parameters
        })
    df_classifier = pd.DataFrame(classifier_rows)
    print(df_classifier)

    # Save all results to a CSV file
    all_results = pd.concat([df_overall, df_dataset, df_classifier], ignore_index=True)
    all_results.to_csv("results_summary.csv", index=False)
    print("\nResults saved to 'results_summary.csv'.")

if __name__ == "__main__":
    runs_folder = "./runs"  # Path to the folder containing all runs
    best_overall, best_per_dataset, best_per_classifier = find_best_results(runs_folder)
    display_results(best_overall, best_per_dataset, best_per_classifier)