import os
import pandas as pd
import matplotlib.pyplot as plt
import re

def plot_model_accuracy_for_best_run(csv_path):
    """Plots the accuracy of each model for each dataset for the same run as the Best Overall."""
    # Load the results summary CSV
    results = pd.read_csv(csv_path)

    # Get the run folder of the Best Overall result
    best_overall_run = results[results["Category"] == "Best Overall"]["Run Folder"].iloc[0]

    # Filter results for the same run folder
    same_run_results = results[results["Run Folder"] == best_overall_run]

    # Filter out rows without a dataset (e.g., Best Per Classifier rows)
    dataset_results = same_run_results[same_run_results["Dataset"].notna()]

    # Pivot the data to create a grouped bar plot
    pivot_data = dataset_results.pivot(index="Dataset", columns="Model", values="Mean Accuracy (%)")

    # Plot the grouped bar plot
    pivot_data.plot(kind="bar", figsize=(10, 6), colormap="viridis", edgecolor="black")
    plt.title(f"Accuracy of Models for Each Dataset (Run: {best_overall_run})", fontsize=14)
    plt.ylabel("Mean Accuracy (%)", fontsize=12)
    plt.xlabel("Dataset", fontsize=12)
    plt.ylim(0, 100)  # Set y-axis limit to 100%
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title="Model", fontsize=10, title_fontsize=12)

    # Annotate the bars with values
    for i, bar_group in enumerate(plt.gca().containers):
        for bar in bar_group:
            height = bar.get_height()
            if height > 0:  # Only annotate non-zero bars
                plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{height:.2f}%", ha="center", fontsize=8)

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_selected_model_dataset_accuracy(csv_path):
    """Plots the accuracy for RCE and RBF models across IRIS, WINE, and HAND datasets for the Best Overall run."""
    # Load the results summary CSV
    results = pd.read_csv(csv_path)

    # Get the run folder of the Best Overall result
    best_overall_run = results[results["Category"] == "Best Overall"]["Run Folder"].iloc[0]

    # Filter results for the same run folder
    same_run_results = results[results["Run Folder"] == best_overall_run]

    # Filter for the specific combinations of models and datasets
    selected_results = same_run_results[same_run_results["Dataset"].isin(["IRIS ===", "WINE ===", "HAND ==="])]

    # Pivot the data to prepare for plotting
    pivot_data = selected_results.pivot(index="Dataset", columns="Model", values="Mean Accuracy (%)")

    # Plot the grouped bar plot
    pivot_data.plot(kind="bar", figsize=(8, 6), colormap="viridis", edgecolor="black")
    plt.title(f"Accuracy of RCE and RBF Models Across Datasets (Run: {best_overall_run})", fontsize=14)
    plt.ylabel("Mean Accuracy (%)", fontsize=12)
    plt.xlabel("Dataset", fontsize=12)
    plt.ylim(0, 100)  # Set y-axis limit to 100%
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title="Model", fontsize=10, title_fontsize=12)

    # Annotate the bars with values
    for i, bar_group in enumerate(plt.gca().containers):
        for bar in bar_group:
            height = bar.get_height()
            if height > 0:  # Only annotate non-zero bars
                plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{height:.2f}%", ha="center", fontsize=8)

    # Show the plot
    plt.tight_layout()
    plt.show()

def parse_results_txt(file_path):
    """Parses the results.txt file and extracts accuracy data."""
    results = {}
    current_dataset = None

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("=== Dataset:"):
                current_dataset = line.split(":")[1].strip().replace(" ===", "")  # Remove " ==="
                results[current_dataset] = {}
            elif line.endswith(":") and current_dataset:
                current_model = line[:-1]
            elif "Mean Accuracy" in line and current_dataset and current_model:
                mean_acc = float(re.search(r"Mean Accuracy: ([\d.]+)%", line).group(1))
                results[current_dataset][current_model] = mean_acc

    return results

def plot_results(results, title):
    """Plots the results as a horizontal bar plot."""
    datasets = list(results.keys())
    models = list(next(iter(results.values())).keys())  # Get models from the first dataset

    # Prepare data for plotting
    data = {model: [results[dataset].get(model, 0) for dataset in datasets] for model in models}

    # Plot horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.4
    y_positions = range(len(datasets))

    for i, (model, accuracies) in enumerate(data.items()):
        ax.barh(
            [y + i * bar_width for y in y_positions],
            accuracies,
            height=bar_width,
            label=model,
        )

    # Add labels and title
    ax.set_yticks([y + bar_width / 2 for y in y_positions])
    ax.set_yticklabels(datasets)
    ax.set_xlabel("Mean Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=14)

    # Move the legend outside the plot
    ax.legend(title="Model", fontsize=10, title_fontsize=12, loc='upper left', bbox_to_anchor=(1.05, 1))

    # Annotate bars with values
    for bars in ax.containers:
        for bar in bars:
            width = bar.get_width()
            if width > 0:  # Only annotate non-zero bars
                ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f"{width:.2f}%", va="center", fontsize=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from numpy import nan
    # Path to the results_summary.csv file
    summary = "results_summary.csv"  # Update the path as needed
    df = pd.read_csv(summary)
    files =     list(df["Run Folder"])
    category =  list(df["Category"])
    model =     list(df["Model"])
    dataset =   list(df["Dataset"])

    runs = zip(files, category, model, dataset)

    for file, cat, mod, data in runs:
        results_txt_path = f"./runs/{file}/results.txt"  # Update the path as needed

        # Parse the results.txt file
        results_data = parse_results_txt(results_txt_path)

        name = f"{str(cat)} - {str(mod) if not nan  else ''} - {str(data).replace(' ===', '') if not nan  else ''}"  # Remove " ===" from dataset name
        # Plot the results
        plot_results(results_data, title="Comparison of Classifiers Across Datasets\n" + name)
        # Save the plot as an image
        plot_path = f"./runs/{file}/plots/{name}.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")