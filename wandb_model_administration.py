import wandb
import pandas as pd
import os


# Specify your local directory where you want to save the models
local_directory = "/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/trained_models/"

# Create the directory if it doesn't exist
if not os.path.exists(local_directory):
    os.makedirs(local_directory)

# Replace with your wandb username and project name
username = "deeprna"
project_name = "deepRNA"

# Initialize wandb API
api = wandb.Api()

# Fetch project
runs = api.runs(f"{username}/{project_name}")

# Initialize a list to hold your data
model_data = []


for run in runs:
    # Get configuration and summary
    config = run.config
    summary = run.summary._json_dict
    is_sweep = "Yes" if run.sweep else "No"

    # Check and download the model file (.pth) from the root directory
    for file in run.files():
        if file.name.endswith('.pth'):
            # Define the local path for the model file
            local_model_path = os.path.join(local_directory, f"{file.name}")
            training_command = config.get('training_command', 'Not logged')


            
            if "new_structure_success" in run.tags and run.state == "finished":
            # Record the information
                model_data.append({
                    "Run ID": run.id,
                    "Run Name": run.name,
                    "Model File Name": file.name,
                    "Config": config,
                    "Summary": summary,
                    "Download Path": local_model_path,
                    "Whether Sweep" : is_sweep,
                    "Tags" : run.tags
                })
                # Download the file to the specified local path
                file.download(root=local_directory, replace=True)

# Create a DataFrame from your list
df = pd.DataFrame(model_data)

# Save the DataFrame to a CSV file
csv_file_path = "/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/model_and_parameter_information.csv"  # Specify your desired file path
df.to_csv(csv_file_path, index=False)

print("Model information saved to:", csv_file_path)
