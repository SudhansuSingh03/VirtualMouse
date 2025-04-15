import subprocess
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)


def run_step(step_description, script_name):
    logging.info(f"Starting: {step_description}")
    result = subprocess.run(["python", script_name], capture_output=True, text=True)

    # Check if the step was successful
    if result.returncode != 0:
        logging.error(f"Error during {step_description}. See details below:\n{result.stderr}")
        raise RuntimeError(f"{step_description} failed.")
    else:
        logging.info(f"Completed: {step_description}")
        logging.info(result.stdout)


if __name__ == "__main__":
    try:
        # Step 1: Collect data
        run_step("Data Collection", "data_collection.py")

        # Step 2: Preprocess data
        run_step("Data Preprocessing", "data_preprocessing.py")

        # Step 3: Train model
        run_step("Model Training", "model_training.py")

        # Step 4: Test model
        run_step("Model Testing", "model_testing.py")

    except RuntimeError as e:
        logging.error(f"Process terminated: {e}")
