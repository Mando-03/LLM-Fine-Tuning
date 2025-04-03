# LLM Fine-Tuning for Arabic News Analysis

This repository contains the code and resources used to fine-tune a Qwen2.5-1.5B-Instruct model for Arabic news analysis. The goal of this project was to enhance the model's ability to extract structured information and perform accurate translations on Arabic news articles.

## Project Overview

This project involved several key stages:

1.  **Environment Setup**: Setting up the necessary libraries and dependencies in Google Colab, including cloning the LLaMA-Factory repository and installing required packages.
2.  **Data Preparation**: Curating and formatting a dataset of Arabic news articles for fine-tuning.
3.  **Knowledge Distillation**: Utilizing the Gemini API to generate high-quality training data for the Qwen model.
4.  **Fine-Tuning**: Fine-tuning the Qwen2.5-1.5B-Instruct model using the LLaMA-Factory framework and LoRA.
5.  **Evaluation**: Evaluating the fine-tuned model's performance on information extraction and translation tasks.
6.  **Deployment**: Deploying the fine-tuned model using vLLM for efficient inference.
7.  **Load Testing**: Conducting load testing using Locust to assess the model's performance under stress.

## Key Technologies

-   **Qwen2.5-1.5B-Instruct**: The base language model used for fine-tuning.
-   **LLaMA-Factory**: A framework for efficient fine-tuning of large language models.
-   **LoRA (Low-Rank Adaptation)**: A parameter-efficient fine-tuning technique.
-   **Gemini API**: Used for knowledge distillation to generate high-quality training data.
-   **vLLM**: A high-throughput and memory-efficient inference and serving engine for LLMs.
-   **Locust**: A load-testing tool for simulating user traffic.
-   **Google Colab**: The development environment for this project.

## Project Structure

-   `LLM-FineTuning.ipynb`: The main Colab notebook containing the complete fine-tuning workflow.
-   `locust.py`: The Locust load-testing script.
-   `/gdrive/MyDrive/LLM-Finetunning/`: Directory for saved models and datasets.
    -   `datasets/`: Contains the original and processed datasets.
        -   `news-sample.jsonl`: Raw dataset file.
        -   `sft.jsonl`: Formatted dataset for fine tuning.
        -   `xsft.jsonl`: Formatted dataset for translation fine tuning.
        -   `llamafactory-finetune-data/`: Directory for train and validation datasets.
            -   `train.json`: Training dataset.
            -   `val.json`: Validation dataset.
    -   `models/`: Contains the fine-tuned model checkpoints.

## Setup Instructions

1.  **Clone the Repository**:
    ```bash
    git clone [repository URL]
    cd [repository directory]
    ```
2.  **Google Colab Setup**:
    -   Upload the `LLM-FineTuning.ipynb` notebook to Google Colab.
    -   Mount your Google Drive to `/gdrive`.
    -   Create a directory `/gdrive/MyDrive/LLM-Finetunning/` and upload the datasets to the datasets folder.
    -   Add secrets in Colab for `gemini`, `wandb`, `huggingface` and `ngrok` tokens.
3.  **Install Dependencies**:
    -   Run the setup cells in the notebook to install the required libraries.
4.  **Run the Notebook**:
    -   Execute the cells in the notebook sequentially to perform the fine-tuning and evaluation.

## Key Steps Explained

1.  **Environment Setup**:
    -   Cloned the LLaMA-Factory repository and installed the necessary dependencies.
    -   Configured Google Colab with the required API keys and tokens.
2.  **Data Preparation**:
    -   Loaded the raw Arabic news dataset.
    -   Formatted the data into JSONL format for fine-tuning.
3.  **Knowledge Distillation**:
    -   Utilized the Gemini API to generate structured training data.
    -   Parsed and stored the Gemini API responses in JSON format.
4.  **Fine-Tuning**:
    -   Configured the LLaMA-Factory framework for fine-tuning the Qwen model.
    -   Used LoRA for parameter-efficient fine-tuning.
    -   Monitored the training process using WandB.
5.  **Evaluation**:
    -   Evaluated the fine-tuned model's performance on information extraction and translation tasks.
    -   Compared the model's performance before and after fine-tuning.
6.  **Deployment**:
    -   Deployed the fine-tuned model using vLLM for efficient inference.
    -   Exposed the vLLM API using ngrok for external access.
7.  **Load Testing**:
    -   Used Locust to simulate user traffic and assess the model's performance under load.
    -   Monitored the model's response time and throughput.

## Usage

-   To perform inference, use the vLLM API endpoint provided by ngrok.
-   To conduct load testing, run the `locust.py` script.

## Results

-   The fine-tuned Qwen model demonstrated improved performance on Arabic news analysis tasks.
-   The model was able to accurately extract structured information and perform translations.
-   The vLLM deployment provided efficient inference with high throughput.
-   Locust load testing provided insights into the model's performance under stress.

## Future Improvements

-   Explore different fine-tuning techniques and hyperparameters.
-   Expand the dataset with more diverse and high-quality data.
-   Implement a more robust evaluation pipeline.
-   Optimize the vLLM deployment for production use.
-   Add more load testing scenarios.

## Acknowledgments

-   This project was inspired by the LLaMA-Factory repository.
-   The Gemini API was used for knowledge distillation.
-   The vLLM library was used for model deployment.
-   The Locust library was used for load testing.
