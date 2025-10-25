# Brain Tumor Classification

This project focuses on classifying brain tumors using machine learning techniques. It leverages deep learning models to predict the presence of brain tumors from medical images, potentially aiding in early diagnosis and treatment. The project specifically classifies brain tumors into the following categories:

Meningioma: A type of tumor that arises from the meninges, the layers of tissue covering the brain and spinal cord.

Pituitary Tumor: A tumor that occurs in the pituitary gland, which is responsible for hormone production.

Glioma: A type of tumor that begins in the glial cells of the brain, which provide support and protection for neurons.

## Current Progress

The baseline ResNet50 model for feature extraction and tumor classification has been implemented and is performing well.

We are currently working on integrating the Radial Basis Probabilistic Neural Network (RBPNN) classifier to further improve classification accuracy and precision.

In addition to improving the model, we plan to expand our dataset to include more variations, such as blurred images, noisy images, and images with no tumors. This will help the model generalize better and improve its robustness across different scenarios.

The hybrid model we are developing will use ResNet50 for feature representation. The extracted embeddings will then be passed into the RBPNN classifier, which will help sharpen decision boundaries, improve robustness across datasets, and provide clearer uncertainty signals.

In parallel, we are planning future ablations to test different models. We will swap the RBPNN for alternative heads and backbones, including:

- Linear SVMs on embeddings

- Shallow MLP classifiers

- Logistic regression

- Alternative feature extractors like EfficientNet or ConvNeXt

This will allow us to separate the gains from the probabilistic head (RBPNN) from those that come from different feature extractors. By doing so, we can ensure that any improvements are not solely tied to a single modeling choice.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/ArpithaPrakash/brain-tumor-cls.git
   cd brain-tumor-cls
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare the Data**: Make sure you have the dataset stored in the appropriate directory (likely under the `data/` folder).
   The current dataset and project resources are available through the following Google Drive folder:
   https://drive.google.com/drive/folders/1UqQtpetjvWrkhwaeD3yaaE_S50YWefEr?usp=drive_link

2. **Run the Jupyter Notebook**: The core implementation can be found in `IDL_Project.ipynb`. Open it with Jupyter and run through the steps to train and evaluate the model.

   ```bash
   jupyter notebook
   ```

3. **Model Training**: The notebook contains steps for training the model on the dataset, including data preprocessing, model creation, training, and evaluation.

4. **Prediction**: After training the model, you can use it to predict brain tumors from new images.

## Directory Structure

```plaintext
brain-tumor-cls/
├── data/                   # Dataset for training/testing the model
├── reports/                # Reports and results of the analysis
├── src/                    # Source code for the project
├── IDL_Project.ipynb       # Jupyter notebook for model implementation
├── .gitignore              # Git ignore file
└── requirements.txt        # Python dependencies for the project
```

## Dependencies

This project requires the following Python libraries:

- TensorFlow/Keras
- NumPy
- Matplotlib
- Scikit-learn
- Pandas
- OpenCV (if using image processing)

You can install all dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to your branch (`git push origin feature-branch`).
5. Create a new pull request.
