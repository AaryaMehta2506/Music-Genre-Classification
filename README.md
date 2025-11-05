AI/ML Beginners Project
# Music Genre Classification

## Overview
This project implements a machine learning pipeline to classify music tracks into genres using audio features extracted from audio files. The repository includes feature extraction, data preprocessing, model training and evaluation, and an optional Streamlit app for inference.

## Dataset
Recommended dataset: GTZAN Music Genre Dataset
Link : https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection
Typical structure:
- 1000 audio files (30 seconds each)
- 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
Place the dataset so that each genre is a subfolder containing audio files (for example .au or .wav).

## Features Extracted
Common audio features used:
- MFCC (Mel-frequency cepstral coefficients)
- Chroma features
- Mel-spectrogram statistics
- Spectral contrast
- Tonnetz features

## Project Workflow
1. Extract audio features from files using librosa.
2. Build a feature matrix and label vector.
3. Encode labels and scale features.
4. Split data into training and test sets with stratified sampling.
5. Train classical ML models (for example Random Forest).
6. Evaluate using accuracy, precision, recall, F1-score, and confusion matrix.
7. Save model artifacts: trained model, scaler, and label encoder.
8. Optionally provide a Streamlit app for uploading an audio file and predicting genre.

## Requirements
Install required Python packages:
pip install librosa scikit-learn pandas numpy joblib matplotlib streamlit

## How to Run
1. Ensure the GTZAN dataset is available and DATASET_PATH in the notebook matches your dataset location.
2. Run the notebook cells in order to extract features, train the model, and save artifacts.
3. To run the Streamlit app if included:
streamlit run app.py

## Repository Structure
project/
│
├── dataset_folder/genre            (place GTZAN or other dataset here; genres as subfolders)
├── main.ipynb             (Jupyter notebook with feature extraction, training, evaluation)
├── music_genre_model.pkl      (saved trained model)
├── music_scaler.pkl           (saved scaler for feature normalization)
├── label_encoder.pkl          (saved label encoder mapping numeric labels to genre names)
└── README.md                  (this file)

## Results
Baseline classical models such as Random Forest provide a reasonable baseline (for example around 70% accuracy on GTZAN with summarized features). Results will vary depending on preprocessing and model choices.

## Notes and Tips
- Feature extraction can be time consuming; save extracted features to disk for reuse.
- Use consistent audio loading and preprocessing during inference and training.
- For improved performance, consider deep learning on spectrogram images using CNNs.
- Set random_state for reproducibility where applicable.

## Future Work
- Add explainability using SHAP for feature importance.
- Implement batch prediction via file upload.
- Explore deep learning approaches using spectrograms and convolutional neural networks.

## Contributing
Contributions are welcome!
Feel free to fork the repository, improve the game, and open a pull request. Let's grow this classic game together!

## License
This project is licensed under the [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

## Author
**Aarya Mehta**  
🔗 [GitHub Profile](https://github.com/AaryaMehta2506)


