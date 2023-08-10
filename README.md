# Music Genre Classification using Neural Networks

Music Genre Classification system using neural networks and audio signal processing to classify different genres of music.

## Table of contents

- Libraries Used
- Data Collection & Preparation
- Feature Extraction
- Building the Model
- Training
- Testing and Making Predictions

## Libraries Used

- NumPy
- Pandas
- Librosa
- Scikit-learn
- Keras
- TensorFlow

You can install them using pip:

```bash
pip install numpy pandas librosa scikit-learn keras tensorflow
```

## Data Collection & Preparation

The GTZAN dataset is used in this project. Which contains 1000 music clips categorized into 10 genres. Each clip lasts for 30 seconds.

## Feature Extraction

Used Librosa to extract features like MFCCs, Tempo, and Mel-scaled spectrogram from the audio files.  

```python
def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
    tempo, _ = librosa.beat.beat_track(y, sr=sr)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    mel = np.mean(librosa.feature.melspectrogram(y, sr=sr))

    return mfccs, tempo, contrast, mel
```

## Building the Model

Built a multi-layered Neural Network with Keras library.

```python
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))
```

## Training

The model has been trained for 20 epochs.

```python
history = model.fit(X_train, y_train, epochs=20, batch_size=128)
```

## Testing and Making Predictions

Finally, tested the model and computed the test accuracy. You can also make predictions using the model:

```python
test_loss, test_acc = model.evaluate(X_test,y_test)
predictions = model.predict(X_test)
```

Enjoy exploring the model and happy coding!
