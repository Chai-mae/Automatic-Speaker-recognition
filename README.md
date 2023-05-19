# Automatic-Speaker-recognition
## Introduction :
Automatic Speaker Recognition is a part of speech analysis that encompasses techniques and algorithms for the identification and automatic authentication of individuals based on their unique vocal characteristics, such as voice patterns, speech patterns, and other acoustic features. There are two types of recognition:

1. Text-dependent recognition: In this case, the algorithm is trained on pre-established phrases spoken by all the speakers to be recognized.
2. Text-independent recognition: In this case, there is no training based on specific phrases.

The majority of speaker recognition solutions aim either at speaker identification, which involves recognizing the person speaking among a group of speakers, or speaker verification/authentication, which verifies, with a minimal level of doubt, that a person is the same one who recorded their voice during the verification process.

Project Description:

In this project, our goal is to build a speaker recognition system specifically for the students in our class. The objective is to first identify the student and then verify their identity. The steps involved in accomplishing this project are as follows:

Step 1: Gather Audio Recordings and Build the Dataset
- Collect audio recordings from each student in our class.
- Each student is responsible for recording two audio samples of 1 minute each, one for training and one for testing purposes.
- Store the recordings in a shared drive and create an Excel file to assign each student with a unique identifier.
- We will have separate training and testing files, each containing sub-files (F) and (H) for female and male speakers, respectively.

Step 2: Audio Playback, MFCC Extraction, and Preprocessing
- Read the audio recordings using the SciPy library, defining a function that takes a file path as input and returns three lists: audios, frequencies, and file paths.
- Extract MFCC (Mel-frequency cepstral coefficients) features and preprocess the data.
- To remove silence, calculate the energy of the vocal signal represented in MFCC form for each frame. Use numpy library to calculate the energy.
- Set a threshold at 40% of the average energy to distinguish between speech frames and silence frames. Speech frames have energy higher than the threshold, while silence frames have energy lower.
- Save the extracted MFCCs in text files based on gender. Create a directory named after the gender if it doesn't exist, and save the MFCCs in a text file with the same name as the original audio file but with a ".mfcc" extension. The MFCC values are comma-separated in the text file.
- Observe the reduction in the number of frames after removing silence.

Step 3: GMM Model Training
- Define a function that takes two inputs: the parent directory path and the number of components for the GMM (Gaussian Mixture Model).
- Read a file containing the MFCCs of an audio sample, initialize a GMM model, train the model on the MFCCs, and save it as a pickle file.
- Each student will have four GMM models: one with 128 Gaussians, another with 256, a third with 512, and the last one with 1024.
- The GMM model files are named using the following format: Identifier.n_component.gmm

Step 4: Split the Test Data into Segments of 3s, 10s, 15s, and 30s
- Divide the MFCC files of the test data into segments of 3s, 10s, 15s, and 30s.
- Assume each second corresponds to 100 frames.
- This division is done to illustrate the influence of segment duration on the model's ability to recognize the speaker.

Step 5: Speaker Identification
- Define a function called "predict_speaker" that takes MFCC features and GMM models as input to calculate the
