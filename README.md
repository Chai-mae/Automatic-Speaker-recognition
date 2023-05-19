# Automatic-Speaker-recognition
**<h2>Table of contents</h2>**

   [Introduction:](#Introduction)
   
   [Project Description:](#Project-Description)
   
   [General structure of the project](#General-structure-of-the-project)
   
   [Obtained Results](#Obtained-Results)

   [Conclusion](#Conclusion)
   
**<h2>Introduction:</h2>**
Automatic speaker recognition is a part of speech analysis, it is the set of techniques and algorithms allowing the automatic identification and authentication of individuals based on their unique speech characteristics, such as their patterns voices, their speech patterns and other acoustic characteristics. There are two types of speaker recognition:

• Text-dependent recognition : in this case the algorithm is trained by pre-established sentences and said by all the speakers to be recognized

• Text-independent recognition : in this case there is no training following one or more specific sentences

The majority of the solutions developed for speaker recognition aim either to identify the speaker, i.e. to recognize the person who spoke among a group of speakers, or else verification or authentication which consists of checking with a minimal level of doubt that a person is the one who recorded their voice during verification.

**<h2>Project Description:</h2>**

As part of this project, we aim to build a speaker recognition system especially for students in our class. The objective being the identification of the student at first and the verification of the latter thereafter
The steps followed to carry out this project are as follows:

[Step 1: Gather all the audio recordings and build the dataset](#step-1:-gather-all-the-audio-recordings-and-build-the-dataset)

[Step 2: Reading of recordings, extraction of MFCCs and pre-processing.    ](#Step-2:)

[Step 3: Train the GMM models for each speaker](#Step-3:-Construction-of-GMM-models)

[Step 4: Divide the Test data set into 3s,10s,15s and 30s segments](#Step-4:-Divide-the-Test-data-set-into-3s,-10s,-15s-and-30s-segments)

[Step 5: Identification](#Step-5:-Identification)

[Step 6: Verification](#Step-6:-Verification)
    
In the rest of this report, we will detail each step of the approach followed.

**<h2>General structure of the project</h2>**

### Step 1: Gather all the audio recordings and build the dataset 

Our dataset is formed by a set of audio recordings made by each student in our class. Indeed each of us was responsible for recording two audios of 1 min one for the train and one for the test and depositing them in the drive as well as mentioning his name in the dedicated Excel file to assign each student his identifier. We have two files Train and Test each contains sub-files (F) and (H) in which the records are stored.
### Step 2:
• Reading of recordings:

In order to read the audio recordings we have defined the following function which allows to read the audios from a given path (filepath) using the Scipy library and returns three lists: audios, freqs, filepaths.
 ```javascript
 def read_audios(path):
    audios = []
    freqs = []
    filepaths = []
    #walking through the directory that contains the dataset and reading each file that has the .wav extension
    for dp, dn, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.wav'):
                filepath = os.path.join(dp, filename)
                filepaths.append(filepath)
                with open(filepath, "rb") as f:
                    # load the audio using scipy
                    freq, data = scipy.io.wavfile.read(f, mmap=False)
                    # append the data and frequency to the respective lists
                    audios.append(data)
                    freqs.append(freq)
    return audios, freqs, filepaths 
 ```

• Extraction of MFFCs and preprocessing:

After reading the audio recordings comes the step where we must extract the Mfcc coefficients and delete the frames that constitute the silence. To do this, we have defined the following function which takes as input the audios list, the freqs list, the filepaths list, and the path where you want to save the MFFCs.
In order to remove the silence, we calculated the energy of the voice signal represented in MFCC form. It is calculated for each frame of the MFCCs using the numpy library. After calculating the energy, a threshold is calculated at 40% of the average energy. This threshold is used to distinguish frames of silence from frames of speech. Speech frames correspond to frames where the energy is above the threshold, while silence frames correspond to frames where the energy is below.
Extracted MFCCs are saved in genre-based files. The function first extracts gender information from the file name by checking whether it contains the string "H" or "F". Then it saves the MFCCs in a directory named after the genre, and if the directory doesn't exist, it creates a new one. Finally, it saves the MFCCs to a text file with the same name as the original audio file, but with an ".mfcc" extension. MFCC functions are saved as comma-separated values in the text file.

 ```javascript
 def extractMfccs_RemoveSilence_saveMfccs(audio, freq, filepath, directory):

    
    mfcc_features = mfcc(audio, freq, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=2048, lowfreq=0,
                         highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=False)

    energy = np.sum(mfcc_features ** 2, axis=1)
    threshold = np.mean(energy) * 0.4
    voiced_indices = np.where(energy > threshold)[0]
    mfccs_voiced = mfcc_features[voiced_indices, :]

    print(f"MFCCs before removing silence: {mfcc_features.shape}")
    print(f"MFCCs after removing silence: {mfccs_voiced.shape}")

    gender = None
    if 'H' in filepath:
        gender = 'H'
    elif 'F' in filepath:
        gender = 'F'

    if gender is not None:
        gender_dir = os.path.join(directory, gender)
        if not os.path.exists(gender_dir):
            os.makedirs(gender_dir)
        mfcc_file = os.path.join(gender_dir, os.path.splitext(os.path.basename(filepath))[0] + ".mfcc")
        np.savetxt(mfcc_file, mfccs_voiced, delimiter=',')

 ```
 It can be seen here that the number of frames decreased after removing silence.
 
 ![Screenshot_872](https://github.com/Chai-mae/Automatic-Speaker-recognition/assets/86806466/f88cf57d-2d3e-4cfe-9653-8359e11220b9)

### Step 3: Construction of GMM models

In this step we have defined a function that takes two parameters as input: parentDir (the parent directory path) and n_components (the number of components for the GMM). This function allows you to read a file containing the MFFcs of an audio, initialize a GMM model, train it on the MFFCs, then save it in a pickle file.
Each student has four Gmm models: one model with 128 Gaussians, a second with 256, a third with 512 and a last with 1024.
The names of the files containing the trained GMM models are in the form: Identifier.n_component.gmm
 ```javascript
 def gmm(parentDir,  n_components):

    # Loop over the two folders "H" and "F"
    for folder in ['H', 'F']:
        # Get the list of files in the folder
        folder_path = os.path.join(parentDir, folder)
         # Get the list of files in the folder
        files = os.listdir(folder_path)

        # Loop over the files in the folder
        for file in files:
            # Load the MFCC features from the file
            mfcc_features =np.loadtxt(os.path.join(folder_path, file), delimiter = ',')
    

            # Create a GMM object
            gmm = GaussianMixture(n_components=n_components)

            # Fit the GMM to the MFCC features
            gmm.fit(mfcc_features)

            # Save the trained GMM to a file with a name of Hi.n_components.gmm
            gmm_file_name = os.path.splitext(file)[0] + '.' + str(n_components) + '.gmm'
            gmm_file_path = os.path.join(r'C:\Users\ASUS ROG STRIX\Desktop\Projet\RAL\GMM', gmm_file_name)
            with open(gmm_file_path, 'wb') as f:
                pickle.dump(gmm, f)

 ```
### Step 4: Divide the Test data set into 3s, 10s, 15s and 30s segments

Then we split the test mfcc files into 3s, 10s, 15s and 30s files that we created. We assume that each second equals 100 frames. This division is made to illustrate the influence of segment duration on the ability of the model to recognize the speaker.
 ```javascript
 def split_audio_test(test_mfccs, segment_length_sec):

    # Compute the number of frames per segment
    frames_per_sec = 100  # Assuming 100 frames per second
    frames_per_segment = int(segment_length_sec * frames_per_sec)

    # Split the test audio into segments
    num_segments = math.ceil(len(test_mfccs) / frames_per_segment)
    test_segments = []
    for i in range(num_segments):
        start_frame = i * frames_per_segment
        end_frame = min(start_frame + frames_per_segment, len(test_mfccs))
        segment = test_mfccs[start_frame:end_frame]
        test_segments.append(segment)

    return test_segments
 ```
### Step 5: Identification

In this step, first we define the predict_speaker function which takes as input the mfcc as well as the GMM model to calculate the score of each in order to return the maximum score as well as the predict_speaker below the function:
 ```javascript
def predict_speaker(mfcc_features, gmm_models):
    highest_score = -float('inf')
    predicted_speaker = None

    # Calculate scores for each GMM model
    for model_name, gmm_model in gmm_models.items():
        score = gmm_model.score(mfcc_features)

        if score > highest_score:
            highest_score = score
            predicted_speaker = model_name.split(".")[0]

    return highest_score, predicted_speaker
 ```
Next, we load all the GMM models for men and women and store them in dictionaries based on n_components and gender:(key: model_name , value: gmm model)
and also we load files containing mfcc features for male and female and store them in dictionaries according to duration and gender: (key: model_name , value: mfcc_features)

We go to the speaker identification stage, this part of code, we apply the predict_speaker function for each segment, and store the results in a list of dictionaries. Each dictionary contains information such as the name of the test speaker, the number of segments, the maximum score obtained and the predicted speaker.
 ```javascript
results_3_H_128 = []

for test_segment_name, test_segment in test_files_3_H.items():
    mydict = {}
    mydict["maxScore"], mydict["IdentifiedSpeaker"] = predict_speaker(test_segment, gmm_models_H_128)
    mydict["TestSpeaker"] = test_segment_name.split(".")[0]  # Add the test speaker name to the dictionary
    mydict["NumSegments"] = test_segment_name.split(".")[2]  # Add the number of segments to the dictionary
    results_3_H_128.append(mydict)

for result in results_3_H_128:
    print("Test Speaker:", result["TestSpeaker"])
    print("Num Segments:", result["NumSegments"])
    print("MAX_Score:", result["maxScore"], "--- Predicted Speaker:", result["IdentifiedSpeaker"])
    print()


 ```
 After running the code described above, the following outcomes can be expected:
 
 ![image](https://github.com/Chai-mae/Automatic-Speaker-recognition/assets/86806466/0368ccc5-0785-4ba2-9d8f-bcebb1125110) ![image](https://github.com/Chai-mae/Automatic-Speaker-recognition/assets/86806466/5e704979-da4b-49b8-9a47-91b11cab3181)


We apply the block of code described above for each segment of each duration of 3s, 10s, 15s, 30s and with each model (128,256,512,1024) for the identification of male and female speakers
We calculate the false prediction rate in the speaker identification results for a specific set (`results_30_F_1024` for example). If the speakers differ, this indicates an incorrect prediction and the false prediction counter is incremented. Once all the predictions have been verified, the false prediction rate is calculated by dividing the number of false predictions by the total number of predictions made. This false prediction rate is then added to the `false_prediction_rate` dictionary with the following notation (`false_prediction_rate_duration_Gender_ncomponents`)
We extract the duration, genre and component information of the model from the keys of a `false_prediction_rate` dictionary. Those dictionaries are used to draw graphs that represent the rate of false predictions according to the duration of the segments, the gender of the speakers and the number of components of the model in different formats to be able to visualize the results well.
 ```javascript
false_prediction_rate = {}
false_predictions = 0
total_predictions = len(results_3_H_128)

for result in results_3_H_128:
    test_speaker = result["TestSpeaker"]
    identified_speaker = result["IdentifiedSpeaker"]
    
    if test_speaker != identified_speaker:
        false_predictions += 1

false_prediction_rate_3_H_128 = false_predictions / total_predictions
false_prediction_rate["false_prediction_rate_3_H_128"] = false_prediction_rate_3_H_128
false_prediction_rate_3_H_128

 ```
### Step 6: Verification

For this last step we first start by defining the function get_scores which calculates the score of a given test segment with a given gmm model:
 ```javascript
def get_scores(mfcc_features, gmm_models):
    scores = {}
    for model_name, gmm_model in gmm_models.items():
        scores[model_name.split(".")[0]] = gmm_model.score(mfcc_features)
    return scores
 ``` 
Then we go to load all the GMM models for men and women and store them in dictionaries as in the identification step, and we load also the files containing the Mfcc features for male and female and store them in dictionaries according to duration and gender ( key: model_name , value: mfcc_features ).

After we go to the step of obtaining the predictions of the segments of the test, we start with the men, we use the get_scores function to calculate the score of each segment with each GMM model and store them in a filename with score
```javascript
results_3_F_128 = []

# Calculate scores for each GMM model and store file name with score
for test_segment_name, test_segment in test_files_3_F.items():
    score = get_scores(test_segment, gmm_models_F_128)
    results_3_F_128.append((test_segment_name, score))

# Printing the results
for result in results_3_F_128:
    file_name, score = result
    print(f"File: {file_name} ,Score: {score}")
  ```
  After running the code described above, the following outcomes can be expected:
 
 ![image](https://github.com/Chai-mae/Automatic-Speaker-recognition/assets/86806466/d64cf960-c482-4aad-9876-ac854c8852bf)
 
After obtaining the scores, we sort them from the minimum score to the maximum then we generate a DET (Detection Error Tradeoff) curve and calculate the equalization error rate EER. We start by extracting the genuine partitions and the impostor partitions from a set of results. Then, we determine the minimum and maximum scores among all the scores to generate a certain number of thresholds using the linspace function. Then, for each threshold, we calculate the number of false rejections and false acceptances and then their rates. These rates are stored in lists and the FAR-FRR curve (DET) is plotted and the EER point is displayed with an annotation.

```javascript
# Initialize variables
genuine_scores = defaultdict(list)  # Dictionary to store genuine scores for each client
impostor_scores = []  # List to store impostor scores

# Extract genuine scores and impostor scores
for result in results_3_F_128:
    file_name, score = result
    client_name = file_name.split('.')[0]  # Extract client name from file name

    genuine_scores[client_name].append(score[client_name])

    impostor_scores.extend([s for key, s in score.items() if key != client_name])

# Compute the minimum and maximum scores
min_score = min(sorted_scores_array)
max_score = max(sorted_scores_array)

# Set the number of thresholds and generate them using linspace
num_thresholds = 1000
thresholds = np.geomspace(max_score, min_score, num_thresholds)

# Initialize lists for FAR and FRR
far = []
frr = []

# Iterate over thresholds
for threshold in thresholds:
    # Compute the number of false accepts (FAR) and false rejects (FRR)
    false_accepts = sum(score >= threshold for score in impostor_scores)
    false_rejects = 0
    for client_scores in genuine_scores.values():
        false_rejects += sum(score < threshold for score in client_scores)

    # Compute the FAR and FRR rates
    far_rate = false_accepts / len(impostor_scores)
    frr_rate = false_rejects / sum(len(client_scores) for client_scores in genuine_scores.values())

    # Append the FAR and FRR rates to the lists
    far.append(far_rate)
    frr.append(frr_rate)

# Find the threshold with the closest FAR and FRR
eer_threshold = thresholds[np.argmin(np.abs(np.array(far) - np.array(frr)))]

# Compute the EER values
eer_far = far[np.argmin(np.abs(np.array(far) - np.array(frr)))]
eer_frr = frr[np.argmin(np.abs(np.array(far) - np.array(frr)))]

# Plot the FAR and FRR curve
plt.plot(far, frr)
plt.xlabel('False Acceptance Rate (FAR)')
plt.ylabel('False Rejection Rate (FRR)')
plt.title('DET Curve')
plt.grid(True)

# Plot the EER point
plt.scatter(eer_far, eer_frr, color='red', marker='o', label='EER')

# Add legend
plt.legend()

# Add text annotation for EER point
plt.annotate(f'EER: ({eer_far:.3f}, {eer_frr:.3f})', (eer_far, eer_frr), xytext=(eer_far + 0.05, eer_frr), color='red')

plt.show()
```
**<h2>Obtained Results</h2>**


    
