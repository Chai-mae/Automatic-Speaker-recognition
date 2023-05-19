# Automatic-Speaker-recognition
**<h2>Table of contents</h2>**

   [Introduction:](#Introduction)
   
   [Project Description:](#1-Project-Description)
   
   [General structure of the project](#2-General-structure-of-the-project)

   [Conclusion](#Conclusion)
   
**<h2>Introduction:</h2>**
Automatic speaker recognition is a part of speech analysis, it is the set of techniques and algorithms allowing the automatic identification and authentication of individuals based on their unique speech characteristics, such as their patterns voices, their speech patterns and other acoustic characteristics. There are two types of recognition:

• Text-dependent recognition in this case the algorithm is trained by pre-established sentences and said by all the speakers to be recognized

• Recognition independent of the text in this case there is no training following one or more specific sentences
The majority of the solutions developed for speaker recognition aim either to identify the speaker, i.e. to recognize the person who spoke among a group of speakers, or else verification or authentication which consists of checking with a minimal level of doubt that a person is the one who recorded their voice during verification.

**<h2>Project Description:</h2>**

As part of this project, we aim to build a speaker recognition system especially for students in our class. The objective being the identification of the student at first and the verification of the latter thereafter
The steps followed to carry out this project are as follows:

•Step 1: Gather all the audio recordings and build the dataset

• Step 2: Reading of recordings, extraction of MFCCs and pre-processing.    

• Step 3: Train the GMM models for each speaker

• Step 4: Divide the Test data set into 3s,10s,15s and 30s segments

• Step 5: Identification

• Step 6: Verification
    
In the rest of this report, we will detail each step of the approach followed.

**<h2>General structure of the project</h2>**

### Step 1 :
• Dataset
Our dataset is formed by a set of audio recordings made by each student in our class. Indeed each of us was responsible for recording two audios of 1 min one for the train and one for the test and depositing them in the drive as well as mentioning his name in the dedicated Excel file to assign each student his identifier. We have two files Train and Test each contains sub-files (F) and (H) in which the records are stored.
### 2nd step :
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
### Step 5: Identification
In this step, first we define the predict_speaker function which takes as input the mfcc as well as the GMM model to calculate the score of each in order to return the maximum score as well as the predict_speaker above the function:
Next, we load all the GMM models for men and women and store them in dictionaries based on n_components and gender:key: model_name , value: gmm model
and also files containing mfcc features for male and female and store them in dictionaries according to duration and gender:key: model_name , value: mfcc_features
We go to the speaker identification stage, this part of code, we apply the predict_speaker function for each segment, and store the results in a list of dictionaries. Each dictionary contains information such as the name of the test speaker, the number of segments, the maximum score obtained and the predicted speaker.
We apply this block of code for each segment of each duration of 3s, 10s, 15s, 30s and with each model (128,256,512,1024) for the identification of male and female speakers
We calculate the false prediction rate in the speaker identification results for a specific set (`results_30_F_1024`). If the speakers differ, this indicates an incorrect prediction and the false prediction counter is incremented. Once all the predictions have been verified, the false prediction rate is calculated by dividing the number of false predictions by the total number of predictions made. This false prediction rate is then added to the `false_prediction_rate` dictionary with the following notation (`false_prediction_rate_30_F_1024`)
We extract the duration, genre and component information of the model from the keys of a `false_prediction_rate` dictionary. They are used to draw graphs that represent the rate of false predictions according to the duration of the segments, the gender of the speakers and the number of components of the model in different formats to be able to visualize the results well.
This code is applied for each segment of the test and with each model
### Step 6: Verification
For this last step we first start by defining the function get_scores which calculates the score
Then we go to load all the GMM models for men and women and store them in dictionaries according to n_components and gender:key: model_name , value: gmm model
and also files containing Mfcc features for male and female and store them in dictionaries according to duration and gender:key: model_name , value: mfcc_features
as in the identification step. After we go to the step of obtaining the predictions of the segments of the test, we start with the men, we use the get_scores function to calculate the score of each segment with each GMM model and store them in a filename with score
After obtaining the scores on the ordinates from the minimum score to the maximum and on pass to generate a DET (Detection Error Tradeoff) curve and calculate the equalization error rate EER. We start by extracting the genuine partitions and the impostor partitions from a set of results. Then, we determine the minimum and maximum scores among all the scores to generate a certain number of thresholds using the linspace function. Then, for each threshold, we calculate the number of false rejections and false acceptances and then their rates. These rates are stored in lists and the FAR-FRR curve (DET) is plotted and the EER point is displayed with an annotation.

    
