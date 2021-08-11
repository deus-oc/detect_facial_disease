## Detecting Visually Observable Disease Symptoms from Faces
- We adopted semi-supervised anomaly detection combining with computer vision features extracted  from normal faces datasets to produce a reliable mechanism of detecting and classifying abnormal  symptoms that are visually observable from faces.
- The process of the detection was implemented via the process mentioned in the research paper Detecting Visually Observable Disease Symptoms from Faces by Kuan Wang and Jiebo Luo.
- The training dataset is composed of around 8000 pictures of normal frontal face images with diverse ethnicities, we further collected around 180 pictures of faces with symptoms paired with 180  pictures randomly picked from normal face datasets as our testing dataset.
- We used active shape models (ASMs) to label this dataset via library DLIB.



## Variants Selection and Extraction

The Variants were selected according to the paper which claimed to
be verified via certified doctor and practitioners.
>α represents aggregate value of the CIELAB alpha channel (red-green channel) of the feature

<p align="center">
  <img src="https://user-images.githubusercontent.com/43948081/129056377-46d59380-4c0d-422b-b206-d7b2bfed3f5f.png">
</p>

We took the aggregate value of the second matrix in the CIELAB format of the picture of the format.
> β represents the aggregate value of the CIELAB beta channel (yellow-blue channel) of the interested feature

We took the aggregate value of the third matrix in the CIELAB format of the picture of the format.

>Σ represents the total count of all the pixels belonging to the feature
-The image obtained of various keypoints is used and their sizes gave the result.

<p align="center">
  <img src="https://user-images.githubusercontent.com/43948081/129056393-414d36a1-2b44-4c12-8021-c56bbac63253.png">
</p>



For right face and left face, we just use the midline going through the nose and seperating the image to obtain the pixels

<p align="center">
  <img src="https://user-images.githubusercontent.com/43948081/129058638-0a862880-4e8e-4647-b9f9-63610c46c070.png">
</p>

>H is the process of applying the well-known Hough Transform on the CIELAB feature of the skin area,

and then further applying a counting function to count how many circular structures we found, the  mechanism is based on Size Invariant Circle Detection

<p align="center">
  <img width="200" height="200" src="https://user-images.githubusercontent.com/43948081/129056446-51e4dc79-7aba-439d-93cf-7f989bfa90bf.png">
</p>



## Results

After calcualtion of the variants given below in the table, we used around 8000 images of the training data to calculate the μ and δ of the data of the assumed data taken in normal distribution.

|Index | Variant | μ | δ |
|---|--------|------|------|
|1| α(Eye)/Σ(Eye)| 133.527| 5.1584|
|2 |β(Eye)/Σ(Eye) |132.528 |7.104|
|3 |α(Lip)/Σ(Lip)| 152.29| 11.581|
|4 |Σ(LFace)/ Σ(RFace)| 1.338| 0.2275|
|5 |Σ(LEye)/ Σ(REye)| 1.012| 0.274|
|6 |H(Face)| 2.871| 3.35|

An outlier is hence defined as a variant whose value is not in <b>μ ± t × δ</b>, where t is the multipler we used  to tighten the degree of normality. We applied the threshold μ ± t × δ on our observations with assumed distribution function and eventually divided the testing dataset into flagged group and unflagged group with respect to different t values.


## Install Dependencies via
Run these before installing python libraries
>sudo apt-get install build-essential cmake
sudo apt-get install libgtk-3-dev
sudo apt-get install libboost-all-dev


`$ sudo apt install python3-pip`
`$ pip3 install numpy`
`$ pip3 install opencv-python`
`$ pip3 install Shapely`
`$ pip3 install imutils`
`$ pip3 install dlib`

Use this link to download the required .DAT file for dlib
https://raw.githubusercontent.com/italojs/facial-landmarks-recognition/master/shape_predictor_68_face_landmarks.dat



## Steps to Run Script
1.Run final_test.py and use space bar to capture image and then Esc to confirm the captured image, 
2.The script will run and find the variants, and tell the disease which could be present according to the model


## Reference Paper
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5007273/
