# glowing-guacamole
This is a repository for ICSI531F20 Project for Team3

## Install
First make sure you have [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and [python3](https://www.python.org/downloads/) installed
Run the following command in your terminal:
```bash
git --version
python3 --version
```

To install the repository run the follwing commands in your terminal:
```bash
git clone https://github.com/JerrisonChang/glowing-guacamole.git # Clones the repository to your local machine
cd glowing-guacamole # Moves you into the repository
pip3 install --upgrade pip # Upgrades pip to its latest version
pip install -r requirements.txt # Will install all of the required packages to run the code
```

## Dataset
Our [Dataset](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=76&access_redir=1) is supplied by K.R Koedinger et. al.
The dataset is from a data repository for the EDM community: The PSLC DataShop.
This dataset has data from the area unit of the Geometry Cognitive Tutor course as it was used during the school year 1996-1997.  

## Bayesian Knowledge Tracing (BKT)

## Deep Knowledge Tracing (DKT)
Performs Deep Knowledge Tracing on out data set using both simple RNN and LSTM networks.
deep knowldge tracing model was based off of [Lucas C. Casagrande's](https://github.com/lccasagrande) [implementation](https://github.com/lccasagrande/Deep-Knowledge-Tracing) of Deep Knoweledge Tracing.

### data_util.py
Configures the data from out dataset into a tensor of shape (student, question, knoweledge component).
Generates tensorflow dataset and splits data set into a traing set (80%) and a testing set (20%).

### deepkt.py
The Deep Knowledge Tracing Model

### metrics.py
The metrics to evaluate our models with

### Run
To run the deep knowldge tracing model, run the following command in your terminal:
```bash
cd DKT
python3 run_dkt.py
```
The code generates 15 diffrent Simple RNN and LSTM networks trained on data from the dataset.
Each model is trained on a diffrent number of knowldge components (KC) ranging from 1 to 15.

### Output
The output will be four files:
1. lstm_roc.csv - predicted values vs actual values from the testing set from the LSTM model trained on all KCs
2. rnn_roc.csv - predicted values vs actual values from the testing set from the RNN model trained on all KCs
3. lstm_accuracy.csv - accuracy of each LSTM model
4. rnn_accuracy.csv - accuracy of each RNN model

## Knowledge Tracing Machine (KTM) 
Performs knowldge tracing using factorization machines
Code is from [Jill-Jênn Vie's](https://github.com/jilljenn) [implementation](https://github.com/jilljenn/ktm) with slight modifications to perform knowldge tracing on our dataset.

### Install
To make the factorization machine libary run:
```bash
cd KTM
make libfm
```

### Perpare Data
To perpare the data for use by the factorization machine run the following:
```bash
cd data/geometry
python3 config_data.py # Outputs data.csv and needed.csv
cd ../..
```
data.csv has the following format:
```bash
user, item, skill, correct, wins, fails
```
needed.csv has the following format:
```bash
user_id, item_id, correct
```

### Run
To encode the data into sparse features run:
```bash
python3 encode.py --dataset geometry [--users] [--items] [--skills] [--wins] [--fails]  # To get the encodings (npz)
```
The options "--users", "--items", etc. denote the features to encode for the Q-matrix.
For example, if we wanted to encode users and items we would run:
```bash
python3 encode.py --dataset geometry --users --items  # Outputs Q-matrix file called X-ui.npz
```
And if we wanted to encode skills, wins, and fails we would run:
```bash
python3 encode.py --dataset geometry --skills --wins --fails  # Outputs Q-matrix file called X-swf.npz
```
Note the change in the Q-matrix file name to reflect the encoded features

To encode the time windows run:
```bash
python3 encode_tw.py --dataset geometry --tw # Will encode DAS3H sparse features into X.npz
```
This will also generate folds for the time windows which will be used in the MIRT and IRT models

To run the logistic regression model (d = 0), run the following:
```bash
python3 lr.py data/geometry/X-swf.npz
```

And to run the factorization machines (d > 0), run:
```bash
python3 fm.py --d 5 data/geometry/X-swf.npz # --d is the number of dimensions to run the factorization machine with
```
The above command will run the factorization machine on a Q-matrix with encodeings on skills, wins, and fails up to 5 dimensions

To run the MIRT model:
```bash
python3 omirt.py --d 0 data/geometry/needed.csv  # Will load LR: coef0.npy
python3 omirt.py --d 5 data/geometry/needed.csv  # Will load FM: w.npy and V.npy
```
The above two examples show the MIRT model will load diffrent parameters depending on the number of dimensions given
When d = 0 the MIRT model will load the logistics regression model
And when d > 0 it will load the factorization machine model

To run the IRT model, run:
```bash
python3 dmirt.py data/assist09/needed.csv
```
