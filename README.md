# SolarSense

# Introduction
SolarSense forecasts 24-hour solar X-ray flux (GOES XRS) and flare class.
It supports two model families (scikit-learn Gradient Boosting and a PyTorch LSTM), a FastAPI backend that orchestrates data fetch + prediction, and a React + Tailwind frontend UI.

---

# Pipeline: 
        -> fetch.py (run 3 times for data on training, seed, and verify actual)
        -> train_(pytorch/sklearn).py 
        -> predict_(pytorch/sklearn).py 
        -> OPTIONAL: process_prediction.py
        -> test_prediction.py
        -> graph_(prediction/pytorch/sklearn).py

Be sure to check out the comments of individual files, as we run them, to see 
any specfic changes you can make to customize your prediction experience.

---

# Tech Stack

## Backend

Python 3.10+ 

pandas, numpy, requests, joblib, matplotlib

scikit-learn (HistGradientBoostingRegressor)

pytorch (LSTM)

sunpy

fastapi, uvicorn

![](https://skillicons.dev/icons?i=python,numpy,pandas,scikit-learn,pytorch)

![](https://skillicons.dev/icons?i=git,github)

![](https://skillicons.dev/icons?i=linux,windows,mac)


## Frontend

React + TailwindCSS (webpack dev server)

![](https://skillicons.dev/icons?i=html,css,javascript,react,tailwind)

---

# Initial Setup

Be sure to download sunpy on your operating system of choice, and setup any additional python libraries while within the sunpy virtual environment.

Setup steps can be found here for Sunpy:
https://docs.sunpy.org/en/stable/tutorial/index.html 

You can enter the sunpy venv using the following once you have set up your environment:
```conda activate sunpy     ```

---

# Directory Structure 

The directory Structure is as follows:
```
SolarSense:.
│   .gitignore
│   README.md
│
├───data
│   │
│   │
│   └───processed
│           actual.csv
│           forecast_pytorch_2025_08_02-2025_08_02.csv
│           forecast_sklearn_2025_08_02-2025_08_02.csv
│           prediction_seed.csv
│           training_data.csv
│
├───models
│   └───regressors
│           flux_hgbr_1step.pkl
│           flux_lstm_60step.pth
│           x_scaler_1step.pkl
│           x_scaler_60step.pkl
│           y_scaler_1step.pkl
│           y_scaler_60step.pkl
│
├───react-tailwind-vanilla
│   │   .babelrc
│   │   package-lock.json
│   │   package.json
│   │   postcss.config.js
│   │   tailwind.config.js
│   │   webpack.config.js
│   │
│   ├───public
│   │       index.html
│   │
│   └───src
│           index.css
│           index.js
│
├───scripts
│   │   .DS_Store
│   │
│   ├───backend
│   │   │   api.py
│   │   │   data_fetch.py
│   │   │   paths.py
│   │   │   predict.py
│   │   │
│   │   └───pycache
│   │           api.cpython-312.pyc
│   │           data_fetch.cpython-312.pyc
│   │           paths.cpython-312.pyc
│   │           predict.cpython-312.pyc
│   │
│   ├───collect
│   │       fetch.py
│   │
│   └───model
│           predict_pytorch.py
│           predict_sklearn.py
│           train_pytorch.py
│           train_sklearn.py
│
├───test
│       process_prediction.py
│       test_prediction.py
│
└───visuals
        forecast_plot.png
        forecast_validation_pytorch.png
        forecast_validation_sklearn.png
        graph_prediction.py
        graph_pytorch.py
        graph_sklearn.py
```
---

# Running the pipeline

Be sure to check out the comments of individual files, as we run them, to see 
any specfic changes you can make to customize your prediction experience.

In General:

All pipeline commands can be run with 
```
python (dir location from above) file_name
```
According to the pipeline shown above, the entire process can be run as follows:
1.         -> fetch.py (run 3 times for data on training, seed, and verify actual)
              -> training data is usually long term used in step 2 (1-2 weeks)
              -> seed data is what step 3 uses for prediction (usually 1 day)
              -> actual is the file used to verify the prediction (eg. 24 hours)
2.         -> train_(pytorch/sklearn).py 
              -> sklearn is basic version that uses HistGradientBoostingRegressor with a 720 (12 hour window)
              -> pytorch is a more advanced version that uses LSTM with a 1440 (24 hour window)
3.         -> predict_(pytorch/sklearn).py 
              -> both versions take in the prediction_seed.csv and predict
              -> To change how long to predict, the HORIZON value can be changed.
4.         -> OPTIONAL: process_prediction.py
              -> simply averages each hours minutes into one value for faster validation
              -> preferable to skip this step unless u want immediate but less accurate validation
5.         -> test_prediction.py
              -> compares forcasted prediction with actual csv. fetched earlier. 
6.         -> graph_(prediction/pytorch/sklearn).py
              -> graphs both forcasted prediction and actual csv on same graph to compare.

Sample Pipeline running (while in main SolarSense Directory):
```

1. python scripts/collect/fetch.py -> fetch training data, changing the right lines according to comments in file (change start/end date, and output file name)

2. python scripts/collect/fetch.py -> fetch seed data

3. python scripts/collect/fetch.py -> fetch actual data

4. 
python scripts/model/train_sklearn.py  
OR 
python scripts/model/train_pytorch.py  

5. 
python scripts/model/predict_sklearn.py  
OR 
python scripts/model/predict_pytorch.py  

OPTIONAL 6 STEP
6. python test/process_prediction.py  

7. python test/test_prediction.py  

8. 
python visuals/graph_sklearn.py  
OR 
python visuals/graph_pytorch.py  

```
---

# Running the Web Server Pipeline (Backend and Frontend)

Note that the web server is a portion of the main pipeline, that is only used to run the entire above pipeline automatically using a frotend Web Interface.

By pressing the left and right arrow keys on the frontend, users can automatically run the pipeline, though there may initially be fetch errors as the pipeline takes some time to perform its action.

Ensure the backend is running before starting the frontend.

Backend:

The Backend uses FastAPI to simply perform some of the above pipeline.
It perform the sklearn version, then aggregates to hourly and returns.
The results are graphed, and users can customize what day they want to see.
Check out the api.py file in scripts/backend for more details.

While in scripts/backend directory:

```
uvicorn api:app --reload --port 8000
```


Frontend:

Simply click the left and right arrow keys to automatically run the pytorch version of the pipeline, results are returned and values are updated automatically in roughly 1-2 minutes.
There is also a dark/light mode feature and full mobile responsiveness.
Check out the index.js file in react-tailwind-vanilla/src for more details.

While in react-tailwind-vanilla:
```
npm install # if node modules is not installed
npm start # start the front end
```

You can access the frontend locally via : http://localhost:8080/


--- 

Thank you
