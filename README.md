# mlops_SmsSpam_Assignment2
SMS Spam Detection using ML, Flask, and Docker

1. [Docker Hub Repository]  Docker Hub Link
2. [Steps to Accomplish Tasks]
   - **(i)   Model Training Approaches**
   - **(ii)  HyperParameter Tuning and logging experiment using MLFlow**
   - **(iii) Creating API Endpoints using Flask (Flask REST API)**
   - **(iv)  Build Docker Image and RUN Container**
   - **(v)   PUSH to Docker Hub**
   - **(vi)  PULL and RUN FROM Docker Hub**
   - **(vii) Dockerized USE of API**

   

# 1. Docker Hub Link

**Docker Image:**  
https://hub.docker.com/r/vishwajeetyadav3597/sms_spam_api

```bash
# Pull the image
docker pull vishwajeetyadav3597/sms_spam_api:latest

# Run the container
docker run -p 5001:5001 vishwajeetyadav3597/sms_spam_api

/prediction
Request Type: POST
URL: http://127.0.0.1:5001/prediction

Sample Request Body:
{
  "text": "Congratulations! You've won a $1,000 gift card. Claim now: http:claimprize.com"
}

/training
Request Type: POST
URL: http://127.0.0.1:5001/training

Sample Request Body:
{
  "C": 0.4,
  "max_iter": 250,
  "solver": "liblinear"
}

/best_model_parameter
Request Type: GET
URL: http://127.0.0.1:5001/best_model_parameter

Request Body: Not required
```
********************************************************************


# 2. **Steps to Accomplish Tasks**
 
##  Email Spam Classifier with Flask API & Docker Deployment

### (i) Model Training Approaches

-   **Multinomial Naive Bayes** with `TfidfVectorizer`
-   **Logistic Regression** with `CountVectorizer`  *(Selected)*

**Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC  
**Selected Combination**: `LogisticRegression + CountVectorizer`  
**Key Files**: `logisticreg.py`, `main.py`
#### Screenshot - roc-auc Curve
![Screenshot to ROC CURVE](saved_models/roc.png)

------------------------------------------------------------------
### (ii) HyperParameter Tuning and logging experiment using MLFlow
------------------------------------------------------------------

![Screenshot to MLFlow](screenshots/mlflow_UI.png)




--------------------------------------------------
###  (iii) Creating API Endpoints using Flask (Flask REST API)
--------------------------------------------------
A lightweight RESTful API to classify SMS messages as **Spam** or **Not Spam**.

------------------------------------
END POINTS and how to USE
------------------------------------
### POST `/prediction`
- **Description**: Predict SMS spam/not spam  
- **Input**: JSON `{ "text": "your message" }`  
- **Response**: JSON `{ "prediction": "spam" | "not spam" }`



###  GET `/best_model_parameter`
- **Description**: Returns best parameters (from Grid/RandomizedSearchCV)  
- **Response**: JSON of hyperparameters


###  POST `/training`
- **Description**: Trains model using predefined dataset path  
- **Note**: Dataset path is hardcoded in script

- **Input**: JSON `{  "C": 0.4,  "max_iter": 250,  "solver": "liblinear"}`
- **Response**: JSON `{"message": "Model retrained using Logistic Regression."}`

---------------------------------------------
### (iv) Build Docker Image and RUN Container
--------------------------------------------

### BUILD IMAGE

```bash
- docker build -t dockerhub_username/sms_spam_api .
```
![Screenshot to Docker Image on Docker Desktop UI](screenshots/username-tag.png)

### RUN the CONTAINER from IMAGE
```bash
- docker run -p 5001:5001 dockerhub_username/sms_spam_api
```
![Screenshot to RUN container](screenshots/run-container.png)

---------------------------------------
### (v) PUSH to Docker Hub
--------------------------------------
```bash
- docker push dockerhub_username/sms_spam_api
```
![Screenshot to DockerHub](screenshots/dockerhub.png)

------------------------------------------
### (vi) PULL and RUN FROM Docker Hub
------------------------------------------
```bash
- docker pull dockerhub_username/sms_spam_api
- docker run -p 5001:5001 dockerhub_username/sms_spam_api
```
![Screenshot to PULL Image ](screenshots/pull-image.png)
![Screenshot to RUN container](screenshots/run-container.png)



##  Dockerized USE of API
  #### Screenshot to Predict
![Screenshot to prediction](screenshots/predict-api.png)
  #### Screenshot to Best HyperParameter
![Screenshot to Best Hyper parameters](screenshots/model-hyperparameters.png)
  #### Training
  Usage As mentioned in [Docker Hub Link] Serial No- 1 At the top



