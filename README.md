# mlops_SmsSpam_Assignment2
Sms Spam Detection

## DockerHub Link

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

