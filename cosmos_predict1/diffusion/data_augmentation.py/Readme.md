# Side by Side documentation

## Depedencies

''' So we need to implement a simple yet effictive comparator for now.
Let us with say something as yolo then we will move up to attention vision tranformers
Step 1. Load the video 
step 2 Run Yolo on video 1 -> get the model embedding and prediction 
step 3 Run yolo on video 2 -> get the model embedding and prediction
Step 4 Run a cosine similarity on two embeddings and cross check with prediction

Test case: feed two different videos of lions, the prediction should be same but the cosine similarity would fail.
Question: will cosine similarity fail for generative videos if the video changes in later stage ?? Good or Bad??'''