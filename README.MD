# Moview Review Rating Prediction
Predicting star rating of a movie (from 1 to 10) based on the review contents.  
This project uses two regression models - distilbert and a bag-of-words lgbm model for comparison.

The inferface includes a flask front-end and a basic API.

POST /get_ratings expects json formated like {"method": "lgbm"|"bert"|"both", "reviews":["text1","text2"...]}  
returns json {"bert":{"ratings":[7,3],"labels":[1,0], "lgbm":{"ratings":[7,3],"labels":[1,0]}
