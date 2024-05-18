# nyc-yellow-taxi-analysis
Analyzing Urban Mobility Patterns Using NYC Yellow Taxi Trip Data, predicting Tip_Amount based on other factors


Milestone 1:
Proposal: Analyzing Urban Mobility Patterns Using NYC Yellow Taxi Trip Data

1. Description of the Data Set: For this project I chose to use the New York city Taxi Trip Records  directly from NYC TLC website. This data set offers a comprehensive view of urban mobility within NYC, capturing detailed information about each yellow  taxi trip. It includes temporal data points such as pickup and drop-off times, spatial dimensions involving locations, trip distance, fare composition, payment methods, and passenger counts. The dataset spans from 2009 to 2023, providing a rich chronicle of urban transit patterns over time.
2. URL/Location for Downloading the Data:  https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page



3. Data Set Attributes (Columns):
Total 19 columns:
 #   Column                 		Dtype         
---  ------              	  	 -----         
 0   VendorID               	int64         
 1   tpep_pickup_datetime       datetime64[ns]
 2   tpep_dropoff_datetime      datetime64[ns]
 3   passenger_count                float64       
 4   trip_distance           	 float64       
 5   RatecodeID            	 float64       
 6   store_and_fwd_flag           object        
 7   PULocationID           	 int64         
 8   DOLocationID                   int64         
 9   payment_type          	 int64         
 10  fare_amount            	float64       
 11  extra               	            float64       
 12  mta_tax                		float64       
 13  tip_amount             	float64       
 14  tolls_amount           	float64       
 15  improvement_surcharge  float64       
 16  total_amount           	float64       
 17  congestion_surcharge      float64       
 18  airport_fee            	object 
4.Intended Analysis:
Our goal is to build a predictive model to estimate the amount of tip left by passengers in NYC taxi rides. We plan to explore the following analyses:
a) Investigating factors influencing the tip amount, such as total fare, trip distance, time of day, and traffic conditions.
b) Identifying patterns and trends in tipping behavior, such as peak tipping hours/days and seasonal variations.
c) Assessing the impact of various features on the likelihood of a "good tip," defined as a tip amount exceeding 10% of the total fare.
d) Exploring the spatial patterns of tipping behavior by analyzing popular pickup/drop-off locations and transit hubs.
5.Model Selection:
For predicting the tip amount, a continuous variable, we will use linear regression. Additionally, we can explore logistic regression to predict whether the tip will be a "good tip" based on predefined criteria (e.g., tip amount exceeding 10% of the total fare).






Milestone 2 

Data Acquisition:

My project ID:  natural-aspect-415016

1. First I created an instance named “my-first-instance”  consisting of 40 GB disk space in the us-central1-c zone  .  I allowed for HTTP / HTTPS traffic and left everything in default mode . 

2. Opened the SSH in the new window 

3. I authorized my login with    “gcloud auth login” command. 

4. Created a bucket : 
gcloud storage buckets create gs://my-bigdata-project-md --project=natural-aspect-415016 --default-storage-class=STANDARD --location=us-central1 --uniform-bucket-level-access

5. Created folders named landing, cleaned, trusted, code and models
within a Google Cloud Storage  bucket by saving an empty placeholder file. 

gsutil -m cp -Z /dev/null gs://my-bigdata-project-md/landing/empty_placeholder
gsutil -m cp -Z /dev/null gs://my-bigdata-project-md/cleaned/empty_placeholder
gsutil -m cp -Z /dev/null gs://my-bigdata-project-md/trusted/empty_placeholder
gsutil -m cp -Z /dev/null gs://my-bigdata-project-md/code/empty_placeholder
gsutil -m cp -Z /dev/null gs://my-bigdata-project-md/models/empty_placeholder

6. Downloaded all the data and copied it in the landing folder of my bucket. I went to the official New York  TLC website https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page to upload documents. I wanted to analyze  Yellow Taxi Trip Records which started in 2009. It was very inconvenient to do it one by one, so I wrote the code using ‘for loop ‘ for both month and year:

for YEAR in 2015 2016 2017 2018 2019 2020 2021 2022 2023
do
  for MONTH in 01 02 03 04 05 06 07 08 09 10 11 12
  do
    curl -L -o yellow_tripdata_${YEAR}-${MONTH}.parquet https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_${YEAR}-${MONTH}.parquet
    gsutil cp yellow_tripdata_${YEAR}-${MONTH}.parquet gs://my-bigdata-project-md/landing/
  done
done



7. Deleted the empty placeholder so it will not be confusing. 



8. Screenshots 














Milestone 3 

Exploratory Data Analysis and Data Cleaning:
Created a Dataproc Cluster. I have an error saying GCP Private Google Access , so I went to the VPC network and edited the default VPC subnet and turned on GCP Private Google access. 

Opened the Jupyter notebook and created a new file under the GCP folder. 

I wrote python code for Exploratory Data Analysis which I included in Appendix B

There were 16 columns at the beginning and increased to 18 through the years , so it needs to be dropped , a lot of columns are not logically related to predicting tip amount and also needs to be dropped 

Data Cleaning : dropped unnecessary columns : 'VendorID', 'RatecodeID', 'store_and_fwd_flag', 'congestion_surcharge', 'airport_fee', 'payment_type'

Dropped null values

Renamed the columns 

Applied schema : 
        'tpep_pickup_datetime': 'datetime64[us]',
        'tpep_dropoff_datetime': 'datetime64[us]',
        'passenger_count': 'int64',
        'trip_distance': 'float64',
        'PULocationID': 'int64',
        'DOLocationID': 'int64',
        'fare_amount': 'float64',
        'extra': 'float64',
        'mta_tax': 'float64',
        'tip_amount': 'float64',
        'tolls_amount': 'float64',
        'improvement_surcharge': 'float64',
        'total_amount': 'float64'

Uploaded it to the ‘cleaned’ folder of the bucket








Milestone 4 


Created a cluster at Dataproc.
Opened Jupyter notebook through cluster and created Pyspark notebook.
Set Up the Environment with spark command and imported libraries : 

import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
# Import some modules we will need later on
from pyspark.sql.functions import col, isnan, when, count, udf, to_date, year, month, date_format, size, split, unix_timestamp
from pyspark.ml.stat import Correlation
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

Only used one month of data  , which is :  ‘cleaned/yellow_tripdata_2023-12.parquet’ to save time 
Loaded the data as Spark dataframe and printed the Schema 
root
 |-- Pick_up_time: timestamp_ntz (nullable = true)
 |-- Drop_off_time: timestamp_ntz (nullable = true)
 |-- Passenger_count: double (nullable = true)
 |-- Trip_distance: double (nullable = true)
 |-- Pick_up_location: long (nullable = true)
 |-- Drop_off_location: long (nullable = true)
 |-- Fare_amount: double (nullable = true)
 |-- Extra_charge: double (nullable = true)
 |-- Mta_tax: double (nullable = true)
 |-- Tip_amount: double (nullable = true)
 |-- Tolls_amount: double (nullable = true)
 |-- Improvement_surcharge: double (nullable = true)
 |-- Total_amount: double (nullable = true)
 |-- __index_level_0__: long (nullable = true)
Convert the date column to an actual date data type    :   'yyyy-MM-dd'
 Looked at statistics for some specific columns
Filter out tips greater than $25 and trips longer than 50 miles
Created Trip_duration column in minutes by subtracting Pick_up_time  from   Drop_off_time
Assembled features into a feature vector
Normalized features
Prepared  the data for modeling
Splitted the data TRAIN/TEST SPLIT
Trained and Evaluated the data
 Root Mean Squared Error (RMSE): 2.9832153745221146


Conclusion :  A lower RMSE value indicates a better fit of the model to the data. Conversely, a higher RMSE suggests that the model's predictions are less accurate. 

On average, the predictions made by the Linear Regression model differ from the actual tip amounts by approximately 2.98 units.

In this case, an RMSE of about 2.98 suggests that the model has a moderate level of accuracy in predicting tip amounts.






  

Milestone 5
VISUALIZATIONS:

Summary of the Correlation Matrix​Strong Correlation:​Fare Amount & Trip Distance: Longer trips have higher fares (0.83). Tolls Amount & Trip Distance: Longer trips incur higher tolls (0.64). Fare Amount & Tolls Amount: Higher fares often include higher tolls (0.61).
Moderate Correlation:​Tip Amount & Fare Amount: Higher fares generally lead to higher tips (0.58). Tip Amount & Trip Distance: Longer trips typically get higher tips (0.56).
Weak Correlation:​Extra Charge & MTA Tax: Minimal impact on tips, fares, or distances.
Conclusion: Fare amount and trip distance are key factors affecting tip amounts, while extra charges and MTA tax have minimal impact.



Summary of the Predicted vs. Actual Tip Amount Plot :
Underprediction: The model generally underpredicts tips, especially higher amounts.
Accurate for Low Tips: Predictions are accurate for low tips (0-5 dollars).
Decreasing Accuracy: Model accuracy decreases as tip amounts increase.
Outliers: Significant prediction errors indicate areas for improvement.




SUMMARY:
Higher Tips for Larger Groups: Groups of 7-9 passengers tend to tip more.
Negative Tips and Outliers: Many outliers, especially for single passengers, suggest data errors or service issues.
Consistent Tips for Smaller Groups: Tips for 1-4 passengers are stable, implying other factors affect tips more.


The residual plot shows:
Underestimation of Higher Tips: The model often underpredicts higher tips.
Overestimation of Lower Tips: The model tends to overpredict lower tips.
Outliers: Significant outliers indicate the model struggles with certain data points.








Milestone 6 Due 5/17/2024 (10 points)  
Summary and Conclusions:  Document the completed data processing pipeline and complete the project report with a summary of the project and the main conclusions you were able to draw from the data. Be sure to include citations for any code examples or other resources used. Include the GitHub URL for the shared project. 
Share the Project:  Post the project description and code on GitHub.  Include the URL  of your GitHub Project in the final report. Be sure to hide any security keys, passwords, credentials, etc. that may appear in your code. When you create your GitHub account, be sure to use your real name (or as close to it as possible) so that prospective employers can find you.
Submission by e-mail with Subject: CIS 4130 Project Milestone 6  attach file name: cis_4130_project_milestone_6_LastName_FirstName.pdf



Milestone 6 

The primary goal of this project was to build a predictive model to estimate the amount of tip left by passengers in NYC taxi rides.

 The project involved the following key steps:

Data Acquisition: Collected yellow taxi trip data from the NYC TLC website for the years 2015 to 2023 and stored it in Google Cloud Storage.
Data Cleaning: Removed unnecessary columns, handled missing values, and standardized the data types.
Exploratory Data Analysis (EDA): Conducted EDA to understand the data and identify key features influencing tip amounts.
Model Building: Developed a linear regression model to predict the tip amount based on selected features.
Visualization: Created visualizations to summarize the data, model predictions, and insights.

Key Steps and Findings:

Data Acquisition:
Collected a comprehensive dataset of NYC yellow taxi trips from 2015 to 2023.
Stored data in a structured format in Google Cloud Storage for easy access and processing.

Data Cleaning:
Dropped irrelevant columns such as VendorID, RatecodeID, store_and_fwd_flag, etc.
Removed rows with null values to ensure data quality.
Renamed columns for better readability and applied appropriate data types.

Exploratory Data Analysis (EDA):

Identified columns with strong correlations, such as fare amount and trip distance, which directly influence the tip amount.
Filtered out outliers to enhance model accuracy.

Model Building:

Created a feature vector and normalized the features.
Split the data into training and testing sets.
Trained a linear regression model and evaluated its performance using RMSE (Root Mean Squared Error).

Visualizations:

Correlation Matrix: Showed strong correlations between fare amount and trip distance, as well as moderate correlations between tip amount and fare amount.
Predicted vs. Actual Tip Amount Plot: Highlighted the model's accuracy for low tip amounts and areas for improvement for higher tips.
Box Plot of Tip Amount by Passenger Count: Revealed tipping patterns based on the number of passengers.
Residual Plot: Showed the model's underestimation of higher tips and overestimation of lower tips.

Conclusions
Model Performance:
The linear regression model achieved a moderate level of accuracy with an RMSE of approximately 3.00. This indicates that on average, the predictions made by the model differ from the actual tip amounts by about 3 dollars.

Key Insights:

Fare Amount and Trip Distance: These are significant predictors of tip amount. Longer trips and higher fares tend to result in higher tips.
Passenger Count: Groups with 7-9 passengers tend to tip more, while tips for 1-4 passengers are relatively stable.
Outliers and Residuals: The model struggles with accurately predicting very high or very low tips, suggesting areas for further improvement, such as incorporating additional features or using more complex models
