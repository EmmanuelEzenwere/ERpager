# Disaster Response Pipeline Project
Machine Learning Web Application to Extract, Transform and Load (ETL) Twitter Messages into an SQL database and classify messages into response categories for Disaster Response Organisations during Disasters.

<img src="assets/DisasterResponseDashboard.png" />
<img src="assets/Plots.png" />







### Getting Started:
1. **To run the python scripts.**
   
   Create a virtual environment to manage application dependencies.
   
    ```
	python3 -m venv myenv  
    ```
    ```
	source myenv/bin/activate
    ```
    ```
	pip install -r requirements.txt
    ```
    ```
	python app.py
	```








## Project Files Overview

- Running the Python scripts 
- Running the Web app, 
- Files in the repository.


#### Running the Python Scripts


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        ```
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        
        ```
    - To run ML pipeline that trains classifier and saves
        ```
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
        
        ```
        

#### Running the Web App



2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:3001/ or http://0.0.0.0:3001/


#### Files in the repository
data
- There is a process data script with a program to extract data from csv files and load into a pandas data frame, clean and transform the data into a dataset for a machine learning model to classify messages into the appropriate disaster response category.

script takes the file paths of the two datasets and database, cleans the datasets, and stores the clean data into a SQLite database in the specified database file path.

It merges the messages and categories datasets, splits the categories column into separate, clearly named columns, converts values to binary, and drops duplicates.


- This makes use of helper functions such as
- -  load_data
- - save_data


## The Model

- There is a train_classifier script with a program to fetch the cleaned data from an sqlite database, load into a pandas data frame and split into training and tests set for a machine learning model to classify messages into the appropriate disaster response category.

The script takes the database file path and model file path, creates and trains a classifier, and stores the classifier into a pickle file to the specified model file path.

The script uses a custom tokenize function using nltk to case normalize, lemmatize, and tokenize text. This function is used in the machine learning pipeline to vectorize and then apply TF-IDF to the text.

The script builds a pipeline that processes text and then performs multi-output classification on the 36 categories in the dataset. GridSearchCV is used to find the best parameters for the model.

The TF-IDF pipeline is only trained with the training data. The f1 score, precision and recall for the test set is outputted for each category.



The web app runs without errors and displays visualizations that describe the training data.




 The main page includes at least two visualizations using data from the SQLite database.



The web app successfully uses the trained model to input text and return classification results.




When a user inputs a message into the app, the app returns classification results for all 36 categories.





- - Tokenize
- -  Starating Verb Extractor
- - save model as a pickel file
- - build_model
- - 

workspace
Jupyter notebooks with functions in cells to perform little experiments before running main program scripts for ETL or ML pipeline.

tests
- Actively in development, unit testing scripts for future development support.
```
python -m tests/test_data_processing.py
python -m tests/test_train_classifier.py
```





Go into more detail about the dataset and your data cleaning and modeling process in your README file, add screenshots of your web app and model results.
Add more visualizations to the web app.
Based on the categories that the ML algorithm classifies text into, advise some organizations to connect to.
Customize the design of the web app.
Deploy the web app to a cloud service provider.
Improve the efficiency of the code in the ETL and ML pipeline.
This dataset is imbalanced (ie some labels like water have few examples). In your README, discuss how this imbalance, how that affects training the model, and your thoughts about emphasizing precision or recall for the various categories.

<div style="display: flex; justify-content: space-between;"/>
    <img src="assets/data summary 1.png" alt="Data Summary 1" style="width: 48%;" />
    <img src="assets/data summary 2.png" alt="Data Summary 2" style="width: 48%;"/>
<div />
