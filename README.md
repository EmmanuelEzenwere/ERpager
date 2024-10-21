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

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:3001/ or http://0.0.0.0:3001/



