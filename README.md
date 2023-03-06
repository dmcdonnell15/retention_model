# Retention Tiering

The goal of this project is to predict which students are at risk of not being retained to the subsequent term, and to target the highest risk students with additional academic supports. Each folder contains at least one of the following:
* SQL query to pull the data used in the model
* A Jupyter Notebook to build a model for a given term using prior terms as training data
* Model evaluation - need to add
* A saved pkl file of the final model used to create retention scores for the given term
* A Python script that pulls in current term data, runs it through the final model, assigns students to tiers, and loads the data into SQL for production 
* A csv snapshot of the data used to create the retention scores at a given point in time - need to add

#### To run the model:
1. Download the jupyter notebook for the appropriate term to create and edit the model
2. Download the python script to use the created model to run on current-term students and upload to SQL server for use in OpenBook
3. When term is complete and students from term have re-enrolled in subsequent term, use the jupyter notebook evaluation section to evaluate the results of your predictions
