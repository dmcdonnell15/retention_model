# Retention Tiering
The goal of this project is to predict which students are at risk of not being retained to the subsequent term, and to target the highest risk students with additional academic supports. Each folder contains at least one of the following:
* SQL query to pull the data used in the model
* A requirements.txt file that contains all the package dependencies needed to run the Jupyter Notebook
* A Jupyter Notebook to build a model for a given term using prior terms as training data
    * Notebook includes model evaluation section to be run after term ends
* A saved pkl file of the final model used to create retention scores for the given term
* A Python script that pulls in current term data, runs it through the final model, assigns students to tiers, and loads the data into SQL for production 
* An evaluation folder with snapshots of the data used to create the retention scores at a given point in time, as well as training data evaluation plots

#### To run the model:
1. Download the jupyter notebook, the requirements.txt file, and SQL query for the appropriate term to create and edit the model
2. Use pip install -r requirements.txt to download the dependencies for the notebook, and open the Jupyter Notebook and run the cells to get the training data from SQL, explore the data, and create the model. The evaluation section can be run later (see below)
3. Download the .py python script to use the created model to run on current-term students and upload list of students and their retention scores to SQL server for use in OpenBook production server
4. When term is complete and students from term have re-enrolled in subsequent term, use the Jupyter Notebook evaluation section to evaluate the results of your predictions