# Disaster_Response_Upgrade

## 1. Project Overview
The Disaster Response Pipeline project is a web application designed to analyze and classify messages related to disasters. The goal of this project is to aid emergency responders in swiftly identifying the most pertinent messages during a disaster, thereby facilitating a quicker response and more effective assistance.

## 2. Project Components
The project consists of the following components:

### 2.1. Data Augmentation
- Loads the data from `disaster_messages.csv` and `disaster_categories.csv` datasets.
- Processes and cleans the data.
- Use **TextAttck** library to augment the data.
- Save it into 2 file `disaster_train.csv` and `disaster_test.csv`.
  - The original data is disaster_test.csv with a shape of (26216, 39).
  - The augmented data is disaster_train.csv with a shape of (129665, 39).

### 2.2. Machine Learning Pipeline
- Load the data from `disaster_train.csv` and `disaster_test.csv`.
- Saves the final model as a pickle file
- The model is used to categorise messages into different disaster-related categories.
- Machine Learning pipeline code can be found in **_models/building_model.ipynb_**
- Pipeline: $\newline$
  
![Pipeline](https://github.com/thangthewinner/Disaster_Response_Upgrade/blob/main/screenshots/ml_pipeline.png?raw=true) $\newline$

- Model evaluation: $\newline$
  
![Model Evaluation](https://github.com/thangthewinner/Disaster_Response_Upgrade/blob/main/screenshots/classification_report.png?raw=true) $\newline$

![Accuracy](https://github.com/thangthewinner/Disaster_Response_Upgrade/blob/main/screenshots/accuracy.png?raw=true) $\newline$


### 2.3. Web Application
- Using Flask web app
- Users can explore visualization of the dataset.

## 3. Getting Started

### 3.1. Dependencies
- **Python Version**: Python 3.11.4
- **Data Augmentation Libraries**:
    - Pandas
    - TextAttack
- **Machine Learning Libraries**:
    - Pandas
    - SciPy
    - Scikit-Learn
- **Natural Language Processing Libraries**:
    - NLTK (Natural Language Toolkit)
- **Model Loading and Saving Libraries**:
    - Pickle
- **Web Application and Data Visualization Libraries**: 
  - Flask
  - Plotly

### 3.2. Cloning the repository

To clone the git repository: 

    
        git clone https://github.com/thangthewinner/Disaster_Response_Upgrade.git

### 3.3. Running the project
1. See the way to augment data

- Open file `Data_Augmentation.ipynb` follow the path **\Disaster_Response_Upgrade\data\data_augmentation\Data_Augmentation.ipynb**

2. See the way the model has been built

- Open file `building_model.ipynb` follow the path **\Disaster_Response_Upgrade\models\building_model.ipynb**
- Evaluate the model:


3. Run the web app
- Run the following command in the app directory:
```bash
python run.py
```
- Go to `http://127.0.0.1:3000/` to use the web application.

## 4. Preview
- Front page: $\newline$
![Front Page](https://github.com/thangthewinner/Disaster_Response_Upgrade/blob/main/screenshots/front_page.png?raw=true)

- Result page: $\newline$
![Result Page](https://github.com/thangthewinner/Disaster_Response_Upgrade/blob/main/screenshots/result_page.png?raw=true)
