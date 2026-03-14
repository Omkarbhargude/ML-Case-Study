import pandas as pd
import numpy as np
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#-----------------------------------------------------------------------------
#   Function Name  : LoadPreservedModel
#   Description    : It is used to load preserve on secondary
#   Parameter      : filename
#   Return         : model
#   Date           : 14/03/2026
#   Author         : Omkar Mahadev Bhargude
#-----------------------------------------------------------------------------
def LoadPreservedModel(FileName):
    loaded_model = joblib.load(FileName)
    print("Model successfully loaded")
    return loaded_model

#-----------------------------------------------------------------------------
#   Function Name  : PreserveModel
#   Description    : It is used to preserve on secondary
#   Parameter      : model, filename
#   Return         : None
#   Date           : 14/03/2026
#   Author         : Omkar Mahadev Bhargude
#-----------------------------------------------------------------------------

def PreserveModel(model, FileName):
    joblib.dump(model,FileName)
    print("Model preserved successfully with name : ",FileName)
    
#-----------------------------------------------------------------------------
#   Function Name  : TrainTitanicModel
#   Description    : It does split X,Y, training data, testing data  
#   Parameter      : df -> pandas dataframe
#   Return         : None
#   Date           : 14/03/2026
#   Author         : Omkar Mahadev Bhargude
#-----------------------------------------------------------------------------

def TrainTitanicModel(df):
    # split features and label
    X = df.drop("Survived", axis = 1)
    Y = df['Survived']

    print("Features : ")
    print(X.head())

    print("\nLabels : ")
    print(Y.head())

    print("Shape of X : ",X.shape)
    print("Shape of Y : ",Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=42)

    print("Shape of X_train : ",X_train.shape)
    print("Shape of X_test : ",X_test.shape)
    print("Shape of Y_train : ",Y_train.shape)
    print("Shape of Y_test : ",Y_test.shape)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train,Y_train)
    print("Model trained successfully")

    print("\nIntercept of model : ")
    print(model.intercept_)

    print("\nCoefficent of model : ")
    for feature, coefficent in zip(X.columns, model.coef_[0]):
        print(feature," : ",coefficent)

    PreserveModel(model,"marvelloustitanic.pkl")

    loaded_model = LoadPreservedModel("marvelloustitanic.pkl")

    Y_pred = loaded_model.predict(X_test)

    accuracy = accuracy_score(Y_pred,Y_test)

    print("Accuracy is : ",accuracy*100)

    cm = confusion_matrix(Y_pred,Y_test)
    print("Confusion matrix is : ")
    print(cm)

#-----------------------------------------------------------------------------
#   Function Name  : DisplayInfo
#   Description    : It displays the formated title  
#   Parameter      : title (str)
#   Return         : None
#   Date           : 14/03/2026
#   Author         : Omkar Mahadev Bhargude
#-----------------------------------------------------------------------------
def DisplayInfo(title):
    print("\n"+"="*70)
    print(title)
    print("="*70)

#-----------------------------------------------------------------------------
#   Function Name  : ShowData
#   Description    : It shows basic information about dataset
#   Parameter      : df
#                    df -> Pandas dataset object
#                    message
#                    message -> Heading text to display
#   Return         : None
#   Date           : 14/03/2026
#   Author         : Omkar Mahadev Bhargude
#-----------------------------------------------------------------------------

def ShowData(df, message):
    DisplayInfo(message)

    print("\nFirst 5 rows of dataset")
    print(df.head())

    print("\nShape of dataset")
    print(df.shape)

    print("\nColumn names ")
    print(df.columns.tolist())

    print("\nMissing values in each column")
    print(df.isnull().sum())

#-----------------------------------------------------------------------------
#   Function Name  : CleanTitanicData
#   Description    : It does preprocessing 
#                    It removes unnecessary columns
#                    It Handles missing value
#                    It converts text data to numeric format 
#                    It does encoding to the categorical columns
#   Parameter      : df -> Pandas dataframe
#   Return         : df -> Cleaned Pandas dataframe
#   Date           : 14/03/2026
#   Author         : Omkar Mahadev Bhargude
#-----------------------------------------------------------------------------

def CleanTitanicData(df):
    DisplayInfo("Step 2 : Original Data")
    print(df.head())

    # Remove unnecessary columns
    drop_columns = ["Passengerid","zero","Name","Cabin"]
    existing_columns = [col for col in drop_columns if col in df.columns]

    print("\n Columns to be dropped : ")
    print(existing_columns)

    print("Drop the unwanted columns : ")
    df = df.drop(columns=existing_columns)

    DisplayInfo("Step 2 : Data after column removal")
    print(df.head())

    # Handle age column
    if "Age" in df.columns:
        print("Age column before preprocessing : ")
        print(df['Age'].head(10))

        # coerce -> Invalid value gets converted as NaN 
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce") 

        age_median = df["Age"].median()

        # Replace missing values with median 
        df["Age"] = df["Age"].fillna(age_median)

        print("\nAge cloumn after preprocessing : ")
        print(df["Age"].head(10))

    # Handle fare column
    if "Fare" in df.columns:
        print("\nFare column before preprocessing : ")
        print(df["Fare"].head(10))

        df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce") 

        fare_median = df["Fare"].median()
        print("Median of Fare column is : ",fare_median)

        # Replace missing values with median 
        df["Fare"] = df["Fare"].fillna(fare_median)

        print("\nFare column preprocessing : ")
        print(df["Fare"].head(10))

    #Handle Embarked column
    if "Embarked" in df.columns:
        print("\nEmbarked column before preprocessing : ")
        print(df["Embarked"].head(10))
    
        # used to convert data from int to string
        df["Embarked"] = df["Embarked"].astype(str).str.strip()         

        # Remove missing values
        df["Embarked"] = df["Embarked"].replace(['nan','None',''],np.nan)

        # get most frequent value
        embarked_mode = df["Embarked"].mode()[0]
        print("Mode of Embarked column : ",embarked_mode)

        df["Embarked"] = df["Embarked"].fillna(embarked_mode)
        
        print("\nEmbarked column after preprocessing : ")
        print(df["Embarked"].head(10))

    # Handle Sex column
    if "Sex" in df.columns:
        print("\nSex column before preprocessing : ")
        print(df["Sex"].head(10))

        df["Sex"] = pd.to_numeric(df["Sex"], errors="coerce") 

        print("\nSex column after preprocessing : ")
        print(df["Sex"].head(10))

    DisplayInfo("Data Afte preprocessing")
    print(df.head())

    print("\nMissing values after preprocessing")
    print(df.isnull().sum())
    
    # Encode Embarked column
    df = pd.get_dummies(df,columns=["Embarked"],drop_first=True)

    print("\nData after encoding : ")
    print(df.head())

    print("Shape of dataset : ",df.shape)

    # convert boolean columns into integer
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    print("\nData after encoding : ")
    print(df.head())

    return df
#-----------------------------------------------------------------------------
#   Function Name  : TitanicLogistic
#   Description    : This is main pipeline controller. It loads the dataset, 
#                    show raw data. It preprocess & train the model
#   Parameter      : Datapath of dataset file
#   Return         : None
#   Date           : 14/03/2026
#   Author         : Omkar Mahadev Bhargude
#-----------------------------------------------------------------------------

def TitanicLogistic(DataPath):
    DisplayInfo("Step 1 : Loading the dataset")
    df = pd.read_csv(DataPath)

    ShowData(df, "Initial dataset")

    df = CleanTitanicData(df)

    TrainTitanicModel(df)

#-----------------------------------------------------------------------------
#   Function Name  : main
#   Description    : Starting point of the application
#   Parameter      : None
#   Return         : None
#   Date           : 14/03/2026
#   Author         : Omkar Mahadev Bhargude
#-----------------------------------------------------------------------------

def main():
    TitanicLogistic("TitanicDataset.csv")

if __name__ == "__main__":
    main()
