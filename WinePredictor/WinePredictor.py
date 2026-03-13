import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

def WineClassifier(DataPath):
    border = "-"*50
    
    # Step 1 : Load the Dataset from CSV file
    print(border)
    print("Step 1 : Load the Dataset from CSV file")
    print(border)

    df = pd.read_csv(DataPath)

    print(border)
    print("Some entries from dataset : ")
    print(df.head())
    print(border)

    # Step 2 : Clean the dataset by removing empty rows
    print(border)
    print("Step 2 : Clean the dataset by removing empty rows")
    print(border)

    df.dropna(inplace=True)        # empty rows will be removed
    print("Total records : ",df.shape[0])
    print("Total columns : ",df.shape[1])
    print(border)

    # Step 3 : Seperate independent and dependent variables 
    print(border)
    print("Step 3 : Seperate independent and dependent variables")
    print(border)

    X = df.drop(columns=['Class'])
    Y = df['Class']

    print("Shape of Independent variable : ",X.shape)
    print("Shape of Dependent varibales : ",Y.shape)

    print(border)
    print("Input columns : ",X.columns.to_list())
    print("Output column : Class")
    
    # Step 4 : Split the dataset for training and testing  
    print(border)
    print("Step 4 : Split the dataset for training and testing ")
    print(border)

    X_train, X_test_, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)

    print(border)
    print("Information of training and testing data : ")
    print("X_train shape : ",X_train.shape)
    print("X_test shape : ",X_test_.shape)
    print("Y_train shape : ",Y_train.shape)
    print("Y_test shape : ",Y_test.shape)
    print(border)

    # Step 5 : Feature Scaling 
    print(border)
    print("Step 5 : Feature Scaling")
    print(border)

    scalar = StandardScaler()

    # Independent variable scaling
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.fit_transform(X_test_)

    print("Feature scaling is done")

    # Step 6 : Explore the multiple values of k
    # Hyperpameter tuning (K)

    accuracy_scores = []
    k_values = range(1,21)

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_scaled,Y_train)
        Y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(Y_test,Y_pred)
        accuracy_scores.append(accuracy)

    print(border)
    print("Accuracy report of all k values from 1 to 20 are : ")
    for value in accuracy_scores:
        print(value)

    print(border)

    # Step 7 : Plot graph of K vs Accuracy 
    print(border)
    print("Step  7 : Plot graph of K vs Accuracy")
    print(border)

    plt.figure(figsize=(8,5))
    plt.plot(k_values, accuracy_scores, marker = 'o')
    plt.title("K values VS Accuracy")
    plt.xlabel("Value of K")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.xticks(list(k_values))
    plt.show()

    # Step 8 : Find best value of K
    print(border)
    print(" Step 8 : Find best value of K")
    print(border)

    best_k = list(k_values)[accuracy_scores.index(max(accuracy_scores))]
    print("best value of k is : ",best_k)

    # Step 9 : Build final model using best value of k
    print(border)
    print("Step 9 : Build final model using best value of k")
    print(border)

    final_model = KNeighborsClassifier(n_neighbors=best_k)
    final_model.fit(X_train_scaled,Y_train)
    Y_pred = final_model.predict(X_test_scaled)

    # Step 10 : Accuracy calculation
    print(border)
    print("Step 10 : Calculate final accuracy")
    print(border)

    accuracy = accuracy_score(Y_test,Y_pred)
    print("Accuracy of model is : ",accuracy*100)

    # Step 11 : Display confusion matrix
    print(border)
    print("Step 11 : Display confusion matrix")
    print(border)

    cm = confusion_matrix(Y_test,Y_pred)
    print(cm)

    # Step 11 : Display classification report 
    print(border)
    print("Step 12 : Display classification report")
    print(border)

    print(classification_report(Y_test,Y_pred))

def main():
    border = "-"*50

    print(border)
    print("Wine Classifier Using KNN")
    print(border)

    WineClassifier("WinePredictor.csv")

if __name__ == "__main__":
    main()  
