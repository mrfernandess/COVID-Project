from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.utils.class_weight import compute_sample_weight
import tensorflow as tf

def train_neural_network(X, T):
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, T, test_size=0.2, random_state=42)
    
    model = MLPClassifier(
        hidden_layer_sizes=(200, 100),
        max_iter=4000,
        activation='tanh',
        solver='adam',
        alpha=0.0001,
        learning_rate='adaptive',
        early_stopping=True,
        batch_size=32
    )
    
    evaluate_with_cross_validation(model, Xtrain, ytrain)
    
    model.fit(Xtrain, ytrain)
    
    return model, Xtest, ytest


def evaluate_model(model, Xtest, ytest):
    #SE,SP,F1score, AUC, ROC, Confusion Matrix, Accuracy
    ypred = model.predict(Xtest)
    cm = confusion_matrix(ytest, ypred)
    TN, FP, FN, TP = cm.ravel()
    SE = TP / (TP + FN)
    SP = TN / (TN + FP)
    F1 = 2 * TP / (2 * TP + FP + FN)
    accuracy = (cm[0,0] + cm[1,1]) / np.sum(cm)
    print("Sensitivity: ", SE)
    print("Specificity: ", SP)
    print("F1 Score: ", F1)
    print("Accuracy: ", accuracy)
    print("Confusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # Plot ROC curve
    yprob = model.predict_proba(Xtest)[:, 1]
    fpr, tpr, _ = roc_curve(ytest, yprob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    
# Function for evaluation with cross-validation
def evaluate_with_cross_validation(model, X, y):
    f1_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
    acc_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    print("Average F1 Score in cross-validation:", np.mean(f1_scores))
    print("Average Accuracy in cross-validation:", np.mean(acc_scores))

    
def tune_hyperparameters(X, y):
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (150,), (100, 50), (150, 75), (200, 100)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [1e-5, 0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [1000, 2000, 3000, 4000, 5000],
        'early_stopping': [True],
        'batch_size': [32, 64, 128]
    }
    
    mlp = MLPClassifier(max_iter=4500)
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    
    print("Best parameters found:")
    print(grid_search.best_params_)
    
    return grid_search.best_estimator_

def train_cnn(X_tabular, X_img, y, img_shape=(21, 21, 1), epochs=50, batch_size=32):
    
    # Reshape image data to the required shape
    X_img = X_img.reshape(X_img.shape[0], *img_shape)
    
    # Split the data into training and testing sets
    X_train_tabular, X_test_tabular, X_train_img, X_test_img, y_train, y_test = train_test_split(X_tabular, X_img, y, test_size=0.2, random_state=42)
    
    # Define the CNN model for image data
    img_input = tf.keras.layers.Input(shape=img_shape)
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(img_input)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    img_output = tf.keras.layers.Dense(64, activation='relu')(x)
    
    # Define the model for tabular data
    tabular_input = tf.keras.layers.Input(shape=(X_tabular.shape[1],))
    y =tf.keras.layers.Dense(64, activation='relu')(tabular_input)
    y = tf.keras.layers.Dropout(0.5)(y)
    tabular_output = tf.keras.layers.Dense(64, activation='relu')(y)
    
    # Combine the outputs of the two models
    combined = tf.keras.layers.concatenate([img_output, tabular_output])
    z = tf.keras.layers.Dense(128, activation='relu')(combined)
    z = tf.keras.layers.Dropout(0.5)(z)
    final_output = tf.keras.layers.Dense(1, activation='sigmoid')(z)
    
    # Define the combined model
    model = tf.keras.models.Model(inputs=[img_input, tabular_input], outputs=final_output)
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    history = model.fit([X_train_img, X_train_tabular], y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
    return model, history, [X_test_img, X_test_tabular], y_test

def evaluate_model_cnn(model, Xtest, ytest):
    # Sensitivity, Specificity, F1 Score, AUC, ROC, Confusion Matrix, Accuracy
    yprob = model.predict(Xtest)
    ypred = (yprob > 0.5).astype(int)  # Convert probabilities to binary classes
    cm = confusion_matrix(ytest, ypred)
    TN, FP, FN, TP = cm.ravel()
    SE_0 = TN / (TN + FP)  # Sensitivity for class 0
    SP_0 = TP / (TP + FN)  # Specificity for class 0
    SE_1 = TP / (TP + FN)  # Sensitivity for class 1
    SP_1 = TN / (TN + FP)  # Specificity for class 1
    F1 = 2 * TP / (2 * TP + FP + FN)
    accuracy = (cm[0,0] + cm[1,1]) / np.sum(cm)
    
    print("Sensitivity and Specificity for each class:")
    print(f"Class 0 - Sensitivity: {SE_0:.2f}, Specificity: {SP_0:.2f}")
    print(f"Class 1 - Sensitivity: {SE_1:.2f}, Specificity: {SP_1:.2f}")
    print("F1 Score: ", F1)
    print("Accuracy: ", accuracy)
    print("Confusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(ytest, yprob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
