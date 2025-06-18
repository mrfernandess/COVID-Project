import projectoRedesNeuronais as prn
import projetoClustering as pcl
import projetoDecisionTrees as pdt

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

def load_data():
    # Load the COVID_numerics.csv file
    colunas = ["GENDER","AGE","MARITAL STATUS","VACINATION","RESPIRATION CLASS","HEART RATE","SYSTOLIC BLOOD PRESSURE","TEMPERATURE","TARGET"]
    df_numerics = pd.read_csv('COVID_numerics.csv', usecols=colunas)
    
    # Load the COVID_IMG.csv file without header
    df_img = pd.read_csv('COVID_IMG.csv', header=None)
    return df_numerics, df_img

def process_img_data(df_img):
    # Flatten the images (21x21 -> 441)
    X_img = df_img.values.reshape(df_img.shape[0], -1)
    return X_img

def add_rule(X):
    X['RULE'] = ((X["RESPIRATION CLASS"] >= 2) & (X["TEMPERATURE"] > 37.8)).astype(int)
    return X

def preprocess_data(df_numerics):
    
    df_numerics = add_rule(df_numerics)
    
    df_numerics.drop_duplicates(inplace=True)
    
    for col in df_numerics.columns:
        if df_numerics[col].dtype == 'object':
            df_numerics[col].fillna(df_numerics[col].mode()[0], inplace=True)
        else:
            df_numerics[col].fillna(df_numerics[col].mean(), inplace=True)
            
    continuous_columns = ["AGE", "HEART RATE", "SYSTOLIC BLOOD PRESSURE", "TEMPERATURE"]
        
    # Normalization
    scaler = MinMaxScaler()
    df_numerics[continuous_columns] = scaler.fit_transform(df_numerics[continuous_columns])
    
    # Correlation
    # Check the correlation between variables
    print("Correlation between variables") 
    corr = df_numerics.corr()
    sns.heatmap(corr, annot=True)
    plt.show()
        
    # Separate the target variable
    target = df_numerics["TARGET"]
    X = df_numerics.drop(columns=["TARGET"])
    
    return X, target

def feature_selection_Random_Forest(X, y):
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y) 
    
    # Feature importance
    feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
    feature_importances = feature_importances.sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title('Feature Importance Random Forest')
    plt.show()
    
    # Select important features
    selector = SelectFromModel(rf, threshold=-np.inf, prefit=True)
    X_selected = selector.transform(X)
    
    # Get the names of the selected features
    selected_features = X.columns[selector.get_support()]
    
    return selected_features

def feature_selection_anova(X, y):
   
    selector = SelectKBest(score_func=f_classif, k='all')
    X_selected = selector.fit_transform(X, y)
    
    scores = selector.scores_
    feature_scores = pd.Series(scores, index=X.columns).sort_values(ascending=False)
    
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_scores, y=feature_scores.index)
    plt.title('Feature Scores using ANOVA')
    plt.show()
    
    return feature_scores.index

def feature_selection_mutual_info(X, y):
    

    selector = SelectKBest(score_func=mutual_info_classif, k='all')
    X_selected = selector.fit_transform(X, y)
    
    scores = selector.scores_
    feature_scores = pd.Series(scores, index=X.columns).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_scores, y=feature_scores.index)
    plt.title('Feature Scores using Mutual Information')
    plt.show()
    
    return feature_scores.index

# Feature selection based on 3 feature selection models
# Select the most important features
def select_features(X, target, num_features=7, selected_features=None):
    if selected_features is None:
        selected_features_rf = feature_selection_Random_Forest(X, target)

        selected_features_anova = feature_selection_anova(X, target)

        selected_features_mutual_info = feature_selection_mutual_info(X, target)

        # Create a weighted global ranking
        feature_scores = {}

        # Assign scores based on position in each list
        for rank, feature in enumerate(selected_features_rf):
            feature_scores[feature] = feature_scores.get(feature, 0) + (len(selected_features_rf) - rank)

        for rank, feature in enumerate(selected_features_anova):
            feature_scores[feature] = feature_scores.get(feature, 0) + (len(selected_features_anova) - rank)

        for rank, feature in enumerate(selected_features_mutual_info):
            feature_scores[feature] = feature_scores.get(feature, 0) + (len(selected_features_mutual_info) - rank)

        # Sort features by accumulated score
        sorted_features = sorted(feature_scores.items(), key=lambda x: -x[1])
        top_features = [feature for feature, score in sorted_features[:num_features]]
    else:
        top_features = selected_features

    X_selected = X[top_features]

    return X_selected, top_features

def compare_rule_target(df_numerics):
    
    # Ensure the RULE column is present
    if 'RULE' not in df_numerics.columns:
        df_numerics = add_rule(df_numerics)
    
    # Compare RULE with TARGET
    comparison = df_numerics['RULE'] == df_numerics['TARGET']
    
    # Calculate the number of coincidences
    num_coincidences = comparison.sum()
    total_cases = len(df_numerics)
    
    print(f"Number of coincidences between RULE and TARGET: {num_coincidences} out of {total_cases} cases")
    print(f"Percentage of coincidences: {num_coincidences / total_cases * 100:.2f}%")
    

def main():
    
    df_numerics, df_img = load_data()
    
    compare_rule_target(df_numerics)
    
    X, target = preprocess_data(df_numerics)
    
    selected_features = ["AGE", "HEART RATE", "SYSTOLIC BLOOD PRESSURE", "TEMPERATURE", "VACINATION", "RESPIRATION CLASS", "GENDER"]
    # Based on these 3 feature selection methods, we can choose the most important features, we will combine a combination of these features
    X_selected, selected_features = select_features(X, target, num_features=7, selected_features=selected_features)
    
    print("Selected features:")
    print(selected_features)
  
    X_img = process_img_data(df_img)
    
    print("--------------------------------------------CLUSTERING--------------------------------------------")
    selected_features_with_target = X.copy()
    selected_features_with_target['TARGET'] = target
    
    pcl.clustering(X,selected_features_with_target)
    
    print("--------------------------------------------DECISION TREES--------------------------------------------")
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, target, test_size=0.2, random_state=42)
    models = pdt.train_decision_trees(X_train, y_train)
    pdt.evaluate_models(models, X_test, y_test)
    
    # Visualize all the trees
    pdt.visualize_all_trees(models, selected_features)

    print("--------------------------------------------NEURAL NETWORKS--------------------------------------------")
    
    # Tune hyperparameters
    #X_train, X_test, y_train, y_test = train_test_split(X_selected, target, test_size=0.2, random_state=42)
    #best_model = trn.tune_hyperparameters(X_train, y_train)
    
    model, Xtest, ytest = prn.train_neural_network(X_selected, target)
    prn.evaluate_model(model, Xtest, ytest)
    
    print("--------------------------------------------CNN--------------------------------------------")
    cnn_model, cnn_history, x_test, y_test = prn.train_cnn(X_selected, X_img, target)
    prn.evaluate_model_cnn(cnn_model, x_test, y_test)
    

if __name__ == '__main__':
    main()