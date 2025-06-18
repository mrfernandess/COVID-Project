from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve

# Training Decision Trees with different parameters
def train_decision_trees(X_train, y_train):
    models = {}
    for criterion in ['gini']:
        for max_depth in [3, 5, 10, None]:
            model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            models[f'{criterion}_depth_{max_depth}'] = model
    
    # ID3 (similar to entropy)
    id3_model = DecisionTreeClassifier(criterion='entropy', splitter='best', random_state=42)
    id3_model.fit(X_train, y_train)
    models['ID3'] = id3_model
    
    # CART (similar to gini)
    cart_model = DecisionTreeClassifier(criterion='gini', splitter='best', random_state=42)
    cart_model.fit(X_train, y_train)
    models['CART'] = cart_model
    
    # Gain Ratio (approximated using entropy and balanced splits)
    gain_ratio_model = DecisionTreeClassifier(criterion='entropy', splitter='random', random_state=42)
    gain_ratio_model.fit(X_train, y_train)
    models['Gain_Ratio'] = gain_ratio_model

    return models

# Evaluating each model
def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        print(f'\nEvaluating Model: {name}')
        y_pred = model.predict(X_test)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualizing each of the trained trees
def visualize_all_trees(models, feature_names):
    for name, model in models.items():
        plt.figure(figsize=(15, 10))
        tree.plot_tree(model, feature_names=feature_names, class_names=['Return Home', 'Stay at Hospital'], filled=True)
        plt.title(f"Visualization of {name}")
        plt.show()

