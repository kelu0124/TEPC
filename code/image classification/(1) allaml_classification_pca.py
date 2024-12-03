import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def main():
    dataset = 'allaml'

    # Load data
    data_feature = np.load(r'/public/home/chenlong666/desktop/my_desk1/coil_20/origin_data/ALLAML.npy')
    print("Original data shape:", data_feature.shape)

    # PCA for dimensionality reduction
    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(data_feature)
    X = np.array(pca_data)
    print('PCA dimensionality reduction completed')

    # Load labels
    path_label = r'/public/home/chenlong666/desktop/my_desk1/coil_20/origin_data/ALLAML_labels.npy'
    data_label = np.load(path_label)
    print("Size of data labels:", data_label.shape)

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, data_label, test_size=0.3, random_state=42)

    # Train Random Forest classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)

    # Predictions
    y_pred = rf_clf.predict(X_test)

    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Calculate and print balanced accuracy
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print(f'Balanced Accuracy: {balanced_acc}')

if __name__ == '__main__':
    main()
