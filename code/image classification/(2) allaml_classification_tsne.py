import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def main():
    # Load data
    data_feature = np.load(r'/public/home/chenlong666/desktop/my_desk1/coil_20/feature/rossler/allaml_feature_combined_couple_1.000.npy')
    print("Original data shape:", data_feature.shape)

    # t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    tsne_data = tsne.fit_transform(data_feature)
    X = np.array(tsne_data)
    print('t-SNE dimensionality reduction completed')

    # Save reduced features
    np.save(r'/public/home/chenlong666/desktop/my_desk1/coil_20/origin_data/coil_20_reduced.npy', X)
    print('Reduced features saved successfully')

    # Load labels
    path_label = r'/public/home/chenlong666/desktop/my_desk1/coil_20/origin_data/ALLAML_labels.npy'
    data_label = np.load(path_label)
    print("Size of data labels:", data_label.shape)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, data_label, test_size=0.3, random_state=42)

    # Train Random Forest classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)

    # Predictions
    y_pred = rf_clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Calculate balanced accuracy
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced Accuracy: {balanced_acc:.2f}")

if __name__ == '__main__':
    main()
