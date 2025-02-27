import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn import preprocessing
import pennylane as qml
import tempfile
import os

st.set_page_config(page_title="Quantum Model Evaluator", layout="wide")

st.title("Quantum Asymptotically Universal Multi-feature(QAUM) Encoding")
st.write("Upload your quantum model and test data to evaluate performance")

# Define the quantum circuit and model
def define_quantum_model(depth=2):
    dev = qml.device("default.qubit.autograd", wires=1)

    def variational_circ(i, w):
        qml.RZ(w[i][0], wires=0)
        qml.RX(w[i][1], wires=0)
        qml.RY(w[i][2], wires=0)

    def quantum_neural_network(x, w, depth=depth):
        qml.Hadamard(wires=0)
        variational_circ(0, w)
        for i in range(0, depth):
            for j in range(8):
                qml.RZ(x[j], wires=0)
                variational_circ(j + 8 * i, w)

    @qml.qnode(dev, diff_method='backprop')
    def get_output(x, w):
        quantum_neural_network(x, w, depth)
        return qml.expval(qml.PauliZ(wires=0))

    def get_parity_prediction(x, w):
        np_measurements = (get_output(x, w) + 1.) / 2.
        return np.array([1. - np_measurements, np_measurements])

    def categorise(x, w):
        out = get_parity_prediction(x, w)
        return np.argmax(out)

    def accuracy(data, w):
        correct = 0
        for ii, (x, y) in enumerate(data):
            cat = categorise(x, w)
            if int(cat) == int(y):
                correct += 1
        return correct / len(data) * 100
    
    model_functions = {
        'get_output': get_output,
        'get_parity_prediction': get_parity_prediction,
        'categorise': categorise,
        'accuracy': accuracy
    }
    
    return model_functions

def load_model(model_file):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model

def calculate_specificity(y_true, y_pred):
    """Calculate specificity (true negative rate)"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def evaluate_model(test_df, model):
    # Extract features and target
    X_test = test_df.iloc[:, :-1].values  # All columns except the last one
    y_test = test_df.iloc[:, -1].values   # Last column is the target
    
    # Apply scaling
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, np.pi))
    X_test = min_max_scaler.fit_transform(X_test)
    
    # Get model parameters
    weights = model['weights']
    depth = model['depth']
    
    # Get model functions
    model_functions = define_quantum_model(depth)
    categorise = model_functions['categorise']
    accuracy = model_functions['accuracy']
    
    # Prepare data for evaluation
    test_data = list(zip(X_test, y_test))
    
    # Calculate accuracy
    test_accuracy = accuracy(test_data, weights)
    
    # Generate predictions for all test instances
    y_pred = []
    for x in X_test:
        pred = categorise(x, weights)
        y_pred.append(pred)
    
    # Generate confusion matrix and classification report
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Calculate additional metrics
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
    specificity = calculate_specificity(y_test, y_pred)
    
    # Create a dictionary of metrics
    metrics = {
        'accuracy': test_accuracy / 100,  # Convert from percentage to decimal
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1
    }
    
    return test_accuracy, conf_matrix, class_report, y_pred, metrics, y_test

# Sidebar for file uploads
st.sidebar.header("Upload Files")

# Upload model file
model_file = st.sidebar.file_uploader("Upload Model (PKL file)", type="pkl")

# Upload test data
test_data_file = st.sidebar.file_uploader("Upload Test Data (CSV file)", type="csv")

# Container for model information
model_info_container = st.container()

# Container for results
results_container = st.container()

# If both files are uploaded
if model_file and test_data_file:
    # Save the uploaded model to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_model:
        tmp_model.write(model_file.getvalue())
        tmp_model_path = tmp_model.name
    
    # Load the model
    try:
        model = load_model(tmp_model_path)
        
        # Display model information
        with model_info_container:
            st.subheader("Model Information")
            st.write(f"Model Depth: {model['depth']}")
            #st.write(f"Number of Parameters: {len(model['weights'])}")
    # Calculate total number of parameters
            total_parameters = model['weights'].size  # This works if weights is a NumPy array
    # If weights is a list of arrays, use this instead
    # total_parameters = sum(len(w) for w in model['weights'])

    

            st.write(f"Number of Parameters: {total_parameters}")
        
        # Load test data
        test_df = pd.read_csv(test_data_file)
        
        # Display data preview
        st.subheader("Test Data Preview")
        st.dataframe(test_df.head())
        
        # Allow user to start evaluation
        if st.button("Evaluate Model"):
            with st.spinner("Evaluating model... This may take a moment."):
                # Evaluate the model
                test_accuracy, conf_matrix, class_report, y_pred, metrics, y_test = evaluate_model(test_df, model)
                
                # Display results
                with results_container:
                    # Create three columns for metrics, confusion matrix, and additional info
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("Performance Metrics")
                        
                        # Create a metrics dashboard
                        metrics_cols = st.columns(3)
                        
                        metrics_cols[0].metric("Accuracy", f"{metrics['accuracy']:.4f}")
                        metrics_cols[1].metric("Precision", f"{metrics['precision']:.4f}")
                        metrics_cols[2].metric("Recall", f"{metrics['recall']:.4f}")
                        
                        metrics_cols2 = st.columns(3)
                        metrics_cols2[0].metric("Specificity", f"{metrics['specificity']:.4f}")
                        metrics_cols2[1].metric("F1 Score", f"{metrics['f1_score']:.4f}")
                        
                        # Create metrics from classification report
                        st.write("Classification Report:")
                        metrics_df = pd.DataFrame(class_report).transpose()
                        st.dataframe(metrics_df.style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1-score": "{:.4f}", "support": "{:.0f}"}))
                        
                        # Add metric explanations
                        with st.expander("Metric Explanations"):
                            st.markdown("""
                            - **Accuracy**: Proportion of correct predictions among the total number of predictions
                            - **Precision**: Proportion of true positive predictions among all positive predictions (TP / (TP + FP))
                            - **Recall**: Proportion of true positive predictions among all actual positives (TP / (TP + FN))
                            - **Specificity**: Proportion of true negative predictions among all actual negatives (TN / (TN + FP))
                            - **F1 Score**: Harmonic mean of precision and recall (2 * (precision * recall) / (precision + recall))
                            """)
                    
                    with col2:
                        st.subheader("Confusion Matrix")
                        # Plot confusion matrix
                        fig, ax = plt.subplots(figsize=(8, 6))
                        im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
                        ax.set_title('Confusion Matrix')
                        plt.colorbar(im, ax=ax)
                        
                        classes = ['Class 0', 'Class 1']
                        tick_marks = np.arange(len(classes))
                        ax.set_xticks(tick_marks)
                        ax.set_xticklabels(classes)
                        ax.set_yticks(tick_marks)
                        ax.set_yticklabels(classes)
                        
                        # Add text annotations to each cell
                        thresh = conf_matrix.max() / 2
                        for i in range(conf_matrix.shape[0]):
                            for j in range(conf_matrix.shape[1]):
                                ax.text(j, i, format(conf_matrix[i, j], 'd'),
                                        ha="center", va="center",
                                        color="white" if conf_matrix[i, j] > thresh else "black")
                        
                        ax.set_xlabel('Predicted label')
                        ax.set_ylabel('True label')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Add confusion matrix explanation
                        with st.expander("Confusion Matrix Explanation"):
                            st.markdown("""
                            The confusion matrix shows:
                            - **True Negatives (TN)**: Top-left - Correctly predicted negative cases
                            - **False Positives (FP)**: Top-right - Negative cases predicted as positive (Type I error)
                            - **False Negatives (FN)**: Bottom-left - Positive cases predicted as negative (Type II error)
                            - **True Positives (TP)**: Bottom-right - Correctly predicted positive cases
                            """)
                    
                    # ROC curve and precision-recall curve
                    if len(np.unique(y_test)) > 1:  # Only if we have both classes in test set
                        st.subheader("Additional Visualizations")
                        cols_viz = st.columns(2)
                        
                        with cols_viz[0]:
                            # Calculate ROC points manually (since we already have predictions)
                            # For binary case with 0-1 predictions, this is simplified
                            st.write("ROC Curve placeholder (requires probability outputs)")
                            st.info("Note: For full ROC curve, model would need to output probabilities instead of just class predictions")
                        
                        with cols_viz[1]:
                            # Create a simple bar chart of metrics
                            fig, ax = plt.subplots(figsize=(8, 5))
                            metric_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score']
                            metric_values = [metrics['accuracy'], metrics['precision'], 
                                             metrics['recall'], metrics['specificity'], 
                                             metrics['f1_score']]
                            
                            ax.bar(metric_names, metric_values, color='skyblue')
                            ax.set_ylim(0, 1.0)
                            ax.set_title('Metric Comparison')
                            ax.set_ylabel('Score')
                            
                            # Add values on top of bars
                            for i, v in enumerate(metric_values):
                                ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
                                
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    # Add predictions vs actual
                    st.subheader("Predictions vs Actual")
                    results_df = test_df.copy()
                    results_df['Predicted'] = y_pred
                    
                    # Add a column for correct/incorrect predictions
                    results_df['Correct'] = results_df.iloc[:, -2] == results_df['Predicted']
                    
                    # Style the dataframe to highlight correct/incorrect predictions
                    def highlight_correct(val):
                        return 'background-color: #CCFFCC' if val else 'background-color: #FFCCCC'
                    
                    st.dataframe(results_df.style.apply(
                        lambda x: [''] * (len(x) - 1) + [highlight_correct(x.iloc[-1])], 
                        axis=1
                    ))
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results CSV",
                        data=csv,
                        file_name="qaum_predictions.csv",
                        mime="text/csv",
                    )
    
    except Exception as e:
        st.error(f"Error loading or evaluating model: {e}")
        st.exception(e)  # This will show the full traceback for debugging
    
    # Clean up temp file
    os.unlink(tmp_model_path)

else:
    with model_info_container:
        st.info("Please upload both a model file (.pkl) and test data (.csv) to begin evaluation.")
        
        st.subheader("Expected File Formats")
        st.write("Model file: A pickle (.pkl) file containing a dictionary with 'weights' and 'depth' keys")
        st.write("Test data: A CSV file with features in all columns except the last one, which should contain the target class (0 or 1)")

# Add explanatory information at the bottom
st.markdown("""
## About This Application

This application evaluates quantum machine learning models created with PennyLane. 
It specifically works with Quantum Asymptotically Universal Multi-feature (QAUM) Encoding, which uses a single qubit 
for binary classification tasks.

### How to use:
1. Upload your trained model (.pkl file)
2. Upload your test dataset (.csv file)
3. Click "Evaluate Model" to run the evaluation

### Evaluation Metrics:
- **Accuracy**: Overall correctness of the model
- **Precision**: Ability to identify only the relevant data points
- **Recall**: Ability to find all relevant instances
- **Specificity**: Ability to identify true negatives
- **F1 Score**: Harmonic mean of precision and recall

### Model Requirements:
The model should be a dictionary with at least:
- `weights`: The trained parameters
- `depth`: The circuit depth parameter

### Data Requirements:
- Features should be in all columns except the last
- The last column should contain the target class (0 or 1)
""")
