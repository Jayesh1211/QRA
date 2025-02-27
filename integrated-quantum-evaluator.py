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

st.title("Quantum Model Evaluator")
st.write("Upload your quantum model and test data to evaluate performance")

# Define the model types
model_types = {
    "QAUM": "Quantum Asymptotically Universal Multi-feature Encoding",
    "QAOA": "Quantum Approximate Optimization Algorithm"
}

# Define the QAUM quantum circuit and model
def define_qaum_model(depth=2):
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

# Define the QAOA quantum circuit and model
def define_qaoa_model(depth=3):
    dev = qml.device("default.qubit.autograd", wires=9)

    def variational_circ(i, w):
        qml.RZ(w[i][0], wires=0)
        qml.RX(w[i][1], wires=0)
        qml.RY(w[i][2], wires=0)

    def quantum_neural_network(x, w, depth=depth):
        qml.templates.embeddings.QAOAEmbedding(features=x, weights=w, local_field='Y', wires=[i for i in range(9)])

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

def evaluate_model(test_df, model, model_type="QAUM"):
    # Extract features and target
    X_test = test_df.iloc[:, :-1].values  # All columns except the last one
    y_test = test_df.iloc[:, -1].values   # Last column is the target
    
    # Apply scaling
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, np.pi))
    X_test = min_max_scaler.fit_transform(X_test)
    
    # Get model parameters
    weights = model['weights']
    depth = model['depth']
    
    # Get model functions based on model type
    if model_type == "QAUM":
        model_functions = define_qaum_model(depth)
    else:  # QAOA
        model_functions = define_qaoa_model(depth)
        
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

# Function to display evaluation results
def display_results(test_accuracy, conf_matrix, class_report, y_pred, metrics, y_test, test_df, results_container, model_type):
    with results_container:
        st.subheader(f"{model_type} Model Evaluation Results")
        
        # Create columns for metrics and confusion matrix
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
            ax.set_title(f'Confusion Matrix - {model_type}')
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
        
        # Metrics visualization
        st.subheader("Metrics Visualization")
        # Create a simple bar chart of metrics
        fig, ax = plt.subplots(figsize=(10, 5))
        metric_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score']
        metric_values = [metrics['accuracy'], metrics['precision'], 
                         metrics['recall'], metrics['specificity'], 
                         metrics['f1_score']]
        
        ax.bar(metric_names, metric_values, color='skyblue')
        ax.set_ylim(0, 1.0)
        ax.set_title(f'Metric Comparison - {model_type}')
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
            label=f"Download {model_type} Results CSV",
            data=csv,
            file_name=f"{model_type.lower()}_predictions.csv",
            mime="text/csv",
        )
    
    return metrics, results_df

# Function to compare model results
def compare_models(qaum_metrics, qaoa_metrics):
    st.subheader("Model Comparison")
    
    # Create a dataframe for comparison
    comparison_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score'],
        'QAUM': [qaum_metrics['accuracy'], qaum_metrics['precision'], 
                qaum_metrics['recall'], qaum_metrics['specificity'], 
                qaum_metrics['f1_score']],
        'QAOA': [qaoa_metrics['accuracy'], qaoa_metrics['precision'], 
                qaoa_metrics['recall'], qaoa_metrics['specificity'], 
                qaoa_metrics['f1_score']],
        'Difference (QAOA - QAUM)': [
            qaoa_metrics['accuracy'] - qaum_metrics['accuracy'],
            qaoa_metrics['precision'] - qaum_metrics['precision'],
            qaoa_metrics['recall'] - qaum_metrics['recall'],
            qaoa_metrics['specificity'] - qaum_metrics['specificity'],
            qaoa_metrics['f1_score'] - qaum_metrics['f1_score']
        ]
    })
    
    # Format the dataframe
    st.dataframe(comparison_df.style.format({
        'QAUM': '{:.4f}',
        'QAOA': '{:.4f}',
        'Difference (QAOA - QAUM)': '{:.4f}'
    }).background_gradient(cmap='coolwarm', subset=['Difference (QAOA - QAUM)']))
    
    # Create a comparison chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(comparison_df['Metric']))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, comparison_df['QAUM'], width, label='QAUM')
    rects2 = ax.bar(x + width/2, comparison_df['QAOA'], width, label='QAOA')
    
    ax.set_ylabel('Score')
    ax.set_title('QAUM vs QAOA Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Metric'])
    ax.legend()
    
    ax.set_ylim(0, 1.0)
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Recommendations based on comparison
    st.subheader("Recommendations")
    best_model = "QAUM" if comparison_df['QAUM'].mean() > comparison_df['QAOA'].mean() else "QAOA"
    st.write(f"Based on the overall metrics, the {best_model} model performs better on this dataset.")
    
    # Detailed breakdown
    strengths = []
    for idx, row in comparison_df.iterrows():
        if row['Difference (QAOA - QAUM)'] > 0.05:
            strengths.append(f"QAOA performs better in {row['Metric']} by {row['Difference (QAOA - QAUM)']:.4f}")
        elif row['Difference (QAOA - QAUM)'] < -0.05:
            strengths.append(f"QAUM performs better in {row['Metric']} by {abs(row['Difference (QAOA - QAUM)']):.4f}")
    
    if strengths:
        st.write("Key differences:")
        for strength in strengths:
            st.write(f"- {strength}")
    else:
        st.write("The models perform similarly across all metrics.")

# Sidebar for file uploads and model selection
st.sidebar.header("Settings")

# Model selection
eval_option = st.sidebar.radio(
    "Evaluation Option",
    ["QAUM Model", "QAOA Model", "Compare Both Models"]
)

# Upload files section
st.sidebar.header("Upload Files")

# Upload model files based on the selected evaluation option
if eval_option in ["QAUM Model", "Compare Both Models"]:
    # Upload QAUM model file
    qaum_model_file = st.sidebar.file_uploader("Upload QAUM Model (PKL file)", type="pkl", key="qaum_model")
else:
    qaum_model_file = None

if eval_option in ["QAOA Model", "Compare Both Models"]:
    # Upload QAOA model file
    qaoa_model_file = st.sidebar.file_uploader("Upload QAOA Model (PKL file)", type="pkl", key="qaoa_model")
else:
    qaoa_model_file = None

# Upload test data
test_data_file = st.sidebar.file_uploader("Upload Test Data (CSV file)", type="csv")

# Container for model information
model_info_container = st.container()

# Container for results
results_container = st.container()

# Container for comparison
comparison_container = st.container()

# Process files based on selected evaluation option
if test_data_file:
    # Load test data
    test_df = pd.read_csv(test_data_file)
    
    # Display data preview
    st.subheader("Test Data Preview")
    st.dataframe(test_df.head())
    
    # QAUM evaluation
    if (eval_option in ["QAUM Model", "Compare Both Models"]) and qaum_model_file:
        # Save the uploaded model to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_model:
            tmp_model.write(qaum_model_file.getvalue())
            tmp_qaum_model_path = tmp_model.name
        
        # Load the model
        try:
            qaum_model = load_model(tmp_qaum_model_path)
            
            # Display model information
            with model_info_container:
                st.subheader("QAUM Model Information")
                st.write(f"Model Depth: {qaum_model['depth']}")
                total_parameters = qaum_model['weights'].size
                st.write(f"Number of Parameters: {total_parameters}")
        except Exception as e:
            st.error(f"Error loading QAUM model: {e}")
            st.exception(e)
    
    # QAOA evaluation
    if (eval_option in ["QAOA Model", "Compare Both Models"]) and qaoa_model_file:
        # Save the uploaded model to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_model:
            tmp_model.write(qaoa_model_file.getvalue())
            tmp_qaoa_model_path = tmp_model.name
        
        # Load the model
        try:
            qaoa_model = load_model(tmp_qaoa_model_path)
            
            # Display model information
            with model_info_container:
                st.subheader("QAOA Model Information")
                st.write(f"Model Depth: {qaoa_model['depth']}")
                total_parameters = qaoa_model['weights'].size
                st.write(f"Number of Parameters: {total_parameters}")
        except Exception as e:
            st.error(f"Error loading QAOA model: {e}")
            st.exception(e)
    
    # Evaluate button
    if st.button("Evaluate Model(s)"):
        qaum_results = None
        qaoa_results = None
        
        # QAUM evaluation
        if eval_option in ["QAUM Model", "Compare Both Models"] and qaum_model_file:
            with st.spinner("Evaluating QAUM model..."):
                try:
                    # Evaluate the model
                    qaum_test_accuracy, qaum_conf_matrix, qaum_class_report, qaum_y_pred, qaum_metrics, qaum_y_test = evaluate_model(test_df, qaum_model, "QAUM")
                    
                    # Display results
                    qaum_metrics, qaum_results_df = display_results(
                        qaum_test_accuracy, qaum_conf_matrix, qaum_class_report, 
                        qaum_y_pred, qaum_metrics, qaum_y_test, 
                        test_df, results_container, "QAUM"
                    )
                    qaum_results = {
                        'metrics': qaum_metrics,
                        'results_df': qaum_results_df
                    }
                except Exception as e:
                    st.error(f"Error evaluating QAUM model: {e}")
                    st.exception(e)
        
        # QAOA evaluation
        if eval_option in ["QAOA Model", "Compare Both Models"] and qaoa_model_file:
            with st.spinner("Evaluating QAOA model..."):
                try:
                    # Evaluate the model
                    qaoa_test_accuracy, qaoa_conf_matrix, qaoa_class_report, qaoa_y_pred, qaoa_metrics, qaoa_y_test = evaluate_model(test_df, qaoa_model, "QAOA")
                    
                    # Display results
                    qaoa_metrics, qaoa_results_df = display_results(
                        qaoa_test_accuracy, qaoa_conf_matrix, qaoa_class_report, 
                        qaoa_y_pred, qaoa_metrics, qaoa_y_test, 
                        test_df, results_container, "QAOA"
                    )
                    qaoa_results = {
                        'metrics': qaoa_metrics,
                        'results_df': qaoa_results_df
                    }
                except Exception as e:
                    st.error(f"Error evaluating QAOA model: {e}")
                    st.exception(e)
        
        # Compare models if both were evaluated
        if eval_option == "Compare Both Models" and qaum_results and qaoa_results:
            with comparison_container:
                compare_models(qaum_results['metrics'], qaoa_results['metrics'])
                
                # Create a comparison of predictions
                st.subheader("Prediction Comparison")
                comparison_df = test_df.copy()
                comparison_df['QAUM_Predicted'] = qaum_results['results_df']['Predicted']
                comparison_df['QAOA_Predicted'] = qaoa_results['results_df']['Predicted']
                comparison_df['Match'] = comparison_df['QAUM_Predicted'] == comparison_df['QAOA_Predicted']
                
                # Style the dataframe
                def highlight_match(val):
                    return 'background-color: #CCFFCC' if val else 'background-color: #FFDDDD'
                
                st.dataframe(comparison_df.style.apply(
                    lambda x: [''] * (len(x) - 1) + [highlight_match(x.iloc[-1])], 
                    axis=1
                ))
                
                # Calculate agreement statistics
                agreement_rate = comparison_df['Match'].mean() * 100
                st.write(f"Models agree on {agreement_rate:.2f}% of predictions")
                
                # Download combined results
                csv = comparison_df.to_csv(index=False)
                st.download_button(
                    label="Download Combined Results CSV",
                    data=csv,
                    file_name="model_comparison.csv",
                    mime="text/csv",
                )
    
    # Clean up temp files
    if 'tmp_qaum_model_path' in locals():
        os.unlink(tmp_qaum_model_path)
    if 'tmp_qaoa_model_path' in locals():
        os.unlink(tmp_qaoa_model_path)

else:
    with model_info_container:
        st.info("Please upload the required files to begin evaluation.")
        
        st.subheader("Expected File Formats")
        st.write("QAUM Model file: A pickle (.pkl) file containing a dictionary with 'weights' and 'depth' keys")
        st.write("QAOA Model file: A pickle (.pkl) file containing a dictionary with 'weights' and 'depth' keys")
        st.write("Test data: A CSV file with features in all columns except the last one, which should contain the target class (0 or 1)")

# Add explanatory information at the bottom
st.markdown("""
## About This Application

This application evaluates quantum machine learning models created with PennyLane. 
It supports two types of quantum encoding:

1. **QAUM (Quantum Asymptotically Universal Multi-feature) Encoding**:
   - Uses a single qubit for binary classification tasks
   - Leverages a Hadamard gate followed by rotations

2. **QAOA (Quantum Approximate Optimization Algorithm) Encoding**:
   - Uses multiple qubits (9 in this implementation)
   - Employs the QAOA embedding template from PennyLane

### How to use:
1. Select the evaluation option (QAUM, QAOA, or Compare Both)
2. Upload your trained model(s) (.pkl files)
3. Upload your test dataset (.csv file)
4. Click "Evaluate Model(s)" to run the evaluation

### Evaluation Metrics:
- **Accuracy**: Overall correctness of the model
- **Precision**: Ability to identify only the relevant data points
- **Recall**: Ability to find all relevant instances
- **Specificity**: Ability to identify true negatives
- **F1 Score**: Harmonic mean of precision and recall

### Model Requirements:
Each model should be a dictionary with at least:
- `weights`: The trained parameters
- `depth`: The circuit depth parameter

### Data Requirements:
- Features should be in all columns except the last
- The last column should contain the target class (0 or 1)
""")
