
# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pennylane as qml
from pennylane.optimize import AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, NesterovMomentumOptimizer
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns  # For better visualizations of the confusion matrix

# Fetching Data with Random Seed
def fetch_data_random_seed_val(n_samples, seed):
    dataset = pd.read_csv('pulsar.csv')

    data0 = dataset[dataset[dataset.columns[8]] == 0]
    data0 = data0.sample(n=n_samples, random_state=seed)
    X0 = data0[data0.columns[0:8]].values
    Y0 = data0[data0.columns[8]].values

    data1 = dataset[dataset[dataset.columns[8]] == 1]
    data1 = data1.sample(n=n_samples, random_state=seed)
    X1 = data1[data1.columns[0:8]].values
    Y1 = data1[data1.columns[8]].values

    X = np.append(X0, X1, axis=0)
    Y = np.append(Y0, Y1, axis=0)

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, np.pi))
    X = min_max_scaler.fit_transform(X)

    # Split into training+validation and testing datasets
    train_val_X, test_X, train_val_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=seed)
    
    # Split the training+validation set into training and validation sets
    train_X, validation_X, train_Y, validation_Y = train_test_split(train_val_X, train_val_Y, test_size=0.25, random_state=seed)

    return train_X, validation_X, test_X, train_Y, validation_Y, test_Y

# Quantum Model Training Function
def quantum_model_train(train_X, train_Y, validation_X, validation_Y, test_X, test_Y, depth=1):
    from pennylane import numpy as np

    train_X = np.array(train_X, requires_grad=False)
    train_Y = np.array(train_Y, requires_grad=False)
    validation_X = np.array(validation_X, requires_grad=False)
    validation_Y = np.array(validation_Y, requires_grad=False)
    test_X = np.array(test_X, requires_grad=False)
    test_Y = np.array(test_Y, requires_grad=False)
    validation_data = list(zip(validation_X, validation_Y))
    test_data = list(zip(test_X, test_Y))
    train_data = list(zip(train_X, train_Y))

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
        quantum_neural_network(x, w)
        return qml.expval(qml.PauliZ(wires=0))

    def get_parity_prediction(x, w):
        np_measurements = (get_output(x, w) + 1.) / 2.
        return np.array([1. - np_measurements, np_measurements])

    def average_loss(w, data):
        cost_value = 0
        for i, (x, y) in enumerate(data):
            cost_value += single_loss(w, x, y)
        return cost_value / len(data)

    def single_loss(w, x, y):
        prediction = get_parity_prediction(x, w)
        return rel_ent(prediction, y)

    def rel_ent(pred, y):
        return -1. * np.log(pred)[int(y)]

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

    def precision(data, w):
        tp = 0  # True Positives
        fp = 0  # False Positives
        for x, y in data:
            cat = categorise(x, w)
            if int(cat) == 1 and int(y) == 1:
                tp += 1
            elif int(cat) == 1 and int(y) == 0:
                fp += 1
        if tp + fp == 0:
            return 0  # Avoid division by zero
        return tp / (tp + fp) * 100

    def recall(data, w):
        tp = 0  # True Positives
        fn = 0  # False Negatives
        for x, y in data:
            cat = categorise(x, w)
            if int(cat) == 1 and int(y) == 1:
                tp += 1
            elif int(cat) == 0 and int(y) == 1:
                fn += 1
        if tp + fn == 0:
            return 0  # Avoid division by zero
        return tp / (tp + fn) * 100

    def specificity(data, w):
        tn = 0  # True Negatives
        fp = 0  # False Positives
        for x, y in data:
            cat = categorise(x, w)
            if int(cat) == 0 and int(y) == 0:
                tn += 1
            elif int(cat) == 1 and int(y) == 0:
                fp += 1
        if tn + fp == 0:
            return 0  # Avoid division by zero
        return tn / (tn + fp) * 100

    def get_predictions(data, w):
        preds = []
        for x, y in data:
            cat = categorise(x, w)
            preds.append(cat)
        return np.array(preds)

    # Initialise weights
    w = np.array(np.split(np.random.uniform(size=(3 * (8 * depth + 1),), low=-1, high=1), 8 * depth + 1),
                 requires_grad=True) * 2 * np.pi
    learning_rate = 0.1

    # Optimiser
    optimiser = AdagradOptimizer(learning_rate)
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    for i in range(num_epochs):
        start = time.time()
        w, train_loss_value = optimiser.step_and_cost(lambda v: average_loss(v, train_data), w)
        end = time.time()
        w.requires_grad = False
        train_acc = accuracy(train_data, w)
        validation_loss_value = average_loss(w, validation_data)
        validation_acc = accuracy(validation_data, w)
        w.requires_grad = True

        train_accs.append(train_acc)
        train_losses.append(train_loss_value)
        val_accs.append(validation_acc)
        val_losses.append(validation_loss_value)

        print(f"Epoch = {i}, Training Loss = {train_loss_value}, Validation Loss = {validation_loss_value}, "
              f"Train Acc = {train_acc}%, Val Acc = {validation_acc}%, Time taken = {end - start}s")

    # Measure test accuracy after training
    test_acc = accuracy(test_data, w)
    test_precision = precision(test_data, w)
    test_recall = recall(test_data, w)
    test_specificity = specificity(test_data, w)
    test_preds = get_predictions(test_data, w)

    print(f"Test Accuracy = {test_acc}%")
    print("Test Precision = ", test_precision, "%")
    print("Test Recall = ", test_recall, "%")
    print("Test Specificity = ", test_specificity, "%")

    return train_accs, val_accs, train_losses, val_losses, test_acc, test_precision, test_recall, test_specificity, test_preds, test_Y

# Parameters
num_epochs = 100
n_iteration = 5


all_train_losses = []
all_val_losses = []
all_train_accs = []
all_val_accs = []
all_test_accs = []
all_test_precisions = []
all_test_recalls = []
all_test_specificities = []
all_preds = []
all_actuals = []

for i in range(n_iteration):
    train_X, validation_X, test_X, train_Y, validation_Y, test_Y = fetch_data_random_seed_val(n_samples=300, seed=i)
    print(f"Iteration {i+1}")
    loss = quantum_model_train(train_X, train_Y, validation_X, validation_Y, test_X, test_Y, depth=2)
    all_train_losses.append(loss[2])  # Append train_losses
    all_val_losses.append(loss[3])  # Append val_losses
    all_train_accs.append(loss[0][-1])  # Append last train_acc
    all_val_accs.append(loss[1][-1])  # Append last val_acc
    all_test_accs.append(loss[4])  # Append test_acc
    all_test_precisions.append(loss[5])  # Append test_precision
    all_test_recalls.append(loss[6])  # Append test_recall
    all_test_specificities.append(loss[7]) 
    all_preds.append(loss[8])
    all_actuals.append(loss[9]) # Append test_specificity

# Compute the mean of all iteration metrics
mean_train_loss = np.mean(all_train_losses)
mean_val_loss = np.mean(all_val_losses)
mean_train_acc = np.mean(all_train_accs)
mean_val_acc = np.mean(all_val_accs)
mean_test_acc = np.mean(all_test_accs)
mean_test_precision = np.mean(all_test_precisions)
mean_test_recall = np.mean(all_test_recalls)
mean_test_specificity = np.mean(all_test_specificities)
min_train_losses = np.min(all_train_losses)
max_train_acc = max(all_train_accs)

# Print the mean metrics
print(f"Mean Training Loss: {mean_train_loss}")
print("minimum Training Loss:", min_train_losses)
print(f"Mean Validation Loss: {mean_val_loss}")
print(f"Mean Training Accuracy: {mean_train_acc}")
print("maximum Traning Accuracy:", max_train_acc)
print(f"Mean Validation Accuracy: {mean_val_acc}")
print(f"Mean Test Accuracy: {mean_test_acc}")
print(f"Mean Test Precision: {mean_test_precision}")
print(f"Mean Test Recall: {mean_test_recall}")
print(f"Mean Test Specificity: {mean_test_specificity}")



# Print the data used for confusion matrix
print("True labels and predicted labels for each iteration:")
for i, (actuals, preds) in enumerate(zip(all_actuals, all_preds)):
    print(f"Iteration {i+1}:")
    print("True labels: ", actuals)
    print("Predicted labels: ", preds)



# Compute confusion matrices for each iteration
conf_matrices = [confusion_matrix(y_true, y_pred) for y_true, y_pred in zip(all_actuals, all_preds)]

# Convert to numpy array for easy manipulation
conf_matrices = np.array(conf_matrices)

# Compute the average confusion matrix
avg_conf_matrix = np.mean(conf_matrices, axis=0)

# Compute the standard error of the confusion matrices
std_error_conf_matrix = np.std(conf_matrices, axis=0, ddof=1) / np.sqrt(n_iteration)

# Combine average and standard error in the form "mean ± error"
conf_matrix_with_error = np.empty(avg_conf_matrix.shape, dtype=object)
for i in range(avg_conf_matrix.shape[0]):
    for j in range(avg_conf_matrix.shape[1]):
        conf_matrix_with_error[i, j] = f"{avg_conf_matrix[i, j]:.2f} ± {std_error_conf_matrix[i, j]:.2f}"

# Plot the combined confusion matrix
plt.figure(figsize=(4, 4))  # Set figure size to make it square
ax = sns.heatmap(avg_conf_matrix, annot=conf_matrix_with_error, fmt="", cmap="Blues", 
                 xticklabels=["Non-Pulsar", "Pulsar"], yticklabels=["Non-Pulsar", "Pulsar"], 
                 annot_kws={"size": 9, "weight": "bold"}, cbar=False, square=True)  # Ensure square shape

# Move x-axis labels to the top
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# Manually add the color bar at the bottom
cbar = ax.figure.colorbar(ax.collections[0], orientation="horizontal", pad=0.05, aspect=30, shrink=0.9)  # Adjust aspect and shrink
cbar.ax.tick_params(labelsize=10)
cbar.set_label('Number of Predictions', fontsize=12)

#plt.title('QAUM (L=2)', fontsize=16)
plt.xlabel('Predicted (QAUM, L=2)', fontsize=14)#, labelpad=20)  # Adjust labelpad for spacing
plt.ylabel('True', fontsize=14)

# Increase size of labels
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Save the plot
#plt.savefig('ConfMatrix(QAUM, L=2_1).png')

plt.show()

#print(np.array(losses, dtype=float).shape)
#print(f"Test Accuracies: {test_accuracies}")











