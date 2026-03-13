# Lab 5: Decision Tree Classifiers

## Objective

Learn decision tree classifiers for non-linear decision boundaries.

## Experiment

### Task

Develop a decision tree model to classify species in the Iris dataset.

### Evaluation Metrics

- **Accuracy**: Overall correctness of the model
- **Classification Report**: Detailed metrics for each species including:
  - **Precision**: True positives / (True positives + False positives)
  - **Recall**: True positives / (True positives + False negatives)
  - **F1-Score**: Harmonic mean of precision and recall

### Visualization

- Use `plot_tree()` to visualize the decision tree structure
- This visualization helps understand how the model splits the data at different nodes
- Each node shows the decision criteria and the distribution of classes

## Implementation Details

- **Dataset**: Iris dataset (150 samples, 4 features, 3 classes)
- **Model**: Decision Tree Classifier
- **Framework**: scikit-learn

## Expected Outcomes

- A trained decision tree model capable of classifying iris species
- Visual representation of the decision tree showing node splits and decision boundaries
- Comprehensive performance metrics demonstrating model effectiveness
- Understanding of how decision trees handle non-linear classification problems
