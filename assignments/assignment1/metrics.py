def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    
    TP = ground_truth[(prediction==ground_truth) & (ground_truth)==1].sum() # ok
    TN = len(ground_truth[prediction==ground_truth]) - TP # ok
    FP = prediction[(prediction!=ground_truth) & (ground_truth==0)].sum()
    FN = len(prediction[(prediction!=ground_truth) & (ground_truth==1)])

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = len(ground_truth[ground_truth == prediction]) / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    return len(ground_truth[ground_truth == prediction]) / len(ground_truth)
