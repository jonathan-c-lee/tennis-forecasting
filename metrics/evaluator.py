"""Model evaluation methods."""
import torch
from torch.utils.data import Dataset

from models.tracknet import TrackNet


def get_scores(tp: int, tn: int, fp: int, fn: int):
    """
    Get model performance metrics.

    Args:
        tp (int): Number of true positives.
        tn (int): Number of true negatives.
        fp (int): Number of false positives.
        fn (int): Number of false negatives.
    
    Returns:
        Dictionary of scores including accuracy, precision, recall, and F1 score.
    """
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1 score': f1_score
    }


def evaluate_tracknet(model: TrackNet, dataset: Dataset, device: torch.device, threshold: int = 4):
    """
    Evaluate TrackNet model performance.

    Args:
        model (TrackNet): TrackNet model to evaluate.
        dataset (Dataset): Dataset to use for evaluation.
        threshold (int): Threshold for true positive.
        device (torch.device): Device that model is attached to.

    Returns:
        Raw statistics and performance scores.
    """
    tp, tn, fp, fn = 0, 0, 0, 0

    model.eval()
    for input, label in dataset:
        input, label = input.to(device), label.to(device)
        output = model(input.unsqueeze(0))
        for i, heatmap in enumerate(output):
            prediction = model.detect_ball(heatmap)
            actual = model.detect_ball(label[i])
            if actual == (-1, -1):
                if prediction == actual:
                    tn += 1
                else:
                    fp += 1
            else:
                if prediction == (-1, -1):
                    fn += 1
                else:
                    distance = ((actual[0] - prediction[0])**2 + (actual[1] - prediction[1])**2)**0.5
                    if distance <= threshold:
                        tp += 1
                    else:
                        fp += 1

    statistics = {
        'true positives': tp,
        'true negatives': tn,
        'false positives': fp,
        'false negatives': fn
    }
    scores = get_scores(tp, tn, fp, fn)
    return statistics, scores