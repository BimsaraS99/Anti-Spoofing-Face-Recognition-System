import numpy as np
from numpy.linalg import norm

def calculate_distances(embedding1, embedding2):
    # Convert to NumPy arrays if not already
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)

    # Flatten 2D array if needed (shape [1, 128] → [128])
    if embedding2.ndim == 2 and embedding2.shape[0] == 1:
        embedding2 = embedding2.flatten()
    if embedding1.ndim == 2 and embedding1.shape[0] == 1:
        embedding1 = embedding1.flatten()

    # Euclidean Distance
    euclidean = norm(embedding1 - embedding2)

    # Cosine Distance
    cosine_similarity = np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
    cosine_distance = 1 - cosine_similarity

    return cosine_distance, euclidean


def is_same_person(embedding1, embedding2,
                            cosine_threshold=0.5,
                            euclidean_threshold=0.95):
    """
    Determines if two embeddings belong to the same person using both cosine and Euclidean distance.

    Parameters:
        embedding1: array-like, shape (128,) or (1, 128)
        embedding2: array-like, shape (128,) or (1, 128)
        cosine_threshold: float, threshold for cosine distance
        euclidean_threshold: float, threshold for Euclidean distance

    Returns:
        is_same: True if both distances agree it's the same person, else False
        results: dictionary with distances and threshold comparisons
    """
    cosine_distance, euclidean_distance = calculate_distances(embedding1, embedding2)

    same_cosine = cosine_distance < cosine_threshold
    same_euclidean = euclidean_distance < euclidean_threshold
    is_same = same_cosine and same_euclidean

    results = {
        'cosine_distance': cosine_distance,
        'euclidean_distance': euclidean_distance,
        'cosine_threshold_passed': same_cosine,
        'euclidean_threshold_passed': same_euclidean
    }

    return is_same, results
