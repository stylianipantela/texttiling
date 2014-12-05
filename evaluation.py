from collections import Counter

def window_diff(true_ls, pred_ls, k, N):
    """
    Calculate the WindowDiff metric as proposed in
    http://people.ischool.berkeley.edu/~hearst/papers/pevzner-01.pdf

    Args:
        true_ls: list of actual boundaries in the text.
        pred_ls: list of predicted boundaries in the text.
        k: length of window as defined in the paper.
        N: total number of possible boundary locations as defined in the paper.
    Returns:
        WindowDiff metric (number between 0 and 1).
    Raises:
        None.
    """
    true_dict = Counter()
    pred_dict = Counter()
    for item in true_ls:
        for index in range(item - k + 1, item + 1):
            true_dict[index] += 1
    for item in pred_ls:
        for index in range(item - k + 1, item + 1):
            pred_dict[index] += 1
    total = 0
    for i in range(0, N - k):
        if true_dict[i] != pred_dict[i]:
            total += 1
    return float(total)/float(N - k)
        
