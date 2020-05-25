def score_norm(score, epsilon=1e-8):
    return (score - score.min()) / (score.max() - score.min() + epsilon)