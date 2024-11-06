import numpy as np
import matplotlib.pyplot as plt

k = 10
num_promotions = 1000

epsilon = 0.4

aligned_articles = [0, 2, 5]
true_view_probabilities = np.random.rand(k)
true_view_probabilities[aligned_articles] += 0.3  # Boost for aligned articles
true_view_probabilities = np.clip(true_view_probabilities, 0, 1)

estimated_view_probabilities = np.zeros(k)
promotion_counts = np.zeros(k)
views = np.zeros(num_promotions)

aligned_views = np.zeros(num_promotions)
non_aligned_views = np.zeros(num_promotions)

for i in range(num_promotions):
    if np.random.rand() < epsilon:
        article = np.random.randint(0, k)
    else:
        article = np.argmax(estimated_view_probabilities)
    view = 1 if np.random.rand() < true_view_probabilities[article] else 0
    
    promotion_counts[article] += 1
    estimated_view_probabilities[article] += (view - estimated_view_probabilities[article]) / promotion_counts[article]
    views[i] = view
    if article in aligned_articles:
        aligned_views[i] = view
    else:
        non_aligned_views[i] = view

print("True View Probabilities:", true_view_probabilities)
print("Estimated View Probabilities:", estimated_view_probabilities)
