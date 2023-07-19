import jax.numpy as jnp
from jax import random
from jax import vmap

def initialize_centroids(embeddings, k, key):
    """
    This function initializes k centroids randomly.

    Args:
        embeddings (jax.numpy.ndarray): The input embeddings.
        k (int): The number of clusters.
        key (jax.random.PRNGKey): The random key.

    Returns:
        jax.numpy.ndarray: The initialized centroids.
    """
    indices = random.choice(key, jnp.arange(embeddings.shape[0]), shape=(k,), replace=False)
    return jnp.take(embeddings, indices, axis=0)

def compute_distances(embedding, centroids):
    """
    This function computes the distance from each centroid to an embedding.

    Args:
        embedding (jax.numpy.ndarray): The input embedding.
        centroids (jax.numpy.ndarray): The centroids.

    Returns:
        jax.numpy.ndarray: The distances.
    """
    return jnp.sqrt(jnp.sum((embedding - centroids)**2, axis=-1))

def assign_clusters(embeddings, centroids):
    """
    This function assigns each embedding to the nearest centroid.

    Args:
        embeddings (jax.numpy.ndarray): The input embeddings.
        centroids (jax.numpy.ndarray): The centroids.

    Returns:
        jax.numpy.ndarray: The cluster assignments for each embedding.
    """
    distances = vmap(compute_distances, in_axes=(0, None))(embeddings, centroids)
    return jnp.argmin(distances, axis=-1)

def update_centroids(embeddings, assignments, k):
    """
    This function updates the centroids by computing the mean of all embeddings in each cluster.

    Args:
        embeddings (jax.numpy.ndarray): The input embeddings.
        assignments (jax.numpy.ndarray): The cluster assignments for each embedding.
        k (int): The number of clusters.

    Returns:
        jax.numpy.ndarray: The updated centroids.
    """
    return jnp.array([embeddings[assignments==i].mean(axis=0) for i in range(k)])

def kmeans(embeddings, k, num_iters=100, seed=0):
    """
    This function applies the K-Means algorithm to input embeddings.

    Args:
        embeddings (jax.numpy.ndarray): The input embeddings.
        k (int): The number of clusters.
        num_iters (int, optional): The number of iterations to run the K-Means algorithm. Default is 100.
        seed (int, optional): The random seed for centroid initialization. Default is 0.

    Returns:
        tuple: The final centroids and the cluster assignments for each embedding.
    """
    key = random.PRNGKey(seed)
    centroids = initialize_centroids(embeddings, k, key)

    for _ in range(num_iters):
        assignments = assign_clusters(embeddings, centroids)
        centroids = update_centroids(embeddings, assignments, k)

    return centroids, assignments
