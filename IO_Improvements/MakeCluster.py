import GaiaSubsetSelection
import numpy as np

RANGE = 0.0
CENTER_OF_OBSERVATIONS = 0.0
SEED = 42

def get_range():
    pass

def get_center_of_observations():
    pass

def create_cluster_centers(number_of_clusters: int) -> List[NDArray[np.float64]]:
    cluster_centers = []
    i = 0
    while i < number_of_clusters:
        center = np.random(0,RANGE / 2) * CENTER_OF_OBSERVATIONS
        if center not in cluster_centers:
            cluster_centers.append(center)
            i++
    return cluster_centers


def main():
    number_of_clusters = 10
    cluster_centers = create_cluster_centers(number_of_clusters)
    for i in range cluster_centers:
        GaiaSubsetSelection.CLUSTER_CENTER = cluster_centers[i]
        GaiaSubsetSelection.main()
        


if __name__ == "__main__":
    main()
