import GaiaSubsetSelection
import numpy as np

RANGE = 0.0
CENTER_OF_OBSERVATIONS = 0.0
SEED = 42

def get_range():
    
    pass

def get_center_of_observations():
    pass

def mean_distance_clusters(cluster_centers):
    pass

def create_cluster_centers(number_of_clusters: int, center_of_obs) -> List[NDArray[np.float64]]:
    cluster_centers = [center_of_obs]
    i = 1
    # The clusters should be within a specific range so that they 
    # Affect at least one other clsuter

    while i < number_of_clusters:
        center = np.random(0,RANGE / 2) * CENTER_OF_OBSERVATIONS
        if center not in cluster_centers:
            if cluster_distance < (mean_distance_clusters(cluster_centers) * 2):
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
