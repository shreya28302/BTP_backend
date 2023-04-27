from algorithms.inprocessing.Proportionality import *
from dataloaders.data_loader_nonbinary import parse_data, random_sample
from metrics.Rho import *
from sklearn.cluster import KMeans
from metrics.Rho import calc_rho_proportionality
import argparse

def run_algo(dataset_name,algorithm_name):
    cost=0
    rho=1
    if algorithm_name=="unfair":
        rho_value = 1.00001

        min_k = 7
        max_k = 7
        k_step = 1
        tot_exp = 1  # Number of Experiments ran for each k
        parsed_data, kmeans_parsed_data = parse_data(dataset_name)
        dim = len(kmeans_parsed_data[0])

        # Assertion makes sure we get identical copy of data in two formats
        assert (len(kmeans_parsed_data[0]) == parsed_data[0].dim)
        assert (len(parsed_data) == len(kmeans_parsed_data))
        print('Succeed in Parsing Data')

        
        all_centers = parsed_data
        all_clients = parsed_data
        for rho in [rho_value]:
            print(rho)
            k_values = range(min_k, max_k + 1, k_step)
            for k in k_values:
                print('k = %d' % k)

                # Find Audit Centers for Rho Proportionality Calculation
                audit_centers = all_centers
                
                # KMeans++ Algorithm Part
                print("Start Kmeans++")
                kmeans = KMeans(n_clusters=k, random_state=0, init='k-means++').fit(np.array(kmeans_parsed_data))
                kmeans_centers = []
                for center in kmeans.cluster_centers_:
                    kmeans_centers.append(data_pt(center, 'kmeans'))
                kmeansobj_kmeans = calc_kmeans_obj(parsed_data, kmeans_centers, k)
                assert (abs(kmeansobj_kmeans - kmeans.inertia_) < 1000)
                kmeans_rho = calc_rho_proportionality(all_clients, kmeans_centers, k, audit_centers=audit_centers)
                print(kmeansobj_kmeans,kmeans_rho)
                cost=kmeansobj_kmeans
                Rho=kmeans_rho

    if algorithm_name=="Proportionality":  
        rho_value = 1.00001

        min_k = 7
        max_k = 7
        k_step = 1
        tot_exp = 1  # Number of Experiments ran for each k
        parsed_data, kmeans_parsed_data = parse_data(dataset_name)
        dim = len(kmeans_parsed_data[0])

        # Assertion makes sure we get identical copy of data in two formats
        assert (len(kmeans_parsed_data[0]) == parsed_data[0].dim)
        assert (len(parsed_data) == len(kmeans_parsed_data))
        print('Succeed in Parsing Data')

        
        all_centers = parsed_data
        all_clients = parsed_data

        for rho in [rho_value]:
            print(rho)
            k_values = range(min_k, max_k + 1, k_step)
            for k in k_values:
                print('k = %d' % k)

                # Find Audit Centers for Rho Proportionality Calculation
                audit_centers = all_centers

                # Greedy Capture Algorithm Part
                print("Start Greedy Capture Algorithm")
                greedy_center = ball_growing_repeated(all_clients, k, alpha=1, distances=None)
                assert (len(greedy_center) == k)
                kmeansobj_greedy = calc_kmeans_obj(parsed_data, greedy_center, k)
                kmeans_rho_greedy = calc_rho_proportionality(all_clients, greedy_center, k, audit_centers=audit_centers)
                print(kmeansobj_greedy,kmeans_rho_greedy)
                cost=kmeansobj_greedy
                Rho=kmeans_rho_greedy

        
    return cost,Rho
                