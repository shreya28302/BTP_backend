
# Importing libraries
from importlib import reload
import time
import json
from tabnanny import verbose
from algorithms.unfairAlgorithms.kCenters import KCenters
from metrics.balance import distance, balance_calculation
from dataloaders.data_loader_binary import DataLoader
from algorithms.preprocessing.fairlet_decomposition import VanillaFairletDecomposition, MCFFairletDecomposition
import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy as np
from collections import defaultdict
import algorithms.preprocessing.scalable_fairlet_decomposition as scalable
import matlab
import time

def run_algo(dataset_name,algorithm_name):

    source = dataset_name
    normalize = False
    degrees = 2
    degree =2
    balance=0.0
    cost=0.0
    
    with opeen('config.json') as jsn_file:
        config = json.load(json_file)
    
    dl = DataLoader(source=source, fair_column=config[source]['fair_column'],
                fair_values=config[source]['fair_values'], distance_columns=config[source]['distance_columns'])
    dl.load(normalize)
    blues, reds = dl.split(split_size=tuple(config[source]['split_size']), random_state=config[source]['random_state'])
    
    # running unfair algo
    if algorithm_name== "unfair":
        unfair_degrees = []
        unfair_costs = []
        unfair_balances = []

        start_time = time.time()
        kcenters = KCenters(k=degree)
        kcenters.fit(dl.data_list)
        mapping = kcenters.assign()
        cost=kcenters.costs[-1]
        balance=balance_calculation(dl.data_list, kcenters.centers, mapping)

    # running fair algo
    if algorithm_name== "MCF":
        # Instantiating the MCF Decomposition
        mcf = MCFFairletDecomposition(blues, reds, 2, config[source]['distance_threshold'], dl.data_list)

        # Computing the distance matrix between blue and red nodes
        mcf.compute_distances()

        # Adding nodes and edges
        mcf.build_graph(plot_graph=True)

        # Decomposing for fairlets and performing traditional clustering
        mcf_fairlets, mcf_fairlet_centers, mcf_fairlet_costs = mcf.decompose()
        curr_degrees = []
        curr_costs = []
        curr_balances = []
        data=dl.data_list
        start_time = time.time()
        kcenters = KCenters(k=degree)
        kcenters.fit([data[i] for i in mcf_fairlet_centers])
        mapping = kcenters.assign()
        final_clusters = []
        for fairlet_id, final_cluster in mapping:
            for point in mcf_fairlets[fairlet_id]:
                final_clusters.append((point, mcf_fairlet_centers[final_cluster]))
                
        centers = [mcf_fairlet_centers[i] for i in kcenters.centers]
        cost=max([min([distance(data[j], i) for j in centers]) for i in data])
        balance=balance_calculation(data, centers, final_clusters)
        if verbose:
                print("Time taken for Degree %d - %.3f seconds."%(degree, time.time() - start_time))
        
    if algorithm_name== "Proportionality":
        
        option = parser.parse_args()
        file_name = "./data/" + option.file_name
        output_file = "./result/" + option.file_name + "_result.txt"
        rho_value = option.rho

        min_k = 2
        max_k = 10
        k_step = 1
        sample_num = option.sample_num
        center_num = option.center_num
        sample_type = "Random" if option.sample_type else "Full" # Choose from Full, Random
        tot_exp = 1  # Number of Experiments ran for each k

        print('Working on %s, Randomly select %d samples, k from %d to %d' % (file_name, sample_num, min_k, max_k))
        parsed_data, kmeans_parsed_data = parse_data(file_name)
        dim = len(kmeans_parsed_data[0])

        # Assertion makes sure we get identical copy of data in two formats
        assert (len(kmeans_parsed_data[0]) == parsed_data[0].dim)
        assert (len(parsed_data) == len(kmeans_parsed_data))
        print('Succeed in Parsing Data')

        if sample_type == "Random":
            all_clients, reverse_map, _ = random_sample(parsed_data, sample_num)
            print('Succeed in Sampling %d Clients' % len(all_clients))
            all_centers, original_centers = kmeansinitialization(kmeans_parsed_data, center_num)
            print('Succeed in Sampling %d Centers' % len(all_centers))
        else:
            all_centers = parsed_data
            all_clients = parsed_data

        for rho in [rho_value]:
            print(rho)
            k_values = range(min_k, max_k + 1, k_step)
            for k in k_values:
                print('k = %d' % k)

                # Find Audit Centers for Rho Proportionality Calculation
                if sample_type == "Random":
                    audit_centers, _ = kmeansinitialization(kmeans_parsed_data, center_num)
                else:
                    audit_centers = all_centers

                # Local Capture Algorithm Part
                print("Start Local Search")
                local_capture_centers = local_capture(all_clients, k, rho=rho, all_centers=all_centers)
                assert (len(local_capture_centers) == k)
                local_capture_kmeans = calc_kmeans_obj(parsed_data, local_capture_centers, k)
                local_capture_rho = calc_rho_proportionality(all_clients, local_capture_centers, k,
                                                            audit_centers=audit_centers)
                printData(output_file, local_capture_kmeans, local_capture_rho, label="Local Search")

                # KMeans++ Algorithm Part
                print("Start Kmeans++")
                kmeans = KMeans(n_clusters=k, random_state=0, init='k-means++').fit(np.array(kmeans_parsed_data))
                kmeans_centers = []
                for center in kmeans.cluster_centers_:
                    kmeans_centers.append(data_pt(center, 'kmeans'))
                kmeansobj_kmeans = calc_kmeans_obj(parsed_data, kmeans_centers, k)
                assert (abs(kmeansobj_kmeans - kmeans.inertia_) < 1000)
                kmeans_rho = calc_rho_proportionality(all_clients, kmeans_centers, k, audit_centers=audit_centers)
                printData(output_file, kmeans.inertia_, kmeans_rho, label="KMeans++")

                if sample_type == "Random":
                    # KMeans Heuristic Part
                    print("Start Center Reduction Heuristic")
                    remain_centers = local_capture_centers + kmeans_centers
                    k2_kmeans = calc_kmeans_obj(parsed_data, remain_centers, k)
                    k2_rho = calc_rho_proportionality(all_clients, remain_centers, k, audit_centers=audit_centers)
                    printData(output_file, k2_kmeans, k2_rho, label="2KCenters")

                    # Reduce as many centers as possible
                    flag = True
                    while (len(remain_centers) > k and flag):
                        flag = False
                        for next_close_center in remain_centers:
                            remain_centers.remove(next_close_center)
                            temp_kmeans = calc_kmeans_obj(parsed_data, remain_centers, k)
                            temp_rho = calc_rho_proportionality(all_clients, remain_centers, k, audit_centers=audit_centers)
                            if (temp_rho <= 1.2 * local_capture_rho and temp_kmeans <= 1.5 * kmeansobj_kmeans):
                                flag = True
                                break
                            remain_centers.append(next_close_center)

                    remain_centers_kmeans = calc_kmeans_obj(parsed_data, remain_centers, k)
                    remain_centers_rho = calc_rho_proportionality(all_clients, remain_centers, k,
                                                                audit_centers=audit_centers)
                    print("Hybrid Heuristic finish with %d centers, Kmeans Objective %d, Rho Objective %f" % (
                    len(remain_centers), remain_centers_kmeans, remain_centers_rho))
                    f = open(output_file, "a")
                    f.write(
                        str(remain_centers_kmeans) + " " + str(remain_centers_rho) + " " + str(len(remain_centers)) + "\n")
                    f.close()
                else:
                    # Greedy Capture Algorithm Part
                    print("Start Greedy Capture Algorithm")
                    greedy_center = ball_growing_repeated(all_clients, k, alpha=1, distances=None)
                    assert (len(greedy_center) == k)
                    kmeansobj_greedy = calc_kmeans_obj(parsed_data, greedy_center, k)
                    kmeans_rho_greedy = calc_rho_proportionality(all_clients, greedy_center, k, audit_centers=audit_centers)
                    printData(output_file, kmeansobj_greedy, kmeans_rho_greedy, label="Greedy Ball Growing", EOL=True)

    if algorithm_name== "Scalable":  
        colors = []
        points = []
        p=1
        q=5
        i = 0
        skipped_lines = 0
        for line in dl.data_list:
            if len(line) == 0:
                skipped_lines += 1
                continue
            tokens = line
            try:
                color = int(tokens[0])
            except:
                print("Invalid color label in line", i, ", skipping")
                skipped_lines += 1
                continue
            try:
                point = [float(x) for x in tokens[1:]]
            except:
                print("Invalid point coordinates in line", i, ", skipping")
                skipped_lines += 1
                continue
            colors.append(color)
            points.append(point)
            i += 1
        
            n_points = len(points)
            if  n_points == 0:
                print("No successfully parsed points in input file, terminating")
                sys.exit(0)
            dimension = len(points[0])
        
            dataset = np.zeros((n_points, dimension))
            for i in range(n_points):
                if len(points[i]) < dimension:
                    print("Insufficient dimension in line", i+skipped_lines, ", terminating")
                    sys.exit(0)
                for j in range(dimension):
                    dataset[i,j] = points[i][j]
        

            
        print("Number of data points:", n_points)
        print("Dimension:", dimension)
        print("Balance:", p, q)
        
        print("Constructing tree...")
        fairlet_s = time.time()
        root = scalable.build_quadtree(dataset)
        
        print("Doing fair clustering...")
        cost = scalable.tree_fairlet_decomposition(p, q, root, dataset, colors)
        fairlet_e = time.time()
        
        curr_degrees = []
        curr_costs = []
        curr_balances = []
        data=dl.data_list
        start_time = time.time()
        kcenters = KCenters(k=degree)
        kcenters.fit([data[i] for i in scalable.FAIRLET_CENTERS])
        mapping = kcenters.assign()
        final_clusters = []
        for fairlet_id, final_cluster in mapping:
            for point in scalable.FAIRLETS[fairlet_id]:
                final_clusters.append((point, scalable.FAIRLET_CENTERS[final_cluster]))
                
        centers = [scalable.FAIRLET_CENTERS[i] for i in kcenters.centers]
        cost=max([min([distance(data[j], i) for j in centers]) for i in data])
        balance=balance_calculation(data, centers, final_clusters)
        if verbose:
                print("Time taken for Degree %d - %.3f seconds."%(degree, time.time() - start_time))
        print(cost)
        print(balance)

    return balance,cost


       