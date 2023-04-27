from balance import distance
def kcenters_cost(data,centers):
    cost=max([min([distance(data[j], i) for j in centers]) for i in data])
    return cost
def kmeans_cost(data,centers):
    cost = 0
    for i in data:
        dist = min([distance(data[j], i) for j in centers]) ** 2
        cost += dist
    return cost

def kmedians_cost(data,centers):
    cost = 0
    for i in data:
        dist = min([distance(data[j], i) for j in centers])
        cost+=dist
    return cost