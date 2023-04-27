import numpy as np
import matplotlib.pyplot as plt

def distance(a, b, order=2):
	"""
	Calculates the specified norm between two vectors.
	
	Args:
		a (list) : First vector
		b (list) : Second vector
		order (int) : Order of the norm to be calculated as distance
	
	Returns:
		Resultant norm value
	"""
	assert len(a) == len(b), "Length of the vectors for distance don't match."
	return np.linalg.norm(x=np.array(a)-np.array(b), ord=order)

def balance_calculation(data, centers, mapping):
	"""
	Checks fairness for each of the clusters defined by k-centers.
	Returns balance using the total and class counts.
	
	Args:
		data (list)
		centers (list)
		mapping (list) : tuples of the form (data, center)
		
	Returns:
		fair (dict) : key=center, value=(sum of 1's corresponding to fairness variable, number of points in center)
	"""
	fair = dict([(i, [0, 0]) for i in centers])
	for i in mapping:
		fair[i[1]][1] += 1
		if data[i[0]][0] == 1: # MARITAL
			fair[i[1]][0] += 1

	curr_b = []
	for i in list(fair.keys()):
		p = fair[i][0]
		q = fair[i][1] - fair[i][0]
		if p == 0 or q == 0:
			balance = 0
		else:
			balance = min(float(p/q), float(q/p))
		curr_b.append(balance)

	return min(curr_b)