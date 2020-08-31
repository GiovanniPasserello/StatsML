from math import *


class KMeansClusterer:
	"""
	A K-Means clusterer
	"""

	def __init__(self, centroids, points):
		self.centroids = centroids
		self.points = points
		self.assignments = [-1] * len(points)
		self.previous_assignments = [0] * len(points)  # used to check convergence
		self.distances = [[-1 for _ in range(len(centroids))] for _ in range(len(points))]

	def cluster(self):
		""" Run k means clusteringtill convergence

		Arguments:
			centroids {np.ndarray} -- A numpy array of K-dimensional centroid points
			points {np.ndarray} -- An N x K dimensional numpy array of points
		Returns:
			{np.ndarray} -- Converged centroids
			{np.ndarray} -- An N dimensional numpy array of point assignments to centroids (corresponding order to points)
			{np.ndarray} -- An N x K dimensional numpy array of euclidean distances of each point to each centroids
		"""

		while self.assignments != self.previous_assignments:
			self.previous_assignments = self.assignments.copy()
			self.assign()
			self.update()

		return self.centroids, self.assignments, self.distances

	def assign(self):
		for i, (p1, p2) in enumerate(self.points):
			for j, (c1, c2) in enumerate(self.centroids):
				self.distances[i][j] = self.euclidean(c1 - p1, c2 - p2)
		self.calculate_assignments()

	def calculate_assignments(self):
		for i, dists in enumerate(self.distances):
			self.assignments[i] = dists.index(min(dists)) + 1

	def update(self):
		self.centroids = [[0, 0] for _ in range(len(self.centroids))]
		for c in range(len(self.centroids)):
			assigned = [self.points[i] for i in range(len(self.points)) if self.assignments[i] == c + 1]

			if len(assigned) > 0:
				for (p1, p2) in assigned:
					self.centroids[c][0] += p1
					self.centroids[c][1] += p2

				self.centroids[c][0] = round(self.centroids[c][0] / len(assigned), 4)
				self.centroids[c][1] = round(self.centroids[c][1] / len(assigned), 4)

	def get_distances(self, centroid_num):
		return list(map(lambda ds: round(ds[centroid_num], 4), self.distances))

	@staticmethod
	def euclidean(x, y):
		return sqrt(x ** 2 + y ** 2)


def example_main():
	c1 = [0.06, 0.37]
	c2 = [1.75, 1.35]
	centroids = [c1, c2]

	x1 = [1.23, 0.83, 0.23, 1.51, -1.09, -0.5, -0.08, 1.49, -0.2, 2.26]
	x2 = [0.11, -0.59, 2.06, 1.35, 0.53, 1.01, 0.25, 1.83, -0.77, 0.88]
	points = list(zip(x1, x2))

	clusterer = KMeansClusterer(centroids, points)
	centroids, assignments, distances = clusterer.cluster()

	print("\nCentroids:", centroids)
	print("Assignments: {0}\n".format(assignments))
	for i in range(len(centroids)):
		print("Distances from centroid {0}: {1}".format(i + 1, clusterer.get_distances(i)))


if __name__ == "__main__":
	example_main()
