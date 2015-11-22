import unittest

from cluster.kmeans_model import KmeansModel
from feature.sift import Sift

class MyTestCase(unittest.TestCase):
    def test_something(self):
        sift = Sift()
        kmeans = KmeansModel()
        image_list = ['3.jpg','4.jpg']
        descriptors_list = sift.compute(image_list)
        kmeans.fit(descriptors_list)
        kmeans.save()
        new_kmeans = KmeansModel()
        new_kmeans.load()
        watch = new_kmeans.kmeans
        print 'ok'



if __name__ == '__main__':
    unittest.main()
