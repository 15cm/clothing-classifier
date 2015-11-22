import unittest

from cluster.kmeans_model import KmeansModel
from feature.sift import Sift
import os

CURPATH = os.path.split(os.path.realpath(__file__))[0]
DATAPATH = os.path.join(os.path.dirname(CURPATH),'src/dataset')

class MyTestCase(unittest.TestCase):
    def test_something(self):
        id_upper = 3900
        sift = Sift()
        kmeans = KmeansModel()
        image_list = sorted([x for x in os.listdir(DATAPATH) if os.path.splitext(x)[1] == '.jpg' and int(os.path.splitext(x)[0]) < id_upper],key=lambda x: int(os.path.splitext(x)[0]))
        print image_list
        descriptors_list = sift.compute(image_list)
        kmeans.fit(descriptors_list)
        kmeans.save('kmeans_sift')
        print 'ok'



if __name__ == '__main__':
    unittest.main()
