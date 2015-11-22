import unittest
import os
from feature.superpixel import SuperPixel

CURPATH = os.path.split(os.path.realpath(__file__))[0]
DATAPATH = os.path.join(os.path.dirname(CURPATH),'/src/dataset')

class MyTestCase(unittest.TestCase):
    def test_something(self):
        sp = SuperPixel('3.jpg')
        sp.segment()
        sp.count_descriptors()
        pixel_list = sp.pixel_list
        print pixel_list


if __name__ == '__main__':
    unittest.main()
