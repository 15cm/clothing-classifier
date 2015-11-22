import unittest


from feature.sift import Sift
class MyTestCase(unittest.TestCase):
    def test_something(self):
        sift = Sift()
        des = sift.compute(['3.jpg','4.jpg'])
        print des


if __name__ == '__main__':
    unittest.main()
