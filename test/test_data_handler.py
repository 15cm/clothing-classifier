import unittest

from data.data_handler import DataHandler

class MyTestCase(unittest.TestCase):
    def test_something(self):
        dh = DataHandler()
        dh.parse_data('design.json')


if __name__ == '__main__':
    unittest.main()
