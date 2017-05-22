import unittest
from report_stat import ErrorStat


class TestErrorStat(unittest.TestCase):
    def testAdd(self):
        es = ErrorStat()
        es.add(0, 1)
        es.add(1, 1)
        es.add(0, 1)
        es.add(2, 1)
        es.add(0, 2)
        es.add(0, 2)

        self.assertEqual(3, len(es.records))

    def testTop(self):
        es = ErrorStat()
        es.add(0, 1)
        es.add(1, 1)
        es.add(0, 1)
        es.add(0, 1)
        es.add(2, 1)
        es.add(0, 2)
        es.add(0, 2)

        res = es.top(2)

        self.assertEqual(2, len(res))
        self.assertEqual(((0, 1), 3), res[0])
        self.assertEqual(((0, 2), 2), res[1])
