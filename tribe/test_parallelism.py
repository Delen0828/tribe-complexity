"""Tests for bounded concurrent preparation and wall-clock runtime estimates."""
import argparse
import threading
import unittest
from parallelism import positive_threads, prefetch
from dataset import estimate_runtime


class ParallelismTests(unittest.TestCase):
    def test_thread_validation(self):
        self.assertEqual(positive_threads('8'),8)
        for value in ('0','-2'):
            with self.assertRaises(argparse.ArgumentTypeError):
                positive_threads(value)

    def test_concurrency_order_and_bound(self):
        barrier = threading.Barrier(3)
        lock = threading.Lock()
        active = 0
        peak = 0
        def work(value):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak,active)
            barrier.wait(timeout=5)
            with lock:
                active -= 1
            return value * 2
        self.assertEqual(list(prefetch(work,range(12),3)),list(range(0,24,2)))
        self.assertEqual(peak,3)

    def test_single_empty_and_exception(self):
        self.assertEqual(list(prefetch(lambda x:x,[],8)),[])
        self.assertEqual(list(prefetch(lambda x:x,range(4),1)),list(range(4)))
        with self.assertRaisesRegex(RuntimeError,'failed'):
            list(prefetch(lambda _: (_ for _ in ()).throw(RuntimeError('failed')), [0],8))

    def test_wall_time_not_sum_of_overlapping_tasks(self):
        receipts=[dict(encoded_windows=1,total_seconds=10,execution='prefetched')]*20
        benchmark=dict(count=20,processing_wall_seconds=80,thread=8,device='cuda')
        result=estimate_runtime(receipts,5800,1,benchmark)
        self.assertEqual(result['mean_seconds_per_stimulus'],4)
        self.assertEqual(result['full_prediction_seconds'],23201)
        self.assertEqual(result['thread'],8)
        self.assertEqual(estimate_runtime(receipts,5800,1)['measured_count'],0)


if __name__ == '__main__':
    unittest.main()
