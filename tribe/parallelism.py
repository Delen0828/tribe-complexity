"""Bounded, ordered CPU preparation ahead of serial GPU inference."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor


def positive_threads(value):
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return count


def prefetch(function, rows, threads):
    """Keep at most `threads` jobs in flight and preserve manifest order."""
    if threads < 1:
        raise ValueError('threads must be positive')
    iterator = iter(rows)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        pending = deque()
        for _ in range(threads):
            row = next(iterator, None)
            if row is None:
                break
            pending.append(pool.submit(function, row))
        while pending:
            result = pending.popleft().result()
            row = next(iterator, None)
            if row is not None:
                pending.append(pool.submit(function, row))
            yield result


def batches(items, size):
    """Yield bounded groups, including the final partial batch."""
    if size < 1:
        raise ValueError('batch size must be positive')
    group = []
    for item in items:
        group.append(item)
        if len(group) == size:
            yield group
            group = []
    if group:
        yield group
