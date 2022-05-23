from heapq import heappush
from heapq import heappop

import numpy as np

class Heap:

    def __init__(self,
                 random=None):
        if not isinstance(random, np.random.RandomState):
            random = np.random.RandomState(random)
        self._random = random
        self._pq = []
        self._REMOVED = '<removed-task>'
        self._element_finder = {}
        self._n = 0

    def push(self,
             priority,
             element,
             extra=None):
        if element in self._element_finder:
            self.remove(element)
        tiebreaker = self._random.random()
        entry = [priority, tiebreaker, element, extra]
        self._element_finder[element] = entry
        heappush(self._pq, entry)
        self._n += 1

    def push_if_better(self,
                       priority,
                       element,
                       extra=None):
        if element in self._element_finder:
            if self._element_finder[element][0] <= priority:
                # The better element is already in the queue. Do nothing
                return 
            else:
                self.remove(element)
        tiebreaker = self._random.random()
        entry = [priority, tiebreaker, element, extra]
        self._element_finder[element] = entry
        heappush(self._pq, entry)
        self._n += 1

    def remove(self,
               element):
        entry = self._element_finder.pop(element)
        entry[-2] = self._REMOVED
        self._n -= 1

    def pop(self):
        while len(self._pq) > 0:
            priority, _, element, extra = heappop(self._pq)
            if element is not self._REMOVED:
                del self._element_finder[element]
                self._n -= 1
                return priority, element, extra
        raise KeyError('Tried to pop from an empty priority queue')

    def __len__(self):
        return self._n
