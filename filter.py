from collections import deque

class Filter:
    def __init__(self, max_length=10):
        self.history = deque(maxlen=max_length)
        self.last_result = None

    def clear(self):
        self.history.clear()
        self.last_result = None

    def filter(self, results):
        if results is None:
            return None

        self.history.append(results)

        if len(self.history) != self.history.maxlen:
            return None

        count_dict = {}
        for i in self.history:
            if i in count_dict:
                count_dict[i] += 1
            else:
                count_dict[i] = 1

        max_count = 0
        max_result = None
        for k, v in count_dict.items():
            if v > max_count:
                max_count = v
                max_result = k

        if max_count >= len(self.history) * 6 / 10:
            return max_result

        return None
