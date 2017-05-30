from bisect import bisect_right


class CustomHeap:
    """
    Note: We can specify the heap to be min or max heap.
     However, our implementation inside is min heap, so we multiply score by -1 if heap type is max heap.
    """
    def __init__(self, max_size, heap_type='max'):
        self.max_size = max_size
        self.items = list()
        if heap_type == 'max':
            self.multiplier = -1
        else:
            self.multiplier = 1

    def add(self, new_item):
        (score, item) = new_item
        score *= self.multiplier

        can_add = (len(self.items) < self.max_size) or (self.items[-1][0] > score)
        if can_add:
            score_list = [x[0] for x in self.items]
            insert_pos = bisect_right(score_list, score)
            self.items.insert(insert_pos, (score, item))
            if len(self.items) > self.max_size:
                self.items = self.items[0:self.max_size]

    def get_items_with_score(self):
        correct_list = [(self.multiplier * score, item) for (score, item) in self.items]
        return correct_list

    def get_items_only(self):
        items_with_score = self.get_items_with_score()
        items_only = [item for (score, item) in items_with_score]
        return items_only
