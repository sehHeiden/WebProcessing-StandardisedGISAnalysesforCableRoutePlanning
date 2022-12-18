from math import sqrt
import queue
import collections

from shapely.geometry import Point


def dijkstra(start_tuple: tuple[tuple[int, int], Point, int], end_tuples: tuple[tuple[tuple[int, int], Point, int]],
             block: list[list[None | float]], find_nearest: bool):
    class Grid:
        def __init__(self, matrix: list[list[None | float]]):
            self.map = matrix
            self.h = len(matrix)
            self.w = len(matrix[0])
            self.manhattan_boundary = None
            self.curr_boundary = None

        def _in_bounds(self, _id: tuple[int, int]):
            x, y = _id
            return 0 <= x < self.h and 0 <= y < self.w

        def _passable(self, _id: tuple[int, int]):
            x, y = _id
            return self.map[x][y] is not None

        def is_valid(self, _id: tuple[int, int]):
            return self._in_bounds(_id) and self._passable(_id)

        def neighbors(self, _id: tuple[int, int]):
            x, y = _id
            results = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1),
                       (x + 1, y - 1), (x + 1, y + 1), (x - 1, y - 1), (x - 1, y + 1)]
            results = list(filter(self.is_valid, results))
            return results

        @staticmethod
        def manhattan_distance(id1: tuple[int, int], id2: tuple[int, int]):
            x1, y1 = id1
            x2, y2 = id2
            return abs(x1 - x2) + abs(y1 - y2)

        @staticmethod
        def min_manhattan(curr_node, end_nodes):
            return min(map(lambda node: Grid.manhattan_distance(curr_node, node), end_nodes))

        @staticmethod
        def max_manhattan(curr_node, end_nodes):
            return max(map(lambda node: Grid.manhattan_distance(curr_node, node), end_nodes))

        @staticmethod
        def all_manhattan(curr_node, end_nodes):
            return {end_node: Grid.manhattan_distance(curr_node, end_node) for end_node in end_nodes}

        def simple_cost(self, current: tuple[int, int], _next: tuple[int, int]):
            cx, cy = current
            nx, ny = _next
            current_value = self.map[cx][cy]
            offset_value = self.map[nx][ny]
            if cx == nx or cy == ny:
                return (current_value + offset_value) / 2
            else:
                return sqrt(2) * (current_value + offset_value) / 2

    result = []
    grid = Grid(block)

    end_dict = collections.defaultdict(list)
    for end_tuple in end_tuples:
        end_dict[end_tuple[0]].append(end_tuple)
    end_row_cols = set(end_dict.keys())
    end_row_col_list = list(end_row_cols)
    start_row_col = start_tuple[0]

    frontier = queue.PriorityQueue()
    frontier.put((0, start_row_col))
    came_from = {}
    cost_so_far = {}
    decided = set()

    if not grid.is_valid(start_row_col):
        return result

    # init progress
    came_from[start_row_col] = None
    cost_so_far[start_row_col] = 0

    while not frontier.empty():
        _, current_node = frontier.get()
        if current_node in decided:
            continue
        decided.add(current_node)

        # reached destination
        if current_node in end_row_cols:
            path = []
            costs = []
            traverse_node = current_node
            while traverse_node is not None:
                path.append(traverse_node)
                costs.append(cost_so_far[traverse_node])
                traverse_node = came_from[traverse_node]

            # start point and end point overlaps
            if len(path) == 1:
                path.append(start_row_col)
                costs.append(0.0)
            path.reverse()
            costs.reverse()
            result.append((path, costs, end_dict[current_node]))

            end_row_cols.remove(current_node)
            end_row_col_list.remove(current_node)
            if len(end_row_cols) == 0 or find_nearest:
                break

        # relax distance
        for nex in grid.neighbors(current_node):
            new_cost = cost_so_far[current_node] + grid.simple_cost(current_node, nex)
            if nex not in cost_so_far or new_cost < cost_so_far[nex]:
                cost_so_far[nex] = new_cost
                frontier.put((new_cost, nex))
                came_from[nex] = current_node

    return result
