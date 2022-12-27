from math import sqrt
import queue
import collections

from shapely.geometry import Point


def dijkstra(start_tuple: tuple[tuple[int, int], Point, int],
             end_tuples: list[tuple[tuple[int, int], Point, int]],
             block: list[list[float | None]],
             find_nearest: bool) -> list[tuple[list[tuple[int, int]], list[float | None], list]]:
    """
    compute the distance between the starting point and any point of he
    :param start_tuple: Starting point (2-tuple of int, in coords of `block`) of the aggregation/length calculation
    :param end_tuples: Ending points(list of 2-tuple of ints) that should be reached
    :param block: The costs/weights as 2S-List of float | None (None == nodata)
    :param find_nearest: bool, True: only compute the path for the nearest point, False: Compute all points
    :return: list of the path, of costs (per path) and end point id
    """
    class Grid:
        def __init__(self, matrix: list[list[float]]):
            self.map = matrix
            self.h = len(matrix)
            self.w = len(matrix[0])

        def _in_bounds(self, _id: tuple[int, int]) -> bool:
            """
            Is the tuple inside the `block` coordinates
            :param _id: tuple[int, int] coords of the point in `block` coords
            :return: bool
            """
            x, y = _id
            return 0 <= x < self.h and 0 <= y < self.w

        def _passable(self, _id: tuple[int, int]) -> bool:
            """
            Can this point be reached (is a valid point, with valid weight?)
            :param _id: tuple[int, int] coords of the point in `block` coords
            :return: bool
            """
            x, y = _id
            return self.map[x][y] is not None

        def is_valid(self, _id: tuple[int, int]) -> bool:
            """
            Can this inside the block and can be reached
            :param _id:tuple[int, int] coords of the point in `block` coords
            :return: bool
            """
            return self._in_bounds(_id) and self._passable(_id)

        def neighbors(self, _id: tuple[int, int]) -> list[tuple[int, int]]:
            """
            Compute all valid neighbors of the current point
            :param _id: tuple[int, int] coords of the point in `block` coords
            :return: list[tuple[int, int]]
            """
            x, y = _id
            results = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1),
                       (x + 1, y - 1), (x + 1, y + 1), (x - 1, y - 1), (x - 1, y + 1)]
            results = list(filter(self.is_valid, results))
            return results

        def simple_cost(self, current: tuple[int, int], _next: tuple[int, int]) -> float:
            """
            Compute the euclidean distance between current and _next point
            :param current: tuple[int, int] coords of the point in `block` coords
            :param _next: tuple[int, int] coords of the point in `block` coords
            :return: float
            """
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
