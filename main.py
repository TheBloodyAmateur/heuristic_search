import copy
import heapq
import random
import time
from itertools import islice

from memory_profiler import memory_usage
from prettytable import PrettyTable


class HeuristicUtils:
    """
    Utility class for calculating different heuristic distances
    between the current state and the goal state of a sliding puzzle.
    """

    @staticmethod
    def manhattan(current: list[list[str]], goal: list[list[str]]) -> int:
        """
        Calculate the Manhattan distance between the current state `c` and the goal state `g`.
        The Manhattan distance is the sum of the absolute differences between the
        positions of each tile in the current state and its position in the goal state.
        :return int: The Manhattan distance.
        """
        distance = 0

        for current_x, current_row in enumerate(current):
            for current_y, current_element in enumerate(current_row):
                for goal_x, goal_row in enumerate(goal):
                    for goal_y, goal_element in enumerate(goal_row):
                        if current_element == goal_element and current_element != "_":
                            distance += abs(goal_x - current_x) + abs(goal_y - current_y)
        return distance

    @staticmethod
    def hamming(current: list[list[str]], goal: list[list[str]]) -> int:
        """
        Calculate the Hamming distance between the current state `c` and the goal state `g`.
        The Hamming distance is the number of tiles that are in the wrong position.

        :param current: Current state of the board as a 2D list.
        :param goal: Goal state of the board as a 2D list.
        :return int: The Hamming distance.
        """

        distance = 0
        for row1, row2 in zip(current, goal):
            if len(row1) != len(row2):
                raise ValueError("Arrays must have the same number of columns")
            for elem1, elem2 in zip(row1, row2):
                if elem1 != elem2:
                    distance += 1
        return distance


class AStarSearch(HeuristicUtils):
    @staticmethod
    def get_nearest_neighbours(state_to_explore: list[list[str]]) -> list[list[list[str]]]:
        """
        Generate all possible neighboring states by sliding tiles into the empty space.
        :param state_to_explore: Current state of the board as a 2D list.
        :return list: A list of neighboring states.
        """
        # valid moves: UP, DOWN, LEFT, RIGHT
        valid_moves: [list[set]] = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        # Generate the neighbors with the valid moves
        valid_state: list[list[list[str]]] = []

        # Iterate through the board to check for valid moves
        for state_x, state_row in enumerate(state_to_explore):
            for state_y, state_element in enumerate(state_row):
                if state_element == "_":
                    for move in valid_moves:
                        new_x = state_x + move[0] if state_x + move[0] <= 2 else -1
                        new_y = state_y + move[1] if state_y + move[1] <= 2 else -1
                        # If the new position is invalid (out of bounds --> -1), skip it, else switch the elements
                        if new_x != -1 and new_y != -1:
                            new_valid_state = copy.deepcopy(state_to_explore)
                            new_valid_state[state_x][state_y], new_valid_state[new_x][new_y] = new_valid_state[new_x][
                                new_y], new_valid_state[state_x][state_y]
                            valid_state.append(new_valid_state)

        return valid_state

    @staticmethod
    def reconstruct_path(state_info, goal_state):
        path = []
        current = goal_state
        while current is not None:
            path.append(current)
            current_tuple = tuple(tuple(row) for row in current)
            _, parent = state_info[current_tuple]
            current = parent
        return path[::-1]

    def isSolvable(self, board: list[list[str]]) -> bool:
        """
        Check if a given 8-puzzle board is solvable.
        An 8-puzzle is solvable if the number of inversions is even.

        :param board: 3x3 matrix representing the puzzle state.
        :return: True if the puzzle is solvable, False otherwise.
        """
        flat_board = [num for row in board for num in row if num != "_"]
        inversions = 0

        for i in range(len(flat_board)):
            for j in range(i + 1, len(flat_board)):
                if flat_board[i] > flat_board[j]:
                    inversions += 1

        return inversions % 2 == 0

    @staticmethod
    def a_star_hamming(current: list[list[str]], goal: list[list[str]], heuristic_function) -> list:
        """
        Perform the A* search algorithm to find the shortest path from the current state to the goal
        state with Hamming heuristic.

        :param heuristic_function: The heuristic function to use (Hamming or Manhattan).
        :param current: Current state of the board as a 2D list.
        :param goal: Goal state of the board as a 2D list.
        :return int: The cost of the shortest path.
        """

        if current == goal:
            print("Already at the goal state!")
            return []

        if not AStarSearch().isSolvable(current):
            print("The puzzle is not solvable.")
            return None

        heap = []
        visited_states = {}
        state_info = {}  # {state_tuple: (g(n), parent_state)}

        current_tuple = tuple(tuple(row) for row in current)
        visited_states[current_tuple] = 0
        state_info[current_tuple] = (0, None)

        # the heap to use, (f(n), g(n), h(n))
        heapq.heappush(heap, (0 + heuristic_function(current, goal), 0, current))

        while heap:
            f_n, g_n, state_to_explore = heapq.heappop(heap)
            if state_to_explore == goal:
                return g_n

            states_to_store: list[list[list[str]]] = AStarSearch.get_nearest_neighbours(state_to_explore)

            for neighbour in states_to_store:
                neighbour_tuple = tuple(tuple(row) for row in neighbour)
                # Each move is 1
                new_g_n = g_n + 1
                if neighbour_tuple not in visited_states or new_g_n < visited_states[neighbour_tuple]:
                    visited_states[neighbour_tuple] = new_g_n
                    state_info[neighbour_tuple] = (new_g_n, state_to_explore)
                    # (f(n), g(n), h(n))
                    heapq.heappush(heap, (new_g_n + heuristic_function(neighbour, goal), new_g_n, neighbour))
        return None


class GenerateUtils:
    @staticmethod
    def generate_solvable_matrices(count):
        """
        Generates a specified number of solvable 3x3 two-dimensional matrices for the 8-puzzle.
        Each matrix contains numbers from 1 to 8 and a blank space represented as an empty string.
        :param count: Number of solvable matrices to generate.
        :return: A generator yielding solvable 3x3 matrices.
        """
        for _ in range(count):
            matrix = GenerateUtils.generate2dmatrix()
            if AStarSearch().isSolvable([list(row) for row in matrix]):
                yield matrix

    @staticmethod
    def generate2dmatrix() -> list[list[str]]:
        """
        Generates a 3x3 two-dimensional matrix which randomly places numbers from 1 to 8 and a blank space.
        The blank space is represented as an empty string.

        :return: A 3x3 matrix with numbers 1-8 and a blank space.
        """
        elements = list(map(str, range(1, 9))) + ["_"]
        random.shuffle(elements)
        matrix = [
            elements[0:3],
            elements[3:6],
            elements[6:9]
        ]
        return matrix


class TestClass:
    memory_increase: float = 0.0
    max_memory: float = 0.0
    execution_time: float = 0.0
    heuristic_function = None

    def __init__(self, function):
        self.heuristic_function = function
        pass

    def test_function(self,
                      list_of_initial_matrix: list[list[list[str]]],
                      list_of_goal_matrix: list[list[list[str]]]):
        mem_usage = memory_usage(
            (self.test_a_star_function, (list_of_initial_matrix, list_of_goal_matrix)),
        )

        self.memory_increase = round(max(mem_usage) - min(mem_usage), 2)
        self.max_memory = round(max(mem_usage), 2)

        self.__print_results_table()

    def test_a_star_function(self,
                             list_of_initial_matrix: list[list[list[str]]],
                             list_of_goal_matrix: list[list[list[str]]]):
        """
        Test the A* search algorithm with the specified heuristic function on multiple test cases.
        :param list_of_initial_matrix:
        :param list_of_goal_matrix:
        :return:
        """
        i: int = 1
        self.start_time = time.time()

        for initial_state, goal_state in zip(list_of_initial_matrix, list_of_goal_matrix):
            AStarSearch().a_star_hamming(initial_state, goal_state, self.heuristic_function)
            i += 1

        self.end_time = time.time()
        self.execution_time = round(self.end_time - self.start_time, 2)

    def __print_results_table(self):
        table = PrettyTable(["Heuristic", "Time (s)", "Memory Increase (MiB)", "Peak Memory (MiB)"])
        table.add_row([self.heuristic_function.__name__, self.execution_time, self.memory_increase, self.max_memory])
        print(table)


if __name__ == "__main__":
    list_of_initial_matrices = list(islice(GenerateUtils.generate_solvable_matrices(200), 100))
    list_of_goal_matrices = list(islice(GenerateUtils.generate_solvable_matrices(200), 100))

    hamming_test: TestClass = TestClass(HeuristicUtils.hamming)
    hamming_test.test_function(list_of_initial_matrices, list_of_goal_matrices)

    print("\n" * 5)
    manhattan_test: TestClass = TestClass(HeuristicUtils.manhattan)
    manhattan_test.test_function(list_of_initial_matrices, list_of_goal_matrices)
