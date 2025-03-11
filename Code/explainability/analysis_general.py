import numpy as np
import networkx as nx
from torch_geometric.nn import MessagePassing
import math
from torch_geometric.utils import to_networkx

class GraphAnalyzer_general:
    def __init__(self, model,  k=100, c0=10, delta=1, gamma=1, theta=1, alpha=1, beta=1, epsilon=1, number_of_graphs=2):

        self.number_of_graphs = number_of_graphs
        self.model = model
        self.num_hops = self.calculate_num_hops()
        
        # C-puct base value
        self.c0 = c0


        """HIPERPARÁMETROS A OPTIMIZAR POR AUTOGOAL, SON TODOS FLOAT"""

        # Sample num hyperparameters
        self.k = k

        # Local Radius Hyperparameters
        self.alpha = alpha
        self.beta = beta

        # C_puct Hyperparameters (exploration-exploitation trade-off)
        self.delta = delta

        # Rollout Hyperparameters
        self.gamma = gamma
        
        # Expand Atoms Hyperparameters
        self.theta = theta

        # Max nodes hyperparameters
        self.epsilon = epsilon

    def validate_positive_N(self, N):
        if N <= 0:
            raise ValueError("N must be greater than 0 to calculate the logarithm")

    def calculate_num_hops(self):
        num_hops = sum(1 for module in self.model.modules() if isinstance(module, MessagePassing))

        num_hops = num_hops // self.number_of_graphs

        return num_hops
        
    def calculate_sample_num(self, N, min_atoms, k):
        self.validate_positive_N(N)
        log_N_plus_1 = np.log(N + 1)
        scale_factor = N / (N + k)
        denominator = 1 - scale_factor
        sample_num = min_atoms + log_N_plus_1 / denominator
        return int(min(N, max(sample_num, min_atoms)))


    def calculate_max_nodes(self, N, D, avg_degree_norm, std_degree_norm, min_atoms, epsilon):
        """
        Calculate the max_nodes parameter for SubgraphX.

        Args:
            N (int): Number of nodes in the graph.
            D (float): Density of the graph.
            avg_degree_norm (float): Normalized average degree.
            std_degree_norm (float): Normalized standard deviation of degrees.
            min_atoms (int): The minimum number of atoms for subgraphs.
            min_max_nodes (int): Minimum value for max_nodes.
            max_max_nodes (int): Maximum value for max_nodes.

        Returns:
            int: Calculated max_nodes value.
        """
        # Base max_nodes on 20% of N
        base_max_nodes = int(0.2 * N)

        # Adjust based on graph metrics
        complexity_score = (D + avg_degree_norm + std_degree_norm) / 3
        adjusted_max_nodes = base_max_nodes * (epsilon + complexity_score)

        # Ensure max_nodes is greater than min_atoms by at least delta
        delta = max(5, int(0.05 * N))  # Ensure delta scales with N
        max_nodes = int(max(adjusted_max_nodes, min_atoms + delta))

        return max_nodes

    def calculate_diameter(self, graph):
        try:
            return nx.diameter(graph)
        except nx.NetworkXError:
            # If the graph is not connected, use the diameter of the largest component
            largest_cc = max(nx.connected_components(graph), key=len)
            subgraph = graph.subgraph(largest_cc)
            return nx.diameter(subgraph) 

    def calculate_local_radius(self, N, M, diameter, local_radius_method, alpha, beta):
        """
        Calculate the local radius based on the given parameters.
        Parameters:
        -----------
        N : int
            The number of nodes.
        M : int
            The number of edges.
        diameter : float
            The diameter of the network.
        local_radius_method : str or float
            The scaling factor for the local radius. If 'log', it will be calculated as the logarithm of N.
            If 'step', it will be calculated based on predefined steps. Otherwise, it should be a numeric value.
        Returns:
        --------
        float
            The calculated local radius, with a minimum value of 1.
        """
        self.validate_positive_N(N)

        # Calculate local radius based on N
        if local_radius_method == 'log':
            local_radius_method = math.log10(N)
        
        elif local_radius_method == 'step':
            if N <= 10:
                local_radius_method = 1
            elif N > 10 and N <= 100:
                local_radius_method = int(1 + 0.015*(N-10))
            elif N > 100:
                local_radius_method = int(2.35+0.005*(N-100))

        # Consider also the diameter, average degree, and the scaling factors alpha and beta
        average_degree = (2 * M) / N
        local_radius = local_radius_method * (alpha*diameter / (beta*average_degree))
        return max(1, local_radius)

    def calculate_rollout(self, N, gamma, rollout_method):
        """
        Calculate the rollout value based on the given parameters.
        Parameters:
        N (int): A positive integer value representing the number of iterations.
        gamma (float): A multiplier used in the calculation of alpha and rollout.
        rollout_method (str): The method to calculate alpha. It can be 'log' or 'step'.
        Returns:
        int: The calculated rollout value.
        Raises:
        ValueError: If N is not a positive integer.
        Notes:
        - If rollout_method is 'log', alpha is calculated as gamma * log10(N).
        - If rollout_method is 'step', alpha is calculated based on the value of N:
            - If N <= 10, alpha is 1.
            - If 10 < N <= 100, alpha is 1 + 0.015 * (N - 10).
            - If N > 100, alpha is 2.35 + 0.005 * (N - 100).
        """
        self.validate_positive_N(N)
        # Calcular alfa basado en N
        if rollout_method == 'log':
            alpha = gamma * math.log10(N)
        
        elif rollout_method == 'step':
            if N <= 10:
                alpha = 1
            elif N > 10 and N <= 100:
                alpha = int(1 + 0.015*(N-10))
            elif N > 100:
                alpha = int(2.35+0.005*(N-100))
        
        rollout = int(alpha * N * (1 + gamma))
        return rollout

    def calculate_c_puct(self, graph, c0, delta):
        """
        Calculate the c_puct value for a given graph.

        Parameters:
        graph (networkx.Graph): The input graph for which the c_puct value is calculated.
        c0 (float): The base value of c_puct.
        delta (float): The adjustment factor for c_puct based on graph metrics.

        Returns:
        float: The adjusted c_puct value.

        The function calculates the c_puct value based on the following steps:
        1. Calculate the number of nodes (N) and edges (E) in the graph.
        2. If the graph has 1 or fewer nodes, return the base value c0.
        3. Calculate the density (D) of the graph.
        4. Calculate the average degree and standard deviation of the degrees of the nodes.
        5. Normalize the average degree and standard deviation using logarithms.
        6. Limit the normalized values to the range [0, 1].
        7. Compute the mean of the normalized metrics.
        8. Adjust the c_puct value based on the mean of the normalized metrics.
        9. Ensure that the adjusted c_puct value is not less than 50% of the base value c0.
        """

        N = graph.number_of_nodes()
        E = graph.number_of_edges()


        if N <= 1:
            return c0

        D = nx.density(graph)  # Densidad del grafo

        # Calcular grado promedio
        avg_degree = (2 * E) / N

        # Calcular desviación estándar del grado
        degrees = [degree for node, degree in graph.degree()]
        std_degree = np.std(degrees)

        # Normalizar grado promedio y desviación estándar usando logaritmo
        avg_degree_norm = np.log(avg_degree + 1) / np.log(N)
        std_degree_norm = np.log(std_degree + 1) / np.log(N)

        # Limitar los valores normalizados a [0, 1]
        avg_degree_norm = min(max(avg_degree_norm, 0), 1)
        std_degree_norm = min(max(std_degree_norm, 0), 1)

        # Calcular el promedio de las métricas normalizadas
        metrics_mean = (D + avg_degree_norm + std_degree_norm) / 3

        # Ajustar c_puct
        c_puct = c0 * (1 + delta * (metrics_mean - 0.5))

        # Asegurar que c_puct no sea menor que una fracción del valor base (por ejemplo, no menos del 50%)
        c_puct = max(c_puct, c0 * 0.5)

        return c_puct

    def calculate_expand_atoms(self, graph, theta):
        """
        Calculate the expand_atoms value for a given graph and theta.

        This function computes a metric called expand_atoms based on the properties
        of the input graph and a parameter theta. The metric is influenced by the 
        average degree, the standard deviation of the degree, and the density of the graph.

        Parameters:
        graph (networkx.Graph): The input graph for which the expand_atoms value is calculated.
        theta (float): A parameter that influences the calculation of expand_atoms.

        Returns:
        int: The calculated expand_atoms value, which is at least 1.
        """
        N = graph.number_of_nodes()
        E = graph.number_of_edges()

        # Evitar divisiones por cero
        if N <= 1:
            return 1

        # Calcular grado promedio
        avg_degree = (2 * E) / N

        # Calcular desviación estándar del grado
        degrees = [degree for node, degree in graph.degree()]
        std_degree = np.std(degrees)

        # Normalizar grado promedio y desviación estándar
        avg_degree_norm = np.log(avg_degree + 1) / np.log(N)
        std_degree_norm = np.log(std_degree + 1) / np.log(N)

        # Limitar los valores normalizados a [0, 1]
        avg_degree_norm = min(avg_degree_norm, 1)
        std_degree_norm = min(std_degree_norm, 1)

        # Calcular densidad
        D = nx.density(graph)  # Ya está en el rango [0, 1]

        # Calcular expand_atoms
        expand_atoms = 1 * (theta * (avg_degree_norm + D + std_degree_norm) / 3)

        # Asegurar que expand_atoms sea al menos 1 y entero
        expand_atoms = max(int(expand_atoms), 1)

        return expand_atoms

    def calculate_graph_parameters(self, data, rollout_method, gamma, c0, delta, theta, 
                                   local_radius_method, k):
        graph = to_networkx(data, to_undirected=True)
        N = graph.number_of_nodes()
        if N == 0:
            return None

        # Parámetros que no dependen de hiperparámetros a calcular
        self.validate_positive_N(N)
        M = graph.number_of_edges()
        D = nx.density(graph)
        avg_degree = (2 * M) / N
        avg_degree_norm = min(max(np.log(avg_degree + 1) / np.log(N), 0), 1)
        degrees = np.fromiter((degree for _, degree in graph.degree()), dtype=float)
        std_degree = np.std(degrees)
        std_degree_norm = min(max(np.log(std_degree + 1) / np.log(N), 0), 1)
        diameter = self.calculate_diameter(graph)

        
       # Parámetros que dependen de hiperparámetros a calcular 
        min_atoms = max(1, int(round(0.1 * N)))  # Adjusted to 10% of N
        sample_num = self.calculate_sample_num(N, min_atoms, k)
        local_radius = self.calculate_local_radius(N, M, diameter, local_radius_method, self.alpha, self.beta)
        rollout = self.calculate_rollout(N, gamma, rollout_method)
        c_puct = self.calculate_c_puct(graph, c0, delta)
        expand_atoms = self.calculate_expand_atoms(graph, theta)
        max_nodes = self.calculate_max_nodes(N, D, avg_degree_norm, std_degree_norm, min_atoms, self.epsilon)

        return {
            'num_hops': self.num_hops,
            'rollout': rollout,
            'min_atoms': min_atoms,
            'c_puct': c_puct,
            'expand_atoms': expand_atoms,
            'local_radius': local_radius,
            'sample_num': sample_num,
            'max_nodes': max_nodes,
        }

    def calculate_subgraphx_parameters(self, graphs, num_hops=None):
        if num_hops is not None:
            self.num_hops = num_hops

        rollout_method = 'step'
        local_radius_method = 'step'

        parameters_list = []
        for data in graphs:
            params = self.calculate_graph_parameters(
                data, rollout_method, self.gamma, self.c0, self.delta, self.theta, local_radius_method, self.k)
            if params is not None:
                parameters_list.append(params)

        return parameters_list
    
    def final_parameters(self, parameters_list):
        """
        Aggregates a list of parameter dictionaries into a single dictionary with averaged values.
        Args:
            parameters_list (list of dict): A list of dictionaries where each dictionary contains the following keys:
                - 'num_hops' (int): Number of hops.
                - 'rollout' (float): Rollout value.
                - 'min_atoms' (float): Minimum number of atoms.
                - 'c_puct' (float): c_puct value.
                - 'expand_atoms' (float): Number of atoms to expand.
                - 'local_radius' (float): Local radius value.
                - 'sample_num' (float): Number of samples.
                - 'max_nodes' (float): Maximum number of nodes.
        Returns:
            dict: A dictionary with the same keys as the input dictionaries, where the values are the aggregated (averaged) values from the input list.
        """
        
        Aggregated_num_hops = parameters_list[0]['num_hops']
        Aggregated_rollout = int(np.round(sum([parameters_list[i]['rollout'] for i in range(len(parameters_list))])/len(parameters_list)))
        Aggregated_min_atoms = int(np.round(sum([parameters_list[i]['min_atoms'] for i in range(len(parameters_list))])/len(parameters_list)))
        Aggregated_c_puct = sum([parameters_list[i]['c_puct'] for i in range(len(parameters_list))])/len(parameters_list)
        Aggregated_expand_atoms = int(np.round(sum([parameters_list[i]['expand_atoms'] for i in range(len(parameters_list))])/len(parameters_list)))
        Aggregated_local_radius = int(np.round(sum([parameters_list[i]['local_radius'] for i in range(len(parameters_list))])/len(parameters_list)))
        Aggregated_sample_num = int(np.round(sum([parameters_list[i]['sample_num'] for i in range(len(parameters_list))])/len(parameters_list)))
        Aggregated_max_nodes = int(np.round(sum([parameters_list[i]['max_nodes'] for i in range(len(parameters_list))])/len(parameters_list)))

        return {    
            'num_hops': Aggregated_num_hops,                   # int
            'rollout': Aggregated_rollout,                     # int
            'min_atoms': Aggregated_min_atoms,                 # int
            'c_puct': Aggregated_c_puct,                       # float
            'expand_atoms': Aggregated_expand_atoms,           # int
            'local_radius': Aggregated_local_radius,           # int
            'sample_num': Aggregated_sample_num,               # int
            'max_nodes': Aggregated_max_nodes,                 # int
        }