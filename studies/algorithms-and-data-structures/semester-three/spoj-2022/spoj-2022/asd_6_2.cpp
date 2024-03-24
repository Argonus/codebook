#include <iostream>;
#include <vector>;
#include <map>;
#include <set>;

using Edge = std::pair<int, int>;
using EdgeVal = std::pair<Edge, int>;
using Edges = std::map<Edge, int>;
const int NaN = 10000007;

class Graph {
	// Intercace
	public:
		signed int max;
		Edges edges;
		std::set<int> nodes;

		Graph() {
			max = -1;
		}
		void addEdge(int a, int b) {
			Edge edge = std::make_pair(a, b);
			EdgeVal edgeVal = std::make_pair(edge, 1);
			edges.insert(edgeVal);
			nodes.insert(a);
			nodes.insert(b);

			if (a > max) { max = a; }
			if (b > max) { max = b; }
		}
};

class GraphMatrix {
	public:
		GraphMatrix(const Graph& graph) {
			size = graph.max - 1;
			buildEmptyMatrix();
			fillDirectConnections(graph);
			fillChildConnections();
		};

		void printMatrix() {
			for (int i = 0; i <= size; i++) {
				for (int j = 0; j <= size; j++) {
					if (matrix[i][j] == NaN) { std::cout << "NaN ";}
					else { std::cout << matrix[i][j] << " ";}
				}
				std::cout << std::endl;
			}
		}

	private:
		int size;
		std::vector<std::vector<int>> matrix;

		void buildEmptyMatrix() {
			for (int i = 0; i <= size; i++) {
				std::vector<int> row;

				for (int j = 0; j <= size; j++) { 
					row.push_back(NaN);
				}

				matrix.push_back(row);
			}
		}

		void fillDirectConnections(const Graph& graph) {
			const Edges edges = graph.edges;
			const std::set<int> nodes = graph.nodes;

			// Fill diagonal
			for (int node = 1; node <= graph.max; node++) {
				int idx = node - 1;
				matrix[idx][idx] = 0; 
			}
			// Fill direct connections
			for (int nodeOne = 1; nodeOne <= graph.max; nodeOne++) {
				for (int nodeTwo = 1; nodeTwo <= graph.max; nodeTwo++) {
					int idxOne = nodeOne - 1;
					int idxTwo = nodeTwo - 1;

					if (edges.find({ nodeOne, nodeTwo }) != edges.end()) { 
						matrix[idxOne][idxTwo] = 1; 
						matrix[idxTwo][idxOne] = 1;
					}

					if (edges.find({ nodeTwo, nodeOne }) != edges.end()) {
						matrix[idxOne][idxTwo] = 1;
						matrix[idxTwo][idxOne] = 1;
					}
				}
			}
		}

		void fillChildConnections() {
			for (int i = 0; i <= size; i++) {
				for (int j = 0; j <= size; j++) {
					for (int k = 0; k <= size; k++) {
						bool valsExist = matrix[i][k] != NaN && matrix[j][i] != NaN;

						if (valsExist && (matrix[j][k] > (matrix[j][i] + matrix[i][k]))) {
							matrix[j][k] = matrix[j][i] + matrix[i][k];
						}
					}
				}
			}
		}
};

void colectPairs(Graph& graph, const int& dataPairNum) {
	for (int pair = 0; pair < dataPairNum; pair++) {
		int a, b;
		std::cin >> a >> b;
		graph.addEdge(a, b);
	}
}

int solve62() {
	short int dataSetsNum;
	std::vector<Graph> graphsList;
	std::cin >> dataSetsNum;

	// Colect Input Data
	for (int set = 0; set < dataSetsNum; set++) {
		int dataPairsNum;
		Graph graph;

		std::cin >> dataPairsNum;
		colectPairs(graph, dataPairsNum);
		graphsList.push_back(graph);
	}
	// Build Matrix based on graph
	std::vector<GraphMatrix> matrixSet;
	for (int set = 0; set < dataSetsNum; set++) {
		Graph currentGraph = graphsList[set];
		GraphMatrix matrix = GraphMatrix(currentGraph);
		matrixSet.push_back(matrix);
	}
	// Print Results
	for (int set = 0; set < dataSetsNum; set++) {
		GraphMatrix matrix = matrixSet[set];
		matrix.printMatrix();
		std::cout << std::endl;
	}

	return 0;
}