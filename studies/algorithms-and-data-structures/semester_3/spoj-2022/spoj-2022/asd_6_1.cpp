#include <iostream>
#include <vector>
#include <map>
#include <algorithm>

using DataSet = std::vector<std::pair<int, int>>;
using DataSets = std::vector<DataSet>;
using Conn = std::vector<int>;

struct Graph {
	int maxNode = 0;
	std::map<int, bool> visited;
	std::map<int, std::vector<int>> adj;
};
using Graphs = std::vector<Graph>;

void colectPairs(DataSet& dataSet, const int& dataPairNum) {
	for (int pair = 0; pair < dataPairNum; pair++) {
		int a, b;
		std::cin >> a >> b;
		dataSet.push_back(std::make_pair(a, b));
	}
}

void buildGraph(Graph& graph, const DataSet& dataSet) {
	for (int i = 0; i < dataSet.size(); i++) {
		int a = dataSet[i].first;
		int b = dataSet[i].second;
		graph.adj[a].push_back(b);
		graph.adj[b].push_back(a);

		if (graph.maxNode < a) { graph.maxNode = a; };
		if (graph.maxNode < b) { graph.maxNode = b; };
	}
}

void printResult(int& group, Conn& conn) {
	if (conn.size() == 0) { return; }
	std::sort(conn.begin(), conn.end());

	std::cout << group << ": ";

	for (const auto& node : conn) {
		std::cout << node << " ";
	}

	std::cout << std::endl;
	group++;
}

void findDFS(Graph& graph, int& node, Conn& conn) {
	graph.visited[node] = true;
	conn.push_back(node);

	for (int i = 0; i < graph.adj[node].size(); i++) {
		int nextNode = graph.adj[node][i];
		if (!graph.visited[nextNode]) {
			findDFS(graph, nextNode, conn);
		}
	}
}

void traverseGraph(Graph& graph) {
	int group = 1;

	for (int node = 1; node <= graph.maxNode; node++) {
		Conn conn;

		if (!graph.visited[node]) {
			findDFS(graph, node, conn);
		}

		printResult(group, conn);
	}
}

int solve61() {
	int dataSetsNum;
	DataSets dataSets;
	std::cin >> dataSetsNum;

	// Colect Input Data
	for (int set = 0; set < dataSetsNum; set++) {
		int dataPairsNum;
		DataSet dataSet;
		
		std::cin >> dataPairsNum;
		colectPairs(dataSet, dataPairsNum);
		dataSets.push_back(dataSet);
	}
	
	// Build Graphs
	Graphs graphs;
	for (int set = 0; set < dataSetsNum; set++) {
		Graph graph;
		buildGraph(graph, dataSets[set]);
		graphs.push_back(graph);
	}
	
	// Solve	
	for (int graphNum = 0; graphNum < dataSetsNum; graphNum++) {
		Graph currentGraph = graphs[graphNum];
		traverseGraph(currentGraph);
		std::cout << std::endl;
	}
	
	return 0;
}