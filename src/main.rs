use std::fs::File;
use std::io::Write;

use gcn_agg::accelerator::system::System;
use gcn_agg::{graph::Graph, node_features::NodeFeatures};
use ramulator_wrapper;
fn main() {
    let graph_name = "test_data/graph.txt";
    let features_name = "test_data/features.txt";
    let data = "f 3\n0 1 2\n1 2 0\n2 0 1\nend\n";
    let mut file = File::create("test_data/graph.txt").unwrap();
    file.write_all(data.as_bytes()).unwrap();
    let data = "0 0 1 0 1 0\n1 0 0 1 1 1\n1 1 0 0 0 1\n";
    let mut file = File::create("test_data/features.txt").unwrap();
    file.write_all(data.as_bytes()).unwrap();

    let mut graph = Graph::from(graph_name);
    let mut node_features = NodeFeatures::from(features_name);
    let mut system = System::new(
        &graph,
        &node_features,
        1,
        1,
        1,
        1,
        100,
        100,
        100,
        1,
        vec![1, 1],
    );
    system.run();
}
