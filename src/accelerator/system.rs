use std::array::IntoIter;

use super::agg_buffer::{self, AggBuffer};
use super::aggregator::Aggregator;
use super::input_buffer::InputBuffer;
use super::mem_interface::MemInterface;
use super::output_buffer::OutputBuffer;
use crate::{graph::Graph, node_features::NodeFeatures};
struct System {
    finished: bool,
    total_cycle: u64,
    aggregator: Aggregator,
    input_buffer: InputBuffer,
    output_buffer: OutputBuffer,
    agg_buffer: AggBuffer,
    mem_interface: MemInterface,
    graph: Graph,
    node_features: NodeFeatures,

    input_buffer_size: usize,
    agg_buffer_size: usize,
    output_buffer_size: usize,
}
#[derive(Debug, Clone, PartialEq)]
struct Window {}
struct WindowIterator<'a> {
    graph: &'a Graph,
    node_features: &'a NodeFeatures,
    current_task_id: usize,
    input_buffer_size: usize,
    agg_buffer_size: usize,
    output_buffer_size: usize,

    // current window information
    current_window_start_x: usize,
    current_window_end_x: usize,
    current_window_start_y: usize,
    current_window_end_y: usize,


}
// impl new for WindowIterator
impl<'a> WindowIterator<'a> {
    fn new(
        graph: &'a Graph,
        node_features: &'a NodeFeatures,
        input_buffer_size: usize,
        agg_buffer_size: usize,
        output_buffer_size: usize,
    ) -> Self {
        WindowIterator {
            graph,
            node_features,
            current_task_id: 0,
            input_buffer_size,
            agg_buffer_size,
            output_buffer_size,
            
            current_window_end_x: 0,
            current_window_end_y: 0,
            current_window_start_x: 0,
            current_window_start_y: 0,

        }
    }
}

impl<'a> Iterator for WindowIterator<'a> {
    type Item = Window;
    fn next(&mut self) -> Option<Self::Item> {
        self.current_task_id += 1;
        if self.current_task_id < self.node_features.len() {
            Some(Window {})
        } else {
            None
        }
    }
}
impl System {
    pub fn new(
        sparse_cores: usize,
        spase_width: usize,
        dense_cores: usize,
        dense_width: usize,
        graph_name: &str,
        features_name: &str,
        input_buffer_size: usize,
        agg_buffer_size: usize,
        output_buffer_size: usize,
    ) -> System {
        let graph = Graph::from(graph_name);
        let node_features = NodeFeatures::from(features_name);
        let aggregator = Aggregator::new(sparse_cores, spase_width, dense_cores, dense_width);

        let input_buffer = InputBuffer::new();
        let output_buffer = OutputBuffer::new();
        let agg_buffer = AggBuffer::new();
        let mem_interface = MemInterface::new(64, 64);

        System {
            finished: false,
            total_cycle: 0,
            aggregator,
            input_buffer,
            output_buffer,
            agg_buffer,
            mem_interface,
            graph,
            node_features,
            input_buffer_size,
            agg_buffer_size,
            output_buffer_size,
        }
    }

    /// # Description
    /// * this function will schedule the requests of memory and aggregator
    /// * will update each component's status
    /// * will ***NOT*** update the cycle
    ///
    pub fn cycle(&mut self) {
        // this code is for testing only!
        self.input_buffer.cycle();
        self.output_buffer.cycle();
        self.agg_buffer.cycle();
        self.mem_interface.cycle();
        self.aggregator.cycle();
    }
    /// # Description
    /// keep running until all finished
    /// * for each cycle, it will call the cycle function
    /// * and increase the total_cycle
    pub fn run(&mut self) {
        while !self.finished {
            self.cycle();
            self.total_cycle += 1;
        }
        self.print_stats();
    }

    pub fn finished(&self) -> bool {
        self.finished
    }
    fn print_stats(&self) {
        println!("Total cycles: {}", self.total_cycle);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::{fs::File, io::Write};
    #[test]
    fn test_system() {
        let graph_name = "test_data/graph.txt";
        let features_name = "test_data/features.txt";
        let data = "f 3\n0 1 2\n1 2 0\n2 0 1\nend\n";
        let mut file = File::create("test_data/graph.txt").unwrap();
        file.write_all(data.as_bytes()).unwrap();
        let data = "0 0 1 0 1 0\n1 0 0 1 1 1\n1 1 0 0 0 1\n";
        let mut file = File::create("test_data/features.txt").unwrap();
        file.write_all(data.as_bytes()).unwrap();

        let mut system = System::new(1, 1, 1, 1, graph_name, features_name, 1, 1, 1);
        system.run();
        assert_eq!(system.finished(), true);
    }
    #[test]
    fn window_iter_test() {
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
        let mut window_iter = WindowIterator::new(&graph, &node_features, 1, 1, 1);
        for i in window_iter {
            println!("{:?}", i);
        }
    }
}
