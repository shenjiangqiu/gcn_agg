/// # Description
/// the state for the system
/// * `Idle` means this is the very first of each layer, need to init the new output and input iter
/// * `Working` means every thing is ok, can correctly use next input
/// * `NoInputIter` means no next input iter,need to get next input layer and get next output iter
/// * `NoWindow` means no next window for this input iter, need to get next input iter from the output iter
/// * `Finished` means all layer is finished
#[derive(Debug)]
enum SystemState {
    Working,
    NoInputIter,
    NoWindow,
    Finished,
}

use super::{
    agg_buffer::AggBuffer,
    aggregator::Aggregator,
    input_buffer::{self, InputBuffer},
    mem_interface::MemInterface,
    output_buffer::OutputBuffer,
    sliding_window::{InputWindowIterator, OutputWindowIterator, Window},
};

use crate::{graph::Graph, node_features::NodeFeatures};
#[derive(Debug)]
pub struct System<'a> {
    state: SystemState,
    finished: bool,
    total_cycle: u64,
    aggregator: Aggregator,
    input_buffer: InputBuffer<'a>,
    output_buffer: OutputBuffer,
    agg_buffer: AggBuffer,
    mem_interface: MemInterface,
    graph: &'a Graph,
    node_features: &'a NodeFeatures,

    input_buffer_size: usize,
    agg_buffer_size: usize,
    output_buffer_size: usize,
    current_layer: usize,
    current_output_iter: Option<OutputWindowIterator<'a>>,
    current_input_iter: Option<InputWindowIterator<'a>>,
    current_window: Option<Window<'a>>,
    gcn_layer_num: usize,
    gcn_hidden_size: Vec<usize>,
}

impl<'a> System<'a> {
    pub fn new(
        graph: &'a Graph,
        node_features: &'a NodeFeatures,
        sparse_cores: usize,
        spase_width: usize,
        dense_cores: usize,
        dense_width: usize,
        input_buffer_size: usize,
        agg_buffer_size: usize,
        output_buffer_size: usize,
        gcn_layer_num: usize,
        gcn_hidden_size: Vec<usize>,
    ) -> System<'a> {
        let aggregator = Aggregator::new(sparse_cores, spase_width, dense_cores, dense_width);

        let input_buffer = InputBuffer::new();
        let output_buffer = OutputBuffer::new();
        let agg_buffer = AggBuffer::new();
        let mem_interface = MemInterface::new(64, 64);
        let mut current_output_iter =
            OutputWindowIterator::new(graph, node_features, agg_buffer_size, input_buffer_size, 0);
        let mut current_input_iter = current_output_iter.next();
        let current_window = match current_input_iter.as_mut() {
            Some(iter) => iter.next(),
            None => None,
        };

        let state = match current_window {
            Some(_) => SystemState::Working,
            None => match current_input_iter {
                Some(_) => SystemState::NoWindow,
                None => SystemState::NoInputIter,
            },
        };
        System {
            state,
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
            current_layer: 0,
            current_output_iter: Some(current_output_iter),
            current_input_iter,
            current_window,
            gcn_layer_num,
            gcn_hidden_size,
        }
    }

    /// # Description
    /// * this function will schedule the requests of memory and aggregator
    /// * will update each component's status
    /// * will ***NOT*** update the cycle
    ///
    pub fn cycle(&mut self) {
        match &self.state {
            SystemState::Working => {
                self.aggregator.cycle();
                self.mem_interface.cycle();
                self.agg_buffer.cycle();
                self.input_buffer.cycle();
                self.output_buffer.cycle();

                match &self.input_buffer.next_state {
                    input_buffer::BufferStatus::Empty => {
                        // add a task to the input buffer
                        // self.input_buffer.send_req(self.current_input_iter.as_ref().unwrap());
                        let window = self.current_window.as_ref().unwrap();
                        let req_id = window.get_task_id();
                        self.input_buffer.add_task(req_id.clone(), window.clone());
                    }
                    input_buffer::BufferStatus::WaitingToLoad(req, window) => {
                        if self.mem_interface.available() {
                            // generate addr from the req and window

                            let mut addr_vec = vec![];
                            let start_addrs = &self.node_features.start_addrs;
                            let mut start_addr = start_addrs[window.start_x];
                            let end_addr = start_addrs[window.end_x];
                            // round start_addr to the nearest 64
                            start_addr = start_addr / 64 * 64;
                            while start_addr < end_addr {
                                addr_vec.push(start_addr);
                                start_addr += 64;
                            }
                            self.mem_interface.send(req.clone(), addr_vec, false);
                            self.input_buffer.send_req();
                        }
                    }
                    _ => {
                        // println!("input buffer is unknown");
                    }
                }
                match &self.input_buffer.current_state {
                    input_buffer::BufferStatus::WaitingToLoad(_, _) => {
                        input_buffer::BufferStatus::Ready()
                }
            }
            SystemState::NoInputIter => {
                // which means the OutputWindowIterator is finished, and cannot generate any more InputWindowIterator,
                // So we need to get the next OutputWindowIterator(chagne the next layer and create a new OutputWindowIterator)
                self.current_layer += 1;
                if self.current_layer == self.gcn_layer_num {
                    self.state = SystemState::Finished;
                    return;
                }
                self.current_output_iter = Some(OutputWindowIterator::new(
                    self.graph,
                    self.node_features,
                    self.agg_buffer_size,
                    self.input_buffer_size,
                    self.current_layer,
                ));
                // rebuild the input iter and window
                self.current_input_iter = self.current_output_iter.as_mut().unwrap().next();
                self.current_window = match self.current_input_iter.as_mut() {
                    Some(iter) => iter.next(),
                    None => None,
                };
                self.state = match self.current_window {
                    Some(_) => SystemState::Working,
                    None => match self.current_input_iter {
                        Some(_) => SystemState::NoWindow,
                        None => SystemState::NoInputIter,
                    },
                };
            }
            SystemState::Finished => {
                self.finished = true;
            }
            SystemState::NoWindow => {
                // which mean the current InputWindowIterator is finished, and cannot generate any more Window,
                // So we need to get the next InputWindowIterator(Get from the OutputWindowIterator)
                self.current_input_iter = self.current_output_iter.as_mut().unwrap().next();
                self.current_window = match self.current_input_iter.as_mut() {
                    Some(iter) => iter.next(),
                    None => None,
                };
                self.state = match self.current_window {
                    Some(_) => SystemState::Working,
                    None => match self.current_input_iter {
                        Some(_) => SystemState::NoWindow,
                        None => SystemState::NoInputIter,
                    },
                };
            }
        }
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
        // let mut window_iter = WindowIterator::new(&graph, &node_features, 1, 1, 1);
        // for i in window_iter {
        //     println!("{:?}", i);
        // }
    }
}
