/// # Description
/// the state for the system
/// * `Idle` means this is the very first of each layer, need to init the new output and input iter
/// * `Working` means every thing is ok, can correctly use next input
/// * `NoInputIter` means no next input iter,need to get next input layer and get next output iter
/// * `NoWindow` means no next window for this input iter, need to get next input iter from the output iter
/// * `Finished` means all layer is finished
#[derive(Debug, PartialEq)]
enum SystemState {
    Empty,
    Working,
    Finished,
}

use super::{
    agg_buffer::AggBuffer,
    aggregator::{self, Aggregator},
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
    node_features: Vec<&'a NodeFeatures>,

    input_buffer_size: usize,
    agg_buffer_size: usize,
    output_buffer_size: usize,
    current_layer: usize,
    current_output_iter: OutputWindowIterator<'a>,
    current_input_iter: InputWindowIterator<'a>,
    current_window: Option<Window<'a>>,
    gcn_layer_num: usize,
    gcn_hidden_size: Vec<usize>,
}

impl<'a> System<'a> {
    pub fn new(
        graph: &'a Graph,
        node_features: Vec<&'a NodeFeatures>,
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
        let mut current_output_iter = OutputWindowIterator::new(
            graph,
            node_features.get(0).expect("node_features is empty"),
            agg_buffer_size,
            input_buffer_size,
            0,
        );
        let mut current_input_iter = current_output_iter
            .next()
            .expect("cannot build the first input iter");
        let current_window = Some(
            current_input_iter
                .next()
                .expect("cannot build the first window"),
        );

        let state = SystemState::Working;
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
            current_output_iter,
            current_input_iter,
            current_window,
            gcn_layer_num,
            gcn_hidden_size,
        }
    }

    pub fn move_to_next_window(&mut self) {
        // go through the current_input_iter and current_output_iter to get the next window
        // if the current_input_iter is finished, then get the next input iter
        // if the current_output_iter is finished, then get the next output iter
        // if both are finished, then the system is finished

        let mut next_window = None;
        while next_window.is_none() {
            if let Some(window) = self.current_input_iter.next() {
                next_window = Some(window);
            } else if let Some(input_iter) = self.current_output_iter.next() {
                self.current_input_iter = input_iter;
                next_window = self.current_input_iter.next();
            } else {
                // need to move to the next layer and reset the output iter
                self.current_layer += 1;
                if self.current_layer >= self.gcn_layer_num {
                    self.finished = true;
                    self.state = SystemState::Finished;
                    return;
                }
                self.current_output_iter = OutputWindowIterator::new(
                    self.graph,
                    self.node_features.get(self.current_layer).expect(
                        format!("node_features is empty, layer: {}", self.current_layer).as_str(),
                    ),
                    self.agg_buffer_size,
                    self.input_buffer_size,
                    self.current_layer,
                );
                self.current_input_iter = self
                    .current_output_iter
                    .next()
                    .expect("cannot build the first input iter");
                next_window = self.current_input_iter.next();
            }
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
                // add task to current input_buffer or send request to memory
                match self.input_buffer.get_current_state() {
                    input_buffer::BufferStatus::Empty => {
                        // add a task to the input buffer
                        // self.input_buffer.send_req(self.current_input_iter.as_ref().unwrap());
                        let window = self.current_window.take().unwrap();
                        let req_id = window.get_task_id();

                        self.input_buffer.add_task_to_current(window);
                        self.move_to_next_window();
                        return;
                    }
                    input_buffer::BufferStatus::WaitingToLoad => {
                        if self.mem_interface.available() {
                            // generate addr from the req and window

                            let mut addr_vec = vec![];
                            let window = self
                                .input_buffer
                                .get_current_window()
                                .expect("no window in input buffer");
                            let window_layer = window.get_task_id().layer_id;
                            let start_addrs = self
                                .node_features
                                .get(window_layer)
                                .expect("no such layer in nodefeatures")
                                .start_addrs;
                            let mut start_addr = start_addrs[window.start_x];
                            let end_addr = start_addrs[window.end_x];
                            // round start_addr to the nearest 64
                            start_addr = start_addr / 64 * 64;
                            while start_addr < end_addr {
                                addr_vec.push(start_addr);
                                start_addr += 64;
                            }
                            self.mem_interface
                                .send(window.get_task_id().clone(), addr_vec, false);
                            self.input_buffer.send_req(true);
                            return;
                        }
                    }
                    _ => {
                        // println!("input buffer is unknown");
                    }
                }
                // add task to next input_buffer or send request to memory
                match self.input_buffer.get_next_state() {
                    input_buffer::BufferStatus::Empty => {
                        // add a task to the input buffer
                        // self.input_buffer.send_req(self.current_input_iter.as_ref().unwrap());
                        let window = self.current_window.take().unwrap();
                        let req_id = window.get_task_id();
                        self.input_buffer.add_task_to_current(window);
                        self.move_to_next_window();
                        return;
                    }
                    input_buffer::BufferStatus::WaitingToLoad => {
                        if self.mem_interface.available() {
                            // generate addr from the req and window

                            let mut addr_vec = vec![];
                            let window = self
                                .input_buffer
                                .get_current_window()
                                .expect("no window in input buffer");
                            let window_layer = window.get_task_id().layer_id;
                            let start_addrs = self
                                .node_features
                                .get(window_layer)
                                .expect("no such layer in nodefeatures")
                                .start_addrs;
                            let mut start_addr = start_addrs[window.start_x];
                            let end_addr = start_addrs[window.end_x];
                            // round start_addr to the nearest 64
                            start_addr = start_addr / 64 * 64;
                            while start_addr < end_addr {
                                addr_vec.push(start_addr);
                                start_addr += 64;
                            }
                            self.mem_interface
                                .send(window.get_task_id().clone(), addr_vec, false);
                            self.input_buffer.send_req(false);
                            return;
                        }
                    }
                    _ => {
                        // println!("input buffer is unknown");
                    }
                }

                // test if there are memory request return
                if self.mem_interface.ret_ready() {
                    let ret_req = self.mem_interface.receive_pop();
                    self.input_buffer.receive(&ret_req);
                    return;
                }

                // test if the aggregator is ready to start
                match (
                    self.input_buffer.get_current_state(),
                    self.aggregator.get_state(),
                ) {
                    (input_buffer::BufferStatus::Ready, aggregator::AggregatorState::Idle) => {
                        let current_window=self.input_buffer.get_current_window().unwrap();
                        let window_layer = current_window.get_task_id().layer_id;
                        self.aggregator.add_task(
                            current_window,
                            self.node_features.get(window_layer).unwrap(),
                        );
                        self.agg_buffer.add_task(current_window);
                        
                        self.input_buffer.current_state = input_buffer::BufferStatus::Reading;
                        return;
                    }
                    _ => {}
                }

                // test if the aggregator is finished
                match self.aggregator.get_state() {
                    aggregator::AggregatorState::Finished => {
                        // 1. make the aggregator idle
                        self.aggregator.state = aggregator::AggregatorState::Idle;
                        // 2. set the input buffer to empty
                        self.input_buffer.current_state = input_buffer::BufferStatus::Empty;
                        return;
                    }
                    _ => {}
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
