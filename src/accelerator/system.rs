use super::{
    agg_buffer::{self, AggBuffer},
    aggregator::{self, Aggregator},
    component::Component,
    input_buffer::{self, InputBuffer},
    mem_interface::MemInterface,
    mlp::{self, Mlp},
    output_buffer::{self, OutputBuffer},
    sliding_window::{InputWindow, InputWindowIterator, OutputWindowIterator},
    sparsifier::{self, Sparsifier},
    sparsify_buffer::{self, SparsifyBuffer},
};

use log::debug;
/// # Description
/// the state for the system
/// * `Idle` means this is the very first of each layer, need to init the new output and input iter
/// * `Working` means every thing is ok, can correctly use next input
/// * `NoInputIter` means no next input iter,need to get next input layer and get next output iter
/// * `NoWindow` means no next window for this input iter, need to get next input iter from the output iter
/// * `Finished` means all layer is finished
#[derive(Debug, PartialEq)]
enum SystemState {
    Working,
    NoMoreWindow,
    Finished,
}

use crate::gcn_result::GcnStatistics;
use crate::{graph::Graph, node_features::NodeFeatures};

#[derive(Debug)]
pub struct System<'a> {
    state: SystemState,
    finished: bool,
    total_cycle: u64,
    aggregator: Aggregator,
    input_buffer: InputBuffer<'a>,
    output_buffer: OutputBuffer,
    sparsify_buffer: SparsifyBuffer,
    agg_buffer: AggBuffer,
    mem_interface: MemInterface,
    sparsifier: Sparsifier,
    mlp: Mlp,

    graph: &'a Graph,
    node_features: &'a Vec<NodeFeatures>,

    input_buffer_size: usize,
    agg_buffer_size: usize,
    output_buffer_size: usize,
    current_layer: usize,
    current_output_iter: OutputWindowIterator<'a>,
    current_input_iter: InputWindowIterator<'a>,
    current_window: Option<InputWindow<'a>>,
    gcn_layer_num: usize,
    gcn_hidden_size: &'a Vec<usize>,
}

impl Component for System<'_> {
    /// # Description
    /// * this function will schedule the requests of memory and aggregator
    /// * will update each component's status
    /// * will ***NOT*** update the cycle
    ///
    fn cycle(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        match &self.state {
            SystemState::Working => {
                debug!("running,working:{}", self.total_cycle);
                // all components are: input_buffer, output_buffer, agg_buffer, mlp, sparsifier, aggregator, mem_interface, mlp

                self.aggregator.cycle()?;
                self.mem_interface.cycle()?;
                self.agg_buffer.cycle()?;
                self.input_buffer.cycle()?;
                self.output_buffer.cycle()?;
                self.sparsifier.cycle()?;
                self.sparsify_buffer.cycle()?;
                self.mlp.cycle()?;

                self.handle_input_buffer_add_task()?;
                self.handle_input_buffer_to_mem()?;
                self.handle_mem_to_input_buffer()?;

                self.handle_start_aggregator()?;
                self.handle_finish_aggregator()?;

                self.handle_start_mlp()?;
                self.handle_finish_mlp()?;

                self.handle_start_sparsify()?;
                self.handle_finish_sparsify()?;
                self.handle_start_writeback()?;
            }
            SystemState::NoMoreWindow => {
                debug!("no more window");
                self.aggregator.cycle()?;
                self.mem_interface.cycle()?;
                self.agg_buffer.cycle()?;
                self.input_buffer.cycle()?;
                self.sparsify_buffer.cycle()?;
                self.output_buffer.cycle()?;

                self.sparsifier.cycle()?;
                self.mlp.cycle()?;
            }

            SystemState::Finished => {
                self.finished = true;
            }
        }
        Ok(())
    }
}

impl<'a> System<'a> {
    pub fn new(
        graph: &'a Graph,
        node_features: &'a Vec<NodeFeatures>,
        sparse_cores: usize,
        sparse_width: usize,
        dense_cores: usize,
        dense_width: usize,
        input_buffer_size: usize,
        agg_buffer_size: usize,
        output_buffer_size: usize,
        gcn_layer_num: usize,
        gcn_hidden_size: &'a Vec<usize>,
        systolic_rows: usize,
        systolic_cols: usize,
        mlp_sparse_cores: usize,
        sparsifier_cores: usize,
    ) -> System<'a> {
        let aggregator = Aggregator::new(sparse_cores, sparse_width, dense_cores, dense_width);

        let input_buffer = InputBuffer::new();
        let output_buffer = OutputBuffer::new();
        let sparsify_buffer = SparsifyBuffer::new();
        let agg_buffer = AggBuffer::new(graph.get_num_node());
        let mem_interface = MemInterface::new(64, 64);
        let mlp = Mlp::new(systolic_rows, systolic_cols, mlp_sparse_cores);

        let mut current_output_iter = OutputWindowIterator::new(
            graph,
            node_features.get(0).expect("node_features is empty"),
            agg_buffer_size,
            input_buffer_size,
            0,
            gcn_hidden_size,
            gcn_layer_num == 1,
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
            sparsify_buffer,
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
            mlp,
            sparsifier: Sparsifier::new(sparsifier_cores),
        }
    }
    /// # Description
    /// - this function just move to the next window, or change the layer. ***don't modify any states here***!!!
    ///
    /// ---
    /// sjq
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
                // here we moved to the next col
                self.current_input_iter = input_iter;
                next_window = self.current_input_iter.next();
            } else {
                // need to move to the next layer and reset the output iter
                self.current_layer += 1;
                if self.current_layer >= self.gcn_layer_num {
                    self.finished = true;
                    self.state = SystemState::NoMoreWindow;
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
                    self.gcn_hidden_size,
                    self.current_layer == self.gcn_layer_num - 1,
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
    /// keep running until all finished
    /// * for each cycle, it will call the cycle function
    /// * and increase the total_cycle
    pub fn run(&mut self) -> Result<GcnStatistics, Box<dyn std::error::Error>> {
        debug!("start running");
        while !self.finished {
            self.cycle()?;
            self.total_cycle += 1;
        }
        self.print_stats();
        let mut gcn_statistics = GcnStatistics::new();
        gcn_statistics.cycle = self.total_cycle;
        Ok(gcn_statistics)
    }

    pub fn finished(&self) -> bool {
        self.finished
    }
    fn print_stats(&self) {
        println!("Total cycles: {}", self.total_cycle);
    }

    fn handle_input_buffer_to_mem(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // add task to current input_buffer or send request to memory
        match self.input_buffer.get_current_state() {
            // need to send request to memory
            input_buffer::BufferStatus::WaitingToLoad => {
                if self.mem_interface.available() {
                    // generate addr from the req and window

                    let mut addr_vec = vec![];
                    let window = self
                        .input_buffer
                        .get_current_window()
                        .expect("no window in input buffer");
                    let window_layer = window.get_task_id().layer_id;
                    let start_addrs = &self
                        .node_features
                        .get(window_layer)
                        .expect("no such layer in nodefeatures")
                        .start_addrs;
                    let mut start_addr = start_addrs[window.start_input_index];
                    let end_addr = start_addrs[window.end_input_index];
                    // round start_addr to the nearest 64
                    start_addr = start_addr / 64 * 64;
                    while start_addr < end_addr {
                        addr_vec.push(start_addr);
                        start_addr += 64;
                    }
                    self.mem_interface
                        .send(window.get_task_id().clone(), addr_vec, false);
                    self.input_buffer.send_req(true);
                    return Ok(());
                }
            }
            _ => {}
        }
        // add task to next input_buffer or send request to memory
        match self.input_buffer.get_next_state() {
            input_buffer::BufferStatus::WaitingToLoad => {
                if self.mem_interface.available() {
                    // generate addr from the req and window

                    let mut addr_vec = vec![];
                    let window = self
                        .input_buffer
                        .get_current_window()
                        .expect("no window in input buffer");
                    let window_layer = window.get_task_id().layer_id;
                    let start_addrs = &self
                        .node_features
                        .get(window_layer)
                        .expect("no such layer in nodefeatures")
                        .start_addrs;
                    let mut start_addr = start_addrs[window.start_input_index];
                    let end_addr = start_addrs[window.end_input_index];
                    // round start_addr to the nearest 64
                    start_addr = start_addr / 64 * 64;
                    while start_addr < end_addr {
                        addr_vec.push(start_addr);
                        start_addr += 64;
                    }
                    self.mem_interface
                        .send(window.get_task_id().clone(), addr_vec, false);
                    self.input_buffer.send_req(false);
                    return Ok(());
                }
            }
            _ => {
                // println!("input buffer is unknown");
            }
        }

        Ok(())
    }

    fn handle_input_buffer_add_task(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // add task to current input_buffer or send request to memory
        match self.input_buffer.get_current_state() {
            input_buffer::BufferStatus::Empty => {
                // add a task to the input buffer
                // self.input_buffer.send_req(self.current_input_iter.as_ref().unwrap());
                let window = self.current_window.take().unwrap();
                debug!("add task to inputbuffer's current window:{:?}", &window);
                
                self.input_buffer.add_task_to_current(window);
                self.move_to_next_window();
                return Ok(());
            }
            _ => {}
        }

        match self.input_buffer.get_next_state() {
            input_buffer::BufferStatus::Empty => {
                // add a task to the input buffer
                // self.input_buffer.send_req(self.current_input_iter.as_ref().unwrap());
                let window = self.current_window.take().unwrap();
                debug!("add task to inputbuffer's next window:{:?}", &window);
                self.input_buffer.add_task_to_next(window);
                self.move_to_next_window();
                return Ok(());
            }
            _ => {}
        }

        Ok(())
    }

    fn handle_mem_to_input_buffer(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // test if there are memory request return
        if let Some(ret_req) = self.mem_interface.receive_pop() {
            self.input_buffer.receive(&ret_req);
            return Ok(());
        }
        Ok(())
    }

    fn handle_start_aggregator(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // test if the aggregator is ready to start
        match (
            self.input_buffer.get_current_state(),
            self.aggregator.get_state(),
            &self.agg_buffer.current_state,
        ) {
            (
                input_buffer::BufferStatus::Ready,
                aggregator::AggregatorState::Idle,
                agg_buffer::BufferStatus::Empty | agg_buffer::BufferStatus::Writing,
            ) => {
                // start the aggregator
                debug!(
                    "start the aggregator,agg window: {:?}",
                    self.input_buffer.get_current_window()
                );
                let current_window = self.input_buffer.get_current_window().unwrap();
                let window_layer = current_window.get_task_id().layer_id;

                // start the aggregator
                self.agg_buffer
                    .add_task(current_window.get_output_window().clone());
                self.aggregator.add_task(
                    current_window,
                    self.node_features.get(window_layer).unwrap(),
                    &mut self.agg_buffer.current_temp_result,
                );
                self.input_buffer.current_state = input_buffer::BufferStatus::Reading;
                return Ok(());
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_finish_aggregator(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // test if the aggregator is finished
        match self.aggregator.get_state() {
            aggregator::AggregatorState::Finished => {
                // 1. make the aggregator idle
                self.aggregator.state = aggregator::AggregatorState::Idle;
                // 2. set the input buffer to empty
                self.input_buffer.current_state = input_buffer::BufferStatus::Empty;
                // 3. set the aggregator buffer to finished or writing
                let window = self.input_buffer.get_current_window().unwrap();

                self.agg_buffer.current_state = match window.is_last_row {
                    true => agg_buffer::BufferStatus::WaitingToMlp,
                    false => agg_buffer::BufferStatus::Writing,
                };

                return Ok(());
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_start_mlp(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // test if start the mlp
        //
        match (
            self.agg_buffer.get_next_state(),
            &self.mlp.state,
            &self.sparsify_buffer.current_state,
        ) {
            (
                agg_buffer::BufferStatus::WaitingToMlp,
                mlp::MlpState::Idle,
                sparsify_buffer::BufferStatus::Empty,
            ) => {
                // start the mlp
                let current_window = self.agg_buffer.get_current_window();
                self.mlp
                    .start_mlp(current_window, &self.agg_buffer.next_temp_result);
                self.agg_buffer.start_mlp();
                self.sparsify_buffer.start_mlp();

                return Ok(());
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_finish_mlp(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // test if the mlp is finished
        match self.mlp.state {
            mlp::MlpState::Finished => {
                // 1. make the mlp idle
                self.mlp.state = mlp::MlpState::Idle;
                // 2. set the output buffer to empty
                self.sparsify_buffer.current_state =
                    sparsify_buffer::BufferStatus::WaitingToSparsify;
                return Ok(());
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_start_sparsify(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // test if start the sparsifier
        //
        match (
            &self.sparsify_buffer.next_state,
            &self.sparsifier.state,
            &self.output_buffer.current_state,
        ) {
            (
                sparsify_buffer::BufferStatus::WaitingToSparsify,
                sparsifier::SparsifierState::Idle,
                output_buffer::BufferStatus::Empty,
            ) => {
                // start the sparsifier
                let current_window = self.sparsify_buffer.next_window.as_ref().unwrap();
                let input_dim = current_window.get_input_dim();
                let output_dim = current_window.get_output_dim();
                let output_layer_id = current_window.get_task_id().layer_id + 1;
                let output_feature = self.node_features.get(output_layer_id).unwrap();

                self.sparsifier
                    .add_task(input_dim, output_dim, &output_feature);
                self.output_buffer.start_sparsify(current_window.clone());

                self.sparsify_buffer.start_sparsify();
                return Ok(());
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_finish_sparsify(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        match (
            &self.sparsifier.state,
            &self.sparsify_buffer.next_state,
            &self.output_buffer.current_state,
        ) {
            (
                sparsifier::SparsifierState::Idle,
                sparsify_buffer::BufferStatus::Sparsifying,
                output_buffer::BufferStatus::Writing,
            ) => {
                // 1. make the sparsifier idle
                self.sparsifier.state = sparsifier::SparsifierState::Idle;
                // 2. set the output buffer to empty
                self.output_buffer.current_state = output_buffer::BufferStatus::Empty;
                self.sparsify_buffer.next_state = sparsify_buffer::BufferStatus::Empty;
                return Ok(());
            }
            _ => {}
        }

        Ok(())
    }

    fn handle_start_writeback(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // test if start the writeback
        match (
            &self.output_buffer.next_state,
            &self.mem_interface.available(),
        ) {
            (output_buffer::BufferStatus::WaitingToWriteBack, true) => {
                // start the writeback
                // the write back traffic is compressed
                let current_window = self.output_buffer.next_window.as_ref().unwrap();
                if current_window.final_layer {
                    // do nothing,
                    // the final layer is not written back
                    if current_window.final_window {
                        // do nothing, this is the class output, just return and set simulator to finished
                        self.state = SystemState::Finished;
                        self.finished = true;
                        return Ok(());
                    }
                }

                // else, the write back traffic is decided be next layer's input.
                let layer_id = current_window.get_task_id().layer_id;
                let node_feature = self.node_features.get(layer_id + 1).unwrap();
                let mut addr_vec = vec![];

                let start_addrs = &node_feature.start_addrs;
                let mut start_addr = start_addrs[current_window.start_output_index];
                let end_addr = start_addrs[current_window.end_output_index];
                // round start_addr to the nearest 64
                start_addr = start_addr / 64 * 64;
                while start_addr < end_addr {
                    addr_vec.push(start_addr);
                    start_addr += 64;
                }
                self.mem_interface
                    .send(current_window.get_task_id().clone(), addr_vec, true);

                return Ok(());
            }
            _ => {}
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use std::{fs::File, io::Write};
    #[test]
    fn test_system() -> Result<(), Box<dyn std::error::Error>> {
        let graph_name = "test_data/graph.txt";
        let features_name = "test_data/features.txt";
        let data = "f 3\n0 1 2\n1 2 0\n2 0 1\nend\n";
        let mut file = File::create("test_data/graph.txt").unwrap();
        file.write_all(data.as_bytes()).unwrap();
        let data = "0 0 1 0 1 0\n1 0 0 1 1 1\n1 1 0 0 0 1\n";
        let mut file = File::create("test_data/features.txt").unwrap();
        file.write_all(data.as_bytes()).unwrap();
        let graph = Graph::new(graph_name)?;
        let node_features = NodeFeatures::new(features_name)?;
        let node_features = vec![node_features];
        let gcn_hidden_size = vec![2, 2];
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
            &gcn_hidden_size,
            1,
            1,
            1,
            1,
        );
        system.run()?;
        assert_eq!(system.finished(), true);
        Ok(())
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

        let _graph = Graph::new(graph_name).unwrap();
        let _node_features = NodeFeatures::new(features_name).unwrap();
        // let mut window_iter = WindowIterator::new(&graph, &node_features, 1, 1, 1);
        // for i in window_iter {
        //     println!("{:?}", i);
        // }
    }
}
