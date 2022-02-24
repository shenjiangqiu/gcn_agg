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

use log::{debug, warn};
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
    ChangedLayer,
}

use crate::{
    accelerator::sliding_window::WindowIterSettings,
    gcn_result::GcnStatistics,
    settings::{
        AcceleratorSettings, AggregatorSettings, MlpSettings, RunningMode, SparsifierSettings,
    },
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
    sparsify_buffer: SparsifyBuffer,
    agg_buffer: AggBuffer,
    mem_interface: MemInterface,
    sparsifier: Sparsifier,
    running_mode: RunningMode,
    mlp: Mlp,

    graph: &'a Graph,
    node_features: &'a [NodeFeatures],

    input_buffer_size: usize,
    agg_buffer_size: usize,
    // output_buffer_size: usize,
    current_layer: usize,
    current_output_iter: OutputWindowIterator<'a>,
    current_input_iter: InputWindowIterator<'a>,
    current_window: Option<InputWindow<'a>>,
    gcn_layer_num: usize,
    gcn_hidden_size: Vec<usize>,

    possible_deadloack_count: usize,
    deadlock_count: usize,
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
                //debug!("running,working:{}", self.total_cycle);
                // all components are: input_buffer, output_buffer, agg_buffer, mlp, sparsifier, aggregator, mem_interface, mlp

                self.aggregator.cycle()?;
                self.mem_interface.cycle()?;
                self.agg_buffer.cycle()?;
                self.input_buffer.cycle()?;
                self.output_buffer.cycle()?;
                self.sparsifier.cycle()?;
                self.sparsify_buffer.cycle()?;
                self.mlp.cycle()?;

                // if the result is true, then return

                if self.handle_input_buffer_add_task()? {
                    return Ok(());
                }

                if self.handle_input_buffer_to_mem()? {
                    return Ok(());
                }
                if self.handle_mem_to_input_buffer()? {
                    return Ok(());
                }

                if self.handle_start_aggregator()? {
                    return Ok(());
                }
                if self.handle_finish_aggregator()? {
                    return Ok(());
                }

                if self.handle_start_mlp()? {
                    return Ok(());
                }
                if self.handle_finish_mlp()? {
                    return Ok(());
                }

                if self.handle_start_sparsify()? {
                    return Ok(());
                }
                if self.handle_finish_sparsify()? {
                    return Ok(());
                }
                if self.handle_start_writeback()? {
                    return Ok(());
                }

                // did nothings
                self.possible_deadloack_count += 1;
                if self.possible_deadloack_count == 200000 {
                    warn!("possible deadlock, current cycle:{}", self.total_cycle);
                    self.possible_deadloack_count = 0;
                    warn!("input_buffer:{:?}", self.input_buffer);
                    warn!("output_buffer:{:?}", self.output_buffer);
                    warn!("agg_buffer:{:?}", self.agg_buffer);
                    warn!("mem_interface:{:?}", self.mem_interface);
                    warn!("sparsifier:{:?}", self.sparsifier);
                    warn!("aggregator:{:?}", self.aggregator);
                    warn!("mlp:{:?}", self.mlp);
                    warn!("sparsify_buffer:{:?}\n\n\n\n\n", self.sparsify_buffer);
                    self.deadlock_count += 1;
                    if self.deadlock_count == 10 {
                        panic!("deadlock");
                    }
                }
            }
            SystemState::NoMoreWindow => {
                // debug!("no more window");
                self.aggregator.cycle()?;
                self.mem_interface.cycle()?;
                self.agg_buffer.cycle()?;
                self.input_buffer.cycle()?;
                self.sparsify_buffer.cycle()?;
                self.output_buffer.cycle()?;

                self.sparsifier.cycle()?;
                self.mlp.cycle()?;
                // no more window, so no need to add task!
                if self.handle_input_buffer_to_mem()? {
                    return Ok(());
                }
                if self.handle_mem_to_input_buffer()? {
                    return Ok(());
                }

                if self.handle_start_aggregator()? {
                    return Ok(());
                }
                if self.handle_finish_aggregator()? {
                    return Ok(());
                }

                if self.handle_start_mlp()? {
                    return Ok(());
                }
                if self.handle_finish_mlp()? {
                    return Ok(());
                }

                if self.handle_start_sparsify()? {
                    return Ok(());
                }
                if self.handle_finish_sparsify()? {
                    return Ok(());
                }
                if self.handle_start_writeback()? {
                    return Ok(());
                }

                // did nothings
                self.possible_deadloack_count += 1;
                if self.possible_deadloack_count == 200000 {
                    warn!("possible deadlock, current cycle:{}", self.total_cycle);
                    self.possible_deadloack_count = 0;
                    warn!("input_buffer:{:?}", self.input_buffer);
                    warn!("output_buffer:{:?}", self.output_buffer);
                    warn!("agg_buffer:{:?}", self.agg_buffer);
                    warn!("mem_interface:{:?}", self.mem_interface);
                    warn!("sparsifier:{:?}", self.sparsifier);
                    warn!("aggregator:{:?}", self.aggregator);
                    warn!("mlp:{:?}", self.mlp);
                    warn!("sparsify_buffer:{:?}\n\n\n\n\n", self.sparsify_buffer);
                    self.deadlock_count += 1;
                    if self.deadlock_count == 10 {
                        panic!("deadlock");
                    }
                }
            }
            &SystemState::ChangedLayer => {
                // cannot add new task until the current layer is finished(triggle by handle_start_writeback)
                self.aggregator.cycle()?;
                self.mem_interface.cycle()?;
                self.agg_buffer.cycle()?;
                self.input_buffer.cycle()?;
                self.sparsify_buffer.cycle()?;
                self.output_buffer.cycle()?;

                self.sparsifier.cycle()?;
                self.mlp.cycle()?;
                // no more window, so no need to add task!
                if self.handle_input_buffer_to_mem()? {
                    return Ok(());
                }
                if self.handle_mem_to_input_buffer()? {
                    return Ok(());
                }

                if self.handle_start_aggregator()? {
                    return Ok(());
                }
                if self.handle_finish_aggregator()? {
                    return Ok(());
                }

                if self.handle_start_mlp()? {
                    return Ok(());
                }
                if self.handle_finish_mlp()? {
                    return Ok(());
                }

                if self.handle_start_sparsify()? {
                    return Ok(());
                }
                if self.handle_finish_sparsify()? {
                    return Ok(());
                }
                if self.handle_start_writeback()? {
                    return Ok(());
                }

                // did nothings
                self.possible_deadloack_count += 1;
                if self.possible_deadloack_count == 200000 {
                    warn!("possible deadlock, current cycle:{}", self.total_cycle);
                    self.possible_deadloack_count = 0;
                    warn!("input_buffer:{:?}", self.input_buffer);
                    warn!("output_buffer:{:?}", self.output_buffer);
                    warn!("agg_buffer:{:?}", self.agg_buffer);
                    warn!("mem_interface:{:?}", self.mem_interface);
                    warn!("sparsifier:{:?}", self.sparsifier);
                    warn!("aggregator:{:?}", self.aggregator);
                    warn!("mlp:{:?}", self.mlp);
                    warn!("sparsify_buffer:{:?}\n\n\n\n\n", self.sparsify_buffer);
                    self.deadlock_count += 1;
                    if self.deadlock_count == 10 {
                        panic!("deadlock");
                    }
                }
            }
            SystemState::Finished => {
                debug!("finished");
                self.finished = true;
            }
        }
        Ok(())
    }
}

impl<'a> System<'a> {
    pub fn new(
        graph: &'a Graph,
        node_features: &'a [NodeFeatures],
        acc_settings: AcceleratorSettings,
        stats_name: &str,
    ) -> System<'a> {
        let AcceleratorSettings {
            input_buffer_size,
            agg_buffer_size,
            gcn_hidden_size,
            aggregator_settings,
            mlp_settings,
            sparsifier_settings,
            // output_buffer_size,
            running_mode,
            mem_config_name,
        } = acc_settings;

        let AggregatorSettings {
            sparse_cores,
            sparse_width,
            dense_cores,
            dense_width,
        } = aggregator_settings;

        let MlpSettings {
            systolic_rows,
            systolic_cols,
            mlp_sparse_cores,
        } = mlp_settings;

        let SparsifierSettings { sparsifier_cores } = sparsifier_settings;
        let aggregator = Aggregator::new(sparse_cores, sparse_width, dense_cores, dense_width);

        let input_buffer = InputBuffer::new();
        let output_buffer = OutputBuffer::new();
        let sparsify_buffer = SparsifyBuffer::new();
        let agg_buffer = AggBuffer::new(graph.get_num_node(), running_mode.clone());

        let mem_interface = MemInterface::new(64, 64, &mem_config_name, stats_name);
        let mlp = Mlp::new(systolic_rows, systolic_cols, mlp_sparse_cores);
        let gcn_layer_num = node_features.len();
        let window_iter_settings = WindowIterSettings {
            agg_buffer_size,
            input_buffer_size,
            gcn_hidden_size: gcn_hidden_size.clone(),
            final_layer: gcn_layer_num == 1,
            running_mode: running_mode.clone(),
            layer: 0,
        };
        let mut current_output_iter = OutputWindowIterator::new(
            graph,
            node_features.get(0).expect("node_features is empty"),
            window_iter_settings,
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
        debug!("finished build the system");
        System {
            state,
            finished: false,
            total_cycle: 0,
            aggregator,
            input_buffer,
            output_buffer,
            sparsify_buffer,
            agg_buffer,
            running_mode,
            mem_interface,
            graph,
            node_features,
            input_buffer_size,
            agg_buffer_size,
            // output_buffer_size,
            current_layer: 0,
            current_output_iter,
            current_input_iter,
            current_window,
            gcn_layer_num,
            gcn_hidden_size,
            mlp,
            sparsifier: Sparsifier::new(sparsifier_cores),
            possible_deadloack_count: 0,
            deadlock_count: 0,
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
        debug!("start to move the window");
        let mut next_window = None;
        while next_window.is_none() {
            if let Some(window) = self.current_input_iter.next() {
                debug!("get the next input window:{:?}", window);
                next_window = Some(window);
            } else if let Some(input_iter) = self.current_output_iter.next() {
                // here we moved to the next col
                self.current_input_iter = input_iter;
                next_window = self.current_input_iter.next();
                debug!("cannot get the next window,move to the next input iter and get the next window:{:?}", next_window);
            } else {
                debug!("get next output iter");
                // need to move to the next layer and reset the output iter
                self.current_layer += 1;
                if self.current_layer >= self.gcn_layer_num {
                    debug!("No more window!");
                    // self.finished = true;
                    self.state = SystemState::NoMoreWindow;
                    return;
                }
                let window_iter_settings = WindowIterSettings {
                    agg_buffer_size: self.agg_buffer_size,
                    input_buffer_size: self.input_buffer_size,
                    gcn_hidden_size: self.gcn_hidden_size.clone(),
                    final_layer: self.current_layer == self.gcn_layer_num - 1,
                    running_mode: self.running_mode.clone(),
                    layer: self.current_layer,
                };

                self.current_output_iter = OutputWindowIterator::new(
                    self.graph,
                    self.node_features
                        .get(self.current_layer)
                        .unwrap_or_else(|| {
                            panic!("node_features is empty, layer: {}", self.current_layer)
                        }),
                    window_iter_settings,
                );
                self.current_input_iter = self
                    .current_output_iter
                    .next()
                    .expect("cannot build the first input iter");
                next_window = self.current_input_iter.next();
                debug!("cannot get the next window,move to the next output iter and input iter and get the next window:{:?}", next_window);
                self.state = SystemState::ChangedLayer;
            }
        }
        self.current_window = next_window;
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

    fn handle_input_buffer_to_mem(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        // add task to current input_buffer or send request to memory
        if let input_buffer::BufferStatus::WaitingToLoad = self.input_buffer.get_current_state() {
            if self.mem_interface.available() {
                // generate addr from the req and window

                let mut addr_vec = vec![];
                let window = self
                    .input_buffer
                    .get_current_window()
                    .expect("no window in input buffer");
                let window_layer = window.get_task_id().layer_id;
                match self.running_mode {
                    RunningMode::Sparse => {
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
                        return Ok(true);
                    }
                    RunningMode::Dense => {
                        // dense
                        let base_addr: u64 = (window_layer * 0x10000000) as u64;
                        let mut start_addr = base_addr
                            + window.start_input_index as u64
                                * window.get_output_window().get_input_dim() as u64
                                * 4;
                        let end_addr = base_addr
                            + window.end_input_index as u64
                                * window.get_output_window().get_input_dim() as u64
                                * 4;
                        while start_addr < end_addr {
                            addr_vec.push(start_addr);
                            start_addr += 64;
                        }
                        self.mem_interface
                            .send(window.get_task_id().clone(), addr_vec, false);
                        self.input_buffer.send_req(true);
                        return Ok(true);
                    }
                    RunningMode::Mixed => {
                        todo!()
                    }
                }
            }
        }
        // add task to next input_buffer or send request to memory
        if let input_buffer::BufferStatus::WaitingToLoad = self.input_buffer.get_next_state() {
            if self.mem_interface.available() {
                // generate addr from the req and window

                let mut addr_vec = vec![];
                let window = self
                    .input_buffer
                    .get_next_window()
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
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn handle_input_buffer_add_task(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        // add task to current input_buffer or send request to memory
        if let input_buffer::BufferStatus::Empty = self.input_buffer.get_current_state() {
            // add a task to the input buffer
            // self.input_buffer.send_req(self.current_input_iter.as_ref().unwrap());
            let window = self.current_window.take().unwrap();
            debug!("add task to inputbuffer's current window:{:?}", &window);

            self.input_buffer.add_task_to_current(window);
            self.move_to_next_window();
            return Ok(true);
        }

        if let input_buffer::BufferStatus::Empty = self.input_buffer.get_next_state() {
            // add a task to the input buffer
            // self.input_buffer.send_req(self.current_input_iter.as_ref().unwrap());
            let window = self.current_window.take().unwrap();
            debug!("add task to inputbuffer's next window:{:?}", &window);
            self.input_buffer.add_task_to_next(window);
            self.move_to_next_window();
            return Ok(true);
        }

        Ok(false)
    }

    fn handle_mem_to_input_buffer(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        // test if there are memory request return
        if let Some(ret_req) = self.mem_interface.receive_pop() {
            self.input_buffer.receive(&ret_req);
            return Ok(true);
        }
        Ok(false)
    }

    fn handle_start_aggregator(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        // test if the aggregator is ready to start
        if let (
            input_buffer::BufferStatus::Ready,
            aggregator::AggregatorState::Idle,
            agg_buffer::BufferStatus::Empty | agg_buffer::BufferStatus::Writing,
        ) = (
            self.input_buffer.get_current_state(),
            self.aggregator.get_state(),
            self.agg_buffer.get_current_state(),
        ) {
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
                self.agg_buffer.get_current_temp_result_mut(),
            );
            self.input_buffer.start_aggragating();
            return Ok(true);
        }
        Ok(false)
    }

    fn handle_finish_aggregator(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        // test if the aggregator is finished
        if self.aggregator.get_state() == &aggregator::AggregatorState::Finished {
            // 1. make the aggregator idle
            self.aggregator.finished_aggregation();
            // 2. set the input buffer to empty
            self.input_buffer.finished_aggregation();
            // 3. set the aggregator buffer to finished or writing
            let window = self.input_buffer.get_current_window().unwrap();
            debug!("finished aggregation, window: {:?}", &window);

            match window.is_last_row {
                true => self.agg_buffer.finished_aggregation(),
                false => {}
            };

            return Ok(true);
        }
        Ok(false)
    }

    fn handle_start_mlp(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        // test if start the mlp
        //
        if let (
            &agg_buffer::BufferStatus::WaitingToMlp,
            &mlp::MlpState::Idle,
            &sparsify_buffer::BufferStatus::Empty,
        ) = (
            &self.agg_buffer.get_next_state(),
            self.mlp.get_state(),
            &self.sparsify_buffer.current_state,
        ) {
            // start the mlp
            let current_window = self.agg_buffer.get_next_window();
            debug!("start the mlp, window: {:?}", &current_window);
            self.mlp
                .start_mlp(current_window, self.agg_buffer.get_next_temp_result());
            self.sparsify_buffer.start_mlp(current_window.clone());
            self.agg_buffer.start_mlp();

            return Ok(true);
        }
        Ok(false)
    }

    fn handle_finish_mlp(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        // test if the mlp is finished
        if self.mlp.get_state() == &mlp::MlpState::Finished {
            // 1. make the mlp idle
            self.mlp.finished_mlp();
            // 2. set the output buffer to empty
            self.sparsify_buffer.finished_mlp();
            let window = self.agg_buffer.get_next_window();
            debug!("finished mlp, window: {:?}", &window);
            self.agg_buffer.finished_mlp();
            return Ok(true);
        }
        Ok(false)
    }

    fn handle_start_sparsify(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        // test if start the sparsifier
        //
        if let (
            sparsify_buffer::BufferStatus::WaitingToSparsify,
            sparsifier::SparsifierState::Idle,
            output_buffer::BufferStatus::Empty,
        ) = (
            &self.sparsify_buffer.next_state,
            &self.sparsifier.state,
            &self.output_buffer.current_state,
        ) {
            // start the sparsifier
            // if it's the last layer, do some special thing
            let current_window = self.sparsify_buffer.next_window.as_ref().unwrap();
            debug!("start the sparsifier: {:?}", &current_window);

            let window_layer = current_window.get_task_id().layer_id;
            if window_layer == self.gcn_layer_num - 1 {
                // no need to sparsify
                debug!("no need to sparsify, layer:{}", window_layer);
                self.sparsifier.add_task_last_layer();
                self.output_buffer.start_sparsify(current_window.clone());
                self.sparsify_buffer.start_sparsify();
            } else {
                let input_dim = current_window.get_input_dim();
                let output_dim = current_window.get_output_dim();
                let output_layer_id = current_window.get_task_id().layer_id + 1;
                let output_feature = self.node_features.get(output_layer_id).unwrap();

                self.sparsifier
                    .add_task(input_dim, output_dim, output_feature);
                self.output_buffer.start_sparsify(current_window.clone());

                self.sparsify_buffer.start_sparsify();
            }
            return Ok(true);
        }
        Ok(false)
    }

    fn handle_finish_sparsify(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        if let (
            sparsifier::SparsifierState::Idle,
            sparsify_buffer::BufferStatus::Sparsifying,
            output_buffer::BufferStatus::Writing,
        ) = (
            &self.sparsifier.state,
            &self.sparsify_buffer.next_state,
            &self.output_buffer.current_state,
        ) {
            let window = self.sparsify_buffer.next_window.as_ref().unwrap();
            debug!("finished sparsify, window: {:?}", &window);
            // 1. make the sparsifier idle
            self.sparsifier.finished_sparsify();
            // 2. set the output buffer to empty
            self.output_buffer.finished_sparsify();
            self.sparsify_buffer.finished_sparsify();
            return Ok(true);
        }

        Ok(false)
    }

    fn handle_start_writeback(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        // test if start the writeback
        if let (output_buffer::BufferStatus::WaitingToWriteBack, true) = (
            &self.output_buffer.next_state,
            &self.mem_interface.available(),
        ) {
            // start the writeback
            // the write back traffic is compressed
            debug!("start writeback");
            let current_window = self.output_buffer.next_window.as_ref().unwrap().clone();
            if current_window.final_layer {
                // do nothing,
                // the final layer is not written back
                if current_window.final_window {
                    // do nothing, this is the class output, just return and set simulator to finished
                    debug!(
                        "finish the simulation, the last window is : {:?}",
                        current_window
                    );
                    self.state = SystemState::Finished;
                    self.finished = true;
                }
                self.output_buffer.start_write_back();
                return Ok(true);
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

            if current_window.final_window {
                // do nothing, this is the class output, just return and set simulator to finished
                debug!("finish current layer: {:?}", current_window);
                assert_eq!(self.state, SystemState::ChangedLayer);
                self.state = SystemState::Working;
            }
            self.output_buffer.start_write_back();

            return Ok(true);
        }

        Ok(false)
    }
}

#[cfg(test)]
mod test {

    use chrono::Local;

    use super::*;
    use std::{fs::File, io::Write};
    #[test]
    fn test_system() -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all("output")?;

        simple_logger::init_with_level(log::Level::Info).unwrap_or_default();

        let graph_name = "test_data/graph_system.txt";
        let graph_data = "f 6\n1 2\n2 3 4\n0 1 4\n0 2 4\n2 4\nend\n";
        let mut file = File::create(graph_name).unwrap();
        file.write_all(graph_data.as_bytes()).unwrap();
        let feature1 = "1 1 0 0 1 1\n1 0 0 1 1 1\n1 1 1 0 0 1\n1 1 1 0 0 1\n1 1 1 0 0 1\n";
        let featrue1_name = "test_data/feature1_system.txt";
        let mut file = File::create(featrue1_name).unwrap();
        file.write_all(feature1.as_bytes()).unwrap();
        let feature2 = "1 1\n1 1 \n1 1\n1 1\n1 1\n";
        let featrue2_name = "test_data/feature2_system.txt";
        let mut file = File::create(featrue2_name).unwrap();
        file.write_all(feature2.as_bytes()).unwrap();

        debug!("graph:\n{}", graph_data);
        debug!("feature1:\n{}", feature1);
        debug!("feature2:\n{}", feature2);

        let graph = Graph::new(graph_name)?;
        debug!("graph:\n{:?}", graph);
        let node_features1 = NodeFeatures::new(featrue1_name)?;
        let node_features2 = NodeFeatures::new(featrue2_name)?;
        let gcn_hidden_size = vec![2];
        let node_features = vec![node_features1, node_features2];
        let acc_settings = AcceleratorSettings {
            agg_buffer_size: 64,
            input_buffer_size: 64,
            running_mode: RunningMode::Sparse,
            gcn_hidden_size,
            mem_config_name: "HBM-config.cfg".into(),
            aggregator_settings: AggregatorSettings {
                dense_cores: 1,
                dense_width: 1,
                sparse_cores: 1,
                sparse_width: 1,
            },
            mlp_settings: MlpSettings {
                mlp_sparse_cores: 2,
                systolic_cols: 2,
                systolic_rows: 2,
            },
            sparsifier_settings: SparsifierSettings {
                sparsifier_cores: 2,
            },
        };
        let stats_name = Local::now()
            .format("output/%Y-%m-%d_%H-%M-%S%.6f-test.txt")
            .to_string();
        let mut system = System::new(&graph, &node_features, acc_settings, &stats_name);
        system.run()?;
        assert!(system.finished());
        Ok(())
    }
    #[test]
    fn window_iter_test() {
        std::fs::create_dir_all("output").unwrap();

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
