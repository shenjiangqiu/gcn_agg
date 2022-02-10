use super::{component::Component, sliding_window::Window};

#[derive(Debug, Clone, PartialEq)]
pub enum MlpState {
    Idle,
    Working,
    Finished,
}

#[derive(Debug, PartialEq)]
pub struct Mlp {
    pub state: MlpState,
    remaining_cycle: u64,
    systolic_rows: usize,
    systolic_cols: usize,
    sparse_cores: usize,
}

impl Component for Mlp {
    fn cycle(&mut self) {
        match self.state {
            MlpState::Working => {
                if self.remaining_cycle == 0 {
                    self.state = MlpState::Finished;
                } else {
                    self.remaining_cycle -= 1;
                }
            }
            _ => {}
        }
    }
}

impl Mlp {
    pub fn new(systolic_rows: usize, systolic_cols: usize, sparse_cores: usize) -> Mlp {
        Mlp {
            state: MlpState::Idle,
            remaining_cycle: 0,
            systolic_rows,
            systolic_cols,
            sparse_cores,
        }
    }

    /// # Description
    /// - start the mlp
    /// - calculate the number of cycles needed to finish the mlp
    /// # Arguments
    /// - output_results: the temporary results from the aggregator(if sparse is true)
    /// - tasks: the window
    ///
    /// # Example
    /// ```ignore
    /// use gcn_agg::accelerator::mlp::Mlp;
    /// let mut mlp = Mlp::new(systolic_rows, systolic_cols, sparse_cores);
    /// mlp.start_mlp(output_results, tasks);
    ///
    /// ```
    ///
    /// # Return
    /// ()
    ///
    pub fn start_mlp(&mut self, tasks: &Window, output_results: &Option<Vec<Vec<usize>>>) {
        match output_results {
            Some(output_results) => {
                todo!();
            }
            None => {
                self.state = MlpState::Working;
                let mut total_cycles = 0;
                // calculate the number of cycles needed to finish the mlp
                let num_nodes = tasks.get_output_node_ids().len();
                let output_node_dim = tasks.output_node_dim;
                let input_node_dim = tasks.input_node_dim;
                let steps = (self.systolic_rows + num_nodes - 1) / self.systolic_rows;
                let elements_steps =
                    (output_node_dim + self.systolic_cols - 1) / self.systolic_cols;
                for _i in 0..steps {
                    for _j in 0..elements_steps - 1 {
                        total_cycles += self.systolic_rows + self.systolic_cols + input_node_dim;
                        total_cycles += self.systolic_rows * self.systolic_rows / 4 / 32;
                    }
                }

                self.remaining_cycle = total_cycles as u64;
            }
        }
    }

    pub fn finished_mlp(&mut self) {
        self.state = MlpState::Finished;
    }

    pub fn get_remaining_cycle(&self) -> u64 {
        self.remaining_cycle
    }
}
