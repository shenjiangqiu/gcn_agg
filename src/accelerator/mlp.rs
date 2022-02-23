use log::info;

use super::{component::Component, sliding_window::OutputWindow, temp_agg_result::TempAggResult};

#[derive(Debug, Clone, PartialEq)]
pub enum MlpState {
    Idle,
    Working,
    Finished,
}

#[derive(Debug, PartialEq)]
pub struct Mlp {
    state: MlpState,
    remaining_cycle: u64,
    systolic_rows: usize,
    systolic_cols: usize,
    sparse_cores: usize,
}

impl Component for Mlp {
    fn cycle(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.state == MlpState::Working {
            if self.remaining_cycle == 0 {
                self.state = MlpState::Finished;
            } else {
                self.remaining_cycle -= 1;
            }
        }
        Ok(())
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
    pub(super) fn get_state(&self) -> &MlpState {
        &self.state
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
    pub fn start_mlp(
        &mut self,
        output_window: &OutputWindow,
        output_results: &Option<TempAggResult>,
    ) {
        match output_results {
            Some(_output_results) => {
                // the sparse mlp
                info!("start sparse mlp");
                self.state = MlpState::Working;
                let total_add = _output_results.iter().fold(0, |acc, x| acc + x.len());
                let mut total_cycle =
                    total_add * output_window.get_output_dim() / (self.sparse_cores);
                total_cycle *= 2;
                self.remaining_cycle = total_cycle as u64;
            }
            None => {
                info!("start dense mlp");
                // the dense mlp
                self.state = MlpState::Working;
                let mut total_cycles = 0;
                // calculate the number of cycles needed to finish the mlp
                let num_nodes = output_window.get_output_len();
                let output_node_dim = output_window.output_node_dim;
                let input_node_dim = output_window.input_node_dim;
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
        self.state = MlpState::Idle;
    }
    #[allow(dead_code)]
    pub fn get_remaining_cycle(&self) -> u64 {
        self.remaining_cycle
    }
}
