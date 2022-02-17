//! the mod sparsifier is a component to translate the dense result to sparse result
//!
//!

use crate::node_features::NodeFeatures;

use super::component::Component;

#[derive(Debug, Clone, PartialEq)]
pub enum SparsifierState {
    Idle,
    Working,
}

#[derive(Debug)]
pub struct Sparsifier {
    pub state: SparsifierState,
    pub remaining_cycle: u64,
    pub num_cores: usize,
}

impl Component for Sparsifier {
    fn cycle(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        match self.state {
            SparsifierState::Working => {
                if self.remaining_cycle == 0 {
                    self.state = SparsifierState::Idle;
                } else {
                    self.remaining_cycle -= 1;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

impl Sparsifier {
    pub fn new(num_cores: usize) -> Sparsifier {
        Sparsifier {
            state: SparsifierState::Idle,
            remaining_cycle: 0,
            num_cores,
        }
    }

    /// # Description
    /// - this is so ***tricky***! be careful!
    /// - we already know the input dimension and know the result sparse vector, so we do not need to know what is the input data!
    /// # Arguments
    /// - input_dim: the input dimension
    /// - input_node_num: the number of input nodes
    /// - output_feature: the csc sparse vector of output nodes
    ///
    pub fn add_task(
        &mut self,
        _input_dim: usize,
        _input_node_num: usize,
        _output_feature: &NodeFeatures,
    ) {
        self.remaining_cycle = 10;
        self.state = SparsifierState::Working;
    }

    pub fn add_task_last_layer(&mut self) {
        self.remaining_cycle = 1;
        self.state = SparsifierState::Working;
    }

    pub fn finished_sparsify(&self) -> bool {
        self.state == SparsifierState::Idle
    }
}
