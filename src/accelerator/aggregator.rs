use std::{
    collections::{btree_set::Range, HashSet},
    vec,
};

use log::error;

use crate::{node_features::NodeFeatures, settings::RunningMode};

use super::{
    component::Component, sliding_window::InputWindow, temp_agg_result::TempAggResult,
    window_id::WindowId,
};
#[derive(Debug, PartialEq)]
pub enum AggregatorState {
    Idle,
    // the task id and remaining cycles
    Working,
    Finished,
}
#[derive(Debug)]
#[allow(dead_code)]
pub struct Aggregator {
    sparse_cores: usize,
    sparse_width: usize,

    dense_cores: usize,
    dense_width: usize,

    pub state: AggregatorState,
    // last_output_id: usize,
    current_task_id: Option<WindowId>,
    current_task_remaining_cycles: u64,
}

impl Component for Aggregator {
    /// # Description
    ///  the cycle function
    /// # Example
    /// ```ignore
    /// aggregator.cycle();
    /// ```
    ///
    fn cycle(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.state == AggregatorState::Working {
            if self.current_task_remaining_cycles == 0 {
                self.state = AggregatorState::Finished;
            } else {
                self.current_task_remaining_cycles -= 1;
            }
        }
        Ok(())
    }
}

impl Aggregator {
    pub fn new(
        sparse_cores: usize,
        sparse_width: usize,
        dense_cores: usize,
        dense_width: usize,
    ) -> Aggregator {
        Aggregator {
            sparse_cores,
            sparse_width,
            dense_cores,
            dense_width,
            state: AggregatorState::Idle,
            // last_output_id: 0,
            current_task_id: None,
            current_task_remaining_cycles: 0,
        }
    }

    pub fn add_task(
        &mut self,
        task: &InputWindow,
        node_features: Option<&NodeFeatures>,
        temp_agg_result: &mut Option<TempAggResult>,
        running_mode: &RunningMode,
    ) {
        match *running_mode {
            RunningMode::Sparse => {
                let tasks = task.get_tasks().clone();
                // collect tasks to Vec<Vec<usize>>
                let output_start = task.start_output_index;
                let output_end = task.end_output_index;

                let cycles = self.get_add_sparse_cycle(
                    tasks,
                    &mut temp_agg_result.as_mut().unwrap()[output_start..output_end],
                    node_features.unwrap(),
                );

                self.state = AggregatorState::Working;
                self.current_task_id = Some(task.get_task_id().clone());
                self.current_task_remaining_cycles = cycles;
            }
            RunningMode::Dense => {
                // dense aggregation
                let num_add = task
                    .get_tasks()
                    .iter()
                    .fold(0, |acc, x| acc + x.clone().count());
                let mut cycles: u64 = 0;
                cycles += (num_add * task.get_output_window().get_input_dim()
                    / (self.dense_width * self.dense_cores)) as u64;
                // extra cycle for load data
                cycles *= 2;
                self.state = AggregatorState::Working;
                self.current_task_id = Some(task.get_task_id().clone());
                self.current_task_remaining_cycles = cycles;
            }
            RunningMode::Mixed => {
                let mut cycles: u64 = 0;

                // first need to unpack the sparse data to dense
                error!("need to decide the unpack algorithm");
                todo!("need to decide the number of the cycles needed for unpack!");
                // then perform dense aggregation
                let num_add = task
                    .get_tasks()
                    .iter()
                    .fold(0, |acc, x| acc + x.clone().count());
                cycles += (num_add * task.get_output_window().get_input_dim()
                    / (self.dense_width * self.dense_cores)) as u64;
                // extra cycle for load data
                cycles *= 2;
                self.state = AggregatorState::Working;
                self.current_task_id = Some(task.get_task_id().clone());
                self.current_task_remaining_cycles = cycles;
            }
        }
    }
    pub fn get_state(&self) -> &AggregatorState {
        &self.state
    }

    ///
    /// # Arguments
    /// * `tasks` - The list of edges to be aggregated
    /// each line is a set of edges that need to be aggregated to taget node, the taget node is returned
    /// echo line is like:
    /// 1 2 3 4
    /// 5 6 7 8
    /// 9 10 11 12
    ///
    /// which mean node 1,2,3,4 will be aggregated to the first node,
    /// node 5,6,7,8 will be aggregated to the second node,
    /// and node 9,10,11,12 will be aggregated to the third node
    /// * node_features - the node features is sparse format, each line is a node, each column is a feature index
    ///
    /// # Return
    /// (the cycles to calculate each node, the node features of result nodes)
    ///
    /// # Example
    /// ```ignore
    /// use gcn_agg::accelerator::aggregator::Aggregator;
    /// let node_features = vec![
    ///  vec![0,4,9],
    ///  vec![1,5,10],
    ///  vec![2],
    /// ];
    /// let tasks = vec![
    /// vec![0,1],
    /// vec![1,2],
    /// ];
    /// let num_sparse_cores = 2;
    /// let num_sparse_width = 2;
    /// let aggregator = Aggregator::new(2,2,2,2);
    /// ```
    ///
    ///
    pub fn get_add_sparse_cycle(
        &mut self,
        tasks: Vec<Range<usize>>,
        output_features: &mut [Vec<usize>],
        node_features: &NodeFeatures,
    ) -> u64 {
        // each task's cycles
        let mut cycle_vec = Vec::new();
        for (task, output_vec) in tasks.into_iter().zip(output_features.iter_mut()) {
            cycle_vec.push(self.get_add_cycle_and_result_sparse(output_vec, task, node_features));
        }

        // each cores current cycles, always push task to the core with the least cycles
        let mut core_cycles = vec![0; self.sparse_cores];
        cycle_vec.into_iter().for_each(|i| {
            core_cycles.sort_unstable();
            core_cycles[0] += i;
        });
        core_cycles.sort_unstable();
        let cycles = *core_cycles.last().unwrap_or(&0);

        cycles as u64
    }

    /// # Description
    /// get the cycle and result for a single output aggregation task
    ///
    /// # Arguments
    /// * `input_nodes` - the input nodes of the task, each element is a edge(node id)
    /// * `output_node_feature` - the result node of the task, there might be temporary result in it, the vector contains
    /// * `node_features` - the node features is sparse format, each line is a node, each column is a feature index
    /// # Example
    /// ```ignore
    /// let data = "0 0 1 0 1 0\n1 0 0 1 1 1\n1 1 0 0 0 1\n";
    /// let file_name = "test_data/node_features.txt";
    /// let mut file = File::create(file_name).unwrap();
    /// file.write_all(data.as_bytes()).unwrap();
    ///
    /// let node_features = NodeFeatures::from(file_name);
    ///
    /// let aggregator = Aggregator::new(2, 2, 2, 2);
    /// let input_node = vec![0, 1];
    /// let mut output_node_feature = vec![0, 3, 5];
    /// let cycles = aggregator.get_add_cycle_and_result_sparse(
    ///     &input_node,
    ///     &mut output_node_feature,
    ///     &node_features,
    /// );
    /// // will be 3+2+5+4=14
    /// assert_eq!(cycles, 14);
    /// // after first round, will be [0,2,3,4,5], after second round , will be the same.
    /// assert_eq!(output_node_feature.iter().collect::<HashSet<_>>(), vec![0, 2, 3, 4, 5].iter().collect());
    /// ```
    ///
    fn get_add_cycle_and_result_sparse(
        &mut self,
        output_feature: &mut Vec<usize>,
        input_nodes: Range<usize>,
        node_features: &NodeFeatures,
    ) -> u64 {
        let mut cycles = 0;
        // type 1, simplely add the features one by one
        let mut temp_set: HashSet<usize> = output_feature.iter().cloned().collect();

        for &i in input_nodes {
            cycles += temp_set.len() + node_features.get_features(i).len();
            for &j in node_features.get_features(i) {
                temp_set.insert(j);
            }
        }
        output_feature.clear();
        output_feature.append(&mut temp_set.into_iter().collect());
        cycles as u64
    }

    /// # Description
    #[allow(dead_code)]
    pub fn add_dense(&self, features_size: usize, num_features: usize) -> u64 {
        let mut cycles = 0;
        cycles += features_size * num_features;
        let total_cores = self.dense_cores * self.dense_width;

        cycles = (cycles + total_cores - 1) / total_cores;

        // for each round, need an extra cycle to load the data and send back the data
        cycles *= 3;
        cycles as u64
    }

    pub fn finished_aggregation(&mut self) {
        self.state = AggregatorState::Idle;
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::node_features::NodeFeatures;
//     use std::fs::File;
//     use std::io::Write;

//     #[test]
//     fn test_sparse_add_single_output() {
//         let data = "0 0 1 0 1 0\n1 0 0 1 1 1\n1 1 0 0 0 1\n";
//         let file_name = "test_data/node_features.txt";
//         let mut file = File::create(file_name).unwrap();
//         file.write_all(data.as_bytes()).unwrap();

//         let node_features = NodeFeatures::from(file_name);

//         let aggregator = Aggregator::new(2, 2, 2, 2,3);
//         let input_node = vec![0, 1];
//         let mut output_node_feature = vec![0, 3, 5];
//         let cycles = aggregator.get_add_cycle_and_result_sparse(
//             &input_node,

//             &mut output_node_feature,
//             &node_features,
//         );
//         // will be 3+2+5+4=14
//         assert_eq!(cycles, 14);
//         // after first round, will be [0,2,3,4,5], after second round , will be the same.
//         assert_eq!(
//             output_node_feature.iter().collect::<HashSet<_>>(),
//             vec![0, 2, 3, 4, 5].iter().collect()
//         );
//     }

//     #[test]
//     fn test_sparse_add() {
//         let data = "0 0 1 0 1 0\n1 0 0 1 1 1\n1 1 0 0 0 1\n";
//         let file_name = "test_data/node_features.txt";
//         let mut file = File::create(file_name).unwrap();
//         file.write_all(data.as_bytes()).unwrap();

//         let node_features = NodeFeatures::from(file_name);

//         let aggregator = Aggregator::new(2, 2, 2, 2);
//         let tasks = vec![vec![0, 1], vec![1, 2], vec![0, 1, 2]];
//         let mut output_node_features = vec![vec![0, 3, 5], vec![0, 3, 5], vec![]];
//         let cycles = aggregator.add_sparse(tasks, &mut output_node_features, &node_features);
//         // the first one will be 3+2+5+4=14, the second one will be 3+4+4+3=14, the third one will be 0+2+2+4+5+3=16
//         assert_eq!(cycles, 30);
//     }

//     #[test]
//     fn test_dense_add() {
//         let aggregator = Aggregator::new(2, 2, 2, 2);

//         let cycles = aggregator.add_dense(10, 10);
//         assert_eq!(cycles, 75);
//     }

//     #[test]
//     fn test_cycle() {
//         let mut aggregator = Aggregator::new(2, 2, 2, 2);
//         let task_cycle = aggregator.add_dense(10, 10);
//         aggregator.add_work(0, task_cycle);
//         aggregator.cycle();
//         assert_eq!(aggregator.state, AggregatorState::Working(0, 74));
//         for _i in 0..100 {
//             aggregator.cycle();
//         }
//         assert_eq!(aggregator.state, AggregatorState::Idle);
//     }
// }
