use std::{collections::HashSet, vec};

use crate::node_features::NodeFeatures;
#[derive(Debug, PartialEq)]
enum AggregatorState {
    Idle,
    // the task id and remaining cycles
    Working(usize, u64),
}
#[derive(Debug)]
pub struct Aggregator {
    sparse_cores: usize,
    sparse_width: usize,

    dense_cores: usize,
    dense_width: usize,

    state: AggregatorState,
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
        }
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
    pub fn add_sparse(
        &self,
        tasks: Vec<Vec<usize>>,
        output_node_features: &mut Vec<Vec<usize>>,
        node_features: &NodeFeatures,
    ) -> u64 {
        // each task's cycles
        let mut cycle_vec = Vec::new();
        for (task, temp) in tasks.iter().zip(output_node_features.iter_mut()) {
            cycle_vec.push(self.get_add_cycle_and_result_sparse(task, temp, node_features));
        }

        // each cores current cycles, always push task to the core with the least cycles
        let mut core_cycles = vec![0; self.sparse_cores];
        for i in cycle_vec {
            core_cycles.sort();
            core_cycles[0] += i;
        }
        core_cycles.sort();
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
        &self,
        input_nodes: &Vec<usize>,
        output_node_feature: &mut Vec<usize>,
        node_features: &NodeFeatures,
    ) -> u64 {
        let mut cycles = 0;

        // type 1, simplely add the features one by one
        let mut temp_set: HashSet<usize> = output_node_feature.iter().cloned().collect();

        for &i in input_nodes {
            cycles += temp_set.len() + node_features.get_features(i).len();
            for &j in node_features.get_features(i) {
                temp_set.insert(j);
            }
        }
        output_node_feature.clear();
        output_node_feature.append(&mut temp_set.into_iter().collect());
        cycles as u64
    }

    /// # Description
    pub fn add_dense(&self, features_size: usize, num_features: usize) -> u64 {
        let mut cycles = 0;
        cycles += features_size * num_features;
        let total_cores = self.dense_cores * self.dense_width;

        cycles = (cycles + total_cores - 1) / total_cores;

        // for each round, need an extra cycle to load the data and send back the data
        cycles *= 3;
        cycles as u64
    }

    /// # Description
    ///  the cycle function
    /// # Example
    /// ```ignore
    /// aggregator.cycle();
    /// ```
    ///
    pub fn cycle(&mut self) {
        match self.state {
            AggregatorState::Idle => {
                // do nothing
            }
            AggregatorState::Working(task_id, remaining_cycles) => {
                if remaining_cycles == 0 {
                    self.state = AggregatorState::Idle;
                } else {
                    self.state = AggregatorState::Working(task_id, remaining_cycles - 1);
                }
            }
        }
    }

    pub fn add_work(&mut self, task_id: usize, cycles: u64) {
        assert_eq!(self.state, AggregatorState::Idle);
        self.state = AggregatorState::Working(task_id, cycles);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node_features::NodeFeatures;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn test_sparse_add_single_output() {
        let data = "0 0 1 0 1 0\n1 0 0 1 1 1\n1 1 0 0 0 1\n";
        let file_name = "test_data/node_features.txt";
        let mut file = File::create(file_name).unwrap();
        file.write_all(data.as_bytes()).unwrap();

        let node_features = NodeFeatures::from(file_name);

        let aggregator = Aggregator::new(2, 2, 2, 2);
        let input_node = vec![0, 1];
        let mut output_node_feature = vec![0, 3, 5];
        let cycles = aggregator.get_add_cycle_and_result_sparse(
            &input_node,
            &mut output_node_feature,
            &node_features,
        );
        // will be 3+2+5+4=14
        assert_eq!(cycles, 14);
        // after first round, will be [0,2,3,4,5], after second round , will be the same.
        assert_eq!(
            output_node_feature.iter().collect::<HashSet<_>>(),
            vec![0, 2, 3, 4, 5].iter().collect()
        );
    }

    #[test]
    fn test_sparse_add() {
        let data = "0 0 1 0 1 0\n1 0 0 1 1 1\n1 1 0 0 0 1\n";
        let file_name = "test_data/node_features.txt";
        let mut file = File::create(file_name).unwrap();
        file.write_all(data.as_bytes()).unwrap();

        let node_features = NodeFeatures::from(file_name);

        let aggregator = Aggregator::new(2, 2, 2, 2);
        let tasks = vec![vec![0, 1], vec![1, 2], vec![0, 1, 2]];
        let mut output_node_features = vec![vec![0, 3, 5], vec![0, 3, 5], vec![]];
        let cycles = aggregator.add_sparse(tasks, &mut output_node_features, &node_features);
        // the first one will be 3+2+5+4=14, the second one will be 3+4+4+3=14, the third one will be 0+2+2+4+5+3=16
        assert_eq!(cycles, 30);
    }

    #[test]
    fn test_dense_add() {
        let aggregator = Aggregator::new(2, 2, 2, 2);

        let cycles = aggregator.add_dense(10, 10);
        assert_eq!(cycles, 75);
    }

    #[test]
    fn test_cycle() {
        let mut aggregator = Aggregator::new(2, 2, 2, 2);
        let task_cycle = aggregator.add_dense(10, 10);
        aggregator.add_work(0, task_cycle);
        aggregator.cycle();
        assert_eq!(aggregator.state, AggregatorState::Working(0, 74));
        for _i in 0..100 {
            aggregator.cycle();
        }
        assert_eq!(aggregator.state, AggregatorState::Idle);
    }
}
