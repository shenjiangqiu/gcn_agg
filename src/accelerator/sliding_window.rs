use super::req::Req;
use crate::{graph::Graph, node_features::NodeFeatures};
use std::{cmp::min, collections::btree_set::Range};

#[derive(Debug, Clone)]
pub struct Window<'a> {
    task_id: Req,
    tasks: Vec<Range<'a, usize>>,
    output_node_ids: Vec<usize>,
    pub start_x: usize,
    pub start_y: usize,
    pub end_x: usize,
    pub end_y: usize,
}
impl<'a> Window<'a> {
    pub fn new(
        task_id: Req,
        tasks: Vec<Range<'a, usize>>,
        start_x: usize,
        start_y: usize,
        end_x: usize,
        end_y: usize,
        output_node_ids: Vec<usize>,
    ) -> Window<'a> {
        Window {
            output_node_ids,
            task_id,
            tasks,
            start_x,
            start_y,
            end_x,
            end_y,
        }
    }
    pub fn get_task_id(&self) -> &Req {
        &self.task_id
    }
    pub fn get_tasks(&self) -> &Vec<Range<'a, usize>> {
        &self.tasks
    }
    pub fn get_location_x(&self) -> (usize, usize) {
        (self.start_x, self.end_x)
    }
    pub fn get_location_y(&self) -> (usize, usize) {
        (self.start_y, self.end_y)
    }
    pub fn get_output_node_ids(&self) -> &Vec<usize> {
        &self.output_node_ids
    }
}

#[derive(Debug)]
pub struct OutputWindowIterator<'a> {
    graph: &'a Graph,
    node_features: &'a NodeFeatures,
    agg_buffer_size: usize,
    input_buffer_size: usize,
    current_start_y: usize,
    task_id: Req,
}
impl<'a> OutputWindowIterator<'a> {
    pub fn new(
        graph: &'a Graph,
        node_features: &'a NodeFeatures,
        agg_buffer_size: usize,
        input_buffer_size: usize,
        layer: usize,
    ) -> OutputWindowIterator<'a> {
        OutputWindowIterator {
            graph,
            node_features,
            agg_buffer_size,
            input_buffer_size,
            current_start_y: 0,
            task_id: Req {
                layer_id: layer,
                col_id: 0,
                row_id: 0,
            },
        }
    }
}
impl<'a> Iterator for OutputWindowIterator<'a> {
    type Item = InputWindowIterator<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_start_y >= self.graph.get_num_node() {
            return None;
        }
        let y_size = self.agg_buffer_size / (self.graph.get_feature_size() * 4);
        let end_y = min(self.current_start_y + y_size, self.graph.get_num_node());
        let intput_iter = InputWindowIterator::new(
            self.task_id,
            self.graph,
            self.node_features,
            self.input_buffer_size,
            self.current_start_y,
            end_y,
        );
        self.task_id.col_id += 1;
        self.current_start_y = end_y;
        Some(intput_iter)
    }
}

#[derive(Debug)]
pub struct InputWindowIterator<'a> {
    task_id: Req,
    graph: &'a Graph,
    node_features: &'a NodeFeatures,
    input_buffer_size: usize,
    start_y: usize,
    end_y: usize,
    // current window information
    current_window_start_x: usize,
    current_window_end_x: usize,
}
// impl new for InputWindowIterator
impl<'a> InputWindowIterator<'a> {
    fn new(
        task_id: Req,
        graph: &'a Graph,
        node_features: &'a NodeFeatures,
        input_buffer_size: usize,
        start_y: usize,
        end_y: usize,
    ) -> Self {
        InputWindowIterator {
            task_id,
            graph,
            node_features,
            input_buffer_size,
            start_y,
            end_y,
            current_window_end_x: 0,
            current_window_start_x: 0,
        }
    }
}

impl<'a> Iterator for InputWindowIterator<'a> {
    type Item = Window<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        // test if no window left
        if self.current_window_start_x >= self.graph.get_num_node() {
            return None;
        } else {
            // first skip all emtpy rows
            while self.current_window_start_x < self.graph.get_num_node() {
                if self.graph.is_row_range_empty(
                    self.current_window_start_x,
                    self.start_y,
                    self.end_y,
                ) {
                    self.current_window_start_x += 1;
                } else {
                    break;
                }
            }
            if self.current_window_start_x == self.graph.get_num_node() {
                return None;
            }
            // build the window
            let mut x_size = 0;
            let mut x_len = 1;
            while x_size < self.input_buffer_size
                && self.current_window_start_x + x_len < self.graph.get_num_node()
            {
                let new_size = self
                    .node_features
                    .get_features(self.current_window_start_x + x_len - 1)
                    .len()
                    * 4;
                if x_size + new_size >= self.input_buffer_size {
                    break;
                }
                x_size += new_size;
                x_len += 1;
            }
            // shrink the window
            self.current_window_end_x = self.current_window_start_x + x_len;
            while self.graph.is_row_range_empty(
                self.current_window_end_x - 1,
                self.start_y,
                self.end_y,
            ) {
                self.current_window_end_x -= 1;
            }

            // build the current window
            let csc = self.graph.get_csc();
            let mut tasks = Vec::new();
            let mut output_node_ids = Vec::new();
            for i in self.start_y..self.end_y {
                let task = csc
                    .get(i)
                    .unwrap()
                    .range(self.current_window_start_x..self.current_window_end_x);

                tasks.push(task);
                output_node_ids.push(i);
            }
            let current_window = Window {
                task_id: Req::new(
                    self.task_id.col_id,
                    self.task_id.row_id,
                    self.task_id.layer_id,
                ),
                output_node_ids,
                tasks,
                start_x: self.current_window_start_x,
                start_y: self.start_y,
                end_x: self.current_window_end_x,
                end_y: self.end_y,
            };

            // prepare the next start x and start y
            self.current_window_start_x = self.current_window_end_x;

            self.task_id.row_id += 1;
            return Some(current_window);
        }
    }
}

#[cfg(test)]
mod test {
    use std::{fs::File, io::Write};

    use super::*;
    #[test]
    fn sliding_window_test() {
        let graph_name = "test_data/graph.txt";
        let features_name = "test_data/features.txt";
        let data = "f 3\n0 1 2\n1 2 0\n2 0 1\nend\n";
        let mut file = File::create("test_data/graph.txt").unwrap();
        file.write_all(data.as_bytes()).unwrap();
        let data = "0 0 1 0 1 0\n1 0 0 1 1 1\n1 1 0 0 0 1\n";
        let mut file = File::create("test_data/features.txt").unwrap();
        file.write_all(data.as_bytes()).unwrap();

        let mut graph = Graph::from(graph_name);
        let node_features = NodeFeatures::from(features_name);

        let output_window_iter = OutputWindowIterator::new(&graph, &node_features, 20, 20, 0);
        for i in output_window_iter {
            println!("{:?}", i);
            for j in i {
                println!("{:?}", j);
            }
        }
    }
    #[test]
    fn sliding_window_test_multi() {
        let graph_name = "test_data/graph.txt";
        let features_name = "test_data/features.txt";
        let data = "f 2\n1 2\n2 3 4\n0 1 4\n0 2 4\n2 4\nend\n";
        let mut file = File::create("test_data/graph.txt").unwrap();
        file.write_all(data.as_bytes()).unwrap();
        let data = "0 1 1 0 1 1\n1 0 0 1 1 1\n1 1 1 0 0 1\n1 1 1 0 0 1\n1 1 1 0 0 1\n";
        let mut file = File::create("test_data/features.txt").unwrap();
        file.write_all(data.as_bytes()).unwrap();

        let mut graph = Graph::from(graph_name);
        let node_features = NodeFeatures::from(features_name);

        let output_window_iter = OutputWindowIterator::new(&graph, &node_features, 16, 32, 0);

        let correct_ranges = vec![
            (vec![1usize, 2], vec![2usize]),
            (vec![], vec![3, 4]),
            (vec![0, 1], vec![0]),
            (vec![], vec![2]),
            (vec![4], vec![4]),
            (vec![2], vec![4]),
        ];
        let correct_task_id = vec![0usize, 1, 0, 1, 2, 0, 1];
        let correcnt_start_end = vec![
            (1usize, 0, 3, 2),
            (3, 0, 5, 2),
            (0, 2, 2, 4),
            (2, 2, 3, 4),
            (4, 2, 5, 4),
            (2, 4, 3, 5),
            (4, 4, 5, 5),
        ];
        // let mut result_=itertools::izip!(&correct_ranges, &correct_task_id, &correcnt_start_end);
        // for i in output_window_iter {
        //     for j in i {
        //         let (range, task_id, start_end) = result_.next().unwrap();
        //         assert_eq!((j.tasks[0].clone().collect(),j.tasks[1].collect()), *range);
        //         assert_eq!(j.task_id, *task_id);
        //         assert_eq!(j.start_end, *start_end);
        //     }
        // }
    }
}
