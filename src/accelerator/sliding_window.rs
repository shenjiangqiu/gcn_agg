use log::{debug, info};

use super::window_id::WindowId;
use crate::{graph::Graph, node_features::NodeFeatures};
use core::panic;
use std::{cmp, collections::btree_set::Range, rc::Rc};
pub struct WindowIterSettings {
    pub agg_buffer_size: usize,
    pub input_buffer_size: usize,
    pub layer: usize,
    pub gcn_hidden_size: Vec<usize>,
    pub final_layer: bool,
    pub is_sparse: bool,
}
#[derive(Debug, Clone)]
pub struct InputWindow<'a> {
    pub task_id: WindowId,
    tasks: Rc<Vec<Range<'a, usize>>>,
    pub start_output_index: usize,
    pub start_input_index: usize,
    pub end_output_index: usize,
    pub end_input_index: usize,
    pub output_window: Rc<OutputWindow>,
    pub is_last_row: bool,
}

#[derive(Debug, Clone)]
pub struct OutputWindow {
    pub start_output_index: usize,
    pub end_output_index: usize,
    pub task_id: WindowId,
    pub output_node_dim: usize,
    pub input_node_dim: usize,
    pub final_window: bool,
    pub final_layer: bool,
}

impl OutputWindow {
    pub fn new(
        start_output_index: usize,
        end_output_index: usize,
        task_id: WindowId,
        output_node_dim: usize,
        input_node_dim: usize,
        final_window: bool,
        final_layer: bool,
    ) -> Self {
        OutputWindow {
            start_output_index,
            end_output_index,
            task_id,
            output_node_dim,
            input_node_dim,
            final_window,
            final_layer,
        }
    }
    pub fn get_output_len(&self) -> usize {
        self.end_output_index - self.start_output_index
    }
    pub fn get_output_dim(&self) -> usize {
        self.output_node_dim
    }
    pub fn get_input_dim(&self) -> usize {
        self.input_node_dim
    }
    pub fn get_task_id(&self) -> &WindowId {
        &self.task_id
    }
}

pub struct RangeIndex {
    start_input_index: usize,
    end_input_index: usize,
    start_output_index: usize,
    end_output_index: usize,
}

impl<'a> InputWindow<'a> {
    pub(super) fn new(
        task_id: WindowId,
        tasks: Rc<Vec<Range<'a, usize>>>,
        range_index: RangeIndex,
        output_window: Rc<OutputWindow>,
        is_last_row: bool,
    ) -> InputWindow<'a> {
        let RangeIndex {
            start_input_index,
            end_input_index,
            start_output_index,
            end_output_index,
        } = range_index;

        InputWindow {
            task_id,
            tasks,
            start_output_index,
            start_input_index,
            end_output_index,
            end_input_index,
            output_window,
            is_last_row,
        }
    }
    pub(super) fn get_task_id(&self) -> &WindowId {
        &self.task_id
    }
    pub(super) fn get_tasks(&self) -> &Vec<Range<'a, usize>> {
        &self.tasks
    }
    #[allow(dead_code)]
    pub(super) fn get_location_x(&self) -> (usize, usize) {
        (self.start_output_index, self.end_output_index)
    }
    #[allow(dead_code)]
    pub(super) fn get_location_y(&self) -> (usize, usize) {
        (self.start_input_index, self.end_input_index)
    }
    pub(super) fn get_output_window(&self) -> &Rc<OutputWindow> {
        &self.output_window
    }
}

#[derive(Debug)]
pub struct OutputWindowIterator<'a> {
    graph: &'a Graph,
    node_features: &'a NodeFeatures,
    agg_buffer_size: usize,
    input_buffer_size: usize,
    current_start_output_index: usize,
    task_id: WindowId,
    gcn_hidden_size: Vec<usize>,
    pub final_layer: bool,
    is_sparse: bool,
}

impl<'a> OutputWindowIterator<'a> {
    pub fn new(
        graph: &'a Graph,
        node_features: &'a NodeFeatures,
        window_iter_settings: WindowIterSettings,
    ) -> OutputWindowIterator<'a> {
        let WindowIterSettings {
            agg_buffer_size,
            input_buffer_size,
            layer,
            gcn_hidden_size,
            final_layer,
            is_sparse,
        } = window_iter_settings;
        OutputWindowIterator {
            graph,
            node_features,
            agg_buffer_size,
            input_buffer_size,
            current_start_output_index: 0,
            task_id: WindowId {
                layer_id: layer,
                output_id: 0,
                input_id: 0,
            },
            gcn_hidden_size,
            final_layer,
            is_sparse,
        }
    }
}
impl<'a> Iterator for OutputWindowIterator<'a> {
    type Item = InputWindowIterator<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_start_output_index >= self.graph.get_num_node() {
            return None;
        }
        // let output_size = (self.agg_buffer_size / 2) / (self.graph.get_feature_size() * 4);
        // fix bug here, the output feature size is gcn_hidden layer size!
        // fix bug again, the aggregated result size is unknown! we need to have enought space to store the aggregated result!
        // let output_size = self.gcn_hidden_size[self.task_id.layer_id] * 4;
        // let output_size = (self.agg_buffer_size / 2) / output_size;
        // fix another bug!, when the layer is not zero, the outout size is the gcn_hidden layer size!
        let output_size = match self.task_id.layer_id {
            0 => {
                debug!(
                    "it's the first layer, the agg buffer is:{}, the node size is:{}",
                    self.agg_buffer_size / 2,
                    self.graph.get_feature_size() * 4
                );
                (self.agg_buffer_size / 2) / (self.graph.get_feature_size() * 4)
            }
            _ => {
                debug!(
                    "it's not the first layer, the agg buffer is:{}, the gcn hidden size is:{}",
                    self.agg_buffer_size / 2,
                    self.gcn_hidden_size[self.task_id.layer_id - 1]
                );
                self.gcn_hidden_size[self.task_id.layer_id - 1]
            }
        };

        if output_size == 0 {
            panic!(
                "Output size is 0,agg_buffer_size:{},feature_size:{}",
                self.agg_buffer_size / 2,
                self.graph.get_feature_size() * 4
            );
        }
        let end_output_index = cmp::min(
            self.current_start_output_index + output_size,
            self.graph.get_num_node(),
        );
        let final_iter = { end_output_index >= self.graph.get_num_node() };
        let input_iter_settings = InputIterSettings {
            input_buffer_size: self.input_buffer_size,
            start_output_index: self.current_start_output_index,
            end_output_index,
            gcn_hidden_size: self.gcn_hidden_size.clone(),
            final_iter,
            final_layer: self.final_layer,
            is_spare: self.is_sparse,
        };
        let intput_iter = InputWindowIterator::new(
            self.task_id.clone(),
            self.graph,
            self.node_features,
            input_iter_settings,
        );
        self.task_id.output_id += 1;
        self.current_start_output_index = end_output_index;
        Some(intput_iter)
    }
}

#[derive(Debug)]
pub struct InputWindowIterator<'a> {
    task_id: WindowId,
    graph: &'a Graph,
    node_features: &'a NodeFeatures,
    input_buffer_size: usize,
    start_output_index: usize,
    end_output_index: usize,
    // current window information
    current_window_start_input_index: usize,
    current_window_end_input_index: usize,
    gcn_hidden_size: Vec<usize>,
    final_iter: bool,
    final_layer: bool,
    is_spare: bool,
}
// impl new for InputWindowIterator
pub struct InputIterSettings {
    pub input_buffer_size: usize,
    pub start_output_index: usize,
    pub end_output_index: usize,
    pub gcn_hidden_size: Vec<usize>,
    pub final_iter: bool,
    pub final_layer: bool,
    pub is_spare: bool,
}
impl<'a> InputWindowIterator<'a> {
    pub fn new(
        task_id: WindowId,
        graph: &'a Graph,
        node_features: &'a NodeFeatures,
        input_iter_settings: InputIterSettings,
    ) -> Self {
        let InputIterSettings {
            input_buffer_size,
            start_output_index,
            end_output_index,
            gcn_hidden_size,
            final_iter,
            final_layer,
            is_spare,
        } = input_iter_settings;
        InputWindowIterator {
            task_id,
            graph,
            node_features,
            input_buffer_size,
            start_output_index,
            end_output_index,
            current_window_end_input_index: 0,
            current_window_start_input_index: 0,
            gcn_hidden_size,
            final_iter,
            final_layer,
            is_spare,
        }
    }
}

impl<'a> Iterator for InputWindowIterator<'a> {
    type Item = InputWindow<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        // test if no window left
        if self.current_window_start_input_index >= self.graph.get_num_node() {
            None
        } else {
            // first skip all emtpy rows
            while self.current_window_start_input_index < self.graph.get_num_node() {
                if self
                    .graph
                    .is_row_range_empty(
                        self.current_window_start_input_index,
                        self.start_output_index,
                        self.end_output_index,
                    )
                    .expect("is_row_range_empty should always return Some")
                {
                    self.current_window_start_input_index += 1;
                } else {
                    break;
                }
            }
            if self.current_window_start_input_index == self.graph.get_num_node() {
                return None;
            }
            let task_id = self.task_id.clone();

            let input_node_dim = match task_id.layer_id {
                0 => self.graph.get_feature_size(),
                _ => *self.gcn_hidden_size.get(task_id.layer_id - 1).unwrap(),
            };

            let output_node_dim = match self.final_layer {
                true => 1,
                false => *self.gcn_hidden_size.get(self.task_id.layer_id).unwrap(),
            };

            // build the window
            let mut x_size = 0;
            // num of nodes in the window
            let mut x_len = 0;
            if self.is_spare {
                info!("build sparse window");
                while x_size < self.input_buffer_size / 2
                    && self.current_window_start_input_index + x_len < self.graph.get_num_node()
                {
                    let new_size = self
                        .node_features
                        .get_features(self.current_window_start_input_index + x_len)
                        .len()
                        * 4;
                    debug!(
                        "old size: {},new size: {}, max size: {}",
                        x_size,
                        new_size,
                        self.input_buffer_size / 2
                    );
                    // fix bug here, it's ok to equal!
                    if x_size + new_size > self.input_buffer_size / 2 {
                        debug!(
                            "break!xsize: {}, new size: {}, max size: {}",
                            x_size,
                            new_size,
                            self.input_buffer_size / 2
                        );
                        break;
                    }
                    x_size += new_size;
                    x_len += 1;
                }
                info!("x_size:{},x_len:{}", x_size, x_len);
            } else {
                // dense
                info!("build dense window");
                x_len += (self.input_buffer_size / 2) / (input_node_dim * 4);
                info!("x_len:{}", x_len);
            }

            debug!("the x_len is {}", x_len);
            if x_len == 0 {
                panic!("x_len is 0, the while input buffer cannot add one more node");
            }
            // shrink the window
            self.current_window_end_input_index = self.current_window_start_input_index + x_len;

            while self
                .graph
                .is_row_range_empty(
                    self.current_window_end_input_index - 1,
                    self.start_output_index,
                    self.end_output_index,
                )
                .expect("is_row_range_empty should always return Some")
            {
                debug!("shrink the window!");
                self.current_window_end_input_index -= 1;
            }

            // build the current window
            let csc = self.graph.get_csc();
            let mut tasks = Vec::new();
            let mut output_node_ids = Vec::new();
            for i in self.start_output_index..self.end_output_index {
                let task = csc.get(i).unwrap().range(
                    self.current_window_start_input_index..self.current_window_end_input_index,
                );

                tasks.push(task);
                output_node_ids.push(i);
            }

            let tasks = Rc::new(tasks);
            let final_window = self.final_iter;

            let mut next_start_row = self.current_window_start_input_index + x_len;
            // test if it't the last row: all the rows after end_input_index should be empty
            let mut is_last_row = true;

            while next_start_row < self.graph.get_num_node() {
                if !self
                    .graph
                    .is_row_range_empty(
                        next_start_row,
                        self.start_output_index,
                        self.end_output_index,
                    )
                    .expect("is_row_range_empty should always return Some")
                {
                    is_last_row = false;
                    break;
                }
                next_start_row += 1;
            }

            //let is_last_row= self.current_window_end_input_index == self.graph.get_num_node();
            let range_index = RangeIndex {
                start_input_index: self.current_window_start_input_index,
                end_input_index: self.current_window_end_input_index,
                start_output_index: self.start_output_index,
                end_output_index: self.end_output_index,
            };
            let current_window = InputWindow::new(
                task_id.clone(),
                tasks,
                range_index,
                Rc::new(OutputWindow::new(
                    self.start_output_index,
                    self.end_output_index,
                    task_id,
                    output_node_dim,
                    input_node_dim,
                    final_window,
                    self.final_layer,
                )),
                is_last_row,
            );

            // prepare the next start x and start y
            self.current_window_start_input_index = next_start_row;

            self.task_id.input_id += 1;
            Some(current_window)
        }
    }
}

#[cfg(test)]
mod test {
    use std::{fs::File, io::Write};

    use log::debug;

    use super::*;
    #[test]
    fn sliding_window_test() {
        simple_logger::init_with_level(log::Level::Info).unwrap_or_default();

        let graph_name = "test_data/graph1.txt";
        let features_name = "test_data/features.txt";
        let data = "f 2\n0 1 2\n1 2 0\n2 0 1\nend\n";
        let mut file = File::create(graph_name).unwrap();
        file.write_all(data.as_bytes()).unwrap();
        let data = "0 0 1 0 1 0\n1 0 0 1 1 1\n1 1 0 0 0 1\n";
        let mut file = File::create("test_data/features.txt").unwrap();
        file.write_all(data.as_bytes()).unwrap();
        debug!("graph:{}", "f 3\n0 1 2\n1 2 0\n2 0 1\nend\n");
        debug!("feature:{}", "0 0 1 0 1 0\n1 0 0 1 1 1\n1 1 0 0 0 1\n");

        let graph = Graph::new(graph_name).unwrap();
        let node_features = NodeFeatures::new(features_name).unwrap();
        let gcn_hidden_size = vec![2];
        let window_iter_settings = WindowIterSettings {
            agg_buffer_size: 32,
            input_buffer_size: 32,
            layer: 0,
            final_layer: false,
            is_sparse: true,
            gcn_hidden_size,
        };
        let output_window_iter =
            OutputWindowIterator::new(&graph, &node_features, window_iter_settings);
        for i in output_window_iter {
            debug!("{:?}\n", i);
            for j in i {
                debug!("{:?}\n", j);
            }
        }
    }
    #[test]
    fn sliding_window_test_multi() -> Result<(), Box<dyn std::error::Error>> {
        simple_logger::init_with_level(log::Level::Info).unwrap_or_default();

        let graph_name = "test_data/graph2.txt";
        let data = "f 6\n1 2\n2 3 4\n0 1 4\n0 2 4\n2 4\nend\n";
        let mut file = File::create(graph_name).unwrap();
        file.write_all(data.as_bytes()).unwrap();
        let feature1 = "1 1 0 0 1 1\n1 0 0 1 1 1\n1 1 1 0 0 1\n1 1 1 0 0 1\n1 1 1 0 0 1\n";
        let mut file = File::create("test_data/features1.txt").unwrap();
        file.write_all(feature1.as_bytes()).unwrap();
        let feature2 = "1 1\n1 1 \n1 1\n1 1\n1 1\n";
        let mut file = File::create("test_data/features2.txt").unwrap();
        file.write_all(feature2.as_bytes()).unwrap();

        debug!("graph:\n{}", "f 2\n1 2\n2 3 4\n0 1 4\n0 2 4\n2 4\nend\n");
        debug!("feature1:\n{}", feature1);
        debug!("feature2:\n{}", feature2);

        let graph = Graph::new(graph_name)?;
        let node_features1 = NodeFeatures::new("test_data/features1.txt")?;
        let node_features2 = NodeFeatures::new("test_data/features2.txt")?;
        let gcn_hidden_size = vec![2];
        // max input num=2, max output num=1
        let window_iter_settings = WindowIterSettings {
            agg_buffer_size: 48,
            input_buffer_size: 32,
            layer: 0,
            final_layer: false,
            is_sparse: true,
            gcn_hidden_size: gcn_hidden_size.clone(),
        };
        let output_window_iter =
            OutputWindowIterator::new(&graph, &node_features1, window_iter_settings);
        let mut total_windows = 0;
        for i in output_window_iter {
            debug!("{:?}\n\n", i);
            for j in i {
                total_windows += 1;
                debug!("{:?}\n\n", j);
            }
        }
        let window_iter_settings = WindowIterSettings {
            agg_buffer_size: 48,
            input_buffer_size: 32,
            layer: 1,
            final_layer: true,
            is_sparse: true,
            gcn_hidden_size,
        };
        let output_window_iter =
            OutputWindowIterator::new(&graph, &node_features2, window_iter_settings);
        for i in output_window_iter {
            debug!("{:?}\n\n", i);
            for j in i {
                total_windows += 1;
                debug!("{:?}\n\n", j);
            }
        }
        assert_eq!(total_windows, 20);
        Ok(())
    }
}
