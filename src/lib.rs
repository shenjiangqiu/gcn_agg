//! the crate gcn_agg is a graph convolutional neural network accelerator simulator.
//! there are 4 parts in the crate:
//!
//! - accelerator: the accelerator is a graph convolutional neural network accelerator.
//! - graph: the data structure to represent the graph.
//! - node_features: the data structure to represent the node features.
//! - statics: the result statics to record the result.
//!
//!

pub mod accelerator;
pub mod gcn_result;
pub mod graph;
pub mod node_features;
pub mod settings;

// default re-export
pub use accelerator::System;
pub use gcn_result::{GcnAggResult, GcnStatistics};
pub use graph::Graph;
pub use node_features::NodeFeatures;
pub use settings::Settings;
