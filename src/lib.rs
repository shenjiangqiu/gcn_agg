//! the crate gcn_agg is a graph convolutional neural network accelerator simulator.
//! there are 4 parts in the crate:
//!
//! - accelerator: the accelerator is a graph convolutional neural network accelerator.
//! - graph: the data structure to represent the graph.
//! - node_features: the data structure to represent the node features.
//! - statics: the result statics to record the result.
//! # Examples
//! ```
//!     use chrono::Local;
//!     use gcn_agg::{
//!     accelerator::System, gcn_result::GcnAggResult, graph::Graph, node_features::NodeFeatures,
//!     settings::Settings,
//!     };
//!     use itertools::Itertools;
//!
//!     fn test_system() -> Result<(), Box<dyn std::error::Error>> {
//!         std::fs::create_dir_all("output")?;
//!
//!         simple_logger::init_with_level(log::Level::Info).unwrap_or(());
//!         let current_time: String = Local::now().format("%Y-%m-%d-%H-%M-%S%.6f").to_string();
//!
//!         let start_time = std::time::Instant::now();
//!         let mut results = GcnAggResult::default();
//!
//!         let settings = Settings::new(vec!["configs/default.toml".into()]).unwrap();
//!         results.settings = Some(settings.clone());
//!         // create the folder for output
//!         std::fs::create_dir_all("output")?;
//!
//!         let graph_name = &settings.graph_path;
//!         let features_name = &settings.features_paths;
//!
//!         let graph = Graph::new(graph_name.as_str())?;
//!
//!         let node_features: Vec<_> = features_name
//!             .iter()
//!             .map(|x| NodeFeatures::new(x.as_str()))
//!             .try_collect()?;
//!
//!         let mem_stat_path = format!("output/{}_mem_stat.txt", current_time);
//!         let mut system = System::new(
//!             &graph,
//!             &node_features,
//!             settings.accelerator_settings,
//!             &mem_stat_path,
//!         );
//!
//!         // run the system
//!         let mut stat = system.run()?;
//!
//!         // record the simulation time
//!         let simulation_time = start_time.elapsed().as_secs();
//!         // record the result
//!         let seconds = simulation_time % 60;
//!         let minutes = (simulation_time / 60) % 60;
//!         let hours = (simulation_time / 60) / 60;
//!         let time_str = format!("{}:{}:{}", hours, minutes, seconds);
//!         stat.simulation_time = time_str;
//!
//!         results.stats = Some(stat);
//!         let output_path = format!("output/{}.json", current_time);
//!
//!         println!("{}", serde_json::to_string_pretty(&results)?);
//!         // write json of results to output_path
//!         std::fs::write(output_path, serde_json::to_string_pretty(&results)?)?;
//!         Ok(())
//!     }
//!     match test_system() {
//!         Ok(_) => println!("test_system success"),
//!         Err(e) => println!("test_system failed: {}", e),
//!    }
//!```  
//!

pub mod accelerator;
pub mod cmd_args;
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
