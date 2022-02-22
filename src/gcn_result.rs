//! # mod gcn result
//! - this mod contains structs for recording the result of gcn simulation.
use crate::settings::Settings;
use serde::Serialize;
///
/// # Description
/// - struct for recording the result of gcn simulation.
/// # Fields
/// - settings: the settings of gcn simulation.`gcn_agg::settings::Settings`
/// - stats: the statistics
#[derive(Debug, Serialize)]
pub struct GcnAggResult {
    pub settings: Option<Settings>,
    pub stats: Option<GcnStatistics>,
}

impl GcnAggResult {
    pub fn new() -> Self {
        GcnAggResult {
            settings: None,
            stats: None,
        }
    }
}
/// # Description
/// - struct for recording the statistics of gcn simulation.
/// # Fields
/// - simulation_time: the simulation time
/// - cycle: the number of cycles
#[derive(Debug, Serialize)]
pub struct GcnStatistics {
    pub cycle: u64,
    pub simulation_time: String,
}

impl GcnStatistics {
    pub fn new() -> Self {
        GcnStatistics {
            cycle: 0,
            simulation_time: String::new(),
        }
    }
}
