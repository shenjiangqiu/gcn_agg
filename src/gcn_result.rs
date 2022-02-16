use crate::settings::Settings;
use serde::Serialize;

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
