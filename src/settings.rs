//! # the settings of the gcn accelerator
//! - this mod contains the settings of the gcn accelerator.
//!
use config::{Config, File};
use glob::glob;
use itertools::Itertools;

use serde::{Deserialize, Serialize};
use std::{error::Error, string::String};

/// # Description
/// - struct for recording the settings of gcn accelerator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub description: String,
    pub graph_path: String,
    pub features_paths: Vec<String>,
    pub accelerator_settings: AcceleratorSettings,
}

/// # Description
/// - struct for recording the settings of gcn accelerator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceleratorSettings {
    pub input_buffer_size: usize,
    pub agg_buffer_size: usize,
    // pub output_buffer_size: usize,
    pub gcn_hidden_size: Vec<usize>,
    pub aggregator_settings: AggregatorSettings,
    pub mlp_settings: MlpSettings,
    pub sparsifier_settings: SparsifierSettings,
    pub is_sparse: bool,
}

/// # Description
/// - struct for recording the settings of aggregator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatorSettings {
    pub sparse_cores: usize,
    pub sparse_width: usize,
    pub dense_cores: usize,
    pub dense_width: usize,
}
/// # Description
/// - struct for recording the settings of mlp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlpSettings {
    pub systolic_rows: usize,
    pub systolic_cols: usize,
    pub mlp_sparse_cores: usize,
}

/// # Description
/// - struct for recording the settings of sparsifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparsifierSettings {
    pub sparsifier_cores: usize,
}

impl Settings {
    /// # Description
    /// - create the settings of gcn accelerator.
    /// - will read all configs provided in the config_path.
    /// - the configs/user_configs/*.toml will also be read.
    /// # Arguments
    /// - `config_path`: the vec of paths of the config file with surfix `.toml`.
    /// # Return
    /// - `Result<Settings, ConfigError>`: the settings of gcn accelerator.
    pub fn new(config_path: Vec<String>) -> Result<Self, Box<dyn Error>> {
        let input_files = config_path.iter().map(|x| File::with_name(x)).collect_vec();
        let default_files: Vec<_> = glob("configs/user_configs/*.toml")?
            .map_ok(File::from)
            .try_collect()?;

        let result: Settings = Config::builder()
            .add_source(input_files)
            .add_source(default_files)
            .build()?
            .try_deserialize()?;

        match result.features_paths.len() - result.accelerator_settings.gcn_hidden_size.len() {
            1 => Ok(result),
            _ => Err(
                "the number of features paths is not equal to the number of gcn hidden size".into(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json;

    #[test]
    fn test_settings() -> Result<(), Box<dyn std::error::Error>> {
        let settings = super::Settings::new(vec!["configs/default.toml".into()])?;
        // serialize settings to json
        let json = serde_json::to_string_pretty(&settings)?;
        println!("{}", json);
        Ok(())
    }
}
