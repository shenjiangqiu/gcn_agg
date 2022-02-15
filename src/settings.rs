use config::{Config, ConfigError, File};
use serde::{Deserialize, Serialize};
use std::string::String;

#[derive(Debug, Serialize, Deserialize)]
pub struct Settings {
    pub graph_path: String,
    pub features_paths: Vec<String>,
    pub accelerator_settings: AcceleratorSettings,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AcceleratorSettings {
    pub input_buffer_size: usize,
    pub agg_buffer_size: usize,
    pub output_buffer_size: usize,
    pub gcn_hidden_size: Vec<usize>,
    pub aggregator_settings: AggregatorSettings,
    pub mlp_settings: MlpSettings,
    pub sparsifier_settings: SparsifierSettings,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AggregatorSettings {
    pub sparse_cores: usize,
    pub sparse_width: usize,
    pub dense_cores: usize,
    pub dense_width: usize,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct MlpSettings {
    pub systolic_rows: usize,
    pub systolic_cols: usize,
    pub mlp_sparse_cores: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SparsifierSettings {
    pub sparsifier_cores: usize,
}

pub struct StringWrapper {
    pub string: String,
}

impl Settings {
    pub fn new(config_path: Vec<String>) -> Result<Self, ConfigError> {
        let mut s = Config::new();
        for i in config_path {
            s.merge(File::with_name(&i))?;
        }
        s.try_into()
    }
}

#[cfg(test)]
mod tests {
    use serde_json;

    #[test]
    fn test_settings() {
        let settings = super::Settings::new(vec!["configs/default.toml".into()]).unwrap();
        // serialize settings to json
        let json = serde_json::to_string_pretty(&settings).unwrap();
        println!("{}", json);
    }
}
