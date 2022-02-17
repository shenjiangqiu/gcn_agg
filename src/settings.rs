use config::{Config, ConfigError, File};
use glob::glob;
use log::debug;
use serde::{Deserialize, Serialize};
use std::string::String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub description: String,
    pub graph_path: String,
    pub features_paths: Vec<String>,
    pub accelerator_settings: AcceleratorSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceleratorSettings {
    pub input_buffer_size: usize,
    pub agg_buffer_size: usize,
    pub output_buffer_size: usize,
    pub gcn_hidden_size: Vec<usize>,
    pub aggregator_settings: AggregatorSettings,
    pub mlp_settings: MlpSettings,
    pub sparsifier_settings: SparsifierSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatorSettings {
    pub sparse_cores: usize,
    pub sparse_width: usize,
    pub dense_cores: usize,
    pub dense_width: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlpSettings {
    pub systolic_rows: usize,
    pub systolic_cols: usize,
    pub mlp_sparse_cores: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
            debug!("Loading config file: {}", i);
            s.merge(File::with_name(&i))?;
        }
        // merge all config files in configs/user_configs/
        let user_files = glob("configs/user_configs/*.toml").unwrap().map(|x| {
            debug!("Loading user config file: {:?}", x);
            File::from(x.unwrap())
        });

        for i in user_files {
            s.merge(i)?;
        }

        let result: Self = s.try_into()?;
        if result.features_paths.len() == result.accelerator_settings.gcn_hidden_size.len() + 1 {
            Ok(result)
        } else {
            Err(ConfigError::Message(String::from(
                "Number of features files does not match the number of hidden layers, feature path should be one more than the number of hidden layers(including the input layer)",
            )))
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
