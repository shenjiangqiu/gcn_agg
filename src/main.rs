use chrono::Local;
use gcn_agg::accelerator::system::System;
use gcn_agg::gcn_result::GcnAggResult;
use gcn_agg::settings::{
    AcceleratorSettings, AggregatorSettings, MlpSettings, Settings, SparsifierSettings,
};
use gcn_agg::{graph::Graph, node_features::NodeFeatures};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    simple_logger::init_with_level(log::Level::Warn)?;
    let start_time = std::time::Instant::now();

    let mut config_names = vec![String::from("configs/default.toml")];
    let margs: Vec<String> = std::env::args().collect();
    if let Some(first_arg) = margs.get(1) {
        if first_arg == "--help" || first_arg == "-h" {
            println!("Usage: gcn_agg [config_file ...]");
            println!("If no config file is specified, the default config file is used.");
            println!("The default config file is configs/default.toml");
            println!("all the config files in configs/user_configs/ are also automaticly loaded.");
            println!("for example: gcn_agg configs/optional_configs/my_config.toml");
            println!("will load configs/default.toml and configs/optional_configs/my_config.toml and all the config files in configs/user_configs/");
            return Ok(());
        }
    }

    // config_names append args
    for arg in margs.into_iter().skip(1) {
        config_names.push(arg);
    }

    let mut results = GcnAggResult::new();
    let settings = Settings::new(config_names)?;
    results.settings = Some(settings.clone());
    println!("{}", serde_json::to_string_pretty(&settings)?);
    // create the folder for output
    std::fs::create_dir_all("output")?;

    let graph_name = &settings.graph_path;
    let features_name = &settings.features_paths;

    let graph = Graph::new(graph_name.as_str())?;

    let node_features = features_name
        .iter()
        .map(|x| NodeFeatures::new(x.as_str()))
        .collect::<Result<_, _>>()?;

    let AcceleratorSettings {
        input_buffer_size,
        agg_buffer_size,
        gcn_hidden_size,
        aggregator_settings,
        mlp_settings,
        sparsifier_settings,
        output_buffer_size,
    } = settings.accelerator_settings;

    let AggregatorSettings {
        sparse_cores,
        sparse_width,
        dense_cores,
        dense_width,
    } = aggregator_settings;

    let MlpSettings {
        systolic_rows,
        systolic_cols,
        mlp_sparse_cores,
    } = mlp_settings;

    let SparsifierSettings { sparsifier_cores } = sparsifier_settings;

    let mut system = System::new(
        &graph,
        &node_features,
        sparse_cores,
        sparse_width,
        dense_cores,
        dense_width,
        input_buffer_size,
        agg_buffer_size,
        output_buffer_size,
        node_features.len(),
        &gcn_hidden_size,
        systolic_rows,
        systolic_cols,
        mlp_sparse_cores,
        sparsifier_cores,
    );

    // run the system
    let mut stat = system.run()?;

    // record the simulation time
    let simulation_time = start_time.elapsed().as_secs();
    // record the result
    let seconds = simulation_time % 60;
    let minutes = (simulation_time / 60) % 60;
    let hours = (simulation_time / 60) / 60;
    let time_str = format!("{}:{}:{}", hours, minutes, seconds);
    stat.simulation_time = time_str;

    results.stats = Some(stat);
    let current_time: String = Local::now().format("%Y-%m-%d-%H-%M-%S%.6f").to_string();
    let output_path = format!("output/{}.json", current_time);

    println!("{}", serde_json::to_string_pretty(&results)?);
    // write json of results to output_path
    std::fs::write(output_path, serde_json::to_string_pretty(&results)?)?;
    return Ok(());
}
