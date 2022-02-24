use chrono::Local;
use clap::{Command, CommandFactory, Parser};
use clap_complete::{generate, Generator};
use gcn_agg::{cmd_args::Args, settings::Settings, GcnAggResult, Graph, NodeFeatures, System};
use itertools::Itertools;
use std::io;
fn print_completions<G: Generator>(gen: G, cmd: &mut Command) {
    generate(gen, cmd, cmd.get_name().to_string(), &mut io::stdout());
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    simple_logger::init_with_level(log::Level::Info)?;
    let start_time = std::time::Instant::now();
    let current_time: String = Local::now().format("%Y-%m-%d-%H-%M-%S%.6f").to_string();

    let mut config_names = vec![String::from("configs/default.toml")];
    let args = Args::parse();
    if let Some(generator) = args.generator {
        let mut cmd = Args::command();
        eprintln!("Generating completion file for {:?}...", generator);
        print_completions(generator, &mut cmd);
        return Ok(());
    }
    println!("{:?}", args);
    let margs = args.config_names;

    // config_names append args
    for arg in margs.into_iter() {
        config_names.push(arg);
    }

    let mut results = GcnAggResult::default();
    let settings = Settings::new(config_names)?;
    results.settings = Some(settings.clone());
    println!("{}", serde_json::to_string_pretty(&settings)?);
    // create the folder for output
    std::fs::create_dir_all("output")?;

    let graph_name = &settings.graph_path;
    let features_name = &settings.features_paths;

    let graph = Graph::new(graph_name.as_str())?;

    let node_features: Vec<_> = features_name
        .iter()
        .map(|x| NodeFeatures::new(x.as_str()))
        .try_collect()?;
    let stats_name = format!("output/{}_mem_stat.txt", current_time);
    let mut system = System::new(
        &graph,
        &node_features,
        settings.accelerator_settings,
        &stats_name,
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
    let output_path = format!("output/{}.json", current_time);

    println!("{}", serde_json::to_string_pretty(&results)?);
    // write json of results to output_path
    std::fs::write(output_path, serde_json::to_string_pretty(&results)?)?;
    Ok(())
}
