use gcn_agg::accelerator::system::System;
use gcn_agg::settings::{
    AcceleratorSettings, AggregatorSettings, MlpSettings, Settings, SparsifierSettings,
};
use gcn_agg::{graph::Graph, node_features::NodeFeatures};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let settings = Settings::new(vec!["configs/default.toml".into()])?;

    let graph_name = &settings.graph_path;
    let features_name = &settings.features_paths;

    let graph = Graph::from(graph_name.as_str());

    let node_features = features_name
        .iter()
        .map(|x| NodeFeatures::from(x.as_str()))
        .collect::<Vec<_>>();

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
        gcn_hidden_size.len(),
        &gcn_hidden_size,
        systolic_rows,
        systolic_cols,
        mlp_sparse_cores,
        sparsifier_cores,
    );
    system.run()?;
    Ok(())
}
