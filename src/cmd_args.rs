use clap::{Parser, ValueHint};
use clap_complete::Shell;

#[derive(Debug, Parser)]
#[clap(author="Jiangqiu Shen. <jshen2@mtu.edu>",version="0.1.0",about="a gcn simulator",long_about=None,trailing_var_arg=true)]
pub struct Args {
    /// generate shell completion
    #[clap(long = "generate", short = 'g', arg_enum)]
    pub generator: Option<Shell>,

    /// the paths of config files
    #[clap(value_hint=ValueHint::FilePath)]
    pub config_names: Vec<String>,
}
