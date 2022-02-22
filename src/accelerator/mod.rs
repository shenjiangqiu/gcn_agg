//! # Description
//! - this module is the accelerator module
//! - the main sub module is system, all the components are in system
//! - read system.rs for more details
//! 
//! # Components
//! - system: the main sub module, all the components are in system
//! - agg_buffer and other buffers: provide data for aggregator and mlp
//! - aggregator and mlp, the module for calculating the result
//! - mem_interface: the interface between system and memory(ramulator)
//! 


pub(self) mod aggregator;
pub(self) mod system;
pub(self) mod mem_interface;
pub(self) mod input_buffer;
pub(self) mod sparsify_buffer;
pub(self) mod agg_buffer;
pub(self) mod sliding_window;
pub(self) mod window_id;
pub(self) mod mlp;
pub(self) mod component;
pub(self) mod temp_agg_result;
pub(self) mod sparsifier;
pub(self) mod output_buffer;
pub use system::System;