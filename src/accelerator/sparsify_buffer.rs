//! this is sparsifier buffer
//! which is use to store the content from mlp and to sparsifier
//! 
//! also see output_buffer.rs
//! 


use std::{mem::swap, rc::Rc};

use super::{component::Component, sliding_window::OutputWindow};
#[derive(Debug, PartialEq)]
pub enum BufferStatus {
    Empty,
    Writing,
    WaitingToSparsify,
    Sparsifying,
}
#[derive(Debug)]
pub struct SparsifyBuffer {
    pub current_state: BufferStatus,
    pub next_state: BufferStatus,
    pub current_window: Option<Rc<OutputWindow>>,
    pub next_window: Option<Rc<OutputWindow>>,
}

impl Component for SparsifyBuffer {
    /// # Description
    /// simply swap the current and next state when current state is Empty
    ///
    /// # Example
    /// ```
    ///
    /// use gcn_agg::accelerator::{sparsify_buffer::{sparsify_buffer, BufferStatus}};
    /// let mut output_buffer = sparsify_buffer::new();
    /// output_buffer.start_writing(1);
    /// assert_eq!(output_buffer.current_state, BufferStatus::Writing(1));
    /// output_buffer.finished_writing(1);
    /// assert_eq!(output_buffer.current_state, BufferStatus::WaitingToWriteBack(1));
    /// output_buffer.cycle();
    /// assert_eq!(output_buffer.current_state, BufferStatus::Empty);
    /// assert_eq!(output_buffer.next_state, BufferStatus::WaitingToWriteBack(1));
    ///
    /// output_buffer.start_writing_back(1);
    /// assert_eq!(output_buffer.next_state, BufferStatus::WritingBack(1));
    /// output_buffer.finished_writing_back(1);
    /// assert_eq!(output_buffer.next_state, BufferStatus::Empty);
    ///
    /// ```
    ///
    fn cycle(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        match (&self.current_state, &self.next_state) {
            (BufferStatus::WaitingToSparsify, BufferStatus::Empty) => {
                swap(&mut self.current_state, &mut self.next_state);
                swap(&mut self.current_window, &mut self.next_window);
            }
            _ => {}
        }
        Ok(())
    }
}

impl SparsifyBuffer {
    pub fn new() -> Self {
        SparsifyBuffer {
            current_state: BufferStatus::Empty,
            next_state: BufferStatus::Empty,
            current_window: None,
            next_window: None,
        }
    }

    pub fn start_mlp(&mut self) {
        assert_eq!(self.current_state, BufferStatus::Empty);
        self.current_state = BufferStatus::Writing;
    }
    pub fn finished_writing(&mut self) {
        assert_eq!(self.current_state, BufferStatus::Writing);
        self.current_state = BufferStatus::WaitingToSparsify;
    }
    pub fn start_sparsify(&mut self) {
        assert_eq!(self.next_state, BufferStatus::WaitingToSparsify);
        self.next_state = BufferStatus::Sparsifying;
    }
    pub fn finished_sparsify(&mut self) {
        assert_eq!(self.next_state, BufferStatus::Sparsifying);
        self.next_state = BufferStatus::Empty;
    }
}
