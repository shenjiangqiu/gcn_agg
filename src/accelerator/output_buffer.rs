//! this is the output buffer for the accelerator
//! the output buffer is used to store the result from sparsifier  and write back to memory
//! also see sparsify_buffer.rs

use std::{mem::swap, rc::Rc};

use super::{component::Component, sliding_window::OutputWindow};
#[derive(Debug, PartialEq)]
pub enum BufferStatus {
    Empty,
    Writing,
    WaitingToWriteBack,
}
#[derive(Debug)]
pub struct OutputBuffer {
    pub current_state: BufferStatus,
    pub next_state: BufferStatus,
    pub current_window: Option<Rc<OutputWindow>>,
    pub next_window: Option<Rc<OutputWindow>>,
}

impl Component for OutputBuffer {
    /// # Description
    /// simply swap the current and next state when current state is Empty
    ///
    /// # Example
    /// ```ignore
    ///
    /// use gcn_agg::accelerator::{sparsify_buffer::{sparsify_buffer, BufferStatus}};
    /// let mut OutputBuffer = sparsify_buffer::new();
    /// OutputBuffer.start_writing(1);
    /// assert_eq!(OutputBuffer.current_state, BufferStatus::Writing(1));
    /// OutputBuffer.finished_writing(1);
    /// assert_eq!(OutputBuffer.current_state, BufferStatus::WaitingToWriteBack(1));
    /// OutputBuffer.cycle();
    /// assert_eq!(OutputBuffer.current_state, BufferStatus::Empty);
    /// assert_eq!(OutputBuffer.next_state, BufferStatus::WaitingToWriteBack(1));
    ///
    /// OutputBuffer.start_writing_back(1);
    /// assert_eq!(OutputBuffer.next_state, BufferStatus::WritingBack(1));
    /// OutputBuffer.finished_writing_back(1);
    /// assert_eq!(OutputBuffer.next_state, BufferStatus::Empty);
    ///
    /// ```
    ///
    fn cycle(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let (BufferStatus::WaitingToWriteBack, BufferStatus::Empty) = (&self.current_state, &self.next_state) {
            swap(&mut self.current_state, &mut self.next_state);
            swap(&mut self.current_window, &mut self.next_window);
        }
        Ok(())
    }
}

impl OutputBuffer {
    pub fn new() -> Self {
        OutputBuffer {
            current_state: BufferStatus::Empty,
            next_state: BufferStatus::Empty,
            current_window: None,
            next_window: None,
        }
    }

    pub fn start_sparsify(&mut self, window: Rc<OutputWindow>) {
        assert_eq!(self.current_state, BufferStatus::Empty);
        self.current_state = BufferStatus::Writing;
        self.current_window = Some(window);
    }
    pub fn finished_sparsify(&mut self) {
        assert_eq!(self.current_state, BufferStatus::Writing);
        self.current_state = BufferStatus::WaitingToWriteBack;
    }
    pub fn start_write_back(&mut self) {
        assert_eq!(self.next_state, BufferStatus::WaitingToWriteBack);
        self.next_state = BufferStatus::Empty;
    }
}
