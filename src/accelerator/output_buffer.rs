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
    /// ```
    ///
    /// use gcn_agg::accelerator::{output_buffer::{OutputBuffer, BufferStatus}};
    /// let mut output_buffer = OutputBuffer::new();
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
            (BufferStatus::WaitingToWriteBack, BufferStatus::Empty) => {
                swap(&mut self.current_state, &mut self.next_state);
            }
            _ => {}
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

    pub fn start_mlp(&mut self) {
        assert_eq!(self.current_state, BufferStatus::Empty);
        self.current_state = BufferStatus::Writing;
    }
    pub fn finished_writing(&mut self) {
        assert_eq!(self.current_state, BufferStatus::Writing);
        self.current_state = BufferStatus::WaitingToWriteBack;
    }
    pub fn start_writing_back(&mut self) {
        assert_eq!(self.next_state, BufferStatus::WaitingToWriteBack);
        self.next_state = BufferStatus::Empty;
    }
}
