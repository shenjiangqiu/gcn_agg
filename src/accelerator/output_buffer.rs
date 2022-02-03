use std::mem::swap;
#[derive(Debug, PartialEq)]
pub enum BufferStatus {
    Empty,
    Writing(usize),
    WaitingToWriteBack(usize),
    WritingBack(usize),
}
pub struct OutputBuffer {
    pub current_state: BufferStatus,
    pub next_state: BufferStatus,
}

impl OutputBuffer {
    pub fn new() -> Self {
        OutputBuffer {
            current_state: BufferStatus::Empty,
            next_state: BufferStatus::Empty,
        }
    }
}

impl OutputBuffer {
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
    pub fn cycle(&mut self) {
        match (&self.current_state, &self.next_state) {
            (BufferStatus::WaitingToWriteBack(_), BufferStatus::Empty) => {
                swap(&mut self.current_state, &mut self.next_state);
            }
            _ => {}
        }
    }

    pub fn start_writing(&mut self, id_: usize) {
        assert_eq!(self.current_state, BufferStatus::Empty);
        self.current_state = BufferStatus::Writing(id_);
    }
    pub fn finished_writing(&mut self, id_: usize) {
        assert_eq!(self.current_state, BufferStatus::Writing(id_));
        self.current_state = BufferStatus::WaitingToWriteBack(id_);
    }
    pub fn start_writing_back(&mut self, id_: usize) {
        assert_eq!(self.next_state, BufferStatus::WaitingToWriteBack(id_));
        self.next_state = BufferStatus::WritingBack(id_);
    }
    pub fn finished_writing_back(&mut self, id_: usize) {
        assert_eq!(self.next_state, BufferStatus::WritingBack(id_));
        self.next_state = BufferStatus::Empty;
    }

}
