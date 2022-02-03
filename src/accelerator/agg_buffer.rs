use std::mem::swap;
#[derive(Debug, PartialEq)]
pub enum BufferStatus {
    Empty,
    Writing(usize),
    WaitingToMlp(usize),
    Mlp(usize),
}
pub struct AggBuffer {
    pub current_state: BufferStatus,
    pub next_state: BufferStatus,
}

impl AggBuffer {
    pub fn new() -> Self {
        AggBuffer {
            current_state: BufferStatus::Empty,
            next_state: BufferStatus::Empty,
        }
    }
}

impl AggBuffer {
    /// # Description
    /// simply swap the current and next state when current state is Empty
    ///
    /// # Example
    /// ```
    /// use gcn_agg::accelerator::{agg_buffer::{AggBuffer, BufferStatus}};
    /// let mut agg_buffer = AggBuffer::new();
    /// agg_buffer.start_writing(1);
    /// assert_eq!(agg_buffer.current_state, BufferStatus::Writing(1));
    /// agg_buffer.finished_writing(1);
    /// assert_eq!(agg_buffer.current_state, BufferStatus::WaitingToMlp(1));
    /// agg_buffer.cycle();
    /// assert_eq!(agg_buffer.current_state, BufferStatus::Empty);
    /// assert_eq!(agg_buffer.next_state, BufferStatus::WaitingToMlp(1));
    /// 
    /// agg_buffer.start_mlp(1);
    /// assert_eq!(agg_buffer.next_state, BufferStatus::Mlp(1));
    /// agg_buffer.finished_mlp(1);
    /// assert_eq!(agg_buffer.next_state, BufferStatus::Empty);
    ///
    /// ```
    ///
    pub fn cycle(&mut self) {
        match (&self.current_state, &self.next_state) {
            (BufferStatus::WaitingToMlp(_), BufferStatus::Empty) => {
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
        self.current_state = BufferStatus::WaitingToMlp(id_);
    }
    pub fn start_mlp(&mut self, id_: usize) {
        assert_eq!(self.next_state, BufferStatus::WaitingToMlp(id_));
        self.next_state = BufferStatus::Mlp(id_);
    }
    pub fn finished_mlp(&mut self, id_: usize) {
        assert_eq!(self.next_state, BufferStatus::Mlp(id_));
        self.next_state = BufferStatus::Empty;
    }

}
