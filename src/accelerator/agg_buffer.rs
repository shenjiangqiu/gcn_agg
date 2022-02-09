use std::mem::swap;

use super::sliding_window::Window;
#[derive(Debug, PartialEq)]
pub enum BufferStatus {
    Empty,
    Writing,
    WaitingToMlp,
    Mlp,
}
/// # Description
/// Aggregator is double buffer
/// 1. write to current buffer
/// 2. read from next buffer
///
/// # Example
///
/// ```ignore
/// use gcn_agg::accelerator::{agg_buffer::{AggBuffer, BufferStatus}};
/// // create a new buffer
/// let mut agg_buffer = AggBuffer::new();
/// // write to current buffer
/// agg_buffer.add_task(window);
/// // read from task
/// let window = agg_buffer.get_task();
/// // finished reading
/// agg_buffer.finish_reading();
/// // finished writing
/// agg_buffer.finish_writing();
///
/// ```
#[derive(Debug)]
pub struct AggBuffer<'a> {
    pub current_state: BufferStatus,
    pub next_state: BufferStatus,
    pub current_window: Option<Window<'a>>,
    pub next_window: Option<Window<'a>>,
}

impl<'a> AggBuffer<'a> {
    pub fn new() -> Self {
        AggBuffer {
            current_state: BufferStatus::Empty,
            next_state: BufferStatus::Empty,
            current_window: None,
            next_window: None,
        }
    }
}

impl<'a> AggBuffer<'a> {
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
            (BufferStatus::WaitingToMlp, BufferStatus::Empty) => {
                swap(&mut self.current_state, &mut self.next_state);
                swap(&mut self.current_window, &mut self.next_window);
            }
            _ => {}
        }
    }
    pub fn add_task(&mut self, window: &Window<'a>) {
        self.current_state = BufferStatus::Writing;
        
        self.current_window = match self.current_window{
            Some(ref mut window) => {
                window.add_window(window);
                Some(window.clone())
            },
            None => {
                Some(window.clone())
            }
        };
        }
    }
    pub fn finish_writing(&mut self) {
        self.current_state = BufferStatus::WaitingToMlp;
    }
    pub fn start_reading(&mut self) {
        self.next_state = BufferStatus::Mlp;
    }
    pub fn finish_reading(&mut self) {
        self.next_state = BufferStatus::Empty;
    }
    pub fn get_current_window(&self) -> Option<&Window<'a>> {
        self.current_window.as_ref()
    }
    pub fn get_next_window(&self) -> Option<&Window<'a>> {
        self.next_window.as_ref()
    }
    pub fn get_current_state(&self) -> &BufferStatus {
        &self.current_state
    }
    pub fn get_next_state(&self) -> &BufferStatus {
        &self.next_state
    }
}
