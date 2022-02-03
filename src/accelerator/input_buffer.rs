use std::mem::swap;
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BufferStatus {
    Empty,
    WaitingToLoad(usize),
    Loading(usize),
    Reading(usize),
    Ready(usize),
}
pub struct InputBuffer {
    pub current_state: BufferStatus,
    pub next_state: BufferStatus,
}
impl InputBuffer {
    pub fn new() -> Self {
        InputBuffer {
            current_state: BufferStatus::Empty,
            next_state: BufferStatus::Empty,
        }
    }
    /// # Description
    /// simply swap the current and next state when current state is Empty
    ///
    /// # Example
    /// ```
    /// use gcn_agg::accelerator::{input_buffer::{InputBuffer, BufferStatus}};
    /// let mut input_buffer = InputBuffer::new();
    /// input_buffer.current_state = BufferStatus::Empty;
    /// input_buffer.next_state = BufferStatus::WaitingToLoad(1);
    /// assert_eq!(input_buffer.current_state, BufferStatus::Empty);
    /// assert_eq!(input_buffer.next_state, BufferStatus::WaitingToLoad(1));
    /// input_buffer.cycle();
    /// assert_eq!(input_buffer.current_state, BufferStatus::WaitingToLoad(1));
    /// assert_eq!(input_buffer.next_state, BufferStatus::Empty);
    ///
    /// ```
    ///
    pub fn cycle(&mut self) {
        match self.current_state {
            BufferStatus::Empty => {
                swap(&mut self.current_state, &mut self.next_state);
            }
            _ => {}
        }
    }

    /// # Description
    /// * make the Loading status to Ready status
    /// * either of current or next state is Loading status
    /// # Example
    /// ```
    /// use gcn_agg::accelerator::{input_buffer::{InputBuffer, BufferStatus}};
    /// let mut input_buffer = InputBuffer::new();
    /// input_buffer.current_state = BufferStatus::Empty;
    /// input_buffer.next_state = BufferStatus::Loading(1);
    /// assert_eq!(input_buffer.current_state, BufferStatus::Empty);
    /// assert_eq!(input_buffer.next_state, BufferStatus::Loading(1));
    /// input_buffer.receive(1);
    /// assert_eq!(input_buffer.current_state, BufferStatus::Empty);
    /// assert_eq!(input_buffer.next_state, BufferStatus::Ready(1));
    ///
    /// ```
    pub fn receive(&mut self, id_: usize) {
        // test if id match any

        match self.current_state {
            BufferStatus::Reading(id) => {
                if id == id_ {
                    self.current_state = BufferStatus::Ready(id);
                } else {
                    assert_eq!(self.next_state, BufferStatus::Loading(id_));
                    self.next_state = BufferStatus::Ready(id);
                }
            }
            _ => {
                assert_eq!(self.next_state, BufferStatus::Loading(id_));
                self.next_state = BufferStatus::Ready(id_);
            }
        }
    }
    /// # Description
    /// * try to get a waiting id to send
    /// * if there is no waiting id, return None
    /// * if there is a waiting id, return Some(id)
    /// # example
    /// ```
    ///
    /// use gcn_agg::accelerator::input_buffer::{InputBuffer, BufferStatus};
    /// let mut input_buffer = InputBuffer::new();
    /// input_buffer.current_state = BufferStatus::WaitingToLoad(1);
    /// assert_eq!(input_buffer.send_req(), Some(1));
    /// assert_eq!(input_buffer.send_req(), None);
    /// ```
    pub fn send_req(&mut self) -> Option<usize> {
        if let BufferStatus::WaitingToLoad(id) = self.current_state {
            self.current_state = BufferStatus::Loading(id);
            Some(id)
        } else if let BufferStatus::WaitingToLoad(id) = self.next_state {
            self.next_state = BufferStatus::Loading(id);
            Some(id)
        } else {
            None
        }
    }
}
