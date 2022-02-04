use super::{req::Req, sliding_window::Window};
use std::mem::swap;
#[derive(Debug, Clone)]
pub enum BufferStatus<'a> {
    Empty,
    WaitingToLoad(Req, Window<'a>),
    Loading(Req, Window<'a>),
    Reading(Req, Window<'a>),
    Ready(Req, Window<'a>),
}
#[derive(Debug)]
pub struct InputBuffer<'a> {
    pub current_state: BufferStatus<'a>,
    pub next_state: BufferStatus<'a>,
}
impl<'a> InputBuffer<'a> {
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
    /// assert_eq!(input_buffer.current_state, BufferStatus::Empty);
    /// assert_eq!(input_buffer.next_state, BufferStatus::WaitingToLoad(1));
    /// input_buffer.next_state = BufferStatus::Ready(1);
    /// input_buffer.cycle();
    /// assert_eq!(input_buffer.current_state, BufferStatus::Ready(1));
    /// assert_eq!(input_buffer.next_state, BufferStatus::Empty);
    ///
    ///
    /// ```
    ///
    pub fn cycle(&mut self) {
        match (&self.current_state, &self.next_state) {
            // both are empty, do nothing
            (BufferStatus::Empty, BufferStatus::Empty) => {}
            // current is empty, next is not empty, swap
            (BufferStatus::Empty, _) => {
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
    pub fn receive(&mut self, id_: &Req) {
        // test if id match any
        match (self.current_state, self.next_state) {
            (BufferStatus::Loading(id, window), _) if &id == id_ => {
                self.current_state = BufferStatus::Ready(id, window);
            }
            (_, BufferStatus::Loading(id, window)) if &id == id_ => {
                self.next_state = BufferStatus::Ready(id, window);
            }
            _ => {
                panic!(
                    "receive id: {:?} but current state is {:?} and next state is {:?}",
                    id_, self.current_state, self.next_state
                );
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
    pub fn send_req(&mut self,id: &Req) {
        match (self.current_state,self.nex) {
            
        }
    }

    pub fn add_task(&mut self, id_: Req, window: Window<'a>) {
        self.next_state = BufferStatus::WaitingToLoad(id_, window);
    }
}
