use log::debug;

use super::{component::Component, sliding_window::InputWindow, window_id::WindowId};
use std::mem::swap;
#[derive(Debug, Clone)]
pub enum BufferStatus {
    Empty,
    WaitingToLoad,
    Loading,
    Reading,
    Ready,
}
#[derive(Debug)]
pub struct InputBuffer<'a> {
    current_state: BufferStatus,
    next_state: BufferStatus,
    current_window: Option<InputWindow<'a>>,
    next_window: Option<InputWindow<'a>>,
}
impl Component for InputBuffer<'_> {
    /// # Description
    /// simply swap the current and next state when current state is Empty
    ///
    /// # Example
    /// ```ignore
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
    fn cycle(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        match (&self.current_state, &self.next_state) {
            // both are empty, do nothing
            (BufferStatus::Empty, BufferStatus::Empty) => {}
            // current is empty, next is not empty, swap
            (BufferStatus::Empty, _) => {
                swap(&mut self.current_state, &mut self.next_state);
                swap(&mut self.current_window, &mut self.next_window);
            }
            // current is not empty, do nothing
            _ => {}
        }
        Ok(())
    }
}

impl<'a> InputBuffer<'a> {
    pub fn new() -> Self {
        InputBuffer {
            current_state: BufferStatus::Empty,
            next_state: BufferStatus::Empty,
            current_window: None,
            next_window: None,
        }
    }

    /// # Description
    /// * make the Loading status to Ready status
    /// * either of current or next state is Loading status
    /// # Example
    /// ```ignore
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
    pub fn receive(&mut self, id_: &WindowId) {
        // test if id match any
        match (
            &self.current_state,
            &self.current_window,
            &self.next_state,
            &self.next_window,
        ) {
            // match current is loading and current window's id match
            (
                BufferStatus::Loading,
                Some(InputWindow {
                    task_id: ref id, ..
                }),
                ..,
            ) if id == id_ => {
                self.current_state = BufferStatus::Ready;
            }

            // match next is loading and next window's id match
            (
                ..,
                BufferStatus::Loading,
                Some(InputWindow {
                    task_id: ref id, ..
                }),
            ) if id == id_ => {
                self.next_state = BufferStatus::Ready;
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
    /// ```ignore
    ///
    /// use gcn_agg::accelerator::input_buffer::{InputBuffer, BufferStatus};
    /// let mut input_buffer = InputBuffer::new();
    /// input_buffer.current_state = BufferStatus::WaitingToLoad(1);
    /// assert_eq!(input_buffer.send_req(), Some(1));
    /// assert_eq!(input_buffer.send_req(), None);
    /// ```
    pub fn send_req(&mut self, is_current: bool) {
        // test if id match any
        match is_current {
            true => {
                self.current_state = BufferStatus::Loading;
            }
            false => {
                self.next_state = BufferStatus::Loading;
            }
        }
    }

    pub fn add_task_to_next(&mut self, window: InputWindow<'a>) {
        self.next_state = BufferStatus::WaitingToLoad;
        self.next_window = Some(window);
    }

    pub fn add_task_to_current(&mut self, window: InputWindow<'a>) {
        debug!("input buffer receive current: {:?}", &window);
        self.current_state = BufferStatus::WaitingToLoad;
        self.current_window = Some(window);
    }

    // pub fn is_current_empty(&self) -> bool {
    //     match self.current_state {
    //         BufferStatus::Empty => true,
    //         _ => false,
    //     }
    // }
    // pub fn is_next_empty(&self) -> bool {
    //     match self.next_state {
    //         BufferStatus::Empty => true,
    //         _ => false,
    //     }
    // }

    // pub fn is_current_ready(&self) -> bool {
    //     match self.current_state {
    //         BufferStatus::Ready => true,
    //         _ => false,
    //     }
    // }

    // pub fn is_next_ready(&self) -> bool {
    //     match self.next_state {
    //         BufferStatus::Ready => true,
    //         _ => false,
    //     }
    // }

    // pub fn is_current_loading(&self) -> bool {
    //     match self.current_state {
    //         BufferStatus::Loading => true,
    //         _ => false,
    //     }
    // }

    // pub fn is_next_loading(&self) -> bool {
    //     match self.next_state {
    //         BufferStatus::Loading => true,
    //         _ => false,
    //     }
    // }

    // pub fn is_current_waiting_to_load(&self) -> bool {
    //     match self.current_state {
    //         BufferStatus::WaitingToLoad => true,
    //         _ => false,
    //     }
    // }

    // pub fn is_next_waiting_to_load(&self) -> bool {
    //     match self.next_state {
    //         BufferStatus::WaitingToLoad => true,
    //         _ => false,
    //     }
    // }

    // pub fn get_current_id(&self) -> Option<&WindowId> {
    //     match &self.current_window {
    //         Some(InputWindow { task_id: id, .. }) => Some(id),
    //         None => None,
    //     }
    // }

    // pub fn get_next_id(&self) -> Option<&WindowId> {
    //     match self.next_window {
    //         Some(InputWindow {
    //             task_id: ref id, ..
    //         }) => Some(id),
    //         None => None,
    //     }
    // }

    pub fn get_current_window(&self) -> Option<&InputWindow<'a>> {
        self.current_window.as_ref()
    }

    pub fn get_next_window(&self) -> Option<&InputWindow<'a>> {
        self.next_window.as_ref()
    }

    pub fn get_current_state(&self) -> &BufferStatus {
        &self.current_state
    }

    pub fn get_next_state(&self) -> &BufferStatus {
        &self.next_state
    }

    // pub fn get_current_layer(&self) -> Option<usize> {
    //     match &self.current_window {
    //         Some(InputWindow { task_id, .. }) => Some(task_id.layer_id),
    //         None => None,
    //     }
    // }

    // pub fn get_next_layer(&self) -> Option<usize> {
    //     match &self.next_window {
    //         Some(InputWindow { task_id, .. }) => Some(task_id.layer_id),
    //         None => None,
    //     }
    // }

    pub fn finished_aggregation(&mut self) {
        self.current_state = BufferStatus::Empty;
    }
    pub(super) fn start_aggragating(&mut self) {
        self.current_state = BufferStatus::Reading;
    }
 }
