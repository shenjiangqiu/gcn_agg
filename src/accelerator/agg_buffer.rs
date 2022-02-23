use std::{mem::swap, rc::Rc};

use super::{component::Component, sliding_window::OutputWindow, temp_agg_result::TempAggResult};
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
pub struct AggBuffer {
    current_state: BufferStatus,
    next_state: BufferStatus,
    current_window: Option<Rc<OutputWindow>>,
    next_window: Option<Rc<OutputWindow>>,

    // the temp result for the aggregation, when the aggregation result is finished, empty those temp result
    current_temp_result: Option<TempAggResult>,
    next_temp_result: Option<TempAggResult>,
}

impl AggBuffer {
    pub(super) fn new(num_nodes: usize, is_parse: bool) -> Self {
        let (current_temp_result, next_temp_result) = match is_parse {
            true => (
                Some(TempAggResult::new(num_nodes)),
                Some(TempAggResult::new(num_nodes)),
            ),
            false => (None, None),
        };

        AggBuffer {
            current_state: BufferStatus::Empty,
            next_state: BufferStatus::Empty,
            current_window: None,
            next_window: None,
            current_temp_result,
            next_temp_result,
        }
    }
}
impl Component for AggBuffer {
    /// # Description
    /// simply swap the current and next state when current state is Empty
    ///
    /// # Example
    /// ```ignore
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
    fn cycle(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let (BufferStatus::WaitingToMlp, BufferStatus::Empty) = (&self.current_state, &self.next_state) {
            swap(&mut self.current_state, &mut self.next_state);
            swap(&mut self.current_window, &mut self.next_window);
            // swap temp result
            swap(&mut self.current_temp_result, &mut self.next_temp_result);
        }
        Ok(())
    }
}
impl AggBuffer {
    /// # Description
    /// - add a new task into current buffer, only when current state is Empty and current window is None this function is called
    /// - if it's current is empty, then add the window to current window, if current is not empty, then test if this window is belong to a new col, if it's the next col
    /// - return err, and set current state to WaitingToMlp
    /// ---
    /// - don't mess up!
    /// ---
    /// ## ........by sjq
    pub(super) fn add_task(&mut self, window: Rc<OutputWindow>) {
        self.current_window = Some(window);
        self.current_state = BufferStatus::Writing;
    }

    pub(super) fn start_mlp(&mut self) {
        if !matches!(self.next_state, BufferStatus::WaitingToMlp) {
            panic!("start_mlp: current state is not waiting to mlp");
        }
        self.next_state = BufferStatus::Mlp;
    }
    pub(super) fn finished_mlp(&mut self) {
        if !matches!(self.next_state, BufferStatus::Mlp) {
            panic!("finished_mlp: current state is not mlp");
        }
        self.next_state = BufferStatus::Empty;
        // fix bug here, clear the temp result after mlp
        if let Some(ref mut temp_result) = self.next_temp_result {
            temp_result.iter_mut().for_each(|x| {
                x.clear();
            });
        }
    }

    #[allow(dead_code)]
    pub(super) fn get_current_window(&self) -> &Rc<OutputWindow> {
        self
            .current_window
            .as_ref()
            .unwrap_or_else(|| panic!("window should not be None!!"))
    }

    pub(super) fn get_next_window(&self) -> &Rc<OutputWindow> {
        self
            .next_window
            .as_ref()
            .unwrap_or_else(|| panic!("window should not be None!!"))
    }
    pub(super) fn get_current_state(&self) -> &BufferStatus {
        &self.current_state
    }
    pub(super) fn get_next_state(&self) -> &BufferStatus {
        &self.next_state
    }

    pub(super) fn finished_aggregation(&mut self) {
        self.current_state = BufferStatus::WaitingToMlp;
    }

    pub(super) fn get_current_temp_result_mut(&mut self) -> &mut Option<TempAggResult> {
        &mut self.current_temp_result
    }
    pub(super) fn get_next_temp_result(&self) -> &Option<TempAggResult> {
        &self.next_temp_result
    }
}
