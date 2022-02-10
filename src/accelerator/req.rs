

/// # Description
/// - struct Req define a window
#[derive(Debug, Clone, PartialEq, Eq,Hash)]
pub struct WindowId {
    pub col_id: usize,
    pub row_id: usize,
    pub layer_id: usize,
}
impl WindowId {
    pub fn new(col_id: usize, row_id: usize, layer_id: usize) -> Self {
        WindowId {
            col_id,
            row_id,
            layer_id,
        }
    }
}
