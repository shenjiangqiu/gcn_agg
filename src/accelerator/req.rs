#[derive(Debug, Clone, PartialEq, Eq,Hash)]
pub struct Req {
    pub col_id: usize,
    pub row_id: usize,
    pub layer_id: usize,
}
impl Req {
    pub fn new(col_id: usize, row_id: usize, layer_id: usize) -> Self {
        Req {
            col_id,
            row_id,
            layer_id,
        }
    }
}
