/// # Description
/// - struct Req define a window
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WindowId {
    pub output_id: usize,
    pub input_id: usize,
    pub layer_id: usize,
}
impl WindowId {
    #[allow(unused)]
    pub fn new(output_id: usize, input_id: usize, layer_id: usize) -> Self {
        WindowId {
            output_id,
            input_id,
            layer_id,
        }
    }
}
