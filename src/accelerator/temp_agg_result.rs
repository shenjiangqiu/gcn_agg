use std::ops::{Deref, DerefMut};

#[derive(Debug)]
pub struct TempAggResult {
    inner: Vec<Vec<usize>>,
}

impl TempAggResult {
    pub fn new(total_nodes: usize) -> Self {
        TempAggResult {
            inner: vec![vec![]; total_nodes],
        }
    }
    pub fn get_line(&self, row_id: usize) -> &Vec<usize> {
        &self.inner[row_id]
    }
    pub fn get_lines_range(&self, start: usize, end: usize) -> &[Vec<usize>] {
        &self.inner[start..end]
    }
}

impl Deref for TempAggResult {
    type Target = Vec<Vec<usize>>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for TempAggResult {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
