use std::error::Error;
use std::fs::File;
use std::io::Read;

#[derive(Debug)]
pub struct NodeFeatures {
    pub features: Vec<Vec<usize>>,
    pub start_addrs: Vec<u64>,
}

impl NodeFeatures {
    ///
    /// # Arguments
    /// * file_name - The name of the file to read
    /// # return
    /// * NodeFeatures - The features of the nodes
    ///
    /// # Description
    /// Reads a file containing a list of features for each node.
    /// note each line of file contains a dense format of a node feature
    /// example file format:
    /// 0 1 0
    /// 1 0 1
    /// 1 1 0
    ///
    /// the result node feature will be stored as csr format
    /// # example
    /// ```ignore
    /// use std::fs::File;
    /// use std::io::{Read,Write};
    /// use gcn_agg::node_features::NodeFeatures;
    ///         let data = "0 0 1 0 1 0\n1 0 0 1 1 1\n1 1 0 0 0 1\n";
    /// let file_name = "test_data/node_features.txt";
    /// // write the data to the file
    /// let mut file = File::create(file_name)?;
    /// file.write_all(data.as_bytes())?;
    ///
    /// let node_features = NodeFeatures::from(file_name);
    /// assert_eq!(node_features.len(), 3);
    /// assert_eq!(node_features.get_features(0).len(), 2);
    /// assert_eq!(node_features.get_features(1).len(), 4);
    /// assert_eq!(node_features.get_features(2).len(), 3);
    /// assert_eq!(node_features.get_features(0)[0], 2);
    /// assert_eq!(node_features.get_features(0)[1], 4);
    ///
    /// assert_eq!(node_features.get_features(1)[0], 0);
    /// assert_eq!(node_features.get_features(1)[1], 3);
    /// assert_eq!(node_features.get_features(1)[2], 4);
    /// assert_eq!(node_features.get_features(1)[3], 5);
    ///
    /// assert_eq!(node_features.get_features(2)[0], 0);
    /// assert_eq!(node_features.get_features(2)[1], 1);
    /// assert_eq!(node_features.get_features(2)[2], 5);
    ///
    /// // delete the file
    /// std::fs::remove_file(file_name)?;
    ///
    /// ```
    pub fn new(file_name: &str) -> Result<Self, Box<dyn Error>> {
        // the file contains adjacency matrix
        // each line is a node
        let mut file = File::open(file_name)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let mut features = Vec::new();

        for line in contents.lines() {
            let mut line_vec = Vec::new();
            for num in line.split_whitespace() {
                line_vec.push(num.parse::<usize>()?);
            }
            // convert the line to csc format
            let mut csc_line = Vec::new();
            for i in 0..line_vec.len() {
                if line_vec[i] != 0 {
                    csc_line.push(i);
                }
            }
            features.push(csc_line);
        }
        // build start addr from the node features

        let mut start_addrs = Vec::new();
        start_addrs.push(0u64);
        for i in 1..=features.len() {
            start_addrs.push(start_addrs[i - 1] + features[i - 1].len() as u64 * 4);
        }
        Ok(NodeFeatures {
            features,
            start_addrs,
        })
    }
}
impl NodeFeatures {
    pub fn get_features(&self, node_id: usize) -> &Vec<usize> {
        &self.features[node_id]
    }
    pub fn len(&self) -> usize {
        self.features.len()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn test_node_features() -> Result<(), Box<dyn Error>> {
        let data = "0 0 1 0 1 0\n1 0 0 1 1 1\n1 1 0 0 0 1\n";
        let file_name = "test_data/node_features.txt";
        // write the data to the file
        let mut file = File::create(file_name)?;
        file.write_all(data.as_bytes())?;

        let node_features = NodeFeatures::new(file_name)?;
        assert_eq!(node_features.len(), 3);
        assert_eq!(node_features.get_features(0).len(), 2);
        assert_eq!(node_features.get_features(1).len(), 4);
        assert_eq!(node_features.get_features(2).len(), 3);
        assert_eq!(node_features.get_features(0)[0], 2);
        assert_eq!(node_features.get_features(0)[1], 4);

        assert_eq!(node_features.get_features(1)[0], 0);
        assert_eq!(node_features.get_features(1)[1], 3);
        assert_eq!(node_features.get_features(1)[2], 4);
        assert_eq!(node_features.get_features(1)[3], 5);

        assert_eq!(node_features.get_features(2)[0], 0);
        assert_eq!(node_features.get_features(2)[1], 1);
        assert_eq!(node_features.get_features(2)[2], 5);

        // delete the file
        std::fs::remove_file(file_name)?;
        Ok(())
    }
}
