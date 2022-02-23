use std::error::Error;
use std::fs::File;
use std::io::Read;

use itertools::Itertools;

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
        // contnents contains all the file in 0 1 0 1\n 1 0 1 1\n 1 1 0 0\n format
        let mut contents = String::new();

        file.read_to_string(&mut contents)?;
        let mut features = Vec::new();

        for line in contents.lines() {
            // each line is a node in 0 1 0 1 format
            let line_vec: Vec<_> = line
                .split_whitespace()
                .map(|x| x.parse::<usize>())
                .try_collect()?;

            // convert the line to csc format
            let mut csc_line = Vec::new();
            // build the csc format: 0 1 0 1 => 1,3
            for (i, &item) in line_vec.iter().enumerate() {
                if item != 0 {
                    csc_line.push(i);
                }
            }

            features.push(csc_line);
        }
        // build start addr from the node features

        let mut start_addrs: Vec<u64> = vec![];
        let last = features.iter().fold(0, |acc, x| {
            start_addrs.push(acc);
            acc + (x.len() * 4) as u64
        });
        start_addrs.push(last);

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
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
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
