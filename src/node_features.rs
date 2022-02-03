use std::fs::File;
use std::io::Read;

pub struct NodeFeatures {
    features: Vec<Vec<usize>>,
}

impl From<&str> for NodeFeatures {
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
    /// ```
    /// use std::fs::File;
    /// use std::io::{Read,Write};
    /// use gcn_agg::node_features::NodeFeatures;
    ///         let data = "0 0 1 0 1 0\n1 0 0 1 1 1\n1 1 0 0 0 1\n";
    /// let file_name = "test_data/node_features.txt";
    /// // write the data to the file
    /// let mut file = File::create(file_name).unwrap();
    /// file.write_all(data.as_bytes()).unwrap();
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
    /// std::fs::remove_file(file_name).unwrap();
    ///
    /// ```
    fn from(file_name: &str) -> Self {
        // the file contains adjacency matrix
        // each line is a node
        let mut file = File::open(file_name).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();
        let mut features = Vec::new();

        for line in contents.lines() {
            let mut line_vec = Vec::new();
            for num in line.split_whitespace() {
                line_vec.push(num.parse::<usize>().unwrap());
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

        NodeFeatures { features }
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
    use std::io::{Read, Write};

    #[test]
    fn test_node_features() {
        let data = "0 0 1 0 1 0\n1 0 0 1 1 1\n1 1 0 0 0 1\n";
        let file_name = "test_data/node_features.txt";
        // write the data to the file
        let mut file = File::create(file_name).unwrap();
        file.write_all(data.as_bytes()).unwrap();

        let node_features = NodeFeatures::from(file_name);
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
        std::fs::remove_file(file_name).unwrap();
    }
}
