use std::{fs::File, io::Read};

// build the structure of the graph
pub struct Graph {
    csc: Vec<Vec<usize>>,
    // the feature size
    feature_size: usize,
}
impl From<&str> for Graph {

    /// read the graph from the file
    /// # Arguments
    /// * `file_name` - the path of the file
    /// # Return
    /// * `Graph` - the graph
    /// 
    /// the file format is:
    /// f feature_size
    /// 0 1 2
    /// 1 2 0
    /// 2 0 1
    /// end
    /// 
    /// the first line is the feature size
    /// the following lines are the edges
    /// the last line is end or END
    /// # Examples
    /// ```
    /// use gcn_agg::graph::Graph;
    /// use std::{fs::File, io::{Read,Write}};
    /// let file_name="test_data/graph.txt";
    /// // write the graph to the file
    /// let data="f 3\n0 1 2\n1 2 0\n2 0 1\nend\n";
    /// let mut f = File::create(file_name).expect("file not found");
    /// f.write_all(data.as_bytes())
    ///     .expect("something went wrong writing the file");
    /// // read the graph from the file
    /// let graph = Graph::from("test_data/graph.txt");
    /// assert_eq!(graph.get_feature_size(), 3);
    /// assert_eq!(graph.get_csc()[0][0], 0);
    /// assert_eq!(graph.get_csc()[0][1], 1);
    /// assert_eq!(graph.get_csc()[0][2], 2);
    /// assert_eq!(graph.get_csc()[1][0], 1);
    /// assert_eq!(graph.get_csc()[1][1], 2);
    /// assert_eq!(graph.get_csc()[1][2], 0);
    /// assert_eq!(graph.get_csc()[2][0], 2);
    /// assert_eq!(graph.get_csc()[2][1], 0);
    /// assert_eq!(graph.get_csc()[2][2], 1);
    /// // delete the file
    /// std::fs::remove_file(file_name).expect("failed to delete the file");
    /// ```
    /// 
    fn from(file_name: &str) -> Self {
        let mut f = File::open(file_name).expect("file not found");
        let mut contents = String::new();
        f.read_to_string(&mut contents)
            .expect("something went wrong reading the file");
        let mut lines = contents.lines();
        // the first line should be like "f {feature_size}"
        let first_line = lines.next().unwrap();
        let mut iter = first_line.split_whitespace();
        let f_char = iter.next();
        match f_char {
            Some("f") => {}
            _ => panic!("the first line should be like \"f feature_size\""),
        }
        let feature_size = iter.next().unwrap().parse::<usize>().unwrap();

        // the remaining lines should be like list of edges in csc format
        // from next line to the second last row, will contain the row index of the edges
        let mut csc = Vec::new();
        for line in lines {
            // test if the line start with END or end
            if line.starts_with("END") || line.starts_with("end") {
                break;
            }
            // break the line into array of usize
            let iter = line.split_whitespace();
            let mut row = Vec::new();
            for i in iter {
                row.push(i.parse::<usize>().unwrap());
            }
            // add the row to the csc format
            csc.push(row);
        }

        Graph { csc, feature_size }
    }
}

impl Graph {
    pub fn get_feature_size(&self) -> usize {
        self.feature_size
    }
    pub fn get_csc(&self) -> &Vec<Vec<usize>> {
        &self.csc
    }
}

// create a mod for testing
#[cfg(test)]
mod graph_test{
    use std::io::Write;

    use super::*;
    #[test]
    fn test_from_str() {
        let file_name="test_data/graph.txt";
        // write the graph to the file
        let data="f 3\n0 1 2\n1 2 0\n2 0 1\nend\n";
        let mut f = File::create(file_name).expect("file not found");
        f.write_all(data.as_bytes())
            .expect("something went wrong writing the file");
        // read the graph from the file

        let graph = Graph::from("test_data/graph.txt");
        assert_eq!(graph.get_feature_size(), 3);
        assert_eq!(graph.get_csc()[0][0], 0);
        assert_eq!(graph.get_csc()[0][1], 1);
        assert_eq!(graph.get_csc()[0][2], 2);
        assert_eq!(graph.get_csc()[1][0], 1);
        assert_eq!(graph.get_csc()[1][1], 2);
        assert_eq!(graph.get_csc()[1][2], 0);
        assert_eq!(graph.get_csc()[2][0], 2);
        assert_eq!(graph.get_csc()[2][1], 0);
        assert_eq!(graph.get_csc()[2][2], 1);
        // delete the file
        std::fs::remove_file(file_name).expect("failed to delete the file");
    }
}