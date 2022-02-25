use std::{collections::BTreeSet, error::Error, fs::File, io::Read, vec};

// build the structure of the graph
#[derive(Debug)]
pub struct Graph {
    csc: Vec<BTreeSet<usize>>,
    csr: Option<Vec<BTreeSet<usize>>>,
    // the feature size
    feature_size: usize,
    // the number of nodes
    pub total_nodes: usize,
}
impl Graph {
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
    ///     use gcn_agg::graph::Graph;
    ///    use std::fs::File;
    ///    use std::io::Write;
    ///    let file_name = "test_data/graph_csr.txt";
    ///    // write the graph to the file
    ///    let data = "f 3\n0 1\n1 2\n2 0\nend\n";
    ///    let mut f = File::create(file_name).expect("file not found");
    ///    f.write_all(data.as_bytes())
    ///        .expect("something went wrong writing the file");
    ///    // read the graph from the file
    ///
    ///    let mut graph = Graph::new(file_name).unwrap();
    ///    
    ///
    ///    if let Some(csr) = graph.get_csr() {
    ///        assert!(csr[0].contains(&0));
    ///        assert!(csr[0].contains(&2));
    ///        assert!(csr[1].contains(&0));
    ///        assert!(csr[1].contains(&1));
    ///        assert!(csr[2].contains(&1));
    ///        assert!(csr[2].contains(&2));
    ///        assert_eq!(csr.len(), 3);
    ///    } else {
    ///        panic!("csr is not generated");
    ///    }
    ///    
    /// ```
    ///
    pub fn new(file_name: &str) -> Result<Self, Box<dyn Error>> {
        let mut f = File::open(file_name).expect("file not found");
        let mut contents = String::new();
        f.read_to_string(&mut contents)
            .expect("something went wrong reading the file");
        let mut lines = contents.lines();
        // the first line should be like "f {feature_size}"
        let first_line = lines.next().ok_or("no new line")?;
        let mut iter = first_line.split_whitespace();
        let f_char = iter.next();
        match f_char {
            Some("f") => {}
            _ => panic!("the first line should be like \"f feature_size\""),
        }
        let feature_size = iter.next().ok_or("no new line")?.parse::<usize>()?;

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
            let mut row = BTreeSet::new();
            for i in iter {
                row.insert(i.parse::<usize>()?);
            }
            // add the row to the csc format
            csc.push(row);
        }
        let total_nodes = csc.len();
        let mut graph = Graph {
            csc,
            csr: None,
            feature_size,
            total_nodes,
        };
        graph.generate_csr();
        Ok(graph)
    }
}

impl Graph {
    pub fn get_feature_size(&self) -> usize {
        self.feature_size
    }
    pub fn get_csc(&self) -> &Vec<BTreeSet<usize>> {
        &self.csc
    }
    pub fn get_csr(&self) -> &Option<Vec<BTreeSet<usize>>> {
        &self.csr
    }
    /// # Description
    /// test if a row is empty from col start to col end, for index i
    pub fn is_row_range_empty(
        &self,
        i: usize,
        start: usize,
        end: usize,
    ) -> Result<bool, Box<dyn Error>> {
        match self
            .csr
            .as_ref()
            .ok_or("no csr")?
            .get(i)
            .ok_or(format!("no that line: {}", i))?
            .range(start..end)
            .next()
        {
            Some(_) => Ok(false),
            None => Ok(true),
        }
    }

    fn generate_csr(&mut self) {
        // build csr from csc
        let mut csr = vec![BTreeSet::<usize>::new(); self.csc.len()];
        for (index, csc_row) in self.csc.iter().enumerate() {
            for j in csc_row {
                csr[*j].insert(index);
            }
        }

        self.csr = Some(csr);
    }
    pub fn get_num_node(&self) -> usize {
        self.csc.len()
    }
}

// create a mod for testing
#[cfg(test)]
mod graph_test {
    use std::io::Write;

    use super::*;
    #[test]
    fn test_from_str() -> Result<(), Box<dyn Error>> {
        let file_name = "test_data/graph_graph_str.txt";
        // write the graph to the file
        let data = "f 3\n0 1 2\n1 2 0\n2 0 1\nend\n";
        let mut f = File::create(file_name).expect("file not found");
        f.write_all(data.as_bytes())
            .expect("something went wrong writing the file");
        // read the graph from the file

        let graph = Graph::new(file_name)?;
        assert_eq!(graph.get_feature_size(), 3);
        assert!(graph.get_csc()[0].contains(&0));
        assert!(graph.get_csc()[0].contains(&1));
        assert!(graph.get_csc()[0].contains(&2));

        assert!(graph.get_csc()[1].contains(&0));
        assert!(graph.get_csc()[1].contains(&1));
        assert!(graph.get_csc()[1].contains(&2));
        assert!(graph.get_csc()[2].contains(&0));
        assert!(graph.get_csc()[2].contains(&1));
        assert!(graph.get_csc()[2].contains(&2));
        // delete the file
        std::fs::remove_file(file_name).expect("failed to delete the file");
        Ok(())
    }
    #[test]
    fn test_csr() -> Result<(), Box<dyn Error>> {
        let file_name = "test_data/graph_csr.txt";
        // write the graph to the file
        let data = "f 3\n0 1\n1 2\n2 0\nend\n";
        let mut f = File::create(file_name).expect("file not found");
        f.write_all(data.as_bytes())
            .expect("something went wrong writing the file");
        // read the graph from the file

        let graph = Graph::new(file_name)?;

        if let Some(csr) = graph.get_csr() {
            assert!(csr[0].contains(&0));
            assert!(csr[0].contains(&2));
            assert!(csr[1].contains(&0));
            assert!(csr[1].contains(&1));
            assert!(csr[2].contains(&1));
            assert!(csr[2].contains(&2));
            assert_eq!(csr.len(), 3);
        } else {
            panic!("csr is not generated");
        }
        Ok(())
    }
}
