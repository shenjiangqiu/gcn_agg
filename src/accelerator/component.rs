pub trait Component {
    fn cycle(&mut self)->Result<(), Box<dyn std::error::Error>>;
}
