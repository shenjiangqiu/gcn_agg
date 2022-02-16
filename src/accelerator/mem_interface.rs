use std::collections::{HashMap, HashSet, VecDeque};

use ramulator_wrapper::RamulatorWrapper;

use super::{component::Component, window_id::WindowId};

#[derive(Debug)]
pub struct MemWindowIdust {
    addr_vec: Vec<u64>,
    id: WindowId,
    is_write: bool,
}

impl MemWindowIdust {
    fn new(addr_vec: Vec<u64>, id: WindowId, is_write: bool) -> Self {
        MemWindowIdust {
            addr_vec,
            id,
            is_write,
        }
    }
}
/// # Description
/// * the MemInterface is the interface between accelerator and memory
/// # Fields
/// * `mem`: the ramulator wrapper
/// * `send_queue`: the queue of requests to be sent to memory
/// * `recv_queue`: the queue of requests to be received from memory
/// * `current_waiting_request`: the current request id on flight,key is the request id, value is the request address
/// * `current_waiting_mem_request`: the current request id on flight,key is the request addr, value is the request id that contains this addr
#[derive(Debug)]
pub struct MemInterface {
    mem: RamulatorWrapper,
    send_queue: VecDeque<MemWindowIdust>,
    send_size: usize,
    recv_queue: VecDeque<WindowId>,
    recv_size: usize,
    current_waiting_request: HashMap<WindowId, HashSet<u64>>,
    current_waiting_mem_request: HashMap<u64, HashSet<WindowId>>,
}

impl Component for MemInterface {
    /// # Description
    /// this is the cycle function that do two things:
    /// 1. send requests to memory
    ///  when the send queue is not empty, send the first request in the queue,
    /// and update the current_waiting_request and current_waiting_mem_request
    ///
    /// 2. receive responses from memory
    /// when the recv queue is not full and self.mem is ret_available, receive the first response from memory,
    /// note that: need to delete the request from current_waiting_request and current_waiting_mem_request
    ///
    fn cycle(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(req) = self.send_queue.front_mut() {
            while let Some(addr) = req.addr_vec.pop() {
                if self.mem.available(addr, req.is_write) {
                    self.mem.send(addr, req.is_write);
                    self.current_waiting_request
                        .entry(req.id.clone())
                        .or_insert(HashSet::new())
                        .insert(addr);
                    self.current_waiting_mem_request
                        .entry(addr)
                        .or_insert(HashSet::new())
                        .insert(req.id.clone());
                } else {
                    req.addr_vec.push(addr);
                    break;
                }
            }
            if req.addr_vec.len() == 0 {
                self.send_queue.pop_front();
            }
        }

        while self.mem.ret_available() && self.recv_queue.len() < self.recv_size {
            let addr = self.mem.get();
            let id_list = self
                .current_waiting_mem_request
                .remove(&addr)
                .expect(format!("no request for addr {}", addr).as_str());
            for id in id_list {
                let req = self
                    .current_waiting_request
                    .get_mut(&id)
                    .expect(format!("no request for id {:?}", id).as_str());
                req.remove(&addr);
                if req.len() == 0 {
                    self.current_waiting_request.remove(&id);
                    self.recv_queue.push_back(id);
                }
            }
        }
        self.mem.cycle();
        Ok(())
    }
}

impl MemInterface {
    pub fn new(send_size: usize, recv_size: usize) -> Self {
        MemInterface {
            mem: RamulatorWrapper::new(),
            send_queue: VecDeque::new(),
            send_size,
            recv_queue: VecDeque::new(),
            recv_size,
            current_waiting_request: HashMap::new(),
            current_waiting_mem_request: HashMap::new(),
        }
    }
    /// # Description
    /// * is the interface ready to receive a request
    pub fn available(&self) -> bool {
        self.send_queue.len() < self.send_size
    }
    /// # Description
    /// * is the interface ready to send a response
    /// - the reason I delete this function: In rust, use let Some is better!
    // pub fn ret_ready(&self) -> bool {
    //     self.recv_queue.len() > 0
    // }

    /// # Description
    /// * send a request to memory
    pub fn send(&mut self, id_: WindowId, addr_vec: Vec<u64>, is_write: bool) {
        self.send_queue
            .push_back(MemWindowIdust::new(addr_vec, id_, is_write));
    }
    /// # Description
    /// * receive a response from memory and keep the request ***still in mem(not pop it)***
    #[allow(dead_code)]
    pub fn receive(&self) -> Option<&WindowId> {
        self.recv_queue.front()
    }
    /// # Description
    /// * receive a response from memory and pop that request
    #[allow(dead_code)]
    pub fn receive_pop(&mut self) -> Option<WindowId> {
        self.recv_queue.pop_front()
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::accelerator::window_id::WindowId;

    #[test]
    fn test_mem_interface() -> Result<(), Box<dyn std::error::Error>> {
        let mut mem_interface = super::MemInterface::new(1, 1);
        assert_eq!(mem_interface.available(), true);
        assert_eq!(mem_interface.receive().is_some(), false);
        mem_interface.send(WindowId::new(1, 1, 1), vec![0], false);
        assert_eq!(mem_interface.available(), false);
        assert_eq!(mem_interface.receive().is_some(), false);

        while !mem_interface.receive().is_some() {
            mem_interface.cycle()?;
        }

        assert_eq!(mem_interface.available(), true);
        assert_eq!(mem_interface.receive().is_some(), true);

        let result = mem_interface.receive().expect("no response");
        assert_eq!(*result, WindowId::new(1, 1, 1));
        assert_eq!(mem_interface.current_waiting_mem_request.is_empty(), true);
        assert_eq!(mem_interface.current_waiting_request.is_empty(), true);
        Ok(())
    }
}
