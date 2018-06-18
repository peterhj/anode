extern crate anode;

use anode::proc::*;

fn main() {
  let mut group = DistProcGroup::default();
  for node in group {
    node.spawn(|proc| {
      println!("DEBUG: hello world: {}", proc.rank());
    }).unwrap().join();
  }
}
