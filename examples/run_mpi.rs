extern crate anode;

#[cfg(feature = "mpi")] use anode::proc_dist::*;

#[cfg(not(feature = "mpi"))]
fn main() {
}

#[cfg(feature = "mpi")]
fn main() {
  let mut group = DistProcGroup::default();
  for node in group {
    node.spawn(|proc| {
      println!("DEBUG: hello world: {}", proc.rank());
    }).unwrap().join();
  }
}
