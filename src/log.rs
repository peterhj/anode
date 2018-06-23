use dot::*;

use std::cell::{RefCell};
use std::fs::{File};
use std::io::{Write};
use std::path::{PathBuf};

thread_local! {
  static STATIC_GRAPH_LOGGING:  RefCell<Option<GraphLogging>> = RefCell::new(None);
  static DYNAMIC_GRAPH_LOGGING: RefCell<Option<GraphLogging>> = RefCell::new(None);
}

pub fn enable_static_graph_logging() {
  STATIC_GRAPH_LOGGING.with(|maybe_logging| {
    let mut maybe_logging = maybe_logging.borrow_mut();
    *maybe_logging = Some(GraphLogging::new("static".to_owned()));
  });
}

pub fn enable_dynamic_graph_logging() {
  DYNAMIC_GRAPH_LOGGING.with(|maybe_logging| {
    let mut maybe_logging = maybe_logging.borrow_mut();
    *maybe_logging = Some(GraphLogging::new("dynamic".to_owned()));
  });
}

pub fn log_static_graph<F>(f: F) where F: FnOnce(&mut GraphLogging) {
  STATIC_GRAPH_LOGGING.with(|maybe_logging| {
    let mut maybe_logging = maybe_logging.borrow_mut();
    if let Some(logging) = maybe_logging.as_mut() {
      f(logging);
    }
  });
}

pub fn log_dynamic_graph<F>(f: F) where F: FnOnce(&mut GraphLogging) {
  DYNAMIC_GRAPH_LOGGING.with(|maybe_logging| {
    let mut maybe_logging = maybe_logging.borrow_mut();
    if let Some(logging) = maybe_logging.as_mut() {
      f(logging);
    }
  });
}

pub fn clear_dynamic_graph() {
  log_dynamic_graph(|logging| logging.clear());
}

pub fn dump_static_graph() {
  log_static_graph(|logging| logging.dump());
}

pub struct GraphLogging {
  // TODO
  name:     String,
  //nodes:    Vec<(u64, String)>,
  //edges:    Vec<((u64, String), (u64, String))>,
  //aliases:  Vec<((u64, String), (u64, String))>,
  lqueue:   Vec<(u64, String)>,
}

impl Default for GraphLogging {
  fn default() -> Self {
    GraphLogging::new("".to_owned())
  }
}

impl GraphLogging {
  pub fn new(name: String) -> Self {
    GraphLogging{
      name:     name,
      lqueue:   vec![],
    }
  }

  pub fn clear(&mut self) {
    // TODO
    self.lqueue.clear();
  }

  pub fn insert_node(&mut self, key: (u64, String)) {
    println!("DEBUG: GraphLogging ({}): insert node: {:?}", self.name, key);
    for lkey in self.lqueue.clone().into_iter() {
      self.insert_edge(lkey, key.clone());
    }
    self.lqueue.clear();
  }

  pub fn insert_edge(&mut self, lkey: (u64, String), rkey: (u64, String)) {
    println!("DEBUG: GraphLogging ({}): insert edge: {:?} -> {:?}", self.name, lkey, rkey);
  }

  pub fn insert_alias_edge(&mut self, lkey: (u64, String), rkey: (u64, String)) {
    println!("DEBUG: GraphLogging ({}): insert alias edge: {:?} -> {:?}", self.name, lkey, rkey);
  }

  pub fn push_pred(&mut self, lkey: (u64, String)) {
    self.lqueue.push(lkey);
  }

  pub fn dump(&mut self) {
  }
}
