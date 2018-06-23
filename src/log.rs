use dot::*;

use std::cell::{RefCell};
use std::collections::{HashMap};
use std::fmt::{Debug};
use std::fs::{File};
use std::hash::{Hash};
use std::io::{Write};
use std::path::{PathBuf};

thread_local! {
  static STATIC_GRAPH_LOGGING:  RefCell<Option<GraphLogging<(u64, String)>>> = RefCell::new(None);
  static DYNAMIC_GRAPH_LOGGING: RefCell<Option<GraphLogging<(u64, String)>>> = RefCell::new(None);
}

pub fn enable_static_graph_logging() {
  STATIC_GRAPH_LOGGING.with(|maybe_logging| {
    let mut maybe_logging = maybe_logging.borrow_mut();
    *maybe_logging = Some(GraphLogging::new("static".to_owned(), false));
  });
}

pub fn enable_dynamic_graph_logging() {
  DYNAMIC_GRAPH_LOGGING.with(|maybe_logging| {
    let mut maybe_logging = maybe_logging.borrow_mut();
    *maybe_logging = Some(GraphLogging::new("dynamic".to_owned(), true));
  });
}

pub fn log_static_graph<F>(f: F) where F: FnOnce(&mut GraphLogging<(u64, String)>) {
  STATIC_GRAPH_LOGGING.with(|maybe_logging| {
    let mut maybe_logging = maybe_logging.borrow_mut();
    if let Some(logging) = maybe_logging.as_mut() {
      f(logging);
    }
  });
}

pub fn log_dynamic_graph<F>(f: F) where F: FnOnce(&mut GraphLogging<(u64, String)>) {
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

pub struct GraphLogging<K> {
  // TODO
  name:     String,
  unique:   bool,
  node_ct:  HashMap<K, usize>,
  //nodes:    Vec<(u64, String)>,
  //edges:    Vec<((u64, String), (u64, String))>,
  //aliases:  Vec<((u64, String), (u64, String))>,
  lqueue:   Vec<K>,
}

/*impl Default for GraphLogging {
  fn default() -> Self {
    GraphLogging::new("".to_owned())
  }
}*/

impl<K: Clone + PartialEq + Eq + Hash + Debug> GraphLogging<K> {
  pub fn new(name: String, unique: bool) -> Self {
    GraphLogging{
      name:     name,
      unique:   unique,
      node_ct:  HashMap::new(),
      lqueue:   vec![],
    }
  }

  pub fn clear(&mut self) {
    // TODO
    self.lqueue.clear();
  }

  pub fn insert_node(&mut self, key: K) {
    match self.unique {
      false => {
        println!("DEBUG: GraphLogging ({}): insert node: {:?}", self.name, key);
      }
      true  => {
        let count = if self.node_ct.contains_key(&key) {
          *self.node_ct.get(&key).unwrap() + 1
        } else {
          0
        };
        self.node_ct.insert(key.clone(), count);
        println!("DEBUG: GraphLogging ({}): insert node: {:?} ctr: {}", self.name, key, count);
      }
    }
    for lkey in self.lqueue.clone().into_iter() {
      self.insert_edge(lkey, key.clone());
    }
    self.lqueue.clear();
  }

  pub fn insert_edge(&mut self, lkey: K, rkey: K) {
    println!("DEBUG: GraphLogging ({}): insert edge: {:?} -> {:?}", self.name, lkey, rkey);
  }

  pub fn insert_alias_edge(&mut self, lkey: K, rkey: K) {
    println!("DEBUG: GraphLogging ({}): insert alias edge: {:?} -> {:?}", self.name, lkey, rkey);
  }

  pub fn push_pred(&mut self, lkey: K) {
    self.lqueue.push(lkey);
  }

  pub fn dump(&mut self) {
  }
}
