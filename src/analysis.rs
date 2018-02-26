use ::{RWVar};

use typemap::{Key};

use std::cell::{RefCell};
use std::collections::{HashSet};
use std::iter::{FromIterator};
use std::rc::{Rc, Weak};

#[derive(Clone)]
pub struct LivenessAnalysis {
  inner:    Rc<RefCell<LivenessAnalysisInner>>,
}

impl Key for LivenessAnalysis {
  type Value = Self;
}

pub struct LivenessAnalysisInner {
  ns:   Vec<LivenessAnalysis>,
  use_: HashSet<RWVar>,
  def:  HashSet<RWVar>,
  in_:  HashSet<RWVar>,
  out:  HashSet<RWVar>,
  pin:  HashSet<RWVar>,
  pout: HashSet<RWVar>,
}

impl LivenessAnalysisInner {
  pub fn _iterate_pass1(&mut self) {
    self.pin.clone_from(&self.in_);
    self.pout.clone_from(&self.out);
  }

  pub fn _iterate_pass2(&mut self) {
    let out_minus_def = HashSet::from_iter(self.out.difference(&self.def).map(|&key| key));
    self.in_ = HashSet::from_iter(self.use_.union(&out_minus_def).map(|&key| key));
  }

  pub fn _iterate_pass3(&mut self) {
    for n in self.ns.iter() {
      let mut pred = n.inner.borrow_mut();
      pred.out.extend(self.in_.iter());
    }
  }
}
