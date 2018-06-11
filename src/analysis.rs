/*
Copyright 2017-2018 Peter Jin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

use ::{RWVar};

use typemap::{Key};

use std::cell::{RefCell};
use std::collections::{HashSet};
use std::iter::{FromIterator};
use std::rc::{Rc};

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

#[derive(Clone)]
pub struct DominanceAnalysis {
  //inner:    Rc<RefCell<DominanceAnalysisInner>>,
}

impl Key for DominanceAnalysis {
  type Value = Self;
}
