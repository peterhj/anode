/*
Copyright 2017 the anode authors

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

#![feature(conservative_impl_trait)]
#![feature(fn_traits)]
#![feature(get_type_id)]
#![feature(slice_patterns)]
#![feature(specialization)]
//#![feature(trait_alias)]
#![feature(unboxed_closures)]

extern crate arithmetic;
#[cfg(feature = "gpu")] extern crate cuda;
#[cfg(feature = "gpu")] extern crate cuda_blas;
#[cfg(feature = "gpu")] extern crate cuda_dnn;
//extern crate float;
#[cfg(feature = "gpu")] extern crate gpudevicemem;
//#[macro_use] extern crate lazy_static;
extern crate memarray;
extern crate rand;
extern crate rng;
extern crate typemap;

use analysis::{LivenessAnalysis};
use ops::{MemIoReader, MemIoWriter, OnesSrcOp, OnesSrcOpMaybeExt, SumJoinOp, SumJoinOpMaybeExt, SumJoinOpExt};
#[cfg(feature = "gpu")] use ops_gpu::{GPUMuxFun};

#[cfg(feature = "gpu")] use gpudevicemem::{GPUDeviceId};
use typemap::{CloneMap, TypeMap};

use std::any::{Any};
use std::cell::{Cell, RefCell, Ref, RefMut};
use std::collections::{HashMap, HashSet};
//use std::collections::hash_map::{Entry};
use std::rc::{Rc};

pub mod analysis;
pub mod config;
pub mod context;
pub mod ffi;
pub mod ops;
#[cfg(feature = "gpu")] pub mod ops_gpu;

thread_local!(static UID_COUNTER: Cell<u64> = Cell::new(0));

pub fn gen_thread_local_uid() -> u64 {
  UID_COUNTER.with(|counter| {
    let prev_count = counter.get();
    counter.set(prev_count + 1);
    let next_count = counter.get();
    assert!(next_count != 0);
    next_count
  })
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Txn(u64);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Epoch(u64);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RVar(u64);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
pub struct RWVar(RVar);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeRef(u64);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ValRef(u64);

pub fn txn() -> Txn {
  Txn::default()
}

impl Default for Txn {
  fn default() -> Self {
    Txn(gen_thread_local_uid())
  }
}

pub fn epoch() -> Epoch {
  Epoch::default()
}

impl Default for Epoch {
  fn default() -> Self {
    Epoch(gen_thread_local_uid())
  }
}

impl Default for RVar {
  fn default() -> Self {
    RVar(gen_thread_local_uid())
  }
}

impl Default for NodeRef {
  fn default() -> Self {
    NodeRef(gen_thread_local_uid())
  }
}

impl Default for ValRef {
  fn default() -> Self {
    ValRef(gen_thread_local_uid())
  }
}

pub struct WalkStackEntry {
  epoch:        Epoch,
  push_degree:  usize,
  pop_degree:   usize,
  // TODO: cycle detection.
  //succ_set:     HashSet<NodeRef>,
}

#[derive(Default)]
pub struct WalkStack {
  entries:  RefCell<Vec<WalkStackEntry>>,
}

impl Walk for WalkStack {
  fn outdegree(&self) -> usize {
    self.outdegree(1)
  }
}

impl WalkStack {
  pub fn push(&self, epoch: Epoch) -> bool {
    let mut entries = self.entries.borrow_mut();
    if entries.is_empty() || entries.last().unwrap().epoch < epoch {
      entries.push(WalkStackEntry{
        epoch:          epoch,
        push_degree:    1,
        pop_degree:     0,
      });
      true
    } else if entries.last().unwrap().epoch > epoch {
      panic!();
    } else if entries.last().unwrap().epoch == epoch {
      entries.last_mut().unwrap().push_degree += 1;
      false
    } else {
      unreachable!();
    }
  }

  pub fn pop(&self, epoch: Epoch) -> bool {
    let mut entries = self.entries.borrow_mut();
    assert!(!entries.is_empty());
    assert_eq!(entries.last().unwrap().epoch, epoch);
    entries.last_mut().unwrap().pop_degree += 1;
    if entries.last().unwrap().push_degree == entries.last().unwrap().pop_degree {
      entries.pop();
      true
    } else {
      false
    }
  }

  pub fn outdegree(&self, depth: usize) -> usize {
    let entries = self.entries.borrow();
    let num_entries = entries.len();
    assert!(num_entries >= depth);
    entries[num_entries - depth].push_degree
  }
}

pub trait Walk {
  fn outdegree(&self) -> usize;
}

pub trait IOVal {
  //fn _load(&self, txn: Txn, writer: &mut Any);
  //fn _store(&self, txn: Txn, reader: &mut Any);
  fn _deserialize(&self, txn: Txn, writer: &mut FnMut(WriteCap, &mut Any));
  fn _serialize(&self, txn: Txn, reader: &mut FnMut(&Any));
}

pub trait Analyses {
  fn _liveness(&self) -> Option<LivenessAnalysis>;
}

pub trait ANode {
  fn _walk(&self) -> &Walk;
  fn _io(&self) -> &IOVal;
  fn _analyses(&self) -> &Analyses { unimplemented!(); }
  //fn _pred_fwd(&self, pred_buf: &mut Vec<Rc<ANode>>);
  //fn _pred_rev(&self, pred_buf: &mut Vec<Rc<ANode>>);

  fn _push(&self, stop_txn: Option<Txn>, epoch: Epoch, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode));
  fn _pop(&self, stop_txn: Option<Txn>, epoch: Epoch, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode));

  fn _txn(&self) -> Option<Txn>;
  //fn _persist(&self, txn: Txn);
  //fn _prepare(&self, txn: Txn);
  //fn _cleanup(&self, txn: Txn);
  fn _apply(&self, txn: Txn, rvar: RVar, xvar: RWVar);
  fn _eval_recursive(&self, txn: Txn, rvar: RVar, xvar: RWVar);

  /*fn deserialize_forward(&self, txn: Txn, writer: &mut FnMut(WriteMode, &mut Any));
  fn deserialize_reverse(&self, txn: Txn, writer: &mut FnMut(WriteMode, &mut Any));
  fn serialize_forward(&self, txn: Txn, reader: &mut FnMut(&mut Any));
  fn serialize_reverse(&self, txn: Txn, reader: &mut FnMut(&mut Any));*/

  //fn eval(&self, txn: Txn);
  //fn cleanup(&self, txn: Txn);
}

pub trait AOp<V>: ANode {
  fn _make_value(&self) -> RWVal<V>;
  fn _value(&self) -> &RWVal<V>;

  fn _make_tangent(&self) -> Val<V> { unimplemented!(); }
  fn tangent(&self) -> Val<V>;

  fn _pop_adjoint(&self, epoch: Epoch, /*filter: &Fn(&ANode) -> bool,*/ this: Val<V>, sink: &mut Sink) { unimplemented!(); }
  //fn adjoint(&self, sink: &mut Sink) -> Val<V>;

  // TODO
  //fn _substitute(&self, subs: Vec<(RWVar, Rc<Any>)>) -> Option<(Rc<ANode>, Rc<AOp<V>>)> { None }
  fn _inplace(&self) -> Option<Val<V>> { None }

  fn _apply_val(&self, txn: Txn, val: AVal<V>);
}

pub trait WrapOpExt<V> {
  //fn substitute(&self) -> Option<Rc<AOp<V>>>;
  fn inplace(&self) -> Option<Val<V>>;
}

#[cfg(feature = "gpu")]
pub trait GPUWrapOpExt<V> {
  fn gpu_mux(&self, dev: GPUDeviceId) -> Val<V>;
}

pub struct Node {
  node: Rc<ANode>,
  xvar: RWVar,
  rvar: RVar,
}

/*impl Clone for Node {
  fn clone(&self) -> Node {
    Node{
      node: self.node.clone(),
    }
  }
}*/

impl Node {
  pub fn node(&self) -> &ANode {
    &*self.node
  }

  pub fn apply(&self) {
  }
}

pub struct Val<V> {
  node: Rc<ANode>,
  op:   Rc<AOp<V>>,
  xvar: RWVar,
  rvar: RVar,
}

impl<V> Clone for Val<V> {
  fn clone(&self) -> Val<V> {
    let rvar = RVar::default();
    Val{
      node: self.node.clone(),
      op:   self.op.clone(),
      xvar: self.xvar,
      rvar: rvar,
    }
  }
}

impl<V> Val<V> where V: 'static {
  pub fn from<Op>(op: Rc<Op>) -> Self where Op: AOp<V> + 'static {
    let rvar = RVar::default();
    Val{
      node: op.clone(),
      op:   op,
      xvar: RWVar(rvar),
      rvar: rvar,
    }
  }

  pub fn to_node(&self) -> Node {
    Node{
      node: self.node.clone(),
      xvar: self.xvar,
      rvar: self.rvar,
    }
  }

  /*pub fn downgrade(&self) -> AVal<V> {
    AVal{
      rvar: self.rvar,
      xvar: self.xvar,
      xval: self.op._value().clone_read(),
    }
  }*/

  pub fn exact_clone(&self) -> Val<V> {
    Val{
      node: self.node.clone(),
      op:   self.op.clone(),
      xvar: self.xvar,
      rvar: self.rvar,
    }
  }

  pub fn accumulate(&self) -> Val<V> {
    self.op._value()._set_accumulate();
    let rvar = RVar::default();
    Val{
      node: self.node.clone(),
      op:   self.op.clone(),
      xvar: RWVar(rvar),
      rvar: rvar,
    }
  }

  pub fn clobber(&self) -> Val<V> {
    self.op._value()._set_clobber();
    let rvar = RVar::default();
    Val{
      node: self.node.clone(),
      op:   self.op.clone(),
      xvar: RWVar(rvar),
      rvar: rvar,
    }
  }

  pub fn node(&self) -> &ANode {
    &*self.node
  }

  pub fn op(&self) -> &AOp<V> {
    &*self.op
  }

  pub fn _apply(&self, txn: Txn) {
    self.op._apply(txn, self.rvar, self.xvar);
  }

  pub fn _eval_recursive(&self, txn: Txn) {
    self.op._eval_recursive(txn, self.rvar, self.xvar);
  }

  pub fn eval(&self, txn: Txn) {
    self._eval_recursive(txn);
  }

  pub fn var(&self) -> RWVar {
    self.xvar
  }

  pub fn persist(&self, txn: Txn) {
    self.op._value().persist(txn, self.xvar);
  }

  pub fn write(&self, txn: Txn) -> Option<(WriteCap, WriteToken)> {
    self.op._value().write(txn, self.xvar)
  }

  pub fn get(&self, txn: Txn) -> Ref<V> {
    self.op._value().get(txn, self.rvar)
  }

  pub fn get_mut(&self, txn: Txn, token: WriteToken) -> RefMut<V> {
    self.op._value().get_mut(txn, self.xvar, token)
  }

  pub fn tangent(&self) -> Val<V> { unimplemented!(); }
  pub fn adjoint(&self, sink: &mut Sink) -> Val<V> { unimplemented!(); }
}

pub struct AVal<V> {
  rvar: RVar,
  xvar: RWVar,
  xval: RWVal<V>,
}

impl<V> AVal<V> where V: 'static {
  pub fn new(rvar: RVar, xvar: RWVar, xval: RWVal<V>) -> Self {
    AVal{
      rvar: rvar,
      xvar: xvar,
      xval: xval,
    }
  }

  pub fn var(&self) -> RWVar {
    self.xvar
  }

  pub fn persist(&self, txn: Txn) {
    self.xval.persist(txn, self.xvar);
  }

  pub fn write(&self, txn: Txn) -> Option<(WriteCap, WriteToken)> {
    self.xval.write(txn, self.xvar)
  }

  pub fn get(&self, txn: Txn) -> Ref<V> {
    self.xval.get(txn, self.rvar)
  }

  pub fn get_mut(&self, txn: Txn, token: WriteToken) -> RefMut<V> {
    self.xval.get_mut(txn, self.xvar, token)
  }
}

pub struct Sink {
  frozen:   HashSet<RWVar>,
  adj_map:  HashMap<RWVar, Vec<(Node, Rc<Any>)>>,
  //join_map: HashMap<RWVar, (Rc<ANode>, Rc<Any>)>,
  join_map: HashMap<RWVar, (Node, Rc<Any>)>,
  //adj_seq:  Vec<Rc<ANode>>,
}

impl Sink {
  //pub fn from<V>(sink_op: Rc<AOp<V>>) -> Self where V: 'static {
  pub fn from<V>(sink_op: Val<V>) -> Self where V: 'static {
    let mut sink = Sink{
      frozen:   HashSet::new(),
      adj_map:  HashMap::new(),
      join_map: HashMap::new(),
      //adj_seq:  Vec::new(),
    };
    // Add a "ones" adjoint op corresponding to `sink_op`.
    let sink_adj_op = match <OnesSrcOp as OnesSrcOpMaybeExt<V>>::maybe_build() {
      None => unimplemented!(),
      Some(op) => op,
    };
    sink.put_adj::<V>(sink_op.var(), sink_adj_op);
    sink
  }

  //pub fn get_adj<V>(&mut self, var: RWVar) -> Option<(Rc<ANode>, Rc<AOp<V>>)> where V: RWVal + 'static, SumJoinOp: SumJoinOpExt<V::T, V> {
  //pub fn get_adj<V>(&mut self, var: RWVar) -> Option<(Rc<ANode>, Rc<AOp<V>>)> where V: AVal + 'static, SumJoinOp: SumJoinOpExt<V::T, V> {
  //pub fn get_adj<V>(&mut self, var: RWVar) -> Option<(Rc<ANode>, Rc<AOp<V>>)> where V: 'static {
  pub fn get_adj<V>(&mut self, var: RWVar) -> Option<Val<V>> where V: 'static {
    self.frozen.insert(var);
    if self.adj_map.contains_key(&var) {
      let adjs = self.adj_map.get(&var).unwrap();
      match adjs.len() {
        0 => {}
        1 => {
          match (adjs[0].1).downcast_ref::<Val<V>>() {
            None => panic!(),
            Some(adj_op) => return Some(adj_op.clone()),
          }
        }
        _ => {
          if self.join_map.contains_key(&var) {
            let &(_, ref join_any_op) = self.join_map.get(&var).unwrap();
            match join_any_op.downcast_ref::<Val<V>>() {
              None => panic!(),
              Some(join_op) => return Some(join_op.clone()),
            }
          } else {
            let adj_ops: Vec<_> = adjs.iter().map(|&(_, ref a)| {
              match a.downcast_ref::<Val<V>>() {
                None => panic!(),
                Some(adj_op) => adj_op.clone(),
              }
            }).collect();
            let join = <SumJoinOp as SumJoinOpMaybeExt<V>>::maybe_build(adj_ops).unwrap();
            self.join_map.insert(var, (join.to_node(), Rc::new(join.clone())));
            return Some(join);
          }
        }
      }
    }
    None
  }

  pub fn put_adj<V>(&mut self, var: RWVar, adj_op: Val<V>) where V: 'static {
    assert!(!self.frozen.contains(&var));
    if self.adj_map.contains_key(&var) {
      let mut adjs = self.adj_map.get_mut(&var).unwrap();
      adjs.push((adj_op.to_node(), Rc::new(adj_op)));
    } else {
      self.adj_map.insert(var, vec![(adj_op.to_node(), Rc::new(adj_op))]);
    }
  }
}

pub struct OpBase<V> {
  ref_:     NodeRef,
  stack:    WalkStack,
  analyses: CloneMap,
  //tng_op:   RefCell<Option<(Rc<ANode>, Rc<AOp<V>>)>>,
  tng_op:   RefCell<Option<Val<V>>>,
}

impl<V> Default for OpBase<V> {
  fn default() -> Self {
    OpBase{
      ref_:     NodeRef::default(),
      stack:    WalkStack::default(),
      analyses: TypeMap::custom(),
      tng_op:   RefCell::new(None),
    }
  }
}

impl<V> Analyses for OpBase<V> {
  fn _liveness(&self) -> Option<LivenessAnalysis> {
    self.analyses.get::<LivenessAnalysis>().map(|x| x.clone())
  }
}

#[derive(Default)]
pub struct NodeVector {
  nodes:    Vec<Rc<ANode>>,
}

impl NodeVector {
  pub fn from(nodes: Vec<Rc<ANode>>) -> Self {
    NodeVector{nodes: nodes}
  }
}

/*impl ANode for NodeVector {
  fn _push(&self, epoch: Epoch, filter: &Fn(&ANode) -> bool, apply: &mut FnMut(&ANode)) {
    // FIXME: priority.
    for node in self.nodes.iter() {
      node._push(epoch, filter, apply);
    }
  }

  fn _pop(&self, epoch: Epoch, filter: &Fn(&ANode) -> bool, apply: &mut FnMut(&ANode)) {
    // FIXME: priority.
    for node in self.nodes.iter().rev() {
      node._pop(epoch, filter, apply);
    }
  }

  fn _io(&self) -> &IOVal {
    unimplemented!();
  }

  fn _apply(&self, _txn: Txn) {
    // TODO
  }
}*/

#[derive(Default)]
pub struct VarCollection {
  vars:     Vec<RVar>,
}

#[derive(Default)]
pub struct NodeRefMap<T> {
  node_map: HashMap<NodeRef, T>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WriteCap {
  Overwrite,
  Accumulate,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WriteMode {
  Exclusive,
  Accumulate,
  Clobber,
}

#[derive(Clone, Copy)]
pub struct WriteToken<'a> {
  //var:      RVar,
  xvar:     RWVar,
  first:    bool,
  borrow:   &'a (),
}

impl<'a> WriteToken<'a> {
  pub fn first_write(&self) -> bool {
    self.first
  }
}

pub struct RWValBuf<T> {
  mode:         WriteMode,
  curr_txn:     Option<Txn>,
  dirty:        bool,
  l_consumers:  HashSet<RVar>,
  d_consumers:  HashSet<RVar>,
  l_producers:  HashSet<RWVar>,
  d_producers:  HashSet<RWVar>,
  data:         Option<T>,
}

impl<T> Default for RWValBuf<T> {
  fn default() -> Self {
    RWValBuf{
      mode:         WriteMode::Exclusive,
      curr_txn:     None,
      dirty:        false,
      l_consumers:  HashSet::new(),
      d_consumers:  HashSet::new(),
      l_producers:  HashSet::new(),
      d_producers:  HashSet::new(),
      data:         None,
    }
  }
}

pub struct RWVal<T> {
  ref_:     ValRef,
  alloc:    Rc<Fn(Txn) -> T>,
  //root:     RWVar,
  //var:      RVar,
  buf:      Rc<RefCell<RWValBuf<T>>>,
  borrow:   (),
}

impl<T> IOVal for RWVal<T> where T: 'static {
  fn _deserialize(&self, txn: Txn, write: &mut FnMut(WriteCap, &mut Any)) {
    // TODO
    unimplemented!();
    /*if let Some((cap, token)) = self.write(txn) {
      let mut buf = self.get_mut(txn, token);
      write(cap, &mut *buf);
    }*/
  }

  fn _serialize(&self, txn: Txn, read: &mut FnMut(&Any)) {
    // TODO
    unimplemented!();
    /*let buf = self.get(txn);
    read(&*buf);*/
  }
}

impl<T> RWVal<T> where T: 'static {
  pub fn from(alloc: Rc<Fn(Txn) -> T>) -> Self {
    //let var = RVar::default();
    RWVal{
      ref_:     ValRef::default(),
      alloc:    alloc,
      //root:     RWVar(var),
      //var:      var,
      buf:      Rc::new(RefCell::new(RWValBuf::default())),
      borrow:   (),
    }
  }

  pub fn exact_duplicate(&self) -> Self {
    RWVal{
      ref_:     self.ref_,
      alloc:    self.alloc.clone(),
      //root:     self.root,
      //var:      self.var,
      buf:      self.buf.clone(),
      borrow:   (),
    }
  }

  pub fn clone_read(&self) -> Self {
    RWVal{
      ref_:     self.ref_,
      alloc:    self.alloc.clone(),
      //root:     self.root,
      //var:      rvar,
      buf:      self.buf.clone(),
      borrow:   (),
    }
  }

  pub fn clone_read_write(&self) -> Self {
    let var = RVar::default();
    RWVal{
      ref_:     self.ref_,
      alloc:    self.alloc.clone(),
      //root:     RWVar(var),
      //var:      var,
      buf:      self.buf.clone(),
      borrow:   (),
    }
  }

  pub fn _set_accumulate(&self) {
    let mut buf = self.buf.borrow_mut();
    match buf.mode {
      WriteMode::Exclusive => {
        buf.mode = WriteMode::Accumulate;
      }
      WriteMode::Accumulate => {}
      _ => panic!(),
    }
  }

  pub fn _set_clobber(&self) {
    let mut buf = self.buf.borrow_mut();
    match buf.mode {
      WriteMode::Exclusive => {
        buf.mode = WriteMode::Clobber;
      }
      WriteMode::Clobber => {}
      _ => panic!(),
    }
  }

  /*pub fn root_var(&self) -> RWVar {
    self.root
  }

  pub fn tmp_var(&self) -> RVar {
    self.var
  }*/

  pub fn txn(&self) -> Option<Txn> {
    let buf = self.buf.borrow();
    buf.curr_txn
  }

  pub fn reset(&self) {
    let mut buf = self.buf.borrow_mut();
    buf.curr_txn = None;
    buf.l_consumers.clear();
    buf.d_consumers.clear();
    buf.l_producers.clear();
    buf.d_producers.clear();
  }

  pub fn release(&self) {
    self.reset();
    let mut buf = self.buf.borrow_mut();
    buf.data = None;
  }

  pub fn persist(&self, txn: Txn, /*rvar: RVar,*/ xvar: RWVar) {
    let new_txn = {
      let buf = self.buf.borrow();
      buf.curr_txn.is_none() || buf.curr_txn.unwrap() != txn
    };
    if new_txn {
      self.reset();
      let mut buf = self.buf.borrow_mut();
      buf.curr_txn = Some(txn);
    }

    let mut buf = self.buf.borrow_mut();
    assert!(!buf.d_producers.contains(&xvar),
        "`persist` should be called before all other writes");
    match buf.l_producers.len() {
      0 => {}
      1 => {
        assert!(buf.l_producers.contains(&xvar),
            "`persist` should be called before all other writes");
        return;
      }
      _ => panic!("`persist` should be called before all other writes"),
    }
    assert!(buf.l_consumers.is_empty(),
        "`persist` should be called before reads");
    buf.l_producers.insert(xvar);
  }

  pub fn write(&self, txn: Txn, xvar: RWVar) -> Option<(WriteCap, WriteToken)> {
    let new_txn = {
      let buf = self.buf.borrow();
      buf.curr_txn.is_none() || buf.curr_txn.unwrap() != txn
    };
    if new_txn {
      self.reset();
      let mut buf = self.buf.borrow_mut();
      buf.curr_txn = Some(txn);
    }

    let mut buf = self.buf.borrow_mut();
    match buf.mode {
      WriteMode::Exclusive => {
        match (buf.l_producers.len(), buf.d_producers.len()) {
          (0, 0) => {}
          (1, 0) => {
            if buf.l_producers.contains(&xvar) {
              return None;
            }
            panic!("attempting second write to `Exclusive` val");
          }
          (_, 0) => panic!("attempting multiple writes to `Exclusive` val"),
          (_, _) => panic!("all writes to `Exclusive` val must be live"),
        }
        assert!(buf.l_consumers.is_empty(),
            "attempting write to `Exclusive` val after read");
      }
      WriteMode::Accumulate => {
        match (buf.l_producers.len(), buf.d_producers.len()) {
          (0, 0) => {}
          (_, 0) => {
            if buf.l_producers.contains(&xvar) {
              return None;
            }
          }
          (_, _) => panic!("all writes to `Accumulate` val must be live"),
        }
        assert!(buf.l_consumers.is_empty(),
            "attempting write to `Accumulate` val after read");
      }
      WriteMode::Clobber => {
        match (buf.l_producers.len(), buf.d_producers.len()) {
          (0, 0) => {}
          (1, _) => {
            if buf.l_producers.contains(&xvar) {
              return None;
            }
          }
          (_, _) => panic!("attempting multiple live writes to `Clobber` val"),
        }
        let &mut RWValBuf{
            ref mut l_consumers,
            ref mut d_consumers,
            ref mut l_producers,
            ref mut d_producers,
            ..} = &mut *buf;
        d_consumers.extend(l_consumers.drain());
        d_producers.extend(l_producers.drain());
      }
    }

    let first = buf.l_producers.is_empty();
    let cap = match (buf.mode, first) {
      (WriteMode::Accumulate, false) => WriteCap::Accumulate,
      (_, true) => WriteCap::Overwrite,
      _ => unreachable!(),
    };
    buf.l_producers.insert(xvar);
    Some((cap, WriteToken{xvar: xvar, first: first, borrow: &self.borrow}))
  }

  pub fn get(&self, txn: Txn, rvar: RVar) -> Ref<T> {
    {
      let mut buf = self.buf.borrow_mut();

      let mut valid_txn = false;
      if let Some(curr_txn) = buf.curr_txn {
        if curr_txn == txn {
          valid_txn = true;
        }
      }
      assert!(valid_txn,
          "attempting a read with an invalid txn (did you forget to `persist` or `write`?)");

      assert!(!buf.d_consumers.contains(&rvar),
          "attempting a stale read (the value has been clobbered)");
      match buf.l_producers.len() {
        0 => panic!("attempting an invalid read (the value was never written)"),
        1 => {}
        _ => panic!("attempting an invalid read (too many live writes)"),
      }
      buf.l_consumers.insert(rvar);

      assert!(buf.data.is_some(),
          "attempting a read on empty data");
    }
    let buf = self.buf.borrow();
    Ref::map(buf, |buf| buf.data.as_ref().unwrap())
  }

  pub fn get_mut(&self, txn: Txn, /*rvar: RVar,*/ xvar: RWVar, token: WriteToken) -> RefMut<T> {
    assert_eq!(xvar, token.xvar);
    let mut buf = self.buf.borrow_mut();
    {
      let mut valid_txn = false;
      if let Some(curr_txn) = buf.curr_txn {
        if curr_txn == txn {
          valid_txn = true;
        }
      }
      assert!(valid_txn,
          "attempting a write with an invalid txn (did you forget to `write`?)");

      assert!(buf.l_consumers.is_empty(),
          "attempting a write-after-read (check your `get` and `get_mut` order)");
      assert!(!buf.d_producers.contains(&xvar),
          "attempting an invalid write (the value has been clobbered)");
      assert!(buf.l_producers.contains(&xvar),
          "attempting an invalid write (did you forget to `write`?)");

      if buf.data.is_none() {
        buf.data = Some((self.alloc)(txn));
      }
    }
    RefMut::map(buf, |buf| buf.data.as_mut().unwrap())
  }

  pub fn set<F>(&self, txn: Txn, xvar: RWVar, f: F) where F: FnOnce(RefMut<T>) {
    if let Some((cap, token)) = self.write(txn, xvar) {
      match cap {
        WriteCap::Overwrite => {
          f(self.get_mut(txn, xvar, token));
        }
        _ => unimplemented!(),
      }
    }
  }
}

pub trait IoReadable<'a>: Sized + 'static {
  fn read(&'a self, reader: &mut IoReader<'a>) {
    reader.read(self);
  }
}

pub trait IoReader<'a> {
  fn read(&mut self, src: &'a Any);
}

impl<'a, R> IoReader<'a> for R where R: MemIoReader<'a> {
  fn read(&mut self, src: &'a Any) {
    let ty_id = src.get_type_id();
    if self.read_mem(src).is_some() {
      return;
    }
    /*if self.read_mem(src).is_some() {
      return;
    }*/
    panic!("IoReader: `src` has an unhandled type: {:?}", ty_id);
  }
}

pub trait IoWriter<'a> {
  fn write(&mut self, cap: WriteCap, dst: &'a mut Any);
}

impl<'a, W> IoWriter<'a> for W where W: MemIoWriter<'a> {
  fn write(&mut self, cap: WriteCap, mut dst: &'a mut Any) {
    let ty_id = (*dst).get_type_id();
    if self.write_mem(cap, &mut dst).is_some() {
      return;
    }
    /*if self.write_mem(mode, &mut dst).is_some() {
      return;
    }*/
    panic!("IoWriter: `dst` has an unhandled type: {:?}", ty_id);
  }
}

pub struct FlatReader<'a, T> where T: 'a {
  pub offset:   usize,
  pub inner:    &'a mut T,
}

impl<'a, T> FlatReader<'a, T> {
  pub fn new(inner: &'a mut T) -> Self {
    FlatReader{
      offset:   0,
      inner:    inner,
    }
  }
}

impl<'a, T> FnOnce<(&'a Any,)> for FlatReader<'a, T> where FlatReader<'a, T>: IoReader<'a> {
  type Output = ();

  extern "rust-call" fn call_once(mut self, args: (&'a Any,)) -> () {
    self.call_mut(args)
  }
}

impl<'a, T> FnMut<(&'a Any,)> for FlatReader<'a, T> where FlatReader<'a, T>: IoReader<'a> {
  extern "rust-call" fn call_mut(&mut self, args: (&'a Any,)) -> () {
    self.read(args.0);
  }
}

pub struct FlatWriter<'a, T> where T: 'a {
  pub offset:   usize,
  pub inner:    &'a mut T,
}

impl<'a, T> FlatWriter<'a, T> {
  pub fn new(inner: &'a mut T) -> Self {
    FlatWriter{
      offset:   0,
      inner:    inner,
    }
  }
}

impl<'a, T> FnOnce<(WriteCap, &'a mut Any)> for FlatWriter<'a, T> where FlatWriter<'a, T>: IoWriter<'a> {
  type Output = ();

  extern "rust-call" fn call_once(mut self, args: (WriteCap, &'a mut Any)) -> () {
    self.call_mut(args)
  }
}

impl<'a, T> FnMut<(WriteCap, &'a mut Any)> for FlatWriter<'a, T> where FlatWriter<'a, T>: IoWriter<'a> {
  extern "rust-call" fn call_mut(&mut self, args: (WriteCap, &'a mut Any)) -> () {
    self.write(args.0, args.1);
  }
}

pub struct OpExt<F, V> {
  build:    Rc<Fn(Vec<Rc<Any>>) -> Val<V>>,
  init:     Rc<Fn() -> RWVal<V>>,
  prepare:  Option<Rc<Fn(Txn, RefMut<F>)>>,
  cleanup:  Option<Rc<Fn(Txn, RefMut<F>)>>,
  apply:    Rc<Fn(Txn, RefMut<F>, AVal<V>)>,
  tangent:  Option<Rc<Fn() -> Val<V>>>,
  adjoint:  Option<Rc<Fn(Val<V>, &mut Sink)>>,
  inplace:  Option<Rc<Fn(Val<V>) -> Val<V>>>,
}

impl<F, V> Clone for OpExt<F, V> {
  fn clone(&self) -> Self {
    OpExt{
      build:    self.build.clone(),
      init:     self.init.clone(),
      prepare:  self.prepare.clone(),
      cleanup:  self.cleanup.clone(),
      apply:    self.apply.clone(),
      tangent:  self.tangent.clone(),
      adjoint:  self.adjoint.clone(),
      inplace:  self.inplace.clone(),
    }
  }
}

impl<V> WrapOpExt<V> for Rc<AOp<V>> where V: 'static {
  fn inplace(&self) -> Option<Val<V>> {
    self._inplace()
  }
}

#[cfg(feature = "gpu")]
impl<V> GPUWrapOpExt<V> for Val<V> where V: 'static {
  fn gpu_mux(&self, dev: GPUDeviceId) -> Val<V> {
    let wrap_ext: OpExt<GPUMuxFun<V>, V> = GPUMuxFun::<V>::build_ext();
    let wrap_fun: GPUMuxFun<V> = GPUMuxFun{
      dev:  dev,
      op:   self.clone(),
    };
    let op: Rc<FSrcOp<GPUMuxFun<V>, V>> = Rc::new(FSrcOp{
      base: OpBase::default(),
      ext:  wrap_ext,
      fun:  RefCell::new(wrap_fun),
      ctrl: vec![],
      val:  self.op()._value().clone_read(),
    });
    Val::from(op)
  }
}

pub struct FSrcOp<F, V> {
  base: OpBase<V>,
  ext:  OpExt<F, V>,
  fun:  RefCell<F>,
  ctrl: Vec<Rc<ANode>>,
  val:  RWVal<V>,
}

impl<F, V> FSrcOp<F, V> {
  pub fn new(fun: F, ext: OpExt<F, V>, val: RWVal<V>) -> Self {
    FSrcOp{
      base: OpBase::default(),
      ext:  ext,
      fun:  RefCell::new(fun),
      ctrl: vec![],
      val:  val,
    }
  }
}

impl<F, V> ANode for FSrcOp<F, V> where RWVal<V>: IOVal + 'static {
  fn _walk(&self) -> &Walk {
    &self.base.stack
  }

  fn _io(&self) -> &IOVal {
    &self.val
  }

  fn _analyses(&self) -> &Analyses {
    &self.base
  }

  /*fn _pred_fwd(&self, _pred_buf: &mut Vec<Rc<ANode>>) {
  }

  fn _pred_rev(&self, _pred_buf: &mut Vec<Rc<ANode>>) {
  }*/

  fn _push(&self, _stop_txn: Option<Txn>, epoch: Epoch, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.push(epoch) {
      apply(self);
    }
  }

  fn _pop(&self, _stop_txn: Option<Txn>, epoch: Epoch, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.pop(epoch) {
      apply(self);
    }
  }

  fn _txn(&self) -> Option<Txn> {
    self.val.txn()
  }

  /*fn _persist(&self, txn: Txn) {
    self.val.persist(txn);
  }*/

  /*fn _prepare(&self, txn: Txn) {
    if let Some(ref prepare) = self.ext.prepare {
      (prepare)(txn, self.fun.borrow_mut());
    }
  }*/

  /*fn _cleanup(&self, txn: Txn) {
    // TODO
    if let Some(ref cleanup) = self.ext.cleanup {
      (cleanup)(txn, self.fun.borrow_mut());
    }
  }*/

  fn _apply(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    self._apply_val(txn, AVal::new(rvar, xvar, self._value().clone_read()));
  }

  fn _eval_recursive(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    if Some(txn) != self._txn() {
      self._apply(txn, rvar, xvar);
    }
  }

  /*fn deserialize_forward(&self, txn: Txn, writer: &mut FnMut(WriteMode, &mut Any)) {
    // TODO
    unimplemented!();
  }

  fn deserialize_reverse(&self, txn: Txn, writer: &mut FnMut(WriteMode, &mut Any)) {
    // TODO
    unimplemented!();
  }

  fn serialize_forward(&self, txn: Txn, reader: &mut FnMut(&mut Any)) {
    // TODO
    unimplemented!();
  }

  fn serialize_reverse(&self, txn: Txn, reader: &mut FnMut(&mut Any)) {
    // TODO
    unimplemented!();
  }*/
}

impl<F, V> AOp<V> for FSrcOp<F, V> where RWVal<V>: IOVal + 'static {
  fn _make_value(&self) -> RWVal<V> {
    (self.ext.init)()
    //(self.ext.init)(self.fun.borrow_mut())
  }

  fn _value(&self) -> &RWVal<V> {
    &self.val
  }

  fn _make_tangent(&self) -> Val<V> {
    match self.ext.tangent {
      None => unimplemented!(),
      Some(ref tangent) => (tangent)(),
    }
  }

  fn tangent(&self) -> Val<V> {
    // TODO
    unimplemented!();
  }

  fn _pop_adjoint(&self, epoch: Epoch, /*filter: &Fn(&ANode) -> bool,*/ this: Val<V>, sink: &mut Sink) {
    if self.base.stack.pop(epoch) {
      match self.ext.adjoint {
        None => panic!(),
        Some(ref adjoint) => (adjoint)(this, sink),
      }
    }
  }

  /*//fn adjoint(&self, sink: &mut Sink) -> (Rc<ANode>, Rc<AOp<V>>) {
  fn adjoint(&self, sink: &mut Sink) -> Val<V> {
    match sink.get_adj(self.var()) {
      None => panic!(),
      Some(adj) => adj,
    }
  }*/

  fn _apply_val(&self, txn: Txn, val: AVal<V>) {
    (self.ext.apply)(txn, self.fun.borrow_mut(), val);
  }
}

/*pub struct Pipe1Op<F, V, W> where W: AVal {
  base: OpBase<W>,
  ext:  OpExt<W>,
  fun:  F,
  // TODO: should not contain an input `x` but instead a "slot" in which to
  // pipe an input of the same type.
  x_:   Rc<AOp<V>>,
  y:    W,
}

impl<F, V, W> Pipe1Op<F, V, W>
where V: AVal, W: AVal {
  pub fn attach(&self, txn: Txn, input: Rc<AOp<V>>) {
    // TODO: figure out `attach` semantics.
  }
}*/

pub struct F1Op<F, V1, W> {
  base: OpBase<W>,
  ext:  OpExt<F, W>,
  fun:  RefCell<F>,
  ctrl: Vec<Rc<ANode>>,
  x_:   Val<V1>,
  y:    RWVal<W>,
}

impl<F, V1, W> F1Op<F, V1, W> {
  pub fn new(fun: F, ext: OpExt<F, W>, x_: Val<V1>, y: RWVal<W>) -> Self {
    F1Op{
      base: OpBase::default(),
      ext:  ext,
      fun:  RefCell::new(fun),
      ctrl: vec![],
      x_:   x_,
      y:    y,
    }
  }
}

impl<F, V1, W> ANode for F1Op<F, V1, W> where V1: 'static, RWVal<W>: IOVal + 'static {
  fn _walk(&self) -> &Walk {
    &self.base.stack
  }

  fn _io(&self) -> &IOVal {
    &self.y
  }

  fn _analyses(&self) -> &Analyses {
    &self.base
  }

  /*fn _pred_fwd(&self, pred_buf: &mut Vec<Rc<ANode>>) {
    pred_buf.push(self.x_.clone());
  }

  fn _pred_rev(&self, pred_buf: &mut Vec<Rc<ANode>>) {
    pred_buf.push(self.x_.clone());
  }*/

  fn _push(&self, stop_txn: Option<Txn>, epoch: Epoch, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.push(epoch) {
      if stop_txn.is_none() || stop_txn != self._txn() {
        self.x_.node()._push(stop_txn, epoch, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, stop_txn: Option<Txn>, epoch: Epoch, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.pop(epoch) {
      apply(self);
      if stop_txn.is_none() || stop_txn != self._txn() {
        self.x_.node()._pop(stop_txn, epoch, apply);
      }
    }
  }

  fn _txn(&self) -> Option<Txn> {
    self.y.txn()
  }

  /*fn _persist(&self, txn: Txn) {
    self.y.persist(txn);
  }*/

  /*fn _prepare(&self, txn: Txn) {
    self.x_.node().eval(txn);
    if let Some(ref prepare) = self.ext.prepare {
      (prepare)(txn, self.fun.borrow_mut());
    }
  }*/

  /*fn _cleanup(&self, txn: Txn) {
    // TODO
    if let Some(ref cleanup) = self.ext.cleanup {
      (cleanup)(txn, self.fun.borrow_mut());
    }
  }*/

  fn _apply(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    self._apply_val(txn, AVal::new(rvar, xvar, self._value().clone_read()));
  }

  fn _eval_recursive(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    if Some(txn) != self._txn() {
      self.x_._eval_recursive(txn);
      self._apply(txn, rvar, xvar);
    }
  }
}

impl<F, V1, W> AOp<W> for F1Op<F, V1, W> where V1: 'static, RWVal<W>: IOVal + 'static {
  fn _make_value(&self) -> RWVal<W> {
    (self.ext.init)()
    //(self.ext.init)(self.fun.borrow_mut())
  }

  fn _value(&self) -> &RWVal<W> {
    &self.y
  }

  fn _make_tangent(&self) -> Val<W> {
    match self.ext.tangent {
      None => unimplemented!(),
      Some(ref tangent) => (tangent)(),
    }
  }

  fn tangent(&self) -> Val<W> {
    let mut tng_op = self.base.tng_op.borrow_mut();
    if tng_op.is_none() {
      *tng_op = Some(self._make_tangent());
    }
    tng_op.as_ref().unwrap().clone()
  }

  fn _pop_adjoint(&self, epoch: Epoch, /*filter: &Fn(&ANode) -> bool,*/ this: Val<W>, sink: &mut Sink) {
    if self.base.stack.pop(epoch) {
      match self.ext.adjoint {
        None => panic!(),
        Some(ref adjoint) => (adjoint)(this, sink),
      }
      self.x_.op()._pop_adjoint(epoch, self.x_.clone(), sink);
    }
  }

  /*//fn adjoint(&self, sink: &mut Sink) -> (Rc<ANode>, Rc<AOp<W>>) {
  fn adjoint(&self, sink: &mut Sink) -> Val<W> {
    match sink.get_adj(self.var()) {
      None => panic!(),
      Some(adj) => adj,
    }
  }*/

  /*fn _substitute(&self, subs: Vec<(RWVar, Rc<Any>)>) -> Option<(Rc<ANode>, Rc<AOp<W>>)> {
    // TODO: what happens when vars are repeated in the subs list?
    for &(var, ref arg) in subs.iter() {
      if self.x_.var() == var {
        let args = vec![arg.clone()];
        let (node, op) = (self.ext.build)(args);
        return Some((node, op));
      }
    }
    None
  }*/

  default fn _inplace(&self) -> Option<Val<W>> {
    None
  }

  fn _apply_val(&self, txn: Txn, val: AVal<W>) {
    (self.ext.apply)(txn, self.fun.borrow_mut(), val);
  }
}

impl<F, V> AOp<V> for F1Op<F, V, V> where V: 'static, RWVal<V>: IOVal + 'static {
  fn _inplace(&self) -> Option<Val<V>> {
    match self.ext.inplace {
      None => None,
      Some(ref inplace) => {
        if self.x_.node()._walk().outdegree() == 1 {
          Some((inplace)(self.x_.clone()))
        } else {
          None
        }
      }
    }
  }
}

pub struct F2Op<F, V1, V2, W> {
  base: OpBase<W>,
  ext:  OpExt<F, W>,
  fun:  RefCell<F>,
  ctrl: Vec<Rc<ANode>>,
  x1_:  Val<V1>,
  x2_:  Val<V2>,
  y:    RWVal<W>,
}

impl<F, V1, V2, W> F2Op<F, V1, V2, W> {
  pub fn new(fun: F, ext: OpExt<F, W>, x1_: Val<V1>, x2_: Val<V2>, y: RWVal<W>) -> Self {
    F2Op{
      base: OpBase::default(),
      ext:  ext,
      fun:  RefCell::new(fun),
      ctrl: vec![],
      x1_:  x1_,
      x2_:  x2_,
      y:    y,
    }
  }
}

impl<F, V1, V2, W> ANode for F2Op<F, V1, V2, W> where V1: 'static, V2: 'static, RWVal<W>: IOVal + 'static {
  fn _walk(&self) -> &Walk {
    &self.base.stack
  }

  fn _io(&self) -> &IOVal {
    &self.y
  }

  fn _analyses(&self) -> &Analyses {
    &self.base
  }

  /*fn _pred_fwd(&self, pred_buf: &mut Vec<Rc<ANode>>) {
    pred_buf.push(self.x1_.clone());
    pred_buf.push(self.x2_.clone());
  }

  fn _pred_rev(&self, pred_buf: &mut Vec<Rc<ANode>>) {
    pred_buf.push(self.x2_.clone());
    pred_buf.push(self.x1_.clone());
  }*/

  fn _push(&self, stop_txn: Option<Txn>, epoch: Epoch, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.push(epoch) {
      if stop_txn.is_none() || stop_txn != self._txn() {
        self.x1_.node()._push(stop_txn, epoch, apply);
        self.x2_.node()._push(stop_txn, epoch, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, stop_txn: Option<Txn>, epoch: Epoch, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.pop(epoch) {
      apply(self);
      if stop_txn.is_none() || stop_txn != self._txn() {
        self.x2_.node()._pop(stop_txn, epoch, apply);
        self.x1_.node()._pop(stop_txn, epoch, apply);
      }
    }
  }

  fn _txn(&self) -> Option<Txn> {
    self.y.txn()
  }

  /*fn _persist(&self, txn: Txn) {
    self.y.persist(txn);
  }*/

  /*fn _prepare(&self, txn: Txn) {
    self.x1_.node().eval(txn);
    self.x2_.node().eval(txn);
    if let Some(ref prepare) = self.ext.prepare {
      (prepare)(txn, self.fun.borrow_mut());
    }
  }*/

  /*fn _cleanup(&self, txn: Txn) {
    // TODO
    if let Some(ref cleanup) = self.ext.cleanup {
      (cleanup)(txn, self.fun.borrow_mut());
    }
  }*/

  fn _apply(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    self._apply_val(txn, AVal::new(rvar, xvar, self._value().clone_read()));
  }

  fn _eval_recursive(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    if Some(txn) != self._txn() {
      self.x1_._eval_recursive(txn);
      self.x2_._eval_recursive(txn);
      self._apply(txn, rvar, xvar);
    }
  }
}

impl<F, V1, V2, W> AOp<W> for F2Op<F, V1, V2, W> where V1: 'static, V2: 'static, RWVal<W>: IOVal + 'static {
  fn _make_value(&self) -> RWVal<W> {
    (self.ext.init)()
    //(self.ext.init)(self.fun.borrow_mut())
  }

  fn _value(&self) -> &RWVal<W> {
    &self.y
  }

  fn _make_tangent(&self) -> Val<W> {
    match self.ext.tangent {
      None => unimplemented!(),
      Some(ref tangent) => (tangent)(),
    }
  }

  fn tangent(&self) -> Val<W> {
    let mut tng_op = self.base.tng_op.borrow_mut();
    if tng_op.is_none() {
      *tng_op = Some(self._make_tangent());
    }
    tng_op.as_ref().unwrap().clone()
  }

  fn _pop_adjoint(&self, epoch: Epoch, /*filter: &Fn(&ANode) -> bool,*/ this: Val<W>, sink: &mut Sink) {
    if self.base.stack.pop(epoch) {
      match self.ext.adjoint {
        None => panic!(),
        Some(ref adjoint) => (adjoint)(this, sink),
      }
      self.x2_.op()._pop_adjoint(epoch, self.x2_.clone(), sink);
      self.x1_.op()._pop_adjoint(epoch, self.x1_.clone(), sink);
    }
  }

  /*//fn adjoint(&self, sink: &mut Sink) -> (Rc<ANode>, Rc<AOp<W>>) {
  fn adjoint(&self, sink: &mut Sink) -> Val<W> {
    match sink.get_adj(self.var()) {
      None => panic!(),
      Some(adj) => adj,
    }
  }*/

  fn _apply_val(&self, txn: Txn, val: AVal<W>) {
    (self.ext.apply)(txn, self.fun.borrow_mut(), val);
  }
}

pub struct F3Op<F, V1, V2, V3, W> {
  base: OpBase<W>,
  ext:  OpExt<F, W>,
  fun:  RefCell<F>,
  ctrl: Vec<Rc<ANode>>,
  //x1_:  Rc<AOp<V1>>,
  //x2_:  Rc<AOp<V2>>,
  //x3_:  Rc<AOp<V3>>,
  x1_:  Val<V1>,
  x2_:  Val<V2>,
  x3_:  Val<V3>,
  y:    RWVal<W>,
}

impl<F, V1, V2, V3, W> ANode for F3Op<F, V1, V2, V3, W> where V1: 'static, V2: 'static, V3: 'static, RWVal<W>: IOVal + 'static {
  fn _walk(&self) -> &Walk {
    &self.base.stack
  }

  fn _io(&self) -> &IOVal {
    &self.y
  }

  fn _analyses(&self) -> &Analyses {
    &self.base
  }

  /*fn _pred_fwd(&self, pred_buf: &mut Vec<Rc<ANode>>) {
    pred_buf.push(self.x1_.clone());
    pred_buf.push(self.x2_.clone());
    pred_buf.push(self.x3_.clone());
  }

  fn _pred_rev(&self, pred_buf: &mut Vec<Rc<ANode>>) {
    pred_buf.push(self.x3_.clone());
    pred_buf.push(self.x2_.clone());
    pred_buf.push(self.x1_.clone());
  }*/

  fn _push(&self, stop_txn: Option<Txn>, epoch: Epoch, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.push(epoch) {
      if stop_txn.is_none() || stop_txn != self._txn() {
        self.x1_.node()._push(stop_txn, epoch, apply);
        self.x2_.node()._push(stop_txn, epoch, apply);
        self.x3_.node()._push(stop_txn, epoch, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, stop_txn: Option<Txn>, epoch: Epoch, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.pop(epoch) {
      apply(self);
      if stop_txn.is_none() || stop_txn != self._txn() {
        self.x3_.node()._pop(stop_txn, epoch, apply);
        self.x2_.node()._pop(stop_txn, epoch, apply);
        self.x1_.node()._pop(stop_txn, epoch, apply);
      }
    }
  }

  fn _txn(&self) -> Option<Txn> {
    self.y.txn()
  }

  /*fn _persist(&self, txn: Txn) {
    self.y.persist(txn);
  }*/

  /*fn _prepare(&self, txn: Txn) {
    self.x1_.node().eval(txn);
    self.x2_.node().eval(txn);
    self.x3_.node().eval(txn);
    if let Some(ref prepare) = self.ext.prepare {
      (prepare)(txn, self.fun.borrow_mut());
    }
  }*/

  /*fn _cleanup(&self, txn: Txn) {
    // TODO
    if let Some(ref cleanup) = self.ext.cleanup {
      (cleanup)(txn, self.fun.borrow_mut());
    }
  }*/

  fn _apply(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    self._apply_val(txn, AVal::new(rvar, xvar, self._value().clone_read()));
  }

  fn _eval_recursive(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    if Some(txn) != self._txn() {
      self.x1_._eval_recursive(txn);
      self.x2_._eval_recursive(txn);
      self.x3_._eval_recursive(txn);
      self._apply(txn, rvar, xvar);
    }
  }
}

impl<F, V1, V2, V3, W> AOp<W> for F3Op<F, V1, V2, V3, W> where V1: 'static, V2: 'static, V3: 'static, RWVal<W>: IOVal + 'static {
  fn _make_value(&self) -> RWVal<W> {
    (self.ext.init)()
    //(self.ext.init)(self.fun.borrow_mut())
  }

  fn _value(&self) -> &RWVal<W> {
    &self.y
  }

  fn _make_tangent(&self) -> Val<W> {
    match self.ext.tangent {
      None => unimplemented!(),
      Some(ref tangent) => (tangent)(),
    }
  }

  //fn tangent(&self) -> (Rc<ANode>, Rc<AOp<W>>) {
  fn tangent(&self) -> Val<W> {
    let mut tng_op = self.base.tng_op.borrow_mut();
    if tng_op.is_none() {
      *tng_op = Some(self._make_tangent());
    }
    tng_op.as_ref().unwrap().clone()
  }

  fn _pop_adjoint(&self, epoch: Epoch, /*filter: &Fn(&ANode) -> bool,*/ this: Val<W>, sink: &mut Sink) {
    if self.base.stack.pop(epoch) {
      match self.ext.adjoint {
        None => panic!(),
        Some(ref adjoint) => (adjoint)(this, sink),
      }
      self.x3_.op()._pop_adjoint(epoch, self.x3_.clone(), sink);
      self.x2_.op()._pop_adjoint(epoch, self.x2_.clone(), sink);
      self.x1_.op()._pop_adjoint(epoch, self.x1_.clone(), sink);
    }
  }

  /*//fn adjoint(&self, sink: &mut Sink) -> (Rc<ANode>, Rc<AOp<W>>) {
  fn adjoint(&self, sink: &mut Sink) -> Val<W> {
    match sink.get_adj(self.var()) {
      None => panic!(),
      Some(adj) => adj,
    }
  }*/

  fn _apply_val(&self, txn: Txn, val: AVal<W>) {
    (self.ext.apply)(txn, self.fun.borrow_mut(), val);
  }
}

pub struct FJoinOp<F, V, W> {
  base: OpBase<W>,
  ext:  OpExt<F, W>,
  fun:  RefCell<F>,
  ctrl: Vec<Rc<ANode>>,
  xs_:  Vec<Val<V>>,
  y:    RWVal<W>,
}

impl<F, V, W> FJoinOp<F, V, W> {
  pub fn new(fun: F, ext: OpExt<F, W>, xs_: Vec<Val<V>>, y: RWVal<W>) -> Self {
    FJoinOp{
      base: OpBase::default(),
      ext:  ext,
      fun:  RefCell::new(fun),
      ctrl: vec![],
      xs_:  xs_,
      y:    y,
    }
  }
}

impl<F, V, W> ANode for FJoinOp<F, V, W> where V: 'static, RWVal<W>: IOVal + 'static {
  fn _walk(&self) -> &Walk {
    &self.base.stack
  }

  fn _io(&self) -> &IOVal {
    &self.y
  }

  fn _analyses(&self) -> &Analyses {
    &self.base
  }

  /*fn _pred_fwd(&self, pred_buf: &mut Vec<Rc<ANode>>) {
    for x_ in self.xs_.iter() {
      pred_buf.push(x_.clone());
    }
  }

  fn _pred_rev(&self, pred_buf: &mut Vec<Rc<ANode>>) {
    for x_ in self.xs_.iter().rev() {
      pred_buf.push(x_.clone());
    }
  }*/

  fn _push(&self, stop_txn: Option<Txn>, epoch: Epoch, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.push(epoch) {
      if stop_txn.is_none() || stop_txn != self._txn() {
        for x_ in self.xs_.iter() {
          x_.node()._push(stop_txn, epoch, apply);
        }
      }
      apply(self);
    }
  }

  fn _pop(&self, stop_txn: Option<Txn>, epoch: Epoch, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.pop(epoch) {
      apply(self);
      if stop_txn.is_none() || stop_txn != self._txn() {
        for x_ in self.xs_.iter().rev() {
          x_.node()._pop(stop_txn, epoch, apply);
        }
      }
    }
  }

  fn _txn(&self) -> Option<Txn> {
    self.y.txn()
  }

  /*fn _persist(&self, txn: Txn) {
    self.y.persist(txn);
  }*/

  /*fn _prepare(&self, txn: Txn) {
    for x in self.xs_.iter() {
      x.node().eval(txn);
    }
    if let Some(ref prepare) = self.ext.prepare {
      (prepare)(txn, self.fun.borrow_mut());
    }
  }*/

  /*fn _cleanup(&self, txn: Txn) {
    // TODO
    if let Some(ref cleanup) = self.ext.cleanup {
      (cleanup)(txn, self.fun.borrow_mut());
    }
  }*/

  fn _apply(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    self._apply_val(txn, AVal::new(rvar, xvar, self._value().clone_read()));
  }

  fn _eval_recursive(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    if Some(txn) != self._txn() {
      for x_ in self.xs_.iter() {
        x_._eval_recursive(txn);
      }
      self._apply(txn, rvar, xvar);
    }
  }
}

impl<F, V, W> AOp<W> for FJoinOp<F, V, W> where V: 'static, RWVal<W>: IOVal + 'static {
  fn _make_value(&self) -> RWVal<W> {
    (self.ext.init)()
    //(self.ext.init)(self.fun.borrow_mut())
  }

  fn _value(&self) -> &RWVal<W> {
    &self.y
  }

  fn _make_tangent(&self) -> Val<W> {
    match self.ext.tangent {
      None => unimplemented!(),
      Some(ref tangent) => (tangent)(),
    }
  }

  fn tangent(&self) -> Val<W> {
    let mut tng_op = self.base.tng_op.borrow_mut();
    if tng_op.is_none() {
      *tng_op = Some(self._make_tangent());
    }
    tng_op.as_ref().unwrap().clone()
  }

  fn _pop_adjoint(&self, epoch: Epoch, /*filter: &Fn(&ANode) -> bool,*/ this: Val<W>, sink: &mut Sink) {
    if self.base.stack.pop(epoch) {
      match self.ext.adjoint {
        None => panic!(),
        Some(ref adjoint) => (adjoint)(this, sink),
      }
      for x_ in self.xs_.iter().rev() {
        x_.op()._pop_adjoint(epoch, x_.clone(), sink);
      }
    }
  }

  /*//fn adjoint(&self, sink: &mut Sink) -> (Rc<ANode>, Rc<AOp<W>>) {
  fn adjoint(&self, sink: &mut Sink) -> Val<W> {
    match sink.get_adj(self.var()) {
      None => panic!(),
      Some(adj) => adj,
    }
  }*/

  default fn _inplace(&self) -> Option<Val<W>> {
    None
  }

  fn _apply_val(&self, txn: Txn, val: AVal<W>) {
    (self.ext.apply)(txn, self.fun.borrow_mut(), val);
  }
}

impl<F, V> AOp<V> for FJoinOp<F, V, V> where RWVal<V>: IOVal + 'static {
  fn _inplace(&self) -> Option<Val<V>> {
    match self.ext.inplace {
      None => {}
      Some(ref inplace) => {
        for x_ in self.xs_.iter() {
          if x_.node()._walk().outdegree() == 1 {
            return Some((inplace)(x_.clone()));
          }
        }
      }
    }
    None
  }
}
