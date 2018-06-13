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

#![feature(const_fn)]
#![feature(core_intrinsics)]
#![feature(fn_traits)]
#![feature(get_type_id)]
#![feature(optin_builtin_traits)]
#![feature(slice_patterns)]
#![feature(specialization)]
//#![feature(trait_alias)]
#![feature(unboxed_closures)]

extern crate arithmetic;
extern crate arrayidx;
#[cfg(feature = "gpu")] extern crate cuda;
#[cfg(feature = "gpu")] extern crate cuda_blas;
#[cfg(feature = "gpu")] extern crate cuda_coll;
#[cfg(feature = "gpu")] extern crate cuda_dnn;
//extern crate float;
#[cfg(feature = "gpu")] extern crate gpudevicemem;
#[macro_use] extern crate lazy_static;
extern crate memarray;
extern crate parking_lot;
extern crate rand;
extern crate rng;
extern crate typemap;

use analysis::{LivenessAnalysis};
use ops::{OnesSrcOp, OnesSrcOpMaybeExt, SumJoinOp, SumJoinOpMaybeExt};
#[cfg(feature = "gpu")] use ops_gpu::{GPUMuxOp};

use arrayidx::{ArrayIndex};
#[cfg(feature = "gpu")] use gpudevicemem::{GPUDeviceId};
use memarray::{Array};
use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use typemap::{CloneMap, TypeMap};

use std::any::{Any};
use std::cell::{Cell, RefCell, RefMut};
use std::collections::{HashMap, HashSet};
//use std::collections::hash_map::{Entry};
//use std::ops::{Deref, DerefMut};
use std::rc::{Rc};
use std::sync::{Arc};
use std::sync::mpsc::{SyncSender, Receiver};

pub mod analysis;
pub mod config;
pub mod context;
pub mod ffi;
#[cfg(feature = "gpu")] pub mod io_gpu;
pub mod ops;
#[cfg(feature = "gpu")] pub mod ops_gpu;
#[cfg(feature = "mpi")] pub mod ops_mpi;
pub mod templates;
pub mod utils;

thread_local! {
  static UID_COUNTER: Cell<u64> = Cell::new(0);

  static WRAP_VAL_STACK: RefCell<Vec<Rc<Any>>> = RefCell::new(Vec::new());
}

pub fn gen_thread_local_uid() -> u64 {
  UID_COUNTER.with(|counter| {
    let prev_count = counter.get();
    counter.set(prev_count + 1);
    let next_count = counter.get();
    assert!(next_count != 0);
    next_count
  })
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Txn(u64);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Pass(u64);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RVar(u64);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RWVar(RVar);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeRef(u64);

/*#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ValRef(u64);*/

pub fn txn() -> Txn {
  Txn::default()
}

impl Default for Txn {
  fn default() -> Self {
    Txn(gen_thread_local_uid())
  }
}

pub fn pass() -> Pass {
  Pass::default()
}

impl Default for Pass {
  fn default() -> Self {
    Pass(gen_thread_local_uid())
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

/*impl Default for ValRef {
  fn default() -> Self {
    ValRef(gen_thread_local_uid())
  }
}*/

pub struct WalkStackEntry {
  pass:         Pass,
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
  pub fn push(&self, pass: Pass) -> bool {
    let mut entries = self.entries.borrow_mut();
    if entries.is_empty() || entries.last().unwrap().pass < pass {
      entries.push(WalkStackEntry{
        pass:           pass,
        push_degree:    1,
        pop_degree:     0,
      });
      true
    } else if entries.last().unwrap().pass > pass {
      panic!();
    } else if entries.last().unwrap().pass == pass {
      entries.last_mut().unwrap().push_degree += 1;
      false
    } else {
      unreachable!();
    }
  }

  pub fn pop(&self, pass: Pass) -> bool {
    let mut entries = self.entries.borrow_mut();
    assert!(!entries.is_empty());
    assert_eq!(entries.last().unwrap().pass, pass);
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

pub trait AnalysisTags {
  fn liveness(&self) -> Option<LivenessAnalysis>;
}

pub trait ANode {
  fn _walk(&self) -> &Walk;
  fn _io(&self) -> &IOVal;
  fn _analysis_tags(&self) -> &AnalysisTags { unimplemented!(); }

  fn _pred_fwd(&self, pred_buf: &mut Vec<Node>) { unimplemented!(); }
  fn _pred_rev(&self, pred_buf: &mut Vec<Node>) { unimplemented!(); }

  fn _push(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode));
  fn _pop(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode));

  fn _push_fwd(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar));
  fn _pop_rev(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar)) { unimplemented!(); }

  fn _txn(&self) -> Option<Txn>;
  fn _reset(&self);
  fn _release(&self);
  //fn _persist(&self, txn: Txn);
  //fn _prepare(&self, txn: Txn);
  //fn _cleanup(&self, txn: Txn);
  fn _eval_recursive(&self, txn: Txn, rvar: RVar, xvar: RWVar);
  fn _apply(&self, txn: Txn, rvar: RVar, xvar: RWVar);
}

pub trait AOp<V>: ANode {
  fn _pred_val_fwd(&self, pred_buf: &mut Vec<Rc<Any>>) { unimplemented!(); }
  fn _pred_val_rev(&self, pred_buf: &mut Vec<Rc<Any>>) { unimplemented!(); }

  fn _make_value(&self) -> RWVal<V>;
  fn _value(&self) -> &RWVal<V>;

  fn _build(&self, pred_vals: Vec<Rc<Any>>) -> Val<V> { unimplemented!(); }

  fn _apply_output(&self, txn: Txn, val: OVal<V>);

  fn _push_tangent(&self, pass: Pass, feedfwd: &mut FeedFwd) -> Val<V> { unimplemented!(); }
  //fn tangent(&self) -> Val<V>;

  fn _pop_adjoint(&self, pass: Pass, this: Val<V>, sink: &mut Sink) { unimplemented!(); }
  //fn adjoint(&self, sink: &mut Sink) -> Val<V>;

  // TODO
  //fn _substitute(&self, subs: Vec<(RWVar, Rc<Any>)>) -> Option<(Rc<ANode>, Rc<AOp<V>>)> { None }
  fn _inplace(&self) -> Option<Val<V>> { None }
}

pub trait IOVal {
  fn _serialize(&self, txn: Txn, rvar: RVar, dst: &mut Any);
  fn _deserialize(&self, txn: Txn, xvar: RWVar, src: &mut Any);

  fn _serialize_vec(&self, txn: Txn, rvar: RVar, off: usize, dst: &mut Any) -> usize;
  fn _deserialize_vec(&self, txn: Txn, rvar: RVar, xvar: RWVar, off: usize, src: &mut Any) -> usize;
}

pub trait IONodeExt {
  fn serialize(&self, txn: Txn, dst: &mut Any);
  fn deserialize(&self, txn: Txn, src: &mut Any);
}

pub trait VIONodeExt {
  fn _serialize_vec(&self, txn: Txn, off: usize, dst: &mut Any) -> usize;
  fn _deserialize_vec(&self, txn: Txn, off: usize, src: &mut Any) -> usize;

  fn serialize_vec(&self, txn: Txn, dst: &mut Any) {
    self._serialize_vec(txn, 0, dst);
  }

  fn deserialize_vec(&self, txn: Txn, src: &mut Any) {
    self._deserialize_vec(txn, 0, src);
  }
}

pub fn push_wrapper<Wrap>(wrapper: Wrap) -> WrapGuard where Wrap: Any + 'static {
  WRAP_VAL_STACK.with(|stack| {
    let mut stack = stack.borrow_mut();
    stack.push(Rc::new(wrapper));
    WrapGuard
  })
}

pub struct WrapGuard;

impl !Send for WrapGuard {}
impl !Sync for WrapGuard {}

impl Drop for WrapGuard {
  fn drop(&mut self) {
    WRAP_VAL_STACK.with(|stack| {
      let mut stack = stack.borrow_mut();
      assert!(stack.pop().is_some());
    });
  }
}

pub struct ChainWrap(pub Vec<Rc<Any>>);

pub struct InplaceWrap;

pub struct GPUMuxWrap {
  pub dev:  GPUDeviceId,
}

/*pub trait WrapVal: Any {
  fn wrap<V>(&self, val: Val<V>) -> Val<V> where Self: 'static;
}*/

pub trait WrapValExt<V> {
  //fn substitute(&self) -> Option<Val<V>>;
  fn inplace(&self) -> Option<Val<V>>;
}

#[cfg(feature = "gpu")]
pub trait GPUWrapValExt<V> {
  fn gpu_mux(&self, dev: GPUDeviceId) -> Val<V>;
}

pub trait SendValExt<V> {
  fn send_thread(&self) -> (SendThreadVal, RecvThreadVal);
  fn send_serial(&self) -> ();
}

pub struct SendThreadVal {
  tx:   SyncSender<()>,
}

impl SendThreadVal {
  pub fn send(self) {
    // TODO
    unimplemented!();
  }
}

pub struct RecvThreadVal {
  rx:   Receiver<()>,
}

impl RecvThreadVal {
  pub fn recv(self) -> () /*Val<V>*/ {
    // TODO
    unimplemented!();
  }
}

pub fn dup_cache<V>(root: Val<V>, cached: &HashSet<RWVar>) -> Val<V> where V: 'static {
  // TODO: need to recursively rebuild the graph; need to store predecessors.
  let pass = Pass::default();
  root._push_fwd(None, pass, &mut |node, rvar, xvar| {
    // TODO
    if cached.contains(&xvar) {
      // TODO
    } else {
      let mut preds = vec![];
      node._pred_fwd(&mut preds);
    }
  });
  root._pop_rev(None, pass, &mut |_, _, _| {});
  // TODO
  unimplemented!();
}

pub fn rep_cache<V>(root: Val<V>, reps: usize, cached: &HashSet<RWVar>) -> Vec<Val<V>> where V: 'static {
  let mut new_roots = vec![];
  for rep in 0 .. reps {
    if rep == 0 {
      new_roots.push(root.clone());
    } else {
      new_roots.push(dup_cache(root.clone(), cached));
    }
  }
  new_roots
}

pub fn gpu_mux<V>(roots: Vec<Val<V>>) -> Vec<Val<V>> {
  // TODO
  unimplemented!();
}

pub struct Torus1d<V=()> {
  idxs: Vec<usize>,
  data: Vec<V>,
}

pub type Ring<V=()> = Torus1d<V>;

impl Torus1d {
  pub fn to(r: usize) -> Self {
    // TODO
    unimplemented!();
  }
}

pub struct Torus2d<V=()> {
  idxs: Vec<[usize; 2]>,
  data: Vec<V>,
}

pub type Torus<V=()> = Torus2d<V>;

impl Torus2d {
  pub fn to(r0: usize, r1: usize) -> Self {
    // TODO
    unimplemented!();
  }
}

pub struct Torus3d<V=()> {
  idxs: Vec<[usize; 3]>,
  data: Vec<V>,
}

pub struct Node {
  node: Rc<ANode>,
  xvar: RWVar,
  rvar: RVar,
  name: Option<String>,
}

impl Clone for Node {
  fn clone(&self) -> Self {
    let rvar = RVar::default();
    Node{
      node: self.node.clone(),
      xvar: self.xvar,
      rvar: rvar,
      name: self.name.clone(),
    }
  }
}

impl Node {
  pub fn _node(&self) -> &ANode {
    &*self.node
  }

  pub fn _push_fwd(&self, stop_txn: Option<Txn>, pass: Pass, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    self.node._push_fwd(stop_txn, pass, self.rvar, self.xvar, apply);
  }

  pub fn _pop_rev(&self, stop_txn: Option<Txn>, pass: Pass, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    self.node._pop_rev(stop_txn, pass, self.rvar, self.xvar, apply);
  }

  pub fn _apply(&self, txn: Txn) {
    self.node._apply(txn, self.rvar, self.xvar);
  }

  pub fn _eval_recursive(&self, txn: Txn) {
    self.node._eval_recursive(txn, self.rvar, self.xvar);
  }

  pub fn persist(&self, txn: Txn) {
    // TODO
    unimplemented!();
    //self.node._persist(txn, self.xvar);
  }

  pub fn eval(&self, txn: Txn) {
    self._eval_recursive(txn);
    self._apply(txn);
  }

  pub fn var(&self) -> RWVar {
    self.xvar
  }
}

impl IONodeExt for Node {
  fn serialize(&self, txn: Txn, dst: &mut Any) {
    // FIXME
    self.node._io()._serialize(txn, self.rvar, dst);
  }

  fn deserialize(&self, txn: Txn, src: &mut Any) {
    self.node._io()._deserialize(txn, self.xvar, src);
  }
}

impl VIONodeExt for Node {
  fn _serialize_vec(&self, txn: Txn, off: usize, dst: &mut Any) -> usize {
    self.node._io()._serialize_vec(txn, self.rvar, off, dst)
  }

  fn _deserialize_vec(&self, txn: Txn, off: usize, src: &mut Any) -> usize {
    self.node._io()._deserialize_vec(txn, self.rvar, self.xvar, off, src)
  }
}

pub struct Val<V> {
  node: Rc<ANode>,
  op:   Rc<AOp<V>>,
  //mode: WriteMode,
  xvar: RWVar,
  rvar: RVar,
  name: Option<String>,
}

impl<V> Clone for Val<V> {
  fn clone(&self) -> Val<V> {
    let rvar = RVar::default();
    Val{
      node: self.node.clone(),
      op:   self.op.clone(),
      xvar: self.xvar,
      rvar: rvar,
      name: self.name.clone(),
    }
  }
}

impl<V> Val<V> where V: 'static {
  pub fn nowrap<Op>(op: Rc<Op>, xvar: RWVar) -> Self where Op: AOp<V> + 'static {
    let rvar = RVar::default();
    Val{
      node: op.clone(),
      op:   op,
      xvar: xvar,
      rvar: rvar,
      name: None,
    }
  }

  pub fn from<Op>(op: Rc<Op>) -> Self where Op: AOp<V> + 'static {
    let rvar = RVar::default();
    let val = Val{
      node: op.clone(),
      op:   op,
      xvar: RWVar(rvar),
      rvar: rvar,
      name: None,
    };
    let val = WRAP_VAL_STACK.with(|stack| {
      let stack = stack.borrow();
      if stack.is_empty() {
        val
      } else {
        let wrapper = stack[stack.len() - 1].clone();
        if let Some(_wrapper) = wrapper.downcast_ref::<InplaceWrap>() {
          val.inplace().unwrap_or(val)
        } else if let Some(wrapper) = wrapper.downcast_ref::<GPUMuxWrap>() {
          val.gpu_mux(wrapper.dev)
        } else {
          // TODO: warn on unsupported wrapper.
          val
        }
      }
    });
    val
  }

  pub fn to_node(&self) -> Node {
    Node{
      node: self.node.clone(),
      xvar: self.xvar,
      // NOTE: Should the node corresponding to a val share the same varkeys?
      rvar: self.rvar,
      name: self.name.clone(),
    }
  }

  pub fn into_node(self) -> Node {
    Node{
      node: self.node.clone(),
      xvar: self.xvar,
      rvar: self.rvar,
      name: self.name.clone(),
    }
  }

  /*pub fn downgrade(&self) -> OVal<V> {
    OVal{
      rvar: self.rvar,
      xvar: self.xvar,
      xval: self.op._value()._clone(),
    }
  }*/

  pub fn _exact_clone(&self) -> Val<V> {
    Val{
      node: self.node.clone(),
      op:   self.op.clone(),
      xvar: self.xvar,
      rvar: self.rvar,
      name: self.name.clone(),
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
      name: self.name.clone(),
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
      name: self.name.clone(),
    }
  }

  pub fn named(&self, name: &str) -> Val<V> {
    let rvar = RVar::default();
    Val{
      node: self.node.clone(),
      op:   self.op.clone(),
      xvar: self.xvar,
      rvar: rvar,
      name: Some(name.to_owned()),
    }
  }

  pub fn _node(&self) -> &ANode {
    &*self.node
  }

  pub fn _op(&self) -> &AOp<V> {
    &*self.op
  }

  pub fn _push_fwd(&self, stop_txn: Option<Txn>, pass: Pass, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    self.op._push_fwd(stop_txn, pass, self.rvar, self.xvar, apply);
  }

  pub fn _pop_rev(&self, stop_txn: Option<Txn>, pass: Pass, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    self.op._pop_rev(stop_txn, pass, self.rvar, self.xvar, apply);
  }

  pub fn _apply(&self, txn: Txn) {
    self.op._apply(txn, self.rvar, self.xvar);
  }

  pub fn _eval_recursive(&self, txn: Txn) {
    self.op._eval_recursive(txn, self.rvar, self.xvar);
  }

  pub fn eval(&self, txn: Txn) {
    self._eval_recursive(txn);
    self._apply(txn);
  }

  pub fn var(&self) -> RWVar {
    self.xvar
  }

  pub fn reset(&self) {
    self.op._reset();
  }

  pub fn release(&self) {
    self.op._release();
  }

  pub fn persist(&self, txn: Txn) {
    self.op._value().persist(txn, self.xvar);
  }

  pub fn write(&self, txn: Txn) -> Option<(WriteCap, WriteToken)> {
    self.op._value().write(txn, self.xvar)
  }

  pub fn get(&self, txn: Txn) -> RwLockReadGuard<V> {
    //self.eval(txn);
    self.op._value().get(txn, self.rvar)
  }

  pub fn get_mut(&self, txn: Txn, token: WriteToken) -> RwLockWriteGuard<V> {
    //self.eval(txn);
    self.op._value().get_mut(txn, self.xvar, token)
  }

  pub fn set<F: FnOnce(RwLockWriteGuard<V>)>(&self, txn: Txn, f: F) {
    self.op._value().set(txn, self.xvar, f);
  }

  pub fn _make_value(&self) -> RWVal<V> {
    self.op._make_value()
  }

  pub fn _push_tangent(&self, pass: Pass, feedfwd: &mut FeedFwd) -> Val<V> {
    self.op._push_tangent(pass, feedfwd)
  }

  pub fn tangent(&self, feedfwd: &mut FeedFwd) -> Val<V> {
    // TODO
    unimplemented!();
  }

  pub fn _pop_adjoint(&self, pass: Pass, sink: &mut Sink) {
    self.op._pop_adjoint(pass, self.clone(), sink);
  }

  pub fn adjoint(&self, sink: &mut Sink) -> Option<Val<V>> {
    sink.get_adj::<V>(self.var())
  }

  pub fn put_adjoint(&self, adj: Val<V>, sink: &mut Sink) {
    sink.put_adj::<V>(self.var(), adj);
  }
}

impl<V> IONodeExt for Val<V> where V: 'static {
  fn serialize(&self, txn: Txn, dst: &mut Any) {
    self.op._io()._serialize(txn, self.rvar, dst);
  }

  fn deserialize(&self, txn: Txn, src: &mut Any) {
    self.op._io()._deserialize(txn, self.xvar, src);
  }
}

impl<V> VIONodeExt for Val<V> where V: 'static {
  fn _serialize_vec(&self, txn: Txn, off: usize, dst: &mut Any) -> usize {
    self.op._io()._serialize_vec(txn, self.rvar, off, dst)
  }

  fn _deserialize_vec(&self, txn: Txn, off: usize, src: &mut Any) -> usize {
    self.op._io()._deserialize_vec(txn, self.rvar, self.xvar, off, src)
  }
}

pub struct OVal<V> {
  val:  RWVal<V>,
  rvar: RVar,
  xvar: RWVar,
}

impl<V> OVal<V> where V: 'static {
  pub fn new(rvar: RVar, xvar: RWVar, val: RWVal<V>) -> Self {
    OVal{
      val:  val,
      rvar: rvar,
      xvar: xvar,
    }
  }

  pub fn var(&self) -> RWVar {
    self.xvar
  }

  pub fn persist(&self, txn: Txn) {
    self.val.persist(txn, self.xvar);
  }

  pub fn write(&self, txn: Txn) -> Option<(WriteCap, WriteToken)> {
    self.val.write(txn, self.xvar)
  }

  pub fn get(&self, txn: Txn) -> RwLockReadGuard<V> {
    self.val.get(txn, self.rvar)
  }

  pub fn get_mut(&self, txn: Txn, token: WriteToken) -> RwLockWriteGuard<V> {
    self.val.get_mut(txn, self.xvar, token)
  }

  pub fn set<F: FnOnce(RwLockWriteGuard<V>)>(&self, txn: Txn, f: F) {
    self.val.set(txn, self.xvar, f);
  }
}

pub struct FeedFwd {
  tng_map:  HashMap<RWVar, (Node, Rc<Any>)>,
}

pub struct Sink {
  frozen:   HashSet<RWVar>,
  adj_map:  HashMap<RWVar, Vec<(Node, Rc<Any>)>>,
  join_map: HashMap<RWVar, (Node, Rc<Any>)>,
}

impl Sink {
  pub fn from<V>(sink_: Val<V>) -> Self where V: 'static {
    // Add a "ones" adjoint op corresponding to `sink_`.
    let sink_adj = match <OnesSrcOp as OnesSrcOpMaybeExt<V>>::maybe_build_like(sink_.clone()) {
      None => unimplemented!("FATAL: Sink: missing `ones` builder for sink val"),
      Some(adj) => adj,
    };
    Sink::with_adj(sink_, sink_adj)
  }

  pub fn with_adj<V>(sink_: Val<V>, sink_adj: Val<V>) -> Self where V: 'static {
    let mut sink = Sink{
      frozen:   HashSet::new(),
      adj_map:  HashMap::new(),
      join_map: HashMap::new(),
    };
    sink_.put_adjoint(sink_adj, &mut sink);
    let p = pass();
    sink_._push_fwd(None, p, &mut |_node, _rvar, _xvar| {});
    sink_._pop_adjoint(p, &mut sink);
    sink
  }

  pub fn get_adj_node(&mut self, var: RWVar) -> Option<Node> {
    self.frozen.insert(var);
    if self.adj_map.contains_key(&var) {
      let adjs = self.adj_map.get(&var).unwrap();
      match adjs.len() {
        0 => {}
        1 => {
          return Some(adjs[0].0.clone());
        }
        _ => {
          if self.join_map.contains_key(&var) {
            let &(ref join_node, _) = self.join_map.get(&var).unwrap();
            return Some(join_node.clone());
          } else {
            // TODO: need an untyped sum op builder.
            unimplemented!();
            /*let adj_nodes: Vec<_> = adjs.iter().map(|&(ref n, _)| n.clone()).collect();
            let join = <SumJoinOp as SumJoinOpMaybeExt<V>>::maybe_build(adj_nodes).unwrap();
            self.join_map.insert(var, (join.clone().into_node(), Rc::new(join.clone())));
            return Some(join.into_node());*/
          }
        }
      }
    }
    None
  }

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
            self.join_map.insert(var, (join.clone().into_node(), Rc::new(join.clone())));
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
      let adjs = self.adj_map.get_mut(&var).unwrap();
      adjs.push((adj_op.clone().into_node(), Rc::new(adj_op)));
    } else {
      self.adj_map.insert(var, vec![(adj_op.clone().into_node(), Rc::new(adj_op))]);
    }
  }
}

pub struct OpBase {
  ref_:     NodeRef,
  stack:    WalkStack,
  tags:     CloneMap,
  //tng_op:   RefCell<Option<Val<V>>>,
}

impl Default for OpBase {
  fn default() -> Self {
    OpBase{
      ref_:     NodeRef::default(),
      stack:    WalkStack::default(),
      tags:     TypeMap::custom(),
      //tng_op:   RefCell::new(None),
    }
  }
}

impl AnalysisTags for OpBase {
  fn liveness(&self) -> Option<LivenessAnalysis> {
    self.tags.get::<LivenessAnalysis>().map(|x| x.clone())
  }
}

#[derive(Clone, Default)]
pub struct NodeVec {
  nodes:    Vec<Node>,
}

impl NodeVec {
  pub fn from(nodes: Vec<Node>) -> Self {
    NodeVec{nodes: nodes}
  }

  pub fn reversed(&self) -> Self {
    let mut rev = self.clone();
    rev.nodes.reverse();
    rev
  }

  pub fn adjoints(&self, sink: &mut Sink) -> Self {
    let mut adjs = self.clone();
    for n in self.nodes.iter() {
      match sink.get_adj_node(n.var()) {
        None => panic!(),
        Some(adj_n) => adjs.push(adj_n),
      }
    }
    adjs
  }

  pub fn push(&mut self, node: Node) {
    self.nodes.push(node);
  }

  pub fn push_val<A: 'static>(&mut self, val: Val<A>) {
    self.nodes.push(val.into_node());
  }

  pub fn extend(&mut self, other: &NodeVec) {
    self.nodes.extend_from_slice(&other.nodes);
  }

  pub fn persist(&self, txn: Txn) {
    for n in self.nodes.iter() {
      n.persist(txn);
    }
  }

  pub fn eval(&self, txn: Txn) {
    for n in self.nodes.iter() {
      n.eval(txn);
    }
  }
}

impl VIONodeExt for NodeVec {
  fn _serialize_vec(&self, txn: Txn, mut off: usize, dst: &mut Any) -> usize {
    for node in self.nodes.iter() {
      off = node._serialize_vec(txn, off, dst);
    }
    off
  }

  fn _deserialize_vec(&self, txn: Txn, mut off: usize, src: &mut Any) -> usize {
    for node in self.nodes.iter() {
      off = node._deserialize_vec(txn, off, src);
    }
    off
  }
}

pub struct FlatIO<Buf> {
  buffer:   Buf,
  offset:   usize,
}

impl<Buf> FlatIO<Buf> {
  pub fn new(buffer: Buf) -> Self {
    FlatIO{
      buffer:   buffer,
      offset:   0,
    }
  }

  pub fn reset(&mut self) {
    self.offset = 0;
  }

  pub fn take(self) -> Buf {
    self.buffer
  }
}

pub struct ArrayIO<Arr> where Arr: Array {
  array:    Arr,
  offset:   <Arr as Array>::Idx,
}

impl<Arr> ArrayIO<Arr> where Arr: Array {
  pub fn new(array: Arr) -> Self {
    ArrayIO{
      array:    array,
      offset:   <<Arr as Array>::Idx as ArrayIndex>::zero(),
    }
  }

  pub fn take(self) -> Arr {
    self.array
  }
}

/*pub struct BatchArrayIO<Arr> where Arr: Array {
  array:    Arr,
  offset:   <Arr as Array>::Idx,
  eloffset: usize,
}*/

#[derive(Default)]
pub struct LazyConst<T> where T: Copy {
  inner:    Cell<Option<T>>,
}

impl<T> LazyConst<T> where T: Copy {
  pub fn get(&self) -> T {
    match self.inner.get() {
      None => panic!("LazyConst was never initialized with set_once"),
      Some(value) => value,
    }
  }

  pub fn set_once<F>(&self, f: F) -> T where F: FnOnce() -> T {
    if self.inner.get().is_none() {
      self.inner.set(Some(f()));
    }
    self.get()
  }
}

#[derive(Clone)]
pub struct TCell<T> where T: Copy {
  inner:    Rc<RefCell<TCellInner<T>>>,
}

impl<T> Default for TCell<T> where T: Copy + Default {
  fn default() -> Self {
    Self::new(T::default())
  }
}

impl<T> TCell<T> where T: Copy {
  pub fn new(init_value: T) -> Self {
    TCell{inner: Rc::new(RefCell::new(TCellInner::new(init_value)))}
  }

  pub fn txn(&self) -> Option<Txn> {
    self.inner.borrow_mut().txn()
  }

  pub fn reset(&self) {
    self.inner.borrow_mut().reset();
  }

  pub fn persist(&self, txn: Txn) {
    self.inner.borrow_mut().persist(txn);
  }

  pub fn always_get(&self) -> T {
    self.inner.borrow_mut().always_get()
  }

  pub fn get(&self, txn: Txn) -> T {
    self.inner.borrow_mut().get(txn)
  }

  pub fn rollback(&self, txn: Txn) {
    self.inner.borrow_mut().rollback(txn);
  }

  pub fn propose<F>(&self, txn: Txn, f: F) -> T where F: Fn(T) -> T {
    self.inner.borrow_mut().propose(txn, f)
  }
}

pub struct TCellInner<T> where T: Copy {
  init:     T,
  state:    (Option<Txn>, T),
  proposal: Option<(Txn, T)>,
}

impl<T> TCellInner<T> where T: Copy {
  pub fn new(init_value: T) -> Self {
    TCellInner{
      init:     init_value,
      state:    (None, init_value),
      proposal: None,
    }
  }

  pub fn txn(&mut self) -> Option<Txn> {
    self.commit();
    self.state.0
  }

  pub fn reset(&mut self) {
    self.state = (None, self.init);
    self.proposal = None;
  }

  pub fn persist(&mut self, txn: Txn) {
    // TODO
    unimplemented!();
  }

  pub fn always_get(&mut self) -> T {
    self.commit();
    match self.state {
      (Some(_), value) => value,
      _ => self.init,
    }
  }

  pub fn get(&mut self, txn: Txn) -> T {
    if self.proposal.is_some() {
      self.conditional_commit(txn);
    }
    assert!(self.proposal.is_none(),
        "cannot read from TCell which has an uncommitted proposal");
    let new_read_txn = match self.state.0 {
      None => true,
      Some(curr_txn) => if curr_txn < txn {
        true
      } else if curr_txn == txn {
        false
      } else {
        panic!("causal violation, probably a bug in your code");
      }
    };
    assert!(!new_read_txn);
    self.state.1
  }

  pub fn propose<F>(&mut self, txn: Txn, f: F) -> T where F: FnOnce(T) -> T {
    let new_write_txn = match self.proposal {
      None => true,
      Some((prop_txn, _)) => if prop_txn < txn {
        true
      } else if prop_txn == txn {
        false
      } else {
        panic!("causal violation, probably a bug in your code");
      }
    };
    if new_write_txn {
      self.commit();
    }
    assert!(self.proposal.is_none(),
        "commit mysteriously failed");
    let prev_value = match self.state.0 {
      None => self.init,
      Some(curr_txn) => if curr_txn <= txn {
        self.state.1
      } else {
        panic!("causal violation, probably a bug in your code");
      }
    };
    if new_write_txn {
      let next_value = f(prev_value);
      //println!("DEBUG: TCell: propose: setting proposal value");
      //println!("DEBUG: TCell: propose: setting proposal value: {:?} -> {:?}", prev_value, next_value);
      self.proposal = Some((txn, next_value));
    }
    prev_value
  }

  pub fn rollback(&mut self, txn: Txn) {
    if let Some(curr_txn) = self.state.0 {
      assert!(curr_txn != txn,
          "unable to rollback, as proposal was already committed");
    }
    if let Some((prop_txn, _)) = self.proposal {
      if prop_txn == txn {
        self.proposal = None;
      }
    }
  }

  pub fn conditional_commit(&mut self, txn: Txn) {
    if let Some((prop_txn, prop_value)) = self.proposal {
      if prop_txn == txn {
        self.state = (Some(prop_txn), prop_value);
        self.proposal = None;
      }
    }
  }

  pub fn commit(&mut self) {
    if let Some((prop_txn, prop_value)) = self.proposal {
      self.state = (Some(prop_txn), prop_value);
      self.proposal = None;
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WriteCap {
  Assign,
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
  l_consumers:  Mutex<HashSet<RVar>>,
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
      l_consumers:  Mutex::new(HashSet::new()),
      d_consumers:  HashSet::new(),
      l_producers:  HashSet::new(),
      d_producers:  HashSet::new(),
      data:         None,
    }
  }
}

impl<T> RWValBuf<T> {
}

pub struct ShareableRWVal<T> {
  alloc:    Arc<Fn(Txn) -> T>,
  buf:      Arc<RwLock<RWValBuf<T>>>,
}

impl<T> ShareableRWVal<T> where T: 'static {
  pub fn into_val(self) -> RWVal<T> {
    // TODO
    unimplemented!();
  }
}

pub struct RWVal<T> {
  //ref_:     ValRef,
  alloc:    Arc<Fn(Txn) -> T>,
  buf:      Arc<RwLock<RWValBuf<T>>>,
  borrow:   (),
}

impl<T> IOVal for RWVal<T> where T: 'static {
  default fn _serialize(&self, txn: Txn, rvar: RVar, dst: &mut Any) {
    unimplemented!();
  }

  default fn _deserialize(&self, txn: Txn, xvar: RWVar, src: &mut Any) {
    unimplemented!();
  }

  default fn _serialize_vec(&self, txn: Txn, rvar: RVar, off: usize, dst: &mut Any) -> usize {
    unimplemented!();
  }

  default fn _deserialize_vec(&self, txn: Txn, rvar: RVar, xvar: RWVar, off: usize, src: &mut Any) -> usize {
    unimplemented!();
  }
}

impl<T> RWVal<T> where T: 'static {
  pub fn from(alloc: Arc<Fn(Txn) -> T>) -> Self {
    let buf = Arc::new(RwLock::new(RWValBuf::default()));
    RWVal{
      //ref_:     ValRef::default(),
      alloc:    alloc,
      buf:      buf,
      borrow:   (),
    }
  }

  /*pub fn shared(alloc: Arc<Fn(Txn) -> T>) -> Self {
    let buf = Arc::new(RwLock::new(RWValBuf::default()));
    RWVal{
      alloc:    alloc,
      buf:      RwBox::Shared(buf),
      borrow:   (),
    }
  }*/

  pub fn share(&self) -> Option<ShareableRWVal<T>> {
    Some(ShareableRWVal{
      alloc:  self.alloc.clone(),
      buf:    self.buf.clone(),
    })
  }

  pub fn _clone(&self) -> Self {
    RWVal{
      //ref_:     self.ref_,
      alloc:    self.alloc.clone(),
      buf:      self.buf.clone(),
      borrow:   (),
    }
  }

  pub fn _set_accumulate(&self) {
    let mut buf = self.buf.write();
    match buf.mode {
      WriteMode::Exclusive => {
        buf.mode = WriteMode::Accumulate;
      }
      WriteMode::Accumulate => {}
      _ => panic!(),
    }
  }

  pub fn _set_clobber(&self) {
    let mut buf = self.buf.write();
    match buf.mode {
      WriteMode::Exclusive => {
        buf.mode = WriteMode::Clobber;
      }
      WriteMode::Clobber => {}
      _ => panic!(),
    }
  }

  pub fn txn(&self) -> Option<Txn> {
    let buf = self.buf.read();
    buf.curr_txn
  }

  pub fn reset(&self) {
    let mut buf = self.buf.write();
    buf.curr_txn = None;
    buf.l_consumers.lock().clear();
    buf.d_consumers.clear();
    buf.l_producers.clear();
    buf.d_producers.clear();
  }

  pub fn release(&self) {
    let mut buf = self.buf.write();
    buf.curr_txn = None;
    buf.l_consumers.lock().clear();
    buf.d_consumers.clear();
    buf.l_producers.clear();
    buf.d_producers.clear();
    buf.data = None;
  }

  pub fn persist(&self, txn: Txn, /*rvar: RVar,*/ xvar: RWVar) {
    let mut buf = self.buf.write();

    let new_txn = buf.curr_txn.is_none() || buf.curr_txn.unwrap() != txn;
    if new_txn {
      buf.curr_txn = Some(txn);
      buf.l_consumers.lock().clear();
      buf.d_consumers.clear();
      buf.l_producers.clear();
      buf.d_producers.clear();
    }

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
    assert!(buf.l_consumers.lock().is_empty(),
        "`persist` should be called before reads");
    buf.l_producers.insert(xvar);
  }

  pub fn write(&self, txn: Txn, xvar: RWVar) -> Option<(WriteCap, WriteToken)> {
    let mut buf = self.buf.write();

    let new_txn = buf.curr_txn.is_none() || buf.curr_txn.unwrap() != txn;
    if new_txn {
      buf.curr_txn = Some(txn);
      buf.l_consumers.lock().clear();
      buf.d_consumers.clear();
      buf.l_producers.clear();
      buf.d_producers.clear();
    }

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
        assert!(buf.l_consumers.lock().is_empty(),
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
        assert!(buf.l_consumers.lock().is_empty(),
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
            ref l_consumers,
            ref mut d_consumers,
            ref mut l_producers,
            ref mut d_producers,
            ..} = &mut *buf;
        let mut l_consumers = l_consumers.lock();
        d_consumers.extend(l_consumers.drain());
        d_producers.extend(l_producers.drain());
      }
    }

    let first = buf.l_producers.is_empty();
    let cap = match (buf.mode, first) {
      (WriteMode::Accumulate, false) => WriteCap::Accumulate,
      (_, true) => WriteCap::Assign,
      _ => unreachable!(),
    };
    buf.l_producers.insert(xvar);
    Some((cap, WriteToken{xvar: xvar, first: first, borrow: &self.borrow}))
  }

  pub fn get(&self, txn: Txn, rvar: RVar) -> RwLockReadGuard<T> {
    let buf = self.buf.read();

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
    buf.l_consumers.lock().insert(rvar);

    assert!(buf.data.is_some(),
        "attempting a read on empty data");

    RwLockReadGuard::map(buf, |buf| buf.data.as_ref().unwrap())
  }

  pub fn get_mut(&self, txn: Txn, xvar: RWVar, token: WriteToken) -> RwLockWriteGuard<T> {
    let mut buf = self.buf.write();
    assert_eq!(xvar, token.xvar);

    let mut valid_txn = false;
    if let Some(curr_txn) = buf.curr_txn {
      if curr_txn == txn {
        valid_txn = true;
      }
    }
    assert!(valid_txn,
        "attempting a write with an invalid txn (did you forget to `write`?)");

    assert!(buf.l_consumers.lock().is_empty(),
        "attempting a write-after-read (check your `get` and `get_mut` order)");
    assert!(!buf.d_producers.contains(&xvar),
        "attempting an invalid write (the value has been clobbered)");
    assert!(buf.l_producers.contains(&xvar),
        "attempting an invalid write (did you forget to `write`?)");

    if buf.data.is_none() {
      buf.data = Some((self.alloc)(txn));
    }

    RwLockWriteGuard::map(buf, |buf| buf.data.as_mut().unwrap())
  }

  pub fn set<F>(&self, txn: Txn, xvar: RWVar, f: F) where F: FnOnce(RwLockWriteGuard<T>) {
    if let Some((cap, token)) = self.write(txn, xvar) {
      match cap {
        WriteCap::Assign => {
          f(self.get_mut(txn, xvar, token));
        }
        _ => unimplemented!(),
      }
    }
  }
}

pub struct OpExt<F, V> {
  make_val: Box<Fn(RefMut<F>) -> RWVal<V>>,
  //prepare:  Option<Box<Fn(Txn, RefMut<F>)>>,
  //cleanup:  Option<Box<Fn(Txn, RefMut<F>)>>,
  apply:    Box<Fn(Txn, RefMut<F>, OVal<V>)>,
  build:    Option<Box<Fn(Vec<Rc<Any>>) -> Val<V>>>,
  //tangent:  Option<Box<Fn() -> Val<V>>>,
  tangent:  Option<Box<Fn(Pass, RefMut<F>, &mut FeedFwd) -> Val<V>>>,
  //adjoint:  Option<Box<Fn(Val<V>, &mut Sink)>>,
  adjoint:  Option<Box<Fn(Pass, Val<V>, RefMut<F>, &mut Sink)>>,
  inplace:  Option<Box<Fn(Val<V>) -> Val<V>>>,
}

/*impl<F, V> Clone for OpExt<F, V> {
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
}*/

impl<V> WrapValExt<V> for Val<V> where V: 'static {
  fn inplace(&self) -> Option<Val<V>> {
    self._op()._inplace()
  }
}

/*#[cfg(not(feature = "gpu"))]
impl<V> GPUWrapValExt<V> for Val<V> where V: 'static {
  fn gpu_mux(&self, _dev: GPUDeviceId) -> Val<V> {
    // TODO: cant quite just impl a no-op, since it still uses the
    // `GPUDeviceId` type.
  }
}*/

#[cfg(feature = "gpu")]
impl<V> GPUWrapValExt<V> for Val<V> where V: 'static {
  fn gpu_mux(&self, dev: GPUDeviceId) -> Val<V> {
    let this = self.clone();
    let wrap_ext: OpExt<GPUMuxOp<V>, V> = GPUMuxOp::<V>::build_ext();
    let wrap_cfg: GPUMuxOp<V> = GPUMuxOp{
      dev:  dev,
      val:  this._exact_clone(),
    };
    let wrap_op = FSrcWrapOp::new(wrap_cfg, wrap_ext, this._exact_clone());
    // FIXME: `nowrap` only makes sense here without join wrappers.
    Val::nowrap(Rc::new(wrap_op), self.var())
  }
}

pub struct FSrcWrapOp<F, V> {
  base: OpBase,
  ext:  OpExt<F, V>,
  cfg:  RefCell<F>,
  ctrl: Vec<Node>,
  val_: Val<V>,
}

impl<F, V> FSrcWrapOp<F, V> {
  pub fn new(cfg: F, ext: OpExt<F, V>, val_: Val<V>) -> Self {
    let cfg = RefCell::new(cfg);
    FSrcWrapOp{
      base: OpBase::default(),
      ext:  ext,
      cfg:  cfg,
      ctrl: vec![],
      val_: val_,
    }
  }
}

impl<F, V> ANode for FSrcWrapOp<F, V> where RWVal<V>: IOVal + 'static {
  fn _walk(&self) -> &Walk {
    &self.base.stack
  }

  fn _io(&self) -> &IOVal {
    self.val_._op()._io()
  }

  fn _analysis_tags(&self) -> &AnalysisTags {
    &self.base
  }

  fn _pred_fwd(&self, _pred_buf: &mut Vec<Node>) {
  }

  fn _pred_rev(&self, _pred_buf: &mut Vec<Node>) {
  }

  fn _push(&self, _stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    unimplemented!();
  }

  fn _pop(&self, _stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    unimplemented!();
  }

  fn _push_fwd(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    if self.base.stack.push(pass) {
      // TODO: `stop_txn`?
      self.val_._push_fwd(stop_txn, pass, apply);
      apply(self, rvar, xvar);
    }
  }

  fn _txn(&self) -> Option<Txn> {
    self.val_._op()._txn()
  }

  fn _reset(&self) {
    self.val_.reset();
  }

  fn _release(&self) {
    self.val_.release();
  }

  fn _apply(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    //println!("DEBUG: FWrap: apply");
    self._apply_output(txn, OVal::new(rvar, xvar, self._value()._clone()));
  }

  fn _eval_recursive(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    //println!("DEBUG: FWrap: eval recursive");
    if Some(txn) != self._txn() {
      for node in self.ctrl.iter() {
        node.eval(txn);
      }
      //println!("DEBUG: FWrap: inner eval recursive");
      self.val_._eval_recursive(txn);
    }
  }
}

impl<F, V> AOp<V> for FSrcWrapOp<F, V> where RWVal<V>: IOVal + 'static {
  fn _make_value(&self) -> RWVal<V> {
    (self.ext.make_val)(self.cfg.borrow_mut())
  }

  fn _value(&self) -> &RWVal<V> {
    self.val_._op()._value()
  }

  fn _apply_output(&self, txn: Txn, val: OVal<V>) {
    (self.ext.apply)(txn, self.cfg.borrow_mut(), val);
  }

  fn _push_tangent(&self, pass: Pass, feedfwd: &mut FeedFwd) -> Val<V> {
    match self.ext.tangent {
      None => unimplemented!(),
      Some(ref tangent) => (tangent)(pass, self.cfg.borrow_mut(), feedfwd),
    }
  }

  fn _pop_adjoint(&self, pass: Pass, this: Val<V>, sink: &mut Sink) {
    if self.base.stack.pop(pass) {
      match self.ext.adjoint {
        None => {}
        Some(ref adjoint) => (adjoint)(pass, this, self.cfg.borrow_mut(), sink),
      }
    }
  }
}

pub struct FSrcOp<F, V> {
  base: OpBase,
  ext:  OpExt<F, V>,
  cfg:  RefCell<F>,
  ctrl: Vec<Node>,
  val:  RWVal<V>,
}

impl<F, V> FSrcOp<F, V> {
  pub fn new(cfg: F, ext: OpExt<F, V>) -> Self {
    let cfg = RefCell::new(cfg);
    let val = (ext.make_val)(cfg.borrow_mut());
    FSrcOp{
      base: OpBase::default(),
      ext:  ext,
      cfg:  cfg,
      ctrl: vec![],
      val:  val,
    }
  }

  /*pub fn with_val(state: F, ext: OpExt<F, V>, val: RWVal<V>) -> Self {
    let state = RefCell::new(state);
    FSrcOp{
      base: OpBase::default(),
      ext:  ext,
      cfg:  state,
      ctrl: vec![],
      val:  val,
    }
  }*/
}

impl<F, V> ANode for FSrcOp<F, V> where RWVal<V>: IOVal + 'static {
  fn _walk(&self) -> &Walk {
    &self.base.stack
  }

  fn _io(&self) -> &IOVal {
    &self.val
  }

  fn _analysis_tags(&self) -> &AnalysisTags {
    &self.base
  }

  fn _pred_fwd(&self, _pred_buf: &mut Vec<Node>) {
  }

  fn _pred_rev(&self, _pred_buf: &mut Vec<Node>) {
  }

  fn _push(&self, _stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.push(pass) {
      apply(self);
    }
  }

  fn _pop(&self, _stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.pop(pass) {
      apply(self);
    }
  }

  fn _push_fwd(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    if self.base.stack.push(pass) {
      apply(self, rvar, xvar);
    }
  }

  fn _txn(&self) -> Option<Txn> {
    self.val.txn()
  }

  fn _reset(&self) {
    self._value().reset();
  }

  fn _release(&self) {
    self._value().release();
  }

  /*fn _persist(&self, txn: Txn) {
    self.val.persist(txn);
  }*/

  /*fn _prepare(&self, txn: Txn) {
    if let Some(ref prepare) = self.ext.prepare {
      (prepare)(txn, self.cfg.borrow_mut());
    }
  }*/

  /*fn _cleanup(&self, txn: Txn) {
    // TODO
    if let Some(ref cleanup) = self.ext.cleanup {
      (cleanup)(txn, self.cfg.borrow_mut());
    }
  }*/

  fn _apply(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    //println!("DEBUG: FSrcOp: apply");
    self._apply_output(txn, OVal::new(rvar, xvar, self._value()._clone()));
  }

  fn _eval_recursive(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    //println!("DEBUG: FSrcOp: eval recursive");
    if Some(txn) != self._txn() {
      for node in self.ctrl.iter() {
        node.eval(txn);
      }
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
    //(self.ext.make_val)()
    (self.ext.make_val)(self.cfg.borrow_mut())
  }

  fn _value(&self) -> &RWVal<V> {
    &self.val
  }

  fn _push_tangent(&self, pass: Pass, feedfwd: &mut FeedFwd) -> Val<V> {
    match self.ext.tangent {
      None => unimplemented!(),
      Some(ref tangent) => (tangent)(pass, self.cfg.borrow_mut(), feedfwd),
    }
  }

  fn _pop_adjoint(&self, pass: Pass, this: Val<V>, sink: &mut Sink) {
    if self.base.stack.pop(pass) {
      match self.ext.adjoint {
        None => {}
        Some(ref adjoint) => (adjoint)(pass, this, self.cfg.borrow_mut(), sink),
      }
    }
  }

  fn _apply_output(&self, txn: Txn, val: OVal<V>) {
    (self.ext.apply)(txn, self.cfg.borrow_mut(), val);
  }
}

/*pub struct Pipe1Op<F, V, W> where W: OVal {
  base: OpBase,
  ext:  OpExt<W>,
  cfg:  F,
  // TODO: should not contain an input `x` but instead a "slot" in which to
  // pipe an input of the same type.
  x_:   Rc<AOp<V>>,
  y:    W,
}

impl<F, V, W> Pipe1Op<F, V, W>
where V: OVal, W: OVal {
  pub fn attach(&self, txn: Txn, input: Rc<AOp<V>>) {
    // TODO: figure out `attach` semantics.
  }
}*/

pub struct F1WrapOp<F, V, W> {
  base: OpBase,
  ext:  OpExt<F, W>,
  cfg:  RefCell<F>,
  ctrl: Vec<Node>,
  x_:   Val<V>,
  y_:   Val<W>,
}

impl<F, V, W> F1WrapOp<F, V, W> {
  pub fn new(cfg: F, ext: OpExt<F, W>, x_: Val<V>, y_: Val<W>) -> Self {
    let cfg = RefCell::new(cfg);
    F1WrapOp{
      base: OpBase::default(),
      ext:  ext,
      cfg:  cfg,
      ctrl: vec![],
      x_:   x_,
      y_:   y_,
    }
  }
}

impl<F, V, W> ANode for F1WrapOp<F, V, W> where V: 'static, RWVal<W>: IOVal + 'static {
  fn _walk(&self) -> &Walk {
    &self.base.stack
  }

  fn _io(&self) -> &IOVal {
    self.y_._op()._io()
  }

  fn _analysis_tags(&self) -> &AnalysisTags {
    &self.base
  }

  fn _pred_fwd(&self, pred_buf: &mut Vec<Node>) {
    pred_buf.push(self.x_.to_node());
  }

  fn _pred_rev(&self, pred_buf: &mut Vec<Node>) {
    pred_buf.push(self.x_.to_node());
  }

  fn _push(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.push(pass) {
      if stop_txn.is_none() || stop_txn != self._txn() {
        self.x_._node()._push(stop_txn, pass, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.pop(pass) {
      apply(self);
      if stop_txn.is_none() || stop_txn != self._txn() {
        self.x_._node()._pop(stop_txn, pass, apply);
      }
    }
  }

  fn _push_fwd(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    if self.base.stack.push(pass) {
      if stop_txn.is_none() || stop_txn != self._txn() {
        self.x_._push_fwd(stop_txn, pass, apply);
      }
      apply(self, rvar, xvar);
    }
  }

  fn _txn(&self) -> Option<Txn> {
    self.y_._op()._txn()
  }

  fn _reset(&self) {
    self._value().reset();
  }

  fn _release(&self) {
    self._value().release();
  }

  fn _apply(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    self._apply_output(txn, OVal::new(rvar, xvar, self._value()._clone()));
  }

  fn _eval_recursive(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    if Some(txn) != self._txn() {
      for node in self.ctrl.iter() {
        node.eval(txn);
      }
      self.x_.eval(txn);
    }
  }
}

impl<F, V, W> AOp<W> for F1WrapOp<F, V, W> where V: 'static, RWVal<W>: IOVal + 'static {
  fn _make_value(&self) -> RWVal<W> {
    (self.ext.make_val)(self.cfg.borrow_mut())
  }

  fn _value(&self) -> &RWVal<W> {
    self.y_._op()._value()
  }

  fn _apply_output(&self, txn: Txn, val: OVal<W>) {
    (self.ext.apply)(txn, self.cfg.borrow_mut(), val);
  }

  fn _push_tangent(&self, pass: Pass, feedfwd: &mut FeedFwd) -> Val<W> {
    match self.ext.tangent {
      None => unimplemented!(),
      Some(ref tangent) => (tangent)(pass, self.cfg.borrow_mut(), feedfwd),
    }
  }

  fn _pop_adjoint(&self, pass: Pass, this: Val<W>, sink: &mut Sink) {
    if self.base.stack.pop(pass) {
      match self.ext.adjoint {
        None => {}
        Some(ref adjoint) => {
          (adjoint)(pass, this, self.cfg.borrow_mut(), sink);
        }
      }
      self.x_._pop_adjoint(pass, sink);
    }
  }

  fn _inplace(&self) -> Option<Val<W>> {
    None
  }
}

pub struct F1Op<F, V1, W> {
  base: OpBase,
  ext:  OpExt<F, W>,
  cfg:  RefCell<F>,
  ctrl: Vec<Node>,
  x_:   Val<V1>,
  y:    RWVal<W>,
}

impl<F, V1, W> F1Op<F, V1, W> {
  //pub fn new(cfg: F, ext: OpExt<F, W>, x_: Val<V1>, y: RWVal<W>) -> Self {
  pub fn new(cfg: F, ext: OpExt<F, W>, x_: Val<V1>) -> Self {
    let state = RefCell::new(cfg);
    let y = (ext.make_val)(state.borrow_mut());
    F1Op{
      base: OpBase::default(),
      ext:  ext,
      //cfg:  RefCell::new(cfg),
      cfg:  state,
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

  fn _analysis_tags(&self) -> &AnalysisTags {
    &self.base
  }

  fn _pred_fwd(&self, pred_buf: &mut Vec<Node>) {
    pred_buf.push(self.x_.to_node());
  }

  fn _pred_rev(&self, pred_buf: &mut Vec<Node>) {
    pred_buf.push(self.x_.to_node());
  }

  fn _push(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.push(pass) {
      if stop_txn.is_none() || stop_txn != self._txn() {
        self.x_._node()._push(stop_txn, pass, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.pop(pass) {
      apply(self);
      if stop_txn.is_none() || stop_txn != self._txn() {
        self.x_._node()._pop(stop_txn, pass, apply);
      }
    }
  }

  fn _push_fwd(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    if self.base.stack.push(pass) {
      if stop_txn.is_none() || stop_txn != self._txn() {
        self.x_._push_fwd(stop_txn, pass, apply);
      }
      apply(self, rvar, xvar);
    }
  }

  fn _txn(&self) -> Option<Txn> {
    self.y.txn()
  }

  fn _reset(&self) {
    self._value().reset();
  }

  fn _release(&self) {
    self._value().release();
  }

  /*fn _persist(&self, txn: Txn) {
    self.y.persist(txn);
  }*/

  /*fn _prepare(&self, txn: Txn) {
    self.x_._node().eval(txn);
    if let Some(ref prepare) = self.ext.prepare {
      (prepare)(txn, self.cfg.borrow_mut());
    }
  }*/

  /*fn _cleanup(&self, txn: Txn) {
    // TODO
    if let Some(ref cleanup) = self.ext.cleanup {
      (cleanup)(txn, self.cfg.borrow_mut());
    }
  }*/

  fn _apply(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    //println!("DEBUG: F1Op: apply");
    self._apply_output(txn, OVal::new(rvar, xvar, self._value()._clone()));
  }

  fn _eval_recursive(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    //println!("DEBUG: F1Op: eval recursive");
    if Some(txn) != self._txn() {
      for node in self.ctrl.iter() {
        node.eval(txn);
      }
      //println!("DEBUG: F1Op: inner eval recursive");
      self.x_.eval(txn);
    }
  }
}

impl<F, V1, W> AOp<W> for F1Op<F, V1, W> where V1: 'static, RWVal<W>: IOVal + 'static {
  fn _make_value(&self) -> RWVal<W> {
    //(self.ext.make_val)()
    (self.ext.make_val)(self.cfg.borrow_mut())
  }

  fn _value(&self) -> &RWVal<W> {
    &self.y
  }

  fn _push_tangent(&self, pass: Pass, feedfwd: &mut FeedFwd) -> Val<W> {
    match self.ext.tangent {
      None => unimplemented!(),
      Some(ref tangent) => (tangent)(pass, self.cfg.borrow_mut(), feedfwd),
    }
  }

  /*fn tangent(&self) -> Val<W> {
    /*let mut tng_op = self.base.tng_op.borrow_mut();
    if tng_op.is_none() {
      *tng_op = Some(self._push_tangent());
    }
    tng_op.as_ref().unwrap().clone()*/
    // TODO
    unimplemented!();
  }*/

  /*fn _push_adjoint(&self, pass: Pass) {
    // TODO
    if self.base.stack.pop(pass) {
      match self.ext.adjoint {
        None => {}
        Some(_) => {
          self.x_._push_adjoint(pass);
        }
      }
    }
  }*/

  fn _pop_adjoint(&self, pass: Pass, this: Val<W>, sink: &mut Sink) {
    if self.base.stack.pop(pass) {
      match self.ext.adjoint {
        None => {}
        Some(ref adjoint) => {
          (adjoint)(pass, this, self.cfg.borrow_mut(), sink);
        }
      }
      self.x_._pop_adjoint(pass, sink);
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

  fn _apply_output(&self, txn: Txn, val: OVal<W>) {
    (self.ext.apply)(txn, self.cfg.borrow_mut(), val);
  }
}

impl<F, V> AOp<V> for F1Op<F, V, V> where V: 'static, RWVal<V>: IOVal + 'static {
  fn _inplace(&self) -> Option<Val<V>> {
    match self.ext.inplace {
      None => None,
      Some(ref inplace) => {
        if self.x_._node()._walk().outdegree() == 1 {
          Some((inplace)(self.x_.clone()))
        } else {
          None
        }
      }
    }
  }
}

pub struct F2Op<F, V1, V2, W> {
  base: OpBase,
  ext:  OpExt<F, W>,
  cfg:  RefCell<F>,
  ctrl: Vec<Node>,
  x1_:  Val<V1>,
  x2_:  Val<V2>,
  y:    RWVal<W>,
}

impl<F, V1, V2, W> F2Op<F, V1, V2, W> {
  //pub fn new(cfg: F, ext: OpExt<F, W>, x1_: Val<V1>, x2_: Val<V2>, y: RWVal<W>) -> Self {
  pub fn new(cfg: F, ext: OpExt<F, W>, x1_: Val<V1>, x2_: Val<V2>) -> Self {
    let state = RefCell::new(cfg);
    let y = (ext.make_val)(state.borrow_mut());
    F2Op{
      base: OpBase::default(),
      ext:  ext,
      //cfg:  RefCell::new(cfg),
      cfg:  state,
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

  fn _analysis_tags(&self) -> &AnalysisTags {
    &self.base
  }

  fn _pred_fwd(&self, pred_buf: &mut Vec<Node>) {
    pred_buf.push(self.x1_.to_node());
    pred_buf.push(self.x2_.to_node());
  }

  fn _pred_rev(&self, pred_buf: &mut Vec<Node>) {
    pred_buf.push(self.x2_.to_node());
    pred_buf.push(self.x1_.to_node());
  }

  fn _push(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.push(pass) {
      if stop_txn.is_none() || stop_txn != self._txn() {
        self.x1_._node()._push(stop_txn, pass, apply);
        self.x2_._node()._push(stop_txn, pass, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.pop(pass) {
      apply(self);
      if stop_txn.is_none() || stop_txn != self._txn() {
        self.x2_._node()._pop(stop_txn, pass, apply);
        self.x1_._node()._pop(stop_txn, pass, apply);
      }
    }
  }

  fn _push_fwd(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    if self.base.stack.push(pass) {
      if stop_txn.is_none() || stop_txn != self._txn() {
        self.x1_._push_fwd(stop_txn, pass, apply);
        self.x2_._push_fwd(stop_txn, pass, apply);
      }
      apply(self, rvar, xvar);
    }
  }

  fn _txn(&self) -> Option<Txn> {
    self.y.txn()
  }

  fn _reset(&self) {
    self._value().reset();
  }

  fn _release(&self) {
    self._value().release();
  }

  /*fn _persist(&self, txn: Txn) {
    self.y.persist(txn);
  }*/

  /*fn _prepare(&self, txn: Txn) {
    self.x1_._node().eval(txn);
    self.x2_._node().eval(txn);
    if let Some(ref prepare) = self.ext.prepare {
      (prepare)(txn, self.cfg.borrow_mut());
    }
  }*/

  /*fn _cleanup(&self, txn: Txn) {
    // TODO
    if let Some(ref cleanup) = self.ext.cleanup {
      (cleanup)(txn, self.cfg.borrow_mut());
    }
  }*/

  fn _apply(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    self._apply_output(txn, OVal::new(rvar, xvar, self._value()._clone()));
  }

  fn _eval_recursive(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    if Some(txn) != self._txn() {
      for node in self.ctrl.iter() {
        node.eval(txn);
      }
      self.x1_.eval(txn);
      self.x2_.eval(txn);
    }
  }
}

impl<F, V1, V2, W> AOp<W> for F2Op<F, V1, V2, W> where V1: 'static, V2: 'static, RWVal<W>: IOVal + 'static {
  fn _make_value(&self) -> RWVal<W> {
    //(self.ext.make_val)()
    (self.ext.make_val)(self.cfg.borrow_mut())
  }

  fn _value(&self) -> &RWVal<W> {
    &self.y
  }

  fn _push_tangent(&self, pass: Pass, feedfwd: &mut FeedFwd) -> Val<W> {
    match self.ext.tangent {
      None => unimplemented!(),
      Some(ref tangent) => (tangent)(pass, self.cfg.borrow_mut(), feedfwd),
    }
  }

  /*fn tangent(&self) -> Val<W> {
    /*let mut tng_op = self.base.tng_op.borrow_mut();
    if tng_op.is_none() {
      *tng_op = Some(self._push_tangent());
    }
    tng_op.as_ref().unwrap().clone()*/
    // TODO
    unimplemented!();
  }*/

  fn _pop_adjoint(&self, pass: Pass, this: Val<W>, sink: &mut Sink) {
    if self.base.stack.pop(pass) {
      match self.ext.adjoint {
        None => {}
        Some(ref adjoint) => {
          (adjoint)(pass, this, self.cfg.borrow_mut(), sink);
        }
      }
      self.x2_._pop_adjoint(pass, sink);
      self.x1_._pop_adjoint(pass, sink);
    }
  }

  /*//fn adjoint(&self, sink: &mut Sink) -> (Rc<ANode>, Rc<AOp<W>>) {
  fn adjoint(&self, sink: &mut Sink) -> Val<W> {
    match sink.get_adj(self.var()) {
      None => panic!(),
      Some(adj) => adj,
    }
  }*/

  fn _apply_output(&self, txn: Txn, val: OVal<W>) {
    (self.ext.apply)(txn, self.cfg.borrow_mut(), val);
  }
}

pub struct F3Op<F, V1, V2, V3, W> {
  base: OpBase,
  ext:  OpExt<F, W>,
  cfg:  RefCell<F>,
  ctrl: Vec<Node>,
  x1_:  Val<V1>,
  x2_:  Val<V2>,
  x3_:  Val<V3>,
  y:    RWVal<W>,
}

impl<F, V1, V2, V3, W> F3Op<F, V1, V2, V3, W> {
  //pub fn new(cfg: F, ext: OpExt<F, W>, x1_: Val<V1>, x2_: Val<V2>, x3_: Val<V3>, y: RWVal<W>) -> Self {
  pub fn new(cfg: F, ext: OpExt<F, W>, x1_: Val<V1>, x2_: Val<V2>, x3_: Val<V3>) -> Self {
    let state = RefCell::new(cfg);
    let y = (ext.make_val)(state.borrow_mut());
    F3Op{
      base: OpBase::default(),
      ext:  ext,
      //cfg:  RefCell::new(cfg),
      cfg:  state,
      ctrl: vec![],
      x1_:  x1_,
      x2_:  x2_,
      x3_:  x3_,
      y:    y,
    }
  }
}

impl<F, V1, V2, V3, W> ANode for F3Op<F, V1, V2, V3, W> where V1: 'static, V2: 'static, V3: 'static, RWVal<W>: IOVal + 'static {
  fn _walk(&self) -> &Walk {
    &self.base.stack
  }

  fn _io(&self) -> &IOVal {
    &self.y
  }

  fn _analysis_tags(&self) -> &AnalysisTags {
    &self.base
  }

  fn _pred_fwd(&self, pred_buf: &mut Vec<Node>) {
    pred_buf.push(self.x1_.to_node());
    pred_buf.push(self.x2_.to_node());
    pred_buf.push(self.x3_.to_node());
  }

  fn _pred_rev(&self, pred_buf: &mut Vec<Node>) {
    pred_buf.push(self.x3_.to_node());
    pred_buf.push(self.x2_.to_node());
    pred_buf.push(self.x1_.to_node());
  }

  fn _push(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.push(pass) {
      if stop_txn.is_none() || stop_txn != self._txn() {
        self.x1_._node()._push(stop_txn, pass, apply);
        self.x2_._node()._push(stop_txn, pass, apply);
        self.x3_._node()._push(stop_txn, pass, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.pop(pass) {
      apply(self);
      if stop_txn.is_none() || stop_txn != self._txn() {
        self.x3_._node()._pop(stop_txn, pass, apply);
        self.x2_._node()._pop(stop_txn, pass, apply);
        self.x1_._node()._pop(stop_txn, pass, apply);
      }
    }
  }

  fn _push_fwd(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    if self.base.stack.push(pass) {
      if stop_txn.is_none() || stop_txn != self._txn() {
        self.x1_._push_fwd(stop_txn, pass, apply);
        self.x2_._push_fwd(stop_txn, pass, apply);
        self.x3_._push_fwd(stop_txn, pass, apply);
      }
      apply(self, rvar, xvar);
    }
  }

  fn _txn(&self) -> Option<Txn> {
    self.y.txn()
  }

  fn _reset(&self) {
    self._value().reset();
  }

  fn _release(&self) {
    self._value().release();
  }

  /*fn _persist(&self, txn: Txn) {
    self.y.persist(txn);
  }*/

  /*fn _prepare(&self, txn: Txn) {
    self.x1_._node().eval(txn);
    self.x2_._node().eval(txn);
    self.x3_._node().eval(txn);
    if let Some(ref prepare) = self.ext.prepare {
      (prepare)(txn, self.cfg.borrow_mut());
    }
  }*/

  /*fn _cleanup(&self, txn: Txn) {
    // TODO
    if let Some(ref cleanup) = self.ext.cleanup {
      (cleanup)(txn, self.cfg.borrow_mut());
    }
  }*/

  fn _apply(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    self._apply_output(txn, OVal::new(rvar, xvar, self._value()._clone()));
  }

  fn _eval_recursive(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    if Some(txn) != self._txn() {
      for node in self.ctrl.iter() {
        node.eval(txn);
      }
      self.x1_.eval(txn);
      self.x2_.eval(txn);
      self.x3_.eval(txn);
    }
  }
}

impl<F, V1, V2, V3, W> AOp<W> for F3Op<F, V1, V2, V3, W> where V1: 'static, V2: 'static, V3: 'static, RWVal<W>: IOVal + 'static {
  fn _make_value(&self) -> RWVal<W> {
    //(self.ext.make_val)()
    (self.ext.make_val)(self.cfg.borrow_mut())
  }

  fn _value(&self) -> &RWVal<W> {
    &self.y
  }

  fn _push_tangent(&self, pass: Pass, feedfwd: &mut FeedFwd) -> Val<W> {
    match self.ext.tangent {
      None => unimplemented!(),
      Some(ref tangent) => (tangent)(pass, self.cfg.borrow_mut(), feedfwd),
    }
  }

  /*//fn tangent(&self) -> (Rc<ANode>, Rc<AOp<W>>) {
  fn tangent(&self) -> Val<W> {
    /*let mut tng_op = self.base.tng_op.borrow_mut();
    if tng_op.is_none() {
      *tng_op = Some(self._push_tangent());
    }
    tng_op.as_ref().unwrap().clone()*/
    // TODO
    unimplemented!();
  }*/

  fn _pop_adjoint(&self, pass: Pass, this: Val<W>, sink: &mut Sink) {
    if self.base.stack.pop(pass) {
      match self.ext.adjoint {
        None => {}
        Some(ref adjoint) => {
          (adjoint)(pass, this, self.cfg.borrow_mut(), sink);
        }
      }
      self.x3_._pop_adjoint(pass, sink);
      self.x2_._pop_adjoint(pass, sink);
      self.x1_._pop_adjoint(pass, sink);
    }
  }

  /*//fn adjoint(&self, sink: &mut Sink) -> (Rc<ANode>, Rc<AOp<W>>) {
  fn adjoint(&self, sink: &mut Sink) -> Val<W> {
    match sink.get_adj(self.var()) {
      None => panic!(),
      Some(adj) => adj,
    }
  }*/

  fn _apply_output(&self, txn: Txn, val: OVal<W>) {
    (self.ext.apply)(txn, self.cfg.borrow_mut(), val);
  }
}

pub struct FSwitchOp<F, V> {
  base: OpBase,
  ext:  OpExt<F, V>,
  cfg:  RefCell<F>,
  ctrl: Vec<Node>,
  flag: TCell<bool>,
  x1_:  Val<V>,
  x2_:  Val<V>,
  done: TCell<()>,
}

impl<F, V> FSwitchOp<F, V> {
  pub fn new(cfg: F, ext: OpExt<F, V>, flag: TCell<bool>, x1_: Val<V>, x2_: Val<V>) -> Self {
    let cfg = RefCell::new(cfg);
    FSwitchOp{
      base: OpBase::default(),
      ext:  ext,
      cfg:  cfg,
      ctrl: vec![],
      flag: flag,
      x1_:  x1_,
      x2_:  x2_,
      done: TCell::new(()),
    }
  }
}

impl<F, V> ANode for FSwitchOp<F, V> where V: 'static, RWVal<V>: IOVal + 'static {
  fn _walk(&self) -> &Walk {
    &self.base.stack
  }

  fn _io(&self) -> &IOVal {
    // NOTE: `always_get` is valid here.
    match self.flag.always_get() {
      false => self.x1_._op()._io(),
      true  => self.x2_._op()._io(),
    }
  }

  fn _analysis_tags(&self) -> &AnalysisTags {
    &self.base
  }

  fn _pred_fwd(&self, pred_buf: &mut Vec<Node>) {
    pred_buf.push(self.x1_.to_node());
    pred_buf.push(self.x2_.to_node());
  }

  fn _pred_rev(&self, pred_buf: &mut Vec<Node>) {
    pred_buf.push(self.x2_.to_node());
    pred_buf.push(self.x1_.to_node());
  }

  fn _push(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.push(pass) {
      if stop_txn.is_none() || stop_txn != self._txn() {
        self.x1_._node()._push(stop_txn, pass, apply);
        self.x2_._node()._push(stop_txn, pass, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.pop(pass) {
      apply(self);
      if stop_txn.is_none() || stop_txn != self._txn() {
        self.x2_._node()._pop(stop_txn, pass, apply);
        self.x1_._node()._pop(stop_txn, pass, apply);
      }
    }
  }

  fn _push_fwd(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    if self.base.stack.push(pass) {
      if stop_txn.is_none() || stop_txn != self._txn() {
        self.x1_._push_fwd(stop_txn, pass, apply);
        self.x2_._push_fwd(stop_txn, pass, apply);
      }
      apply(self, rvar, xvar);
    }
  }

  fn _txn(&self) -> Option<Txn> {
    self.done.txn()
  }

  fn _reset(&self) {
    self.done.reset();
  }

  fn _release(&self) {
    self.done.reset();
  }

  fn _apply(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    match self.flag.get(txn) {
      false => self._apply_output(txn, OVal::new(rvar, xvar, self.x1_._op()._value()._clone())),
      true  => self._apply_output(txn, OVal::new(rvar, xvar, self.x2_._op()._value()._clone())),
    }
    self.done.propose(txn, |_| ());
  }

  fn _eval_recursive(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    if Some(txn) != self._txn() {
      for node in self.ctrl.iter() {
        node.eval(txn);
      }
      match self.flag.get(txn) {
        false   => self.x1_.eval(txn),
        true    => self.x2_.eval(txn),
      }
    }
  }
}

impl<F, V> AOp<V> for FSwitchOp<F, V> where V: 'static, RWVal<V>: IOVal + 'static {
  fn _make_value(&self) -> RWVal<V> {
    (self.ext.make_val)(self.cfg.borrow_mut())
  }

  fn _value(&self) -> &RWVal<V> {
    // NOTE: `always_get` is valid here.
    match self.flag.always_get() {
      false => self.x1_._op()._value(),
      true  => self.x2_._op()._value(),
    }
  }

  fn _apply_output(&self, txn: Txn, val: OVal<V>) {
    (self.ext.apply)(txn, self.cfg.borrow_mut(), val);
  }

  fn _push_tangent(&self, pass: Pass, feedfwd: &mut FeedFwd) -> Val<V> {
    match self.ext.tangent {
      None => unimplemented!(),
      Some(ref tangent) => (tangent)(pass, self.cfg.borrow_mut(), feedfwd),
    }
  }

  fn _pop_adjoint(&self, pass: Pass, this: Val<V>, sink: &mut Sink) {
    if self.base.stack.pop(pass) {
      match self.ext.adjoint {
        None => {}
        Some(ref adjoint) => {
          (adjoint)(pass, this, self.cfg.borrow_mut(), sink);
        }
      }
      self.x2_._pop_adjoint(pass, sink);
      self.x1_._pop_adjoint(pass, sink);
    }
  }
}

pub struct FMuxOp<F, V> {
  base: OpBase,
  ext:  OpExt<F, V>,
  cfg:  RefCell<F>,
  ctrl: Vec<Node>,
  sel:  TCell<usize>,
  xs_:  Vec<Val<V>>,
  done: TCell<()>,
}

pub struct FJoinOp<F, V, W> {
  base: OpBase,
  ext:  OpExt<F, W>,
  cfg:  RefCell<F>,
  ctrl: Vec<Node>,
  xs_:  Vec<Val<V>>,
  y:    RWVal<W>,
}

impl<F, V, W> FJoinOp<F, V, W> {
  //pub fn new(cfg: F, ext: OpExt<F, W>, xs_: Vec<Val<V>>, y: RWVal<W>) -> Self {
  pub fn new(cfg: F, ext: OpExt<F, W>, xs_: Vec<Val<V>>) -> Self {
    let state = RefCell::new(cfg);
    let y = (ext.make_val)(state.borrow_mut());
    FJoinOp{
      base: OpBase::default(),
      ext:  ext,
      //cfg:  RefCell::new(cfg),
      cfg:  state,
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

  fn _analysis_tags(&self) -> &AnalysisTags {
    &self.base
  }

  fn _pred_fwd(&self, pred_buf: &mut Vec<Node>) {
    for x_ in self.xs_.iter() {
      pred_buf.push(x_.to_node());
    }
  }

  fn _pred_rev(&self, pred_buf: &mut Vec<Node>) {
    for x_ in self.xs_.iter().rev() {
      pred_buf.push(x_.to_node());
    }
  }

  fn _push(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.push(pass) {
      if stop_txn.is_none() || stop_txn != self._txn() {
        for x_ in self.xs_.iter() {
          x_._node()._push(stop_txn, pass, apply);
        }
      }
      apply(self);
    }
  }

  fn _pop(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.pop(pass) {
      apply(self);
      if stop_txn.is_none() || stop_txn != self._txn() {
        for x_ in self.xs_.iter().rev() {
          x_._node()._pop(stop_txn, pass, apply);
        }
      }
    }
  }

  fn _push_fwd(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    if self.base.stack.push(pass) {
      if stop_txn.is_none() || stop_txn != self._txn() {
        for x_ in self.xs_.iter() {
          x_._push_fwd(stop_txn, pass, apply);
        }
      }
      apply(self, rvar, xvar);
    }
  }

  fn _txn(&self) -> Option<Txn> {
    self.y.txn()
  }

  fn _reset(&self) {
    self._value().reset();
  }

  fn _release(&self) {
    self._value().release();
  }

  /*fn _persist(&self, txn: Txn) {
    self.y.persist(txn);
  }*/

  /*fn _prepare(&self, txn: Txn) {
    for x in self.xs_.iter() {
      x._node().eval(txn);
    }
    if let Some(ref prepare) = self.ext.prepare {
      (prepare)(txn, self.cfg.borrow_mut());
    }
  }*/

  /*fn _cleanup(&self, txn: Txn) {
    // TODO
    if let Some(ref cleanup) = self.ext.cleanup {
      (cleanup)(txn, self.cfg.borrow_mut());
    }
  }*/

  fn _apply(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    self._apply_output(txn, OVal::new(rvar, xvar, self._value()._clone()));
  }

  fn _eval_recursive(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    if Some(txn) != self._txn() {
      for node in self.ctrl.iter() {
        node.eval(txn);
      }
      for x_ in self.xs_.iter() {
        x_.eval(txn);
      }
    }
  }
}

impl<F, V, W> AOp<W> for FJoinOp<F, V, W> where V: 'static, RWVal<W>: IOVal + 'static {
  fn _make_value(&self) -> RWVal<W> {
    //(self.ext.make_val)()
    (self.ext.make_val)(self.cfg.borrow_mut())
  }

  fn _value(&self) -> &RWVal<W> {
    &self.y
  }

  fn _push_tangent(&self, pass: Pass, feedfwd: &mut FeedFwd) -> Val<W> {
    if self.base.stack.push(pass) {
      for x_ in self.xs_.iter() {
        x_._push_tangent(pass, feedfwd);
      }
      match self.ext.tangent {
        None => unimplemented!(),
        Some(ref tangent) => (tangent)(pass, self.cfg.borrow_mut(), feedfwd),
      }
    } else {
      // TODO
      unimplemented!();
    }
  }

  /*fn tangent(&self) -> Val<W> {
    /*let mut tng_op = self.base.tng_op.borrow_mut();
    if tng_op.is_none() {
      *tng_op = Some(self._push_tangent());
    }
    tng_op.as_ref().unwrap().clone()*/
    // TODO
    unimplemented!();
  }*/

  fn _pop_adjoint(&self, pass: Pass, this: Val<W>, sink: &mut Sink) {
    if self.base.stack.pop(pass) {
      match self.ext.adjoint {
        None => {}
        Some(ref adjoint) => {
          (adjoint)(pass, this, self.cfg.borrow_mut(), sink);
        }
      }
      for x_ in self.xs_.iter().rev() {
        x_._pop_adjoint(pass, sink);
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

  fn _apply_output(&self, txn: Txn, val: OVal<W>) {
    (self.ext.apply)(txn, self.cfg.borrow_mut(), val);
  }
}

impl<F, V> AOp<V> for FJoinOp<F, V, V> where RWVal<V>: IOVal + 'static {
  fn _inplace(&self) -> Option<Val<V>> {
    match self.ext.inplace {
      None => {}
      Some(ref inplace) => {
        for x_ in self.xs_.iter() {
          if x_._node()._walk().outdegree() == 1 {
            return Some((inplace)(x_.clone()));
          }
        }
      }
    }
    None
  }
}
