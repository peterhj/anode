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
#![feature(nll)]
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
extern crate dot;
//extern crate float;
#[cfg(feature = "gpu")] extern crate gpudevicemem;
#[macro_use] extern crate lazy_static;
extern crate memarray;
#[cfg(feature = "mpi")] extern crate mpich;
extern crate parking_lot;
extern crate rand;
extern crate rng;
extern crate time;
extern crate typemap;

use analysis::{LivenessAnalysis};
use log::*;
use ops::{OnesSrcOp, OnesSrcOpMaybeExt, SumJoinOp, SumJoinOpMaybeExt, PassExt};
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
use std::intrinsics::{type_name};
//use std::ops::{Deref, DerefMut};
use std::rc::{Rc};
use std::sync::{Arc};
use std::sync::mpsc::{SyncSender, Receiver};

pub mod analysis;
pub mod config;
pub mod context;
pub mod ffi;
#[cfg(feature = "gpu")] pub mod io_gpu;
pub mod log;
pub mod ops;
#[cfg(feature = "gpu")] pub mod ops_gpu;
#[cfg(feature = "mpi")] pub mod ops_mpi;
#[cfg(feature = "mpi")] pub mod proc;
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
pub struct OpRef(u64);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RWValRef(u64);

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

impl RVar {
  fn _raw(self) -> u64 {
    self.0
  }
}

impl RWVar {
  fn _raw(self) -> u64 {
    self.0._raw()
  }
}

impl Default for OpRef {
  fn default() -> Self {
    OpRef(gen_thread_local_uid())
  }
}

impl Default for RWValRef {
  fn default() -> Self {
    RWValRef(gen_thread_local_uid())
  }
}

pub struct WalkStackEntry {
  pass:         Pass,
  push_degree:  usize,
  pop_degree:   usize,
  // TODO: cycle detection.
  //succ_set:     HashSet<OpRef>,
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
  fn _key(&self) -> String;
  fn _walk(&self) -> &Walk;
  fn _analysis_tags(&self) -> &AnalysisTags { unimplemented!(); }

  fn _pred_fwd(&self, pred_buf: &mut Vec<Node>) { unimplemented!(); }
  fn _pred_rev(&self, pred_buf: &mut Vec<Node>) { unimplemented!(); }

  fn _push(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) { unimplemented!(); }
  fn _pop(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) { unimplemented!(); }

  fn _push_fwd(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar));
  fn _pop_rev(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar));

  //fn _txn(&self) -> Option<Txn>;
  //fn _reset(&self);
  //fn _release(&self);
  //fn _persist(&self, txn: Txn);
  //fn _prepare(&self, txn: Txn);
  //fn _cleanup(&self, txn: Txn);

  //fn _io(&self) -> &IOVal;
  fn _io<'a>(&'a self, txn: Txn, static_any_value: &'a Any) -> &'a IOVal;

  fn _eval_recursive(&self, txn: Txn, rvar: RVar, xvar: RWVar);
  //fn _apply(&self, txn: Txn, rvar: RVar, xvar: RWVar, mode: WriteMode);
  fn _apply_any(&self, txn: Txn, rvar: RVar, xvar: RWVar, mode: WriteMode, any_value: Rc<Any>);
}

pub trait AOp<V>: ANode {
  fn _pred_val_fwd(&self, pred_buf: &mut Vec<Rc<Any>>) { unimplemented!(); }
  fn _pred_val_rev(&self, pred_buf: &mut Vec<Rc<Any>>) { unimplemented!(); }

  //fn _value(&self) -> &RWVal<V>;
  fn _value2(&self, txn: Txn, static_value: Option<RWVal<V>>) -> RWVal<V>;
  fn _value3<'a>(&'a self, txn: Txn, static_value: Option<&'a RWVal<V>>) -> &'a RWVal<V>;

  fn _make_value(&self) -> RWVal<V>;
  fn _build(&self, pred_vals: Vec<Rc<Any>>) -> Val<V> { unimplemented!(); }
  fn _apply_output(&self, txn: Txn, output: OVal<V>);
  fn _push_tangent(&self, pass: Pass, feedfwd: &mut FeedFwd) -> Val<V> { unimplemented!(); }
  //fn tangent(&self) -> Val<V>;
  fn _pop_adjoint(&self, pass: Pass, this: Val<V>, sink: &mut Sink) { unimplemented!(); }
  //fn adjoint(&self, sink: &mut Sink) -> Val<V>;
  // TODO
  //fn _substitute(&self, subs: Vec<(RWVar, Rc<Any>)>) -> Option<(Rc<ANode>, Rc<AOp<V>>)> { None }
  fn _inplace(&self) -> Option<Val<V>> { None }
}

pub trait AnyRWVal {
  //fn _txn(&self) -> Option<Txn>;
  fn _complete_txn(&self) -> Option<Txn>;
  fn _persist(&self, txn: Txn, xvar: RWVar);
}

pub trait IOVal: AnyRWVal {
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
  fn inplace_at(&self, arg_idx: Option<usize>) -> Option<Val<V>>;

  fn inplace(&self) -> Option<Val<V>> {
    self.inplace_at(None)
  }
}

#[cfg(feature = "gpu")]
pub trait GPUWrapValExt<V> {
  fn gpu_mux(&self, dev: GPUDeviceId) -> Val<V>;
}

impl<V> WrapValExt<V> for Val<V> where V: 'static {
  fn inplace_at(&self, arg_idx: Option<usize>) -> Option<Val<V>> {
    // TODO
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
  node:     Rc<ANode>,
  mode:     WriteMode,
  value:    Rc<Any>,
  xvar:     RWVar,
  rvar:     RVar,
  name:     Option<String>,
}

impl Clone for Node {
  fn clone(&self) -> Self {
    let rvar = RVar::default();
    Node{
      node:     self.node.clone(),
      mode:     self.mode,
      value:    self.value.clone(),
      xvar:     self.xvar,
      rvar:     rvar,
      name:     self.name.clone(),
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

  fn _io<'a>(&'a self, txn: Txn) -> &'a IOVal {
    self.node._io(txn, &*self.value)
  }

  pub fn _apply(&self, txn: Txn) {
    //self.node._apply(txn, self.rvar, self.xvar, self.mode);
    self.node._apply_any(txn, self.rvar, self.xvar, self.mode, self.value.clone());
  }

  pub fn _eval_recursive(&self, txn: Txn) {
    if Some(txn) == self.node._io(txn, &*self.value)._complete_txn() {
      self.node._eval_recursive(txn, self.rvar, self.xvar);
    }
  }

  pub fn persist(&self, txn: Txn) {
    self.node._io(txn, &*self.value)._persist(txn, self.xvar);
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
    self.node._io(txn, &*self.value)._serialize(txn, self.rvar, dst);
  }

  fn deserialize(&self, txn: Txn, src: &mut Any) {
    self.node._io(txn, &*self.value)._deserialize(txn, self.xvar, src);
  }
}

impl VIONodeExt for Node {
  fn _serialize_vec(&self, txn: Txn, off: usize, dst: &mut Any) -> usize {
    self.node._io(txn, &*self.value)._serialize_vec(txn, self.rvar, off, dst)
  }

  fn _deserialize_vec(&self, txn: Txn, off: usize, src: &mut Any) -> usize {
    self.node._io(txn, &*self.value)._deserialize_vec(txn, self.rvar, self.xvar, off, src)
  }
}

pub struct Val<V> {
  node:     Rc<ANode>,
  op:       Rc<AOp<V>>,
  value:    Option<RWVal<V>>,
  mode:     WriteMode,
  xref:     Rc<()>,
  xvar:     RWVar,
  rvar:     RVar,
  name:     Option<String>,
}

impl<V> Clone for Val<V> where V: 'static {
  fn clone(&self) -> Val<V> {
    let rvar = RVar::default();
    Val{
      node:     self.node.clone(),
      op:       self.op.clone(),
      //value:    self.value.as_ref().map(|v| v._clone()),
      value:    self._clone_value(),
      mode:     self.mode,
      xref:     self.xref.clone(),
      xvar:     self.xvar,
      rvar:     rvar,
      name:     self.name.clone(),
    }
  }
}

impl<V> Val<V> where V: 'static {
  pub fn nowrap<Op>(op: Rc<Op>, xvar: RWVar) -> Self where Op: AOp<V> + 'static {
    println!("WARNING: Val::nowrap: semantics of this are currently very unstable");
    let rvar = RVar::default();
    Val{
      node:     op.clone(),
      op:       op.clone(),
      value:    Some(op._make_value()),
      mode:     WriteMode::Exclusive,
      // FIXME
      xref:     Rc::new(()),
      xvar:     xvar,
      rvar:     rvar,
      name:     None,
    }
  }

  pub fn from<Op>(op: Rc<Op>) -> Self where Op: AOp<V> + 'static {
    let rvar = RVar::default();
    let xvar = RWVar(rvar);
    let val = Val{
      node:     op.clone(),
      op:       op.clone(),
      value:    Some(op._make_value()),
      mode:     WriteMode::Exclusive,
      xref:     Rc::new(()),
      xvar:     xvar,
      rvar:     rvar,
      name:     None,
    };
    log_static_graph(|logging| {
      logging.insert_node(val._graph_key());
    });
    // FIXME(peter, 20180619): disable wrappers while revamping ops.
    /*let val = WRAP_VAL_STACK.with(|stack| {
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
    });*/
    val
  }

  pub fn with_value<Op>(op: Rc<Op>, value: Option<RWVal<V>>) -> Self where Op: AOp<V> + 'static {
    let rvar = RVar::default();
    let xvar = RWVar(rvar);
    let val = Val{
      node:     op.clone(),
      op:       op.clone(),
      value:    value,
      mode:     WriteMode::Exclusive,
      xref:     Rc::new(()),
      xvar:     xvar,
      rvar:     rvar,
      name:     None,
    };
    log_static_graph(|logging| {
      logging.insert_node(val._graph_key());
    });
    val
  }

  pub fn with_value_mode<Op>(op: Rc<Op>, value: Option<RWVal<V>>, mode: WriteMode) -> Self where Op: AOp<V> + 'static {
    let rvar = RVar::default();
    let xvar = RWVar(rvar);
    let val = Val{
      node:     op.clone(),
      op:       op.clone(),
      value:    value,
      mode:     mode,
      xref:     Rc::new(()),
      xvar:     xvar,
      rvar:     rvar,
      name:     None,
    };
    log_static_graph(|logging| {
      logging.insert_node(val._graph_key());
    });
    val
  }

  pub fn _to_node(&self) -> Node {
    Node{
      node:     self.node.clone(),
      mode:     self.mode,
      //value:    Rc::new(self.value.as_ref().map(|v| v._clone())),
      value:    Rc::new(self._clone_value()),
      //xref:     self.xref.clone(),
      xvar:     self.xvar,
      // NOTE: Should the node corresponding to a val share the same varkeys?
      rvar:     self.rvar,
      name:     self.name.clone(),
    }
  }

  pub fn into_node(self) -> Node {
    Node{
      node:     self.node.clone(),
      mode:     self.mode,
      //value:    Rc::new(self.value.as_ref().map(|v| v._clone())),
      value:    Rc::new(self._clone_value()),
      //xref:     self.xref.clone(),
      xvar:     self.xvar,
      rvar:     self.rvar,
      name:     self.name.clone(),
    }
  }

  /*pub fn downgrade(&self) -> OVal<V> {
    OVal{
      rvar: self.rvar,
      xvar: self.xvar,
      xval: self.op._value()._clone(),
    }
  }*/

  fn _exact_clone(&self) -> Val<V> {
    Val{
      node:     self.node.clone(),
      op:       self.op.clone(),
      //value:    self.value._clone(),
      //value:    self.value.as_ref().map(|v| v._clone()),
      value:    self._clone_value(),
      mode:     self.mode,
      xref:     self.xref.clone(),
      xvar:     self.xvar,
      rvar:     self.rvar,
      name:     self.name.clone(),
    }
  }

  pub fn duplicate(&self) -> Val<V> {
    let rvar = RVar::default();
    let xvar = RWVar(rvar);
    let val = Val{
      node:     self.node.clone(),
      op:       self.op.clone(),
      value:    Some(self._make_value()),
      mode:     self.mode,
      xref:     Rc::new(()),
      xvar:     xvar,
      rvar:     rvar,
      name:     None,
    };
    log_static_graph(|logging| {
      logging.insert_node(val._graph_key());
      logging.insert_alias_edge(self._graph_key(), val._graph_key());
    });
    val
  }

  pub fn accumulate_value(&self, new_value: Option<RWVal<V>>) -> Val<V> {
    let rvar = RVar::default();
    let xvar = RWVar(rvar);
    let val = Val{
      node:     self.node.clone(),
      op:       self.op.clone(),
      value:    new_value,
      mode:     WriteMode::Accumulate,
      xref:     Rc::new(()),
      xvar:     xvar,
      rvar:     rvar,
      name:     None,
    };
    log_static_graph(|logging| {
      logging.insert_node(val._graph_key());
      logging.insert_alias_edge(self._graph_key(), val._graph_key());
    });
    val
  }

  pub fn accumulate(&self) -> Val<V> {
    let rvar = RVar::default();
    let xvar = RWVar(rvar);
    let val = Val{
      node:     self.node.clone(),
      op:       self.op.clone(),
      value:    Some(self._make_value()),
      mode:     WriteMode::Accumulate,
      xref:     Rc::new(()),
      xvar:     xvar,
      rvar:     rvar,
      name:     None,
    };
    log_static_graph(|logging| {
      logging.insert_node(val._graph_key());
      logging.insert_alias_edge(self._graph_key(), val._graph_key());
    });
    val
  }

  pub fn clobber(&self) -> Val<V> {
    let rvar = RVar::default();
    let xvar = RWVar(rvar);
    let val = Val{
      node:     self.node.clone(),
      op:       self.op.clone(),
      value:    Some(self._make_value()),
      mode:     WriteMode::Clobber,
      xref:     Rc::new(()),
      xvar:     xvar,
      rvar:     rvar,
      name:     None,
    };
    log_static_graph(|logging| {
      logging.insert_node(val._graph_key());
      logging.insert_alias_edge(self._graph_key(), val._graph_key());
    });
    val
  }

  pub fn named(&self, name: &str) -> Val<V> {
    let rvar = RVar::default();
    Val{
      node:     self.node.clone(),
      op:       self.op.clone(),
      //value:    self.value._clone(),
      //value:    self.value.as_ref().map(|v| v._clone()),
      value:    self._clone_value(),
      mode:     self.mode,
      xref:     self.xref.clone(),
      xvar:     self.xvar,
      rvar:     rvar,
      name:     Some(name.to_owned()),
    }
  }

  pub fn name(&self) -> Option<String> {
    self.name.clone()
  }

  pub fn _graph_key(&self) -> (u64, String) {
    let base_opkey = self.op._key().replace("anode::", "").replace("ops::", "");
    let opkey = match self.mode {
      WriteMode::Exclusive => base_opkey,
      WriteMode::Accumulate => format!("{}(Accumulate)", base_opkey),
      WriteMode::Clobber => format!("{}(Clobber)", base_opkey),
    };
    (self.xvar._raw(), opkey)
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
    //self.op._apply(txn, self.rvar, self.xvar, self.mode);
    self.op._apply_output(txn, OVal::with_value(self.rvar, self.xvar, self.mode, self._static_value()));
  }

  pub fn _eval_recursive(&self, txn: Txn) {
    if 80 == self.xvar._raw() {
      println!("DEBUG: Val: eval_recursive: this is {:?}", self._graph_key());
    }
    if 89 == self.xvar._raw() {
      println!("DEBUG: Val: eval_recursive: this is {:?}", self._graph_key());
    }
    //if Some(txn) != self._value3(txn).txn() || self._value3(txn).incomplete() {
    if Some(txn) != self._value3(txn)._complete_txn() {
      if 80 == self.xvar._raw() {
        println!("DEBUG: Val: eval_recursive:   enter");
      }
      if 89 == self.xvar._raw() {
        println!("DEBUG: Val: eval_recursive:   enter");
      }
      self.op._eval_recursive(txn, self.rvar, self.xvar);
    }
  }

  pub fn eval(&self, txn: Txn) {
    self._eval_recursive(txn);
    self._apply(txn);
  }

  pub fn _valref(&self) -> Option<RWValRef> {
    self.value.as_ref().map(|v| v._ref())
  }

  pub fn var(&self) -> RWVar {
    self.xvar
  }

  pub fn _ref_count(&self) -> usize {
    // FIXME: this is not the refcount we want.
    //Rc::strong_count(&self.op)
    Rc::strong_count(&self.xref)
  }

  pub fn reset(&self) {
    //self.op._reset();
    self.value.as_ref().map(|v| v.reset());
  }

  pub fn release(&self) {
    //self.op._release();
    self.value.as_ref().map(|v| v.release());
  }

  pub fn persist(&self, txn: Txn) {
    //self.op._value().persist(txn, self.xvar);
    //self.op._persist_output(txn, self.xvar, self._clone_value());
    let xvalue = self.op._value3(txn, self.value.as_ref());
    xvalue.persist(txn, self.xvar);
  }

  pub fn write(&self, txn: Txn) -> Option<(WriteCap, WriteToken)> {
    //self.op._value().write(txn, self.xvar, self.mode)
    //self.op._write_output(txn, self.xvar, self.mode, self._clone_value());
    //let xvalue = self.op._value2(txn, self._static_value());
    let xvalue = self.op._value3(txn, self.value.as_ref());
    xvalue.write(txn, self.xvar, self.mode)
  }

  pub fn get(&self, txn: Txn) -> RwLockReadGuard<V> {
    //self.op._value().get(txn, self.rvar)
    //self.op._get_output(txn, self.rvar, self._clone_value());
    //let xvalue = self.op._value2(txn, self._static_value());
    let xvalue = self.op._value3(txn, self.value.as_ref());
    //xvalue.get(txn, self.rvar)
    xvalue._get_debug(txn, self.rvar, self.name.as_ref().map(|s| s.as_ref()), self._graph_key())
  }

  pub fn get_mut(&self, txn: Txn, token: WriteToken) -> RwLockWriteGuard<V> {
    //self.op._value().get_mut(txn, self.xvar, token)
    //self.op._get_mut_output(txn, self.xvar, token, self._clone_value());
    //let xvalue = self.op._value2(txn, self._static_value());
    let xvalue = self.op._value3(txn, self.value.as_ref());
    xvalue.get_mut(txn, self.xvar, token)
  }

  pub fn finish_write(&self, txn: Txn, token: WriteToken) {
    let xvalue = self.op._value3(txn, self.value.as_ref());
    xvalue.finish_write(txn, self.xvar, token);
  }

  pub fn set<F: FnOnce(RwLockWriteGuard<V>)>(&self, txn: Txn, f: F) {
    //self.op._value().set(txn, self.xvar, self.mode, f);
    //self.op._set_output(txn, self.xvar, self.mode, self._clone_value());
    let xvalue = self.op._value3(txn, self.value.as_ref());
    xvalue.set(txn, self.xvar, self.mode, f);
  }

  fn _io<'a>(&'a self, txn: Txn) -> &'a IOVal {
    self._value3(txn)
  }

  fn _value2(&self, txn: Txn) -> RWVal<V> {
    self.op._value2(txn, self._static_value())
  }

  fn _value3<'a>(&'a self, txn: Txn) -> &'a RWVal<V> {
    self.op._value3(txn, self.value.as_ref())
  }

  fn _clone_value(&self) -> Option<RWVal<V>> {
    self.value.as_ref().map(|v| v._clone())
  }

  //pub fn _static_value(&self) -> RWVal<V> {
  pub fn _static_value(&self) -> Option<RWVal<V>> {
    //self.op._value()._clone()
    self._clone_value()
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
    // FIXME: for debugging.
    let adj = match self.name {
      None => adj,
      Some(ref name) => {
        match adj.name() {
          None => adj.named(&format!("adj.{}", name)),
          Some(ref prev_name) => adj.named(&format!("adj.{} (was: {})", name, prev_name)),
        }
      }
    };
    sink.put_adj::<V>(self.var(), adj);
  }
}

impl<V> IONodeExt for Val<V> where V: 'static {
  fn serialize(&self, txn: Txn, dst: &mut Any) {
    self._value3(txn)._serialize(txn, self.rvar, dst);
  }

  fn deserialize(&self, txn: Txn, src: &mut Any) {
    self._value3(txn)._deserialize(txn, self.xvar, src);
  }
}

impl<V> VIONodeExt for Val<V> where V: 'static {
  fn _serialize_vec(&self, txn: Txn, off: usize, dst: &mut Any) -> usize {
    self._value3(txn)._serialize_vec(txn, self.rvar, off, dst)
  }

  fn _deserialize_vec(&self, txn: Txn, off: usize, src: &mut Any) -> usize {
    self._value3(txn)._deserialize_vec(txn, self.rvar, self.xvar, off, src)
  }
}

pub struct OVal<V> {
  value:    Option<RWVal<V>>,
  rvar:     RVar,
  xvar:     RWVar,
  mode:     WriteMode,
}

impl<V> OVal<V> where V: 'static {
  pub fn new(rvar: RVar, xvar: RWVar, mode: WriteMode, value: RWVal<V>) -> Self {
    OVal{
      value:    Some(value),
      rvar:     rvar,
      xvar:     xvar,
      mode:     mode,
    }
  }

  pub fn with_value(rvar: RVar, xvar: RWVar, mode: WriteMode, value: Option<RWVal<V>>) -> Self {
    OVal{
      value:    value,
      rvar:     rvar,
      xvar:     xvar,
      mode:     mode,
    }
  }

  pub fn _valref(&self) -> Option<RWValRef> {
    self.value.as_ref().map(|v| v._ref())
  }

  pub fn var(&self) -> RWVar {
    self.xvar
  }

  pub fn persist(&self, txn: Txn) {
    assert!(self.value.is_some());
    self.value.as_ref().unwrap().persist(txn, self.xvar);
  }

  pub fn write(&self, txn: Txn) -> Option<(WriteCap, WriteToken)> {
    assert!(self.value.is_some());
    self.value.as_ref().unwrap().write(txn, self.xvar, self.mode)
  }

  pub fn write_<F>(&self, txn: Txn, f: F) -> bool where F: FnOnce(WriteCap, WriteToken) {
    assert!(self.value.is_some());
    match self.value.as_ref().unwrap().write(txn, self.xvar, self.mode) {
      Some((cap, token)) => {
        f(cap, token);
        true
      }
      None => {
        false
      }
    }
  }

  pub fn get(&self, txn: Txn) -> RwLockReadGuard<V> {
    assert!(self.value.is_some());
    self.value.as_ref().unwrap().get(txn, self.rvar)
  }

  pub fn get_mut(&self, txn: Txn, token: WriteToken) -> RwLockWriteGuard<V> {
    assert!(self.value.is_some());
    self.value.as_ref().unwrap().get_mut(txn, self.xvar, token)
  }

  pub fn finish_write(&self, txn: Txn, token: WriteToken) {
    assert!(self.value.is_some());
    self.value.as_ref().unwrap().finish_write(txn, self.xvar, token);
  }

  pub fn set<F: FnOnce(RwLockWriteGuard<V>)>(&self, txn: Txn, f: F) {
    assert!(self.value.is_some());
    self.value.as_ref().unwrap().set(txn, self.xvar, self.mode, f);
  }
}

pub struct FeedFwd {
  tng_map:  HashMap<RWVar, (Node, Rc<Any>)>,
}

pub fn sink<V: 'static>(sink_: Val<V>) -> Sink {
  Sink::new(sink_)
}

pub struct Sink {
  adj_map:  HashMap<RWVar, Vec<(Node, Rc<Any>)>>,
  join_map: HashMap<RWVar, (Node, Rc<Any>)>,
  frozen:   HashSet<RWVar>,
  volatile: HashSet<RWVar>,
  nopass:   HashSet<RWVar>,
  pass:     HashSet<RWVar>,
}

impl Sink {
  pub fn new<V>(sink_: Val<V>) -> Self where V: 'static {
    let sink_adj = match <OnesSrcOp as OnesSrcOpMaybeExt<V>>::maybe_build_like(sink_.clone()) {
      None => unimplemented!("FATAL: Sink: missing `ones` builder for sink val"),
      Some(adj) => adj,
    };
    Sink::with_adj(sink_, sink_adj)
  }

  pub fn with_adj<V>(sink_: Val<V>, sink_adj: Val<V>) -> Self where V: 'static {
    let mut sink = Sink{
      adj_map:  HashMap::new(),
      join_map: HashMap::new(),
      frozen:   HashSet::new(),
      volatile: HashSet::new(),
      nopass:   HashSet::new(),
      pass:     HashSet::new(),
    };
    let p = pass();
    sink_._push_fwd(None, p, &mut |_node, _rvar, _xvar| {});
    sink_.put_adjoint(sink_adj, &mut sink);
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
    let &mut Sink{
      ref mut adj_map,
      ref mut join_map,
      ref mut frozen,
      ref mut volatile,
      ref mut nopass,
      ref mut pass,
    } = self;
    frozen.insert(var);
    if adj_map.contains_key(&var) {
      let adjs = adj_map.get(&var).unwrap();
      match adjs.len() {
        0 => {}
        1 => {
          match (adjs[0].1).downcast_ref::<Val<V>>() {
            None => panic!(),
            Some(adj_op) => {
              //println!("DEBUG: Sink: single:   adjoint ref count: {} ({:?})", adj_op._ref_count(), adj_op.name());
              return Some(adj_op.clone());
            }
          }
        }
        _ => {
          if join_map.contains_key(&var) {
            let &(_, ref join_any_op) = join_map.get(&var).unwrap();
            match join_any_op.downcast_ref::<Val<V>>() {
              None => panic!(),
              Some(join_op) => {
                //println!("DEBUG: Sink: join:     adjoint ref count: {} ({:?})", join_op._ref_count(), join_op.name());
                return Some(join_op.clone());
              }
            }
          } else {
            //let no_volatile_adjs = false;
            let no_volatile_adjs = true;
            let adj_ops: Vec<_> = adjs.iter().map(|&(_, ref a)| {
              match a.downcast_ref::<Val<V>>() {
                None => panic!(),
                Some(adj_op) => {
                  //println!("DEBUG: Sink: multiple: adjoint ref count: {} ({:?})", adj_op._ref_count(), adj_op.name());
                  /*// TODO
                  if volatile.contains(&adj_op.var()) {
                    no_volatile_adjs = false;
                  }*/
                  if adj_op._ref_count() <= 1 {
                    nopass.insert(adj_op.var());
                    //adj_op.clone()
                  } else {
                    pass.insert(adj_op.var());
                    if nopass.contains(&adj_op.var()) {
                      println!("WARNING: Sink: adjoint op may be executed twice");
                    }
                    //println!("DEBUG: Sink: multiple:   create pass");
                    //adj_op.clone().pass()
                  }
                  //adj_op.clone().pass()
                  adj_op.clone()
                }
              }
            }).collect();
            // TODO: if none of `adj_ops` are volatile (TODO: define volatile),
            // transform this join into an in-place/accumulate op.
            // FIXME: also gate on an "explicitly allow non-volatile" flag.
            let join = if no_volatile_adjs {
              for adj_op in adj_ops.iter() {
                //println!("DEBUG: Sink: multiple: adjoint ref count: {} ({:?})", adj_op._ref_count(), adj_op.name());
                frozen.insert(adj_op.var());
              }
              match <SumJoinOp as SumJoinOpMaybeExt<V>>::maybe_build_inplace(adj_ops) {
                None => panic!("FATAL: Sink::get_adj(): failed to sum adjoints inplace"),
                Some((join, _)) => join,
              }
            } else {
              match <SumJoinOp as SumJoinOpMaybeExt<V>>::maybe_build(adj_ops) {
                None => panic!("FATAL: Sink::get_adj(): failed to sum adjoints"),
                Some(join) => join,
              }
            };
            let join_clone = join.clone();
            join_map.insert(var, (join_clone._to_node(), Rc::new(join_clone)));
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
      /*adjs.push((adj_op.clone().into_node(), Rc::new(adj_op)));*/
      adjs.push((adj_op._to_node(), Rc::new(adj_op)));
    } else {
      /*self.adj_map.insert(var, vec![(adj_op.clone().into_node(), Rc::new(adj_op))]);*/
      self.adj_map.insert(var, vec![(adj_op._to_node(), Rc::new(adj_op))]);
    }
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
  //mode:         WriteMode,
  curr_txn:     Option<Txn>,
  complete:     bool,
  l_consumers:  Mutex<HashSet<RVar>>,
  d_consumers:  HashSet<RVar>,
  l_producers:  HashSet<RWVar>,
  d_producers:  HashSet<RWVar>,
  data:         Option<T>,
}

impl<T> Default for RWValBuf<T> {
  fn default() -> Self {
    RWValBuf{
      //mode:         WriteMode::Exclusive,
      curr_txn:     None,
      complete:     false,
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
  ref_:     RWValRef,
  alloc:    Arc<Fn(Txn) -> T>,
  buf:      Arc<RwLock<RWValBuf<T>>>,
  borrow:   (),
}

impl<T> AnyRWVal for RWVal<T> where T: 'static {
  /*fn _txn(&self) -> Option<Txn> {
    self.txn()
  }*/

  fn _complete_txn(&self) -> Option<Txn> {
    match self.incomplete() {
      false => self.txn(),
      true  => None,
    }
  }

  fn _persist(&self, txn: Txn, xvar: RWVar) {
    self.persist(txn, xvar);
  }
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
    let ref_ = RWValRef::default();
    let buf = Arc::new(RwLock::new(RWValBuf::default()));
    RWVal{
      ref_:     ref_,
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
      ref_:     self.ref_,
      alloc:    self.alloc.clone(),
      buf:      self.buf.clone(),
      borrow:   (),
    }
  }

  /*pub fn _set_accumulate(&self) {
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
  }*/

  pub fn _ref(&self) -> RWValRef {
    self.ref_
  }

  pub fn txn(&self) -> Option<Txn> {
    let buf = self.buf.read();
    buf.curr_txn
  }

  pub fn incomplete(&self) -> bool {
    let buf = self.buf.read();
    !buf.complete
  }

  pub fn reset(&self) {
    let mut buf = self.buf.write();
    buf.curr_txn = None;
    buf.complete = false;
    buf.l_consumers.lock().clear();
    buf.d_consumers.clear();
    buf.l_producers.clear();
    buf.d_producers.clear();
  }

  pub fn release(&self) {
    let mut buf = self.buf.write();
    buf.curr_txn = None;
    buf.complete = false;
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
      buf.complete = true;
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

    if buf.data.is_none() {
      buf.data = Some((self.alloc)(txn));
    }
  }

  pub fn write(&self, txn: Txn, xvar: RWVar, mode: WriteMode) -> Option<(WriteCap, WriteToken)> {
    let mut buf = self.buf.write();
    let &mut RWValBuf{
        ref mut curr_txn,
        ref mut complete,
        ref l_consumers,
        ref mut d_consumers,
        ref mut l_producers,
        ref mut d_producers,
        ..} = &mut *buf;
    let mut l_consumers = l_consumers.lock();

    let new_txn = curr_txn.is_none() || curr_txn.unwrap() != txn;
    if new_txn {
      *curr_txn = Some(txn);
      *complete = false;
      l_consumers.clear();
      d_consumers.clear();
      l_producers.clear();
      d_producers.clear();
    }

    match mode {
      WriteMode::Exclusive => {
        match (l_producers.len(), d_producers.len()) {
          (0, 0) => {}
          (1, 0) => {
            if l_producers.contains(&xvar) {
              return None;
            }
            panic!("attempting second write to `Exclusive` val");
          }
          (_, 0) => panic!("attempting multiple writes to `Exclusive` val"),
          (_, _) => panic!("all writes to `Exclusive` val must be live"),
        }
        *complete = true;
        assert!(l_consumers.is_empty(),
            "attempting write to `Exclusive` val after read");
      }
      WriteMode::Accumulate => {
        match (l_producers.len(), d_producers.len()) {
          (0, 0) => {}
          (_, 0) => {
            if l_producers.contains(&xvar) {
              return None;
            }
          }
          (_, _) => panic!("all writes to `Accumulate` val must be live"),
        }
        assert!(!*complete);
        assert!(l_consumers.is_empty(),
            "attempting write to `Accumulate` val after read");
      }
      WriteMode::Clobber => {
        match (l_producers.len(), d_producers.len()) {
          (0, 0) => {}
          (1, _) => {
            if l_producers.contains(&xvar) {
              return None;
            }
          }
          (_, _) => panic!("attempting multiple live writes to `Clobber` val"),
        }
        *complete = true;
        d_consumers.extend(l_consumers.drain());
        d_producers.extend(l_producers.drain());
      }
    }

    let first = l_producers.is_empty();
    let cap = match (mode, first) {
      (WriteMode::Accumulate, false) => WriteCap::Accumulate,
      (_, true) => WriteCap::Assign,
      _ => unreachable!(),
    };
    l_producers.insert(xvar);
    Some((cap, WriteToken{xvar: xvar, first: first, borrow: &self.borrow}))
  }

  pub fn _get_debug(&self, txn: Txn, rvar: RVar, name: Option<&str>, key: (u64, String)) -> RwLockReadGuard<T> {
    let buf = self.buf.read();

    let mut valid_txn = false;
    if let Some(curr_txn) = buf.curr_txn {
      if curr_txn == txn {
        valid_txn = true;
      }
    }
    assert!(valid_txn,
        //"attempting a read with an invalid txn (did you forget to `persist` or `write`?) name: {:?}", name);
        "attempting a read with an invalid txn (did you forget to `persist` or `write`?) key: {:?}", key);

    assert!(buf.complete,
        "attempting an incomplete read");
    assert!(!buf.d_consumers.contains(&rvar),
        "attempting a stale read (the value has been clobbered)");
    match buf.l_producers.len() {
      0 => panic!("attempting an invalid read (the value was never written)"),
      _ => {}
    }
    buf.l_consumers.lock().insert(rvar);

    assert!(buf.data.is_some(),
        "attempting a read on empty data");

    RwLockReadGuard::map(buf, |buf| buf.data.as_ref().unwrap())
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

    assert!(buf.complete,
        "attempting an incomplete read");
    assert!(!buf.d_consumers.contains(&rvar),
        "attempting a stale read (the value has been clobbered)");
    match buf.l_producers.len() {
      0 => panic!("attempting an invalid read (the value was never written)"),
      _ => {}
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

  pub fn finish_write(&self, txn: Txn, xvar: RWVar, token: WriteToken) {
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

    assert!(!buf.complete,
        "attempting to finish write, but txn already complete");
    buf.complete = true;
    assert!(buf.l_consumers.lock().is_empty(),
        "attempting a write-after-read (check your `get` and `get_mut` order)");
    assert!(!buf.d_producers.contains(&xvar),
        "attempting an invalid write (the value has been clobbered)");
    assert!(buf.l_producers.contains(&xvar),
        "attempting an invalid write (did you forget to `write`?)");

    assert!(buf.data.is_some(),
        "attempting to finish write on empty data");
  }

  pub fn set<F>(&self, txn: Txn, xvar: RWVar, mode: WriteMode, f: F) where F: FnOnce(RwLockWriteGuard<T>) {
    if let Some((cap, token)) = self.write(txn, xvar, mode) {
      match cap {
        WriteCap::Assign => {
          f(self.get_mut(txn, xvar, token));
        }
        _ => unimplemented!(),
      }
    }
  }
}

pub struct OpBase {
  ref_:     OpRef,
  stack:    WalkStack,
  tags:     CloneMap,
  //tng_op:   RefCell<Option<Val<V>>>,
}

impl Default for OpBase {
  fn default() -> Self {
    OpBase{
      ref_:     OpRef::default(),
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

pub struct FSrcWrapOp<F, V> {
  base: OpBase,
  ext:  OpExt<F, V>,
  cfg:  RefCell<F>,
  ctrl: Vec<Node>,
  val_: Val<V>,
}

impl<F, V> FSrcWrapOp<F, V> where V: 'static {
  pub fn new(cfg: F, ext: OpExt<F, V>, val_: Val<V>) -> Self {
    log_static_graph(|logging| {
      logging.push_pred(val_._graph_key());
    });
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

impl<F, V> ANode for FSrcWrapOp<F, V> where V: 'static {
  fn _key(&self) -> String {
    unsafe { type_name::<F>() }.to_owned()
  }

  fn _walk(&self) -> &Walk {
    &self.base.stack
  }

  /*fn _io(&self) -> &IOVal {
    self.val_._op()._io()
  }*/

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
      self.val_._push_fwd(stop_txn, pass, apply);
      apply(self, rvar, xvar);
    }
  }

  fn _pop_rev(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    if self.base.stack.pop(pass) {
      apply(self, rvar, xvar);
      self.val_._pop_rev(stop_txn, pass, apply);
    }
  }

  /*fn _txn(&self) -> Option<Txn> {
    self.val_._op()._txn()
  }*/

  /*fn _reset(&self) {
    self.val_.reset();
  }

  fn _release(&self) {
    self.val_.release();
  }*/

  fn _io<'a>(&'a self, txn: Txn, static_any_value: &'a Any) -> &'a IOVal {
    if let Some(static_value) = static_any_value.downcast_ref::<Option<RWVal<V>>>() {
      if static_value.is_some() {
        return static_value.as_ref().unwrap();
      } else {
        // FIXME
        unimplemented!();
      }
    } else {
      unreachable!();
    }
  }

  /*fn _apply(&self, txn: Txn, rvar: RVar, xvar: RWVar, mode: WriteMode) {
    //println!("DEBUG: FWrap: apply");
    self._apply_output(txn, OVal::new(rvar, xvar, mode, self._value()._clone()));
  }*/

  fn _apply_any(&self, txn: Txn, rvar: RVar, xvar: RWVar, mode: WriteMode, any_value: Rc<Any>) {
    if let Some(value) = any_value.downcast_ref::<Option<RWVal<V>>>() {
      self._apply_output(txn, OVal::with_value(rvar, xvar, mode, value.as_ref().map(|v| v._clone())));
    } else {
      unreachable!();
    }
  }

  fn _eval_recursive(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    //println!("DEBUG: FWrap: eval recursive");
    //if Some(txn) != self._txn() {
      for node in self.ctrl.iter() {
        node.eval(txn);
      }
      //println!("DEBUG: FWrap: inner eval recursive");
      self.val_._eval_recursive(txn);
    //}
  }
}

impl<F, V> AOp<V> for FSrcWrapOp<F, V> where V: 'static {
  fn _value2(&self, txn: Txn, static_value: Option<RWVal<V>>) -> RWVal<V> {
    if static_value.is_some() {
      return static_value.as_ref().unwrap()._clone();
    } else {
      // FIXME
      unimplemented!();
    }
  }

  fn _value3<'a>(&'a self, txn: Txn, static_value: Option<&'a RWVal<V>>) -> &'a RWVal<V> {
    if static_value.is_some() {
      return static_value.unwrap();
    } else {
      // FIXME
      unimplemented!();
    }
  }

  fn _make_value(&self) -> RWVal<V> {
    (self.ext.make_val)(self.cfg.borrow_mut())
  }

  /*fn _value(&self) -> &RWVal<V> {
    self.val_._op()._value()
  }*/

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
        None => {
          self.val_._pop_rev(None, pass, &mut |_, _, _| {});
        }
        Some(ref adjoint) => {
          (adjoint)(pass, this, self.cfg.borrow_mut(), sink);
          self.val_._pop_adjoint(pass, sink);
        }
      }
    }
  }
}

pub struct FSrcOp<F, V> {
  base: OpBase,
  ext:  OpExt<F, V>,
  cfg:  RefCell<F>,
  ctrl: Vec<Node>,
  //val:  RWVal<V>,
}

impl<F, V> FSrcOp<F, V> where V: 'static {
  pub fn new(cfg: F, ext: OpExt<F, V>) -> Self {
    let cfg = RefCell::new(cfg);
    //let val = (ext.make_val)(cfg.borrow_mut());
    FSrcOp{
      base: OpBase::default(),
      ext:  ext,
      cfg:  cfg,
      ctrl: vec![],
      //val:  val,
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

impl<F, V> ANode for FSrcOp<F, V> where V: 'static {
  fn _key(&self) -> String {
    unsafe { type_name::<F>() }.to_owned()
  }

  fn _walk(&self) -> &Walk {
    &self.base.stack
  }

  /*fn _io(&self) -> &IOVal {
    &self.val
  }*/

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

  fn _pop_rev(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    if self.base.stack.pop(pass) {
      apply(self, rvar, xvar);
    }
  }

  /*fn _txn(&self) -> Option<Txn> {
    self.val.txn()
  }*/

  /*fn _reset(&self) {
    self._value().reset();
  }

  fn _release(&self) {
    self._value().release();
  }*/

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

  fn _io<'a>(&'a self, txn: Txn, static_any_value: &'a Any) -> &'a IOVal {
    if let Some(static_value) = static_any_value.downcast_ref::<Option<RWVal<V>>>() {
      if static_value.is_some() {
        return static_value.as_ref().unwrap();
      } else {
        // FIXME
        unimplemented!();
      }
    } else {
      unreachable!();
    }
  }

  /*fn _apply(&self, txn: Txn, rvar: RVar, xvar: RWVar, mode: WriteMode) {
    //println!("DEBUG: FSrcOp: apply");
    self._apply_output(txn, OVal::new(rvar, xvar, mode, self._value()._clone()));
  }*/

  fn _apply_any(&self, txn: Txn, rvar: RVar, xvar: RWVar, mode: WriteMode, any_value: Rc<Any>) {
    if let Some(value) = any_value.downcast_ref::<Option<RWVal<V>>>() {
      self._apply_output(txn, OVal::with_value(rvar, xvar, mode, value.as_ref().map(|v| v._clone())));
    } else {
      unreachable!();
    }
  }

  fn _eval_recursive(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    //println!("DEBUG: FSrcOp: eval recursive");
    //if Some(txn) != self._txn() {
      for node in self.ctrl.iter() {
        node.eval(txn);
      }
    //}
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

impl<F, V> AOp<V> for FSrcOp<F, V> where V: 'static {
  fn _value2(&self, txn: Txn, static_value: Option<RWVal<V>>) -> RWVal<V> {
    if static_value.is_some() {
      return static_value.as_ref().unwrap()._clone();
    } else {
      // FIXME
      unimplemented!();
    }
  }

  fn _value3<'a>(&'a self, txn: Txn, static_value: Option<&'a RWVal<V>>) -> &'a RWVal<V> {
    if static_value.is_some() {
      return static_value.unwrap();
    } else {
      // FIXME
      unimplemented!();
    }
  }

  fn _make_value(&self) -> RWVal<V> {
    //(self.ext.make_val)()
    (self.ext.make_val)(self.cfg.borrow_mut())
  }

  /*fn _value(&self) -> &RWVal<V> {
    &self.val
  }*/

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

impl<F, V, W> F1WrapOp<F, V, W> where V: 'static, W: 'static {
  pub fn new(cfg: F, ext: OpExt<F, W>, x_: Val<V>, y_: Val<W>) -> Self {
    log_static_graph(|logging| {
      logging.push_pred(x_._graph_key());
      logging.push_pred(y_._graph_key());
    });
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

impl<F, V, W> ANode for F1WrapOp<F, V, W> where V: 'static, W: 'static {
  fn _key(&self) -> String {
    unsafe { type_name::<F>() }.to_owned()
  }

  fn _walk(&self) -> &Walk {
    &self.base.stack
  }

  /*fn _io(&self) -> &IOVal {
    self.y_._op()._io()
  }*/

  fn _analysis_tags(&self) -> &AnalysisTags {
    &self.base
  }

  fn _pred_fwd(&self, pred_buf: &mut Vec<Node>) {
    pred_buf.push(self.x_._to_node());
  }

  fn _pred_rev(&self, pred_buf: &mut Vec<Node>) {
    pred_buf.push(self.x_._to_node());
  }

  fn _push(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.push(pass) {
      //if stop_txn.is_none() || stop_txn != self._txn() {
        self.x_._node()._push(stop_txn, pass, apply);
      //}
      apply(self);
    }
  }

  fn _pop(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.pop(pass) {
      apply(self);
      //if stop_txn.is_none() || stop_txn != self._txn() {
        self.x_._node()._pop(stop_txn, pass, apply);
      //}
    }
  }

  fn _push_fwd(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    if self.base.stack.push(pass) {
      //if stop_txn.is_none() || stop_txn != self._txn() {
        self.x_._push_fwd(stop_txn, pass, apply);
      //}
      self.y_._push_fwd(stop_txn, pass, apply);
      apply(self, rvar, xvar);
    }
  }

  fn _pop_rev(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    if self.base.stack.pop(pass) {
      apply(self, rvar, xvar);
      self.y_._pop_rev(stop_txn, pass, apply);
      //if stop_txn.is_none() || stop_txn != self._txn() {
        self.x_._pop_rev(stop_txn, pass, apply);
      //}
    }
  }

  /*fn _txn(&self) -> Option<Txn> {
    self.y_._op()._txn()
  }*/

  /*fn _reset(&self) {
    self._value().reset();
  }

  fn _release(&self) {
    self._value().release();
  }*/

  fn _io<'a>(&'a self, txn: Txn, static_any_value: &'a Any) -> &'a IOVal {
    if let Some(static_value) = static_any_value.downcast_ref::<Option<RWVal<W>>>() {
      if static_value.is_some() {
        return static_value.as_ref().unwrap();
      } else {
        // FIXME
        unimplemented!();
      }
    } else {
      unreachable!();
    }
  }

  /*fn _apply(&self, txn: Txn, rvar: RVar, xvar: RWVar, mode: WriteMode) {
    self._apply_output(txn, OVal::new(rvar, xvar, mode, self._value()._clone()));
  }*/

  fn _apply_any(&self, txn: Txn, rvar: RVar, xvar: RWVar, mode: WriteMode, any_value: Rc<Any>) {
    if let Some(value) = any_value.downcast_ref::<Option<RWVal<W>>>() {
      self._apply_output(txn, OVal::with_value(rvar, xvar, mode, value.as_ref().map(|v| v._clone())));
    } else {
      unreachable!();
    }
  }

  fn _eval_recursive(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    //if Some(txn) != self._txn() {
      for node in self.ctrl.iter() {
        node.eval(txn);
      }
      self.x_.eval(txn);
      self.y_._eval_recursive(txn);
    //}
  }
}

impl<F, V, W> AOp<W> for F1WrapOp<F, V, W> where V: 'static, W: 'static {
  fn _value2(&self, txn: Txn, static_value: Option<RWVal<W>>) -> RWVal<W> {
    if static_value.is_some() {
      return static_value.as_ref().unwrap()._clone();
    } else {
      // FIXME
      unimplemented!();
    }
  }

  fn _value3<'a>(&'a self, txn: Txn, static_value: Option<&'a RWVal<W>>) -> &'a RWVal<W> {
    if static_value.is_some() {
      return static_value.unwrap();
    } else {
      // FIXME
      unimplemented!();
    }
  }

  fn _make_value(&self) -> RWVal<W> {
    (self.ext.make_val)(self.cfg.borrow_mut())
  }

  /*fn _value(&self) -> &RWVal<W> {
    self.y_._op()._value()
  }*/

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
        None => {
          self.y_._pop_rev(None, pass, &mut |_, _, _| {});
          self.x_._pop_rev(None, pass, &mut |_, _, _| {});
        }
        Some(ref adjoint) => {
          (adjoint)(pass, this, self.cfg.borrow_mut(), sink);
          self.y_._pop_adjoint(pass, sink);
          self.x_._pop_adjoint(pass, sink);
        }
      }
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
  //y:    RWVal<W>,
}

impl<F, V1, W> F1Op<F, V1, W> where V1: 'static, W: 'static {
  //pub fn new(cfg: F, ext: OpExt<F, W>, x_: Val<V1>, y: RWVal<W>) -> Self {
  pub fn new(cfg: F, ext: OpExt<F, W>, x_: Val<V1>) -> Self {
    log_static_graph(|logging| {
      logging.push_pred(x_._graph_key());
    });
    let state = RefCell::new(cfg);
    //let y = (ext.make_val)(state.borrow_mut());
    F1Op{
      base: OpBase::default(),
      ext:  ext,
      //cfg:  RefCell::new(cfg),
      cfg:  state,
      ctrl: vec![],
      x_:   x_,
      //y:    y,
    }
  }
}

impl<F, V1, W> ANode for F1Op<F, V1, W> where V1: 'static, W: 'static {
  fn _key(&self) -> String {
    unsafe { type_name::<F>() }.to_owned()
  }

  fn _walk(&self) -> &Walk {
    &self.base.stack
  }

  /*fn _io(&self) -> &IOVal {
    &self.y
  }*/

  fn _analysis_tags(&self) -> &AnalysisTags {
    &self.base
  }

  fn _pred_fwd(&self, pred_buf: &mut Vec<Node>) {
    pred_buf.push(self.x_._to_node());
  }

  fn _pred_rev(&self, pred_buf: &mut Vec<Node>) {
    pred_buf.push(self.x_._to_node());
  }

  fn _push(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.push(pass) {
      //if stop_txn.is_none() || stop_txn != self._txn() {
        self.x_._node()._push(stop_txn, pass, apply);
      //}
      apply(self);
    }
  }

  fn _pop(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.pop(pass) {
      apply(self);
      //if stop_txn.is_none() || stop_txn != self._txn() {
        self.x_._node()._pop(stop_txn, pass, apply);
      //}
    }
  }

  fn _push_fwd(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    if self.base.stack.push(pass) {
      //if stop_txn.is_none() || stop_txn != self._txn() {
        self.x_._push_fwd(stop_txn, pass, apply);
      //}
      apply(self, rvar, xvar);
    }
  }

  fn _pop_rev(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    if self.base.stack.pop(pass) {
      apply(self, rvar, xvar);
      //if stop_txn.is_none() || stop_txn != self._txn() {
        self.x_._pop_rev(stop_txn, pass, apply);
      //}
    }
  }

  /*fn _txn(&self) -> Option<Txn> {
    self.y.txn()
  }*/

  /*fn _reset(&self) {
    self._value().reset();
  }

  fn _release(&self) {
    self._value().release();
  }*/

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

  fn _io<'a>(&'a self, txn: Txn, static_any_value: &'a Any) -> &'a IOVal {
    if let Some(static_value) = static_any_value.downcast_ref::<Option<RWVal<W>>>() {
      if static_value.is_some() {
        return static_value.as_ref().unwrap();
      } else {
        // FIXME
        unimplemented!();
      }
    } else {
      unreachable!();
    }
  }

  /*fn _apply(&self, txn: Txn, rvar: RVar, xvar: RWVar, mode: WriteMode) {
    //println!("DEBUG: F1Op: apply");
    self._apply_output(txn, OVal::new(rvar, xvar, mode, self._value()._clone()));
  }*/

  fn _apply_any(&self, txn: Txn, rvar: RVar, xvar: RWVar, mode: WriteMode, any_value: Rc<Any>) {
    if let Some(value) = any_value.downcast_ref::<Option<RWVal<W>>>() {
      self._apply_output(txn, OVal::with_value(rvar, xvar, mode, value.as_ref().map(|v| v._clone())));
    } else {
      unreachable!();
    }
  }

  fn _eval_recursive(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    //println!("DEBUG: F1Op: eval recursive");
    //if Some(txn) != self._txn() {
      for node in self.ctrl.iter() {
        node.eval(txn);
      }
      //println!("DEBUG: F1Op: inner eval recursive");
      self.x_.eval(txn);
    //}
  }
}

impl<F, V1, W> AOp<W> for F1Op<F, V1, W> where V1: 'static, W: 'static {
  fn _value2(&self, txn: Txn, static_value: Option<RWVal<W>>) -> RWVal<W> {
    if static_value.is_some() {
      return static_value.as_ref().unwrap()._clone();
    } else {
      // FIXME
      unimplemented!();
    }
  }

  fn _value3<'a>(&'a self, txn: Txn, static_value: Option<&'a RWVal<W>>) -> &'a RWVal<W> {
    if static_value.is_some() {
      return static_value.unwrap();
    } else {
      // FIXME
      unimplemented!();
    }
  }

  fn _make_value(&self) -> RWVal<W> {
    //(self.ext.make_val)()
    (self.ext.make_val)(self.cfg.borrow_mut())
  }

  /*fn _value(&self) -> &RWVal<W> {
    &self.y
  }*/

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
        None => {
          self.x_._pop_rev(None, pass, &mut |_, _, _| {});
        }
        Some(ref adjoint) => {
          (adjoint)(pass, this, self.cfg.borrow_mut(), sink);
          self.x_._pop_adjoint(pass, sink);
        }
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

impl<F, V> AOp<V> for F1Op<F, V, V> where V: 'static, V: 'static {
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
  //y:    RWVal<W>,
}

impl<F, V1, V2, W> F2Op<F, V1, V2, W> where V1: 'static, V2: 'static, W: 'static {
  //pub fn new(cfg: F, ext: OpExt<F, W>, x1_: Val<V1>, x2_: Val<V2>, y: RWVal<W>) -> Self {
  pub fn new(cfg: F, ext: OpExt<F, W>, x1_: Val<V1>, x2_: Val<V2>) -> Self {
    log_static_graph(|logging| {
      logging.push_pred(x1_._graph_key());
      logging.push_pred(x2_._graph_key());
    });
    let state = RefCell::new(cfg);
    //let y = (ext.make_val)(state.borrow_mut());
    F2Op{
      base: OpBase::default(),
      ext:  ext,
      //cfg:  RefCell::new(cfg),
      cfg:  state,
      ctrl: vec![],
      x1_:  x1_,
      x2_:  x2_,
      //y:    y,
    }
  }
}

impl<F, V1, V2, W> ANode for F2Op<F, V1, V2, W> where V1: 'static, V2: 'static, W: 'static {
  fn _key(&self) -> String {
    unsafe { type_name::<F>() }.to_owned()
  }

  fn _walk(&self) -> &Walk {
    &self.base.stack
  }

  /*fn _io(&self) -> &IOVal {
    &self.y
  }*/

  fn _analysis_tags(&self) -> &AnalysisTags {
    &self.base
  }

  fn _pred_fwd(&self, pred_buf: &mut Vec<Node>) {
    pred_buf.push(self.x1_._to_node());
    pred_buf.push(self.x2_._to_node());
  }

  fn _pred_rev(&self, pred_buf: &mut Vec<Node>) {
    pred_buf.push(self.x2_._to_node());
    pred_buf.push(self.x1_._to_node());
  }

  fn _push(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.push(pass) {
      //if stop_txn.is_none() || stop_txn != self._txn() {
        self.x1_._node()._push(stop_txn, pass, apply);
        self.x2_._node()._push(stop_txn, pass, apply);
      //}
      apply(self);
    }
  }

  fn _pop(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.pop(pass) {
      apply(self);
      //if stop_txn.is_none() || stop_txn != self._txn() {
        self.x2_._node()._pop(stop_txn, pass, apply);
        self.x1_._node()._pop(stop_txn, pass, apply);
      //}
    }
  }

  fn _push_fwd(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    if self.base.stack.push(pass) {
      //if stop_txn.is_none() || stop_txn != self._txn() {
        self.x1_._push_fwd(stop_txn, pass, apply);
        self.x2_._push_fwd(stop_txn, pass, apply);
      //}
      apply(self, rvar, xvar);
    }
  }

  fn _pop_rev(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    if self.base.stack.pop(pass) {
      apply(self, rvar, xvar);
      //if stop_txn.is_none() || stop_txn != self._txn() {
        self.x2_._pop_rev(stop_txn, pass, apply);
        self.x1_._pop_rev(stop_txn, pass, apply);
      //}
    }
  }

  /*fn _txn(&self) -> Option<Txn> {
    self.y.txn()
  }*/

  /*fn _reset(&self) {
    self._value().reset();
  }

  fn _release(&self) {
    self._value().release();
  }*/

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

  fn _io<'a>(&'a self, txn: Txn, static_any_value: &'a Any) -> &'a IOVal {
    if let Some(static_value) = static_any_value.downcast_ref::<Option<RWVal<W>>>() {
      if static_value.is_some() {
        return static_value.as_ref().unwrap();
      } else {
        // FIXME
        unimplemented!();
      }
    } else {
      unreachable!();
    }
  }

  /*fn _apply(&self, txn: Txn, rvar: RVar, xvar: RWVar, mode: WriteMode) {
    self._apply_output(txn, OVal::new(rvar, xvar, mode, self._value()._clone()));
  }*/

  fn _apply_any(&self, txn: Txn, rvar: RVar, xvar: RWVar, mode: WriteMode, any_value: Rc<Any>) {
    if let Some(value) = any_value.downcast_ref::<Option<RWVal<W>>>() {
      self._apply_output(txn, OVal::with_value(rvar, xvar, mode, value.as_ref().map(|v| v._clone())));
    } else {
      unreachable!();
    }
  }

  fn _eval_recursive(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    //if Some(txn) != self._txn() {
      for node in self.ctrl.iter() {
        node.eval(txn);
      }
      self.x1_.eval(txn);
      self.x2_.eval(txn);
    //}
  }
}

impl<F, V1, V2, W> AOp<W> for F2Op<F, V1, V2, W> where V1: 'static, V2: 'static, W: 'static {
  fn _value2(&self, txn: Txn, static_value: Option<RWVal<W>>) -> RWVal<W> {
    if static_value.is_some() {
      return static_value.as_ref().unwrap()._clone();
    } else {
      // FIXME
      unimplemented!();
    }
  }

  fn _value3<'a>(&'a self, txn: Txn, static_value: Option<&'a RWVal<W>>) -> &'a RWVal<W> {
    if static_value.is_some() {
      return static_value.unwrap();
    } else {
      // FIXME
      unimplemented!();
    }
  }

  fn _make_value(&self) -> RWVal<W> {
    //(self.ext.make_val)()
    (self.ext.make_val)(self.cfg.borrow_mut())
  }

  /*fn _value(&self) -> &RWVal<W> {
    &self.y
  }*/

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
        None => {
          self.x2_._pop_rev(None, pass, &mut |_, _, _| {});
          self.x1_._pop_rev(None, pass, &mut |_, _, _| {});
        }
        Some(ref adjoint) => {
          (adjoint)(pass, this, self.cfg.borrow_mut(), sink);
          self.x2_._pop_adjoint(pass, sink);
          self.x1_._pop_adjoint(pass, sink);
        }
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
  //y:    RWVal<W>,
}

impl<F, V1, V2, V3, W> F3Op<F, V1, V2, V3, W> where V1: 'static, V2: 'static, V3: 'static, W: 'static {
  //pub fn new(cfg: F, ext: OpExt<F, W>, x1_: Val<V1>, x2_: Val<V2>, x3_: Val<V3>, y: RWVal<W>) -> Self {
  pub fn new(cfg: F, ext: OpExt<F, W>, x1_: Val<V1>, x2_: Val<V2>, x3_: Val<V3>) -> Self {
    log_static_graph(|logging| {
      logging.push_pred(x1_._graph_key());
      logging.push_pred(x2_._graph_key());
      logging.push_pred(x3_._graph_key());
    });
    let state = RefCell::new(cfg);
    //let y = (ext.make_val)(state.borrow_mut());
    F3Op{
      base: OpBase::default(),
      ext:  ext,
      //cfg:  RefCell::new(cfg),
      cfg:  state,
      ctrl: vec![],
      x1_:  x1_,
      x2_:  x2_,
      x3_:  x3_,
      //y:    y,
    }
  }
}

impl<F, V1, V2, V3, W> ANode for F3Op<F, V1, V2, V3, W> where V1: 'static, V2: 'static, V3: 'static, W: 'static {
  fn _key(&self) -> String {
    unsafe { type_name::<F>() }.to_owned()
  }

  fn _walk(&self) -> &Walk {
    &self.base.stack
  }

  /*fn _io(&self) -> &IOVal {
    &self.y
  }*/

  fn _analysis_tags(&self) -> &AnalysisTags {
    &self.base
  }

  fn _pred_fwd(&self, pred_buf: &mut Vec<Node>) {
    pred_buf.push(self.x1_._to_node());
    pred_buf.push(self.x2_._to_node());
    pred_buf.push(self.x3_._to_node());
  }

  fn _pred_rev(&self, pred_buf: &mut Vec<Node>) {
    pred_buf.push(self.x3_._to_node());
    pred_buf.push(self.x2_._to_node());
    pred_buf.push(self.x1_._to_node());
  }

  fn _push(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.push(pass) {
      //if stop_txn.is_none() || stop_txn != self._txn() {
        self.x1_._node()._push(stop_txn, pass, apply);
        self.x2_._node()._push(stop_txn, pass, apply);
        self.x3_._node()._push(stop_txn, pass, apply);
      //}
      apply(self);
    }
  }

  fn _pop(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.pop(pass) {
      apply(self);
      //if stop_txn.is_none() || stop_txn != self._txn() {
        self.x3_._node()._pop(stop_txn, pass, apply);
        self.x2_._node()._pop(stop_txn, pass, apply);
        self.x1_._node()._pop(stop_txn, pass, apply);
      //}
    }
  }

  fn _push_fwd(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    if self.base.stack.push(pass) {
      //if stop_txn.is_none() || stop_txn != self._txn() {
        self.x1_._push_fwd(stop_txn, pass, apply);
        self.x2_._push_fwd(stop_txn, pass, apply);
        self.x3_._push_fwd(stop_txn, pass, apply);
      //}
      apply(self, rvar, xvar);
    }
  }

  fn _pop_rev(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    if self.base.stack.pop(pass) {
      apply(self, rvar, xvar);
      //if stop_txn.is_none() || stop_txn != self._txn() {
        self.x3_._pop_rev(stop_txn, pass, apply);
        self.x2_._pop_rev(stop_txn, pass, apply);
        self.x1_._pop_rev(stop_txn, pass, apply);
      //}
    }
  }

  /*fn _txn(&self) -> Option<Txn> {
    self.y.txn()
  }*/

  /*fn _reset(&self) {
    self._value().reset();
  }

  fn _release(&self) {
    self._value().release();
  }*/

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

  fn _io<'a>(&'a self, txn: Txn, static_any_value: &'a Any) -> &'a IOVal {
    if let Some(static_value) = static_any_value.downcast_ref::<Option<RWVal<W>>>() {
      if static_value.is_some() {
        return static_value.as_ref().unwrap();
      } else {
        // FIXME
        unimplemented!();
      }
    } else {
      unreachable!();
    }
  }

  /*fn _apply(&self, txn: Txn, rvar: RVar, xvar: RWVar, mode: WriteMode) {
    self._apply_output(txn, OVal::new(rvar, xvar, mode, self._value()._clone()));
  }*/

  fn _apply_any(&self, txn: Txn, rvar: RVar, xvar: RWVar, mode: WriteMode, any_value: Rc<Any>) {
    if let Some(value) = any_value.downcast_ref::<Option<RWVal<W>>>() {
      self._apply_output(txn, OVal::with_value(rvar, xvar, mode, value.as_ref().map(|v| v._clone())));
    } else {
      unreachable!();
    }
  }

  fn _eval_recursive(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    //if Some(txn) != self._txn() {
      for node in self.ctrl.iter() {
        node.eval(txn);
      }
      self.x1_.eval(txn);
      self.x2_.eval(txn);
      self.x3_.eval(txn);
    //}
  }
}

impl<F, V1, V2, V3, W> AOp<W> for F3Op<F, V1, V2, V3, W> where V1: 'static, V2: 'static, V3: 'static, W: 'static {
  fn _value2(&self, txn: Txn, static_value: Option<RWVal<W>>) -> RWVal<W> {
    if static_value.is_some() {
      return static_value.as_ref().unwrap()._clone();
    } else {
      // FIXME
      unimplemented!();
    }
  }

  fn _value3<'a>(&'a self, txn: Txn, static_value: Option<&'a RWVal<W>>) -> &'a RWVal<W> {
    if static_value.is_some() {
      return static_value.unwrap();
    } else {
      // FIXME
      unimplemented!();
    }
  }

  fn _make_value(&self) -> RWVal<W> {
    //(self.ext.make_val)()
    (self.ext.make_val)(self.cfg.borrow_mut())
  }

  /*fn _value(&self) -> &RWVal<W> {
    &self.y
  }*/

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
        None => {
          self.x3_._pop_rev(None, pass, &mut |_, _, _| {});
          self.x2_._pop_rev(None, pass, &mut |_, _, _| {});
          self.x1_._pop_rev(None, pass, &mut |_, _, _| {});
        }
        Some(ref adjoint) => {
          (adjoint)(pass, this, self.cfg.borrow_mut(), sink);
          self.x3_._pop_adjoint(pass, sink);
          self.x2_._pop_adjoint(pass, sink);
          self.x1_._pop_adjoint(pass, sink);
        }
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

  fn _apply_output(&self, txn: Txn, val: OVal<W>) {
    (self.ext.apply)(txn, self.cfg.borrow_mut(), val);
  }
}

pub struct F4Op<F, V1, V2, V3, V4, W> {
  base: OpBase,
  ext:  OpExt<F, W>,
  cfg:  RefCell<F>,
  ctrl: Vec<Node>,
  x1_:  Val<V1>,
  x2_:  Val<V2>,
  x3_:  Val<V3>,
  x4_:  Val<V4>,
}

impl<F, V1, V2, V3, V4, W> F4Op<F, V1, V2, V3, V4, W> where V1: 'static, V2: 'static, V3: 'static, V4: 'static, W: 'static {
  pub fn new(cfg: F, ext: OpExt<F, W>, x1_: Val<V1>, x2_: Val<V2>, x3_: Val<V3>, x4_: Val<V4>) -> Self {
    log_static_graph(|logging| {
      logging.push_pred(x1_._graph_key());
      logging.push_pred(x2_._graph_key());
      logging.push_pred(x3_._graph_key());
      logging.push_pred(x4_._graph_key());
    });
    let cfg = RefCell::new(cfg);
    F4Op{
      base: OpBase::default(),
      ext:  ext,
      cfg:  cfg,
      ctrl: vec![],
      x1_:  x1_,
      x2_:  x2_,
      x3_:  x3_,
      x4_:  x4_,
    }
  }
}

impl<F, V1, V2, V3, V4, W> ANode for F4Op<F, V1, V2, V3, V4, W> where V1: 'static, V2: 'static, V3: 'static, V4: 'static, W: 'static {
  fn _key(&self) -> String {
    unsafe { type_name::<F>() }.to_owned()
  }

  fn _walk(&self) -> &Walk {
    &self.base.stack
  }

  fn _analysis_tags(&self) -> &AnalysisTags {
    &self.base
  }

  fn _pred_fwd(&self, pred_buf: &mut Vec<Node>) {
    pred_buf.push(self.x1_._to_node());
    pred_buf.push(self.x2_._to_node());
    pred_buf.push(self.x3_._to_node());
    pred_buf.push(self.x4_._to_node());
  }

  fn _pred_rev(&self, pred_buf: &mut Vec<Node>) {
    pred_buf.push(self.x4_._to_node());
    pred_buf.push(self.x3_._to_node());
    pred_buf.push(self.x2_._to_node());
    pred_buf.push(self.x1_._to_node());
  }

  fn _push_fwd(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    if self.base.stack.push(pass) {
      self.x1_._push_fwd(stop_txn, pass, apply);
      self.x2_._push_fwd(stop_txn, pass, apply);
      self.x3_._push_fwd(stop_txn, pass, apply);
      self.x4_._push_fwd(stop_txn, pass, apply);
      apply(self, rvar, xvar);
    }
  }

  fn _pop_rev(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    if self.base.stack.pop(pass) {
      apply(self, rvar, xvar);
      self.x4_._pop_rev(stop_txn, pass, apply);
      self.x3_._pop_rev(stop_txn, pass, apply);
      self.x2_._pop_rev(stop_txn, pass, apply);
      self.x1_._pop_rev(stop_txn, pass, apply);
    }
  }

  fn _io<'a>(&'a self, txn: Txn, static_any_value: &'a Any) -> &'a IOVal {
    if let Some(static_value) = static_any_value.downcast_ref::<Option<RWVal<W>>>() {
      if static_value.is_some() {
        return static_value.as_ref().unwrap();
      } else {
        // FIXME
        unimplemented!();
      }
    } else {
      unreachable!();
    }
  }

  fn _eval_recursive(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    for node in self.ctrl.iter() {
      node.eval(txn);
    }
    self.x1_.eval(txn);
    self.x2_.eval(txn);
    self.x3_.eval(txn);
    self.x4_.eval(txn);
  }

  fn _apply_any(&self, txn: Txn, rvar: RVar, xvar: RWVar, mode: WriteMode, any_value: Rc<Any>) {
    if let Some(value) = any_value.downcast_ref::<Option<RWVal<W>>>() {
      self._apply_output(txn, OVal::with_value(rvar, xvar, mode, value.as_ref().map(|v| v._clone())));
    } else {
      unreachable!();
    }
  }
}

impl<F, V1, V2, V3, V4, W> AOp<W> for F4Op<F, V1, V2, V3, V4, W> where V1: 'static, V2: 'static, V3: 'static, V4: 'static, W: 'static {
  fn _value2(&self, txn: Txn, static_value: Option<RWVal<W>>) -> RWVal<W> {
    if static_value.is_some() {
      return static_value.as_ref().unwrap()._clone();
    } else {
      // FIXME
      unimplemented!();
    }
  }

  fn _value3<'a>(&'a self, txn: Txn, static_value: Option<&'a RWVal<W>>) -> &'a RWVal<W> {
    if static_value.is_some() {
      return static_value.unwrap();
    } else {
      // FIXME
      unimplemented!();
    }
  }

  fn _make_value(&self) -> RWVal<W> {
    (self.ext.make_val)(self.cfg.borrow_mut())
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
        None => {
          self.x4_._pop_rev(None, pass, &mut |_, _, _| {});
          self.x3_._pop_rev(None, pass, &mut |_, _, _| {});
          self.x2_._pop_rev(None, pass, &mut |_, _, _| {});
          self.x1_._pop_rev(None, pass, &mut |_, _, _| {});
        }
        Some(ref adjoint) => {
          (adjoint)(pass, this, self.cfg.borrow_mut(), sink);
          self.x4_._pop_adjoint(pass, sink);
          self.x3_._pop_adjoint(pass, sink);
          self.x2_._pop_adjoint(pass, sink);
          self.x1_._pop_adjoint(pass, sink);
        }
      }
    }
  }

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

impl<F, V> FSwitchOp<F, V> where V: 'static {
  pub fn new(cfg: F, ext: OpExt<F, V>, flag: TCell<bool>, x1_: Val<V>, x2_: Val<V>) -> Self {
    log_static_graph(|logging| {
      logging.push_pred(x1_._graph_key());
      logging.push_pred(x2_._graph_key());
    });
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

impl<F, V> ANode for FSwitchOp<F, V> where V: 'static, V: 'static {
  fn _key(&self) -> String {
    unsafe { type_name::<F>() }.to_owned()
  }

  fn _walk(&self) -> &Walk {
    &self.base.stack
  }

  /*fn _io(&self) -> &IOVal {
    // NOTE: `always_get` is valid here.
    match self.flag.always_get() {
      false => self.x1_._op()._io(),
      true  => self.x2_._op()._io(),
    }
  }*/

  fn _analysis_tags(&self) -> &AnalysisTags {
    &self.base
  }

  fn _pred_fwd(&self, pred_buf: &mut Vec<Node>) {
    pred_buf.push(self.x1_._to_node());
    pred_buf.push(self.x2_._to_node());
  }

  fn _pred_rev(&self, pred_buf: &mut Vec<Node>) {
    pred_buf.push(self.x2_._to_node());
    pred_buf.push(self.x1_._to_node());
  }

  fn _push(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.push(pass) {
      //if stop_txn.is_none() || stop_txn != self._txn() {
        self.x1_._node()._push(stop_txn, pass, apply);
        self.x2_._node()._push(stop_txn, pass, apply);
      //}
      apply(self);
    }
  }

  fn _pop(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.pop(pass) {
      apply(self);
      //if stop_txn.is_none() || stop_txn != self._txn() {
        self.x2_._node()._pop(stop_txn, pass, apply);
        self.x1_._node()._pop(stop_txn, pass, apply);
      //}
    }
  }

  fn _push_fwd(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    if self.base.stack.push(pass) {
      //if stop_txn.is_none() || stop_txn != self._txn() {
        self.x1_._push_fwd(stop_txn, pass, apply);
        self.x2_._push_fwd(stop_txn, pass, apply);
      //}
      apply(self, rvar, xvar);
    }
  }

  fn _pop_rev(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    if self.base.stack.pop(pass) {
      apply(self, rvar, xvar);
      //if stop_txn.is_none() || stop_txn != self._txn() {
        self.x2_._pop_rev(stop_txn, pass, apply);
        self.x1_._pop_rev(stop_txn, pass, apply);
      //}
    }
  }

  /*fn _txn(&self) -> Option<Txn> {
    self.done.txn()
  }*/

  /*fn _reset(&self) {
    self.done.reset();
  }

  fn _release(&self) {
    self.done.reset();
  }*/

  fn _io<'a>(&'a self, txn: Txn, static_any_value: &'a Any) -> &'a IOVal {
    if let Some(static_value) = static_any_value.downcast_ref::<Option<RWVal<V>>>() {
      if static_value.is_some() {
        return static_value.as_ref().unwrap();
      } else {
        match self.flag.get(txn) {
          false => self.x1_._io(txn),
          true  => self.x2_._io(txn),
        }
      }
    } else {
      unreachable!();
    }
  }

  /*fn _apply(&self, txn: Txn, rvar: RVar, xvar: RWVar, mode: WriteMode) {
    match self.flag.get(txn) {
      false => self._apply_output(txn, OVal::new(rvar, xvar, mode, self.x1_._op()._value()._clone())),
      true  => self._apply_output(txn, OVal::new(rvar, xvar, mode, self.x2_._op()._value()._clone())),
    }
    self.done.propose(txn, |_| ());
  }*/

  fn _apply_any(&self, txn: Txn, rvar: RVar, xvar: RWVar, mode: WriteMode, any_value: Rc<Any>) {
    if let Some(value) = any_value.downcast_ref::<Option<RWVal<V>>>() {
      self._apply_output(txn, OVal::with_value(rvar, xvar, mode, value.as_ref().map(|v| v._clone())));
    } else {
      unreachable!();
    }
  }

  fn _eval_recursive(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    //if Some(txn) != self._txn() {
      for node in self.ctrl.iter() {
        node.eval(txn);
      }
      match self.flag.get(txn) {
        false   => self.x1_.eval(txn),
        true    => self.x2_.eval(txn),
      }
    //}
  }
}

impl<F, V> AOp<V> for FSwitchOp<F, V> where V: 'static, V: 'static {
  fn _value2(&self, txn: Txn, static_value: Option<RWVal<V>>) -> RWVal<V> {
    if static_value.is_some() {
      return static_value.as_ref().unwrap()._clone();
    } else {
      match self.flag.get(txn) {
        false => self.x1_._value2(txn),
        true  => self.x2_._value2(txn),
      }
    }
  }

  fn _value3<'a>(&'a self, txn: Txn, static_value: Option<&'a RWVal<V>>) -> &'a RWVal<V> {
    if static_value.is_some() {
      return static_value.unwrap();
    } else {
      match self.flag.get(txn) {
        false => self.x1_._value3(txn),
        true  => self.x2_._value3(txn),
      }
    }
  }

  fn _make_value(&self) -> RWVal<V> {
    (self.ext.make_val)(self.cfg.borrow_mut())
  }

  /*fn _value(&self) -> &RWVal<V> {
    // NOTE: `always_get` is valid here.
    match self.flag.always_get() {
      false => self.x1_._op()._value(),
      true  => self.x2_._op()._value(),
    }
  }*/

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
  //y:    RWVal<W>,
}

impl<F, V, W> FJoinOp<F, V, W> where V: 'static, W: 'static {
  //pub fn new(cfg: F, ext: OpExt<F, W>, xs_: Vec<Val<V>>, y: RWVal<W>) -> Self {
  pub fn new(cfg: F, ext: OpExt<F, W>, xs_: Vec<Val<V>>) -> Self {
    log_static_graph(|logging| {
      for x_ in xs_.iter() {
        logging.push_pred(x_._graph_key());
      }
    });
    let state = RefCell::new(cfg);
    //let y = (ext.make_val)(state.borrow_mut());
    FJoinOp{
      base: OpBase::default(),
      ext:  ext,
      //cfg:  RefCell::new(cfg),
      cfg:  state,
      ctrl: vec![],
      xs_:  xs_,
      //y:    y,
    }
  }
}

impl<F, V, W> ANode for FJoinOp<F, V, W> where V: 'static, W: 'static {
  fn _key(&self) -> String {
    unsafe { type_name::<F>() }.to_owned()
  }

  fn _walk(&self) -> &Walk {
    &self.base.stack
  }

  /*fn _io(&self) -> &IOVal {
    &self.y
  }*/

  fn _analysis_tags(&self) -> &AnalysisTags {
    &self.base
  }

  fn _pred_fwd(&self, pred_buf: &mut Vec<Node>) {
    for x_ in self.xs_.iter() {
      pred_buf.push(x_._to_node());
    }
  }

  fn _pred_rev(&self, pred_buf: &mut Vec<Node>) {
    for x_ in self.xs_.iter().rev() {
      pred_buf.push(x_._to_node());
    }
  }

  fn _push(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.push(pass) {
      //if stop_txn.is_none() || stop_txn != self._txn() {
        for x_ in self.xs_.iter() {
          x_._node()._push(stop_txn, pass, apply);
        }
      //}
      apply(self);
    }
  }

  fn _pop(&self, stop_txn: Option<Txn>, pass: Pass, /*filter: &Fn(&ANode) -> bool,*/ apply: &mut FnMut(&ANode)) {
    if self.base.stack.pop(pass) {
      apply(self);
      //if stop_txn.is_none() || stop_txn != self._txn() {
        for x_ in self.xs_.iter().rev() {
          x_._node()._pop(stop_txn, pass, apply);
        }
      //}
    }
  }

  fn _push_fwd(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    if self.base.stack.push(pass) {
      //if stop_txn.is_none() || stop_txn != self._txn() {
        for x_ in self.xs_.iter() {
          x_._push_fwd(stop_txn, pass, apply);
        }
      //}
      apply(self, rvar, xvar);
    }
  }

  fn _pop_rev(&self, stop_txn: Option<Txn>, pass: Pass, rvar: RVar, xvar: RWVar, apply: &mut FnMut(&ANode, RVar, RWVar)) {
    if self.base.stack.pop(pass) {
      apply(self, rvar, xvar);
      //if stop_txn.is_none() || stop_txn != self._txn() {
        for x_ in self.xs_.iter().rev() {
          x_._pop_rev(stop_txn, pass, apply);
        }
      //}
    }
  }

  /*fn _txn(&self) -> Option<Txn> {
    self.y.txn()
  }*/

  /*fn _reset(&self) {
    self._value().reset();
  }

  fn _release(&self) {
    self._value().release();
  }*/

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

  fn _io<'a>(&'a self, txn: Txn, static_any_value: &'a Any) -> &'a IOVal {
    if let Some(static_value) = static_any_value.downcast_ref::<Option<RWVal<W>>>() {
      if static_value.is_some() {
        return static_value.as_ref().unwrap();
      } else {
        // FIXME
        unimplemented!();
      }
    } else {
      unreachable!();
    }
  }

  /*fn _apply(&self, txn: Txn, rvar: RVar, xvar: RWVar, mode: WriteMode) {
    self._apply_output(txn, OVal::new(rvar, xvar, mode, self._value()._clone()));
  }*/

  fn _apply_any(&self, txn: Txn, rvar: RVar, xvar: RWVar, mode: WriteMode, any_value: Rc<Any>) {
    if let Some(value) = any_value.downcast_ref::<Option<RWVal<W>>>() {
      self._apply_output(txn, OVal::with_value(rvar, xvar, mode, value.as_ref().map(|v| v._clone())));
    } else {
      unreachable!();
    }
  }

  fn _eval_recursive(&self, txn: Txn, rvar: RVar, xvar: RWVar) {
    //if Some(txn) != self._txn() {
      for node in self.ctrl.iter() {
        node.eval(txn);
      }
      for x_ in self.xs_.iter() {
        x_.eval(txn);
      }
    //}
  }
}

impl<F, V, W> AOp<W> for FJoinOp<F, V, W> where V: 'static, W: 'static {
  fn _value2(&self, txn: Txn, static_value: Option<RWVal<W>>) -> RWVal<W> {
    if static_value.is_some() {
      return static_value.as_ref().unwrap()._clone();
    } else {
      // FIXME
      unimplemented!();
    }
  }

  fn _value3<'a>(&'a self, txn: Txn, static_value: Option<&'a RWVal<W>>) -> &'a RWVal<W> {
    if static_value.is_some() {
      return static_value.unwrap();
    } else {
      // FIXME
      unimplemented!();
    }
  }

  fn _make_value(&self) -> RWVal<W> {
    //(self.ext.make_val)()
    (self.ext.make_val)(self.cfg.borrow_mut())
  }

  /*fn _value(&self) -> &RWVal<W> {
    &self.y
  }*/

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
        None => {
          for x_ in self.xs_.iter().rev() {
            x_._pop_rev(None, pass, &mut |_, _, _| {});
          }
        }
        Some(ref adjoint) => {
          (adjoint)(pass, this, self.cfg.borrow_mut(), sink);
          for x_ in self.xs_.iter().rev() {
            x_._pop_adjoint(pass, sink);
          }
        }
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

impl<F, V> AOp<V> for FJoinOp<F, V, V> where V: 'static {
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
