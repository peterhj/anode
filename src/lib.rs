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
#![feature(unboxed_closures)]

extern crate arithmetic;
//extern crate async_execution;
#[cfg(feature = "gpu")] extern crate cuda;
#[cfg(feature = "gpu")] extern crate cuda_blas;
#[cfg(feature = "gpu")] extern crate cuda_dnn;
#[cfg(feature = "gpu")] extern crate devicemem_gpu;
//extern crate fnv;
extern crate rng;

//#[macro_use] extern crate lazy_static;
//extern crate libc;
extern crate rand;

use ops::{MemIoReader, MemIoWriter, SumJoinOp, SumJoinOpExt};

use std::any::{Any};
use std::cell::{Cell, RefCell, Ref, RefMut};
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::{Entry};
use std::rc::{Rc};

pub mod context;
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Epoch(u64);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TmpVar(u64);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
pub struct RootVar(TmpVar);

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

impl Default for TmpVar {
  fn default() -> Self {
    TmpVar(gen_thread_local_uid())
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

pub struct NodeStackEntry {
  epoch:        Epoch,
  push_degree:  usize,
  pop_degree:   usize,
}

#[derive(Default)]
pub struct NodeStack {
  entries:  RefCell<Vec<NodeStackEntry>>,
}

impl NodeStack {
  pub fn push(&self, epoch: Epoch) -> bool {
    let mut entries = self.entries.borrow_mut();
    if entries.is_empty() || entries.last().unwrap().epoch != epoch {
      entries.push(NodeStackEntry{
        epoch:          epoch,
        push_degree:    1,
        pop_degree:     0,
      });
      true
    } else {
      entries.last_mut().unwrap().push_degree += 1;
      false
    }
  }

  pub fn outdegree(&self) -> usize {
    let entries = self.entries.borrow();
    assert!(!entries.is_empty());
    entries.last().unwrap().push_degree
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
}

pub trait ANode {
  fn _push(&self, epoch: Epoch, priority: &NodeRefMap<usize>, filter: &Fn(&ANode) -> bool, apply: &mut FnMut(&ANode));
  fn _pop(&self, epoch: Epoch, priority: &NodeRefMap<usize>, filter: &Fn(&ANode) -> bool, apply: &mut FnMut(&ANode));

  fn _io(&self) -> &AIo;
  fn _eval(&self, txn: Txn);

  /*fn load_forward(&self, txn: Txn, priority: &NodeRefMap<usize>, writer: &mut Any) {
    let epoch = Epoch::default();
    self._push(epoch, priority, &|_| true, &mut |node| node._io()._load(txn, writer));
    self._pop(epoch, priority, &|_| true, &mut |_node| {});
  }

  fn load_reverse(&self, txn: Txn, priority: &NodeRefMap<usize>, writer: &mut Any) {
    let epoch = Epoch::default();
    self._push(epoch, priority, &|_| true, &mut |_node| {});
    self._pop(epoch, priority, &|_| true, &mut |node| node._io()._load(txn, writer));
  }

  fn store_forward(&self, txn: Txn, priority: &NodeRefMap<usize>, reader: &mut Any) {
    let epoch = Epoch::default();
    self._push(epoch, priority, &|_| true, &mut |node| node._io()._store(txn, reader));
    self._pop(epoch, priority, &|_| true, &mut |_node| {});
  }

  fn store_reverse(&self, txn: Txn, priority: &NodeRefMap<usize>, reader: &mut Any) {
    let epoch = Epoch::default();
    self._push(epoch, priority, &|_| true, &mut |_node| {});
    self._pop(epoch, priority, &|_| true, &mut |node| node._io()._store(txn, reader));
  }*/

  fn v_deserialize_forward(&self, txn: Txn, priority: &NodeRefMap<usize>, writer: &mut FnMut(WriteMode, &mut Any)) {
    // TODO
    unimplemented!();
  }

  fn v_deserialize_reverse(&self, txn: Txn, priority: &NodeRefMap<usize>, writer: &mut FnMut(WriteMode, &mut Any)) {
    // TODO
    unimplemented!();
  }

  fn v_serialize_forward(&self, txn: Txn, priority: &NodeRefMap<usize>, reader: &mut FnMut(&mut Any)) {
    // TODO
    unimplemented!();
  }

  fn v_serialize_reverse(&self, txn: Txn, priority: &NodeRefMap<usize>, reader: &mut FnMut(&mut Any)) {
    // TODO
    unimplemented!();
  }

  fn eval(&self, txn: Txn) {
    // TODO
    self._eval(txn);
  }

  fn eval_prioritized(&self, txn: Txn, priority: &NodeRefMap<usize>) {
    let epoch = Epoch::default();
    // TODO
    self._push(epoch, priority, &|_| true, &mut |node| node._eval(txn));
    self._pop(epoch, priority, &|_| true, &mut |_node| {});
  }
}

pub fn swap_val<V, Op>(this: &mut Rc<Op>, new_val: V) where V: AVal, Op: AOp<V=V> {
  match Rc::get_mut(this) {
    None => panic!(),
    Some(this) => {
      this._swap(new_val);
    }
  }
}

pub trait AOp: ANode {
  type V: AVal;

  fn _swap(&mut self, new_val: Self::V) { unimplemented!(); }

  fn var(&self) -> RootVar { unimplemented!(); }
  fn _make_value(&self) -> Self::V;
  fn value(&self) -> Self::V;
  fn _make_tangent(&self) -> (Rc<ANode>, Rc<AOp<V=Self::V>>) { unimplemented!(); }
  fn tangent(&self) -> (Rc<ANode>, Rc<AOp<V=Self::V>>);
  fn _pop_adjoint(&self, epoch: Epoch, priority: &NodeRefMap<usize>, filter: &Fn(&ANode) -> bool, this: Rc<AOp<V=Self::V>>, sink: &mut Sink) { unimplemented!(); }
  fn adjoint(&self, sink: &mut Sink);
}

pub trait AIo {
  fn _persist(&self, txn: Txn);
  //fn _load(&self, txn: Txn, writer: &mut Any);
  //fn _store(&self, txn: Txn, reader: &mut Any);
  fn _deserialize(&self, txn: Txn, writer: &mut FnMut(WriteMode, &mut Any));
  fn _serialize(&self, txn: Txn, reader: &mut FnMut(&Any));
}

pub trait AVal: AIo + Clone {
  type T;

  fn duplicate(&self) -> Self where Self: Sized;
  fn root_var(&self) -> RootVar;
  fn var(&self) -> TmpVar;
  //fn load(&self, txn: Txn, writer: &mut Any);
  //fn store(&self, txn: Txn, reader: &mut Any);
}

/*pub trait ASink {
  fn get_adj<V>(&mut self, var: RootVar) -> Option<Rc<AOp<V=V>>> where Self: Sized, V: AVal + 'static;
  fn put_adj<V>(&mut self, var: RootVar, adj_node: Rc<ANode>, adj_op: Rc<AOp<V=V>>) where Self: Sized, V: AVal + 'static;
}*/

pub struct Sink {
  freeze:   HashSet<RootVar>,
  adj_map:  HashMap<RootVar, Vec<(Rc<ANode>, Rc<Any>)>>,
  join_map: HashMap<RootVar, (Rc<ANode>, Rc<Any>)>,
  adj_seq:  Vec<Rc<ANode>>,
}

impl Sink {
  pub fn get_adj<V>(&mut self, var: RootVar) -> Option<Rc<AOp<V=V>>> where V: RWVal + 'static, SumJoinOp: SumJoinOpExt<V::T, V> {
    self.freeze.insert(var);
    if self.adj_map.contains_key(&var) {
      let adjs = self.adj_map.get(&var).unwrap();
      match adjs.len() {
        0 => {}
        1 => {
          match (adjs[0].1).downcast_ref::<Rc<AOp<V=V>>>() {
            None => panic!(),
            Some(adj_op) => return Some(adj_op.clone()),
          }
        }
        _ => {
          if self.join_map.contains_key(&var) {
            let &(_, ref join) = self.join_map.get(&var).unwrap();
            match join.downcast_ref::<Rc<AOp<V=V>>>() {
              None => panic!(),
              Some(join_op) => return Some(join_op.clone()),
            }
          } else {
            let adj_ops: Vec<_> = adjs.iter().map(|&(_, ref a)| {
              match a.downcast_ref::<Rc<AOp<V=V>>>() {
                None => panic!(),
                Some(adj_op) => adj_op.clone(),
              }
            }).collect();
            let join = SumJoinOp::build(adj_ops);
            let join_node: Rc<ANode> = join.clone();
            let join_any_op: Rc<AOp<V=V>> = join;
            self.join_map.insert(var, (join_node, Rc::new(join_any_op.clone())));
            return Some(join_any_op);
          }
        }
      }
    }
    None
  }

  pub fn put_adj<V, Op>(&mut self, var: RootVar, mut adj_op: Rc<Op>) where V: AVal + 'static, Op: AOp<V=V> + 'static {
    assert!(!self.freeze.contains(&var));
    if self.adj_map.contains_key(&var) {
      let mut adjs = self.adj_map.get_mut(&var).unwrap();
      // NOTE: Not using accumulate-mode join, so do not explicitly swap values.
      /*if !adjs.is_empty() {
        match (adjs[0].1).downcast_ref::<Rc<AOp<V=V>>>() {
          None => panic!(),
          Some(ex_adj_op) => {
            swap_val(&mut adj_op, ex_adj_op.value());
          }
        }
      }*/
      let adj_node: Rc<ANode> = adj_op.clone();
      let adj_any_op: Rc<AOp<V=V>> = adj_op;
      adjs.push((adj_node.clone(), Rc::new(adj_any_op)));
      self.adj_seq.push(adj_node);
    } else {
      let adj_node: Rc<ANode> = adj_op.clone();
      let adj_any_op: Rc<AOp<V=V>> = adj_op;
      self.adj_map.insert(var, vec![(adj_node.clone(), Rc::new(adj_any_op))]);
      self.adj_seq.push(adj_node);
    }
  }
}

pub struct OpBase<V> where V: AVal {
  ref_:     NodeRef,
  stack:    NodeStack,
  tng_op:   RefCell<Option<(Rc<ANode>, Rc<AOp<V=V>>)>>,
}

impl<V> Default for OpBase<V> where V: AVal {
  fn default() -> Self {
    OpBase{
      ref_:     NodeRef::default(),
      stack:    NodeStack::default(),
      tng_op:   RefCell::new(None),
    }
  }
}

#[derive(Default)]
pub struct NodeCollection {
  nodes:    Vec<Rc<ANode>>,
}

impl ANode for NodeCollection {
  fn _push(&self, epoch: Epoch, priority: &NodeRefMap<usize>, filter: &Fn(&ANode) -> bool, apply: &mut FnMut(&ANode)) {
    // FIXME: priority.
    for node in self.nodes.iter() {
      node._push(epoch, priority, filter, apply);
    }
  }

  fn _pop(&self, epoch: Epoch, priority: &NodeRefMap<usize>, filter: &Fn(&ANode) -> bool, apply: &mut FnMut(&ANode)) {
    // FIXME: priority.
    for node in self.nodes.iter().rev() {
      node._pop(epoch, priority, filter, apply);
    }
  }

  fn _io(&self) -> &AIo {
    unimplemented!();
  }

  fn _eval(&self, _txn: Txn) {
    // TODO
  }
}

#[derive(Default)]
pub struct VarCollection {
  vars:     Vec<TmpVar>,
}

#[derive(Default)]
pub struct NodeRefMap<T> {
  node_map: HashMap<NodeRef, T>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WriteMode {
  Exclusive,
  Accumulate,
  Clobber,
}

#[derive(Clone, Copy)]
pub struct WriteToken<'a> {
  var:      TmpVar,
  first:    bool,
  borrow:   &'a (),
}

impl<'a> WriteToken<'a> {
  pub fn first_write(&self) -> bool {
    self.first
  }
}

pub trait RWVal: AVal {
  fn from(alloc: Rc<Fn(Txn) -> Self::T>) -> Self where Self: Sized;

  /*// Set `WriteMode`.
  fn set_exclusive(&self);
  fn set_accumulate(&self);
  fn set_clobber(&self);*/

  fn reset(&self);
  fn dealloc(&self);
  fn force_persist(&self, txn: Txn);
  fn write(&self, txn: Txn) -> Option<(WriteMode, WriteToken)>;
  fn get(&self, txn: Txn) -> Ref<Self::T>;
  fn get_mut(&self, txn: Txn, token: WriteToken) -> RefMut<Self::T>;
}

pub struct RWValBuf<T> {
  mode:         WriteMode,
  curr_txn:     Option<Txn>,
  dirty:        bool,
  l_consumers:  HashSet<TmpVar>,
  d_consumers:  HashSet<TmpVar>,
  l_producers:  HashSet<TmpVar>,
  d_producers:  HashSet<TmpVar>,
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

pub struct Val<T> {
  ref_:     ValRef,
  alloc:    Rc<Fn(Txn) -> T>,
  root:     RootVar,
  var:      TmpVar,
  buf:      Rc<RefCell<RWValBuf<T>>>,
  borrow:   (),
}

impl<T> Val<T> {
  pub fn new(alloc: Rc<Fn(Txn) -> T>) -> Self {
    let ref_ = ValRef::default();
    let var = TmpVar::default();
    Val{
      ref_:     ref_,
      alloc:    alloc,
      root:     RootVar(var),
      var:      var,
      buf:      Rc::new(RefCell::new(RWValBuf::default())),
      borrow:   (),
    }
  }

  pub fn mode(self, mode: WriteMode) -> Self {
    // TODO
    unimplemented!();
  }
}

impl<T> Clone for Val<T> {
  fn clone(&self) -> Self {
    Val{
      ref_:     self.ref_,
      alloc:    self.alloc.clone(),
      root:     self.root,
      var:      TmpVar::default(),
      buf:      self.buf.clone(),
      borrow:   (),
    }
  }
}

impl<T> AIo for Val<T> where T: 'static {
  fn _persist(&self, txn: Txn) {
    self.force_persist(txn);
  }

  /*fn _load(&self, txn: Txn, writer: &mut Any) {
    // TODO
    unimplemented!();
  }

  fn _store(&self, txn: Txn, reader: &mut Any) {
    // TODO
    unimplemented!();
  }*/

  fn _deserialize(&self, txn: Txn, writer: &mut FnMut(WriteMode, &mut Any)) {
    if let Some((mode, token)) = self.write(txn) {
      let mut buf = self.get_mut(txn, token);
      writer(mode, &mut *buf);
    }
  }

  fn _serialize(&self, txn: Txn, reader: &mut FnMut(&Any)) {
    let buf = self.get(txn);
    reader(&*buf);
  }
}

impl<T> AVal for Val<T> where T: 'static {
  type T = T;

  fn duplicate(&self) -> Self {
    Val{
      ref_:     self.ref_,
      alloc:    self.alloc.clone(),
      root:     self.root,
      var:      self.var,
      buf:      self.buf.clone(),
      borrow:   (),
    }
  }

  fn root_var(&self) -> RootVar {
    self.root
  }

  fn var(&self) -> TmpVar {
    self.var
  }
}

impl<T> RWVal for Val<T> where T: 'static {
  fn from(alloc: Rc<Fn(Txn) -> T>) -> Self {
    let var = TmpVar::default();
    Val{
      ref_:     ValRef::default(),
      //mode:     mode,
      alloc:    alloc,
      root:     RootVar(var),
      var:      var,
      buf:      Rc::new(RefCell::new(RWValBuf::default())),
      borrow:   (),
    }
  }

  /*fn set_exclusive(&self) {
    let mut buf = self.buf.borrow_mut();
    assert!(!buf.dirty);
    buf.mode = WriteMode::Exclusive;
  }

  fn set_accumulate(&self) {
    let mut buf = self.buf.borrow_mut();
    assert!(!buf.dirty);
    buf.mode = WriteMode::Accumulate;
  }

  fn set_clobber(&self) {
    let mut buf = self.buf.borrow_mut();
    assert!(!buf.dirty);
    buf.mode = WriteMode::Clobber;
  }*/

  fn reset(&self) {
    let mut buf = self.buf.borrow_mut();
    buf.curr_txn = None;
    buf.l_consumers.clear();
    buf.d_consumers.clear();
    buf.l_producers.clear();
    buf.d_producers.clear();
  }

  fn dealloc(&self) {
    self.reset();
    let mut buf = self.buf.borrow_mut();
    buf.data = None;
  }

  fn force_persist(&self, txn: Txn) {
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
    assert!(!buf.d_producers.contains(&self.var),
        "`persist` should be called before all other writes");
    match buf.l_producers.len() {
      0 => {}
      1 => {
        assert!(buf.l_producers.contains(&self.var),
            "`persist` should be called before all other writes");
      }
      _ => panic!("`persist` should be called before all other writes"),
    }
    assert!(buf.l_consumers.is_empty(),
        "`persist` should be called before reads");
    buf.l_producers.insert(self.var);
  }

  fn write(&self, txn: Txn) -> Option<(WriteMode, WriteToken)> {
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
            if buf.l_producers.contains(&self.var) {
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
            if buf.l_producers.contains(&self.var) {
              return None;
            }
          }
          (_, _) => panic!("all writes to `Accumulate` val must be live"),
        }
        assert!(buf.l_consumers.is_empty(),
            "attempting write to `Exclusive` val after read");
      }
      WriteMode::Clobber => {
        match (buf.l_producers.len(), buf.d_producers.len()) {
          (0, 0) => {}
          (1, _) => {
            if buf.l_producers.contains(&self.var) {
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
    buf.l_producers.insert(self.var);
    Some((buf.mode, WriteToken{var: self.var, first: first, borrow: &self.borrow}))
  }

  fn get(&self, txn: Txn) -> Ref<T> {
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

      assert!(!buf.d_consumers.contains(&self.var),
          "attempting an invalid read (the value has been clobbered)");
      buf.l_consumers.insert(self.var);

      assert!(buf.data.is_some(),
          "attempting a read on empty data");
    }
    let buf = self.buf.borrow();
    Ref::map(buf, |buf| buf.data.as_ref().unwrap())
  }

  fn get_mut(&self, txn: Txn, token: WriteToken) -> RefMut<T> {
    assert_eq!(self.var, token.var);
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
      assert!(!buf.d_producers.contains(&self.var),
          "attempting an invalid write (the value has been clobbered)");
      assert!(buf.l_producers.contains(&self.var),
          "attempting an unpermitted write (did you use the wrong `WriteToken`?)");

      if buf.data.is_none() {
        buf.data = Some((self.alloc)(txn));
      }
    }
    RefMut::map(buf, |buf| buf.data.as_mut().unwrap())
  }
}

pub struct ClkVal<T> {
  val:      ValRef,
  clock:    Rc<Cell<usize>>,
  alloc:    Rc<Fn(Txn) -> T>,
  clk_vars: Vec<TmpVar>,
  clk_bufs: Rc<RefCell<Vec<RWValBuf<T>>>>,
  borrow:   (),
}

impl<T> Clone for ClkVal<T> {
  fn clone(&self) -> Self {
    let mut new_clk_vars = Vec::with_capacity(self.clk_vars.len());
    for _ in 0 .. self.clk_vars.len() {
      new_clk_vars.push(TmpVar::default());
    }
    ClkVal{
      val:      self.val,
      clock:    self.clock.clone(),
      alloc:    self.alloc.clone(),
      clk_vars: new_clk_vars,
      clk_bufs: self.clk_bufs.clone(),
      borrow:   (),
    }
  }
}

impl<T> AIo for ClkVal<T> where T: 'static {
  fn _persist(&self, txn: Txn) {
    self.force_persist(txn);
  }

  /*fn _load(&self, txn: Txn, writer: &mut Any) {
    // TODO
  }

  fn _store(&self, txn: Txn, reader: &mut Any) {
    // TODO
  }*/

  fn _deserialize(&self, txn: Txn, writer: &mut FnMut(WriteMode, &mut Any)) {
    // TODO
  }

  fn _serialize(&self, txn: Txn, reader: &mut FnMut(&Any)) {
    // TODO
  }
}

impl<T> AVal for ClkVal<T> where T: 'static {
  type T = T;

  fn duplicate(&self) -> Self {
    ClkVal{
      val:      self.val,
      clock:    self.clock.clone(),
      alloc:    self.alloc.clone(),
      clk_vars: self.clk_vars.clone(),
      clk_bufs: self.clk_bufs.clone(),
      borrow:   (),
    }
  }

  fn root_var(&self) -> RootVar {
    // TODO
    unimplemented!();
  }

  fn var(&self) -> TmpVar {
    let clk = self.clock.get();
    self.clk_vars[clk]
  }

  /*fn load(&self, txn: Txn, deserializer: &mut Any) {
    // TODO
  }

  fn store(&self, txn: Txn, serializer: &mut Any) {
    // TODO
  }*/
}

impl<T> RWVal for ClkVal<T> where T: 'static {
  fn from(alloc: Rc<Fn(Txn) -> T>) -> Self {
    // FIXME
    unimplemented!();
  }

  /*fn set_exclusive(&self) {
    // FIXME
    //let mut buf = self.buf.borrow_mut();
    //buf.mode = WriteMode::Exclusive;
    unimplemented!();
  }

  fn set_clobber(&self) {
    // FIXME
    //let mut buf = self.buf.borrow_mut();
    //buf.mode = WriteMode::Clobber;
    unimplemented!();
  }

  fn set_accumulate(&self) {
    // FIXME
    //let mut buf = self.buf.borrow_mut();
    //buf.mode = WriteMode::Accumulate;
    unimplemented!();
  }*/

  fn reset(&self) {
    // FIXME
    unimplemented!();
  }

  fn dealloc(&self) {
    // FIXME
    unimplemented!();
  }

  fn force_persist(&self, txn: Txn) {
    // FIXME
    unimplemented!();
  }

  fn write(&self, txn: Txn) -> Option<(WriteMode, WriteToken)> {
    // FIXME
    //Some((WriteMode::Exclusive, WriteToken{node: node, borrow: &self.borrow}))
    unimplemented!();
  }

  fn get(&self, txn: Txn) -> Ref<T> {
    // TODO
    unimplemented!();
  }

  fn get_mut(&self, txn: Txn, token: WriteToken) -> RefMut<T> {
    // TODO
    unimplemented!();
  }
}

impl<T> ClkVal<T> {
  fn get_prev(&self, txn: Txn) -> Ref<T> {
    // TODO
    unimplemented!();
  }

  fn get_clk(&self, clk: usize, txn: Txn) -> RefMut<T> {
    // TODO
    unimplemented!();
  }
}

/*pub trait KeyValue<K, T>: RWVal<T> {
  fn write_key(&self, key: &K, txn: Txn) -> Option<(WriteMode, WriteToken)>;
  fn get_key(&self, key: &K, txn: Txn) -> Ref<T>;
  fn get_key_mut(&self, key: &K, txn: Txn, token: WriteToken) -> RefMut<T>;
}

pub struct KeyValInner<K, T> {
  key_bufs: HashMap<K, RWValBuf<T>>,
}*/

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
  fn write(&mut self, mode: WriteMode, dst: &'a mut Any);
}

impl<'a, W> IoWriter<'a> for W where W: MemIoWriter<'a> {
  fn write(&mut self, mode: WriteMode, mut dst: &'a mut Any) {
    let ty_id = (*dst).get_type_id();
    if self.write_mem(mode, &mut dst).is_some() {
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

impl<'a, T> FnOnce<(WriteMode, &'a mut Any)> for FlatWriter<'a, T> where FlatWriter<'a, T>: IoWriter<'a> {
  type Output = ();

  extern "rust-call" fn call_once(mut self, args: (WriteMode, &'a mut Any)) -> () {
    self.call_mut(args)
  }
}

impl<'a, T> FnMut<(WriteMode, &'a mut Any)> for FlatWriter<'a, T> where FlatWriter<'a, T>: IoWriter<'a> {
  extern "rust-call" fn call_mut(&mut self, args: (WriteMode, &'a mut Any)) -> () {
    self.write(args.0, args.1);
  }
}

pub struct OpExt<V> {
  make:     Rc<Fn() -> V>,
  func:     Rc<Fn(Txn, V)>,
  tangent:  Option<Rc<Fn() -> (Rc<ANode>, Rc<AOp<V=V>>)>>,
  adjoint:  Option<Rc<Fn(Rc<AOp<V=V>>, &mut Sink)>>,
}

pub struct SrcOp<S, V> where V: AVal {
  base: OpBase<V>,
  ext:  OpExt<V>,
  spec: S,
  val:  V,
}

impl<S, V> SrcOp<S, V>
where V: AVal {
  pub fn new(spec: S, ext: OpExt<V>, val: V) -> Self {
    SrcOp{
      base: OpBase::default(),
      ext:  ext,
      spec: spec,
      val:  val,
    }
  }
}

impl<S, V> ANode for SrcOp<S, V>
where V: AVal {
  fn _push(&self, epoch: Epoch, priority: &NodeRefMap<usize>, filter: &Fn(&ANode) -> bool, apply: &mut FnMut(&ANode)) {
    // TODO
  }

  fn _pop(&self, epoch: Epoch, priority: &NodeRefMap<usize>, filter: &Fn(&ANode) -> bool, apply: &mut FnMut(&ANode)) {
    // TODO
  }

  fn _io(&self) -> &AIo {
    &self.val
  }

  fn _eval(&self, txn: Txn) {
    (self.ext.func)(txn, self.val.duplicate());
  }
}

impl<S, V> AOp for SrcOp<S, V>
where V: AVal {
  type V = V;

  fn _make_value(&self) -> Self::V {
    (self.ext.make)()
  }

  fn value(&self) -> Self::V {
    self.val.clone()
  }

  fn _make_tangent(&self) -> (Rc<ANode>, Rc<AOp<V=Self::V>>) {
    match self.ext.tangent {
      None => panic!(),
      Some(ref tangent) => (tangent)(),
    }
  }

  fn tangent(&self) -> (Rc<ANode>, Rc<AOp<V=Self::V>>) {
    // TODO
    unimplemented!();
  }

  fn _pop_adjoint(&self, epoch: Epoch, priority: &NodeRefMap<usize>, filter: &Fn(&ANode) -> bool, this: Rc<AOp<V=Self::V>>, sink: &mut Sink) {
    if self.base.stack.pop(epoch) {
      match self.ext.adjoint {
        None => panic!(),
        Some(ref adjoint) => (adjoint)(this, sink),
      }
    }
  }

  fn adjoint(&self, sink: &mut Sink) {
    // TODO
    unimplemented!();
  }
}

/*pub struct Pipe1Op<S, V, W> where W: AVal {
  base: OpBase<W>,
  ext:  OpExt<W>,
  spec: S,
  // TODO: should not contain an input `x` but instead a "slot" in which to
  // pipe an input of the same type.
  x_:   Rc<AOp<V=V>>,
  y:    W,
}

impl<S, V, W> Pipe1Op<S, V, W>
where V: AVal, W: AVal {
  pub fn attach(&self, txn: Txn, input: Rc<AOp<V=V>>) {
    // TODO: figure out `attach` semantics.
  }
}*/

pub struct F1Op<S, V1, W> where W: AVal {
  base: OpBase<W>,
  ext:  OpExt<W>,
  spec: S,
  x_:   Rc<AOp<V=V1>>,
  y:    W,
}

impl<S, V1, W> F1Op<S, V1, W>
where V1: AVal, W: AVal {
  pub fn new(spec: S, ext: OpExt<W>, x_: Rc<AOp<V=V1>>, y: W) -> Self {
    F1Op{
      base: OpBase::default(),
      ext:  ext,
      spec: spec,
      x_:   x_,
      y:    y,
    }
  }
}

impl<S, V1, W> ANode for F1Op<S, V1, W>
where V1: AVal, W: AVal {
  fn _push(&self, epoch: Epoch, priority: &NodeRefMap<usize>, filter: &Fn(&ANode) -> bool, apply: &mut FnMut(&ANode)) {
    if self.base.stack.push(epoch) {
      // TODO: apply priority.
      self.x_._push(epoch, priority, filter, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, priority: &NodeRefMap<usize>, filter: &Fn(&ANode) -> bool, apply: &mut FnMut(&ANode)) {
    if self.base.stack.pop(epoch) {
      // TODO: apply priority.
      apply(self);
      self.x_._pop(epoch, priority, filter, apply);
    }
  }

  fn _io(&self) -> &AIo {
    &self.y
  }

  fn _eval(&self, txn: Txn) {
    (self.ext.func)(txn, self.y.duplicate());
  }
}

impl<S, V1, W> AOp for F1Op<S, V1, W>
where V1: AVal, W: AVal {
  type V = W;

  fn _make_value(&self) -> Self::V {
    (self.ext.make)()
  }

  fn value(&self) -> Self::V {
    self.y.clone()
  }

  fn _make_tangent(&self) -> (Rc<ANode>, Rc<AOp<V=Self::V>>) {
    match self.ext.tangent {
      None => panic!(),
      Some(ref tangent) => (tangent)(),
    }
  }

  fn tangent(&self) -> (Rc<ANode>, Rc<AOp<V=Self::V>>) {
    let mut tng_op = self.base.tng_op.borrow_mut();
    if tng_op.is_none() {
      *tng_op = Some(self._make_tangent());
    }
    tng_op.as_ref().unwrap().clone()
  }

  fn _pop_adjoint(&self, epoch: Epoch, priority: &NodeRefMap<usize>, filter: &Fn(&ANode) -> bool, this: Rc<AOp<V=Self::V>>, sink: &mut Sink) {
    if self.base.stack.pop(epoch) {
      // TODO: priority.
      match self.ext.adjoint {
        None => panic!(),
        Some(ref adjoint) => (adjoint)(this, sink),
      }
      self.x_._pop_adjoint(epoch, priority, filter, self.x_.clone(), sink);
    }
  }

  fn adjoint(&self, sink: &mut Sink) {
    // TODO
    unimplemented!();
  }
}

pub struct F2Op<S, V1, V2, W> where W: AVal {
  base: OpBase<W>,
  ext:  OpExt<W>,
  spec: S,
  x1_:  Rc<AOp<V=V1>>,
  x2_:  Rc<AOp<V=V2>>,
  y:    W,
}

impl<S, V1, V2, W> F2Op<S, V1, V2, W>
where V1: AVal, V2: AVal, W: AVal {
  pub fn new(spec: S, ext: OpExt<W>, x1_: Rc<AOp<V=V1>>, x2_: Rc<AOp<V=V2>>, y: W) -> Self {
    F2Op{
      base: OpBase::default(),
      ext:  ext,
      spec: spec,
      x1_:  x1_,
      x2_:  x2_,
      y:    y,
    }
  }
}

impl<S, V1, V2, W> ANode for F2Op<S, V1, V2, W>
where V1: AVal, V2: AVal, W: AVal {
  fn _push(&self, epoch: Epoch, priority: &NodeRefMap<usize>, filter: &Fn(&ANode) -> bool, apply: &mut FnMut(&ANode)) {
    if self.base.stack.push(epoch) {
      // TODO: apply priority.
      self.x1_._push(epoch, priority, filter, apply);
      self.x2_._push(epoch, priority, filter, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, priority: &NodeRefMap<usize>, filter: &Fn(&ANode) -> bool, apply: &mut FnMut(&ANode)) {
    if self.base.stack.pop(epoch) {
      // TODO: apply priority.
      apply(self);
      self.x2_._pop(epoch, priority, filter, apply);
      self.x1_._pop(epoch, priority, filter, apply);
    }
  }

  fn _io(&self) -> &AIo {
    &self.y
  }

  fn _eval(&self, txn: Txn) {
    (self.ext.func)(txn, self.y.duplicate());
  }
}

impl<S, V1, V2, W> AOp for F2Op<S, V1, V2, W>
where V1: AVal, V2: AVal, W: AVal {
  type V = W;

  fn _make_value(&self) -> Self::V {
    (self.ext.make)()
  }

  fn value(&self) -> Self::V {
    self.y.clone()
  }

  fn _make_tangent(&self) -> (Rc<ANode>, Rc<AOp<V=Self::V>>) {
    match self.ext.tangent {
      None => panic!(),
      Some(ref tangent) => (tangent)(),
    }
  }

  fn tangent(&self) -> (Rc<ANode>, Rc<AOp<V=Self::V>>) {
    let mut tng_op = self.base.tng_op.borrow_mut();
    if tng_op.is_none() {
      *tng_op = Some(self._make_tangent());
    }
    tng_op.as_ref().unwrap().clone()
  }

  fn _pop_adjoint(&self, epoch: Epoch, priority: &NodeRefMap<usize>, filter: &Fn(&ANode) -> bool, this: Rc<AOp<V=Self::V>>, sink: &mut Sink) {
    if self.base.stack.pop(epoch) {
      // TODO: priority.
      match self.ext.adjoint {
        None => panic!(),
        Some(ref adjoint) => (adjoint)(this, sink),
      }
      self.x2_._pop_adjoint(epoch, priority, filter, self.x2_.clone(), sink);
      self.x1_._pop_adjoint(epoch, priority, filter, self.x1_.clone(), sink);
    }
  }

  fn adjoint(&self, sink: &mut Sink) {
    // TODO
    unimplemented!();
  }
}

pub struct F3Op<S, V1, V2, V3, W> where W: AVal {
  base: OpBase<W>,
  ext:  OpExt<W>,
  spec: S,
  x1_:  Rc<AOp<V=V1>>,
  x2_:  Rc<AOp<V=V2>>,
  x3_:  Rc<AOp<V=V3>>,
  y:    W,
}

impl<S, V1, V2, V3, W> ANode for F3Op<S, V1, V2, V3, W>
where V1: AVal, V2: AVal, V3: AVal, W: AVal {
  fn _push(&self, epoch: Epoch, priority: &NodeRefMap<usize>, filter: &Fn(&ANode) -> bool, apply: &mut FnMut(&ANode)) {
    if self.base.stack.push(epoch) {
      // TODO: apply priority.
      self.x1_._push(epoch, priority, filter, apply);
      self.x2_._push(epoch, priority, filter, apply);
      self.x3_._push(epoch, priority, filter, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, priority: &NodeRefMap<usize>, filter: &Fn(&ANode) -> bool, apply: &mut FnMut(&ANode)) {
    if self.base.stack.pop(epoch) {
      // TODO: apply priority.
      apply(self);
      self.x3_._pop(epoch, priority, filter, apply);
      self.x2_._pop(epoch, priority, filter, apply);
      self.x1_._pop(epoch, priority, filter, apply);
    }
  }

  fn _io(&self) -> &AIo {
    &self.y
  }

  fn _eval(&self, txn: Txn) {
    (self.ext.func)(txn, self.y.duplicate());
  }
}

impl<S, V1, V2, V3, W> AOp for F3Op<S, V1, V2, V3, W>
where V1: AVal, V2: AVal, V3: AVal, W: AVal {
  type V = W;

  fn _make_value(&self) -> Self::V {
    (self.ext.make)()
  }

  fn value(&self) -> Self::V {
    self.y.clone()
  }

  fn _make_tangent(&self) -> (Rc<ANode>, Rc<AOp<V=Self::V>>) {
    match self.ext.tangent {
      None => panic!(),
      Some(ref tangent) => (tangent)(),
    }
  }

  fn tangent(&self) -> (Rc<ANode>, Rc<AOp<V=Self::V>>) {
    // TODO
    unimplemented!();
  }

  fn _pop_adjoint(&self, epoch: Epoch, priority: &NodeRefMap<usize>, filter: &Fn(&ANode) -> bool, this: Rc<AOp<V=Self::V>>, sink: &mut Sink) {
    if self.base.stack.pop(epoch) {
      // TODO: priority.
      match self.ext.adjoint {
        None => panic!(),
        Some(ref adjoint) => (adjoint)(this, sink),
      }
      self.x3_._pop_adjoint(epoch, priority, filter, self.x3_.clone(), sink);
      self.x2_._pop_adjoint(epoch, priority, filter, self.x2_.clone(), sink);
      self.x1_._pop_adjoint(epoch, priority, filter, self.x1_.clone(), sink);
    }
  }

  fn adjoint(&self, sink: &mut Sink) {
    // TODO
    unimplemented!();
  }
}

pub struct JoinOp<S, V, W> where W: AVal {
  base: OpBase<W>,
  ext:  OpExt<W>,
  spec: S,
  xs_:  Vec<Rc<AOp<V=V>>>,
  y:    W,
}

impl<S, V, W> JoinOp<S, V, W>
where V: AVal, W: AVal {
  pub fn new(spec: S, ext: OpExt<W>, xs_: Vec<Rc<AOp<V=V>>>, y: W) -> Self {
    JoinOp{
      base: OpBase::default(),
      ext:  ext,
      spec: spec,
      xs_:  xs_,
      y:    y,
    }
  }
}

impl<S, V, W> ANode for JoinOp<S, V, W>
where V: AVal, W: AVal {
  fn _push(&self, epoch: Epoch, priority: &NodeRefMap<usize>, filter: &Fn(&ANode) -> bool, apply: &mut FnMut(&ANode)) {
    if self.base.stack.push(epoch) {
      // TODO: apply priority.
      for x_ in self.xs_.iter() {
        x_._push(epoch, priority, filter, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, priority: &NodeRefMap<usize>, filter: &Fn(&ANode) -> bool, apply: &mut FnMut(&ANode)) {
    if self.base.stack.pop(epoch) {
      // TODO: apply priority.
      apply(self);
      for x_ in self.xs_.iter().rev() {
        x_._pop(epoch, priority, filter, apply);
      }
    }
  }

  fn _io(&self) -> &AIo {
    &self.y
  }

  fn _eval(&self, txn: Txn) {
    (self.ext.func)(txn, self.y.duplicate());
  }
}

impl<S, V, W> AOp for JoinOp<S, V, W>
where V: AVal, W: AVal {
  type V = W;

  fn _make_value(&self) -> Self::V {
    (self.ext.make)()
  }

  fn value(&self) -> Self::V {
    self.y.clone()
  }

  fn tangent(&self) -> (Rc<ANode>, Rc<AOp<V=Self::V>>) {
    // TODO
    unimplemented!();
  }

  fn _pop_adjoint(&self, epoch: Epoch, priority: &NodeRefMap<usize>, filter: &Fn(&ANode) -> bool, this: Rc<AOp<V=Self::V>>, sink: &mut Sink) {
    if self.base.stack.pop(epoch) {
      match self.ext.adjoint {
        None => panic!(),
        Some(ref adjoint) => (adjoint)(this, sink),
      }
      for x_ in self.xs_.iter().rev() {
        x_._pop_adjoint(epoch, priority, filter, x_.clone(), sink);
      }
    }
  }

  fn adjoint(&self, sink: &mut Sink) {
    // TODO
    unimplemented!();
  }
}
