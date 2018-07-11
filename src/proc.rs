pub trait Proc: Clone {
  fn rank(&self);
  fn num_ranks(&self);
}

pub trait ProcIO {
  type Tx;
  type Rx;

  fn message(&self, src: usize, dst: usize) -> (Self::Tx, Self::Rx);
  fn allreduce_sum(&self) -> (Self::Tx, Self::Rx);
  fn broadcast(&self, root: usize) -> (Self::Tx, Self::Rx);
}

pub trait ProcTxOnce<Buf: ?Sized> {
  fn send(self, buf: &Buf);
}

pub trait ProcRxOnce<Buf: ?Sized> {
  fn recv(self, buf: &mut Buf);
}
