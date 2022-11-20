// use crate::tensor::{ops::binary_ops::BinaryOps};
use std::rc::Rc;

// pub trait ForwardPass {
//     fn forward(&self, input: &Rc<Tensor>) -> Rc<Tensor>;
// }

// pub struct Linear {
//     w: Rc<Tensor>,
//     b: Option<Rc<Tensor>>,
// }

// impl Linear {
//     pub fn new(dim: (usize, usize)) -> Self {
//         Self {
//             w: Tensor::randn(dim, Some(true)),
//             b: Some(Tensor::randn((dim.0, 1), Some(true))),
//         }
//     }
// }

// impl ForwardPass for Linear {
//     fn forward(&self, input: &Rc<Tensor>) -> Rc<Tensor> {
//         let ret = self.w.matmul(input);

//         match self.b.as_ref() {
//            Some(b) => ret.add(b),
//            None => ret
//         }
//     }
// }

// pub struct Conv2d {
//     kernels: Vec<Rc<Tensor>>,
//     b: Option<Rc<Tensor>>,
//     in_channels: usize,
// }
