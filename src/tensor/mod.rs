pub mod ops;
pub mod tensor_ref;

// use self::ops::binary_ops::{BinaryOpType, BinaryOps, Matmul, Mul, Sub};
// use self::ops::processing_ops::{Conv2d, ProcessingOpType, ProcessingOps};
// use self::ops::reduce_ops::{Mean, ReduceOpType, ReduceOps, Sum};
// use self::ops::unary_ops::{NaturalLog, ReLU, Sigmoid, Square, UnaryOpType, UnaryOps};
use self::tensor_ref::RefMut;
use ndarray::{Array, Dimension, ShapeBuilder};
use ndarray_rand::rand_distr::num_traits::{One, Zero};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use std::cell::RefCell;
// use ops::binary_ops::Add;
// use ops::OpFunction;
use std::collections::HashSet;
use std::rc::Rc;
use std::{
    cell::{Cell, UnsafeCell},
    ptr::NonNull,
};
use tensor_ref::{BorrowRef, BorrowRefMut, Ref};

pub trait TensorBaseImpl {
    type A: Clone;
    type D: Dimension;

    fn new(a: Array<Self::A, Self::D>, requires_grad: Option<bool>) -> Rc<Self>;
    fn from(a: &Array<Self::A, Self::D>, requires_grad: Option<bool>) -> Rc<Self>;
    fn update_grad(&self, grad: Option<Array<Self::A, Self::D>>);
    fn try_grad(&self) -> Result<Ref<Option<Array<Self::A, Self::D>>>, isize>;
    fn grad(&self) -> Ref<Option<Array<Self::A, Self::D>>>;
    fn try_ndarray(&self) -> Result<Ref<Array<Self::A, Self::D>>, isize>;
    fn ndarray(&self) -> Ref<Array<Self::A, Self::D>>;
    fn try_ndarray_mut(&self) -> Result<RefMut<Array<Self::A, Self::D>>, isize>;
    fn ndarray_mut(&self) -> RefMut<Array<Self::A, Self::D>>;
    fn backward(&self)
    where
        Self::A: Clone + One;
    fn _backward(&self, incoming_grad: &Array<Self::A, Self::D>);
    fn zero_grad(&self);
}

pub struct TensorBase<A, D>
where
    A: Clone,
    D: Dimension,
{
    data: UnsafeCell<Array<A, D>>,
    grad_value: UnsafeCell<Option<Array<A, D>>>,
    grad_borrow: Cell<isize>,
    data_borrow: Cell<isize>,
    backward: RefCell<Option<Box<dyn Fn(&Array<A, D>)>>>,
    pub requires_grad: Cell<Option<bool>>,
}

impl<A, D> TensorBase<A, D>
where
    A: Clone,
    D: Dimension,
{
    fn construct(
        a: Array<A, D>,
        requires_grad: Option<bool>,
    ) -> Rc<TensorBase<A, D>> {
        Rc::new(TensorBase {
            data: UnsafeCell::new(a),
            grad_value: UnsafeCell::new(None),
            grad_borrow: Cell::new(0),
            data_borrow: Cell::new(0),
            backward: RefCell::new(None),
            requires_grad: Cell::new(requires_grad),
        })
    }

    fn construct_with_backward_fn(
        a: Array<A, D>,
        backward_fn: impl Fn(&Array<A, D>) + 'static,
        requires_grad: Option<bool>,
    ) -> Rc<TensorBase<A, D>> {
        Rc::new(TensorBase {
            data: UnsafeCell::new(a),
            grad_value: UnsafeCell::new(None),
            grad_borrow: Cell::new(0),
            data_borrow: Cell::new(0),
            backward: RefCell::new(Some(Box::new(backward_fn))),
            requires_grad: Cell::new(requires_grad),
        })
    }

    fn _backward(&self, incoming: &Array<A, D>) {
    }
}

impl<A, D> TensorBase<A, D>
where
    A: Clone,
    D: Dimension,
{
    pub fn zeros<Sh>(shape: Sh, requires_grad: Option<bool>) -> Rc<Self>
    where
        Sh: ShapeBuilder<Dim = D>,
        A: Clone + Zero,
    {
        let a: Array<A, D> = Array::zeros(shape);
        TensorBase::construct(a, requires_grad)
    }

    pub fn ones<Sh>(shape: Sh, requires_grad: Option<bool>) -> Rc<Self>
    where
        Sh: ShapeBuilder<Dim = D>,
        A: Clone + One,
    {
        let a: Array<A, D> = Array::ones(shape);
        TensorBase::construct(a, requires_grad)
    }

    fn dim(&self) -> D::Pattern {
        unsafe { (*self.data.get()).dim() }
    }

    fn deepwalk(&self) -> Vec<*const Self> {
        todo!()
    }
}

impl<D> TensorBase<f32, D>
where
    D: Dimension,
{
    pub fn randn<Sh>(shape: Sh, requires_grad: Option<bool>) -> Rc<Self>
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        let a: Array<f32, D> = Array::random(shape, StandardNormal);
        TensorBase::construct(a, requires_grad)
    }
}

impl<D> TensorBase<f64, D>
where
    D: Dimension,
{
    pub fn randn<Sh>(shape: Sh, requires_grad: Option<bool>) -> Rc<Self>
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        let a: Array<f64, D> = Array::random(shape, StandardNormal);
        TensorBase::construct(a, requires_grad)
    }
}

impl<A, D> TensorBaseImpl for TensorBase<A, D>
where
    A: Clone,
    D: Dimension,
{
    type A = A;
    type D = D;

    fn new(a: Array<A, D>, requires_grad: Option<bool>) -> Rc<Self> {
        TensorBase::construct(a, requires_grad)
    }

    fn from(a: &Array<A, D>, requires_grad: Option<bool>) -> Rc<Self> {
        TensorBase::construct(a.to_owned(), requires_grad)
    }

    fn try_grad(&self) -> Result<Ref<Option<Array<Self::A, Self::D>>>, isize> {
        match BorrowRef::new(&self.grad_borrow) {
            Some(b) => {
                let value = unsafe { NonNull::new_unchecked(self.grad_value.get()) };
                Ok(Ref { value, borrow: b })
            }
            None => Err(self.grad_borrow.get()),
        }
    }

    fn grad(&self) -> Ref<Option<Array<Self::A, Self::D>>> {
        self.try_grad().expect("already mutably borrowed")
    }

    fn try_ndarray(&self) -> Result<Ref<Array<Self::A, Self::D>>, isize> {
        match BorrowRef::new(&self.data_borrow) {
            Some(b) => {
                let value = unsafe { NonNull::new_unchecked(self.data.get()) };
                Ok(Ref { value, borrow: b })
            }
            None => Err(self.data_borrow.get()),
        }
    }

    fn try_ndarray_mut(&self) -> Result<RefMut<Array<Self::A, Self::D>>, isize> {
        match BorrowRefMut::new(&self.data_borrow) {
            Some(b) => {
                let value = unsafe { NonNull::new_unchecked(self.data.get()) };
                Ok(RefMut {
                    value,
                    borrow: b,
                    marker: std::marker::PhantomData,
                })
            }
            None => Err(self.data_borrow.get()),
        }
    }

    fn ndarray(&self) -> Ref<Array<Self::A, Self::D>> {
        self.try_ndarray().expect("already mutably borrowed")
    }

    fn ndarray_mut(&self) -> RefMut<Array<Self::A, Self::D>> {
        self.try_ndarray_mut().expect("already mutably borrowed")
    }

    fn update_grad(&self, grad: Option<Array<Self::A, Self::D>>) {
        unsafe { *self.grad_value.get() = grad };
    }

    fn _backward(&self, incoming: &Array<A, D>) {
        let b_fn = self.backward.borrow();
        b_fn.as_ref().unwrap()(incoming);
    }

    fn backward(&self)
    where
        A: Clone + One,
    {
        let tensors = self.deepwalk();
        let self_grad: Array<A, D> = Array::ones(self.dim());

        for t in &tensors {
            unsafe {
                let pg = (**t).grad();
                if let Some(p_grad) = pg.as_ref() {
                    (**t)._backward(p_grad);
                } else {
                    (**t)._backward(&self_grad);
                }
            }
        }

        self.update_grad(Some(self_grad));
    }

    fn zero_grad(&self) {
        let tensors = self.deepwalk();
        for t in &tensors {
            unsafe {
                (**t).update_grad(None);
            }
        }
    }
}

#[cfg(test)]
mod binary_ops_tests {
    use crate::tensor::{TensorBase, TensorBaseImpl};

    use super::{ops::binary_ops::BinaryOps};
    use ndarray::{array, Array2};

    #[test]
    fn add_tensors() {
        let a = TensorBase::new(array![[1., 2.], [3., 4.]], None);
        let b = TensorBase::new(array![[2., 3.], [4., 5.]], None);
        let out = a.add(&b);
        assert_eq!(&out.ndarray() as &Array2<f64>, array![[3., 5.], [7., 9.]]);
    }

    #[test]
    fn add_tensors_grad_test() {
        let a = TensorBase::new(array![[1., 2.], [3., 4.]], Some(true));
        let b = TensorBase::new(array![[2., 3.], [4., 5.]], Some(true));
        let out = a.add(&b);

        let out_grad_array = array![[1., 1.], [1., 1.]];
        out._backward(&out_grad_array);

        assert_eq!(a.grad().as_ref().unwrap(), array![[1., 1.], [1., 1.]]);
        assert_eq!(b.grad().as_ref().unwrap(), array![[1., 1.], [1., 1.]]);
    }

    // #[test]
    // fn sub_tensors() {
    //     let a = Tensor::new(array![[1., 2.], [3., 4.]], None);
    //     let b = Tensor::new(array![[2., 3.], [4., 5.]], None);
    //     let out = b.sub(&a);
    //     assert_eq!(&out.ndarray() as &Array2<f64>, array![[1., 1.], [1., 1.]]);
    // }

    // #[test]
    // fn sub_tensors_grad_test() {
    //     let a = Tensor::new(array![[1., 2.], [3., 4.]], Some(true));
    //     let b = Tensor::new(array![[2., 3.], [4., 5.]], Some(true));
    //     let out = b.sub(&a);

    //     let out_grad_array = array![[1., 1.], [1., 1.]];
    //     out.__backward(&out_grad_array);

    //     assert_eq!(a.grad().as_ref().unwrap(), array![[-1., -1.], [-1., -1.]]);
    //     assert_eq!(b.grad().as_ref().unwrap(), array![[1., 1.], [1., 1.]]);
    // }

    // #[test]
    // fn mul_tensors() {
    //     let a = Tensor::new(array![[1., 2.], [3., 4.]], None);
    //     let b = Tensor::new(array![[2., 3.], [4., 5.]], None);
    //     let out = a.mul(&b);
    //     assert_eq!(&out.ndarray() as &Array2<f64>, array![[2., 6.], [12., 20.]]);
    // }

    // #[test]
    // fn mul_tensors_grad_test() {
    //     let a = Tensor::new(array![[1., 2.], [3., 4.]], Some(true));
    //     let b = Tensor::new(array![[2., 3.], [4., 5.]], Some(true));
    //     let out = a.mul(&b);

    //     let out_grad_array = array![[1., 1.], [1., 1.]];
    //     out.__backward(&out_grad_array);

    //     assert_eq!(a.grad().as_ref().unwrap(), array![[2., 3.], [4., 5.]]);
    //     assert_eq!(b.grad().as_ref().unwrap(), array![[1., 2.], [3., 4.]]);
    // }

    // #[test]
    // fn matmul_tensors() {
    //     let a = Tensor::new(array![[1., 2.], [3., 4.]], None);
    //     let b = Tensor::new(array![[2., 3.], [4., 5.]], None);
    //     let out = a.matmul(&b);
    //     assert_eq!(
    //         &out.ndarray() as &Array2<f64>,
    //         array![[10., 13.], [22., 29.]]
    //     );
    // }

    // #[test]
    // fn matmul_tensors_grad_test() {
    //     let a = Tensor::new(array![[1., 2., 3.], [4., 5., 6.]], Some(true));
    //     let b = Tensor::new(array![[2., 3.], [4., 5.], [6., 7.]], Some(true));
    //     let out = a.matmul(&b);

    //     let out_grad_array = array![[1., 1.], [1., 1.]];
    //     out.__backward(&out_grad_array);

    //     assert_eq!(
    //         a.grad().as_ref().unwrap(),
    //         array![[5., 9., 13.], [5., 9., 13.]]
    //     );
    //     assert_eq!(
    //         b.grad().as_ref().unwrap(),
    //         array![[5., 5.], [7., 7.], [9., 9.]]
    //     );
    // }
}

// #[cfg(test)]
// mod process_ops_tests {
//     use crate::tensor::{ops::processing_ops::ProcessingOps, Tensor};
//     use ndarray::{array, Array2};

//     #[test]
//     fn conv2d_tensors() {
//         let a = Tensor::new(array![[1., 2., 3.], [3., 4., 5.], [5., 6., 7.]], None);
//         let b = Tensor::new(array![[0.1409, 0.2612], [0.2657, -0.1486]], None);
//         let out = a.conv2d(&b, (1, 1));
//         assert_eq!(
//             &out.ndarray() as &Array2<f64>,
//             array![
//                 [0.8659999999999999, 1.3851999999999995],
//                 [1.9043999999999999, 2.4236]
//             ]
//         );
//     }

//     #[test]
//     fn conv2d_grad_test() {
//         let a = Tensor::new(array![[1., 2., 3.], [3., 4., 5.], [5., 6., 7.]], Some(true));
//         let b = Tensor::new(array![[0.4078, 0.1711], [-0.3865, 0.3107]], Some(true));
//         let out = a.conv2d(&b, (1, 1));

//         let out_grad_array = array![[1., 1.], [1., 1.]];
//         out.__backward(&out_grad_array);

//         assert_eq!(
//             a.grad().as_ref().unwrap(),
//             array![
//                 [0.4078, 0.5789, 0.1711],
//                 [0.021299999999999986, 0.5031, 0.4818],
//                 [-0.3865, -0.07580000000000003, 0.3107]
//             ]
//         );
//         assert_eq!(b.grad().as_ref().unwrap(), array![[10., 14.], [18., 22.]]);
//     }
// }

// #[cfg(test)]
// mod unary_ops_tests {
//     use crate::tensor::{ops::unary_ops::UnaryOps, Tensor};
//     use ndarray::{array, Array2};

//     #[test]
//     fn sigmoid_test() {
//         let a = Tensor::new(array![[1., 2.], [3., 4.]], None);
//         let out = a.sigmoid();
//         assert_eq!(
//             &out.ndarray() as &Array2<f64>,
//             array![
//                 [0.7310585786300049, 0.8807970779778823],
//                 [0.9525741268224334, 0.9820137900379085]
//             ]
//         );
//     }

//     #[test]
//     fn sigmoid_grad_test() {
//         let a = Tensor::new(array![[1., 2.], [3., 4.]], Some(true));
//         let out = a.sigmoid();

//         let out_grad_array = array![[1., 1.], [1., 1.]];
//         out.__backward(&out_grad_array);

//         assert_eq!(
//             a.grad().as_ref().unwrap(),
//             array![
//                 [0.19661193324148188, 0.1049935854035065],
//                 [0.045176659730912144, 0.017662706213291114]
//             ]
//         );
//     }

//     #[test]
//     fn square_test() {
//         let a = Tensor::new(array![[1., 2.], [3., 4.]], None);
//         let out = a.square();
//         assert_eq!(&out.ndarray() as &Array2<f64>, array![[1., 4.], [9., 16.]]);
//     }

//     #[test]
//     fn square_grad_test() {
//         let a = Tensor::new(array![[1., 2.], [3., 4.]], Some(true));
//         let out = a.square();

//         let out_grad_array = array![[1., 1.], [1., 1.]];
//         out.__backward(&out_grad_array);

//         assert_eq!(a.grad().as_ref().unwrap(), array![[2., 4.], [6., 8.]]);
//     }

//     #[test]
//     fn relu_test() {
//         let a = Tensor::new(array![[-1., 0.], [3., 4.]], None);
//         let out = a.relu();
//         assert_eq!(&out.ndarray() as &Array2<f64>, array![[0., 0.], [3., 4.]]);
//     }

//     #[test]
//     fn relu_grad_test() {
//         let a = Tensor::new(array![[-1., -8.], [3., 4.]], Some(true));
//         let out = a.relu();

//         let out_grad_array = array![[1., 1.], [1., 1.]];
//         out.__backward(&out_grad_array);

//         assert_eq!(a.grad().as_ref().unwrap(), array![[0., 0.], [1., 1.]]);
//     }

//     #[test]
//     fn natural_log_test() {
//         let a = Tensor::new(array![[1., 10.], [3., 4.]], None);
//         let out = a.ln();
//         assert_eq!(
//             &out.ndarray() as &Array2<f64>,
//             array![
//                 [0., 2.302585092994046],
//                 [1.0986122886681098, 1.3862943611198906]
//             ]
//         );
//     }

//     #[test]
//     fn natural_log_grad_test() {
//         let a = Tensor::new(array![[1., 10.], [3., 4.]], Some(true));
//         let out = a.ln();

//         let out_grad_array = array![[1., 1.], [1., 1.]];
//         out.__backward(&out_grad_array);

//         assert_eq!(
//             a.grad().as_ref().unwrap(),
//             array![[1., 0.1], [0.3333333333333333, 0.25]]
//         );
//     }
// }

// #[cfg(test)]
// mod reduce_ops_tests {
//     use crate::tensor::{ops::reduce_ops::ReduceOps, Tensor};
//     use ndarray::{array, Array2};

//     #[test]
//     fn mean_test() {
//         let a = Tensor::new(array![[1., 2.], [3., 4.]], None);
//         let out = a.mean();
//         assert_eq!(&out.ndarray() as &Array2<f64>, array![[2.5]]);
//     }

//     #[test]
//     fn mean_grad_test() {
//         let a = Tensor::new(array![[1., 2.], [3., 4.]], Some(true));
//         let out = a.mean();

//         let out_grad_array = array![[1.]];
//         out.__backward(&out_grad_array);

//         assert_eq!(
//             a.grad().as_ref().unwrap(),
//             array![[0.25, 0.25], [0.25, 0.25]]
//         );
//     }

//     #[test]
//     fn sum_test() {
//         let a = Tensor::new(array![[1., 2.], [3., 4.]], None);
//         let out = a.sum();
//         assert_eq!(&out.ndarray() as &Array2<f64>, array![[10.]]);
//     }

//     #[test]
//     fn sum_grad_test() {
//         let a = Tensor::new(array![[1., 2.], [3., 4.]], Some(true));
//         let out = a.sum();

//         let out_grad_array = array![[1.]];
//         out.__backward(&out_grad_array);

//         assert_eq!(a.grad().as_ref().unwrap(), array![[1., 1.], [1., 1.]]);
//     }
// }

// #[cfg(test)]
// mod autodiff_tests {
//     use ndarray::array;

//     use crate::tensor::ops::{reduce_ops::ReduceOps, unary_ops::UnaryOps};

//     use super::{ops::binary_ops::BinaryOps, Tensor};

//     #[test]
//     fn topo_order() {
//         let a = Tensor::new(array![[1., 2.], [3., 4.]], Some(true));
//         let b = Tensor::new(array![[5., 6.], [7., 8.]], Some(true));
//         let c = a.add(&b);
//         let d = c.mul(&a);

//         let order = d.__deepwalk();

//         assert_eq!(order.len(), 4);
//         assert_eq!(order[0], d.as_ref() as *const Tensor);
//         assert_eq!(order[1], c.as_ref() as *const Tensor);
//         assert_eq!(order[2], b.as_ref() as *const Tensor);
//         assert_eq!(order[3], a.as_ref() as *const Tensor);
//     }

//     #[test]
//     fn chain_rule_test() {
//         let a = Tensor::new(array![[1., 2.], [3., 4.]], Some(true));
//         let b = Tensor::new(array![[5., 6.], [7., 8.]], Some(true));
//         let c = a.add(&b);
//         let d = c.mul(&a);
//         let e = d.square().sum();

//         e.backward();

//         assert_eq!(
//             a.grad().as_ref().unwrap(),
//             array![[84., 320.], [780., 1536.]]
//         );
//         assert_eq!(b.grad().as_ref().unwrap(), array![[12., 64.], [180., 384.]]);
//     }
// }
