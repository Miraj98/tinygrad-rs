pub mod loss {
    use crate::tensor::{
        ops::{binary_ops::BinaryOps, reduce_ops::ReduceOps, unary_ops::UnaryOps},
        Tensor,
    };
    use std::rc::Rc;

    #[allow(non_snake_case)]
    pub fn CrossEntropy(input: &Rc<Tensor>, target: &Rc<Tensor>) -> Rc<Tensor> {
        assert_eq!(input.dim(), target.dim());
        let input_ln = input.ln();
        let lhs = target.mul(&input_ln);

        let ones = Tensor::ones(input.dim(), Some(true));
        let _a = ones.sub(&input);
        let _y = ones.sub(&target);
        let rhs = _y.mul(&_a.ln());

        let l = lhs.add(&rhs).mul_scalar(-1.);
        let loss = l.mean();
        loss
    }

    #[allow(non_snake_case)]
    pub fn QuadraticLoss(input: &Rc<Tensor>, target: &Rc<Tensor>) -> Rc<Tensor> {
        assert_eq!(input.dim(), target.dim());
        let l = target.sub(&input).square().mul_scalar(0.5);
        let loss = l.mean();
        loss
    }
}

pub mod layer {
    use ndarray::arr2;

    use crate::tensor::Tensor;
    use std::rc::Rc;

    // #[allow(non_snake_case)]
    // pub fn Conv2d(
    //     in_channels: usize,
    //     feature_maps: usize,
    //     kernel_dim: (usize, usize),
    // ) -> impl Fn(&Vec<Rc<Tensor>>) -> (Vec<Vec<Rc<Tensor>>>, Vec<Vec<Rc<Tensor>>>, Vec<Rc<Tensor>>)  {
    //     move |x| {
    //         let mut w: Vec<Vec<Rc<Tensor>>> = Vec::new();
    //         let mut b: Vec<Vec<Rc<Tensor>>> = Vec::new();
    //         for _ in 0..feature_maps {
    //             let mut _w: Vec<Rc<Tensor>> = Vec::new();
    //             let mut _b: Vec<Rc<Tensor>> = Vec::new();
    //             for _ in 0..in_channels {
    //                 _w.push(Tensor::randn(kernel_dim, Some(true)));
    //                 _b.push(Tensor::randn((1, 1), Some(true)));

    //             }
    //             w.push(_w);
    //             b.push(_b);
    //         }

    //         let ret = x.conv2d(&w, &b);
    //         (w, b, ret)
    //     }
    // }
}
