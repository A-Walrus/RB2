use std::collections::HashMap;

use ndarray::prelude::*;
use num_complex::{Complex, Complex32};

type Matrix = Array2<Complex<i8>>;
// I
//  1,  0
//  0,  1
// X
//  0, 1
//  1, 0
// Y
//  0,-i
//  i, 0
// Z
//  1,  0
//  0, -1
// z
//  1, 0
//  0, i

// z
//  1, 0
//  0, i
// x
//  1+i, 1-i
//  1-i, 1+i  1/2

macro_rules! c {
    (0) => {
        Complex::new(0, 0)
    };
    (i) => {
        Complex::new(0, 1)
    };
    (-i) => {
        Complex::new(0, -1)
    };
    ($real:literal + i) => {
        Complex::new($real, 1)
    };
    ($real:literal - i) => {
        Complex::new($real, -1)
    };
    ($imag:literal i) => {
        Complex::new(0, $imag)
    };
    ($real:literal + $imag:literal i) => {
        Complex::new($real, $imag)
    };
    ($real:literal - $imag:literal i) => {
        Complex::new($real, -$imag)
    };
    ($real:literal) => {
        Complex::new($real, 0)
    };
}

fn dot(a: &Matrix, b: &Matrix) -> Matrix {
    let mul = a.dot(b);
    debug_assert!(mul
        .iter()
        .find(|v| v.re % 2 != 0 || v.im % 2 != 0)
        .is_none()); // There should be no odd numbers at this point
    mul / 2
}

fn push(map: &mut HashMap<Matrix, usize>, key: &Matrix) -> bool {
    // TODO canonicalization
    if !map.contains_key(key) {
        let id = map.len();
        map.insert(key.clone(), id);
        true
    } else {
        false
    }
}

fn main() {
    let originals = vec![sz1(), sx1(), sz2(), sx2(), cnot()];

    let mut map = HashMap::new();
    for m in &originals {
        push(&mut map, m);
    }

    let mut new: Vec<_> = map.keys().cloned().collect();
    let mut newer: Vec<_> = Vec::new();
    loop {
        dbg!(map.len(), new.len(), newer.len());
        dbg!();
        for m1 in &new {
            for m2 in &originals {
                let m = dot(m1, m2);
                if push(&mut map, &m) {
                    newer.push(m);
                }
            }
        }
        if newer.is_empty() {
            break;
        }
        std::mem::swap(&mut new, &mut newer);
        newer.clear();
    }

    dbg!(map.len());
    dbg!(map.len() / 4);
}

fn id() -> Matrix {
    Array2::from_diag_elem(4, 1.into()) * 2
}

fn sz1() -> Matrix {
    arr2(&[
        [c!(1), c!(0), c!(0), c!(0)],
        [c!(0), c!(i), c!(0), c!(0)],
        [c!(0), c!(0), c!(1), c!(0)],
        [c!(0), c!(0), c!(0), c!(i)],
    ]) * 2
}

fn sx1() -> Matrix {
    arr2(&[
        [c!(1 + i), c!(1 - i), c!(0), c!(0)],
        [c!(1 - i), c!(1 + i), c!(0), c!(0)],
        [c!(0), c!(0), c!(1 + i), c!(1 - i)],
        [c!(0), c!(0), c!(1 - i), c!(1 + i)],
    ])
}

fn sx2() -> Matrix {
    arr2(&[
        [c!(1 + i), c!(0), c!(1 - i), c!(0)],
        [c!(0), c!(1 + i), c!(0), c!(1 - i)],
        [c!(1 - i), c!(0), c!(1 + i), c!(0)],
        [c!(0), c!(1 - i), c!(0), c!(1 + i)],
    ])
}

fn sz2() -> Matrix {
    arr2(&[
        [c!(1), c!(0), c!(0), c!(0)],
        [c!(0), c!(1), c!(0), c!(0)],
        [c!(0), c!(0), c!(i), c!(0)],
        [c!(0), c!(0), c!(0), c!(i)],
    ]) * 2
}

fn cnot() -> Matrix {
    arr2(&[
        [c!(1), c!(0), c!(0), c!(0)],
        [c!(0), c!(1), c!(0), c!(0)],
        [c!(0), c!(0), c!(0), c!(1)],
        [c!(0), c!(0), c!(1), c!(0)],
    ]) * 2
}
