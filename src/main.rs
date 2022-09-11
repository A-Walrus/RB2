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

const ARR: [Complex<i8>; 4] = [c!(1), c!(-1), c!(i), c!(-i)];

fn dot(a: &Matrix, b: &Matrix) -> Matrix {
    let mul = a.dot(b);
    debug_assert!(mul
        .iter()
        .find(|v| v.re % 2 != 0 || v.im % 2 != 0)
        .is_none()); // There should be no odd numbers at this point
    mul / 2
}

fn push(map: &mut HashMap<Matrix, usize>, key: &Matrix) -> bool {
    let mut variants = variants(key);
    let contains_variant = variants.any(|k| map.contains_key(&k));
    if !contains_variant {
        let id = map.len();
        map.insert(key.clone(), id);
        true
    } else {
        false
    }
}

fn variants(key: &Matrix) -> impl Iterator<Item = Matrix> {
    let key = key.clone();
    let variants = ARR.iter().map(move |s| key.map(|c| c * s));
    variants
}

fn main() {
    let map = generate_map();
    let lut = generate_lut(&map);
    eprintln!("Done!");
}

fn generate_lut(map: &HashMap<Matrix, usize>) -> Array2<u16> {
    let len = map.len();
    let mut keys: Vec<_> = map.keys().collect();
    keys.sort_by_key(|m| map.get(m).unwrap());

    let lut = Array2::from_shape_fn((len, len), |(i, j)| {
        let m = dot(keys[i], keys[j]);
        let id = variants(&m).map(|m| map.get(&m)).find_map(|m| m).unwrap();
        *id as u16
    });

    lut
}

fn generate_map() -> HashMap<Matrix, usize> {
    let originals = vec![
        sz1(),
        sx1(),
        sz2(),
        sx2(),
        cnot(),
        // f
    ];

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
    map
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
