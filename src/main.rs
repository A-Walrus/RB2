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

fn dot(a: &Matrix, b: &Matrix) -> Matrix {
    let mul = a.dot(b);
    debug_assert!(mul
        .iter()
        .find(|v| v.re % 2 != 0 || v.im % 2 != 0)
        .is_none()); // There should be no odd numbers at this point
    mul / 2
}

fn push(map: &mut HashMap<Matrix, usize>, key: Matrix) -> bool {
    if !map.contains_key(&key) {
        let id = map.len();
        map.insert(key, id);
        true
    } else {
        false
    }
}

fn main() {
    let mut map = HashMap::new();
    push(&mut map, sz());
    push(&mut map, sx());

    loop {
        let mut added = false;
        let keys: Vec<Matrix> = map.keys().cloned().collect();
        for m1 in &keys {
            for m2 in &keys {
                let m = dot(m1, m2);
                added = added || push(&mut map, m);
            }
        }
        if !added {
            break;
        }
    }

    dbg!(map.len());
    dbg!(map.len() / 4);
}

fn id() -> Matrix {
    Array2::from_diag_elem(2, 1.into()) * 2
}

fn sy() -> Matrix {
    arr2(&[
        [Complex::new(1, 1), Complex::new(-1, -1)],
        [Complex::new(1, 1), Complex::new(1, 1)],
    ])
}

fn sz() -> Matrix {
    arr2(&[
        [Complex::new(1, 0), Complex::new(0, 0)],
        [Complex::new(0, 0), Complex::new(0, 1)],
    ]) * 2
}

fn sx() -> Matrix {
    arr2(&[
        [Complex::new(1, 1), Complex::new(1, -1)],
        [Complex::new(1, -1), Complex::new(1, 1)],
    ])
}

fn x() -> Matrix {
    arr2(&[
        [Complex::new(0, 0), Complex::new(1, 0)],
        [Complex::new(1, 0), Complex::new(0, 0)],
    ]) * 2
}

fn z() -> Matrix {
    arr2(&[
        [Complex::new(1, 0), Complex::new(0, 0)],
        [Complex::new(0, 0), Complex::new(-1, 0)],
    ]) * 2
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn id_check() {
        let id = id();
        let sz = sz();
        let sx = sx();
        let x = x();
        assert_eq!(dot(&id, &id), id);
        assert_eq!(dot(&x, &x), id);
        assert_eq!(dot(&sx, &sx), x);
        assert_eq!(dot(&dot(&dot(&sx, &sx), &sx), &sx), id);
        assert_eq!(dot(&dot(&dot(&sz, &sz), &sz), &sz), id);
    }
}
