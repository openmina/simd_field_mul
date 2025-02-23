pub mod avx;

#[cfg(test)]
mod tests {
    use ark_ff::One;
    use ark_ff::UniformRand;
    use mina_hasher::Fp;
    use rand::rngs::OsRng;

    use crate::avx::mul_assign_fp4;
    use crate::avx::pack4fields;
    use crate::avx::unpack4fields;


    // Single iteration test based on compare_multiplications
    #[test]
    fn test_single_multiplication() {
        let mut rng = OsRng;

        let random_x_fp = [
            Fp::rand(&mut rng),
            Fp::rand(&mut rng),
            Fp::rand(&mut rng),
            Fp::rand(&mut rng),
        ];
        let random_y_fp = [
            Fp::rand(&mut rng),
            Fp::rand(&mut rng),
            Fp::rand(&mut rng),
            Fp::rand(&mut rng),
        ];

        let mut a = pack4fields(
            random_x_fp[0],
            random_x_fp[1],
            random_x_fp[2],
            random_x_fp[3],
        );
        let b = pack4fields(
            random_y_fp[0],
            random_y_fp[1],
            random_y_fp[2],
            random_y_fp[3],
        );

        let mut x_fp: [Fp; 4] = Default::default();
        mul_assign_fp4(&mut a, &b);
        unpack4fields(a, &mut x_fp);

        let result_fp_ark: [Fp; 4] = random_x_fp
            .iter()
            .zip(random_y_fp.iter())
            .map(|(x, y)| *x * y)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        assert_eq!(x_fp, result_fp_ark);
    }

    // Multiple iterations test with a smaller iteration count
    #[test]
    fn test_multiple_multiplications() {
        let mut rng = OsRng;

        for _ in 0..100000 {
            let random_x_fp = [
                Fp::rand(&mut rng),
                Fp::rand(&mut rng),
                Fp::rand(&mut rng),
                Fp::rand(&mut rng),
            ];
            let random_y_fp = [
                Fp::rand(&mut rng),
                Fp::rand(&mut rng),
                Fp::rand(&mut rng),
                Fp::rand(&mut rng),
            ];

            let mut a = pack4fields(
                random_x_fp[0],
                random_x_fp[1],
                random_x_fp[2],
                random_x_fp[3],
            );
            let b = pack4fields(
                random_y_fp[0],
                random_y_fp[1],
                random_y_fp[2],
                random_y_fp[3],
            );

            let mut x_fp: [Fp; 4] = Default::default();
            mul_assign_fp4(&mut a, &b);
            unpack4fields(a, &mut x_fp);

            let result_fp_ark: [Fp; 4] = random_x_fp
                .iter()
                .zip(random_y_fp.iter())
                .map(|(x, y)| *x * y)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();

            assert_eq!(x_fp, result_fp_ark);
        }
    }

    // Identity multiplication test: multiplying by one should yield the same element.
    #[test]
    fn test_identity_multiplication() {
        let one = Fp::one();
        let identity_array = [one; 4];
        let mut rng = OsRng;
        let random_fp = [
            Fp::rand(&mut rng),
            Fp::rand(&mut rng),
            Fp::rand(&mut rng),
            Fp::rand(&mut rng),
        ];

        let mut a = pack4fields(random_fp[0], random_fp[1], random_fp[2], random_fp[3]);
        let b = pack4fields(
            identity_array[0],
            identity_array[1],
            identity_array[2],
            identity_array[3],
        );
        let mut x_fp: [Fp; 4] = Default::default();
        mul_assign_fp4(&mut a, &b);
        unpack4fields(a, &mut x_fp);

        assert_eq!(x_fp, random_fp);
    }
}
