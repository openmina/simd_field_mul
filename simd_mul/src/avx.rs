use mina_hasher::Fp;
use std::arch::x86_64::*;

#[inline]
pub fn pack4fields(a: Fp, b: Fp, c: Fp, d: Fp) -> [__m256i; 8] {
    // TODO: optimize in the same way as unpack4fields by loading a,b,c,d in ymm registers and permuting them
    unsafe {
        let a: [i32; 8] = std::mem::transmute(a.0.0);
        let b: [i32; 8] = std::mem::transmute(b.0.0);
        let c: [i32; 8] = std::mem::transmute(c.0.0);
        let d: [i32; 8] = std::mem::transmute(d.0.0);

        [
            _mm256_set_epi32(0, a[0], 0, b[0], 0, c[0], 0, d[0]),
            _mm256_set_epi32(0, a[1], 0, b[1], 0, c[1], 0, d[1]),
            _mm256_set_epi32(0, a[2], 0, b[2], 0, c[2], 0, d[2]),
            _mm256_set_epi32(0, a[3], 0, b[3], 0, c[3], 0, d[3]),
            _mm256_set_epi32(0, a[4], 0, b[4], 0, c[4], 0, d[4]),
            _mm256_set_epi32(0, a[5], 0, b[5], 0, c[5], 0, d[5]),
            _mm256_set_epi32(0, a[6], 0, b[6], 0, c[6], 0, d[6]),
            _mm256_set_epi32(0, a[7], 0, b[7], 0, c[7], 0, d[7]),
        ]
    }
}

#[inline]
pub fn unpack4fields(a: [__m256i; 8], dst: &mut [Fp; 4]) {
    unsafe {
        let tmp1 = _mm256_blend_epi32(a[0], _mm256_slli_epi64(a[1], 32), 0b10101010);
        let tmp2 = _mm256_blend_epi32(a[2], _mm256_slli_epi64(a[3], 32), 0b10101010);
        let tmp3 = _mm256_blend_epi32(a[4], _mm256_slli_epi64(a[5], 32), 0b10101010);
        let tmp4 = _mm256_blend_epi32(a[6], _mm256_slli_epi64(a[7], 32), 0b10101010);
        let tmp1_lo = _mm256_unpacklo_epi64(tmp1, tmp2);
        let tmp1_hi = _mm256_unpackhi_epi64(tmp1, tmp2);
        let tmp2_lo = _mm256_unpacklo_epi64(tmp3, tmp4);
        let tmp2_hi = _mm256_unpackhi_epi64(tmp3, tmp4);

        _mm256_storeu_si256(
            dst[0].0.as_mut().as_mut_ptr() as *mut __m256i,
            _mm256_permute2x128_si256(tmp1_hi, tmp2_hi, 0x31),
        );
        _mm256_storeu_si256(
            dst[1].0.as_mut().as_mut_ptr() as *mut __m256i,
            _mm256_permute2x128_si256(tmp1_lo, tmp2_lo, 0x31),
        );
        _mm256_storeu_si256(
            dst[2].0.as_mut().as_mut_ptr() as *mut __m256i,
            _mm256_permute2x128_si256(tmp1_hi, tmp2_hi, 0x20),
        );
        _mm256_storeu_si256(
            dst[3].0.as_mut().as_mut_ptr() as *mut __m256i,
            _mm256_permute2x128_si256(tmp1_lo, tmp2_lo, 0x20),
        );
    }
}

#[inline]
fn is_valid_mask4(a: &[__m256i; 8]) -> __m256i {
    unsafe {
        let offset = _mm256_set1_epi64x(0x8000000000000000u64 as i64);
        let mod_offset = [
            _mm256_set1_epi64x(0x8000000000000001u64 as i64),
            _mm256_set1_epi64x(0x80000000992d30edu64 as i64),
            _mm256_set1_epi64x(0x80000000094cf91bu64 as i64),
            _mm256_set1_epi64x(0x80000000224698fcu64 as i64),
            offset,
            offset,
            offset,
            _mm256_set1_epi64x(0x8000000040000000u64 as i64),
        ];
        let a_offset = [
            _mm256_add_epi64(a[0], offset),
            _mm256_add_epi64(a[1], offset),
            _mm256_add_epi64(a[2], offset),
            _mm256_add_epi64(a[3], offset),
            _mm256_add_epi64(a[4], offset),
            _mm256_add_epi64(a[5], offset),
            _mm256_add_epi64(a[6], offset),
            _mm256_add_epi64(a[7], offset),
        ];

        let gt = _mm256_cmpgt_epi64(a_offset[7], mod_offset[7]);
        let gte = _mm256_or_si256(_mm256_cmpeq_epi64(a_offset[7], mod_offset[7]), gt);

        let gt2 = _mm256_cmpgt_epi64(a_offset[6], mod_offset[6]);
        let gte2 = _mm256_or_si256(_mm256_cmpeq_epi64(a_offset[6], mod_offset[6]), gt2);
        let gt2 = _mm256_and_si256(_mm256_or_si256(gt2, gt), gte);
        let gte2 = _mm256_and_si256(_mm256_or_si256(gte2, gt), gte);

        let gte3 = _mm256_cmpeq_epi64(a_offset[5], mod_offset[5]);
        let gt3 = _mm256_and_si256(gt2, gte2);
        let gte3 = _mm256_and_si256(_mm256_or_si256(gte3, gt2), gte2);

        let gte4 = _mm256_cmpeq_epi64(a_offset[4], mod_offset[4]);
        let gt4 = _mm256_and_si256(gt3, gte3);
        let gte4 = _mm256_and_si256(_mm256_or_si256(gte4, gt3), gte3);

        let gte5 = _mm256_cmpeq_epi64(a_offset[3], mod_offset[3]);
        let gt5 = _mm256_and_si256(gt4, gte4);
        let gte5 = _mm256_and_si256(_mm256_or_si256(gte5, gt4), gte4);

        let gt6 = _mm256_cmpgt_epi64(a_offset[2], mod_offset[2]);
        let gte6 = _mm256_or_si256(_mm256_cmpeq_epi64(a_offset[2], mod_offset[2]), gt6);
        let gt6 = _mm256_and_si256(_mm256_or_si256(gt6, gt5), gte5);
        let gte6 = _mm256_and_si256(_mm256_or_si256(gte6, gt5), gte5);

        let gt7 = _mm256_cmpgt_epi64(a_offset[1], mod_offset[1]);
        let gte7 = _mm256_or_si256(_mm256_cmpeq_epi64(a_offset[1], mod_offset[1]), gt7);
        let gt7 = _mm256_and_si256(_mm256_or_si256(gt7, gt6), gte6);
        let gte7 = _mm256_and_si256(_mm256_or_si256(gte7, gt6), gte6);

        let gt8 = _mm256_cmpgt_epi64(a_offset[0], mod_offset[0]);
        let gte8 = _mm256_or_si256(_mm256_cmpeq_epi64(a_offset[0], mod_offset[0]), gt8);
        let gt8 = _mm256_and_si256(_mm256_or_si256(gt8, gt7), gte7);
        let gte8 = _mm256_and_si256(_mm256_or_si256(gte8, gt7), gte7);

        let valid_mask = _mm256_or_si256(gte8, gt8);
        //println!("comparison result: {:?}", valid_mask);
        valid_mask
    }
}

#[inline]
fn sbb_mask4(a: &mut [__m256i; 8], valid_mask: __m256i) {
    unsafe {
        let mod0 = _mm256_and_si256(valid_mask, _mm256_set1_epi64x(0x00000001));
        let mod1 = _mm256_and_si256(valid_mask, _mm256_set1_epi64x(0x992d30ed));
        let mod2 = _mm256_and_si256(valid_mask, _mm256_set1_epi64x(0x94cf91b));
        let mod3 = _mm256_and_si256(valid_mask, _mm256_set1_epi64x(0x224698fc));
        let mod7 = _mm256_and_si256(valid_mask, _mm256_set1_epi64x(0x40000000));
        let borrow_mask = _mm256_set1_epi64x(0x0000000100000000);
        let mask32 = _mm256_set1_epi64x(0x00000000FFFFFFFF);

        a[0] = _mm256_or_si256(a[0], borrow_mask);
        a[0] = _mm256_sub_epi64(a[0], mod0);
        let borrow = _mm256_srli_epi64(_mm256_xor_si256(a[0], borrow_mask), 32);
        a[0] = _mm256_and_si256(a[0], mask32);
        a[1] = _mm256_or_si256(a[1], borrow_mask);
        a[1] = _mm256_sub_epi64(_mm256_sub_epi64(a[1], mod1), borrow);
        let borrow = _mm256_srli_epi64(_mm256_xor_si256(a[1], borrow_mask), 32);
        a[1] = _mm256_and_si256(a[1], mask32);
        a[2] = _mm256_or_si256(a[2], borrow_mask);
        a[2] = _mm256_sub_epi64(_mm256_sub_epi64(a[2], mod2), borrow);
        let borrow = _mm256_srli_epi64(_mm256_xor_si256(a[2], borrow_mask), 32);
        a[2] = _mm256_and_si256(a[2], mask32);
        a[3] = _mm256_or_si256(a[3], borrow_mask);
        a[3] = _mm256_sub_epi64(_mm256_sub_epi64(a[3], mod3), borrow);
        let borrow = _mm256_srli_epi64(_mm256_xor_si256(a[3], borrow_mask), 32);
        a[3] = _mm256_and_si256(a[3], mask32);
        a[4] = _mm256_or_si256(a[4], borrow_mask);
        a[4] = _mm256_sub_epi64(a[4], borrow);
        let borrow = _mm256_srli_epi64(_mm256_xor_si256(a[4], borrow_mask), 32);
        a[4] = _mm256_and_si256(a[4], mask32);
        a[5] = _mm256_or_si256(a[5], borrow_mask);
        a[5] = _mm256_sub_epi64(a[5], borrow);
        let borrow = _mm256_srli_epi64(_mm256_xor_si256(a[5], borrow_mask), 32);
        a[5] = _mm256_and_si256(a[5], mask32);
        a[6] = _mm256_or_si256(a[6], borrow_mask);
        a[6] = _mm256_sub_epi64(a[6], borrow);
        let borrow = _mm256_srli_epi64(_mm256_xor_si256(a[6], borrow_mask), 32);
        a[6] = _mm256_and_si256(a[6], mask32);
        a[7] = _mm256_or_si256(a[7], borrow_mask);
        a[7] = _mm256_sub_epi64(_mm256_sub_epi64(a[7], mod7), borrow);
    }
}

#[inline]
fn reduce_fp4(a: &mut [__m256i; 8]) {
    unsafe {
        let valid_mask = is_valid_mask4(a);

        // we can avoid the sbb if all multiplications are below modulus
        if _mm256_movemask_epi8(valid_mask) != 0 {
            sbb_mask4(a, valid_mask);
        }
    }
}

#[inline]
fn mac4(a: __m256i, b: __m256i, c: __m256i, carry: &mut __m256i) -> __m256i {
    unsafe {
        let tmp = _mm256_add_epi64(_mm256_mul_epu32(b, c), a);
        *carry = _mm256_srli_epi64(tmp, 32);
        _mm256_and_si256(tmp, _mm256_set1_epi64x(0x00000000FFFFFFFF))
    }
}

#[inline]
fn mac_with_carry4(a: __m256i, b: __m256i, c: __m256i, carry: &mut __m256i) -> __m256i {
    unsafe {
        let tmp = _mm256_add_epi64(_mm256_add_epi64(_mm256_mul_epu32(b, c), a), *carry);
        *carry = _mm256_srli_epi64(tmp, 32);
        _mm256_and_si256(tmp, _mm256_set1_epi64x(0x00000000FFFFFFFF))
    }
}

#[inline]
fn add_carry4(a: __m256i, carry: &mut __m256i) -> __m256i {
    unsafe {
        let tmp = _mm256_add_epi64(a, *carry);
        *carry = _mm256_srli_epi64(tmp, 32);
        _mm256_and_si256(tmp, _mm256_set1_epi64x(0x00000000FFFFFFFF))
    }
}

#[inline]
fn mul_fp_inner4(r: &mut [__m256i; 8], a: &[__m256i; 8], b: __m256i, carry1: &mut __m256i) {
    unsafe {
        let zero = _mm256_setzero_si256();
        let mod1 = _mm256_set1_epi64x(0x992d30ed);
        let mod2 = _mm256_set1_epi64x(0x094cf91b);
        let mod3 = _mm256_set1_epi64x(0x224698fc);
        let mod7 = _mm256_set1_epi64x(0x40000000);
        let one = _mm256_set1_epi64x(1);

        r[0] = mac4(r[0], a[0], b, carry1);
        let k = _mm256_sub_epi64(zero, r[0]);
        let mut carry2 = _mm256_add_epi64(_mm256_cmpeq_epi64(r[0], zero), one);
        r[1] = mac_with_carry4(r[1], a[1], b, carry1);
        r[0] = mac_with_carry4(r[1], k, mod1, &mut carry2);
        r[2] = mac_with_carry4(r[2], a[2], b, carry1);
        r[1] = mac_with_carry4(r[2], k, mod2, &mut carry2);
        r[3] = mac_with_carry4(r[3], a[3], b, carry1);
        r[2] = mac_with_carry4(r[3], k, mod3, &mut carry2);
        r[4] = mac_with_carry4(r[4], a[4], b, carry1);
        r[3] = add_carry4(r[4], &mut carry2);
        r[5] = mac_with_carry4(r[5], a[5], b, carry1);
        r[4] = add_carry4(r[5], &mut carry2);
        r[6] = mac_with_carry4(r[6], a[6], b, carry1);
        r[5] = add_carry4(r[6], &mut carry2);
        r[7] = mac_with_carry4(r[7], a[7], b, carry1);
        r[6] = mac_with_carry4(r[7], k, mod7, &mut carry2);
        r[7] = _mm256_add_epi64(*carry1, carry2);
    }
}

#[inline]
pub fn mul_assign_fp4(a: &mut [__m256i; 8], b: &[__m256i; 8]) {
    *a = mul_fp4(a, b);
}

#[inline]
pub fn mul_fp4(a: &[__m256i; 8], b: &[__m256i; 8]) -> [__m256i; 8] {
    unsafe {
        let zero = _mm256_setzero_si256();
        let mut r = [zero; 8];
        let mut carry1 = zero;

        mul_fp_inner4(&mut r, a, b[0], &mut carry1);
        mul_fp_inner4(&mut r, a, b[1], &mut carry1);
        mul_fp_inner4(&mut r, a, b[2], &mut carry1);
        mul_fp_inner4(&mut r, a, b[3], &mut carry1);
        mul_fp_inner4(&mut r, a, b[4], &mut carry1);
        mul_fp_inner4(&mut r, a, b[5], &mut carry1);
        mul_fp_inner4(&mut r, a, b[6], &mut carry1);
        mul_fp_inner4(&mut r, a, b[7], &mut carry1);
        reduce_fp4(&mut r);
        r
    }
}
