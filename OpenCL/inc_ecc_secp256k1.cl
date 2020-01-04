/**
 * Author......: See docs/credits.txt
 * License.....: MIT
 *
 * Furthermore, since elliptic curve operations are highly researched and optimized,
 * we've consulted a lot of online resources to implement this, including several papers and
 * example code.
 *
 * Credits where credits are due: there are a lot of nice projects that explain and/or optimize
 * elliptic curve operations (especially elliptic curve multiplications by a scalar).
 *
 * We want to shout out following projects, which were quite helpful when implementing this:
 * - secp256k1 by Pieter Wuille (https://github.com/bitcoin-core/secp256k1/, MIT)
 * - secp256k1-cl by hhanh00 (https://github.com/hhanh00/secp256k1-cl/, MIT)
 * - ec_pure_c by masterzorag (https://github.com/masterzorag/ec_pure_c/)
 * - ecc-gmp by leivaburto (https://github.com/leivaburto/ecc-gmp)
 * - micro-ecc by Ken MacKay (https://github.com/kmackay/micro-ecc/, BSD)
 * - curve_example by willem (https://gist.github.com/nlitsme/c9031c7b9bf6bb009e5a)
 * - py_ecc by Vitalik Buterin (https://github.com/ethereum/py_ecc/, MIT)
 *
 *
 * Some BigNum operations are implemented similar to micro-ecc which is licensed under these terms:
 *  Copyright 2014 Ken MacKay, 2-Clause BSD License
 *
 *  Redistribution and use in source and binary forms, with or without modification, are permitted
 *  provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice, this list of
 *     conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice, this list of
 *     conditions and the following disclaimer in the documentation and/or other materials
 *     provided with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 *  IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 *  AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * ATTENTION: this code is NOT meant to be used in security critical environments that are at risk
 * of side-channel or timing attacks etc, it's only purpose is to make it work fast for GPGPU
 * (OpenCL/CUDA). Some attack vectors like side-channel and timing-attacks might be possible,
 * because of some optimizations used within this code (non-constant time etc).
 */

/*
 * Implementation considerations:
 * point double and point add are implemented similar to algorithms mentioned in this 2011 paper:
 * http://eprint.iacr.org/2011/338.pdf
 * (Fast and Regular Algorithms for Scalar Multiplication over Elliptic Curves by Matthieu Rivain)
 *
 * In theory we could use the Jacobian Co-Z enhancement to get rid of the larger buffer caused by
 * the z coordinates (and in this way reduce register pressure etc).
 * For the Co-Z improvement there are a lot of fast algorithms, but we might still be faster
 * with this implementation (b/c we allow non-constant time) without the Brier/Joye Montgomery-like
 * ladder. Of course, this claim would need to be verified and tested to see which one is faster
 * for our specific scenario at the end.
 *
 * A speedup could also be possible by using scalars converted to (w)NAF (non-adjacent form) or by
 * just using the windowed (precomputed zi) method or similar improvements:
 * The general idea of w-NAF would be to pre-compute some zi coefficients like below to reduce the
 * costly point additions by using a non-binary ("signed") number system (values other than just
 * 0 and 1, but ranging from -2^(w-1)-1 to 2^(w-1)-1). This would work best with the left-to-right
 * binary algorithm such that we could just add zi * P when adding point P (pre-compute all the
 * possible zi * P values because the x/y coordinates are known before the kernel starts):
 *
 *  // Example with window size w = 2 (i.e. mod 4 => & 3):
 *  // 173 => 1 0 -1 0 -1 0 -1 0 1 = 2^8 - 2^6 - 2^4 - 2^2 + 1
 *  int e = 0b10101101;   // 173
 *  int z[8 + 1] = { 0 }; // our zi/di, we need one extra slot to make the subtraction work
 *
 *  int i = 0;
 *
 *  while (e)
 *  {
 *    if (e & 1)
 *    {
 *      // for window size w = 3 it would be:
 *      // => 2^(w-0) = 2^3 = 8
 *      // => 2^(w-1) = 2^2 = 4
 *
 *      int bit; // = 2 - (e & 3) for w = 2
 *
 *      if ((e & 3) >= 2) // e % 4 == e & 3, use (e & 7) >= 4 for w = 3
 *        bit = (e & 3) - 4; // (e & 7) - 8 for w = 3
 *      else
 *        bit = e & 3; // e & 7 for w = 3
 *
 *      z[i] = bit;
 *      e   -= bit;
 *    }
 *
 *    e >>= 1; // e / 2
 *    i++;
 *  }
*/

#include "inc_ecc_secp256k1.h"

DECLSPEC u32 sub (u32 r[8], const u32 a[8], const u32 b[8])
{
  u32 c = 0; // carry/borrow

  #if defined IS_NV && HAS_SUB == 1 && HAS_SUBC == 1
  asm volatile
  (
    "sub.cc.u32   %0,  %9, %17;"
    "subc.cc.u32  %1, %10, %18;"
    "subc.cc.u32  %2, %11, %19;"
    "subc.cc.u32  %3, %12, %20;"
    "subc.cc.u32  %4, %13, %21;"
    "subc.cc.u32  %5, %14, %22;"
    "subc.cc.u32  %6, %15, %23;"
    "subc.cc.u32  %7, %16, %24;"
    "subc.u32     %8,   0,   0;"
    : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
      "=r"(c)
    :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),  "r"(a[4]),  "r"(a[5]),  "r"(a[6]),  "r"(a[7]),
       "r"(b[0]),  "r"(b[1]),  "r"(b[2]),  "r"(b[3]),  "r"(b[4]),  "r"(b[5]),  "r"(b[6]),  "r"(b[7])
  );
  #elif defined IS_AMD && HAS_VSUB == 1 && HAS_VSUBB == 1
  __asm__ __volatile__
  (
    "V_SUB_U32   %0,  %9, %17;"
    "V_SUBB_U32  %1, %10, %18;"
    "V_SUBB_U32  %2, %11, %19;"
    "V_SUBB_U32  %3, %12, %20;"
    "V_SUBB_U32  %4, %13, %21;"
    "V_SUBB_U32  %5, %14, %22;"
    "V_SUBB_U32  %6, %15, %23;"
    "V_SUBB_U32  %7, %16, %24;"
    "V_SUBB_U32  %8,   0,   0;"
    : "=v"(r[0]), "=v"(r[1]), "=v"(r[2]), "=v"(r[3]), "=v"(r[4]), "=v"(r[5]), "=v"(r[6]), "=v"(r[7]),
      "=v"(c)
    :  "v"(a[0]),  "v"(a[1]),  "v"(a[2]),  "v"(a[3]),  "v"(a[4]),  "v"(a[5]),  "v"(a[6]),  "v"(a[7]),
       "v"(b[0]),  "v"(b[1]),  "v"(b[2]),  "v"(b[3]),  "v"(b[4]),  "v"(b[5]),  "v"(b[6]),  "v"(b[7])
  );
  #else
  for (u32 i = 0; i < 8; i++)
  {
    const u32 diff = a[i] - b[i] - c;

    if (diff != a[i]) c = (diff > a[i]);

    r[i] = diff;
  }
  #endif

  return c;
}

DECLSPEC u32 add (u32 r[8], const u32 a[8], const u32 b[8])
{
  u32 c = 0; // carry/borrow

  #if defined IS_NV && HAS_ADD == 1 && HAS_ADDC == 1
  asm volatile
  (
    "add.cc.u32   %0,  %9, %17;"
    "addc.cc.u32  %1, %10, %18;"
    "addc.cc.u32  %2, %11, %19;"
    "addc.cc.u32  %3, %12, %20;"
    "addc.cc.u32  %4, %13, %21;"
    "addc.cc.u32  %5, %14, %22;"
    "addc.cc.u32  %6, %15, %23;"
    "addc.cc.u32  %7, %16, %24;"
    "addc.u32     %8,   0,   0;"
    : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
      "=r"(c)
    :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),  "r"(a[4]),  "r"(a[5]),  "r"(a[6]),  "r"(a[7]),
       "r"(b[0]),  "r"(b[1]),  "r"(b[2]),  "r"(b[3]),  "r"(b[4]),  "r"(b[5]),  "r"(b[6]),  "r"(b[7])
  );
  #elif defined IS_AMD && HAS_VADD == 1 && HAS_VADDC == 1
  __asm__ __volatile__
  (
    "V_ADD_U32   %0,  %9, %17;"
    "V_ADDC_U32  %1, %10, %18;"
    "V_ADDC_U32  %2, %11, %19;"
    "V_ADDC_U32  %3, %12, %20;"
    "V_ADDC_U32  %4, %13, %21;"
    "V_ADDC_U32  %5, %14, %22;"
    "V_ADDC_U32  %6, %15, %23;"
    "V_ADDC_U32  %7, %16, %24;"
    "V_ADDC_U32  %8,   0,   0;"
    : "=v"(r[0]), "=v"(r[1]), "=v"(r[2]), "=v"(r[3]), "=v"(r[4]), "=v"(r[5]), "=v"(r[6]), "=v"(r[7]),
      "=v"(c)
    :  "v"(a[0]),  "v"(a[1]),  "v"(a[2]),  "v"(a[3]),  "v"(a[4]),  "v"(a[5]),  "v"(a[6]),  "v"(a[7]),
       "v"(b[0]),  "v"(b[1]),  "v"(b[2]),  "v"(b[3]),  "v"(b[4]),  "v"(b[5]),  "v"(b[6]),  "v"(b[7])
  );
  #else
  for (u32 i = 0; i < 8; i++)
  {
    const u32 t = a[i] + b[i] + c;

    if (t != a[i]) c = (t < a[i]);

    r[i] = t;
  }
  #endif

  return c;
}

DECLSPEC void sub_mod (u32 r[8], const u32 a[8], const u32 b[8])
{
  const u32 c = sub (r, a, b); // carry

  if (c)
  {
    u32 t[8];

    t[0] = SECP256K1_P0;
    t[1] = SECP256K1_P1;
    t[2] = SECP256K1_P2;
    t[3] = SECP256K1_P3;
    t[4] = SECP256K1_P4;
    t[5] = SECP256K1_P5;
    t[6] = SECP256K1_P6;
    t[7] = SECP256K1_P7;

    add (r, r, t);
  }
}

DECLSPEC void add_mod (u32 r[8], const u32 a[8], const u32 b[8])
{
  const u32 c = add (r, a, b); // carry

  /*
   * Modulo operation:
   */

  // note: we could have an early exit in case of c == 1 => sub ()

  u32 t[8];

  t[0] = SECP256K1_P0;
  t[1] = SECP256K1_P1;
  t[2] = SECP256K1_P2;
  t[3] = SECP256K1_P3;
  t[4] = SECP256K1_P4;
  t[5] = SECP256K1_P5;
  t[6] = SECP256K1_P6;
  t[7] = SECP256K1_P7;

  // check if modulo operation is needed

  u32 mod = 1;

  if (c == 0)
  {
    for (int i = 7; i >= 0; i--)
    {
      if (r[i] < t[i])
      {
        mod = 0;

        break; // or return ! (check if faster)
      }

      if (r[i] > t[i]) break;
    }
  }

  if (mod == 1)
  {
    sub (r, r, t);
  }
}

DECLSPEC void mod_512 (u32 n[16])
{
  // we need to perform a modulo operation with 512-bit % 256-bit (bignum modulo):
  // the modulus is the secp256k1 group order

  // ATTENTION: for this function the byte-order is reversed (most significant bytes
  // at the left)

  /*
    the general modulo by shift and substract code (a = a % b):

    x = b;

    t = a >> 1;

    while (x <= t) x <<= 1;

    while (a >= b)
    {
      if (a >= x) a -= x;

      x >>= 1;
    }

    return a; // remainder
  */

  u32 a[16];

  a[ 0] = n[ 0];
  a[ 1] = n[ 1];
  a[ 2] = n[ 2];
  a[ 3] = n[ 3];
  a[ 4] = n[ 4];
  a[ 5] = n[ 5];
  a[ 6] = n[ 6];
  a[ 7] = n[ 7];
  a[ 8] = n[ 8];
  a[ 9] = n[ 9];
  a[10] = n[10];
  a[11] = n[11];
  a[12] = n[12];
  a[13] = n[13];
  a[14] = n[14];
  a[15] = n[15];

  u32 b[16];

  b[ 0] = 0x00000000;
  b[ 1] = 0x00000000;
  b[ 2] = 0x00000000;
  b[ 3] = 0x00000000;
  b[ 4] = 0x00000000;
  b[ 5] = 0x00000000;
  b[ 6] = 0x00000000;
  b[ 7] = 0x00000000;
  b[ 8] = SECP256K1_N7;
  b[ 9] = SECP256K1_N6;
  b[10] = SECP256K1_N5;
  b[11] = SECP256K1_N4;
  b[12] = SECP256K1_N3;
  b[13] = SECP256K1_N2;
  b[14] = SECP256K1_N1;
  b[15] = SECP256K1_N0;

  /*
   * Start:
   */

  // x = b (but with a fast "shift" trick to avoid the while loop)

  u32 x[16];

  x[ 0] = b[ 8]; // this is a trick: we just put the group order's most significant bit all the
  x[ 1] = b[ 9]; // way to the top to avoid doing the initial: while (x <= t) x <<= 1
  x[ 2] = b[10];
  x[ 3] = b[11];
  x[ 4] = b[12];
  x[ 5] = b[13];
  x[ 6] = b[14];
  x[ 7] = b[15];
  x[ 8] = 0x00000000;
  x[ 9] = 0x00000000;
  x[10] = 0x00000000;
  x[11] = 0x00000000;
  x[12] = 0x00000000;
  x[13] = 0x00000000;
  x[14] = 0x00000000;
  x[15] = 0x00000000;

  // a >= b

  while (a[0] >= b[0])
  {
    const u32 l1 = (a[ 0]  < b[ 0]) <<  0
                 | (a[ 1]  < b[ 1]) <<  1
                 | (a[ 2]  < b[ 2]) <<  2
                 | (a[ 3]  < b[ 3]) <<  3
                 | (a[ 4]  < b[ 4]) <<  4
                 | (a[ 5]  < b[ 5]) <<  5
                 | (a[ 6]  < b[ 6]) <<  6
                 | (a[ 7]  < b[ 7]) <<  7
                 | (a[ 8]  < b[ 8]) <<  8
                 | (a[ 9]  < b[ 9]) <<  9
                 | (a[10]  < b[10]) << 10
                 | (a[11]  < b[11]) << 11
                 | (a[12]  < b[12]) << 12
                 | (a[13]  < b[13]) << 13
                 | (a[14]  < b[14]) << 14
                 | (a[15]  < b[15]) << 15;

    const u32 e1 = (a[ 0] == b[ 0]) <<  0
                 | (a[ 1] == b[ 1]) <<  1
                 | (a[ 2] == b[ 2]) <<  2
                 | (a[ 3] == b[ 3]) <<  3
                 | (a[ 4] == b[ 4]) <<  4
                 | (a[ 5] == b[ 5]) <<  5
                 | (a[ 6] == b[ 6]) <<  6
                 | (a[ 7] == b[ 7]) <<  7
                 | (a[ 8] == b[ 8]) <<  8
                 | (a[ 9] == b[ 9]) <<  9
                 | (a[10] == b[10]) << 10
                 | (a[11] == b[11]) << 11
                 | (a[12] == b[12]) << 12
                 | (a[13] == b[13]) << 13
                 | (a[14] == b[14]) << 14
                 | (a[15] == b[15]) << 15;

    if (l1)
    {
      if (l1 & 0x0001)                              break;
      if (l1 & 0x0002) if ((e1 & 0x0001) == 0x0001) break;
      if (l1 & 0x0004) if ((e1 & 0x0003) == 0x0003) break;
      if (l1 & 0x0008) if ((e1 & 0x0007) == 0x0007) break;
      if (l1 & 0x0010) if ((e1 & 0x000f) == 0x000f) break;
      if (l1 & 0x0020) if ((e1 & 0x001f) == 0x001f) break;
      if (l1 & 0x0040) if ((e1 & 0x003f) == 0x003f) break;
      if (l1 & 0x0080) if ((e1 & 0x007f) == 0x007f) break;
      if (l1 & 0x0100) if ((e1 & 0x00ff) == 0x00ff) break;
      if (l1 & 0x0200) if ((e1 & 0x01ff) == 0x01ff) break;
      if (l1 & 0x0400) if ((e1 & 0x03ff) == 0x03ff) break;
      if (l1 & 0x0800) if ((e1 & 0x07ff) == 0x07ff) break;
      if (l1 & 0x1000) if ((e1 & 0x0fff) == 0x0fff) break;
      if (l1 & 0x2000) if ((e1 & 0x1fff) == 0x1fff) break;
      if (l1 & 0x4000) if ((e1 & 0x3fff) == 0x3fff) break;
      if (l1 & 0x8000) if ((e1 & 0x7fff) == 0x7fff) break;
    }

    // r = x (copy it to have the original values for the subtraction)

    u32 r[16];

    r[ 0] = x[ 0];
    r[ 1] = x[ 1];
    r[ 2] = x[ 2];
    r[ 3] = x[ 3];
    r[ 4] = x[ 4];
    r[ 5] = x[ 5];
    r[ 6] = x[ 6];
    r[ 7] = x[ 7];
    r[ 8] = x[ 8];
    r[ 9] = x[ 9];
    r[10] = x[10];
    r[11] = x[11];
    r[12] = x[12];
    r[13] = x[13];
    r[14] = x[14];
    r[15] = x[15];

    // x >>= 1

    x[15] = x[15] >> 1 | (x[14] & 1) << 31;
    x[14] = x[14] >> 1 | (x[13] & 1) << 31;
    x[13] = x[13] >> 1 | (x[12] & 1) << 31;
    x[12] = x[12] >> 1 | (x[11] & 1) << 31;
    x[11] = x[11] >> 1 | (x[10] & 1) << 31;
    x[10] = x[10] >> 1 | (x[ 9] & 1) << 31;
    x[ 9] = x[ 9] >> 1 | (x[ 8] & 1) << 31;
    x[ 8] = x[ 8] >> 1 | (x[ 7] & 1) << 31;
    x[ 7] = x[ 7] >> 1 | (x[ 6] & 1) << 31;
    x[ 6] = x[ 6] >> 1 | (x[ 5] & 1) << 31;
    x[ 5] = x[ 5] >> 1 | (x[ 4] & 1) << 31;
    x[ 4] = x[ 4] >> 1 | (x[ 3] & 1) << 31;
    x[ 3] = x[ 3] >> 1 | (x[ 2] & 1) << 31;
    x[ 2] = x[ 2] >> 1 | (x[ 1] & 1) << 31;
    x[ 1] = x[ 1] >> 1 | (x[ 0] & 1) << 31;
    x[ 0] = x[ 0] >> 1;

    // if (a >= r) a -= r;

    const u32 l2 = (a[ 0]  < r[ 0]) <<  0
                 | (a[ 1]  < r[ 1]) <<  1
                 | (a[ 2]  < r[ 2]) <<  2
                 | (a[ 3]  < r[ 3]) <<  3
                 | (a[ 4]  < r[ 4]) <<  4
                 | (a[ 5]  < r[ 5]) <<  5
                 | (a[ 6]  < r[ 6]) <<  6
                 | (a[ 7]  < r[ 7]) <<  7
                 | (a[ 8]  < r[ 8]) <<  8
                 | (a[ 9]  < r[ 9]) <<  9
                 | (a[10]  < r[10]) << 10
                 | (a[11]  < r[11]) << 11
                 | (a[12]  < r[12]) << 12
                 | (a[13]  < r[13]) << 13
                 | (a[14]  < r[14]) << 14
                 | (a[15]  < r[15]) << 15;

    const u32 e2 = (a[ 0] == r[ 0]) <<  0
                 | (a[ 1] == r[ 1]) <<  1
                 | (a[ 2] == r[ 2]) <<  2
                 | (a[ 3] == r[ 3]) <<  3
                 | (a[ 4] == r[ 4]) <<  4
                 | (a[ 5] == r[ 5]) <<  5
                 | (a[ 6] == r[ 6]) <<  6
                 | (a[ 7] == r[ 7]) <<  7
                 | (a[ 8] == r[ 8]) <<  8
                 | (a[ 9] == r[ 9]) <<  9
                 | (a[10] == r[10]) << 10
                 | (a[11] == r[11]) << 11
                 | (a[12] == r[12]) << 12
                 | (a[13] == r[13]) << 13
                 | (a[14] == r[14]) << 14
                 | (a[15] == r[15]) << 15;

    if (l2)
    {
      if (l2 & 0x0001)                              continue;
      if (l2 & 0x0002) if ((e2 & 0x0001) == 0x0001) continue;
      if (l2 & 0x0004) if ((e2 & 0x0003) == 0x0003) continue;
      if (l2 & 0x0008) if ((e2 & 0x0007) == 0x0007) continue;
      if (l2 & 0x0010) if ((e2 & 0x000f) == 0x000f) continue;
      if (l2 & 0x0020) if ((e2 & 0x001f) == 0x001f) continue;
      if (l2 & 0x0040) if ((e2 & 0x003f) == 0x003f) continue;
      if (l2 & 0x0080) if ((e2 & 0x007f) == 0x007f) continue;
      if (l2 & 0x0100) if ((e2 & 0x00ff) == 0x00ff) continue;
      if (l2 & 0x0200) if ((e2 & 0x01ff) == 0x01ff) continue;
      if (l2 & 0x0400) if ((e2 & 0x03ff) == 0x03ff) continue;
      if (l2 & 0x0800) if ((e2 & 0x07ff) == 0x07ff) continue;
      if (l2 & 0x1000) if ((e2 & 0x0fff) == 0x0fff) continue;
      if (l2 & 0x2000) if ((e2 & 0x1fff) == 0x1fff) continue;
      if (l2 & 0x4000) if ((e2 & 0x3fff) == 0x3fff) continue;
      if (l2 & 0x8000) if ((e2 & 0x7fff) == 0x7fff) continue;
    }

    // substract (a -= r):

    r[ 0] = a[ 0] - r[ 0];
    r[ 1] = a[ 1] - r[ 1];
    r[ 2] = a[ 2] - r[ 2];
    r[ 3] = a[ 3] - r[ 3];
    r[ 4] = a[ 4] - r[ 4];
    r[ 5] = a[ 5] - r[ 5];
    r[ 6] = a[ 6] - r[ 6];
    r[ 7] = a[ 7] - r[ 7];
    r[ 8] = a[ 8] - r[ 8];
    r[ 9] = a[ 9] - r[ 9];
    r[10] = a[10] - r[10];
    r[11] = a[11] - r[11];
    r[12] = a[12] - r[12];
    r[13] = a[13] - r[13];
    r[14] = a[14] - r[14];
    r[15] = a[15] - r[15];

    // take care of the "borrow" (we can't do it the other way around 15...1 because r[x] is changed!)

    if (r[ 1] > a[ 1]) r[ 0]--;
    if (r[ 2] > a[ 2]) r[ 1]--;
    if (r[ 3] > a[ 3]) r[ 2]--;
    if (r[ 4] > a[ 4]) r[ 3]--;
    if (r[ 5] > a[ 5]) r[ 4]--;
    if (r[ 6] > a[ 6]) r[ 5]--;
    if (r[ 7] > a[ 7]) r[ 6]--;
    if (r[ 8] > a[ 8]) r[ 7]--;
    if (r[ 9] > a[ 9]) r[ 8]--;
    if (r[10] > a[10]) r[ 9]--;
    if (r[11] > a[11]) r[10]--;
    if (r[12] > a[12]) r[11]--;
    if (r[13] > a[13]) r[12]--;
    if (r[14] > a[14]) r[13]--;
    if (r[15] > a[15]) r[14]--;

    a[ 0] = r[ 0];
    a[ 1] = r[ 1];
    a[ 2] = r[ 2];
    a[ 3] = r[ 3];
    a[ 4] = r[ 4];
    a[ 5] = r[ 5];
    a[ 6] = r[ 6];
    a[ 7] = r[ 7];
    a[ 8] = r[ 8];
    a[ 9] = r[ 9];
    a[10] = r[10];
    a[11] = r[11];
    a[12] = r[12];
    a[13] = r[13];
    a[14] = r[14];
    a[15] = r[15];
  }

  n[ 0] = a[ 0];
  n[ 1] = a[ 1];
  n[ 2] = a[ 2];
  n[ 3] = a[ 3];
  n[ 4] = a[ 4];
  n[ 5] = a[ 5];
  n[ 6] = a[ 6];
  n[ 7] = a[ 7];
  n[ 8] = a[ 8];
  n[ 9] = a[ 9];
  n[10] = a[10];
  n[11] = a[11];
  n[12] = a[12];
  n[13] = a[13];
  n[14] = a[14];
  n[15] = a[15];
}

DECLSPEC void mul_mod (u32 r[8], const u32 a[8], const u32 b[8]) // TODO get rid of u64 ?
{
  u32 t[16] = { 0 }; // we need up to double the space (2 * 8)

  /*
   * First start with the basic a * b multiplication:
   */

  u32 t0 = 0;
  u32 t1 = 0;
  u32 c  = 0;

  for (u32 i = 0; i < 8; i++)
  {
    for (u32 j = 0; j <= i; j++)
    {
      u64 p = ((u64) a[j]) * b[i - j];

      u64 d = ((u64) t1) << 32 | t0;

      d += p;

      t0 = (u32) d;
      t1 = d >> 32;

      c += d < p; // carry
    }

    t[i] = t0;

    t0 = t1;
    t1 = c;

    c = 0;
  }

  for (u32 i = 8; i < 15; i++)
  {
    for (u32 j = i - 7; j < 8; j++)
    {
      u64 p = ((u64) a[j]) * b[i - j];

      u64 d = ((u64) t1) << 32 | t0;

      d += p;

      t0 = (u32) d;
      t1 = d >> 32;

      c += d < p;
    }

    t[i] = t0;

    t0 = t1;
    t1 = c;

    c = 0;
  }

  t[15] = t0;



  /*
   * Now do the modulo operation:
   * (r = t % p)
   *
   * http://www.isys.uni-klu.ac.at/PDF/2001-0126-MT.pdf (p.354 or p.9 in that document)
   */

  u32 tmp[16] = { 0 };

  // c = 0;

  // Note: SECP256K1_P = 2^256 - 2^32 - 977 (0x03d1 = 977)
  // multiply t[8]...t[15] by omega:

  for (u32 i = 0, j = 8; i < 8; i++, j++)
  {
    u64 p = ((u64) 0x03d1) * t[j] + c;

    tmp[i] = (u32) p;

    c = p >> 32;
  }

  tmp[8] = c;

  c = add (tmp + 1, tmp + 1, t + 8); // modifies tmp[1]...tmp[8]

  tmp[9] = c;


  // r = t + tmp

  c = add (r, t, tmp);

  // multiply t[0]...t[7] by omega:

  u32 c2 = 0;

  // memset (t, 0, sizeof (t));

  for (u32 i = 0, j = 8; i < 8; i++, j++)
  {
    u64 p = ((u64) 0x3d1) * tmp[j] + c2;

    t[i] = (u32) p;

    c2 = p >> 32;
  }

  t[8] = c2;

  c2 = add (t + 1, t + 1, tmp + 8); // modifies t[1]...t[8]

  t[9] = c2;


  // r = r + t

  c2 = add (r, r, t);

  c += c2;

  t[0] = SECP256K1_P0;
  t[1] = SECP256K1_P1;
  t[2] = SECP256K1_P2;
  t[3] = SECP256K1_P3;
  t[4] = SECP256K1_P4;
  t[5] = SECP256K1_P5;
  t[6] = SECP256K1_P6;
  t[7] = SECP256K1_P7;

  for (u32 i = c; i > 0; i--)
  {
    sub (r, r, t);
  }

  for (int i = 7; i >= 0; i--)
  {
    if (r[i] < t[i]) break;

    if (r[i] > t[i])
    {
      sub (r, r, t);

      break;
    }
  }
}

DECLSPEC void sqrt_mod (u32 r[8])
{
  // Fermat's Little Theorem
  // secp256k1: y^2 = x^3 + 7 % p
  // y ^ (p - 1) = 1
  // y ^ (p - 1) = (y^2) ^ ((p - 1) / 2) = 1 => y^2 = (y^2) ^ (((p - 1) / 2) + 1)
  // => y = (y^2) ^ ((((p - 1) / 2) + 1) / 2)
  // y = (y^2) ^ (((p - 1 + 2) / 2) / 2) = (y^2) ^ ((p + 1) / 4)

  // y1 = (x^3 + 7) ^ ((p + 1) / 4)
  // y2 = p - y1 (or y2 = y1 * -1 % p)

  u32 s[8];

  s[0] = SECP256K1_P0 + 1; //  because of (p + 1) / 4 or use add (s, s, 1)
  s[1] = SECP256K1_P1;
  s[2] = SECP256K1_P2;
  s[3] = SECP256K1_P3;
  s[4] = SECP256K1_P4;
  s[5] = SECP256K1_P5;
  s[6] = SECP256K1_P6;
  s[7] = SECP256K1_P7;

  u32 t[8] = { 0 };

  t[0] = 1;

  for (u32 i = 255; i > 1; i--) // we just skip the last 2 multiplications (=> exp / 4)
  {
    mul_mod (t, t, t); // r * r

    u32 idx  = i >> 5;
    u32 mask = 1 << (i & 0x1f);

    if (s[idx] & mask)
    {
      mul_mod (t, t, r); // t * r
    }
  }

  r[0] = t[0];
  r[1] = t[1];
  r[2] = t[2];
  r[3] = t[3];
  r[4] = t[4];
  r[5] = t[5];
  r[6] = t[6];
  r[7] = t[7];
}

// (inverse (a, p) * a) % p == 1 (or think of a * a^-1 = a / a = 1)

DECLSPEC void inv_mod (u32 a[8])
{
  // How often does this really happen? it should "almost" never happen (but would be safer)
  // if ((a[0] | a[1] | a[2] | a[3] | a[4] | a[5] | a[6] | a[7]) == 0) return;

  u32 t0[8];

  t0[0] = a[0];
  t0[1] = a[1];
  t0[2] = a[2];
  t0[3] = a[3];
  t0[4] = a[4];
  t0[5] = a[5];
  t0[6] = a[6];
  t0[7] = a[7];

  u32 p[8];

  p[0] = SECP256K1_P0;
  p[1] = SECP256K1_P1;
  p[2] = SECP256K1_P2;
  p[3] = SECP256K1_P3;
  p[4] = SECP256K1_P4;
  p[5] = SECP256K1_P5;
  p[6] = SECP256K1_P6;
  p[7] = SECP256K1_P7;

  u32 t1[8];

  t1[0] = SECP256K1_P0;
  t1[1] = SECP256K1_P1;
  t1[2] = SECP256K1_P2;
  t1[3] = SECP256K1_P3;
  t1[4] = SECP256K1_P4;
  t1[5] = SECP256K1_P5;
  t1[6] = SECP256K1_P6;
  t1[7] = SECP256K1_P7;

  u32 t2[8] = { 0 };

  t2[0] = 0x00000001;

  u32 t3[8] = { 0 };

  u32 b = (t0[0] != t1[0])
        | (t0[1] != t1[1])
        | (t0[2] != t1[2])
        | (t0[3] != t1[3])
        | (t0[4] != t1[4])
        | (t0[5] != t1[5])
        | (t0[6] != t1[6])
        | (t0[7] != t1[7]);

  while (b)
  {
    if ((t0[0] & 1) == 0) // even
    {
      t0[0] = t0[0] >> 1 | t0[1] << 31;
      t0[1] = t0[1] >> 1 | t0[2] << 31;
      t0[2] = t0[2] >> 1 | t0[3] << 31;
      t0[3] = t0[3] >> 1 | t0[4] << 31;
      t0[4] = t0[4] >> 1 | t0[5] << 31;
      t0[5] = t0[5] >> 1 | t0[6] << 31;
      t0[6] = t0[6] >> 1 | t0[7] << 31;
      t0[7] = t0[7] >> 1;

      u32 c = 0;

      if (t2[0] & 1) c = add (t2, t2, p);

      t2[0] = t2[0] >> 1 | t2[1] << 31;
      t2[1] = t2[1] >> 1 | t2[2] << 31;
      t2[2] = t2[2] >> 1 | t2[3] << 31;
      t2[3] = t2[3] >> 1 | t2[4] << 31;
      t2[4] = t2[4] >> 1 | t2[5] << 31;
      t2[5] = t2[5] >> 1 | t2[6] << 31;
      t2[6] = t2[6] >> 1 | t2[7] << 31;
      t2[7] = t2[7] >> 1 | c     << 31;
    }
    else if ((t1[0] & 1) == 0)
    {
      t1[0] = t1[0] >> 1 | t1[1] << 31;
      t1[1] = t1[1] >> 1 | t1[2] << 31;
      t1[2] = t1[2] >> 1 | t1[3] << 31;
      t1[3] = t1[3] >> 1 | t1[4] << 31;
      t1[4] = t1[4] >> 1 | t1[5] << 31;
      t1[5] = t1[5] >> 1 | t1[6] << 31;
      t1[6] = t1[6] >> 1 | t1[7] << 31;
      t1[7] = t1[7] >> 1;

      u32 c = 0;

      if (t3[0] & 1) c = add (t3, t3, p);

      t3[0] = t3[0] >> 1 | t3[1] << 31;
      t3[1] = t3[1] >> 1 | t3[2] << 31;
      t3[2] = t3[2] >> 1 | t3[3] << 31;
      t3[3] = t3[3] >> 1 | t3[4] << 31;
      t3[4] = t3[4] >> 1 | t3[5] << 31;
      t3[5] = t3[5] >> 1 | t3[6] << 31;
      t3[6] = t3[6] >> 1 | t3[7] << 31;
      t3[7] = t3[7] >> 1 | c     << 31;
    }
    else
    {
      u32 gt = 0;

      for (int i = 7; i >= 0; i--)
      {
        if (t0[i] > t1[i])
        {
          gt = 1;

          break;
        }

        if (t0[i] < t1[i]) break;
      }

      if (gt)
      {
        sub (t0, t0, t1);

        t0[0] = t0[0] >> 1 | t0[1] << 31;
        t0[1] = t0[1] >> 1 | t0[2] << 31;
        t0[2] = t0[2] >> 1 | t0[3] << 31;
        t0[3] = t0[3] >> 1 | t0[4] << 31;
        t0[4] = t0[4] >> 1 | t0[5] << 31;
        t0[5] = t0[5] >> 1 | t0[6] << 31;
        t0[6] = t0[6] >> 1 | t0[7] << 31;
        t0[7] = t0[7] >> 1;

        u32 lt = 0;

        for (int i = 7; i >= 0; i--)
        {
          if (t2[i] < t3[i])
          {
            lt = 1;

            break;
          }

          if (t2[i] > t3[i]) break;
        }

        if (lt) add (t2, t2, p);

        sub (t2, t2, t3);

        u32 c = 0;

        if (t2[0] & 1) c = add (t2, t2, p);

        t2[0] = t2[0] >> 1 | t2[1] << 31;
        t2[1] = t2[1] >> 1 | t2[2] << 31;
        t2[2] = t2[2] >> 1 | t2[3] << 31;
        t2[3] = t2[3] >> 1 | t2[4] << 31;
        t2[4] = t2[4] >> 1 | t2[5] << 31;
        t2[5] = t2[5] >> 1 | t2[6] << 31;
        t2[6] = t2[6] >> 1 | t2[7] << 31;
        t2[7] = t2[7] >> 1 | c     << 31;
      }
      else
      {
        sub (t1, t1, t0);

        t1[0] = t1[0] >> 1 | t1[1] << 31;
        t1[1] = t1[1] >> 1 | t1[2] << 31;
        t1[2] = t1[2] >> 1 | t1[3] << 31;
        t1[3] = t1[3] >> 1 | t1[4] << 31;
        t1[4] = t1[4] >> 1 | t1[5] << 31;
        t1[5] = t1[5] >> 1 | t1[6] << 31;
        t1[6] = t1[6] >> 1 | t1[7] << 31;
        t1[7] = t1[7] >> 1;

        u32 lt = 0;

        for (int i = 7; i >= 0; i--)
        {
          if (t3[i] < t2[i])
          {
            lt = 1;

            break;
          }

          if (t3[i] > t2[i]) break;
        }

        if (lt) add (t3, t3, p);

        sub (t3, t3, t2);

        u32 c = 0;

        if (t3[0] & 1) c = add (t3, t3, p);

        t3[0] = t3[0] >> 1 | t3[1] << 31;
        t3[1] = t3[1] >> 1 | t3[2] << 31;
        t3[2] = t3[2] >> 1 | t3[3] << 31;
        t3[3] = t3[3] >> 1 | t3[4] << 31;
        t3[4] = t3[4] >> 1 | t3[5] << 31;
        t3[5] = t3[5] >> 1 | t3[6] << 31;
        t3[6] = t3[6] >> 1 | t3[7] << 31;
        t3[7] = t3[7] >> 1 | c     << 31;
      }
    }

    // update b:

    b = (t0[0] != t1[0])
      | (t0[1] != t1[1])
      | (t0[2] != t1[2])
      | (t0[3] != t1[3])
      | (t0[4] != t1[4])
      | (t0[5] != t1[5])
      | (t0[6] != t1[6])
      | (t0[7] != t1[7]);
  }

  // set result:

  a[0] = t2[0];
  a[1] = t2[1];
  a[2] = t2[2];
  a[3] = t2[3];
  a[4] = t2[4];
  a[5] = t2[5];
  a[6] = t2[6];
  a[7] = t2[7];
}

/*
  // everything from the formulas below of course MOD the prime:

  // we use this formula:

  X = (3/2 * x^2)^2 - 2 * x * y^2
  Y = (3/2 * x^2) * (x * y^2 - X) - y^4
  Z = y * z

  this is identical to the more frequently used form:

  X = (3 * x^2)^2 - 8 * x * y^2
  Y =  3 * x^2 * (4 * x * y^2 - X) - 8 * y^4
  Z =  2 * y * z
*/

DECLSPEC void point_double (u32 x[8], u32 y[8], u32 z[8])
{
  // How often does this really happen? it should "almost" never happen (but would be safer)

  /*
  if ((y[0] | y[1] | y[2] | y[3] | y[4] | y[5] | y[6] | y[7]) == 0)
  {
    x[0] = 0;
    x[1] = 0;
    x[2] = 0;
    x[3] = 0;
    x[4] = 0;
    x[5] = 0;
    x[6] = 0;
    x[7] = 0;

    y[0] = 0;
    y[1] = 0;
    y[2] = 0;
    y[3] = 0;
    y[4] = 0;
    y[5] = 0;
    y[6] = 0;
    y[7] = 0;

    z[0] = 0;
    z[1] = 0;
    z[2] = 0;
    z[3] = 0;
    z[4] = 0;
    z[5] = 0;
    z[6] = 0;
    z[7] = 0;

    return;
  }
  */

  u32 t1[8];

  t1[0] = x[0];
  t1[1] = x[1];
  t1[2] = x[2];
  t1[3] = x[3];
  t1[4] = x[4];
  t1[5] = x[5];
  t1[6] = x[6];
  t1[7] = x[7];

  u32 t2[8];

  t2[0] = y[0];
  t2[1] = y[1];
  t2[2] = y[2];
  t2[3] = y[3];
  t2[4] = y[4];
  t2[5] = y[5];
  t2[6] = y[6];
  t2[7] = y[7];

  u32 t3[8];

  t3[0] = z[0];
  t3[1] = z[1];
  t3[2] = z[2];
  t3[3] = z[3];
  t3[4] = z[4];
  t3[5] = z[5];
  t3[6] = z[6];
  t3[7] = z[7];

  u32 t4[8];
  u32 t5[8];
  u32 t6[8];

  mul_mod (t4, t1, t1); // t4 = x^2

  mul_mod (t5, t2, t2); // t5 = y^2

  mul_mod (t1, t1, t5); // t1 = x*y^2

  mul_mod (t5, t5, t5); // t5 = t5^2 = y^4

  // here the z^2 and z^4 is not needed for a = 0

  mul_mod (t3, t2, t3); // t3 = x * z

  add_mod (t2, t4, t4); // t2 = 2 * t4 = 2 * x^2
  add_mod (t4, t4, t2); // t4 = 3 * t4 = 3 * x^2

  // a * z^4 = 0 * 1^4 = 0

  // don't discard the least significant bit it's important too!

  u32 c = 0;

  if (t4[0] & 1)
  {
    u32 t[8];

    t[0] = SECP256K1_P0;
    t[1] = SECP256K1_P1;
    t[2] = SECP256K1_P2;
    t[3] = SECP256K1_P3;
    t[4] = SECP256K1_P4;
    t[5] = SECP256K1_P5;
    t[6] = SECP256K1_P6;
    t[7] = SECP256K1_P7;

    c = add (t4, t4, t); // t4 + SECP256K1_P
  }

  // right shift (t4 / 2):

  t4[0] = t4[0] >> 1 | t4[1] << 31;
  t4[1] = t4[1] >> 1 | t4[2] << 31;
  t4[2] = t4[2] >> 1 | t4[3] << 31;
  t4[3] = t4[3] >> 1 | t4[4] << 31;
  t4[4] = t4[4] >> 1 | t4[5] << 31;
  t4[5] = t4[5] >> 1 | t4[6] << 31;
  t4[6] = t4[6] >> 1 | t4[7] << 31;
  t4[7] = t4[7] >> 1 | c     << 31;

  mul_mod (t6, t4, t4); // t6 = t4^2 = (3/2 * x^2)^2

  add_mod (t2, t1, t1); // t2 = 2 * t1

  sub_mod (t6, t6, t2); // t6 = t6 - t2
  sub_mod (t1, t1, t6); // t1 = t1 - t6

  mul_mod (t4, t4, t1); // t4 = t4 * t1

  sub_mod (t1, t4, t5); // t1 = t4 - t5

  // => x = t6, y = t1, z = t3:

  x[0] = t6[0];
  x[1] = t6[1];
  x[2] = t6[2];
  x[3] = t6[3];
  x[4] = t6[4];
  x[5] = t6[5];
  x[6] = t6[6];
  x[7] = t6[7];

  y[0] = t1[0];
  y[1] = t1[1];
  y[2] = t1[2];
  y[3] = t1[3];
  y[4] = t1[4];
  y[5] = t1[5];
  y[6] = t1[6];
  y[7] = t1[7];

  z[0] = t3[0];
  z[1] = t3[1];
  z[2] = t3[2];
  z[3] = t3[3];
  z[4] = t3[4];
  z[5] = t3[5];
  z[6] = t3[6];
  z[7] = t3[7];
}

DECLSPEC void point_add (u32 x1[8], u32 y1[8], u32 z1[8], const u32 x2[8], const u32 y2[8])
{
  // How often does this really happen? it should "almost" never happen (but would be safer)

  /*
  if ((y2[0] | y2[1] | y2[2] | y2[3] | y2[4] | y2[5] | y2[6] | y2[7]) == 0) return;

  if ((y1[0] | y1[1] | y1[2] | y1[3] | y1[4] | y1[5] | y1[6] | y1[7]) == 0)
  {
    x1[0] = x2[0];
    x1[1] = x2[1];
    x1[2] = x2[2];
    x1[3] = x2[3];
    x1[4] = x2[4];
    x1[5] = x2[5];
    x1[6] = x2[6];
    x1[7] = x2[7];

    y1[0] = y2[0];
    y1[1] = y2[1];
    y1[2] = y2[2];
    y1[3] = y2[3];
    y1[4] = y2[4];
    y1[5] = y2[5];
    y1[6] = y2[6];
    y1[7] = y2[7];

    z1[0] = z2[0];
    z1[1] = z2[1];
    z1[2] = z2[2];
    z1[3] = z2[3];
    z1[4] = z2[4];
    z1[5] = z2[5];
    z1[6] = z2[6];
    z1[7] = z2[7];

    return;
  }
  */

  // if x1 == x2 and y1 == y2 and z1 == z2 we need to double instead?

  // x1/y1/z1:

  u32 t1[8];

  t1[0] = x1[0];
  t1[1] = x1[1];
  t1[2] = x1[2];
  t1[3] = x1[3];
  t1[4] = x1[4];
  t1[5] = x1[5];
  t1[6] = x1[6];
  t1[7] = x1[7];

  u32 t2[8];

  t2[0] = y1[0];
  t2[1] = y1[1];
  t2[2] = y1[2];
  t2[3] = y1[3];
  t2[4] = y1[4];
  t2[5] = y1[5];
  t2[6] = y1[6];
  t2[7] = y1[7];

  u32 t3[8];

  t3[0] = z1[0];
  t3[1] = z1[1];
  t3[2] = z1[2];
  t3[3] = z1[3];
  t3[4] = z1[4];
  t3[5] = z1[5];
  t3[6] = z1[6];
  t3[7] = z1[7];

  // x2/y2:

  u32 t4[8];

  t4[0] = x2[0];
  t4[1] = x2[1];
  t4[2] = x2[2];
  t4[3] = x2[3];
  t4[4] = x2[4];
  t4[5] = x2[5];
  t4[6] = x2[6];
  t4[7] = x2[7];

  u32 t5[8];

  t5[0] = y2[0];
  t5[1] = y2[1];
  t5[2] = y2[2];
  t5[3] = y2[3];
  t5[4] = y2[4];
  t5[5] = y2[5];
  t5[6] = y2[6];
  t5[7] = y2[7];

  u32 t6[8];

  u32 t7[8];

  mul_mod (t7, t3, t3); // t7 = z1^2
  mul_mod (t4, t4, t7); // t4 = x2 * z1^2 = B

  mul_mod (t5, t5, t3); // t5 = y2 * z1
  mul_mod (t5, t5, t7); // t5 = y2 * z1^3 = D

  sub_mod (t1, t1, t4); // t1 = A - B = E

  mul_mod (t3, t1, t3); // t3 = z1 * z2 * E = Z3

  sub_mod (t2, t2, t5); // t2 = C - D = F

  mul_mod (t7, t1, t1); // t7 = E^2
  mul_mod (t6, t2, t2); // t6 = F^2

  mul_mod (t4, t4, t7); // t4 = B * E^2
  mul_mod (t1, t7, t1); // t1 = E^3

  sub_mod (t6, t6, t1); // t6 = F^2 - E^3

  add_mod (t7, t4, t4); // t7 = 2 * B * E^2

  sub_mod (t6, t6, t7); // t6 = F^2 - E^2 - 2 * B * E^2 = X3
  sub_mod (t4, t4, t6); // t4 = B * E^2 - X3

  mul_mod (t2, t2, t4); // t2 = F * (B * E^2 - X3)
  mul_mod (t7, t5, t1); // t7 = D * E^3

  sub_mod (t7, t2, t7); // t7 = F * (B * E^2 - X3) - D * E^3 = Y3

  x1[0] = t6[0];
  x1[1] = t6[1];
  x1[2] = t6[2];
  x1[3] = t6[3];
  x1[4] = t6[4];
  x1[5] = t6[5];
  x1[6] = t6[6];
  x1[7] = t6[7];

  y1[0] = t7[0];
  y1[1] = t7[1];
  y1[2] = t7[2];
  y1[3] = t7[3];
  y1[4] = t7[4];
  y1[5] = t7[5];
  y1[6] = t7[6];
  y1[7] = t7[7];

  z1[0] = t3[0];
  z1[1] = t3[1];
  z1[2] = t3[2];
  z1[3] = t3[3];
  z1[4] = t3[4];
  z1[5] = t3[5];
  z1[6] = t3[6];
  z1[7] = t3[7];
}

DECLSPEC void point_get_coords (secp256k1_t *r, const u32 x[8], const u32 y[8])
{
  // init the values with x and y:

  u32 x1[8];

  x1[0] = x[0];
  x1[1] = x[1];
  x1[2] = x[2];
  x1[3] = x[3];
  x1[4] = x[4];
  x1[5] = x[5];
  x1[6] = x[6];
  x1[7] = x[7];

  u32 y1[8];

  y1[0] = y[0];
  y1[1] = y[1];
  y1[2] = y[2];
  y1[3] = y[3];
  y1[4] = y[4];
  y1[5] = y[5];
  y1[6] = y[6];
  y1[7] = y[7];

  u32 z1[8] = { 0 };

  z1[0] = 1;

  // r->xy[0..15] = 1*(x, y)

  r->xy[ 0] = x[0];
  r->xy[ 1] = x[1];
  r->xy[ 2] = x[2];
  r->xy[ 3] = x[3];
  r->xy[ 4] = x[4];
  r->xy[ 5] = x[5];
  r->xy[ 6] = x[6];
  r->xy[ 7] = x[7];

  r->xy[ 8] = y[0];
  r->xy[ 9] = y[1];
  r->xy[10] = y[2];
  r->xy[11] = y[3];
  r->xy[12] = y[4];
  r->xy[13] = y[5];
  r->xy[14] = y[6];
  r->xy[15] = y[7];

  // r->xy[16..31] = 2*(x, y)

  point_double (x1, y1, z1);

  // convert to affine coordinates and store it:

  inv_mod (z1);

  mul_mod (z2, z1, z1); // z2 = z1^2
  mul_mod (z1, z1, z2); // z1 = z1^3

  // output to x1

  mul_mod (x1, x1, z2);

  r->xy[16] = x1[0];
  r->xy[17] = x1[1];
  r->xy[18] = x1[2];
  r->xy[19] = x1[3];
  r->xy[20] = x1[4];
  r->xy[21] = x1[5];
  r->xy[22] = x1[6];
  r->xy[23] = x1[7];

  // output to y1

  mul_mod (y1, y1, z1);

  r->xy[24] = y1[0];
  r->xy[25] = y1[1];
  r->xy[26] = y1[2];
  r->xy[27] = y1[3];
  r->xy[28] = y1[4];
  r->xy[29] = y1[5];
  r->xy[30] = y1[6];
  r->xy[31] = y1[7];

  // r->xy[32..47] = 3*(x, y)

  point_add (x1, y1, z1, x, y);

  // convert to affine coordinates and store it:

  inv_mod (z1);

  mul_mod (z2, z1, z1); // z2 = z1^2
  mul_mod (z1, z1, z2); // z1 = z1^3

  // output to x1

  mul_mod (x1, x1, z2);

  r->xy[32] = x1[0];
  r->xy[33] = x1[1];
  r->xy[34] = x1[2];
  r->xy[35] = x1[3];
  r->xy[36] = x1[4];
  r->xy[37] = x1[5];
  r->xy[38] = x1[6];
  r->xy[39] = x1[7];

  // output to y1

  mul_mod (y1, y1, z1);

  r->xy[40] = y1[0];
  r->xy[41] = y1[1];
  r->xy[42] = y1[2];
  r->xy[43] = y1[3];
  r->xy[44] = y1[4];
  r->xy[45] = y1[5];
  r->xy[46] = y1[6];
  r->xy[47] = y1[7];
}

DECLSPEC void point_mul (u32 r[9], const u32 k[8], GLOBAL_AS const secp256k1_t *tmps)
{
  u32 base_x[3][8];

  u32 base_y[3][8];

  base_x[0][0] = tmps->xy[ 0];
  base_x[0][1] = tmps->xy[ 1];
  base_x[0][2] = tmps->xy[ 2];
  base_x[0][3] = tmps->xy[ 3];
  base_x[0][4] = tmps->xy[ 4];
  base_x[0][5] = tmps->xy[ 5];
  base_x[0][6] = tmps->xy[ 6];
  base_x[0][7] = tmps->xy[ 7];

  base_y[0][0] = tmps->xy[ 8];
  base_y[0][1] = tmps->xy[ 9];
  base_y[0][2] = tmps->xy[10];
  base_y[0][3] = tmps->xy[11];
  base_y[0][4] = tmps->xy[12];
  base_y[0][5] = tmps->xy[13];
  base_y[0][6] = tmps->xy[14];
  base_y[0][7] = tmps->xy[15];

  base_x[1][0] = tmps->xy[16];
  base_x[1][1] = tmps->xy[17];
  base_x[1][2] = tmps->xy[18];
  base_x[1][3] = tmps->xy[19];
  base_x[1][4] = tmps->xy[20];
  base_x[1][5] = tmps->xy[21];
  base_x[1][6] = tmps->xy[22];
  base_x[1][7] = tmps->xy[23];

  base_y[1][0] = tmps->xy[24];
  base_y[1][1] = tmps->xy[25];
  base_y[1][2] = tmps->xy[26];
  base_y[1][3] = tmps->xy[27];
  base_y[1][4] = tmps->xy[28];
  base_y[1][5] = tmps->xy[29];
  base_y[1][6] = tmps->xy[30];
  base_y[1][7] = tmps->xy[31];

  base_x[2][0] = tmps->xy[32];
  base_x[2][1] = tmps->xy[33];
  base_x[2][2] = tmps->xy[34];
  base_x[2][3] = tmps->xy[35];
  base_x[2][4] = tmps->xy[36];
  base_x[2][5] = tmps->xy[37];
  base_x[2][6] = tmps->xy[38];
  base_x[2][7] = tmps->xy[39];

  base_y[2][0] = tmps->xy[40];
  base_y[2][1] = tmps->xy[41];
  base_y[2][2] = tmps->xy[42];
  base_y[2][3] = tmps->xy[43];
  base_y[2][4] = tmps->xy[44];
  base_y[2][5] = tmps->xy[45];
  base_y[2][6] = tmps->xy[46];
  base_y[2][7] = tmps->xy[47];

  u32 x[8] = { 0 };

  u32 y[8] = { 0 };

  u32 z[8] = { 0 };

  z[0] = 1;

  int zero = 1;

  // main loop (left-to-right binary algorithm):

  for (int i = 254; i >= 0; i -= 2)
  {
    u32 idx = pos >> 5;

    u32 bits = (k[idx] >> (pos & 0x1f)) & 3;

    if (!zero)
    {
      point_double (x, y, z);
      point_double (x, y, z);
    }

    // add only if needed:

    if (bits)
    {
      bits--;

      if (zero)
      {
        zero = 0;

        x[0] = base_x[bits][0];
        x[1] = base_x[bits][1];
        x[2] = base_x[bits][2];
        x[3] = base_x[bits][3];
        x[4] = base_x[bits][4];
        x[5] = base_x[bits][5];
        x[6] = base_x[bits][6];
        x[7] = base_x[bits][7];

        y[0] = base_y[bits][0];
        y[1] = base_y[bits][1];
        y[2] = base_y[bits][2];
        y[3] = base_y[bits][3];
        y[4] = base_y[bits][4];
        y[5] = base_y[bits][5];
        y[6] = base_y[bits][6];
        y[7] = base_y[bits][7];
      }
      else
      {
        point_add (x, y, z, base_x[bits], base_y[bits]);
      }
    }
  }

  /*
   * Get the corresponding affine coordinates x/y:
   *
   * Note:
   * x1_affine = x1_jacobian / z1^2 = x1_jacobian * z1_inv^2
   * y1_affine = y1_jacobian / z1^2 = y1_jacobian * z1_inv^2
   *
   */

  inv_mod (z);

  u32 t[8];

  mul_mod (t, z, z); // z^2
  mul_mod (x, x, t); // x_affine

  mul_mod (z, t, z); // z^3
  mul_mod (y, y, z); // y_affine

  /*
   * output:
   */

  // shift by 1 byte (8 bits) to make room and add the parity/sign (for odd/even y):

  r[8] =               (x[0] << 24);
  r[7] = (x[0] >> 8) | (x[1] << 24);
  r[6] = (x[1] >> 8) | (x[2] << 24);
  r[5] = (x[2] >> 8) | (x[3] << 24);
  r[4] = (x[3] >> 8) | (x[4] << 24);
  r[3] = (x[4] >> 8) | (x[5] << 24);
  r[2] = (x[5] >> 8) | (x[6] << 24);
  r[1] = (x[6] >> 8) | (x[7] << 24);
  r[0] = (x[7] >> 8);

  const u32 type = 0x02 | (y[0] & 1); // (note: 0b10 | 0b01 = 0x03)

  r[0] = r[0] | type << 24; // 0x02 or 0x03
}

DECLSPEC u32 parse_public (secp256k1_t *r, const u32 k[9])
{
  // verify:

  const u32 first_byte = k[0] & 0xff;

  if ((first_byte != '\x02') && (first_byte != '\x03'))
  {
    return 1;
  }

  // load k into x without the first byte:

  u32 x[8];

  x[0] = (k[7] & 0xff00) << 16 | (k[7] & 0xff0000) | (k[7] & 0xff000000) >> 16 | (k[8] & 0xff);
  x[1] = (k[6] & 0xff00) << 16 | (k[6] & 0xff0000) | (k[6] & 0xff000000) >> 16 | (k[7] & 0xff);
  x[2] = (k[5] & 0xff00) << 16 | (k[5] & 0xff0000) | (k[5] & 0xff000000) >> 16 | (k[6] & 0xff);
  x[3] = (k[4] & 0xff00) << 16 | (k[4] & 0xff0000) | (k[4] & 0xff000000) >> 16 | (k[5] & 0xff);
  x[4] = (k[3] & 0xff00) << 16 | (k[3] & 0xff0000) | (k[3] & 0xff000000) >> 16 | (k[4] & 0xff);
  x[5] = (k[2] & 0xff00) << 16 | (k[2] & 0xff0000) | (k[2] & 0xff000000) >> 16 | (k[3] & 0xff);
  x[6] = (k[1] & 0xff00) << 16 | (k[1] & 0xff0000) | (k[1] & 0xff000000) >> 16 | (k[2] & 0xff);
  x[7] = (k[0] & 0xff00) << 16 | (k[0] & 0xff0000) | (k[0] & 0xff000000) >> 16 | (k[1] & 0xff);

  u32 p[8];

  p[0] = SECP256K1_P0;
  p[1] = SECP256K1_P1;
  p[2] = SECP256K1_P2;
  p[3] = SECP256K1_P3;
  p[4] = SECP256K1_P4;
  p[5] = SECP256K1_P5;
  p[6] = SECP256K1_P6;
  p[7] = SECP256K1_P7;

  // x must be smaller than p (because of y ^ 2 = x ^ 3 % p)

  for (int i = 7; i >= 0; i--)
  {
    if (x[i] < p[i]) break;
    if (x[i] > p[i]) return 1;
  }


  // get y^2 = x^3 + 7:

  u32 b[8] = { 0 };

  b[0] = SECP256K1_B;

  u32 y[8];

  mul_mod (y, x, x);
  mul_mod (y, y, x);
  add_mod (y, y, b);

  // get y = sqrt (y^2):

  sqrt_mod (y);

  // check if it's of the correct parity that we want (odd/even):

  if ((first_byte & 1) != (y[0] & 1))
  {
    // y2 = p - y1 (or y2 = y1 * -1)

    sub_mod (y, p, y);
  }

  // get xy:

  point_get_coords (r, x, y);

  return 0;
}
