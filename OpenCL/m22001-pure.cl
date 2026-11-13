/**
 * Author......: See docs/credits.txt
 * License.....: MIT
 */

#define NEW_SIMD_CODE

#ifdef KERNEL_STATIC
#include M2S(INCLUDE_PATH/inc_vendor.h)
#include M2S(INCLUDE_PATH/inc_types.h)
#include M2S(INCLUDE_PATH/inc_platform.cl)
#include M2S(INCLUDE_PATH/inc_common.cl)
#include M2S(INCLUDE_PATH/inc_simd.cl)
#include M2S(INCLUDE_PATH/inc_hash_md5.cl)
#include M2S(INCLUDE_PATH/inc_hash_sha1.cl)
#include M2S(INCLUDE_PATH/inc_hash_sha256.cl)
#include M2S(INCLUDE_PATH/inc_cipher_aes.cl)
#else
#include "inc_vendor.h"
#include "inc_types.h"
#include "inc_platform.h"
#include "inc_common.h"
#include "inc_simd.h"
#include "inc_hash_md5.h"
#include "inc_hash_sha1.h"
#include "inc_hash_sha256.h"
#include "inc_cipher_aes.h"
#endif

// The host compiles this file into the module that wants its helpers. There DECLSPEC says the
// function is part of what the core offers a plugin, and these are not: they are the module's own
// copy of a kernel and they stay inside it. inc_rp_common.cl says the same thing the same way.
//
// A module takes the helpers it wants and leaves the rest, so the ones it left have to be allowed to
// go unused.

#ifdef IS_NATIVE
#undef DECLSPEC
#define DECLSPEC static MAYBE_UNUSED
#endif

#define COMPARE_S M2S(INCLUDE_PATH/inc_comp_single.cl)
#define COMPARE_M M2S(INCLUDE_PATH/inc_comp_multi.cl)

typedef struct wpa_pmk_tmp
{
  u32 out[8];

} wpa_pmk_tmp_t;

#ifdef KERNEL_STATIC
#include M2S(INCLUDE_PATH/inc_wpa_common.cl)
#else
#include "inc_wpa_common.cl"
#endif

#ifdef KERNEL_STATIC
DECLSPEC u8 hex_convert (const u8 c)
{
  return (c & 15) + (c >> 6) * 9;
}

DECLSPEC u8 hex_to_u8 (PRIVATE_AS const u8 *hex)
{
  u8 v = 0;

  v |= ((u8) hex_convert (hex[1]) << 0);
  v |= ((u8) hex_convert (hex[0]) << 4);

  return (v);
}
#endif

KERNEL_FQ KERNEL_FA void m22001_init (KERN_ATTR_TMPS_ESALT (wpa_pmk_tmp_t, wpa_t))
{
  const u64 gid = get_global_id (0);

  if (gid >= GID_CNT) return;

  u32 in[16];

  in[ 0] = pws[gid].i[ 0];
  in[ 1] = pws[gid].i[ 1];
  in[ 2] = pws[gid].i[ 2];
  in[ 3] = pws[gid].i[ 3];
  in[ 4] = pws[gid].i[ 4];
  in[ 5] = pws[gid].i[ 5];
  in[ 6] = pws[gid].i[ 6];
  in[ 7] = pws[gid].i[ 7];
  in[ 8] = pws[gid].i[ 8];
  in[ 9] = pws[gid].i[ 9];
  in[10] = pws[gid].i[10];
  in[11] = pws[gid].i[11];
  in[12] = pws[gid].i[12];
  in[13] = pws[gid].i[13];
  in[14] = pws[gid].i[14];
  in[15] = pws[gid].i[15];

  u32 out[8];

  PRIVATE_AS u8 *in_ptr  = (PRIVATE_AS u8 *) in;
  PRIVATE_AS u8 *out_ptr = (PRIVATE_AS u8 *) out;

  for (int i = 0, j = 0; i < 32; i += 1, j += 2)
  {
    out_ptr[i] = hex_to_u8 (in_ptr + j);
  }

  tmps[gid].out[0] = hc_swap32_S (out[0]);
  tmps[gid].out[1] = hc_swap32_S (out[1]);
  tmps[gid].out[2] = hc_swap32_S (out[2]);
  tmps[gid].out[3] = hc_swap32_S (out[3]);
  tmps[gid].out[4] = hc_swap32_S (out[4]);
  tmps[gid].out[5] = hc_swap32_S (out[5]);
  tmps[gid].out[6] = hc_swap32_S (out[6]);
  tmps[gid].out[7] = hc_swap32_S (out[7]);
}

KERNEL_FQ KERNEL_FA void m22001_loop (KERN_ATTR_TMPS_ESALT (wpa_pmk_tmp_t, wpa_t))
{
  // not in use here, special case...
}

KERNEL_FQ KERNEL_FA void m22001_comp (KERN_ATTR_TMPS_ESALT (wpa_pmk_tmp_t, wpa_t))
{
  // not in use here, special case...
}

#define WPA_TMPS_T wpa_pmk_tmp_t

#define WPA_AUX1_NAME m22001_aux1
#define WPA_AUX2_NAME m22001_aux2
#define WPA_AUX3_NAME m22001_aux3
#define WPA_AUX4_NAME m22001_aux4
#define WPA_AUX5_NAME m22001_aux5
#define WPA_AUX6_NAME m22001_aux6

#ifdef KERNEL_STATIC
#include M2S(INCLUDE_PATH/inc_wpa_aux.cl)
#else
#include "inc_wpa_aux.cl"
#endif

#undef WPA_AUX1_NAME
#undef WPA_AUX2_NAME
#undef WPA_AUX3_NAME
#undef WPA_AUX4_NAME
#undef WPA_AUX5_NAME
#undef WPA_AUX6_NAME
#undef WPA_TMPS_T
