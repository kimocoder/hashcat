/**
 * Author......: See docs/credits.txt
 * License.....: MIT
 */

#include "common.h"
#include "types.h"
#include "modules.h"
#include "bitops.h"
#include "convert.h"
#include "shared.h"
#include "filehandling.h"
#include "memory.h"
#include "limits.h"

#define DGST_ELEM 4

#include "emu_general.h"
#include "emu_inc_cipher_aes.h"
#include "emu_inc_hash_md5.h"
#include "m02500-pure.cl"
#include "wpa_common.c"

static const u32   ATTACK_EXEC    = ATTACK_EXEC_OUTSIDE_KERNEL;
static const u32   DGST_POS0      = 0;
static const u32   DGST_POS1      = 1;
static const u32   DGST_POS2      = 2;
static const u32   DGST_POS3      = 3;
static const u32   DGST_SIZE      = DGST_SIZE_4_4;
static const u32   HASH_CATEGORY  = HASH_CATEGORY_NETWORK_PROTOCOL;
static const char *HASH_NAME      = "WPA-EAPOL-PBKDF2";
static const u64   KERN_TYPE      = 2500;
static const u32   OPTI_TYPE      = OPTI_TYPE_ZERO_BYTE
                                  | OPTI_TYPE_SLOW_HASH_SIMD_LOOP;
static const u64   OPTS_TYPE      = OPTS_TYPE_STOCK_MODULE
                                  | OPTS_TYPE_PT_GENERATE_LE
                                  | OPTS_TYPE_BINARY_HASHFILE
                                  | OPTS_TYPE_DEEP_COMP_KERNEL
                                  | OPTS_TYPE_COPY_TMPS;
static const u64   KERN_BITS      = KERN_BIT_AUX1
                                  | KERN_BIT_AUX2
                                  | KERN_BIT_AUX3;
static const u32   SALT_TYPE      = SALT_TYPE_EMBEDDED;
static const char *ST_PASS        = "hashcat!";
static const char *ST_HASH        = "4843505804000000000235380000000000000000000000000000000000000000000000000000000000000151aecc428f182acefbd1a9e62d369a079265784da83b9cced04f552c5b5acfebf06ae96a09c96d9a01c443a17aa62258c4f651a68aa6a1cd402aed44a4cf88375c44c830e6e5aa5d6faf352aa496a9ee129fb8292f7435df5420b8237b0001030077fe010900200000000000000001a4cf88375c44c830e6e5aa5d6faf352aa496a9ee129fb8292f7435df5420b8230000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000018dd160050f20101000050f20201000050f20201000050f20200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000";

#include "hooks_common.c"
// Called above, before its definition further down.
static int         module_hash_decode              (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED       void *digest_buf, MAYBE_UNUSED       salt_t *salt, MAYBE_UNUSED       void *esalt_buf, MAYBE_UNUSED       void *hook_salt_buf, MAYBE_UNUSED       hashinfo_t *hash_info, const char *line_buf, MAYBE_UNUSED const int line_len);

static u64         module_kern_bits      (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const user_options_t *user_options, MAYBE_UNUSED const user_options_extra_t *user_options_extra) { return KERN_BITS;       }

static const u32 ROUNDS_WPA_PBKDF2 = 4096;

/*
typedef struct wpa_eapol
{
  u32  pke[32];
  u32  eapol[64 + 16];
  u16  eapol_len;
  u8   message_pair;
  int  message_pair_chgd;
  u8   keyver;
  u8   orig_mac_ap[6];
  u8   orig_mac_sta[6];
  u8   orig_nonce_ap[32];
  u8   orig_nonce_sta[32];
  u8   essid_len;
  u8   essid[32];
  u32  keymic[4];
  int  nonce_compare;
  int  nonce_error_corrections;
  int  detected_le;
  int  detected_be;

} wpa_eapol_t;

typedef struct wpa_pbkdf2_tmp
{
  u32 ipad[5];
  u32 opad[5];

  u32 dgst[10];
  u32 out[10];

} wpa_pbkdf2_tmp_t;
*/

#include "wpa_hccapx.h"

static void to_hccapx_t (const hashes_t *hashes, hccapx_t *hccapx, const u32 salt_pos, const u32 digest_pos)
{
  const salt_t *salts_buf   = hashes->salts_buf;
  const void   *esalts_buf  = hashes->esalts_buf;

  memset (hccapx, 0, sizeof (hccapx_t));

  hccapx->signature = HCCAPX_SIGNATURE;
  hccapx->version   = HCCAPX_VERSION;

  const salt_t *salt = &salts_buf[salt_pos];

  const u32 digest_cur = salt->digests_offset + digest_pos;

  hccapx->essid_len = salt->salt_len;

  memcpy (hccapx->essid, salt->salt_buf, hccapx->essid_len);

  const wpa_eapol_t *wpa_eapols = (const wpa_eapol_t *) esalts_buf;
  const wpa_eapol_t *wpa_eapol  = &wpa_eapols[digest_cur];

  hccapx->message_pair = wpa_eapol->message_pair;
  hccapx->keyver = wpa_eapol->keyver;

  hccapx->eapol_len = wpa_eapol->eapol_len;

  if (wpa_eapol->keyver != 1)
  {
    u32 eapol_tmp[64] = { 0 };

    for (u32 i = 0; i < 64; i++)
    {
      eapol_tmp[i] = byte_swap_32 (wpa_eapol->eapol[i]);
    }

    memcpy (hccapx->eapol, eapol_tmp, wpa_eapol->eapol_len);
  }
  else
  {
    memcpy (hccapx->eapol, wpa_eapol->eapol, wpa_eapol->eapol_len);
  }

  memcpy (hccapx->mac_ap,    wpa_eapol->orig_mac_ap,    6);
  memcpy (hccapx->mac_sta,   wpa_eapol->orig_mac_sta,   6);
  memcpy (hccapx->nonce_ap,  wpa_eapol->orig_nonce_ap,  32);
  memcpy (hccapx->nonce_sta, wpa_eapol->orig_nonce_sta, 32);

  if (wpa_eapol->keyver != 1)
  {
    u32 digest_tmp[4];

    digest_tmp[0] = byte_swap_32 (wpa_eapol->keymic[0]);
    digest_tmp[1] = byte_swap_32 (wpa_eapol->keymic[1]);
    digest_tmp[2] = byte_swap_32 (wpa_eapol->keymic[2]);
    digest_tmp[3] = byte_swap_32 (wpa_eapol->keymic[3]);

    memcpy (hccapx->keymic, digest_tmp, 16);
  }
  else
  {
    memcpy (hccapx->keymic, wpa_eapol->keymic, 16);
  }
}

/*
static int check_old_hccap (const char *hashfile)
{
  FILE *fp = fopen (hashfile, "rb");

  if (fp == NULL) return -1;

  u32 signature;

  const size_t nread = hc_fread (&signature, sizeof (u32), 1, fp);

  fclose (fp);

  if (nread != 1) return -1;

  if (signature == HCCAPX_SIGNATURE) return 0;

  return 1;
}
*/

static const char *module_benchmark_mask (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const user_options_t *user_options, MAYBE_UNUSED const user_options_extra_t *user_options_extra)
{
  const char *mask = "?a?a?a?a?a?a?a?a";

  return mask;
}

static u64 module_tmp_size (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const user_options_t *user_options, MAYBE_UNUSED const user_options_extra_t *user_options_extra)
{
  const u64 tmp_size = (const u64) sizeof (wpa_pbkdf2_tmp_t);

  return tmp_size;
}

static u64 module_esalt_size (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const user_options_t *user_options, MAYBE_UNUSED const user_options_extra_t *user_options_extra)
{
  const u64 esalt_size = (const u64) sizeof (wpa_eapol_t);

  return esalt_size;
}

static u32 module_pw_min (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const user_options_t *user_options, MAYBE_UNUSED const user_options_extra_t *user_options_extra)
{
  const u32 pw_min = 8;

  return pw_min;
}

static u32 module_pw_max (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const user_options_t *user_options, MAYBE_UNUSED const user_options_extra_t *user_options_extra)
{
  const u32 pw_max = 63;

  return pw_max;
}

static int module_hash_decode_potfile (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED void *digest_buf, MAYBE_UNUSED salt_t *salt, MAYBE_UNUSED void *esalt_buf, MAYBE_UNUSED void *hook_salt_buf, MAYBE_UNUSED hashinfo_t *hash_info, const char *line_buf, MAYBE_UNUSED const int line_len, MAYBE_UNUSED void *tmps)
{
  wpa_eapol_t *wpa_eapol = (wpa_eapol_t *) esalt_buf;

  wpa_pbkdf2_tmp_t *wpa_pbkdf2_tmp = (wpa_pbkdf2_tmp_t *) tmps;

  // here we have in line_hash_buf: PMK*essid:password
  // but we don't care about the password

  // The 8 reads below take a fixed 64 characters out of the line, and the check that the separator
  // sits at offset 64 comes after them. A shorter potfile line is read past its end.

  if (line_len < 64) return (PARSER_HASH_LENGTH);

  // PMK

  wpa_pbkdf2_tmp->out[0] = hex_to_u32 ((const u8 *) line_buf +  0);
  wpa_pbkdf2_tmp->out[1] = hex_to_u32 ((const u8 *) line_buf +  8);
  wpa_pbkdf2_tmp->out[2] = hex_to_u32 ((const u8 *) line_buf + 16);
  wpa_pbkdf2_tmp->out[3] = hex_to_u32 ((const u8 *) line_buf + 24);
  wpa_pbkdf2_tmp->out[4] = hex_to_u32 ((const u8 *) line_buf + 32);
  wpa_pbkdf2_tmp->out[5] = hex_to_u32 ((const u8 *) line_buf + 40);
  wpa_pbkdf2_tmp->out[6] = hex_to_u32 ((const u8 *) line_buf + 48);
  wpa_pbkdf2_tmp->out[7] = hex_to_u32 ((const u8 *) line_buf + 56);

  // essid

  const char *sep_pos = strrchr (line_buf, '*');

  if (sep_pos == NULL) return (PARSER_SEPARATOR_UNMATCHED);

  if ((line_buf + 64) != sep_pos) return (PARSER_HASH_LENGTH);

  const char *essid_pos = sep_pos + 1;

  const int essid_len = strlen (essid_pos);

  if (essid_len & 1) return (PARSER_SALT_VALUE);

  if (essid_len > 64) return (PARSER_SALT_VALUE);

  wpa_eapol->essid_len = hex_decode ((const u8 *) essid_pos, essid_len, (u8 *) wpa_eapol->essid);

  return PARSER_OK;
}

static int module_hash_encode_potfile (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const void *digest_buf, MAYBE_UNUSED const salt_t *salt, MAYBE_UNUSED const void *esalt_buf, MAYBE_UNUSED const void *hook_salt_buf, MAYBE_UNUSED const hashinfo_t *hash_info, char *line_buf, MAYBE_UNUSED const int line_size, MAYBE_UNUSED const void *tmps)
{
  const wpa_eapol_t *wpa_eapol = (const wpa_eapol_t *) esalt_buf;

  const wpa_pbkdf2_tmp_t *wpa_pbkdf2_tmp = (const wpa_pbkdf2_tmp_t *) tmps;

  char tmp_buf[128];

  const int tmp_len = hex_encode ((const u8 *) wpa_eapol->essid, wpa_eapol->essid_len, (u8 *) tmp_buf);

  tmp_buf[tmp_len] = 0;

  const int line_len = snprintf (line_buf, line_size, "%08x%08x%08x%08x%08x%08x%08x%08x*%s",
    wpa_pbkdf2_tmp->out[0],
    wpa_pbkdf2_tmp->out[1],
    wpa_pbkdf2_tmp->out[2],
    wpa_pbkdf2_tmp->out[3],
    wpa_pbkdf2_tmp->out[4],
    wpa_pbkdf2_tmp->out[5],
    wpa_pbkdf2_tmp->out[6],
    wpa_pbkdf2_tmp->out[7],
    tmp_buf);

  return line_len;
}

static int module_hash_encode_status (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const void *digest_buf, MAYBE_UNUSED const salt_t *salt, MAYBE_UNUSED const void *esalt_buf, MAYBE_UNUSED const void *hook_salt_buf, MAYBE_UNUSED const hashinfo_t *hash_info, char *line_buf, MAYBE_UNUSED const int line_size)
{
  const wpa_eapol_t *wpa_eapol = (const wpa_eapol_t *) esalt_buf;

  const int line_len = snprintf (line_buf, line_size, "%s (AP:%02x:%02x:%02x:%02x:%02x:%02x STA:%02x:%02x:%02x:%02x:%02x:%02x)",
    (const char *) salt->salt_buf,
    wpa_eapol->orig_mac_ap[0],
    wpa_eapol->orig_mac_ap[1],
    wpa_eapol->orig_mac_ap[2],
    wpa_eapol->orig_mac_ap[3],
    wpa_eapol->orig_mac_ap[4],
    wpa_eapol->orig_mac_ap[5],
    wpa_eapol->orig_mac_sta[0],
    wpa_eapol->orig_mac_sta[1],
    wpa_eapol->orig_mac_sta[2],
    wpa_eapol->orig_mac_sta[3],
    wpa_eapol->orig_mac_sta[4],
    wpa_eapol->orig_mac_sta[5]);

  return line_len;
}

static int module_hash_init_selftest (MAYBE_UNUSED const hashconfig_t *hashconfig, hash_t *hash)
{
  const size_t st_hash_len = strlen (hashconfig->st_hash);

  char *tmpdata = (char *) hcmalloc (st_hash_len / 2);

  for (size_t i = 0, j = 0; j < st_hash_len; i += 1, j += 2)
  {
    const u8 c = hex_to_u8 ((const u8 *) hashconfig->st_hash + j);

    tmpdata[i] = c;
  }

  const int parser_status = module_hash_decode (hashconfig, hash->digest, hash->salt, hash->esalt, hash->hook_salt, hash->hash_info, tmpdata, st_hash_len / 2);

  wpa_eapol_t *wpa_eapol = (wpa_eapol_t *) hash->esalt;

  wpa_eapol->detected_le = 1;
  wpa_eapol->detected_be = 0;

  wpa_eapol->nonce_error_corrections = 3;

  hcfree (tmpdata);

  return parser_status;
}

static int module_hash_binary_save (MAYBE_UNUSED const hashes_t *hashes, MAYBE_UNUSED const u32 salt_pos, MAYBE_UNUSED const u32 digest_pos, char **buf)
{
  hccapx_t *hccapx = hcmalloc (sizeof (hccapx_t));

  to_hccapx_t (hashes, hccapx, salt_pos, digest_pos);

  *buf = (char *) hccapx;

  return sizeof (hccapx_t);
}

static int module_hash_binary_parse (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const user_options_t *user_options, MAYBE_UNUSED const user_options_extra_t *user_options_extra, hashes_t *hashes)
{
  hash_t *hashes_buf = hashes->hashes_buf;

  int hashes_cnt = 0;

  const int hashes_avail = (int) hashes->hashes_cnt;

  HCFILE fp;

  if (hc_fopen (&fp, hashes->hashfile, "rb") == false) return (PARSER_HAVE_ERRNO);

  char *in = (char *) hcmalloc (sizeof (hccapx_t));

  while (!hc_feof (&fp))
  {
    if (hashes_cnt >= hashes_avail) break;

    const size_t nread = hc_fread (in, sizeof (hccapx_t), 1, &fp);

    if (nread == 0) break;

    const hccapx_t *hccapx = (const hccapx_t *) in;

    memset (hashes_buf[hashes_cnt].salt, 0, sizeof (salt_t));

    memset (hashes_buf[hashes_cnt].esalt, 0, sizeof (wpa_eapol_t));

    wpa_eapol_t *wpa_eapol = (wpa_eapol_t *) hashes_buf[hashes_cnt].esalt;

    if (module_option_given (user_options, OPTION_EAPOL_MESSAGE_PAIR) == true)
    {
      wpa_eapol->message_pair_chgd = (int) module_option_given (user_options, OPTION_EAPOL_MESSAGE_PAIR);
      wpa_eapol->message_pair      = (u8)  module_option_u32 (user_options, OPTION_EAPOL_MESSAGE_PAIR);
    }

    if (hccapx->message_pair & (1 << 4))
    {
      // ap-less attack detected, nc not needed

      wpa_eapol->nonce_error_corrections = 0;
    }
    else
    {
      if (hccapx->message_pair & (1 << 7))
      {
        // replaycount not checked, nc needed

        wpa_eapol->nonce_error_corrections = NONCE_ERROR_CORRECTIONS;

        if (module_option_given (user_options, OPTION_NONCE_ERROR_CORRECTIONS) == true)
        {
          wpa_eapol->nonce_error_corrections = module_option_u32 (user_options, OPTION_NONCE_ERROR_CORRECTIONS);
        }
      }
      else
      {
        // replaycount checked, nc not needed, but we allow user overwrites

        if (module_option_given (user_options, OPTION_NONCE_ERROR_CORRECTIONS) == true)
        {
          wpa_eapol->nonce_error_corrections = module_option_u32 (user_options, OPTION_NONCE_ERROR_CORRECTIONS);
        }
        else
        {
          wpa_eapol->nonce_error_corrections = 0;
        }
      }
    }

    // now some optimization related to replay counter endianess
    // hcxtools has techniques to detect them
    // since we can not guarantee to get our handshakes from hcxtools we enable both by default
    // this means that we check both even if both are not set!
    // however if one of them is set, we can assume that the endianess has been checked and the other one is not needed

    wpa_eapol->detected_le = 1;
    wpa_eapol->detected_be = 1;

    if (hccapx->message_pair & (1 << 5))
    {
      wpa_eapol->detected_le = 1;
      wpa_eapol->detected_be = 0;
    }
    else if (hccapx->message_pair & (1 << 6))
    {
      wpa_eapol->detected_le = 0;
      wpa_eapol->detected_be = 1;
    }

    hash_t *hash = &hashes_buf[hashes_cnt];

    const int parser_status = module_hash_decode (hashconfig, hash->digest, hash->salt, hash->esalt, hash->hook_salt, hash->hash_info, in, sizeof (hccapx_t));

    if (parser_status != PARSER_OK) continue;

    hashes_cnt++;
  }

  hcfree (in);

  hc_fclose (&fp);

  return hashes_cnt;
}

static int module_hash_binary_count (MAYBE_UNUSED const hashes_t *hashes)
{
  // stat () measures the file on disk, and hc_fopen () transparently decompresses gzip, xz and zstd.
  // A compressed hash file therefore holds far more records than its size on disk suggests, and the
  // count decides how many hash entries are allocated for module_hash_binary_parse () to fill.
  // Counting the records the same way the parse reads them keeps the two in step.

  HCFILE fp;

  if (hc_fopen (&fp, hashes->hashfile, "rb") == false) return (PARSER_HAVE_ERRNO);

  char *in = (char *) hcmalloc (sizeof (hccapx_t));

  u64 count = 0;

  while (hc_feof (&fp) == false)
  {
    const size_t nread = hc_fread (in, sizeof (hccapx_t), 1, &fp);

    if (nread == 0) break;

    count++;

    if (count == INT_MAX) break;
  }

  hcfree (in);

  hc_fclose (&fp);

  return (int) count;
}

static u32 module_deep_comp_kernel (MAYBE_UNUSED const hashes_t *hashes, MAYBE_UNUSED const u32 salt_pos, MAYBE_UNUSED const u32 digest_pos)
{
  const u32 digests_offset = hashes->salts_buf[salt_pos].digests_offset;

  wpa_eapol_t *wpa_eapols = (wpa_eapol_t *) hashes->esalts_buf;

  wpa_eapol_t *wpa_eapol = &wpa_eapols[digests_offset + digest_pos];

  if (wpa_eapol->keyver == 1)
  {
    return KERN_RUN_AUX1;
  }
  else if (wpa_eapol->keyver == 2)
  {
    return KERN_RUN_AUX2;
  }
  else if (wpa_eapol->keyver == 3)
  {
    return KERN_RUN_AUX3;
  }

  return 0;
}

static bool module_potfile_custom_check (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const hash_t *db, MAYBE_UNUSED const hash_t *entry_hash, MAYBE_UNUSED const void *entry_tmps)
{
  const wpa_eapol_t *wpa_eapol_entry = (const wpa_eapol_t *) entry_hash->esalt;
  const wpa_eapol_t *wpa_eapol_db    = (const wpa_eapol_t *) db->esalt;

  if (wpa_eapol_db->essid_len != wpa_eapol_entry->essid_len) return false;

  if (memcmp (wpa_eapol_db->essid, wpa_eapol_entry->essid, wpa_eapol_db->essid_len) != 0) return false;

  const wpa_pbkdf2_tmp_t *wpa_pbkdf2_tmp = (const wpa_pbkdf2_tmp_t *) entry_tmps;

  wpa_pbkdf2_tmp_t tmps;

  tmps.out[0] = byte_swap_32 (wpa_pbkdf2_tmp->out[0]);
  tmps.out[1] = byte_swap_32 (wpa_pbkdf2_tmp->out[1]);
  tmps.out[2] = byte_swap_32 (wpa_pbkdf2_tmp->out[2]);
  tmps.out[3] = byte_swap_32 (wpa_pbkdf2_tmp->out[3]);
  tmps.out[4] = byte_swap_32 (wpa_pbkdf2_tmp->out[4]);
  tmps.out[5] = byte_swap_32 (wpa_pbkdf2_tmp->out[5]);
  tmps.out[6] = byte_swap_32 (wpa_pbkdf2_tmp->out[6]);
  tmps.out[7] = byte_swap_32 (wpa_pbkdf2_tmp->out[7]);

  plain_t plains_buf;

  u32 hashes_shown = 0;

  u32 d_return_buf = 0;

  void (*m02500_aux) (KERN_ATTR_TMPS_ESALT (wpa_pbkdf2_tmp_t, wpa_eapol_t));

  if (wpa_eapol_db->keyver == 1)
  {
    m02500_aux = m02500_aux1;
  }
  else if (wpa_eapol_db->keyver == 2)
  {
    m02500_aux = m02500_aux2;
  }
  else if (wpa_eapol_db->keyver == 3)
  {
    m02500_aux = m02500_aux3;
  }
  else
  {
    return false;
  }

  kernel_param_t kernel_param;

  kernel_param.bitmap_mask         = 0;
  kernel_param.salt_pos_host       = 0;
  kernel_param.loop_pos            = 0;
  kernel_param.loop_cnt            = 0;
  kernel_param.il_cnt              = 0;
  kernel_param.digests_cnt         = 1;
  kernel_param.digests_offset_host = 0;
  kernel_param.combs_mode          = 0;
  kernel_param.salt_repeat         = 0;
  kernel_param.pws_pos             = 0;
  kernel_param.gid_max             = 1;

  m02500_aux
  (
    NULL,               // pws
    NULL,               // rules_buf
    NULL,               // combs_buf
    NULL,               // bfs_buf
    &tmps,              // tmps
    NULL,               // hooks
    NULL,               // bitmaps_buf_s1_a
    NULL,               // bitmaps_buf_s1_b
    NULL,               // bitmaps_buf_s1_c
    NULL,               // bitmaps_buf_s1_d
    NULL,               // bitmaps_buf_s2_a
    NULL,               // bitmaps_buf_s2_b
    NULL,               // bitmaps_buf_s2_c
    NULL,               // bitmaps_buf_s2_d
    &plains_buf,        // plains_buf
    db->digest,         // digests_buf
    &hashes_shown,      // hashes_shown
    db->salt,           // salt_bufs
    db->esalt,          // esalt_bufs
    &d_return_buf,      // d_return_buf
    NULL,               // d_extra0_buf
    NULL,               // d_extra1_buf
    NULL,               // d_extra2_buf
    NULL,               // d_extra3_buf
    &kernel_param       // kernel_param
  );

  const bool r = (d_return_buf == 0) ? false : true;

  return r;
}

#define WPA_SALT_ITER       (ROUNDS_WPA_PBKDF2 - 1)
#define WPA_KEYVER_VALID(k) (((k) == 1) || ((k) == 2) || ((k) == 3))
#define WPA_KEYVER_CMAC(k)  ((k) == 3)

#include "wpa_decode_hccapx.h"

static int module_hash_encode (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const void *digest_buf, MAYBE_UNUSED const salt_t *salt, MAYBE_UNUSED const void *esalt_buf, MAYBE_UNUSED const void *hook_salt_buf, MAYBE_UNUSED const hashinfo_t *hash_info, char *line_buf, MAYBE_UNUSED const int line_size)
{
  const wpa_eapol_t *wpa_eapol = (const wpa_eapol_t *) esalt_buf;

  int line_len = 0;

  if (need_hexify (wpa_eapol->essid, wpa_eapol->essid_len, ':', 0) == true)
  {
    char tmp_buf[128];

    int tmp_len = 0;

    tmp_buf[tmp_len++] = '$';
    tmp_buf[tmp_len++] = 'H';
    tmp_buf[tmp_len++] = 'E';
    tmp_buf[tmp_len++] = 'X';
    tmp_buf[tmp_len++] = '[';

    const size_t hex_len = exec_hexify (wpa_eapol->essid, wpa_eapol->essid_len, (u8 *) tmp_buf + tmp_len);

    tmp_len += (int) hex_len;

    tmp_buf[tmp_len++] = ']';

    tmp_buf[tmp_len++] = 0;

    line_len = snprintf (line_buf, line_size, "%02x%02x%02x%02x%02x%02x:%02x%02x%02x%02x%02x%02x:%s",
      wpa_eapol->orig_mac_ap[0],
      wpa_eapol->orig_mac_ap[1],
      wpa_eapol->orig_mac_ap[2],
      wpa_eapol->orig_mac_ap[3],
      wpa_eapol->orig_mac_ap[4],
      wpa_eapol->orig_mac_ap[5],
      wpa_eapol->orig_mac_sta[0],
      wpa_eapol->orig_mac_sta[1],
      wpa_eapol->orig_mac_sta[2],
      wpa_eapol->orig_mac_sta[3],
      wpa_eapol->orig_mac_sta[4],
      wpa_eapol->orig_mac_sta[5],
      tmp_buf);
  }
  else
  {
    line_len = snprintf (line_buf, line_size, "%02x%02x%02x%02x%02x%02x:%02x%02x%02x%02x%02x%02x:%s",
      wpa_eapol->orig_mac_ap[0],
      wpa_eapol->orig_mac_ap[1],
      wpa_eapol->orig_mac_ap[2],
      wpa_eapol->orig_mac_ap[3],
      wpa_eapol->orig_mac_ap[4],
      wpa_eapol->orig_mac_ap[5],
      wpa_eapol->orig_mac_sta[0],
      wpa_eapol->orig_mac_sta[1],
      wpa_eapol->orig_mac_sta[2],
      wpa_eapol->orig_mac_sta[3],
      wpa_eapol->orig_mac_sta[4],
      wpa_eapol->orig_mac_sta[5],
      wpa_eapol->essid);
  }

  return line_len;
}

static const char *module_deprecated_notice (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const user_options_t *user_options, MAYBE_UNUSED const user_options_extra_t *user_options_extra)
{
  const char *deprecated_notice = "The plugin 2500 is deprecated and was replaced with plugin 22000. For more details, please read: https://hashcat.net/forum/thread-10253.html";

  return deprecated_notice;
}

void module_init (module_ctx_t *module_ctx)
{
  module_init_common (module_ctx);

  module_ctx->module_benchmark_mask           = module_benchmark_mask;
  module_ctx->module_deep_comp_kernel         = module_deep_comp_kernel;
  module_ctx->module_deprecated_notice        = module_deprecated_notice;
  module_ctx->module_esalt_size               = module_esalt_size;
  module_ctx->module_hash_binary_count        = module_hash_binary_count;
  module_ctx->module_hash_binary_parse        = module_hash_binary_parse;
  module_ctx->module_hash_binary_save         = module_hash_binary_save;
  module_ctx->module_hash_decode_potfile      = module_hash_decode_potfile;
  module_ctx->module_hash_decode              = module_hash_decode;
  module_ctx->module_hash_encode_status       = module_hash_encode_status;
  module_ctx->module_hash_encode_potfile      = module_hash_encode_potfile;
  module_ctx->module_hash_encode              = module_hash_encode;
  module_ctx->module_hash_init_selftest       = module_hash_init_selftest;
  module_ctx->module_kern_bits                = module_kern_bits;
  module_ctx->module_potfile_custom_check     = module_potfile_custom_check;
  module_ctx->module_pw_max                   = module_pw_max;
  module_ctx->module_pw_min                   = module_pw_min;
  module_ctx->module_register_options         = wpa_module_register_options;
  module_ctx->module_tmp_size                 = module_tmp_size;
}
