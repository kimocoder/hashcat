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
#include "parser.h"
#include "memory.h"

#define DGST_ELEM 4

#include "emu_general.h"
#include "m16801-pure.cl"

static const u32   ATTACK_EXEC    = ATTACK_EXEC_OUTSIDE_KERNEL;
static const u32   DGST_POS0      = 0;
static const u32   DGST_POS1      = 1;
static const u32   DGST_POS2      = 2;
static const u32   DGST_POS3      = 3;
static const u32   DGST_SIZE      = DGST_SIZE_4_4;
static const u32   HASH_CATEGORY  = HASH_CATEGORY_NETWORK_PROTOCOL;
static const char *HASH_NAME      = "WPA-PMKID-PMK";
static const u64   KERN_TYPE      = 16801;
static const u32   OPTI_TYPE      = OPTI_TYPE_ZERO_BYTE
                                  | OPTI_TYPE_SLOW_HASH_SIMD_LOOP;
static const u64   OPTS_TYPE      = OPTS_TYPE_STOCK_MODULE
                                  | OPTS_TYPE_PT_GENERATE_LE
                                  | OPTS_TYPE_DEEP_COMP_KERNEL
                                  | OPTS_TYPE_POTFILE_NOPASS
                                  | OPTS_TYPE_COPY_TMPS
                                  | OPTS_TYPE_AUTODETECT_DISABLE;
static const u64   KERN_BITS      = KERN_BIT_AUX1;
static const u32   SALT_TYPE      = SALT_TYPE_EMBEDDED;
static const char *ST_PASS        = "5b13d4babb3714ccc62c9f71864bc984efd6a55f237c7a87fc2151e1ca658a9d";
static const char *ST_HASH        = "2582a8281bf9d4308d6f5731d0e61c61:4604ba734d4e:89acf0e761f4";

#include "hooks_common.c"

static u64         module_kern_bits      (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const user_options_t *user_options, MAYBE_UNUSED const user_options_extra_t *user_options_extra) { return KERN_BITS;       }

static const u32 ROUNDS_WPA_PMK = 1;

/*
typedef struct wpa_pmkid
{
  u32  pmkid[4];
  u32  pmkid_data[16];
  u8   orig_mac_ap[6];
  u8   orig_mac_sta[6];
  u8   essid_len;
  u32  essid_buf[16];

} wpa_pmkid_t;

typedef struct wpa_pmk_tmp
{
  u32 out[8];

} wpa_pmk_tmp_t;
*/

static const char *module_benchmark_mask (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const user_options_t *user_options, MAYBE_UNUSED const user_options_extra_t *user_options_extra)
{
  const char *mask = "?a?a?a?a?a?a?a?axxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";

  return mask;
}

static u64 module_tmp_size (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const user_options_t *user_options, MAYBE_UNUSED const user_options_extra_t *user_options_extra)
{
  const u64 tmp_size = (const u64) sizeof (wpa_pmk_tmp_t);

  return tmp_size;
}

static u64 module_esalt_size (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const user_options_t *user_options, MAYBE_UNUSED const user_options_extra_t *user_options_extra)
{
  const u64 esalt_size = (const u64) sizeof (wpa_pmkid_t);

  return esalt_size;
}

static u32 module_pw_min (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const user_options_t *user_options, MAYBE_UNUSED const user_options_extra_t *user_options_extra)
{
  const u32 pw_min = 64;

  return pw_min;
}

static u32 module_pw_max (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const user_options_t *user_options, MAYBE_UNUSED const user_options_extra_t *user_options_extra)
{
  const u32 pw_max = 64;

  return pw_max;
}

static int module_hash_decode_potfile (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED void *digest_buf, MAYBE_UNUSED salt_t *salt, MAYBE_UNUSED void *esalt_buf, MAYBE_UNUSED void *hook_salt_buf, MAYBE_UNUSED hashinfo_t *hash_info, const char *line_buf, MAYBE_UNUSED const int line_len, MAYBE_UNUSED void *tmps)
{
  wpa_pmkid_t *wpa_pmkid = (wpa_pmkid_t *) esalt_buf;

  wpa_pmk_tmp_t *wpa_pmk_tmp = (wpa_pmk_tmp_t *) tmps;

  // here we have in line_hash_buf: PMK*essid:password
  // but we don't care about the password

  // The 8 reads below take a fixed 64 characters out of the line, and the check that the separator
  // sits at offset 64 comes after them. A shorter potfile line is read past its end.

  if (line_len < 64) return (PARSER_HASH_LENGTH);

  // PMK

  wpa_pmk_tmp->out[0] = hex_to_u32 ((const u8 *) line_buf +  0);
  wpa_pmk_tmp->out[1] = hex_to_u32 ((const u8 *) line_buf +  8);
  wpa_pmk_tmp->out[2] = hex_to_u32 ((const u8 *) line_buf + 16);
  wpa_pmk_tmp->out[3] = hex_to_u32 ((const u8 *) line_buf + 24);
  wpa_pmk_tmp->out[4] = hex_to_u32 ((const u8 *) line_buf + 32);
  wpa_pmk_tmp->out[5] = hex_to_u32 ((const u8 *) line_buf + 40);
  wpa_pmk_tmp->out[6] = hex_to_u32 ((const u8 *) line_buf + 48);
  wpa_pmk_tmp->out[7] = hex_to_u32 ((const u8 *) line_buf + 56);

  // essid

  const char *sep_pos = strrchr (line_buf, '*');

  if (sep_pos == NULL) return (PARSER_SEPARATOR_UNMATCHED);

  if ((line_buf + 64) != sep_pos) return (PARSER_HASH_LENGTH);

  const char *essid_pos = sep_pos + 1;

  const int essid_len = strlen (essid_pos);

  if (essid_len & 1) return (PARSER_SALT_VALUE);

  if (essid_len > 64) return (PARSER_SALT_VALUE);

  wpa_pmkid->essid_len = hex_decode ((const u8 *) essid_pos, essid_len, (u8 *) wpa_pmkid->essid_buf);

  return PARSER_OK;
}

static int module_hash_encode_potfile (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const void *digest_buf, MAYBE_UNUSED const salt_t *salt, MAYBE_UNUSED const void *esalt_buf, MAYBE_UNUSED const void *hook_salt_buf, MAYBE_UNUSED const hashinfo_t *hash_info, char *line_buf, MAYBE_UNUSED const int line_size, MAYBE_UNUSED const void *tmps)
{
  const wpa_pmkid_t *wpa_pmkid = (const wpa_pmkid_t *) esalt_buf;

  const wpa_pmk_tmp_t *wpa_pmk_tmp = (const wpa_pmk_tmp_t *) tmps;

  char tmp_buf[128];

  const int tmp_len = hex_encode ((const u8 *) wpa_pmkid->essid_buf, wpa_pmkid->essid_len, (u8 *) tmp_buf);

  tmp_buf[tmp_len] = 0;

  const int line_len = snprintf (line_buf, line_size, "%08x%08x%08x%08x%08x%08x%08x%08x*%s",
    wpa_pmk_tmp->out[0],
    wpa_pmk_tmp->out[1],
    wpa_pmk_tmp->out[2],
    wpa_pmk_tmp->out[3],
    wpa_pmk_tmp->out[4],
    wpa_pmk_tmp->out[5],
    wpa_pmk_tmp->out[6],
    wpa_pmk_tmp->out[7],
    tmp_buf);

  return line_len;
}

static int module_hash_binary_save (MAYBE_UNUSED const hashes_t *hashes, MAYBE_UNUSED const u32 salt_pos, MAYBE_UNUSED const u32 digest_pos, char **buf)
{
  const salt_t *salts_buf   = hashes->salts_buf;
  const void   *esalts_buf  = hashes->esalts_buf;

  const salt_t *salt = &salts_buf[salt_pos];

  const u32 digest_cur = salt->digests_offset + digest_pos;

  const wpa_pmkid_t *wpa_pmkids = (const wpa_pmkid_t *) esalts_buf;
  const wpa_pmkid_t *wpa_pmkid  = &wpa_pmkids[digest_cur];

  int len = 0;

  if (wpa_pmkid->essid_len)
  {
    char tmp_buf[128];

    const int tmp_len = hex_encode ((const u8 *) wpa_pmkid->essid_buf, wpa_pmkid->essid_len, (u8 *) tmp_buf);

    tmp_buf[tmp_len] = 0;

    len = hc_asprintf (buf, "%08x%08x%08x%08x:%02x%02x%02x%02x%02x%02x:%02x%02x%02x%02x%02x%02x:%s" EOL,
      byte_swap_32 (wpa_pmkid->pmkid[0]),
      byte_swap_32 (wpa_pmkid->pmkid[1]),
      byte_swap_32 (wpa_pmkid->pmkid[2]),
      byte_swap_32 (wpa_pmkid->pmkid[3]),
      wpa_pmkid->orig_mac_ap[0],
      wpa_pmkid->orig_mac_ap[1],
      wpa_pmkid->orig_mac_ap[2],
      wpa_pmkid->orig_mac_ap[3],
      wpa_pmkid->orig_mac_ap[4],
      wpa_pmkid->orig_mac_ap[5],
      wpa_pmkid->orig_mac_sta[0],
      wpa_pmkid->orig_mac_sta[1],
      wpa_pmkid->orig_mac_sta[2],
      wpa_pmkid->orig_mac_sta[3],
      wpa_pmkid->orig_mac_sta[4],
      wpa_pmkid->orig_mac_sta[5],
      tmp_buf);
  }
  else
  {
    len = hc_asprintf (buf, "%08x%08x%08x%08x:%02x%02x%02x%02x%02x%02x:%02x%02x%02x%02x%02x%02x" EOL,
      byte_swap_32 (wpa_pmkid->pmkid[0]),
      byte_swap_32 (wpa_pmkid->pmkid[1]),
      byte_swap_32 (wpa_pmkid->pmkid[2]),
      byte_swap_32 (wpa_pmkid->pmkid[3]),
      wpa_pmkid->orig_mac_ap[0],
      wpa_pmkid->orig_mac_ap[1],
      wpa_pmkid->orig_mac_ap[2],
      wpa_pmkid->orig_mac_ap[3],
      wpa_pmkid->orig_mac_ap[4],
      wpa_pmkid->orig_mac_ap[5],
      wpa_pmkid->orig_mac_sta[0],
      wpa_pmkid->orig_mac_sta[1],
      wpa_pmkid->orig_mac_sta[2],
      wpa_pmkid->orig_mac_sta[3],
      wpa_pmkid->orig_mac_sta[4],
      wpa_pmkid->orig_mac_sta[5]);
  }

  return len;
}

static u32 module_deep_comp_kernel (MAYBE_UNUSED const hashes_t *hashes, MAYBE_UNUSED const u32 salt_pos, MAYBE_UNUSED const u32 digest_pos)
{
  return KERN_RUN_AUX1;
}

static bool module_potfile_custom_check (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const hash_t *db, MAYBE_UNUSED const hash_t *entry_hash, MAYBE_UNUSED const void *entry_tmps)
{
  const wpa_pmkid_t *wpa_pmkid_entry = (const wpa_pmkid_t *) entry_hash->esalt;
  const wpa_pmkid_t *wpa_pmkid_db    = (const wpa_pmkid_t *) db->esalt;

  if (wpa_pmkid_db->essid_len != wpa_pmkid_entry->essid_len) return false;

  if (strcmp ((const char *) wpa_pmkid_db->essid_buf, (const char *) wpa_pmkid_entry->essid_buf)) return false;

  const wpa_pmk_tmp_t *wpa_pmk_tmp = (const wpa_pmk_tmp_t *) entry_tmps;

  wpa_pmk_tmp_t tmps;

  tmps.out[0] = byte_swap_32 (wpa_pmk_tmp->out[0]);
  tmps.out[1] = byte_swap_32 (wpa_pmk_tmp->out[1]);
  tmps.out[2] = byte_swap_32 (wpa_pmk_tmp->out[2]);
  tmps.out[3] = byte_swap_32 (wpa_pmk_tmp->out[3]);
  tmps.out[4] = byte_swap_32 (wpa_pmk_tmp->out[4]);
  tmps.out[5] = byte_swap_32 (wpa_pmk_tmp->out[5]);
  tmps.out[6] = byte_swap_32 (wpa_pmk_tmp->out[6]);
  tmps.out[7] = byte_swap_32 (wpa_pmk_tmp->out[7]);

  plain_t plains_buf;

  u32 hashes_shown = 0;

  u32 d_return_buf = 0;

  void (*m16801_aux) (KERN_ATTR_TMPS_ESALT (wpa_pmk_tmp_t, wpa_pmkid_t));

  m16801_aux = m16801_aux1;

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

  m16801_aux
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

#define WPA_PMKID_PMK_MODE 1
#define WPA_SALT_ITER (ROUNDS_WPA_PMK - 1)

#include "wpa_decode_pmkid.h"

static int module_hash_encode (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const void *digest_buf, MAYBE_UNUSED const salt_t *salt, MAYBE_UNUSED const void *esalt_buf, MAYBE_UNUSED const void *hook_salt_buf, MAYBE_UNUSED const hashinfo_t *hash_info, char *line_buf, MAYBE_UNUSED const int line_size)
{
  const wpa_pmkid_t *wpa_pmkid = (const wpa_pmkid_t *) esalt_buf;

  int line_len = 0;

  if (wpa_pmkid->essid_len)
  {
    if (need_hexify ((const u8 *) wpa_pmkid->essid_buf, wpa_pmkid->essid_len, ':', 0) == true)
    {
      char tmp_buf[128];

      int tmp_len = 0;

      tmp_buf[tmp_len++] = '$';
      tmp_buf[tmp_len++] = 'H';
      tmp_buf[tmp_len++] = 'E';
      tmp_buf[tmp_len++] = 'X';
      tmp_buf[tmp_len++] = '[';

      const size_t hex_len = exec_hexify ((const u8 *) wpa_pmkid->essid_buf, wpa_pmkid->essid_len, (u8 *) tmp_buf + tmp_len);

      tmp_len += (int) hex_len;

      tmp_buf[tmp_len++] = ']';

      tmp_buf[tmp_len++] = 0;

      line_len = snprintf (line_buf, line_size, "%02x%02x%02x%02x%02x%02x:%02x%02x%02x%02x%02x%02x:%s",
        wpa_pmkid->orig_mac_ap[0],
        wpa_pmkid->orig_mac_ap[1],
        wpa_pmkid->orig_mac_ap[2],
        wpa_pmkid->orig_mac_ap[3],
        wpa_pmkid->orig_mac_ap[4],
        wpa_pmkid->orig_mac_ap[5],
        wpa_pmkid->orig_mac_sta[0],
        wpa_pmkid->orig_mac_sta[1],
        wpa_pmkid->orig_mac_sta[2],
        wpa_pmkid->orig_mac_sta[3],
        wpa_pmkid->orig_mac_sta[4],
        wpa_pmkid->orig_mac_sta[5],
        tmp_buf);
    }
    else
    {
      line_len = snprintf (line_buf, line_size, "%02x%02x%02x%02x%02x%02x:%02x%02x%02x%02x%02x%02x:%s",
        wpa_pmkid->orig_mac_ap[0],
        wpa_pmkid->orig_mac_ap[1],
        wpa_pmkid->orig_mac_ap[2],
        wpa_pmkid->orig_mac_ap[3],
        wpa_pmkid->orig_mac_ap[4],
        wpa_pmkid->orig_mac_ap[5],
        wpa_pmkid->orig_mac_sta[0],
        wpa_pmkid->orig_mac_sta[1],
        wpa_pmkid->orig_mac_sta[2],
        wpa_pmkid->orig_mac_sta[3],
        wpa_pmkid->orig_mac_sta[4],
        wpa_pmkid->orig_mac_sta[5],
        (const char *) wpa_pmkid->essid_buf);
    }
  }
  else
  {
    line_len = snprintf (line_buf, line_size, "%02x%02x%02x%02x%02x%02x:%02x%02x%02x%02x%02x%02x",
      wpa_pmkid->orig_mac_ap[0],
      wpa_pmkid->orig_mac_ap[1],
      wpa_pmkid->orig_mac_ap[2],
      wpa_pmkid->orig_mac_ap[3],
      wpa_pmkid->orig_mac_ap[4],
      wpa_pmkid->orig_mac_ap[5],
      wpa_pmkid->orig_mac_sta[0],
      wpa_pmkid->orig_mac_sta[1],
      wpa_pmkid->orig_mac_sta[2],
      wpa_pmkid->orig_mac_sta[3],
      wpa_pmkid->orig_mac_sta[4],
      wpa_pmkid->orig_mac_sta[5]);
  }

  return line_len;
}

static const char *module_deprecated_notice (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const user_options_t *user_options, MAYBE_UNUSED const user_options_extra_t *user_options_extra)
{
  const char *deprecated_notice = "The plugin 16801 is deprecated and was replaced with plugin 22001. For more details, please read: https://hashcat.net/forum/thread-10253.html";

  return deprecated_notice;
}

void module_init (module_ctx_t *module_ctx)
{
  module_init_common (module_ctx);

  module_ctx->module_benchmark_mask           = module_benchmark_mask;
  module_ctx->module_deep_comp_kernel         = module_deep_comp_kernel;
  module_ctx->module_deprecated_notice        = module_deprecated_notice;
  module_ctx->module_esalt_size               = module_esalt_size;
  module_ctx->module_hash_binary_save         = module_hash_binary_save;
  module_ctx->module_hash_decode_potfile      = module_hash_decode_potfile;
  module_ctx->module_hash_decode              = module_hash_decode;
  module_ctx->module_hash_encode_potfile      = module_hash_encode_potfile;
  module_ctx->module_hash_encode              = module_hash_encode;
  module_ctx->module_kern_bits                = module_kern_bits;
  module_ctx->module_potfile_custom_check     = module_potfile_custom_check;
  module_ctx->module_pw_max                   = module_pw_max;
  module_ctx->module_pw_min                   = module_pw_min;
  module_ctx->module_tmp_size                 = module_tmp_size;
}
