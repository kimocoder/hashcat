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
#include "parser.h"
#include "memory.h"
#include "limits.h"

#define DGST_ELEM 4

#include "emu_general.h"
#include "emu_inc_cipher_aes.h"
#include "emu_inc_hash_md5.h"
#include "m22001-pure.cl"
#include "wpa_common.c"

static const u32   ATTACK_EXEC    = ATTACK_EXEC_OUTSIDE_KERNEL;
static const u32   DGST_POS0      = 0;
static const u32   DGST_POS1      = 1;
static const u32   DGST_POS2      = 2;
static const u32   DGST_POS3      = 3;
static const u32   DGST_SIZE      = DGST_SIZE_4_4;
static const u32   HASH_CATEGORY  = HASH_CATEGORY_NETWORK_PROTOCOL;
static const char *HASH_NAME      = "WPA-PMK-PMKID+EAPOL";
static const u64   KERN_TYPE      = 22001;
static const u32   OPTI_TYPE      = OPTI_TYPE_ZERO_BYTE
                                  | OPTI_TYPE_SLOW_HASH_SIMD_LOOP;
static const u64   OPTS_TYPE      = OPTS_TYPE_STOCK_MODULE
                                  | OPTS_TYPE_PT_GENERATE_LE
                                  | OPTS_TYPE_PT_ALWAYS_ASCII
                                  | OPTS_TYPE_BINARY_HASHFILE
                                  | OPTS_TYPE_BINARY_HASHFILE_OPTIONAL
                                  | OPTS_TYPE_DEEP_COMP_KERNEL
                                  | OPTS_TYPE_HASH_COPY
                                  | OPTS_TYPE_COPY_TMPS
                                  | OPTS_TYPE_POTFILE_NOPASS
                                  | OPTS_TYPE_AUTODETECT_DISABLE;
static const u64   KERN_BITS      = KERN_BIT_AUX1
                                  | KERN_BIT_AUX2
                                  | KERN_BIT_AUX3
                                  | KERN_BIT_AUX4
                                  | KERN_BIT_AUX5
                                  | KERN_BIT_AUX6;
static const u32   SALT_TYPE      = SALT_TYPE_EMBEDDED;
static const char *ST_PASS        = "88f43854ae7b1624fc2ab7724859e795130f4843c7535729e819cf92f39535dc";
static const char *ST_HASH        = "WPA*01*5ce7ebe97a1bbfeb2822ae627b726d5b*27462da350ac*accd10fb464e*686173686361742d6573736964***";

#include "hooks_common.c"
// Called above, before its definition further down.
static int         module_hash_decode              (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED       void *digest_buf, MAYBE_UNUSED       salt_t *salt, MAYBE_UNUSED       void *esalt_buf, MAYBE_UNUSED       void *hook_salt_buf, MAYBE_UNUSED       hashinfo_t *hash_info, const char *line_buf, MAYBE_UNUSED const int line_len);
static int         module_hash_decode_postprocess  (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED       void *digest_buf, MAYBE_UNUSED       salt_t *salt, MAYBE_UNUSED       void *esalt_buf, MAYBE_UNUSED       void *hook_salt_buf, MAYBE_UNUSED       hashinfo_t *hash_info, MAYBE_UNUSED const user_options_t *user_options, MAYBE_UNUSED const user_options_extra_t *user_options_extra);

static u64         module_kern_bits      (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const user_options_t *user_options, MAYBE_UNUSED const user_options_extra_t *user_options_extra) { return KERN_BITS;       }

static const u32 ROUNDS_WPA_PMK = 0;

#include "wpa_hccapx.h"

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
  const u64 esalt_size = (const u64) sizeof (wpa_t);

  return esalt_size;
}

static bool is_hccapx (HCFILE *fp)
{
  hccapx_t hccapx;

  const size_t nread = hc_fread (&hccapx, sizeof (hccapx_t), 1, fp);

  if (nread == 1)
  {
    if (hccapx.signature == HCCAPX_SIGNATURE)
    {
      return true;
    }
  }

  return false;
}

static int module_hash_init_selftest (MAYBE_UNUSED const hashconfig_t *hashconfig, hash_t *hash)
{
  const int parser_status = module_hash_decode (hashconfig, hash->digest, hash->salt, hash->esalt, hash->hook_salt, hash->hash_info, hashconfig->st_hash, strlen (hashconfig->st_hash));

  wpa_t *wpa = (wpa_t *) hash->esalt;

  wpa->detected_le = 1;
  wpa->detected_be = 0;

  wpa->nonce_error_corrections = 3;

  return parser_status;
}

static int module_hash_binary_parse (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const user_options_t *user_options, MAYBE_UNUSED const user_options_extra_t *user_options_extra, hashes_t *hashes)
{
  hash_t *hashes_buf = hashes->hashes_buf;

  int hashes_cnt = 0;

  // the count the core allocated entries for. Parsing must not write past it, whatever the file
  // holds and however the two loops below are reached.

  const int hashes_avail = (int) hashes->hashes_cnt;

  HCFILE fp;

  if (hc_fopen (&fp, hashes->hashfile, "rb") == false) return (PARSER_HAVE_ERRNO);

  const bool r = is_hccapx (&fp);

  hc_rewind (&fp);

  if (r == true)
  {
    char *in = (char *) hcmalloc (sizeof (hccapx_t));

    while (!hc_feof (&fp))
    {
      if (hashes_cnt >= hashes_avail) break;

      const size_t nread = hc_fread (in, sizeof (hccapx_t), 1, &fp);

      if (nread == 0) break;

      memset (hashes_buf[hashes_cnt].salt, 0, sizeof (salt_t));

      memset (hashes_buf[hashes_cnt].esalt, 0, sizeof (wpa_t));

      /* moved to module_hash_decode_postprocess()
      wpa_t *wpa = (wpa_t *) hashes_buf[hashes_cnt].esalt;

      wpa->message_pair_chgd = module_option_given (user_options, OPTION_EAPOL_MESSAGE_PAIR);
      wpa->message_pair      = module_option_u32 (user_options, OPTION_EAPOL_MESSAGE_PAIR);

      wpa->nonce_error_corrections_chgd = module_option_given (user_options, OPTION_NONCE_ERROR_CORRECTIONS);
      wpa->nonce_error_corrections      = module_option_u32 (user_options, OPTION_NONCE_ERROR_CORRECTIONS);
      */

      hash_t *hash = &hashes_buf[hashes_cnt];

      const int parser_status = module_hash_decode (hashconfig, hash->digest, hash->salt, hash->esalt, hash->hook_salt, hash->hash_info, in, sizeof (hccapx_t));

      if (parser_status != PARSER_OK) continue;

      hashes_cnt++;
    }

    hcfree (in);
  }
  else
  {
    char *line_buf = (char *) hcmalloc (HCBUFSIZ_LARGE);

    while (!hc_feof (&fp))
    {
      if (hashes_cnt >= hashes_avail) break;

      const size_t line_len = fgetl (&fp, line_buf, HCBUFSIZ_LARGE);

      if (line_len == 0) continue;

      memset (hashes_buf[hashes_cnt].salt, 0, sizeof (salt_t));

      memset (hashes_buf[hashes_cnt].esalt, 0, sizeof (wpa_t));

      /* moved to module_hash_decode_postprocess()
      wpa_t *wpa = (wpa_t *) hashes_buf[hashes_cnt].esalt;

      wpa->message_pair_chgd = module_option_given (user_options, OPTION_EAPOL_MESSAGE_PAIR);
      wpa->message_pair      = module_option_u32 (user_options, OPTION_EAPOL_MESSAGE_PAIR);

      wpa->nonce_error_corrections_chgd = module_option_given (user_options, OPTION_NONCE_ERROR_CORRECTIONS);
      wpa->nonce_error_corrections      = module_option_u32 (user_options, OPTION_NONCE_ERROR_CORRECTIONS);
      */

      hash_t *hash = &hashes_buf[hashes_cnt];

      const int parser_status = module_hash_decode (hashconfig, hash->digest, hash->salt, hash->esalt, hash->hook_salt, hash->hash_info, line_buf, line_len);

      if (parser_status != PARSER_OK) continue;

      hashes_cnt++;
    }

    hcfree (line_buf);
  }

  hc_fclose (&fp);

  return hashes_cnt;
}

static int module_hash_binary_count (MAYBE_UNUSED const hashes_t *hashes)
{
  // this mode actually works on a plaintext file
  // but to stay in a .hccapx backward compatibility mode we have to tell the module
  // the file is in binary.
  // we then have to iterated through the file ourself

  HCFILE fp;

  if (hc_fopen (&fp, hashes->hashfile, "rb") == false) return (PARSER_HAVE_ERRNO);

  const bool r = is_hccapx (&fp);

  hc_rewind (&fp);

  int count = 0;

  if (r == true)
  {
    // stat () would measure the file on disk, and hc_fopen () above transparently decompresses gzip,
    // xz and zstd. A compressed hccapx therefore holds far more records than its size on disk
    // suggests, and the count decides how many hash entries module_hash_binary_parse () may fill.

    char *in = (char *) hcmalloc (sizeof (hccapx_t));

    u64 records = 0;

    while (hc_feof (&fp) == false)
    {
      const size_t nread = hc_fread (in, sizeof (hccapx_t), 1, &fp);

      if (nread == 0) break;

      records++;

      if (records == INT_MAX) break;
    }

    hcfree (in);

    count = (int) records;
  }
  else
  {
    count = count_lines (&fp);
  }

  hc_fclose (&fp);

  return count;
}

static bool module_hlfmt_disable (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const user_options_t *user_options, MAYBE_UNUSED const user_options_extra_t *user_options_extra)
{
  const bool hlfmt_disable = true;

  return hlfmt_disable;
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
  wpa_t *wpa = (wpa_t *) esalt_buf;

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

  // module_hash_encode_potfile writes a colon here and always has. Mode 22000 uses '*' in the same
  // place, but matching it would leave every potfile this mode has already written unreadable.

  const char *sep_pos = strrchr (line_buf, ':');

  if (sep_pos == NULL) return (PARSER_SEPARATOR_UNMATCHED);

  if ((line_buf + 64) != sep_pos) return (PARSER_HASH_LENGTH);

  const char *essid_pos = sep_pos + 1;

  const int essid_len = strlen (essid_pos);

  if (essid_len & 1) return (PARSER_SALT_VALUE);

  if (essid_len > 64) return (PARSER_SALT_VALUE);

  wpa->essid_len = hex_decode ((const u8 *) essid_pos, essid_len, (u8 *) wpa->essid_buf);

  return PARSER_OK;
}

static int module_hash_encode_potfile (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const void *digest_buf, MAYBE_UNUSED const salt_t *salt, MAYBE_UNUSED const void *esalt_buf, MAYBE_UNUSED const void *hook_salt_buf, MAYBE_UNUSED const hashinfo_t *hash_info, char *line_buf, MAYBE_UNUSED const int line_size, MAYBE_UNUSED const void *tmps)
{
  const wpa_t *wpa = (const wpa_t *) esalt_buf;

  const wpa_pmk_tmp_t *wpa_pmk_tmp = (const wpa_pmk_tmp_t *) tmps;

  char tmp_buf[128];

  const int tmp_len = hex_encode ((const u8 *) wpa->essid_buf, wpa->essid_len, (u8 *) tmp_buf);

  tmp_buf[tmp_len] = 0;

  const int line_len = snprintf (line_buf, line_size, "%08x%08x%08x%08x%08x%08x%08x%08x:%s",
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

  const wpa_t *wpas = (const wpa_t *) esalts_buf;
  const wpa_t *wpa  = &wpas[digest_cur];

  char tmp_buf[128];

  const int tmp_len = hex_encode ((const u8 *) wpa->essid_buf, wpa->essid_len, (u8 *) tmp_buf);

  tmp_buf[tmp_len] = 0;

  const u8 *mac_ap  = (const u8 *) wpa->mac_ap;
  const u8 *mac_sta = (const u8 *) wpa->mac_sta;

  // the three fields an FT record ends in, empty for the two plain types

  char ft_buf[1 + 4 + 1 + 96 + 1 + 96 + 1];

  ft_buf[0] = 0;

  if ((wpa->type == 3) || (wpa->type == 4))
  {
    const u8 *mdid = (const u8 *) &wpa->mdid;

    char r0khid_buf[96 + 1];
    char r1khid_buf[96 + 1];

    const int r0khid_len = hex_encode ((const u8 *) wpa->r0khid, wpa->r0khid_len, (u8 *) r0khid_buf);
    const int r1khid_len = hex_encode ((const u8 *) wpa->r1khid, wpa->r1khid_len, (u8 *) r1khid_buf);

    r0khid_buf[r0khid_len] = 0;
    r1khid_buf[r1khid_len] = 0;

    snprintf (ft_buf, sizeof (ft_buf), "*%02x%02x*%s*%s", mdid[0], mdid[1], r0khid_buf, r1khid_buf);
  }

  if (wpa->type == 1)
  {
    // keep the bit that selects HMAC-SHA256, written the way hcxpcapngtool writes it

    const char *message_pair = (wpa->keyver == 3) ? "03" : "";

    const int len = hc_asprintf (buf, "WPA*01*%08x%08x%08x%08x*%02x%02x%02x%02x%02x%02x*%02x%02x%02x%02x%02x%02x*%s***%s" EOL,
      byte_swap_32 (wpa->pmkid[0]),
      byte_swap_32 (wpa->pmkid[1]),
      byte_swap_32 (wpa->pmkid[2]),
      byte_swap_32 (wpa->pmkid[3]),
      mac_ap[0],
      mac_ap[1],
      mac_ap[2],
      mac_ap[3],
      mac_ap[4],
      mac_ap[5],
      mac_sta[0],
      mac_sta[1],
      mac_sta[2],
      mac_sta[3],
      mac_sta[4],
      mac_sta[5],
      tmp_buf,
      message_pair);

    return len;
  }
  else if (wpa->type == 3)
  {
    const int len = hc_asprintf (buf, "WPA*03*%08x%08x%08x%08x*%02x%02x%02x%02x%02x%02x*%02x%02x%02x%02x%02x%02x*%s***%02x%s" EOL,
      byte_swap_32 (wpa->pmkid[0]),
      byte_swap_32 (wpa->pmkid[1]),
      byte_swap_32 (wpa->pmkid[2]),
      byte_swap_32 (wpa->pmkid[3]),
      mac_ap[0],
      mac_ap[1],
      mac_ap[2],
      mac_ap[3],
      mac_ap[4],
      mac_ap[5],
      mac_sta[0],
      mac_sta[1],
      mac_sta[2],
      mac_sta[3],
      mac_sta[4],
      mac_sta[5],
      tmp_buf,
      wpa->message_pair,
      ft_buf);

    return len;
  }
  else if ((wpa->type == 2) || (wpa->type == 4))
  {
    u32 eapol_swapped[(WPA_EAPOL_LEN_MAX / 4) + 2];

    for (int i = 0; i < (WPA_EAPOL_LEN_MAX / 4); i++)
    {
      eapol_swapped[i] = wpa->eapol[i];

      if (wpa->keyver == 2)
      {
        eapol_swapped[i] = byte_swap_32 (eapol_swapped[i]);
      }
    }

    eapol_swapped[(WPA_EAPOL_LEN_MAX / 4) + 0] = 0;
    eapol_swapped[(WPA_EAPOL_LEN_MAX / 4) + 1] = 0;

    char tmp2_buf[(WPA_EAPOL_LEN_MAX * 2) + 1];

    const int tmp2_len = hex_encode ((const u8 *) eapol_swapped, wpa->eapol_len, (u8 *) tmp2_buf);

    tmp2_buf[tmp2_len] = 0;

    const int len = hc_asprintf (buf, "WPA*%02x*%08x%08x%08x%08x*%02x%02x%02x%02x%02x%02x*%02x%02x%02x%02x%02x%02x*%s*%08x%08x%08x%08x%08x%08x%08x%08x*%s*%02x%s" EOL,
      wpa->type,
      wpa->keymic[0],
      wpa->keymic[1],
      wpa->keymic[2],
      wpa->keymic[3],
      mac_ap[0],
      mac_ap[1],
      mac_ap[2],
      mac_ap[3],
      mac_ap[4],
      mac_ap[5],
      mac_sta[0],
      mac_sta[1],
      mac_sta[2],
      mac_sta[3],
      mac_sta[4],
      mac_sta[5],
      tmp_buf,
      byte_swap_32 (wpa->anonce[0]),
      byte_swap_32 (wpa->anonce[1]),
      byte_swap_32 (wpa->anonce[2]),
      byte_swap_32 (wpa->anonce[3]),
      byte_swap_32 (wpa->anonce[4]),
      byte_swap_32 (wpa->anonce[5]),
      byte_swap_32 (wpa->anonce[6]),
      byte_swap_32 (wpa->anonce[7]),
      tmp2_buf,
      wpa->message_pair,
      ft_buf);

    return len;
  }

  return 0;
}

static u32 module_deep_comp_kernel (MAYBE_UNUSED const hashes_t *hashes, MAYBE_UNUSED const u32 salt_pos, MAYBE_UNUSED const u32 digest_pos)
{
  const u32 digests_offset = hashes->salts_buf[salt_pos].digests_offset;

  wpa_t *wpas = (wpa_t *) hashes->esalts_buf;

  wpa_t *wpa = &wpas[digests_offset + digest_pos];

  if (wpa->type == 1)
  {
    return KERN_RUN_AUX4;
  }
  else if (wpa->type == 2)
  {
    if (wpa->keyver == 1)
    {
      return KERN_RUN_AUX1;
    }
    else if (wpa->keyver == 2)
    {
      return KERN_RUN_AUX2;
    }
    else if (wpa->keyver == 3)
    {
      return KERN_RUN_AUX3;
    }
  }
  else if (wpa->type == 3)
  {
    return KERN_RUN_AUX5;
  }
  else if (wpa->type == 4)
  {
    return KERN_RUN_AUX6;
  }

  return 0;
}

static bool module_potfile_custom_check (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const hash_t *db, MAYBE_UNUSED const hash_t *entry_hash, MAYBE_UNUSED const void *entry_tmps)
{
  const wpa_t *wpa_entry = (const wpa_t *) entry_hash->esalt;
  const wpa_t *wpa_db    = (const wpa_t *) db->esalt;

  if (wpa_db->essid_len != wpa_entry->essid_len) return false;

  if (strcmp ((const char *) wpa_db->essid_buf, (const char *) wpa_entry->essid_buf)) return false;

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

  void (*m22001_aux) (KERN_ATTR_TMPS_ESALT (wpa_pmk_tmp_t, wpa_t));

  if (wpa_db->type == 1)
  {
    m22001_aux = m22001_aux4;
  }
  else if (wpa_db->type == 2)
  {
    if (wpa_db->keyver == 1)
    {
      m22001_aux = m22001_aux1;
    }
    else if (wpa_db->keyver == 2)
    {
      m22001_aux = m22001_aux2;
    }
    else if (wpa_db->keyver == 3)
    {
      m22001_aux = m22001_aux3;
    }
    else
    {
      return false;
    }
  }
  else if (wpa_db->type == 3)
  {
    m22001_aux = m22001_aux5;
  }
  else if (wpa_db->type == 4)
  {
    if (wpa_db->keyver != 3) return false;

    m22001_aux = m22001_aux6;
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

  m22001_aux
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

#define WPA_SALT_ITER (ROUNDS_WPA_PMK)

#include "wpa_decode_eapol.h"

static int module_hash_encode (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED const void *digest_buf, MAYBE_UNUSED const salt_t *salt, MAYBE_UNUSED const void *esalt_buf, MAYBE_UNUSED const void *hook_salt_buf, MAYBE_UNUSED const hashinfo_t *hash_info, char *line_buf, MAYBE_UNUSED const int line_size)
{
  return snprintf (line_buf, line_size, "%s", hash_info->orighash);
}

static int module_hash_decode_postprocess (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED void *digest_buf, MAYBE_UNUSED salt_t *salt, MAYBE_UNUSED void *esalt_buf, MAYBE_UNUSED void *hook_salt_buf, MAYBE_UNUSED hashinfo_t *hash_info, MAYBE_UNUSED const user_options_t *user_options, MAYBE_UNUSED const user_options_extra_t *user_options_extra)
{
  wpa_t *wpa = (wpa_t *) esalt_buf;

  wpa->message_pair_chgd = module_option_given (user_options, OPTION_EAPOL_MESSAGE_PAIR);
  //wpa->message_pair      = module_option_u32 (user_options, OPTION_EAPOL_MESSAGE_PAIR);

  wpa->nonce_error_corrections_chgd = module_option_given (user_options, OPTION_NONCE_ERROR_CORRECTIONS);
  //wpa->nonce_error_corrections      = module_option_u32 (user_options, OPTION_NONCE_ERROR_CORRECTIONS);

  if (wpa->message_pair_chgd == true)
  {
    // we can filter some message types here

    if (module_option_u32 (user_options, OPTION_EAPOL_MESSAGE_PAIR) != (wpa->message_pair & 0x7f)) return (PARSER_HCCAPX_MESSAGE_PAIR);
  }

  if (wpa->nonce_error_corrections_chgd == true)
  {
    wpa->nonce_error_corrections = module_option_u32 (user_options, OPTION_NONCE_ERROR_CORRECTIONS);
  }
  else
  {
    wpa->nonce_error_corrections = NONCE_ERROR_CORRECTIONS;

    if (wpa->message_pair & (1 << 4))
    {
      // ap-less attack detected, nc not needed

      wpa->nonce_error_corrections = 0;
    }
    else
    {
      if (wpa->message_pair & (1 << 7))
      {
        // replaycount not checked, nc needed
      }
      else
      {
        wpa->nonce_error_corrections = 0;
      }
    }
  }

  return (PARSER_OK);
}

void module_init (module_ctx_t *module_ctx)
{
  module_init_common (module_ctx);

  module_ctx->module_benchmark_mask           = module_benchmark_mask;
  module_ctx->module_deep_comp_kernel         = module_deep_comp_kernel;
  module_ctx->module_esalt_size               = module_esalt_size;
  module_ctx->module_hash_binary_count        = module_hash_binary_count;
  module_ctx->module_hash_binary_parse        = module_hash_binary_parse;
  module_ctx->module_hash_binary_save         = module_hash_binary_save;
  module_ctx->module_hash_decode_postprocess  = module_hash_decode_postprocess;
  module_ctx->module_hash_decode_potfile      = module_hash_decode_potfile;
  module_ctx->module_hash_decode              = module_hash_decode;
  module_ctx->module_hash_encode_potfile      = module_hash_encode_potfile;
  module_ctx->module_hash_encode              = module_hash_encode;
  module_ctx->module_hash_init_selftest       = module_hash_init_selftest;
  module_ctx->module_hlfmt_disable            = module_hlfmt_disable;
  module_ctx->module_kern_bits                = module_kern_bits;
  module_ctx->module_potfile_custom_check     = module_potfile_custom_check;
  module_ctx->module_pw_max                   = module_pw_max;
  module_ctx->module_pw_min                   = module_pw_min;
  module_ctx->module_register_options         = wpa_module_register_options;
  module_ctx->module_tmp_size                 = module_tmp_size;
}
