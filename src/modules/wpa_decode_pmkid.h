/**
 * Author......: See docs/credits.txt
 * License.....: MIT
 */

// The PMKID line, parsed by both modes. 16800 is the passphrase mode and its line ends in the ESSID
// the PMK is derived from. 16801 is the PMK mode and knows a three-field line of its own, while also
// accepting the four-field one. WPA_PMKID_PMK_MODE says which shape is tried first, and WPA_SALT_ITER
// carries the one difference the salt itself has.

#ifndef WPA_DECODE_PMKID_H
#define WPA_DECODE_PMKID_H

static void wpa_pmkid_token4 (hc_token_t *token, const u8 sep)
{
  token->token_cnt  = 4;

  token->sep[3]     = sep;
  token->len_min[3] = 0;
  token->len_max[3] = 64;
  token->attr[3]    = TOKEN_ATTR_VERIFY_LENGTH
                    | TOKEN_ATTR_VERIFY_HEX;
}

static int wpa_pmkid_essid (hc_token_t *token, wpa_pmkid_t *wpa_pmkid)
{
  const u8 *essid_buf = token->buf[3];
  const int essid_len = token->len[3];

  if (essid_len & 1) return (PARSER_SALT_VALUE);

  wpa_pmkid->essid_len = hex_decode (essid_buf, essid_len, (u8 *) wpa_pmkid->essid_buf);

  return (PARSER_OK);
}

static int module_hash_decode (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED void *digest_buf, MAYBE_UNUSED salt_t *salt, MAYBE_UNUSED void *esalt_buf, MAYBE_UNUSED void *hook_salt_buf, MAYBE_UNUSED hashinfo_t *hash_info, const char *line_buf, MAYBE_UNUSED const int line_len)
{
  u32 *digest = (u32 *) digest_buf;

  wpa_pmkid_t *wpa_pmkid = (wpa_pmkid_t *) esalt_buf;

  // detect old/new format

  int old_sep = 0;
  int new_sep = 0;

  for (int i = 0; i < line_len; i++)
  {
    const char c = line_buf[i];

    if (c == '*') old_sep++;
    if (c == ':') new_sep++;
  }

  const u8 sep = (new_sep > old_sep) ? ':' : '*';

  // start normal parsing

  hc_token_t token;

  memset (&token, 0, sizeof (hc_token_t));

  token.sep[0]     = sep;
  token.len[0]     = 32;
  token.attr[0]    = TOKEN_ATTR_FIXED_LENGTH
                   | TOKEN_ATTR_VERIFY_HEX;

  token.sep[1]     = sep;
  token.len[1]     = 12;
  token.attr[1]    = TOKEN_ATTR_FIXED_LENGTH
                   | TOKEN_ATTR_VERIFY_HEX;

  token.sep[2]     = sep;
  token.len[2]     = 12;
  token.attr[2]    = TOKEN_ATTR_FIXED_LENGTH
                   | TOKEN_ATTR_VERIFY_HEX;

#if WPA_PMKID_PMK_MODE

  // the three-field form first; the four-field one is retried below

  token.token_cnt  = 3;

  const int rc_tokenizer = input_tokenizer ((const u8 *) line_buf, line_len, &token);

  if (rc_tokenizer != PARSER_OK)
  {
    // we accept normal 16800 pmkid hash-lines, too

    wpa_pmkid_token4 (&token, sep);

    const int rc_tokenizer2 = input_tokenizer ((const u8 *) line_buf, line_len, &token);

    if (rc_tokenizer2 != PARSER_OK) return (rc_tokenizer);

    const int rc_essid = wpa_pmkid_essid (&token, wpa_pmkid);

    if (rc_essid != PARSER_OK) return (rc_essid);
  }

#else

  wpa_pmkid_token4 (&token, sep);

  const int rc_tokenizer = input_tokenizer ((const u8 *) line_buf, line_len, &token);

  if (rc_tokenizer != PARSER_OK) return (rc_tokenizer);

  const int rc_essid = wpa_pmkid_essid (&token, wpa_pmkid);

  if (rc_essid != PARSER_OK) return (rc_essid);

#endif

  // pmkid

  const u8 *pmkid_buf = token.buf[0];

  wpa_pmkid->pmkid[0] = hex_to_u32 (pmkid_buf +  0);
  wpa_pmkid->pmkid[1] = hex_to_u32 (pmkid_buf +  8);
  wpa_pmkid->pmkid[2] = hex_to_u32 (pmkid_buf + 16);
  wpa_pmkid->pmkid[3] = hex_to_u32 (pmkid_buf + 24);

  // mac_ap

  const u8 *macap_buf = token.buf[1];

  wpa_pmkid->orig_mac_ap[0] = hex_to_u8 (macap_buf +  0);
  wpa_pmkid->orig_mac_ap[1] = hex_to_u8 (macap_buf +  2);
  wpa_pmkid->orig_mac_ap[2] = hex_to_u8 (macap_buf +  4);
  wpa_pmkid->orig_mac_ap[3] = hex_to_u8 (macap_buf +  6);
  wpa_pmkid->orig_mac_ap[4] = hex_to_u8 (macap_buf +  8);
  wpa_pmkid->orig_mac_ap[5] = hex_to_u8 (macap_buf + 10);

  // mac_sta

  const u8 *macsta_buf = token.buf[2];

  wpa_pmkid->orig_mac_sta[0] = hex_to_u8 (macsta_buf +  0);
  wpa_pmkid->orig_mac_sta[1] = hex_to_u8 (macsta_buf +  2);
  wpa_pmkid->orig_mac_sta[2] = hex_to_u8 (macsta_buf +  4);
  wpa_pmkid->orig_mac_sta[3] = hex_to_u8 (macsta_buf +  6);
  wpa_pmkid->orig_mac_sta[4] = hex_to_u8 (macsta_buf +  8);
  wpa_pmkid->orig_mac_sta[5] = hex_to_u8 (macsta_buf + 10);

  // pmkid_data

  // the salt below copies all eight words, and only the first five are set here. The rest would
  // carry whatever was in the esalt before this record, so zero the array first.

  memset (wpa_pmkid->pmkid_data, 0, sizeof (wpa_pmkid->pmkid_data));

  wpa_pmkid->pmkid_data[0] = 0x204b4d50; // "PMK "
  wpa_pmkid->pmkid_data[1] = 0x656d614e; // "Name"
  wpa_pmkid->pmkid_data[2] = ((u32) wpa_pmkid->orig_mac_ap[0]  <<  0)
                           | ((u32) wpa_pmkid->orig_mac_ap[1]  <<  8)
                           | ((u32) wpa_pmkid->orig_mac_ap[2]  << 16)
                           | ((u32) wpa_pmkid->orig_mac_ap[3]  << 24);
  wpa_pmkid->pmkid_data[3] = ((u32) wpa_pmkid->orig_mac_ap[4]  <<  0)
                           | ((u32) wpa_pmkid->orig_mac_ap[5]  <<  8)
                           | ((u32) wpa_pmkid->orig_mac_sta[0] << 16)
                           | ((u32) wpa_pmkid->orig_mac_sta[1] << 24);
  wpa_pmkid->pmkid_data[4] = ((u32) wpa_pmkid->orig_mac_sta[2] <<  0)
                           | ((u32) wpa_pmkid->orig_mac_sta[3] <<  8)
                           | ((u32) wpa_pmkid->orig_mac_sta[4] << 16)
                           | ((u32) wpa_pmkid->orig_mac_sta[5] << 24);

  // salt

#if WPA_PMKID_PMK_MODE

  salt->salt_buf[0] = wpa_pmkid->pmkid_data[0];
  salt->salt_buf[1] = wpa_pmkid->pmkid_data[1];
  salt->salt_buf[2] = wpa_pmkid->pmkid_data[2];
  salt->salt_buf[3] = wpa_pmkid->pmkid_data[3];
  salt->salt_buf[4] = wpa_pmkid->pmkid_data[4];
  salt->salt_buf[5] = wpa_pmkid->pmkid_data[5];
  salt->salt_buf[6] = wpa_pmkid->pmkid_data[6];
  salt->salt_buf[7] = wpa_pmkid->pmkid_data[7];

  salt->salt_len  = 32;

#else

  memcpy (salt->salt_buf, wpa_pmkid->essid_buf, wpa_pmkid->essid_len);

  salt->salt_len  = wpa_pmkid->essid_len;

#endif

  salt->salt_iter = WPA_SALT_ITER;

  // hash

  digest[0] = wpa_pmkid->pmkid[0];
  digest[1] = wpa_pmkid->pmkid[1];
  digest[2] = wpa_pmkid->pmkid[2];
  digest[3] = wpa_pmkid->pmkid[3];

  digest[0] = byte_swap_32 (digest[0]);
  digest[1] = byte_swap_32 (digest[1]);
  digest[2] = byte_swap_32 (digest[2]);
  digest[3] = byte_swap_32 (digest[3]);

  return (PARSER_OK);
}

#endif