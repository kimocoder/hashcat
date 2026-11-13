/**
 * Author......: See docs/credits.txt
 * License.....: MIT
 */

// The WPA* text line, parsed the same way by 22000 and 22001. They differ only in what the password
// is, which reaches the parser as the salt iteration count: 22000 derives the PMK from a passphrase
// and 22001 is handed the PMK itself. WPA_SALT_ITER carries that difference in.

#ifndef WPA_DECODE_EAPOL_H
#define WPA_DECODE_EAPOL_H

static int module_hash_decode (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED void *digest_buf, MAYBE_UNUSED salt_t *salt, MAYBE_UNUSED void *esalt_buf, MAYBE_UNUSED void *hook_salt_buf, MAYBE_UNUSED hashinfo_t *hash_info, const char *line_buf, MAYBE_UNUSED const int line_len)
{
  u32 *digest = (u32 *) digest_buf;

  wpa_t *wpa = (wpa_t *) esalt_buf;

  const char *input_buf = line_buf;
  int   input_len = line_len;

  // start old pmkid/hccapx compatibility parsing
  // idea is to find out if parsing succeeds and in this case to build a
  // valid hash line and replace line_buf pointer

  char tmp_buf[1024];
  int  tmp_len;

  // hccapx parser

  if (line_len == sizeof (hccapx_t))
  {
    const hccapx_t *hccapx = (const hccapx_t *) line_buf;

    if ((hccapx->signature == HCCAPX_SIGNATURE) && (hccapx->version == HCCAPX_VERSION))
    {
      if (hccapx->essid_len > 32) return (PARSER_SALT_LENGTH);
      if (hccapx->eapol_len < 1 || hccapx->eapol_len > 256) return (PARSER_HCCAPX_EAPOL_LEN);

      tmp_len = 0;

      tmp_len += snprintf (tmp_buf, sizeof (tmp_buf) - tmp_len, "WPA*02*");

      tmp_len += hex_encode ((const u8 *) hccapx->keymic, 16, (u8 *) tmp_buf + tmp_len);

      tmp_buf[tmp_len] = '*';

      tmp_len++;

      tmp_len += hex_encode ((const u8 *) hccapx->mac_ap, 6, (u8 *) tmp_buf + tmp_len);

      tmp_buf[tmp_len] = '*';

      tmp_len++;

      tmp_len += hex_encode ((const u8 *) hccapx->mac_sta, 6, (u8 *) tmp_buf + tmp_len);

      tmp_buf[tmp_len] = '*';

      tmp_len++;

      tmp_len += hex_encode ((const u8 *) hccapx->essid, hccapx->essid_len, (u8 *) tmp_buf + tmp_len);

      tmp_buf[tmp_len] = '*';

      tmp_len++;

      tmp_len += hex_encode ((const u8 *) hccapx->nonce_ap, 32, (u8 *) tmp_buf + tmp_len);

      tmp_buf[tmp_len] = '*';

      tmp_len++;

      tmp_len += hex_encode ((const u8 *) hccapx->eapol, hccapx->eapol_len, (u8 *) tmp_buf + tmp_len);

      tmp_buf[tmp_len] = '*';

      tmp_len++;

      tmp_len += hex_encode (&hccapx->message_pair, 1, (u8 *) tmp_buf + tmp_len);

      tmp_buf[tmp_len] = 0;

      input_buf = tmp_buf;
      input_len = tmp_len;

      // keep the line this record was converted to, so --show and --outfile have a hash to write.
      // The buffer stage 1 hands down is 256 bytes, which a WPA*02 line with a long EAPOL exceeds.

      if (hash_info != NULL)
      {
        hcfree (hash_info->orighash);

        hash_info->orighash = hcstrdup (tmp_buf);
      }
    }
  }

  // pmkid parser

  if (1)
  {
    // detect super-old/old format

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

    token.token_cnt  = 4;

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

    token.sep[3]     = sep;
    token.len_min[3] = 0;
    token.len_max[3] = 64;
    token.attr[3]    = TOKEN_ATTR_VERIFY_LENGTH
                     | TOKEN_ATTR_VERIFY_HEX;

    const int rc_tokenizer = input_tokenizer ((const u8 *) line_buf, line_len, &token);

    // The line is rebuilt from the tokens rather than copied, because the old format is written with
    // either separator and the new one only knows '*'. Copying a line that used ':' left all 4 fields
    // as one token for the parser below, which then refused its own old format.
    if (rc_tokenizer == PARSER_OK)
    {
      tmp_len = snprintf (tmp_buf, sizeof (tmp_buf), "WPA*01*%.*s*%.*s*%.*s*%.*s***",
        token.len[0], (const char *) token.buf[0],
        token.len[1], (const char *) token.buf[1],
        token.len[2], (const char *) token.buf[2],
        token.len[3], (const char *) token.buf[3]);

      input_buf = tmp_buf;
      input_len = tmp_len;
    }
  }

  // start normal parsing

  hc_token_t token;

  memset (&token, 0, sizeof (hc_token_t));

  // an FT record carries three fields the other types do not, and every field is hex, so the
  // separators can be counted before the line is tokenized

  int sep_cnt = 0;

  for (int i = 0; i < input_len; i++)
  {
    if (input_buf[i] == '*') sep_cnt++;
  }

  token.token_cnt  = (sep_cnt == 11) ? 12 : 9;

  token.signatures_cnt    = 1;
  token.signatures_buf[0] = "WPA";

  token.sep[0]     = '*';
  token.len[0]     = 3;
  token.attr[0]    = TOKEN_ATTR_FIXED_LENGTH
                   | TOKEN_ATTR_VERIFY_SIGNATURE;

  token.sep[1]     = '*';
  token.len[1]     = 2;
  token.attr[1]    = TOKEN_ATTR_FIXED_LENGTH
                   | TOKEN_ATTR_VERIFY_HEX;

  token.sep[2]     = '*';
  token.len[2]     = 32;
  token.attr[2]    = TOKEN_ATTR_FIXED_LENGTH
                   | TOKEN_ATTR_VERIFY_HEX;

  token.sep[3]     = '*';
  token.len[3]     = 12;
  token.attr[3]    = TOKEN_ATTR_FIXED_LENGTH
                   | TOKEN_ATTR_VERIFY_HEX;

  token.sep[4]     = '*';
  token.len[4]     = 12;
  token.attr[4]    = TOKEN_ATTR_FIXED_LENGTH
                   | TOKEN_ATTR_VERIFY_HEX;

  token.sep[5]     = '*';
  token.len_min[5] = 0;
  token.len_max[5] = 64;
  token.attr[5]    = TOKEN_ATTR_VERIFY_LENGTH
                   | TOKEN_ATTR_VERIFY_HEX;

  token.sep[6]     = '*';
  token.len_min[6] = 0;
  token.len_max[6] = 64;
  token.attr[6]    = TOKEN_ATTR_VERIFY_LENGTH
                   | TOKEN_ATTR_VERIFY_HEX;

  token.sep[7]     = '*';
  token.len_min[7] = 0;
  token.len_max[7] = WPA_EAPOL_LEN_MAX * 2;
  token.attr[7]    = TOKEN_ATTR_VERIFY_LENGTH
                   | TOKEN_ATTR_VERIFY_HEX;

  token.sep[8]     = '*';
  token.len_min[8] = 0;
  token.len_max[8] = 2;
  token.attr[8]    = TOKEN_ATTR_VERIFY_LENGTH
                   | TOKEN_ATTR_VERIFY_HEX;

  if (token.token_cnt == 12)
  {
    token.sep[9]      = '*';
    token.len[9]      = 4;
    token.attr[9]     = TOKEN_ATTR_FIXED_LENGTH
                      | TOKEN_ATTR_VERIFY_HEX;

    token.sep[10]     = '*';
    token.len_min[10] = 0;
    token.len_max[10] = 96;
    token.attr[10]    = TOKEN_ATTR_VERIFY_LENGTH
                      | TOKEN_ATTR_VERIFY_HEX;

    token.sep[11]     = '*';
    token.len_min[11] = 0;
    token.len_max[11] = 96;
    token.attr[11]    = TOKEN_ATTR_VERIFY_LENGTH
                      | TOKEN_ATTR_VERIFY_HEX;
  }

  const int rc_tokenizer = input_tokenizer ((const u8 *) input_buf, input_len, &token);

  if (rc_tokenizer != PARSER_OK) return (rc_tokenizer);

  // mac_ap

  u8 *mac_ap  = (u8 *) wpa->mac_ap;
  u8 *mac_sta = (u8 *) wpa->mac_sta;

  const u8 *macap_buf = token.buf[3];

  mac_ap[0] = hex_to_u8 (macap_buf +  0);
  mac_ap[1] = hex_to_u8 (macap_buf +  2);
  mac_ap[2] = hex_to_u8 (macap_buf +  4);
  mac_ap[3] = hex_to_u8 (macap_buf +  6);
  mac_ap[4] = hex_to_u8 (macap_buf +  8);
  mac_ap[5] = hex_to_u8 (macap_buf + 10);

  // mac_sta

  const u8 *macsta_buf = token.buf[4];

  mac_sta[0] = hex_to_u8 (macsta_buf +  0);
  mac_sta[1] = hex_to_u8 (macsta_buf +  2);
  mac_sta[2] = hex_to_u8 (macsta_buf +  4);
  mac_sta[3] = hex_to_u8 (macsta_buf +  6);
  mac_sta[4] = hex_to_u8 (macsta_buf +  8);
  mac_sta[5] = hex_to_u8 (macsta_buf + 10);

  // essid

  const u8 *essid_buf = token.buf[5];
  const int essid_len = token.len[5];

  if (essid_len & 1) return (PARSER_SALT_VALUE);

  wpa->essid_len = hex_decode (essid_buf, essid_len, (u8 *) wpa->essid_buf);

  // salt

  memcpy (salt->salt_buf, wpa->essid_buf, wpa->essid_len);

  salt->salt_len = wpa->essid_len;

  salt->salt_iter = WPA_SALT_ITER;

  // type

  const u8 *type_buf = token.buf[1];

  const u8 type = hex_to_u8 (type_buf);

  if ((type != 1) && (type != 2) && (type != 3) && (type != 4)) return (PARSER_SALT_VALUE);

  const bool is_ft = ((type == 3) || (type == 4));

  // the extra separators were what the token count was taken from, so a record has to carry either
  // all three FT fields or none of them

  if ((is_ft == true)  && (token.token_cnt != 12)) return (PARSER_SALT_VALUE);
  if ((is_ft == false) && (token.token_cnt !=  9)) return (PARSER_SALT_VALUE);

  wpa->type = type;

  // FT specific code, shared by both FT types

  if (is_ft == true)
  {
    // mdid

    const u8 *mdid_pos = token.buf[9];

    u8 *mdid_ptr = (u8 *) &wpa->mdid;

    mdid_ptr[0] = hex_to_u8 (mdid_pos + 0);
    mdid_ptr[1] = hex_to_u8 (mdid_pos + 2);

    // r0kh-id

    if (token.len[10] & 1) return (PARSER_SALT_VALUE);

    u8 *r0khid_ptr = (u8 *) wpa->r0khid;

    wpa->r0khid_len = hex_decode (token.buf[10], token.len[10], r0khid_ptr);

    // r1kh-id

    if (token.len[11] & 1) return (PARSER_SALT_VALUE);

    u8 *r1khid_ptr = (u8 *) wpa->r1khid;

    wpa->r1khid_len = hex_decode (token.buf[11], token.len[11], r1khid_ptr);

    // PMK-R0 = KDF-384 (PMK, "FT-R0", ESSIDlen || ESSID || MDID || R0KHIDlen || R0KH-ID || STA-MAC).
    // The EAPOL path uses the first of the two KDF blocks, which is the key itself. The PMKID path
    // uses the second one, whose first 16 bytes are the PMK-R0-Name salt, so it never derives the key
    // at all. The two inputs differ only in the counter in front of them.

    u8 *pke_r0_ptr = (u8 *) wpa->pke_r0;

    memset (pke_r0_ptr, 0, sizeof (wpa->pke_r0));

    pke_r0_ptr[0] = (type == 3) ? 2 : 1;
    pke_r0_ptr[1] = 0;

    memcpy (pke_r0_ptr + 2, "FT-R0", 5);

    pke_r0_ptr[7] = (u8) wpa->essid_len;

    memcpy (pke_r0_ptr + 8, wpa->essid_buf, wpa->essid_len);
    memcpy (pke_r0_ptr + 8 + wpa->essid_len, mdid_ptr, 2);

    pke_r0_ptr[10 + wpa->essid_len] = (u8) wpa->r0khid_len;

    memcpy (pke_r0_ptr + 11 + wpa->essid_len, r0khid_ptr, wpa->r0khid_len);
    memcpy (pke_r0_ptr + 11 + wpa->essid_len + wpa->r0khid_len, mac_sta, 6);

    // the KDF output size, 384 bits, little endian

    pke_r0_ptr[17 + wpa->essid_len + wpa->r0khid_len] = 0x80;
    pke_r0_ptr[18 + wpa->essid_len + wpa->r0khid_len] = 1;

    for (u32 i = 0; i < (sizeof (wpa->pke_r0) / 4); i++)
    {
      wpa->pke_r0[i] = byte_swap_32 (wpa->pke_r0[i]);
    }

    // an FT record from hcxpcapngtool carries a message pair even where it carries no EAPOL frame

    if (token.len[8] == 2) wpa->message_pair = hex_to_u8 (token.buf[8]);
  }

  // PMKID specific code

  if (type == 1)
  {
    // pmkid

    const u8 *pmkid_buf = token.buf[2];

    wpa->pmkid[0] = hex_to_u32 (pmkid_buf +  0);
    wpa->pmkid[1] = hex_to_u32 (pmkid_buf +  8);
    wpa->pmkid[2] = hex_to_u32 (pmkid_buf + 16);
    wpa->pmkid[3] = hex_to_u32 (pmkid_buf + 24);

    // pmkid_data

    wpa->pmkid_data[0] = 0x204b4d50; // "PMK "
    wpa->pmkid_data[1] = 0x656d614e; // "Name"
    wpa->pmkid_data[2] = ((u32) mac_ap[0]  <<  0)
                       | ((u32) mac_ap[1]  <<  8)
                       | ((u32) mac_ap[2]  << 16)
                       | ((u32) mac_ap[3]  << 24);
    wpa->pmkid_data[3] = ((u32) mac_ap[4]  <<  0)
                       | ((u32) mac_ap[5]  <<  8)
                       | ((u32) mac_sta[0] << 16)
                       | ((u32) mac_sta[1] << 24);
    wpa->pmkid_data[4] = ((u32) mac_sta[2] <<  0)
                       | ((u32) mac_sta[3] <<  8)
                       | ((u32) mac_sta[4] << 16)
                       | ((u32) mac_sta[5] << 24);

    // hcxpcapngtool sets bit 1 of the message pair on a PMKID it took from an M1 of key version 3.
    // That M1 comes from an AKM 6 (PSK-SHA256) network, whose PMKID is HMAC-SHA256 and not HMAC-SHA1.

    wpa->keyver = 0;

    if (token.len[8] == 2)
    {
      const u8 message_pair = hex_to_u8 (token.buf[8]);

      if ((message_pair & 0x02) == 0x02) wpa->keyver = 3;
    }

    // hash

    digest[0] = wpa->pmkid[0];
    digest[1] = wpa->pmkid[1];
    digest[2] = wpa->pmkid[2];
    digest[3] = wpa->pmkid[3];

    digest[0] = byte_swap_32 (digest[0]);
    digest[1] = byte_swap_32 (digest[1]);
    digest[2] = byte_swap_32 (digest[2]);
    digest[3] = byte_swap_32 (digest[3]);
  }

  // FT PMKID specific code

  if (type == 3)
  {
    // pmkid

    const u8 *pmkid_buf = token.buf[2];

    wpa->pmkid[0] = hex_to_u32 (pmkid_buf +  0);
    wpa->pmkid[1] = hex_to_u32 (pmkid_buf +  8);
    wpa->pmkid[2] = hex_to_u32 (pmkid_buf + 16);
    wpa->pmkid[3] = hex_to_u32 (pmkid_buf + 24);

    // The PMKID of an FT record is PMK-R1-Name, which is a SHA-256 over
    // "FT-R1N" || PMK-R0-Name || R1KH-ID || STA-MAC. PMK-R0-Name follows from the candidate, so the
    // input is stored with a 16 byte gap and the kernel ORs the name into it rather than writing the
    // whole input back to global memory.

    u8 *pmkid_data_ptr = (u8 *) wpa->pmkid_data;

    memset (pmkid_data_ptr, 0, sizeof (wpa->pmkid_data));

    memcpy (pmkid_data_ptr, "FT-R1N", 6);
    memcpy (pmkid_data_ptr + 22, (const u8 *) wpa->r1khid, wpa->r1khid_len);
    memcpy (pmkid_data_ptr + 22 + wpa->r1khid_len, mac_sta, 6);

    for (u32 i = 0; i < (sizeof (wpa->pmkid_data) / 4); i++)
    {
      wpa->pmkid_data[i] = byte_swap_32 (wpa->pmkid_data[i]);
    }

    // hash

    digest[0] = byte_swap_32 (wpa->pmkid[0]);
    digest[1] = byte_swap_32 (wpa->pmkid[1]);
    digest[2] = byte_swap_32 (wpa->pmkid[2]);
    digest[3] = byte_swap_32 (wpa->pmkid[3]);
  }

  // EAPOL specific code, shared by the plain and the FT record

  if ((type == 2) || (type == 4))
  {
    // checks

    if (token.len[6] != 64) return (PARSER_SALT_LENGTH);

    if (token.len[7] < (int) sizeof (auth_packet_t) * 2) return (PARSER_SALT_LENGTH);

    if (token.len[8] != 2) return (PARSER_SALT_LENGTH);

    // anonce

    const u8 *anonce_pos = token.buf[6];

    wpa->anonce[0] = hex_to_u32 (anonce_pos +  0);
    wpa->anonce[1] = hex_to_u32 (anonce_pos +  8);
    wpa->anonce[2] = hex_to_u32 (anonce_pos + 16);
    wpa->anonce[3] = hex_to_u32 (anonce_pos + 24);
    wpa->anonce[4] = hex_to_u32 (anonce_pos + 32);
    wpa->anonce[5] = hex_to_u32 (anonce_pos + 40);
    wpa->anonce[6] = hex_to_u32 (anonce_pos + 48);
    wpa->anonce[7] = hex_to_u32 (anonce_pos + 56);

    // eapol

    const u8 *eapol_pos = token.buf[7];

    u8 *eapol_ptr = (u8 *) wpa->eapol;

    wpa->eapol_len = hex_decode (eapol_pos, token.len[7], eapol_ptr);

    memset (eapol_ptr + wpa->eapol_len, 0, (WPA_EAPOL_LEN_MAX + 64) - wpa->eapol_len);

    auth_packet_t *auth_packet = (auth_packet_t *) wpa->eapol;

    // keyver

    // The key version is bits 0..2 of the key information word. Reading only two bits would fold a
    // reserved value into one of the three defined ones, so the whole field is taken and checked.

    const u16 key_information = byte_swap_16 (auth_packet->key_information);

    wpa->keyver = key_information & 7;

    if ((wpa->keyver != 1) && (wpa->keyver != 2) && (wpa->keyver != 3)) return (PARSER_SALT_VALUE);

    // 802.11r defines the FT key hierarchy over AES-CMAC only

    if ((type == 4) && (wpa->keyver != 3)) return (PARSER_SALT_VALUE);

    // pke

    u8 *pke_ptr = (u8 *) wpa->pke;

    memset (pke_ptr, 0, 128);

    if (type == 4)
    {
      // PTK = KDF-384 (PMK-R1, "FT-PTK", SNonce || ANonce || AP-MAC || STA-MAC)

      pke_ptr[0] = 1;
      pke_ptr[1] = 0;

      memcpy (pke_ptr + 2, "FT-PTK", 6);

      // Message pairs 3 and 4 store M3, which carries the ANonce, and put the SNonce in the nonce
      // field. The plain WPA PRF sorts the two nonces, but the FT KDF takes them in a fixed order.

      const u8 eapol_msg = wpa->message_pair & 7;

      if ((eapol_msg == 3) || (eapol_msg == 4))
      {
        memcpy (pke_ptr +  8, wpa->anonce, 32);
        memcpy (pke_ptr + 40, auth_packet->wpa_key_nonce, 32);
      }
      else
      {
        memcpy (pke_ptr +  8, auth_packet->wpa_key_nonce, 32);
        memcpy (pke_ptr + 40, wpa->anonce, 32);
      }

      memcpy (pke_ptr + 72, mac_ap,  6);
      memcpy (pke_ptr + 78, mac_sta, 6);

      pke_ptr[84] = 0x80;
      pke_ptr[85] = 1;

      // PMK-R1 = KDF-256 (PMK-R0, "FT-R1", R1KH-ID || STA-MAC)

      u8 *pke_r1_ptr = (u8 *) wpa->pke_r1;

      memset (pke_r1_ptr, 0, sizeof (wpa->pke_r1));

      pke_r1_ptr[0] = 1;
      pke_r1_ptr[1] = 0;

      memcpy (pke_r1_ptr + 2, "FT-R1", 5);

      memcpy (pke_r1_ptr + 7, (const u8 *) wpa->r1khid, wpa->r1khid_len);
      memcpy (pke_r1_ptr + 7 + wpa->r1khid_len, mac_sta, 6);

      // the KDF output size, 256 bits, little endian

      pke_r1_ptr[13 + wpa->r1khid_len] = 0;
      pke_r1_ptr[14 + wpa->r1khid_len] = 1;

      for (u32 i = 0; i < (sizeof (wpa->pke_r1) / 4); i++)
      {
        wpa->pke_r1[i] = byte_swap_32 (wpa->pke_r1[i]);
      }
    }
    else if ((wpa->keyver == 1) || (wpa->keyver == 2))
    {
      memcpy (pke_ptr, "Pairwise key expansion\x00", 23);

      if (memcmp (mac_ap, mac_sta, 6) < 0)
      {
        memcpy (pke_ptr + 23, mac_ap,  6);
        memcpy (pke_ptr + 29, mac_sta, 6);
      }
      else
      {
        memcpy (pke_ptr + 23, mac_sta, 6);
        memcpy (pke_ptr + 29, mac_ap,  6);
      }

      wpa->nonce_compare = memcmp (wpa->anonce, auth_packet->wpa_key_nonce, 32);

      if (wpa->nonce_compare < 0)
      {
        memcpy (pke_ptr + 35, wpa->anonce,  32);
        memcpy (pke_ptr + 67, auth_packet->wpa_key_nonce, 32);
      }
      else
      {
        memcpy (pke_ptr + 35, auth_packet->wpa_key_nonce, 32);
        memcpy (pke_ptr + 67, wpa->anonce,  32);
      }
    }
    else if (wpa->keyver == 3)
    {
      pke_ptr[0] = 1;
      pke_ptr[1] = 0;

      memcpy (pke_ptr + 2, "Pairwise key expansion", 22);

      if (memcmp (mac_ap, mac_sta, 6) < 0)
      {
        memcpy (pke_ptr + 24, mac_ap,  6);
        memcpy (pke_ptr + 30, mac_sta, 6);
      }
      else
      {
        memcpy (pke_ptr + 24, mac_sta, 6);
        memcpy (pke_ptr + 30, mac_ap,  6);
      }

      wpa->nonce_compare = memcmp (wpa->anonce, auth_packet->wpa_key_nonce, 32);

      if (wpa->nonce_compare < 0)
      {
        memcpy (pke_ptr + 36, wpa->anonce,  32);
        memcpy (pke_ptr + 68, auth_packet->wpa_key_nonce, 32);
      }
      else
      {
        memcpy (pke_ptr + 36, auth_packet->wpa_key_nonce, 32);
        memcpy (pke_ptr + 68, wpa->anonce,  32);
      }

      pke_ptr[100] = 0x80;
      pke_ptr[101] = 1;
    }

    for (int i = 0; i < 32; i++)
    {
      wpa->pke[i] = byte_swap_32 (wpa->pke[i]);
    }

    if (wpa->keyver == 2)
    {
      for (int i = 0; i < (WPA_EAPOL_LEN_MAX / 4); i++)
      {
        wpa->eapol[i] = byte_swap_32 (wpa->eapol[i]);
      }
    }

    if (wpa->keyver == 3)
    {
      eapol_ptr[wpa->eapol_len] = 0x80;
    }

    // message_pair

    const u8 *message_pair_pos = token.buf[8];

    const u8 message_pair = hex_to_u8 (message_pair_pos);

    wpa->message_pair = message_pair;

    /* moved to module_hash_decode_postprocess()
    if (wpa->message_pair_chgd == true)
    {
      // we can filter some message types here

      if (wpa->message_pair != (message_pair & 0x7f)) return (PARSER_HCCAPX_MESSAGE_PAIR);
    }
    else
    {
      wpa->message_pair = message_pair;
    }

    if (wpa->nonce_error_corrections_chgd == true)
    {
      // value was set in module_hash_binary_parse()
    }
    else
    {
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
          wpa->nonce_error_corrections = NONCE_ERROR_CORRECTIONS; // temporary until architectural change done (module_hash_decode_postprocess?)
        }
        else
        {
          wpa->nonce_error_corrections = 0;
        }
      }
    }
    */

    // now some optimization related to replay counter endianess
    // hcxtools has techniques to detect them
    // since we can not guarantee to get our handshakes from hcxtools we enable both by default
    // this means that we check both even if both are not set!
    // however if one of them is set, we can assume that the endianess has been checked and the other one is not needed

    wpa->detected_le = 1;
    wpa->detected_be = 1;

    if (wpa->message_pair & (1 << 5))
    {
      wpa->detected_le = 1;
      wpa->detected_be = 0;
    }
    else if (wpa->message_pair & (1 << 6))
    {
      wpa->detected_le = 0;
      wpa->detected_be = 1;
    }

    // mic

    const u8 *mic_pos = token.buf[2];

    wpa->keymic[0] = hex_to_u32 (mic_pos +  0);
    wpa->keymic[1] = hex_to_u32 (mic_pos +  8);
    wpa->keymic[2] = hex_to_u32 (mic_pos + 16);
    wpa->keymic[3] = hex_to_u32 (mic_pos + 24);

    wpa->keymic[0] = byte_swap_32 (wpa->keymic[0]);
    wpa->keymic[1] = byte_swap_32 (wpa->keymic[1]);
    wpa->keymic[2] = byte_swap_32 (wpa->keymic[2]);
    wpa->keymic[3] = byte_swap_32 (wpa->keymic[3]);

    // Create a hash of the nonce as ESSID is not unique enough
    // Not a regular MD5 but good enough
    // We can also ignore cases where we should bzero the work buffer

    u32 hash[4];

    hash[0] = 0;
    hash[1] = 1;
    hash[2] = 2;
    hash[3] = 3;

    u32 block[16];

    memset (block, 0, sizeof (block));

    u8 *block_ptr = (u8 *) block;

    for (int i = 0; i < 16; i++) block[i] = salt->salt_buf[i];

    md5_transform (block + 0, block + 4, block + 8, block + 12, hash);

    for (int i = 0; i < 16; i++) block[i] = wpa->pke[i +  0];

    md5_transform (block + 0, block + 4, block + 8, block + 12, hash);

    for (int i = 0; i < 16; i++) block[i] = wpa->pke[i + 16];

    md5_transform (block + 0, block + 4, block + 8, block + 12, hash);

    for (int i = 0; i < 16; i++) block[i] = wpa->eapol[i +  0];

    md5_transform (block + 0, block + 4, block + 8, block + 12, hash);

    for (int i = 0; i < 16; i++) block[i] = wpa->eapol[i + 16];

    md5_transform (block + 0, block + 4, block + 8, block + 12, hash);

    for (int i = 0; i < 16; i++) block[i] = wpa->eapol[i + 32];

    md5_transform (block + 0, block + 4, block + 8, block + 12, hash);

    for (int i = 0; i < 16; i++) block[i] = wpa->eapol[i + 48];

    md5_transform (block + 0, block + 4, block + 8, block + 12, hash);

    for (int i = 0; i <  2; i++) block[0 + i] = wpa->mac_ap[i];
    for (int i = 0; i <  2; i++) block[2 + i] = wpa->mac_ap[i];
    for (int i = 0; i < 12; i++) block[4 + i] = 0;

    md5_transform (block + 0, block + 4, block + 8, block + 12, hash);

    memcpy (block_ptr +  0, wpa->anonce,  32);
    memcpy (block_ptr + 32, auth_packet->wpa_key_nonce, 32);

    md5_transform (block + 0, block + 4, block + 8, block + 12, hash);

    block[0] = wpa->keymic[0];
    block[1] = wpa->keymic[1];
    block[2] = wpa->keymic[2];
    block[3] = wpa->keymic[3];

    md5_transform (block + 0, block + 4, block + 8, block + 12, hash);

    // make all this stuff unique

    digest[0] = hash[0];
    digest[1] = hash[1];
    digest[2] = hash[2];
    digest[3] = hash[3];
  }

  return (PARSER_OK);
}

#endif
