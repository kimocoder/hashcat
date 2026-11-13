/**
 * Author......: See docs/credits.txt
 * License.....: MIT
 */

// One binary hccapx record, parsed the same way by 2500 and 2501. The two differ in the salt iteration
// count and in whether a record whose key version is 0 is accepted, which the PMK mode keeps for old
// captures. WPA_SALT_ITER, WPA_KEYVER_VALID and WPA_KEYVER_CMAC carry those differences in.

#ifndef WPA_DECODE_HCCAPX_H
#define WPA_DECODE_HCCAPX_H

static int module_hash_decode (MAYBE_UNUSED const hashconfig_t *hashconfig, MAYBE_UNUSED void *digest_buf, MAYBE_UNUSED salt_t *salt, MAYBE_UNUSED void *esalt_buf, MAYBE_UNUSED void *hook_salt_buf, MAYBE_UNUSED hashinfo_t *hash_info, const char *line_buf, MAYBE_UNUSED const int line_len)
{
  u32 *digest = (u32 *) digest_buf;

  wpa_eapol_t *wpa_eapol = (wpa_eapol_t *) esalt_buf;

  // the esalt was partially initialized beforehand, we can not simply memset it to zero

  // module_hash_binary_parse is the only caller today and it always hands over a full record, but
  // the copy takes a fixed size out of the line either way, so the length is checked here rather
  // than assumed

  if (line_len < (int) sizeof (hccapx_t)) return (PARSER_HASH_LENGTH);

  hccapx_t in;

  memcpy (&in, line_buf, sizeof (hccapx_t));

  if (in.signature != HCCAPX_SIGNATURE) return (PARSER_HCCAPX_SIGNATURE);

  if (in.version != HCCAPX_VERSION) return (PARSER_HCCAPX_VERSION);

  if (in.eapol_len < 1 || in.eapol_len > 255) return (PARSER_HCCAPX_EAPOL_LEN);

  memcpy (wpa_eapol->keymic, in.keymic, 16);

  /*
    http://www.one-net.eu/jsw/j_sec/m_ptype.html
    The phrase "Pairwise key expansion"
    Access Point Address (referred to as Authenticator Address AA)
    Supplicant Address (referred to as Supplicant Address SA)
    Access Point Nonce (referred to as Authenticator Anonce)
    Wireless Device Nonce (referred to as Supplicant Nonce Snonce)
  */

  u32 salt_len = in.essid_len;

  if (salt_len > 32) return (PARSER_SALT_LENGTH);

  memcpy (salt->salt_buf, in.essid, in.essid_len);

  salt->salt_len = salt_len;

  salt->salt_iter = WPA_SALT_ITER;

  memcpy (wpa_eapol->essid, in.essid, in.essid_len);

  wpa_eapol->essid_len = in.essid_len;

  // The EAPOL frame carries the key version its MIC was computed with. The hccapx keyver byte is a
  // copy of that field, and the WPA* modes read the frame. Take the frame here too, so a record where
  // the two disagree is checked the same way in both modes. A frame too short to hold the key
  // information falls back to the byte. The field is three bits wide, so a reserved value above 3 is
  // rejected below rather than folded into one of the defined ones.

  wpa_eapol->keyver = in.keyver;

  if (in.eapol_len >= 7)
  {
    wpa_eapol->keyver = (((u32) in.eapol[5] << 8) | (u32) in.eapol[6]) & 7;
  }

  if (WPA_KEYVER_VALID (wpa_eapol->keyver) == false) return (PARSER_SALT_VALUE);

  u8 *pke_ptr = (u8 *) wpa_eapol->pke;

  memset (pke_ptr, 0, 128);

  if ((wpa_eapol->keyver == 1) || (wpa_eapol->keyver == 2))
  {
    memcpy (pke_ptr, "Pairwise key expansion", 23);

    if (memcmp (in.mac_ap, in.mac_sta, 6) < 0)
    {
      memcpy (pke_ptr + 23, in.mac_ap,  6);
      memcpy (pke_ptr + 29, in.mac_sta, 6);
    }
    else
    {
      memcpy (pke_ptr + 23, in.mac_sta, 6);
      memcpy (pke_ptr + 29, in.mac_ap,  6);
    }

    wpa_eapol->nonce_compare = memcmp (in.nonce_ap, in.nonce_sta, 32);

    if (wpa_eapol->nonce_compare < 0)
    {
      memcpy (pke_ptr + 35, in.nonce_ap,  32);
      memcpy (pke_ptr + 67, in.nonce_sta, 32);
    }
    else
    {
      memcpy (pke_ptr + 35, in.nonce_sta, 32);
      memcpy (pke_ptr + 67, in.nonce_ap,  32);
    }
  }
  else if (WPA_KEYVER_CMAC (wpa_eapol->keyver) == true)
  {
    pke_ptr[0] = 1;
    pke_ptr[1] = 0;

    memcpy (pke_ptr + 2, "Pairwise key expansion", 22);

    if (memcmp (in.mac_ap, in.mac_sta, 6) < 0)
    {
      memcpy (pke_ptr + 24, in.mac_ap,  6);
      memcpy (pke_ptr + 30, in.mac_sta, 6);
    }
    else
    {
      memcpy (pke_ptr + 24, in.mac_sta, 6);
      memcpy (pke_ptr + 30, in.mac_ap,  6);
    }

    wpa_eapol->nonce_compare = memcmp (in.nonce_ap, in.nonce_sta, 32);

    if (wpa_eapol->nonce_compare < 0)
    {
      memcpy (pke_ptr + 36, in.nonce_ap,  32);
      memcpy (pke_ptr + 68, in.nonce_sta, 32);
    }
    else
    {
      memcpy (pke_ptr + 36, in.nonce_sta, 32);
      memcpy (pke_ptr + 68, in.nonce_ap,  32);
    }

    pke_ptr[100] = 0x80;
    pke_ptr[101] = 1;
  }

  for (int i = 0; i < 32; i++)
  {
    wpa_eapol->pke[i] = byte_swap_32 (wpa_eapol->pke[i]);
  }

  memcpy (wpa_eapol->orig_mac_ap,    in.mac_ap,    6);
  memcpy (wpa_eapol->orig_mac_sta,   in.mac_sta,   6);
  memcpy (wpa_eapol->orig_nonce_ap,  in.nonce_ap,  32);
  memcpy (wpa_eapol->orig_nonce_sta, in.nonce_sta, 32);

  u8 message_pair_orig = in.message_pair;

  in.message_pair &= 0x0f; // compare the message pair only, the flags sit above it

  if (wpa_eapol->message_pair_chgd == true)
  {
    if (wpa_eapol->message_pair != in.message_pair) return (PARSER_HCCAPX_MESSAGE_PAIR);
  }

  wpa_eapol->message_pair = message_pair_orig;

  wpa_eapol->eapol_len = in.eapol_len;

  u8 *eapol_ptr = (u8 *) wpa_eapol->eapol;

  memcpy (eapol_ptr, in.eapol, wpa_eapol->eapol_len);

  memset (eapol_ptr + wpa_eapol->eapol_len, 0, (256 + 64) - wpa_eapol->eapol_len);

  eapol_ptr[wpa_eapol->eapol_len] = 0x80;

  if (wpa_eapol->keyver == 1)
  {
    // nothing to do
  }
  else if (wpa_eapol->keyver == 2)
  {
    wpa_eapol->keymic[0] = byte_swap_32 (wpa_eapol->keymic[0]);
    wpa_eapol->keymic[1] = byte_swap_32 (wpa_eapol->keymic[1]);
    wpa_eapol->keymic[2] = byte_swap_32 (wpa_eapol->keymic[2]);
    wpa_eapol->keymic[3] = byte_swap_32 (wpa_eapol->keymic[3]);

    for (int i = 0; i < 64; i++)
    {
      wpa_eapol->eapol[i] = byte_swap_32 (wpa_eapol->eapol[i]);
    }
  }
  else if (WPA_KEYVER_CMAC (wpa_eapol->keyver) == true)
  {
    // nothing to do
  }

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

  for (int i = 0; i < 16; i++) block[i] = wpa_eapol->pke[i +  0];

  md5_transform (block + 0, block + 4, block + 8, block + 12, hash);

  for (int i = 0; i < 16; i++) block[i] = wpa_eapol->pke[i + 16];

  md5_transform (block + 0, block + 4, block + 8, block + 12, hash);

  for (int i = 0; i < 16; i++) block[i] = wpa_eapol->eapol[i +  0];

  md5_transform (block + 0, block + 4, block + 8, block + 12, hash);

  for (int i = 0; i < 16; i++) block[i] = wpa_eapol->eapol[i + 16];

  md5_transform (block + 0, block + 4, block + 8, block + 12, hash);

  for (int i = 0; i < 16; i++) block[i] = wpa_eapol->eapol[i + 32];

  md5_transform (block + 0, block + 4, block + 8, block + 12, hash);

  for (int i = 0; i < 16; i++) block[i] = wpa_eapol->eapol[i + 48];

  md5_transform (block + 0, block + 4, block + 8, block + 12, hash);

  for (int i = 0; i <  6; i++) block_ptr[i + 0] = wpa_eapol->orig_mac_ap[i];
  for (int i = 0; i <  6; i++) block_ptr[i + 6] = wpa_eapol->orig_mac_sta[i];

  md5_transform (block + 0, block + 4, block + 8, block + 12, hash);

  for (int i = 0; i < 32; i++) block_ptr[i +  0] = wpa_eapol->orig_nonce_ap[i];
  for (int i = 0; i < 32; i++) block_ptr[i + 32] = wpa_eapol->orig_nonce_sta[i];

  md5_transform (block + 0, block + 4, block + 8, block + 12, hash);

  block[0] = wpa_eapol->keymic[0];
  block[1] = wpa_eapol->keymic[1];
  block[2] = wpa_eapol->keymic[2];
  block[3] = wpa_eapol->keymic[3];

  md5_transform (block + 0, block + 4, block + 8, block + 12, hash);

  // make all this stuff unique

  digest[0] = hash[0];
  digest[1] = hash[1];
  digest[2] = hash[2];
  digest[3] = hash[3];

  return (PARSER_OK);
}

#endif
